"""
Neware file ingestion for auto_echem.

Supports two input formats:
    .ndax   – primary format; read via NewareNDA + XML metadata extraction
    .csv    – legacy Neware BTSDA CSV export

Both paths return the same (DataFrame, metadata_dict) pair so downstream
analysis (eva_neware, plotting) never needs to know which format it came from.

Normalized DataFrame columns
─────────────────────────────────────────────────────────────────────────────
cycle                   int     Neware cycle number
step                    int     Step number within the test program
status                  str     Step type: 'Rest', 'CCCV_Chg', 'CCCV_DChg', …
time/s                  float   Elapsed seconds from test start
voltage/V               float   Cell voltage (V)
current/mA              float   Signed current; + = charge, – = discharge (mA)
charge_capacity/mAh     float   Cumulative charge capacity within step (mAh)
discharge_capacity/mAh  float   Cumulative discharge capacity within step (mAh)
charge_energy/mWh       float   Cumulative charge energy within step (mWh)
discharge_energy/mWh    float   Cumulative discharge energy within step (mWh)
timestamp               datetime Absolute datetime of each data point

Metadata dict keys always present
─────────────────────────────────────────────────────────────────────────────
ID                      str     Experiment identifier (Remark field or folder name)
source_file             str     Absolute path to the source file
format                  str     'ndax' or 'csv'
Remark                  str     Free-text sample code — join key for electrolab
TestGuid                str     Unique test GUID — join key for electrolab ('' for csv)
active material mass    float   mg; np.nan — fill from electrolab before eva_neware()
electrode surface area  float   cm²; np.nan — fill from electrolab before eva_neware()

Additional keys present for .ndax only
─────────────────────────────────────────────────────────────────────────────
device_id, unit_id, channel_id, test_id
current_range/mA, voltage_range/V
step_schedule           str     Name of the Neware step program file
start_time, end_time    datetime
"""

import os
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.integrate import simpson

from auto_echem.general_functions import isclose

def _simps(y, x):
    """Thin wrapper so energy integration matches the rest of the package."""
    return simpson(y, x=x)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_status(s: str) -> str:
    """Normalize step-type strings: 'CCCV Chg' → 'CCCV_Chg'."""
    return str(s).replace(' ', '_')


def _hms_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a Series of 'HH:MM:SS' strings (hours may exceed 24) to float seconds."""
    return pd.to_timedelta(series).dt.total_seconds()


# ─────────────────────────────────────────────────────────────────────────────
# .ndax path
# ─────────────────────────────────────────────────────────────────────────────

def _parse_ndax_metadata(pathway: str) -> dict:
    """Parse TestInfo.xml from an .ndax ZIP and return a metadata dict."""
    meta = {
        'ID': os.path.basename(os.path.dirname(os.path.abspath(pathway))),
        'source_file': os.path.abspath(pathway),
        'format': 'ndax',
        'Remark': '',
        'TestGuid': '',
        'device_id': '',
        'unit_id': '',
        'channel_id': '',
        'test_id': '',
        'current_range/mA': np.nan,
        'voltage_range/V': np.nan,
        'step_schedule': '',
        'start_time': None,
        'end_time': None,
        'active material mass': np.nan,
        'electrode surface area': np.nan,
    }
    try:
        with zipfile.ZipFile(pathway) as z:
            # TestInfo.xml declares encoding="GB2312"; decode to str first so ET
            # doesn't try to handle the multi-byte encoding itself.
            raw_xml = z.read('TestInfo.xml').decode('GB2312', errors='replace')
            root = ET.fromstring(raw_xml)
            ti = root.find('.//TestInfo')
            if ti is None:
                return meta
            meta['TestGuid'] = ti.get('TestGuid', '')
            meta['Remark'] = ti.get('Remark', '')
            meta['device_id'] = ti.get('DevID', '')
            meta['unit_id'] = ti.get('UnitID', '')
            meta['channel_id'] = ti.get('ChlID', '')
            meta['test_id'] = ti.get('TestID', '')
            meta['step_schedule'] = ti.get('StepName', '')
            cr = ti.get('CurrRange', '')
            meta['current_range/mA'] = float(cr) if cr else np.nan
            vr = ti.get('VoltRange', '')
            meta['voltage_range/V'] = float(vr) if vr else np.nan
            st = ti.get('StartTime', '')
            if st:
                meta['start_time'] = pd.to_datetime(st)
            et = ti.get('EndTime', '')
            if et:
                meta['end_time'] = pd.to_datetime(et)
            # Prefer Remark as the human-readable experiment ID
            if meta['Remark']:
                meta['ID'] = meta['Remark']
    except Exception as e:
        print(f'Warning: could not fully parse .ndax metadata: {e}')
    return meta


def _read_ndax(pathway: str) -> tuple:
    """
    Read a .ndax file using NewareNDA and return (normalized_df, metadata_dict).

    Requires: pip install NewareNDA
    """
    try:
        import NewareNDA
    except ImportError:
        raise ImportError(
            "NewareNDA is required to read .ndax files. "
            "Install it with: pip install NewareNDA"
        )

    meta = _parse_ndax_metadata(pathway)
    raw = NewareNDA.read(pathway)

    df = pd.DataFrame({
        'cycle':                  raw['Cycle'].astype(int),
        'step':                   raw['Step'].astype(int),
        'status':                 raw['Status'].astype(str).apply(_normalize_status),
        'time/s':                 raw['Time'].astype(float),
        'voltage/V':              raw['Voltage'].astype(float),
        'current/mA':             raw['Current(mA)'].astype(float),
        'charge_capacity/mAh':    raw['Charge_Capacity(mAh)'].astype(float),
        'discharge_capacity/mAh': raw['Discharge_Capacity(mAh)'].astype(float),
        'charge_energy/mWh':      raw['Charge_Energy(mWh)'].astype(float),
        'discharge_energy/mWh':   raw['Discharge_Energy(mWh)'].astype(float),
        'timestamp':              raw['Timestamp'],
    })

    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# .csv / .txt path
# ─────────────────────────────────────────────────────────────────────────────

def _read_neware_csv(pathway: str) -> tuple:
    """
    Read a Neware BTSDA CSV export and return (normalized_df, metadata_dict).

    Expected CSV columns (BTSDA v8 format):
        DataPoint, Cycle Index, Step Index, Step Type, Time, Total Time,
        Current(A), Voltage(V), Capacity(Ah), Spec. Cap.(mAh/g),
        Energy(Wh), Spec. Energy(mWh/g), Date, Power(W)
    """
    folder = os.path.basename(os.path.dirname(os.path.abspath(pathway)))
    meta = {
        'ID': folder,
        'source_file': os.path.abspath(pathway),
        'format': 'csv',
        'Remark': folder,
        'TestGuid': '',
        'active material mass': np.nan,
        'electrode surface area': np.nan,
    }

    raw = pd.read_csv(pathway)

    # Split the single Capacity and Energy column into charge / discharge using
    # current sign. 'Chg' string matching is avoided because 'DChg' also
    # contains 'Chg'. Rest rows (current ≈ 0) get 0.0 in both columns.
    is_chg  = raw['Current(A)'] > 0
    is_dchg = raw['Current(A)'] < 0
    cap_mah = raw['Capacity(Ah)'] * 1000.0
    eng_mwh = raw['Energy(Wh)']   * 1000.0

    df = pd.DataFrame({
        'cycle':                  raw['Cycle Index'].astype(int),
        'step':                   raw['Step Index'].astype(int),
        'status':                 raw['Step Type'].apply(_normalize_status),
        'time/s':                 _hms_to_seconds(raw['Total Time']),
        'voltage/V':              raw['Voltage(V)'].astype(float),
        'current/mA':             raw['Current(A)'].astype(float) * 1000.0,
        'charge_capacity/mAh':    cap_mah.where(is_chg,  0.0).astype(float),
        'discharge_capacity/mAh': cap_mah.where(is_dchg, 0.0).astype(float),
        'charge_energy/mWh':      eng_mwh.where(is_chg,  0.0).astype(float),
        'discharge_energy/mWh':   eng_mwh.where(is_dchg, 0.0).astype(float),
        'timestamp':              pd.to_datetime(raw['Date']),
    })

    return df, meta


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def read_neware(pathway: str) -> tuple:
    """
    Read a Neware data file and return a normalized (DataFrame, metadata) pair.

    Dispatches on file extension:
        .ndax       → NewareNDA reader + XML metadata
        .csv / .txt → BTSDA CSV reader

    Parameters
    ----------
    pathway : str
        Path to the .ndax, .csv, or .txt file.

    Returns
    -------
    df : pd.DataFrame
        Normalized time-series data. See module docstring for column definitions.
    meta : dict
        Experiment metadata. 'active material mass' and 'electrode surface area'
        are np.nan until filled from electrolab. Use meta['Remark'] or
        meta['TestGuid'] as the join key for that lookup.

    Example
    -------
    df, meta = read_neware('path/to/experiment.ndax')
    meta['active material mass'] = 1.5   # mg, from electrolab
    meta['electrode surface area'] = 0.785  # cm²
    """
    ext = os.path.splitext(pathway)[1].lower()
    if ext == '.ndax':
        return _read_ndax(pathway)
    elif ext in ('.csv', '.txt'):
        return _read_neware_csv(pathway)
    else:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Expected .ndax, .csv, or .txt."
        )


# ─────────────────────────────────────────────────────────────────────────────
# GCPL evaluation
# ─────────────────────────────────────────────────────────────────────────────

# Status strings that indicate a charge or discharge step. Extend if new step
# types appear (e.g. 'CC_Chg', 'CC_DChg').
_CHG_STATUSES  = {'CCCV_Chg',  'CC_Chg'}
_DCHG_STATUSES = {'CCCV_DChg', 'CC_DChg'}


def eva_neware(pathway, m_am=np.nan, A_el=np.nan):
    """
    Read a Neware file and evaluate GCPL cycling data.

    Returns a metadata dict whose structure mirrors the BioLogic auto()
    output so all existing plotting functions (plot_galv, plot_CR) work
    without changes.

    Parameters
    ----------
    pathway : str
        Path to a .ndax, .csv, or .txt Neware file.
    m_am : float
        Active material mass in mg. Required for gravimetric quantities.
        Can be set after the fact: result['active material mass'] = 1.5
    A_el : float
        Electrode surface area in cm². Required for areal quantities.

    Returns
    -------
    meta : dict
        Keys always present:
          'ID', 'source_file', 'format', 'Remark', 'TestGuid'
          'active material mass' (mg), 'electrode surface area' (cm²)
          'data'   – normalized raw DataFrame from read_neware()
          'eva'    – tuple (eva_df, galv), same structure as eva_GCPL output:
              eva_df  : DataFrame, one row per full cycle
              galv    : list of per-cycle dicts with voltage/capacity curves

    Example
    -------
    result = eva_neware('exp.csv', m_am=1.5, A_el=0.785)
    plot_galv(result['eva'], cy=[1, 2, 5])
    plot_CR(result['eva'])
    """
    df, meta = read_neware(pathway)
    meta['active material mass'] = m_am
    meta['electrode surface area'] = A_el

    m_g = 0.001 * m_am  # mass in grams (avoids repeated conversion)

    cap_dis,      cap_dis_calc      = [], []
    cap_cha,      cap_cha_calc      = [], []
    energy_dis,   energy_dis_calc   = [], []
    energy_cha,   energy_cha_calc   = [], []
    I_areal,      I_specific        = [], []
    galv = []
    cy_nos = []

    for cy in sorted(df['cycle'].unique()):
        df_cy = df[df['cycle'] == cy]

        dchg = df_cy[df_cy['status'].isin(_DCHG_STATUSES)]
        chg  = df_cy[df_cy['status'].isin(_CHG_STATUSES)]

        if len(dchg) < 2 or len(chg) < 2:
            # Skip cycles that don't have both a discharge and a charge step
            # (e.g. a Rest-only cycle at the very start or end of the file).
            continue

        # ── Measured capacity (mAh/g) from Neware's integrated capacity column
        grav_cap_dis = dchg['discharge_capacity/mAh'] / m_g
        grav_cap_cha = chg['charge_capacity/mAh']     / m_g

        # ── Calculated capacity: integrate |I|·dt cumulatively (matches BioLogic)
        t_dis = dchg['time/s'] - dchg['time/s'].iloc[0]
        t_cha = chg['time/s']  - chg['time/s'].iloc[0]
        grav_cap_dis_calc = abs(dchg['current/mA']) * (t_dis / 3600) / m_g
        grav_cap_cha_calc = abs(chg['current/mA'])  * (t_cha / 3600) / m_g

        cap_dis.append(grav_cap_dis.iloc[-1])
        cap_cha.append(grav_cap_cha.iloc[-1])
        cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
        cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])

        # ── Energy: area under V–Q curve (mWh/g), matches simps usage in eva_GCPL
        en_dis      = _simps(dchg['voltage/V'].values, grav_cap_dis.values)
        en_cha      = _simps(chg['voltage/V'].values,  grav_cap_cha.values)
        en_dis_calc = _simps(dchg['voltage/V'].values, grav_cap_dis_calc.values)
        en_cha_calc = _simps(chg['voltage/V'].values,  grav_cap_cha_calc.values)
        energy_dis.append(en_dis)
        energy_cha.append(en_cha)
        energy_dis_calc.append(en_dis_calc)
        energy_cha_calc.append(en_cha_calc)

        # ── Areal and specific current (modal current, in mA)
        I_d = abs(dchg['current/mA']).mode().mean()
        I_c = abs(chg['current/mA']).mode().mean()
        if isclose(I_d, I_c, rel_tol=0.01):
            I = (I_d + I_c) / 2
            I_areal.append(round(I * 1000 / A_el, 3))   # µA/cm²
            I_specific.append(round(I / m_am, 3))        # mA/g
        else:
            I_areal.append([round(I_d * 1000 / A_el, 3),
                            round(I_c * 1000 / A_el, 3)])
            I_specific.append([round(I_d / m_am, 3),
                               round(I_c / m_am, 3)])

        cy_nos.append(cy)

        # ── Per-cycle voltage/capacity curves for plot_galv
        galv.append({
            'Gravimetric Discharge Capacity (mAh/g)':            grav_cap_dis,
            'Gravimetric Discharge Capacity Calculated (mAh/g)': grav_cap_dis_calc,
            'Gravimetric Charge Capacity (mAh/g)':               grav_cap_cha,
            'Gravimetric Charge Capacity Calculated (mAh/g)':    grav_cap_cha_calc,
            'Discharge Potential (V)':  dchg['voltage/V'],
            'Charge Potential (V)':     chg['voltage/V'],
            'Discharge Time (s)':       t_dis,
            'Charge Time (s)':          t_cha,
            'Discharge Current (mA)':   dchg['current/mA'],
            'Charge Current (mA)':      chg['current/mA'],
        })

    if not cy_nos:
        print('Warning: no complete charge+discharge cycles found.')
        meta['data'] = df
        meta['eva'] = None
        return meta

    cap_dis  = np.array(cap_dis)
    cap_cha  = np.array(cap_cha)
    cap_dis_areal = cap_dis * m_g / A_el
    cap_cha_areal = cap_cha * m_g / A_el

    ce          = 100 * cap_dis      / cap_cha
    ce_calc     = 100 * np.array(cap_dis_calc) / np.array(cap_cha_calc)
    energy_ef      = 100 * np.array(energy_dis)      / np.array(energy_cha)
    energy_ef_calc = 100 * np.array(energy_dis_calc) / np.array(energy_cha_calc)

    eva = pd.DataFrame({
        'Cycle':                                          cy_nos,
        'Gravimetric Discharge Capacity (mAh/g)':        cap_dis,
        'Gravimetric Discharge Capacity Calculated (mAh/g)': cap_dis_calc,
        'Areal Discharge Capacity (mAh/cm$^2$)':         cap_dis_areal,
        'Gravimetric Charge Capacity (mAh/g)':           cap_cha,
        'Gravimetric Charge Capacity Calculated (mAh/g)': cap_cha_calc,
        'Areal Charge Capacity (mAh/cm$^2$)':            cap_cha_areal,
        'Coulombic Efficency (%)':                        ce,
        'Coulombic Efficency Calculated (%)':             ce_calc,
        'Discharge Energy (mWh/g)':                       energy_dis,
        'Discharge Energy Calculated (mWh/g)':            energy_dis_calc,
        'Charge Energy (mWh/g)':                          energy_cha,
        'Charge Energy Calculated (mWh/g)':               energy_cha_calc,
        'Energy Efficency (%)':                           energy_ef,
        'Energy Efficenecy Calculated (%)':               energy_ef_calc,
        'Areal Current (μA/cm$^2$)':                I_areal,
        'Specific Current (mA/g)':                        I_specific,
    })

    meta['data'] = df
    meta['eva']  = (eva, galv)
    return meta
