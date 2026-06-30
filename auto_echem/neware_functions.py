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
