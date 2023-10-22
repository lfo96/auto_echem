import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import numpy as np
import datetime as dt
import math

from stat import ST_CTIME
from matplotlib import cm
from IPython.display import set_matplotlib_formats
from itertools import islice
from galvani import BioLogic as BL
from contextlib import contextmanager
from IPython.display import clear_output

from auto_echem.general_functions import info
from auto_echem.general_functions import find_nearest
from auto_echem.general_functions import LSV_cond
from auto_echem.general_functions import layout
from auto_echem.general_functions import isclose

from auto_echem.impedance_functions import Nyquist
from auto_echem.impedance_functions import parameter
from auto_echem.impedance_functions import plot_PEIS
from auto_echem.impedance_functions import plot_R
from auto_echem.impedance_functions import eva_PEIS
from auto_echem.impedance_functions import strip_plate

from auto_echem.GCPL_functions import eva_GCPL
from auto_echem.GCPL_functions import has_decreasing_numbers
from auto_echem.GCPL_functions import eva_GCPL_index
from auto_echem.GCPL_functions import plot_galv
from auto_echem.GCPL_functions import plot_CR
from auto_echem.GCPL_functions import plot_CP_raman
from auto_echem.GCPL_functions import cy_index




def auto(pathway, circ = ['Rp', 'Rp'], plot = '',resttime = 50, save = '',fit_para = 0, cy = [1,2,5,10,20,30,50],l=0.14, r_cc=5, PEIS_evaluation = True, lf_limit=0,hf_limit = math.inf,clear_cell = False):
    '''
    Automatically detect and analysis the measurements produced by EC-lab.
    Insert pathway with EC-lab settings file (.mps)
    Returns a dictionary (dict) with meta data, all the raw data  (dict['data']), and the evaluated raw data (dict['eva']). 

    Arguments explanation:
    circ: specify the circuit used for impedance fitting. Standard is 'Rp' which is a simple R0-p(R1-CPE1) circuit.
    plot: if set False, does not plot any evaluated data
    resttime: resttime in between measurements (usually when having a loop over impedance measurements). Will exctract and overwrite the value from the setting file.
    save: specify a name if you wish to save your plots
    fit_para: different initial conditions for impedance fit. Depending on the electronic circuit specified in "circ" various conditions are used. The specific condition can be found inside the "impedance_functions.py". 
    cy: specify the number of cycles you wish to display in your GCPL plot.
    l: used for linear sweep voltametry to caluclate electronic conductivity
    r_cc: actually forgot what this is 
    PEIS_evaluation: if set to False, will not fit any impedance data
    lf_limit: sets a low frequenecy limit for impedance fit, essentially cuts of every measurement point below a frequency determinede in Hz
    '''
    meta = info(pathway)
    d_eva = {}
    circ_count = 0
    cc_switch = [False,False] # Switch to track if a constant curret file has been evaluated: (negative I, positive I)
    for entry in meta['data']:
        start = time.time()
        if entry.split(' ')[1] == 'PEIS':
            if PEIS_evaluation == False:
                continue
            d = {}
            if meta['MB'] == True:
                if len(meta['data'][entry])==0:
                    print(str([entry])+' is empty.')
                    continue
                # MB files do not contain the Re and Im part of the impedance and therefore need to be calculated from the absolute value and the phase shift. 
                hyp = meta['data'][entry]['|Z|/Ohm']
                alpha_deg = meta['data'][entry]['Phase(Z)/deg']*(np.pi/180)
                Re=np.cos(alpha_deg)*hyp
                Im=np.sin(alpha_deg)*hyp
                meta['data'][entry]['Re(Z)/Ohm'] = Re
                meta['data'][entry]['-Im(Z)/Ohm'] = -1*Im
                meta['data'][entry]['<I>/mA'] = meta['data'][entry]['control/V/mA']
                meta['data'][entry]['<Ewe>/V'] = meta['data'][entry]['Ewe/V']

            if circ == False:
                evaluated = eva_PEIS(meta['data'][entry])
                d['Nyquist data'] = [evaluated]
                d_eva[entry] = d
                break
            try:
                print('Fitting with '+circ[circ_count])
            except IndexError:
                print('Please specify the circuit fit for '+str(entry))
                continue

            if circ[circ_count] == 'R':
                R_ac = meta['data'][entry]['Re(Z)/Ohm'].iloc[-1]
                A = (r_cc*0.001)*(r_cc*0.001)*3.141592 #m2
                l = l*0.001 #m
                cond = (1/R_ac)*l/A
                d['AC Conductivity (S/m)'] = cond
                d_eva[entry] = d
                end = time.time()
                print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
                if plot =='':
                    plot_PEIS([[meta['data'][entry]['freq/Hz'],meta['data'][entry]['Re(Z)/Ohm'],meta['data'][entry]['-Im(Z)/Ohm']]], tit = meta['filename'])
                break
            data_eva = Nyquist(meta['data'][entry], circ = circ[circ_count], fit_para = fit_para,lf_limit=lf_limit,hf_limit=hf_limit)
            d['Nyquist data'] = data_eva
                        
            try:
                data_para = parameter(data_eva)
                d['Nyquist parameter'] = data_para
            except IndexError:
                print('IndexError in parameter function.')
                #continue
                break

            if plot == '':
                if save != '':
                    plot_PEIS(data_eva[0],data_eva[1],label = 'off', tit = meta['filename'], save=meta['filename']+'_PEIS')
                    try:
                        plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area'], tit = meta['filename'], save=meta['filename']+'_R_ct')
                        print('Plots saved.')
                    except KeyError:
                        print('Key Error - could not plot impedance parameter.')
                elif plot =='none':
                    pass
                else:
                    plot_PEIS(data_eva[0],data_eva[1],label = 'off', tit = meta['filename'])
                    #plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area'], tit = meta['filename'])
                    try:
                        plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area'], tit = meta['filename'])
                    except KeyError:
                        print('Key Error - could not plot impedance parameter.')

                


            if np.isnan(meta['waiting time']) == False:
                try:
                    t_rest = meta['waiting time']*np.array(range(len(data_para['R1'])))
                    R_ct_rest = find_nearest(t_rest,resttime)
                    R_ct = round(data_para['R1'][R_ct_rest[0]]*meta['electrode surface area'],2)
                    d['R_'+str(R_ct_rest[1])] = R_ct
                    print(entry+'_R_'+str(R_ct_rest[1])+' : '+str(R_ct))
                except KeyError:
                    print('Key Error - could not extract RCT.')


            d_eva[entry] = d
            end = time.time()
            print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
            circ_count += 1
            if circ_count > len(circ):
                print('Please specify the circut model for the '+str(circ_count+1)+' PEIS data set.')
                break

        elif entry.split(' ')[1] == 'GCPL':
            try:
                if has_decreasing_numbers(meta['data'][entry]['half cycle'])== True:
                    print('Alternative index based GCPL analysis.')
                    d = eva_GCPL_index(meta['data'][entry], meta['active material mass'])
                else:
                    d = eva_GCPL(meta['data'][entry], meta['active material mass'], meta['electrode surface area'])
                if d[0]['Gravimetric Discharge Capacity (mAh/g)'].isna().sum()==len(d[0]['Gravimetric Discharge Capacity (mAh/g)']):
                    '''
                    This is triggered in the stripping plating GCPL. Not entirley sure why but it populates all columns with np.nan...
                    '''
                    d = {}
                    data = meta['data'][entry]
                    op = []
                    cy_in = cy_index(data)
                    for i in cy_in[1]:
                        op.append(data['Ewe/V'].loc[data.index[i]])
                    print(entry + ' evaluated.')
                    I_area = round(data['control/V/mA'].max()/meta['electrode surface area'],3)
                    d['over potential'] = op
                    d['areal current (mA/cm2)'] = I_area
                    if plot == '':
                        strip_plate(data, title = str(I_area)+' mA/cm2')
                    d_eva[entry] = d
                    end = time.time()
                    print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
                else:
                    if plot == '':
                        plot_galv(d, save = save, cy = cy)
                        plot_CR(d, save = save)
                    d_eva[entry] = d
                    end = time.time()
                    print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
            except UnboundLocalError:
                print('Unbound Local Error: Potential incomplete first cycle.')
        
        elif entry.split(' ')[1] == 'LSV':
            LSV_eva = LSV_cond(meta['data'][entry],d=l,r_cc=r_cc) # Conductivity in S/m
            if plot == '':
                fig,ax = plt.subplots()
                plt.scatter(LSV_eva[1][0],LSV_eva[1][1])
                layout(ax,x_label='E (V)',y_label='I (A)')
            d = {}
            d['DC Conductivity (S/m)'] = LSV_eva[0]
            d_eva[entry] = d

        elif entry.split(' ')[1] == 'GEIS':
            start = time.time()
            d = {}
            try:
                print('Fitting with '+circ[circ_count])
            except IndexError:
                print('Please specify the circuit fit for '+str(entry))
                break
            if circ[circ_count] == 'R':
                R_ac = meta['data'][entry]['Re(Z)/Ohm'].iloc[-1]
                A = (r_cc*0.001)*(r_cc*0.001)*3.141592 #m2
                l = l*0.001 #m
                cond = (1/R_ac)*l/A
                d['AC Conductivity (S/m)'] = cond
                d_eva[entry] = d
                end = time.time()
                print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
                if plot =='':
                    plot_PEIS([[meta['data'][entry]['freq/Hz'],meta['data'][entry]['Re(Z)/Ohm'],meta['data'][entry]['-Im(Z)/Ohm']]], tit = meta['filename'])
                break

            data_eva = Nyquist(meta['data'][entry], circ = circ[circ_count], fit_para = fit_para, lf_limit = lf_limit,hf_limit=hf_limit)
            d['Nyquist data'] = data_eva
                        
            try:
                data_para = parameter(data_eva)
                d['Nyquist parameter'] = data_para
            except IndexError:
                print('IndexError in parameter function.')
                continue

            if plot == '':
                if save != '':
                    plot_PEIS(data_eva[0],data_eva[1],label = 'off', tit = meta['filename'], save=meta['filename']+'_PEIS')
                    try:
                        plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area'], tit = meta['filename'], save=meta['filename']+'_R_ct')
                        print('Plots saved.')
                    except KeyError:
                        print('Key Error - could not plot impedance parameter.')
                elif plot =='none':
                    pass
                else:
                    plot_PEIS(data_eva[0],data_eva[1],label = 'off', tit = meta['filename'])
                    try:
                        plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area'], tit = meta['filename'])
                    except KeyError:
                        print('Key Error - could not plot impedance parameter.')

                


            if np.isnan(meta['waiting time']) == False:
                try:
                    t_rest = meta['waiting time']*np.array(range(len(data_para['R1'])))
                    R_ct_rest = find_nearest(t_rest,resttime)
                    R_ct = round(data_para['R1'][R_ct_rest[0]]*meta['electrode surface area'],2)
                    d['R_'+str(R_ct_rest[1])] = R_ct
                    print(entry+'_R_'+str(R_ct_rest[1])+' : '+str(R_ct))
                except KeyError:
                    print('Key Error - could not extract RCT.')

                #meta['eva'][entry+'_R_'+str(R_ct_rest[1])] = R_ct
                

            d_eva[entry] = d
            end = time.time()
            print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
            circ_count += 1
            if circ_count > len(circ):
                print('Please specify the circut model for the '+str(circ_count+1)+' PEIS data set.')
                break
            print('Yes')
        elif entry.split(' ')[1] == 'OCV':
            if plot == '':
                fig,ax = plt.subplots()
                ax.plot(meta['data'][entry]['time/s']/3600,meta['data'][entry]['Ewe/V'])
                layout(ax, x_label='time (h)',y_label = 'Potential (V)',title='OCV',x_lim=[meta['data'][entry]['time/s'].iloc[0]/3600,meta['data'][entry]['time/s'].iloc[-1]/3600])
            end = time.time()
            print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')

        elif entry.split(' ')[1] == 'CC':
            d = {}
            data = meta['data'][entry]
            if len(data) == 0:
                print(entry+' is empty.')
                continue 
            I_area = round(data['control/V/mA'].mean()/meta['electrode surface area'],3)
            if I_area <= 0:
                cc_switch[0] = (entry,I_area)
            elif I_area >=0:
                cc_switch[1] = (entry,I_area)

            op_last = []
            op_first = []
            
            cy_in = cy_index(data)
            
            for i in cy_in[0]:
                op_first.append(data['Ewe/V'].loc[i])
            for i in cy_in[1]:
                op_last.append(data['Ewe/V'].loc[i])
                
            print(entry + ' evaluated.')
            d['over potential first (V)'] = op_first
            d['over potential last (V)'] = op_last
            d['areal current (mA/cm2)'] = I_area

            d_eva[entry] = d
            end = time.time()
            print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
            
            if not (cc_switch[0] and cc_switch[1]) is False:
                # Merge the CC files from negative and positive currents
                d_cc = {}
                ID = 'CC_'+str(cc_switch[0][0].split(' ')[0])+'_'+str(cc_switch[1][0].split(' ')[0])
                df_cc = pd.concat([meta['data'][cc_switch[0][0]], meta['data'][cc_switch[1][0]]], sort=False).sort_index()
                op_last = []
                op_first = []
                
                cy_in = cy_index(df_cc)
                
                for i in cy_in[0]:
                    op_first.append(df_cc['Ewe/V'].loc[i+1])
                for i in cy_in[1]:
                    op_last.append(df_cc['Ewe/V'].loc[i])

                if isclose(abs(cc_switch[0][1]),abs(cc_switch[1][1])) is True:
                    I_cc_areal = (abs(cc_switch[0][1])+abs(cc_switch[1][1]))/2
                else:
                    I_cc_areal = (cc_switch[0][1],cc_switch[1][1])
                    print('Chage and Discharge Currents are different.')

                d_cc['Areal Current (mA/cm2)'] = I_cc_areal
                d_cc['df'] = df_cc
                d_cc['over potential first (V)'] = op_first
                d_cc['over potential last (V)'] = op_last
                d_eva[ID] = d_cc
                meta['eva'] = d_eva
                cc_switch = [False,False]
                if plot == '':
                    strip_plate(df_cc, title = str(I_cc_areal)+' $\mathregular{mA\,cm^{-2}}$')
                    
        elif entry.split(' ')[1] == 'CP':
            ''''
            Analysis for Constant Potential set up with Raman measurement. The A_el is automatically set to 0.4 cm radius.
            '''
            A_el = 0.4*0.4*3.141592
            data = meta['data'][entry]
            if plot == '':
                plot_CP_raman(data,A_el, save = '')
            
    meta['eva'] = d_eva
    if clear_cell==True:
        clear_output(wait=False)

    return(meta)
    