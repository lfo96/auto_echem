import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import numpy as np
import datetime as dt

from stat import ST_CTIME
from matplotlib import cm
from IPython.display import set_matplotlib_formats
from itertools import islice
from galvani import BioLogic as BL
from contextlib import contextmanager

from Python_scripts.general_functions import info
from Python_scripts.general_functions import find_nearest
from Python_scripts.general_functions import LSV_cond
from Python_scripts.general_functions import layout
from Python_scripts.general_functions import isclose

from Python_scripts.impedance_functions import Nynquist
from Python_scripts.impedance_functions import parameter
from Python_scripts.impedance_functions import plot_PEIS
from Python_scripts.impedance_functions import plot_R
from Python_scripts.impedance_functions import eva_PEIS
from Python_scripts.impedance_functions import strip_plate

from Python_scripts.GCPL_functions import eva_GCPL
from Python_scripts.GCPL_functions import plot_galv
from Python_scripts.GCPL_functions import cy_index



def auto(pathway, circ = ['Rp', 'Rp'], plot = '',resttime = 50, save = '',fit_para = 0, cy = [1,2,5,10,20,30,50],l=0.14, r_cc=5, PEIS_evaluation = True, lf_limit=''):

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
            data_eva = Nynquist(meta['data'][entry], circ = circ[circ_count], fit_para = fit_para,lf_limit=lf_limit)
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
                d = eva_GCPL(meta['data'][entry], meta['active material mass'], meta['electrode surface area'])
                if plot == '':
                    plot_galv(d, save = save, cy = cy)
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

            data_eva = Nynquist(meta['data'][entry], circ = circ[circ_count], fit_para = fit_para)
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

            op = []
            cy_in = cy_index(data)
            for i in cy_in[1]:
                op.append(data['Ewe/V'].loc[i])
            print(entry + ' evaluated.')
            d['over potential (V)'] = op
            d['areal current (mA/cm2)'] = I_area

            d_eva[entry] = d
            end = time.time()
            print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
            
            if not (cc_switch[0] and cc_switch[1]) is False:
                # Merge the CC files from negative and positive currents
                d_cc = {}
                ID = 'CC_'+str(cc_switch[0][0].split(' ')[0])+'_'+str(cc_switch[1][0].split(' ')[0])
                df_cc = pd.concat([meta['data'][cc_switch[0][0]], meta['data'][cc_switch[1][0]]], sort=False).sort_index()
                op = []
                cy_in = cy_index(df_cc)
                for i in cy_in[1]:
                    op.append(df_cc['Ewe/V'].loc[i])

                if isclose(abs(cc_switch[0][1]),abs(cc_switch[1][1])) is True:
                    I_cc_areal = (abs(cc_switch[0][1])+abs(cc_switch[1][1]))/2
                else:
                    I_cc_areal = (cc_switch[0][1],cc_switch[1][1])
                    print('Chage and Discharge Currents are different.')

                d_cc['Areal Current (mA/cm2)'] = I_cc_areal
                d_cc['df'] = df_cc
                d_cc['over potential (V)'] = op
                d_eva[ID] = d_cc
                meta['eva'] = d_eva
                cc_switch = [False,False]
                if plot == '':
                    strip_plate(df_cc, title = str(I_cc_areal)+' $\mathregular{mA\,cm^{-2}}$')

            
            

    meta['eva'] = d_eva

    return(meta)
    