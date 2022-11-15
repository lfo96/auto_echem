import time
import cmath
import numpy as np
from numpy.lib.nanfunctions import _nanquantile_dispatcher
from scipy.integrate import simps
import pandas as pd

#from auto_echem.general_functions import info
#from auto_echem.general_functions import loop_finder
#from auto_echem.general_functions import tech
#from auto_echem.general_functions import time_limit
#from auto_echem.general_functions import color_gradient
#from auto_echem.general_functions import layout
#from auto_echem.general_functions import find_nearest
#from auto_echem.general_functions import calc_I
#from auto_echem.impedance_functions import eva_PEIS

from impedance.preprocessing import ignoreBelowX
from impedance.models.circuits  import *

from scipy.signal import argrelextrema

def threeEl(pathway):
    files = info(pathway)
    d = {}
    for entry in files['data']:
        if entry.split(' ')[1] == 'PEIS':
            d['PEIS_'+str(entry.split(' ')[0])] = eva_PEIS(files['data'][entry])
    
        elif entry.split(' ')[1] == 'GCPL': 
            d['GCPL_'+str(entry.split(' ')[0])] = files['data'][entry]
    return(d,files['active material mass'])


def eva_GCPL_3El(dis_index,dis_data,cha_index,cha_data, m_am, I_calc = False):
    galv = []
    cap_dis = []
    cap_cha = []
    en_dis = []
    en_cha = []
    cy_no = len(dis_index[0])

    for cycle in range(cy_no):
        dis = dis_data.loc[dis_data['(Q-Qo)/mA.h']!= 0].loc[dis_index[0][cycle]:dis_index[1][cycle]]
        grav_dis = abs(dis['(Q-Qo)/mA.h']/(0.001*m_am))
        cap_dis.append(grav_dis.iloc[-1])

        try:
            cha = cha_data.loc[cha_data['(Q-Qo)/mA.h']!= 0].loc[cha_index[0][cycle]:cha_index[1][cycle]]
        except IndexError:
            cha = pd.DataFrame(np.array([[np.nan,np.nan,np.nan,np.nan,np.nan]]), columns=['(Q-Qo)/mA.h', 'Ewe/V', 'Ece/V','time/s','control/V/mA'])
        grav_cha = abs(cha['(Q-Qo)/mA.h']/(0.001*m_am))
        cap_cha.append(grav_cha.iloc[-1])

        d_galv = {
                'Discharge Time (s)' : dis['time/s'],
                'Gravimetric Discharge Capacity (mAh/g)' : grav_dis,
                'WE Discharge Potential (V)' : dis['Ewe/V'],
                'CE Discharge Potential (V)' : dis['Ece/V'],
                'Discharge Current (mA)' : dis['control/V/mA'],
                'Discharge Q (mAh)' : dis['(Q-Qo)/mA.h'],
                'Charge Time (s)' : cha['time/s'],
                'Gravimetric Charge Capacity (mAh/g)' : grav_cha,
                'WE Charge Potential (V)' : cha['Ewe/V'],
                'CE Charge Potential (V)' : cha['Ece/V'],
                'Charge Current (mA)' : cha['control/V/mA'],
                'Charge Q (mAh)' : cha['(Q-Qo)/mA.h'],
        }

        if I_calc == True:
            I_calculated_d = pd.Series(calc_I(dis['time/s'],dis['(Q-Qo)/mA.h']),index=[dis['time/s'].index])
            #I_calculated_d.reindex([dis['time/s'].index])
            I_calculated_c = pd.Series(calc_I(cha['time/s'],cha['(Q-Qo)/mA.h']),index=[cha['time/s'].index])
            #I_calculated_c.reindex([cha['time/s'].index])
            d_galv['I_calc discharge (mA)'] = I_calculated_d
            d_galv['I_calc charge (mA)'] = I_calculated_c

        galv.append(d_galv)
        
        en_dis_cy, en_cha_cy = simps(dis['Ewe/V']+dis['Ece/V'],grav_dis), simps(cha['Ewe/V']+cha['Ece/V'],grav_cha)

        en_dis.append(en_dis_cy)
        en_cha.append(en_cha_cy)

    ce = np.array(cap_dis)/np.array(cap_cha)
    ee = np.array(en_dis)/np.array(en_cha)

    eva = {
        "Cycle" : range(1,cy_no+1),
        "Gravimetric Discharge Capacity (mAh/g)" : cap_dis,
        "Gravimetric Charge Capacity (mAh/g)": cap_cha,
        "Coulombic Efficency (%)" : ce,
        "Discharge Energy (mWh/g)": en_dis,
        "Charge Energy (mWh/g)": en_cha,
        "Energy Efficency (%)" : ee,
    }
    eva = pd.DataFrame(eva)
    return(galv,eva)



def eva_threeEl(pathway, I_calc = False, plot = ''):
    _3Elec = threeEl(pathway)
    m_am = _3Elec[1]
    d = {
        'data': _3Elec[0],
        'active material mass': m_am
    }
    count = 0
    counter = 1

    for entry in _3Elec[0]:
        if entry.split('_')[0] == 'PEIS':
            if plot == '':
                plot_PEIS(_3Elec[0][entry], tit = entry, label = 'off')
        if entry.split('_')[0] == 'GCPL':
            if count == 0:
                dis_index = cy_index(_3Elec[0][entry])
                dis_data = _3Elec[0][entry]
                count += 1
            else:
                cha_index = cy_index(_3Elec[0][entry])
                cha_data = _3Elec[0][entry]
                if plot == '':
                    plot_GCPL(dis_data, dis_index, cha_data, cha_index, m_am)
                    plot_GCPL(dis_data, dis_index, cha_data, cha_index, m_am, electrode = 'Ece/V')
                eva = eva_GCPL_3El(dis_index,dis_data,cha_index,cha_data,m_am, I_calc=I_calc)
                d['galv_GCPL_'+str(counter)] = eva[0]
                d['eva_GCPL_'+str(counter)] = eva[1]
                counter += 1
                count = 0
    return(d)


def plot_PEIS(data, fit = '', limit = 0, tit = '', save = '', label = ''):
    fig, ax = plt.subplots()
    counter = 0
    colors = color_gradient(len(data))

    if label == 'off':
        des = ''
    elif label != '':
        des = label

    if fit == '':
        #plotting only the experimental data
        for entry in data:
            if label == '':
                des = str(counter+1)+". cycle"

            trace = ax.plot(entry[1], entry[2],'o--', linewidth = 1, label = des)
            trace[0].set_color(colors[counter])
            counter += 1
    else:
        # if fitting data is provided add this to the plot
        for exp,fit in zip(data,fit):
            if label == '':
                des = str(counter+1)+". cycle"
            trace_exp = ax.plot(exp[1],exp[2],'o', linewidth = 1 , label = des)
            trace_fit = ax.plot(fit[1],fit[2],'-', linewidth = 1)# , label = str(counter+1)+". cycle")
            trace_exp[0].set_color(colors[counter])
            trace_fit[0].set_color(colors[counter])
            counter += 1
    if limit == 0:
        layout(ax, x_label = 'Re(Z) (Ohm)', y_label = '-Im(Z) (Ohm)', square = "yes", title = tit)
    else:
        layout(ax, x_label = 'Re(Z) (Ohm)', y_label = '-Im(Z) (Ohm)',x_lim = [0,limit], y_lim = [0,limit], square = "yes", title = tit)
    if save != '':
        plt.savefig(save+'.svg', bbox_inches='tight',transparent = True)


def cy_index(GCPL):
# determine the start end point indexes for each cycle.
    index_list = GCPL.loc[GCPL['control/V/mA']!= 0].index
    j = index_list[0]-1
    start = [index_list[0]]
    end  = []
    for i in index_list:
        #j is a number increasing by 1 every repetion through the loop. i are indix points of the GCPL cycle where the current unequeals 0. If i is more than +1 of j that means there was an index jump. This marks the new half cycle. 
        if i == j+1:
            j = i
        else:
            end.append(j)
            j = i
            start.append(i)
    if len(start) != len(end):
        # in case the cycle is incomplete, add the last point of  the index list as end point.
        end.append(index_list[-1])
    return(start,end)


def plot_GCPL(dis_data, dis_index, cha_data, cha_index, m_am, electrode = 'Ewe/V', save = '', x_lim = '', y_lim = ''):
    '''
    Plot the GCPL corresponding to a three electrode measurement. 
    Insert data in the dictionary form of the threeEl function, the start and end points of the charging cycles obtained by the cy_index function, and specify the electrode system to be plotted.
    Specify electrodes: 'Ewe/V' or 'Ece/V' 
    '''
    cycle = 1
    colors = color_gradient(len(dis_index[0]))
    fig, ax = plt.subplots()
    for i in range(len(dis_index[0])): 
        plt.plot(abs(dis_data.loc[dis_data['control/V/mA']!= 0].loc[dis_index[0][i]:dis_index[1][i]]['(Q-Qo)/mA.h'])/(0.001*m_am),dis_data.loc[dis_data['control/V/mA']!= 0].loc[dis_index[0][i]:dis_index[1][i]][electrode], color = colors[i])#, label = str(cycle)+". cycle")

        try:
            plt.plot(abs(cha_data.loc[cha_data['control/V/mA']!= 0].loc[cha_index[0][i]:cha_index[1][i]]['(Q-Qo)/mA.h'])/(0.001*m_am),cha_data.loc[cha_data['control/V/mA']!= 0].loc[cha_index[0][i]:cha_index[1][i]][electrode], color = colors[i])
        except IndexError:
            break
        cycle += 1
    layout(ax, x_label = r'Gravimetric Capacity ($\mathregular{mAh\,g^{-1}}$)', y_label = r'$\mathregular{E\ (V\ vs\ Li^+/Li)}$', x_lim = x_lim, y_lim = y_lim, title = electrode)
    if save != '':
        plt.savefig(save+'.svg',bbox_inches='tight', transparent = True)
        
        

def eva_potstat(galv):
    colors = color_gradient(len(galv))
    fig,ax = plt.subplots()
    #ax2 = ax.twinx()
    for cy in range(len(galv)):
        index_d = galv[cy]['WE Discharge Potential (V)'].loc[galv[cy]['WE Discharge Potential (V)']<=1.20].index[0]
        t_d = galv[cy]['Discharge Time (s)'].loc[index_d:]-galv[cy]['Discharge Time (s)'][index_d]

        try:
            index_c = galv[cy]['WE Charge Potential (V)'].loc[galv[cy]['WE Charge Potential (V)']>=3.95].index[0]
        except IndexError:
            ax.plot(t_d,galv[cy]['I_calc discharge (mA)'].loc[index_d:], color = colors[cy])
            break
        t_c = galv[cy]['Charge Time (s)'].loc[index_c:]-galv[cy]['Charge Time (s)'][index_c]
                       
        ax.plot(t_d,galv[cy]['I_calc discharge (mA)'].loc[index_d:], color = colors[cy])
        ax.plot(t_c,galv[cy]['I_calc charge (mA)'].loc[index_c:], color = colors[cy])
        #ax2.plot(t_d,galv[cy]['CE Discharge Potential (V)'].loc[index_d:],color = colors[cy],linestyle='--')
    layout(ax,x_label='time (s)',y_label = 'Current (mA)', y_lim = [-0.010,0.010])
    #layout(ax2,y_label = 'Potential CE (V)')


# Evaluation of a potentiostatic step after GCPL. 
def eva_potstat_d(galv):
    colors = color_gradient(len(galv))
    cap_noPotstat_lst = []
    cap_lst = []
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    for cy in range(len(galv)):
        try:    
            index_d = galv[cy]['WE Discharge Potential (V)'].loc[galv[cy]['WE Discharge Potential (V)']<=1.20].index[0]
        except IndexError:
            break
        t_d = (galv[cy]['Discharge Time (s)'].loc[index_d:]-galv[cy]['Discharge Time (s)'][index_d])/3600
        cap_noPotstat = galv[cy]['Gravimetric Discharge Capacity (mAh/g)'].loc[index_d]
        cap = galv[cy]['Gravimetric Discharge Capacity (mAh/g)'].iloc[-1]
        cap_noPotstat_lst.append(cap_noPotstat)
        cap_lst.append(cap)   
                          
        ax.scatter(t_d,galv[cy]['I_calc discharge (mA)'].loc[index_d:], color = colors[cy])
        ax2.plot(t_d,galv[cy]['CE Discharge Potential (V)'].loc[index_d:],color = colors[cy],linestyle='--')
    layout(ax,x_label='time (h)',y_label = 'Current (mA)', y_lim = [-0.010,0.010])
    layout(ax2,y_label = 'Potential CE (V)')
    return(cap_noPotstat_lst,cap_lst)



def eva_potstat_c(galv):
    '''
    Insert the galv raw data from three El file. Returns a list of capacities prior to the 
    '''
    colors = color_gradient(len(galv))
    fig,ax = plt.subplots()
    ax2 = ax.twinx()
    cap_noPotstat_lst = []
    cap_lst = []
    for cy in range(len(galv)):
        try:
            index_c = galv[cy]['WE Charge Potential (V)'].loc[galv[cy]['WE Charge Potential (V)']>=3.98].index[0]
        except IndexError:
            break
        t_c = (galv[cy]['Charge Time (s)'].loc[index_c:]-galv[cy]['Charge Time (s)'][index_c])/3600  
        cap_noPotstat = galv[cy]['Gravimetric Charge Capacity (mAh/g)'].loc[index_c]
        cap = galv[cy]['Gravimetric Charge Capacity (mAh/g)'].iloc[-1]
        cap_noPotstat_lst.append(cap_noPotstat)
        cap_lst.append(cap)    
        ax.scatter(t_c,galv[cy]['I_calc charge (mA)'].loc[index_c:], color = colors[cy])
        ax2.plot(t_c,galv[cy]['CE Charge Potential (V)'].loc[index_c:],color = colors[cy],linestyle='--')
    layout(ax,x_label='time (h)',y_label = 'Current (mA)', y_lim = [-0.010,0.010])
    layout(ax2,y_label = 'Potential CE (V)')
    return(cap_noPotstat_lst,cap_lst)






