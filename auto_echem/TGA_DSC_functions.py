import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import numpy as np
import datetime as dt

from itertools import islice
from auto_echem.general_functions import layout

dict_ms = {
    12 : ['C+','grey', '-'],
    13 : ['CH+','blue', '--'],
    14 : ['N+, CH2+','blue', '--'],
    15 : ['CH3+, NH+','red', '--'],
    16 : ['O+, CH4+','red', '--'],
    17 : ['OH+, NH3+','red', '--'],
    18 : ['H2O+','orange', '--'],
    19 : ['F+, A++?','orange', '--'],
    20 : ['Ar++','orange', '--'],
    21 : ['Unkonwn Ar contamination','orange', '--'],
    22 : ['CO2++','orange', '-'],
    26 : ['C2H2+','orange', '-'],
    27 : ['C2H3+','orange', '-'],
    28 : ['N2+, CO+','blue', '-'],
    29 : ['N2+','blue', '-'],
    30 : ['NO+','blue', '-'],
    32 : ['O2+, S+','red', '-'],
    34 : ['H2S+, S+','red', '-'],
    35 : ['Cl+ isotope','orange', '-'],
    36 : ['Ar+ isotope','orange', '-'],
    37 : ['Cl+','orange', '-'],
    38 : ['Ar+ isotope','orange', '-'],
    39 : ['K+ isotope','orange', '-'],
    40 : ['Ar+','orange', '-'],
    41 : ['C3H5+','orange', '-'],
    42 : ['C3H6+','orange', '-'],
    43 : ['C3H7+','orange', '-'],
    44 : ['CO2+','green', '-'],
    45 : ['CO2+ isotope','green', '-'],
    46 : ['C2H5OH+','orange', '-'],
    48 : ['SO+','green', '-'],
    64 : ['SO2+','green', '-'],
    }

def TGA_DSC(pathway, plot = True, save = '', x_lim='', y_lim ='', y2_lim = '', m_real = ''):
    with open(pathway,encoding= 'unicode_escape') as fin:
        header = 50
        head_len = 0
        for line in islice(fin, 0,header):
                #print(line.split(":")[0])
                if line.split(":")[0]=="#SAMPLE MASS /mg": #extract the active material mass
                    m_am = float(line.split(",")[1][0:5])
                if line[0:2] == '##':
                    break
                head_len += 1

    df = pd.read_csv(pathway,encoding= 'unicode_escape', header=head_len-1)
    try:
        df['Mass loss (mg)'] = ((df['Mass/%'].iloc[0]-df['Mass/%'])/100)*m_am*-1
        df['DSC/uV'] = df['DSC/(uV/mg)']*m_am
    except KeyError:
        print('Blank measurement found.')
        df['Mass/%'] = 100*df['Mass loss/mg']/df['Mass loss/mg'].iloc[0]
    if m_real != '':
        df['Mass corrected (%)'] = 100-((df['Mass loss (mg)']/m_real)*-100)

    meta = {
        'sample mass' : m_am,
        'data' : df   
    }

    if plot == True:
        if m_real != '':
            plot_TGA_DSC(meta['data'], save = save, correction = True, x_lim = x_lim, y_lim = y_lim, y2_lim = y2_lim)
        else:
            plot_TGA_DSC(meta['data'], save = save, x_lim = x_lim, y_lim = y_lim, y2_lim = y2_lim)


    return(meta)

def plot_TGA_DSC(data, save = '', correction = False, x_lim= '', y_lim = '', y2_lim = ''):
    fig,ax = plt.subplots()
    if correction == True:
        ax.plot(data['##Temp./C'],data['Mass corrected (%)'], color = 'black')
    else:
        ax.plot(data['##Temp./C'],data['Mass/%'], color = 'black')
    
    #plt.axis('off')

    if x_lim != '':
        layout(ax,x_lim=x_lim,x_label='Temperature (\N{DEGREE SIGN}C)', y_label='Mass (%)')
    if y_lim != '':
        layout(ax,y_lim=y_lim,x_label='Temperature (\N{DEGREE SIGN}C)', y_label='Mass (%)')
    else:
        layout(ax,x_label='Temperature (\N{DEGREE SIGN}C)', y_label='Mass (%)')

    color_ax2 = 'blue'
    ax2 = ax.twinx()
    ax2.set_ylabel('DSC', color=color_ax2,fontsize = 16)  # we already handled the x-label with ax1
    ax2.tick_params(axis='y', labelcolor=color_ax2)
    ax2.tick_params(direction='in', length=6, width=1.5, color = color_ax2)
    ax2.spines["right"].set_color(color_ax2)
    figure = plt.gca()
    y_axis = figure.axes.get_yaxis()
    y_axis.set_visible(False)
    try: 
        ax2.plot(data['##Temp./C'],data['DSC/(uV/mg)']*-1, color = color_ax2)
    except KeyError:
        print('No DSC file found.')
    if y2_lim == '':
        layout(ax2)
    else:
        layout(ax2, y_lim = y2_lim)
    if save != '':
        plt.savefig(save+'.svg')
    layout(ax2)


def eva_MS(pathway, plot = True, save = '', x_lim='', y_lim ='', y2_lim = '', m_real = ''):
    with open(pathway,encoding= 'unicode_escape') as fin:
        header = 50
        head_len = 0
        for line in islice(fin, 0,header):
                if line[0:2] == '##':
                    break
                head_len += 1

    df = pd.read_csv(pathway,encoding= 'unicode_escape', header=head_len-1)
    header_name = ['Temperature (C)','Time (min)','TIC (A)']
    for i in range(10,len(df.columns.values)+7):
        header_name.append('mz_'+str(i)+' (A)')

    df = pd.read_csv(pathway,encoding= 'unicode_escape', header=head_len-1, names=header_name)
    return(df)

def correction_BL_blank(file,blank,multiplier=1,BL_T = 70, BL_m_T = False):
    # Simply substraction 
    correction = file['data']['DSC/uV']-multiplier*(blank[:file['data'].index[-1]+1])
    cor = {}
    cor['sample mass'] = file['sample mass']
    data = {}
    data['DSC/uV'] = correction
    data['##Temp./C'] = file['data']['##Temp./C']
    data['Time/min'] = file['data']['Time/min']
    data['DSC/(uV/mg)'] = correction/cor['sample mass']
    data['Mass corrected (%)'] = file['data']['Mass corrected (%)']
    if BL_m_T != False:
        BL_m_cor = 100-file['data']['Mass corrected (%)'].loc[file['data']['##Temp./C']>=BL_m_T].iloc[0]
        data['Mass corrected (%)'] = file['data']['Mass corrected (%)']+BL_m_cor
        
    cor['data'] = pd.DataFrame(data)
    baseline_cor = cor['data']['DSC/(uV/mg)'].loc[cor['data']['##Temp./C']>=BL_T].iloc[0]
    cor['data']['DSC/(uV/mg)'] = data['DSC/(uV/mg)']-baseline_cor
    return(cor)

def m_bl(file,T_m_bl):
    try:
        BL_m_cor = 100-file['data']['Mass corrected (%)'].loc[file['data']['##Temp./C']>=T_m_bl].iloc[0]
    except KeyError:
        BL_m_cor = 100-file['data']['Mass/%'].loc[file['data']['##Temp./C']>=T_m_bl].iloc[0]
    file['data']['Mass corrected (%)'] = file['data']['Mass/%']+BL_m_cor
    return(file)


def Li_melt(file, T1 = 180,T2 = 184.2):
    '''
    Insert evaluated TGA-DSC file and T1, T2 of the Lithium melting point and returns the area under the Lithium melting peak [0] and the T vs DSC uV/mg at [1].
    Li melting from inSitu DSC Cell is 0.48943397434941976 J. See calculation below.
    '''
    Li_melt_en = 0.48943397434941976 # J
    Li_melt_df = file['data'][(file['data']['##Temp./C']<=T2) & (file['data']['##Temp./C']>=T1)]
    BL_subs = Li_melt_df['DSC/(uV/mg)'].iloc[0]
    Li_area = simps((Li_melt_df['DSC/(uV/mg)']-BL_subs),Li_melt_df['##Temp./C'])
    
    
    plt.plot(Li_melt_df['##Temp./C'],Li_melt_df['DSC/(uV/mg)']-BL_subs)
    plt.hlines(0,T1,T2, color = 'black', linestyles='--')
    return(Li_area,[Li_melt_df['##Temp./C'],Li_melt_df['DSC/(uV/mg)']])
