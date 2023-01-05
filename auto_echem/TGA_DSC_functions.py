import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import numpy as np
import datetime as dt

from itertools import islice
from auto_echem.general_functions import layout

def TGA_DSC(pathway, plot = True, save = '', x_lim='', y_lim ='', y2_lim = '', m_real = ''):
#pathway = FSI_p
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
    

