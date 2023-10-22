import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import datetime
from matplotlib import cm
# import gspread

from auto_echem.general_functions import info
from auto_echem.general_functions import layout
from auto_echem.general_functions import isclose
from auto_echem.general_functions import data_set



# from .neware_reader_master.neware import *
from scipy.integrate import simps

def has_decreasing_numbers(data):
    """
    Check if a data set has any decreasing numbers.

    Parameters:
    - data: List or array of data points.

    Returns:
    - True if there are decreasing numbers; False otherwise.
    """
    for i in range(1, len(data)):
        if data[i] < data[i - 1]:
            return True
    return False

def eva_GCPL_index(data, m_am, A_el):
    '''
    Alternative GCPL analysis for files that are werid where the half cycle number is not increasing linarly but rather has jumps in tehre...
    '''
    index = cy_index(data)
    galv = []
    cap_dis = []
    cap_cha = []
    en_dis = []
    en_cha = []
    cy_no = len(index[0])
    I_areal = []
    I_specific = []
    dis_counter = 0
    cha_counter = 0
    cy_counter = 0
    for i in range(cy_no):
        df_cy = data.loc[data['control/V/mA']!= 0].loc[index[0][i]:index[1][i]]
        I = df_cy['control/V/mA'].mean()
        if I <= 0:
            dis = df_cy
            t_0_d = dis['time/s'].iloc[0]
            grav_cap_dis = abs(dis['Q charge/discharge/mA.h'])/(0.001*m_am) 
            cap_dis.append(grav_cap_dis.iloc[-1])
            dis_counter += 1

        elif I >= 0:
            cha = df_cy
            t_0_c = cha['time/s'].iloc[0]
            grav_cap_cha = abs(cha['Q charge/discharge/mA.h'])/(0.001*m_am)
            cap_cha.append(grav_cap_cha.iloc[-1]) 
            cha_counter += 1

        if cha_counter == 1 and dis_counter == 1:
            # end of dis and recharge half cycle. Add the values to galv file
            d_galv = {
                    'Discharge Time (s)' : dis['time/s']-t_0_d,
                    'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                    'Discharge Potential (V)' : dis['Ewe/V'],
                    'Discharge Current (mA)' : dis['control/V/mA'],
                    'Discharge Q (mAh)' : dis['(Q-Qo)/mA.h'],
                    'Charge Time (s)' : cha['time/s']-t_0_c,
                    'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                    'Charge Potential (V)' : cha['Ewe/V'],
                    'Charge Current (mA)' : cha['control/V/mA'],
                    'Charge Q (mAh)' : cha['(Q-Qo)/mA.h'],
            }

            galv.append(d_galv)            
            en_dis_cy, en_cha_cy = simps(dis['Ewe/V'],grav_cap_dis), simps(cha['Ewe/V'],grav_cap_cha)

            en_dis.append(en_dis_cy)
            en_cha.append(en_cha_cy)
            I_areal.append(round(1000*I/A_el,3))
            I_specific.append(round(I/m_am,3))
            
            cy_counter += 1
            dis_counter = 0
            cha_counter = 0
    ce = np.array(cap_dis)/np.array(cap_cha)
    ee = np.array(en_dis)/np.array(en_cha)
    eva = {
        "Cycle" : range(1,cy_counter+1),
        "Gravimetric Discharge Capacity (mAh/g)" : cap_dis,
        "Gravimetric Charge Capacity (mAh/g)": cap_cha,
        "Coulombic Efficency (%)" : ce,
        "Discharge Energy (mWh/g)": en_dis,
        "Charge Energy (mWh/g)": en_cha,
        "Energy Efficency (%)" : ee,
        "Areal Current (\u03BCA/cm$^2$)" : I_areal,
        "Specific Current (mA/g)" : I_specific,
    }
    eva = pd.DataFrame(eva)
    return(eva,galv)


def eva_GCPL(df,m_am,A_el):
    cycles_tot = df["half cycle"].iloc[-1] #get the total number of half cycles
    cap_dis = []
    cap_dis_calc = []
    cap_cha = []
    cap_cha_calc = []
    cap_dis_areal = []
    cap_cha_areal = []
    ce = []
    energy_dis = []
    energy_dis_calc = []
    energy_cha = []
    energy_cha_calc = []
    galv = []
    I_areal = []
    I_specific = []
    dis_counter = 0
    cha_counter = 0
    cy_no = 1
    for cycle in range(int(cycles_tot)+1):
        data_cycle = df.loc[(df["half cycle"]==cycle) & (df['control/V/mA']!= 0)] #only display corresponding to a cycle and with current different to zero
        if len(data_cycle)<=1:
            continue 
        I = data_cycle['control/V/mA'].mean()
        if I <= 0:
            dis = data_cycle
            t_0_d = dis['time/s'].iloc[0]
            cap_dis.append(abs(dis['Q charge/discharge/mA.h']).iloc[-1]/(0.001*m_am))
            dis_counter += 1
        elif I >= 0:
            cha = data_cycle
            t_0_c = cha['time/s'].iloc[0]
            cap_cha.append(abs(cha['Q charge/discharge/mA.h']).iloc[-1]/(0.001*m_am)) 
            cha_counter += 1

        if cha_counter == 1 and dis_counter == 1:
            # end of dis and recharge half cycle. Add the values to galv file.
            grav_cap_dis = abs(dis['Q charge/discharge/mA.h'])/(0.001*m_am) 

            grav_cap_cha = abs(cha['Q charge/discharge/mA.h'])/(0.001*m_am)

            t_dis = dis['time/s']-t_0_d
            grav_cap_dis_calc = abs(dis['control/V/mA'])*(t_dis/3600)/(0.001*m_am)
            cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
            t_cha = cha['time/s']-t_0_c
            grav_cap_cha_calc = abs(cha['control/V/mA'])*(t_cha/3600)/(0.001*m_am)
            cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
            #galv.append(((grav_cap_dis,dis["Ewe/V"],t_dis),(grav_cap_cha,cha["Ewe/V"], t_cha)))
            d_galv = {
                'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
                'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
                'Discharge Potential (V)' : dis["Ewe/V"],
                'Charge Potential (V)' : cha["Ewe/V"],
                'Discharge Time (s)' : t_dis,
                'Charge Time (s)' : t_cha,
                'Discharge Current (mA)' : dis['control/V/mA'],
                'Charge Current (mA)' : cha['control/V/mA']
            }
            galv.append(d_galv)
            # calculate energy by integrating the capacity in mWh/g. 
            en_dis, en_cha = simps(dis["Ewe/V"],grav_cap_dis), simps(cha["Ewe/V"],grav_cap_cha)
            energy_dis.append(en_dis)
            energy_cha.append(en_cha)

            en_dis_calc, en_cha_calc = simps(dis["Ewe/V"],grav_cap_dis_calc), simps(cha["Ewe/V"],grav_cap_cha_calc)
            energy_dis_calc.append(en_dis_calc)
            energy_cha_calc.append(en_cha_calc)




            #Areal Current: Determine if charge and discharge current is the same and include in the panda file
            if isclose(abs(cha['control/V/mA'].mode().mean()),abs(dis['control/V/mA'].mode().mean()), rel_tol=0.01) == True:
                I = ((1000*abs(cha['control/V/mA'].mode().mean()))+(1000*abs(dis['control/V/mA'].mode().mean())))/2
                I_areal.append(round(I/A_el,3))
                I_specific.append(round(I/m_am,3))
            else:
                I = [1000*dis['control/V/mA'].mode().mean(),1000*cha['control/V/mA'].mode().mean()]
                I_areal.append([round(I[0]/A_el,3), round(I[1]/A_el,3)])
                I_specific.append([round(I[0]/m_am,3), round(I[1]/m_am,3)])
                
            cy_no += 1
            dis_counter = 0
            cha_counter = 0
        elif cycle == int(cycles_tot):
            if cha_counter ==1:
                # define the missing parameter which would have been determined if cha_counter was 1.
                cap_dis.append(abs(dis['Q charge/discharge/mA.h']).iloc[-1]/(0.001*m_am))
                t_0_d = dis['time/s'].iloc[0]
                
                # proceed as normal...


                grav_cap_dis = abs(dis['Q charge/discharge/mA.h'])/(0.001*m_am)
                grav_cap_cha = abs(cha['Q charge/discharge/mA.h'])/(0.001*m_am)
                
                t_dis = dis['time/s']-t_0_d
                grav_cap_dis_calc = abs(dis['control/V/mA'])*(t_dis/3600)/(0.001*m_am)
                cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
                t_cha = cha['time/s']-t_0_c
                grav_cap_cha_calc = abs(cha['control/V/mA'])*(t_cha/3600)/(0.001*m_am)
                cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
            #galv.append(((grav_cap_dis,dis["Ewe/V"],t_dis),(grav_cap_cha,cha["Ewe/V"], t_cha)))
                d_galv = {
                    'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                    'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
                    'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                    'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
                    'Discharge Potential (V)' : dis["Ewe/V"],
                    'Charge Potential (V)' : cha["Ewe/V"],
                    'Discharge Time (s)' : t_dis,
                    'Charge Time (s)' : t_cha,
                    'Discharge Current (mA)' : dis['control/V/mA'],
                    'Charge Current (mA)' : cha['control/V/mA']
                }
                galv.append(d_galv)

                # calculate energy by integrating the capacity in mWh/g. 
                en_dis, en_cha = simps(dis["Ewe/V"],grav_cap_dis), simps(cha["Ewe/V"],grav_cap_cha)
                energy_dis.append(en_dis)
                energy_cha.append(en_cha)

                en_dis_calc, en_cha_calc = simps(dis["Ewe/V"],grav_cap_dis_calc), simps(cha["Ewe/V"],grav_cap_cha_calc)
                energy_dis_calc.append(en_dis_calc)
                energy_cha_calc.append(en_cha_calc)

                I_areal.append(round(I/A_el,3))
                I_specific.append(round(I/m_am,3))

                cy_no += 1
                print("Incomplete last cycle - discharge added.")
            elif dis_counter ==1:
                # define the missing parameter which would have been determined if cha_counter was 1.
                cap_cha.append(abs(cha['Q charge/discharge/mA.h']).iloc[-1]/(0.001*m_am))
                t_0_c = cha['time/s'].iloc[0]
                
                # proceed as normal...

                #galv.append((0,0,(abs(cha['Q charge/discharge/mA.h'])/(0.001*m_am),cha["Ewe/V"])))
                grav_cap_dis = abs(dis['Q charge/discharge/mA.h'])/(0.001*m_am)
                grav_cap_cha = abs(cha['Q charge/discharge/mA.h'])/(0.001*m_am)
                
                t_dis = dis['time/s']-t_0_d
                grav_cap_dis_calc = abs(dis['control/V/mA'])*(t_dis/3600)/(0.001*m_am)
                cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
                t_cha = cha['time/s']-t_0_c
                grav_cap_cha_calc = abs(cha['control/V/mA'])*(t_cha/3600)/(0.001*m_am)
                cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
            #galv.append(((grav_cap_dis,dis["Ewe/V"],t_dis),(grav_cap_cha,cha["Ewe/V"], t_cha)))
                d_galv = {
                    'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                    'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
                    'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                    'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
                    'Discharge Potential (V)' : dis["Ewe/V"],
                    'Charge Potential (V)' : cha["Ewe/V"],
                    'Discharge Time (s)' : t_dis,
                    'Charge Time (s)' : t_cha,
                    'Discharge Current (mA)' : dis['control/V/mA'],
                    'Charge Current (mA)' : cha['control/V/mA']
                }
                galv.append(d_galv)

                # calculate energy by integrating the capacity in mWh/g. 
                en_dis, en_cha = simps(dis["Ewe/V"],grav_cap_dis), simps(cha["Ewe/V"],grav_cap_cha)
                energy_dis.append(en_dis)
                energy_cha.append(en_cha)

                en_dis_calc, en_cha_calc = simps(dis["Ewe/V"],grav_cap_dis_calc), simps(cha["Ewe/V"],grav_cap_cha_calc)
                energy_dis_calc.append(en_dis_calc)
                energy_cha_calc.append(en_cha_calc)

                I_areal.append(round(I/A_el,3))
                I_specific.append(round(I/m_am,3))

                cy_no += 1
                print("Incomplete last cycle - charge added.")
    
    ce = 100*np.array(cap_dis)/np.array(cap_cha) 
    ce_calc = 100*np.array(cap_dis_calc)/np.array(cap_cha_calc) 
    energy_ef = 100*np.array(energy_dis)/np.array(energy_cha)
    energy_ef_calc = 100*np.array(energy_dis_calc)/np.array(energy_cha_calc)
        
    cap_dis_areal = np.array(cap_dis)*0.001*m_am/A_el
    cap_cha_areal = np.array(cap_cha)*0.001*m_am/A_el
    
    d = {"Cycle" : range(1,cy_no),
         "Gravimetric Discharge Capacity (mAh/g)" : cap_dis,
         "Gravimetric Discharge Capacity Calculated (mAh/g)" : cap_dis_calc,
         "Areal Discharge Capacity (mAh/cm$^2$)" : cap_dis_areal,
         "Gravimetric Charge Capacity (mAh/g)": cap_cha,
         "Gravimetric Charge Capacity Calculated (mAh/g)" : cap_cha_calc,
         "Areal Charge Capacity (mAh/cm$^2$)" : cap_cha_areal,
         "Coulombic Efficency (%)" : ce,
         "Coulombic Efficency Calculated (%)" : ce_calc,
         "Discharge Energy (mWh/g)": energy_dis,
         "Discharge Energy Calculated (mWh/g)": energy_dis_calc,
         "Charge Energy (mWh/g)": energy_cha,
         "Charge Energy Calculated (mWh/g)": energy_cha_calc,
         "Energy Efficency (%)" : energy_ef,
         "Energy Efficenecy Calculated (%)" : energy_ef_calc,
         "Areal Current (\u03BCA/cm$^2$)" : I_areal,
         "Specific Current (mA/g)" : I_specific
         }
    eva = pd.DataFrame(d)
    return (eva,galv)

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

def add_first(first, subs):
    '''
    Merge the first GCPL cycle into the evaluated GCPL data file.
    Insert eva_GCPL files. 
    Returns a tuple in the same (eva,galv) format as the eva_GCPL function.
    '''
    eva_merged = pd.concat([first[0],subs[0]],ignore_index=True)
    # adjust cycle number
    eva_merged['Cycle'] = range(1,len(eva_merged.index)+1) 

    galv_merged = first[1] + subs[1]
    
    return(eva_merged,galv_merged)

# Plotting Functions

def plot_GCPL(data, dis_index, cha_index, electrode = 'Ewe/V', save = ''):
    '''
    Plot the GCPL corresponding to a three electrode measurement. 
    Insert data in the dictionary form of the threeEl function, the start and end points of the charging cycles obtained by the cy_index function, and specify the electrode system to be plotted.
    Specify electrodes: 'Ewe/V' or 'Ece/V' 
    '''
    m_am = data['active material mass']
    cycle = 1
    colors = color_gradient(len(dis_index[0]))
    fig, ax = plt.subplots()
    for i in range(len(dis_index[0])): 
        plt.plot(abs(data['dis'].loc[data['dis']['control/V/mA']!= 0].loc[dis_index[0][i]:dis_index[1][i]]['(Q-Qo)/mA.h'])/(0.001*m_am),data['dis'].loc[data['dis']['control/V/mA']!= 0].loc[dis_index[0][i]:dis_index[1][i]][electrode], color = colors[i])#, label = str(cycle)+". cycle")

        try:
            plt.plot(abs(data['cha'].loc[data['cha']['control/V/mA']!= 0].loc[cha_index[0][i]:cha_index[1][i]]['(Q-Qo)/mA.h'])/(0.001*m_am),data['cha'].loc[data['cha']['control/V/mA']!= 0].loc[cha_index[0][i]:cha_index[1][i]][electrode], color = colors[i])
        except IndexError:
            break
        cycle += 1
    layout(ax, x_label = r'Discharge Capacity ($\mathregular{mAh\,g^{-1}}$)', y_label = r'$\mathregular{E\ (V\ vs\ Li^+/Li)}$', title = electrode)
    if save != '':
        plt.savefig(save+'.svg', transparent = True)


def plot_galv(evaluate, cy = [1,2,5,10,20,30,50], save = "", x_lim = "", y_lim = "", title = '', calc=False):
    """
    Plot the galvanostatic data evaluated with eva_GCPL function.
    Insert evaluated file and a list of cycle numbers to plot.
    """  
    color_lst = ["tab:blue", "tab:green","tab:red","tab:orange","tab:purple","tab:olive", "tab:pink", "tab:grey", "tab:brown","tab:cyan"]
    fig, ax = plt.subplots()
    i = 0

    C_d = 'Gravimetric Discharge Capacity (mAh/g)'
    C_c = 'Gravimetric Charge Capacity (mAh/g)'

    if calc == True:
        C_d = 'Gravimetric Discharge Capacity Calculated (mAh/g)'
        C_c = 'Gravimetric Charge Capacity Calculated (mAh/g)'

    try:
        for cycle in cy:
            #if cycle == len(evaluate[1]):
            #    lines_dis = ax.plot(evaluate[1][cycle-1][C_d], evaluate[1][cycle-1]['Discharge Potential (V)'], linewidth = 2)
            #else:
                #lines_dis = ax.plot(evaluate[1][cycle-1][C_d], evaluate[1][cycle-1]['Discharge Potential (V)'], linewidth = 2, label = str(cycle)+". cycle")
            lines_dis = ax.plot(evaluate[1][cycle-1][C_d], evaluate[1][cycle-1]['Discharge Potential (V)'], linewidth = 2, label = str(cycle)+". cycle")
            lines_dis[0].set_color(color_lst[i])
            lines_cha = ax.plot(evaluate[1][cycle-1][C_c],evaluate[1][cycle-1]['Charge Potential (V)'], linewidth = 2)
            lines_cha[0].set_color(color_lst[i])
            i += 1 
    except IndexError:
        pass


    layout(ax, x_label = r'Gravimetric Capacity ($\mathregular{mAh\,g^{-1}}$)', y_label = r'$\mathregular{E\ (V\ vs\ Li^+/Li)}$', x_lim = x_lim, y_lim = y_lim, title = title)
    
    try:
        if evaluate[0]["Areal Current (\u03BCA/cm$^2$)"].std()<=1:
            ax.text(0.12,
                    0.95,
                    str(round(evaluate[0]["Areal Current (\u03BCA/cm$^2$)"].mean(),1))+r' $\mathregular{\mu A\,cm^{-2}}$',
                    size = 12,
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform = ax.transAxes)
    except TypeError:
        pass
    
    if save != "":
         plt.savefig(str(save)+".svg", bbox_inches='tight', transparent = True)

def plot_CR(d, save = ''):
    fig,ax = plt.subplots()
    ax.scatter(d[0]['Cycle'],d[0]['Gravimetric Discharge Capacity (mAh/g)'], color='black')
    layout(ax, x_label = 'Cycle',y_label = r'Gravimetric Capacity ($\mathregular{mAh\,g^{-1}}$)')
    ax2 = ax.twinx()
    ax2.scatter(d[0]['Cycle'],d[0]['Coulombic Efficency (%)'], color = 'tab:red', marker = '*',  s= 100,facecolors='none')
    ax2.set_ylabel('Coulombic Efficiency (%)',fontsize = 16,color='white',path_effects=[pe.withStroke(linewidth=1.5, foreground="tab:red")])  #
    plt.yticks(fontsize=12,color='white',path_effects=[pe.withStroke(linewidth=1.5, foreground="tab:red")])
    ax2.tick_params(direction='in', length=6, width=1.5, color = 'tab:red')
    ax2.spines["right"].set_color('tab:red')
    ax2.hlines(100,0,len(d[0]['Cycle']),color='tab:red', linestyles='--')
    layout(ax2, y_label = 'Coulombic Efficiency')
    if save != "":
        plt.savefig(str(save)+"_CR.svg", bbox_inches='tight', transparent = True)
    return()      

def plot_CP_raman(data,A_el, save = ''):
    I_area = round(data['control/mA'].mean()/A_el,3)
    fig,ax = plt.subplots()
    ax.plot(data['time/s']/3600,data['Ewe/V'], color = 'black')
    layout(ax, x_label='time (h)',y_label = 'Potential (V)',title='Constant Current: '+str(I_area)+'$\mathregular{mA\,cm^{-2}}$')
    ax2 = ax.twinx()
    ax2_color = 'tab:red'
    ax2.plot(data['time/s']/3600,data['control/mA']/A_el, color = ax2_color)
    ax2.set_ylabel('$\mathregular{Current \, mA\,cm^{-2}}$',fontsize = 16,color='white',path_effects=[pe.withStroke(linewidth=1.5, foreground=ax2_color)])  #
    plt.yticks(fontsize=12,color='white',path_effects=[pe.withStroke(linewidth=1.5, foreground=ax2_color)])
    ax2.tick_params(direction='in', length=6, width=1.5, color = ax2_color)
    ax2.spines["right"].set_color(ax2_color)
    layout(ax2, y_label = '$\mathregular{Current \, mA\,cm^{-2}}$')
    if save != "":
        plt.savefig(str(save)+"_CC.svg", bbox_inches='tight', transparent = True)
    return()                         

def quick(path, save = '', cy = [1,2,5,10,20,30,50], title = '', calc = False):
    file = info(path)
    if save != '':
        save = file['filename']
    if title == 'name':
        title = file['filename']
    
    for entry in file['data']:
        if entry.split(' ')[1] == 'GCPL':
            eva = eva_GCPL(file['data'][entry], file['active material mass'],file['electrode surface area'])
            plot_galv(eva, save = save, cy = cy, title = title, calc=calc)

    return(eva[0],eva[1],file)



# OSCAR Evaluation 
def OSCAR_GCPL(df,m_am,A_el):
    cycles_tot = df["Cycle"].iloc[-1] #get the total number of half cycles
    cap_dis = []
    cap_dis_calc = []
    cap_cha = []
    cap_cha_calc = []
    cap_dis_areal = []
    cap_cha_areal = []
    ce = []
    energy_dis = []
    energy_dis_calc = []
    energy_cha = []
    energy_cha_calc = []
    galv = []
    I_areal = []
    I_specific = []
    dis_counter = 0
    cha_counter = 0
    cy_no = 1
    for cycle in range(1,int(cycles_tot)+1):
        data_cycle = df.loc[(df["Cycle"]==cycle) & (df['Cur(mA)']!= 0)] #only display corresponding to a cycle and with current different to zero
        I = data_cycle['Cur(mA)'].mean()
        dis = data_cycle.loc[data_cycle['Cur(mA)']<= 0]
        cha = data_cycle.loc[data_cycle['Cur(mA)']>= 0]
        if len(cha)==0 or len(dis)==0:
            # if charge cycle is incomplete break the loop here.
            continue

        t_0_d = datetime_sec(dis['Relative Time(h:min:s.ms)'].iloc[0])
        cap_dis.append(abs(dis['CapaCity(mAh)']).iloc[-1]/(0.001*m_am))
        t_0_c = datetime_sec(cha['Relative Time(h:min:s.ms)'].iloc[0])
        cap_cha.append(abs(cha['CapaCity(mAh)']).iloc[-1]/(0.001*m_am)) 
        grav_cap_dis = abs(dis['CapaCity(mAh)'])/(0.001*m_am) 

        grav_cap_cha = abs(cha['CapaCity(mAh)'])/(0.001*m_am)

        t_dis = timestamp_format(dis['Relative Time(h:min:s.ms)'])-t_0_d
        grav_cap_dis_calc = abs(dis['Cur(mA)'])*(t_dis/3600)/(0.001*m_am)
        cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
        t_cha = timestamp_format(cha['Relative Time(h:min:s.ms)'])-t_0_c
        grav_cap_cha_calc = abs(cha['Cur(mA)'])*(t_cha/3600)/(0.001*m_am)
        cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
            #galv.append(((grav_cap_dis,dis["Ewe/V"],t_dis),(grav_cap_cha,cha["Ewe/V"], t_cha)))
        d_galv = {
            'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
            'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
            'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
            'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
            'Discharge Potential (V)' : dis["Voltage(V)"],
            'Charge Potential (V)' : cha["Voltage(V)"],
            'Discharge Time (s)' : t_dis,
            'Charge Time (s)' : t_cha,
            'Discharge Current (mA)' : dis['Cur(mA)'],
            'Charge Current (mA)' : cha['Cur(mA)']
        }
        galv.append(d_galv)
        # calculate energy by integrating the capacity in mWh/g. 
        en_dis, en_cha = simps(dis["Voltage(V)"],grav_cap_dis), simps(cha["Voltage(V)"],grav_cap_cha)
        energy_dis.append(en_dis)
        energy_cha.append(en_cha)

        en_dis_calc, en_cha_calc = simps(dis["Voltage(V)"],grav_cap_dis_calc), simps(cha["Voltage(V)"],grav_cap_cha_calc)
        energy_dis_calc.append(en_dis_calc)
        energy_cha_calc.append(en_cha_calc)

        if isclose(abs(cha['Cur(mA)'].mode().mean()),abs(dis['Cur(mA)'].mode().mean()), rel_tol=0.01) == True:
            I = ((1000*abs(cha['Cur(mA)'].mode().mean()))+(1000*abs(dis['Cur(mA)'].mode().mean())))/2
            I_areal.append(round(I/A_el,3))
            I_specific.append(round(I/m_am,3))
        else:
            I = [1000*dis['Cur(mA)'].mode().mean(),1000*cha['Cur(mA)'].mode().mean()]
            I_areal.append([round(I[0]/A_el,3), round(I[1]/A_el,3)])
            I_specific.append([round(I[0]/m_am,3), round(I[1]/m_am,3)])

        cy_no += 1

    ce = 100*np.array(cap_dis)/np.array(cap_cha) 
    ce_calc = 100*np.array(cap_dis_calc)/np.array(cap_cha_calc) 
    energy_ef = 100*np.array(energy_dis)/np.array(energy_cha)
    energy_ef_calc = 100*np.array(energy_dis_calc)/np.array(energy_cha_calc)
        
    cap_dis_areal = np.array(cap_dis)*0.001*m_am/A_el
    cap_cha_areal = np.array(cap_cha)*0.001*m_am/A_el

    d = {"Cycle" : range(1,cy_no),
            "Gravimetric Discharge Capacity (mAh/g)" : cap_dis,
            "Gravimetric Discharge Capacity Calculated (mAh/g)" : cap_dis_calc,
            "Areal Discharge Capacity (mAh/cm$^2$)" : cap_dis_areal,
            "Gravimetric Charge Capacity (mAh/g)": cap_cha,
            "Gravimetric Charge Capacity Calculated (mAh/g)" : cap_cha_calc,
            "Areal Charge Capacity (mAh/cm$^2$)" : cap_cha_areal,
            "Coulombic Efficency (%)" : ce,
            "Coulombic Efficency Calculated (%)" : ce_calc,
            "Discharge Energy (mWh/g)": energy_dis,
            "Discharge Energy Calculated (mWh/g)": energy_dis_calc,
            "Charge Energy (mWh/g)": energy_cha,
            "Charge Energy Calculated (mWh/g)": energy_cha_calc,
            "Energy Efficency (%)" : energy_ef,
            "Energy Efficenecy Calculated (%)" : energy_ef_calc,
            "Areal Current (\u03BCA/cm$^2$)" : I_areal,
            "Specific Current (mA/g)" : I_specific
            }
    eva = pd.DataFrame(d)

    return (eva,galv)


def timestamp_format(timestamp):
    '''
    Insert panda series of timestamp in h:min:s.ms format returns array of float seconds.
    '''
    time_lst = []
    for time in timestamp:
        time_lst.append(datetime_sec(time))
    return(np.array(time_lst))

def datetime_sec(string):
    '''
    Insert timestamp in the h:min:s.ms format and return the number of seconds as float
    '''
    h,m,s = string.split(':')
    seconds = float(datetime.timedelta(hours=int(h),minutes=int(m),seconds=float(s)).total_seconds())
    return(seconds)


def eva_nda(pathway,m_am=np.nan,A_el=np.nan,data_log = False):
    '''
    Insert pathway of .nda file, active material mass and electrode surface area of nda file.
    # Important: Only set data_log on if you want to read out from Google form. Requires gspread.
    '''
    df = read_nda(pathway)
    meta = {}
    meta['active material mass'] = m_am
    meta['electrode surface area'] = A_el
    meta['data'] = df
    meta['ID'] = pathway.split('\\')[-1].split('.')[0]
    
    if data_log == True:
        try:
            OSCAR_log = OSCAR_data_logging(meta['ID'])
            meta['OSCAR_log'] = OSCAR_log
            meta['OSCAR_log_extract'] = OSCAR_log_extraction(OSCAR_log)
            meta['active material mass'],m_am = meta['OSCAR_log_extract']['Cathode Active Material (mg)'],meta['OSCAR_log_extract']['Cathode Active Material (mg)']
            meta['electrode surface area'],A_el = meta['OSCAR_log_extract']['Cathode Surface Area (cm2)'],meta['OSCAR_log_extract']['Cathode Surface Area (cm2)']
        except IndexError:
            print('No corresponding ID entry in the TMF data logging sheet found: '+meta['ID'])     

    # Create framework with data to be filled in.
    cycles_tot = df["step_ID"].iloc[-1] #get the total number of half cycles
    cap_dis = []
    cap_dis_calc = []
    cap_cha = []
    cap_cha_calc = []
    cap_dis_areal = []
    cap_cha_areal = []
    ce = []
    energy_dis = []
    energy_dis_calc = []
    energy_cha = []
    energy_cha_calc = []
    galv = []
    I_areal = []
    I_specific = []
    dis_counter = 0
    cha_counter = 0
    cy_no = 1
    
    for cycle in range(int(cycles_tot)+1):
        data_cycle = df.loc[(df["step_ID"]==cycle) & (df['current_mA']!= 0)] #only display corresponding to a cycle and with current different to zer
        # I will be np.nan if resting period.
        I = data_cycle['current_mA'].mean()
        if I <= 0:
            # Cell discharge
            dis = data_cycle
            t_0_d = dis['time_in_step'].iloc[0]
            cap_dis.append(abs(dis['capacity_mAh']).iloc[-1]/(0.001*m_am))
            dis_counter += 1
        elif I >= 0:
            # Cell charge
            cha = data_cycle
            t_0_c = cha['time_in_step'].iloc[0]
            cap_cha.append(abs(cha['capacity_mAh']).iloc[-1]/(0.001*m_am)) 
            cha_counter += 1

        if cha_counter == 1 and dis_counter == 1:
            # end of dis and recharge half cycle. Add the values to galv file.
            grav_cap_dis = abs(dis['capacity_mAh'])/(0.001*m_am) 

            grav_cap_cha = abs(cha['capacity_mAh'])/(0.001*m_am)

            t_dis = dis['time_in_step']-t_0_d
            grav_cap_dis_calc = abs(dis['current_mA'])*(t_dis/3600)/(0.001*m_am)
            cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
            t_cha = cha['time_in_step']-t_0_c
            grav_cap_cha_calc = abs(cha['current_mA'])*(t_cha/3600)/(0.001*m_am)
            cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
            #galv.append(((grav_cap_dis,dis["Ewe/V"],t_dis),(grav_cap_cha,cha["Ewe/V"], t_cha)))
            d_galv = {
                'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
                'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
                'Discharge Potential (V)' : dis["voltage_V"],
                'Charge Potential (V)' : cha["voltage_V"],
                'Discharge Time (s)' : t_dis,
                'Charge Time (s)' : t_cha,
                'Discharge Current (mA)' : dis['current_mA'],
                'Charge Current (mA)' : cha['current_mA']
            }
            galv.append(d_galv)
            # calculate energy by integrating the capacity in mWh/g. 
            en_dis, en_cha = simps(dis["voltage_V"],grav_cap_dis), simps(cha["voltage_V"],grav_cap_cha)
            energy_dis.append(en_dis)
            energy_cha.append(en_cha)

            en_dis_calc, en_cha_calc = simps(dis["voltage_V"],grav_cap_dis_calc), simps(cha["voltage_V"],grav_cap_cha_calc)
            energy_dis_calc.append(en_dis_calc)
            energy_cha_calc.append(en_cha_calc)




            #Areal Current: Determine if charge and discharge current is the same and include in the panda file
            if isclose(abs(cha['current_mA'].mode().mean()),abs(dis['current_mA'].mode().mean()), rel_tol=0.01) == True:
                I = ((1000*abs(cha['current_mA'].mode().mean()))+(1000*abs(dis['current_mA'].mode().mean())))/2
                I_areal.append(round(I/A_el,3))
                I_specific.append(round(I/m_am,3))
            else:
                I = [1000*dis['current_mA'].mode().mean(),1000*cha['current_mA'].mode().mean()]
                I_areal.append([round(I[0]/A_el,3), round(I[1]/A_el,3)])
                I_specific.append([round(I[0]/m_am,3), round(I[1]/m_am,3)])
                
            cy_no += 1
            dis_counter = 0
            cha_counter = 0
            
        if cycle == int(cycles_tot):
            if len(cap_dis) == len(cap_cha):
                continue
            if cha_counter ==1:
                # define the missing parameter which would have been determined if cha_counter was 1.
                cap_dis.append(abs(dis['capacity_mAh']).iloc[-1]/(0.001*m_am))
                t_0_d = dis['time_in_step'].iloc[0]
                
                # proceed as normal...


                grav_cap_dis = abs(dis['capacity_mAh'])/(0.001*m_am)
                grav_cap_cha = abs(cha['capacity_mAh'])/(0.001*m_am)
                
                t_dis = dis['time_in_step']-t_0_d
                grav_cap_dis_calc = abs(dis['current_mA'])*(t_dis/3600)/(0.001*m_am)
                cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
                t_cha = cha['time_in_step']-t_0_c
                grav_cap_cha_calc = abs(cha['current_mA'])*(t_cha/3600)/(0.001*m_am)
                cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
                d_galv = {
                    'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                    'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
                    'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                    'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
                    'Discharge Potential (V)' : dis["voltage_V"],
                    'Charge Potential (V)' : cha["voltage_V"],
                    'Discharge Time (s)' : t_dis,
                    'Charge Time (s)' : t_cha,
                    'Discharge Current (mA)' : dis['current_mA'],
                    'Charge Current (mA)' : cha['current_mA']
                }
                galv.append(d_galv)

                # calculate energy by integrating the capacity in mWh/g. 
                en_dis, en_cha = simps(dis["voltage_V"],grav_cap_dis), simps(cha["voltage_V"],grav_cap_cha)
                energy_dis.append(en_dis)
                energy_cha.append(en_cha)

                en_dis_calc, en_cha_calc = simps(dis["voltage_V"],grav_cap_dis_calc), simps(cha["voltage_V"],grav_cap_cha_calc)
                energy_dis_calc.append(en_dis_calc)
                energy_cha_calc.append(en_cha_calc)

                I_areal.append(round(I/A_el,3))
                I_specific.append(round(I/m_am,3))

                cy_no += 1
                print("Incomplete last cycle - discharge added.")
            elif dis_counter ==1:
                # define the missing parameter which would have been determined if cha_counter was 1.
                cap_cha.append(abs(cha['capacity_mAh']).iloc[-1]/(0.001*m_am))
                t_0_c = cha['time_in_step'].iloc[0]
                
                # proceed as normal...

                grav_cap_dis = abs(dis['capacity_mAh'])/(0.001*m_am)
                grav_cap_cha = abs(cha['capacity_mAh'])/(0.001*m_am)
                
                t_dis = dis['time_in_step']-t_0_d
                grav_cap_dis_calc = abs(dis['current_mA'])*(t_dis/3600)/(0.001*m_am)
                cap_dis_calc.append(grav_cap_dis_calc.iloc[-1])
                t_cha = cha['time_in_step']-t_0_c
                grav_cap_cha_calc = abs(cha['current_mA'])*(t_cha/3600)/(0.001*m_am)
                cap_cha_calc.append(grav_cap_cha_calc.iloc[-1])
            
                d_galv = {
                    'Gravimetric Discharge Capacity (mAh/g)' : grav_cap_dis,
                    'Gravimetric Discharge Capacity Calculated (mAh/g)' : grav_cap_dis_calc,
                    'Gravimetric Charge Capacity (mAh/g)' : grav_cap_cha,
                    'Gravimetric Charge Capacity Calculated (mAh/g)' : grav_cap_cha_calc,
                    'Discharge Potential (V)' : dis["voltage_V"],
                    'Charge Potential (V)' : cha["voltage_V"],
                    'Discharge Time (s)' : t_dis,
                    'Charge Time (s)' : t_cha,
                    'Discharge Current (mA)' : dis['current_mA'],
                    'Charge Current (mA)' : cha['current_mA']
                }
                galv.append(d_galv)

                # calculate energy by integrating the capacity in mWh/g. 
                en_dis, en_cha = simps(dis["voltage_V"],grav_cap_dis), simps(cha["voltage_V"],grav_cap_cha)
                energy_dis.append(en_dis)
                energy_cha.append(en_cha)

                en_dis_calc, en_cha_calc = simps(dis["voltage_V"],grav_cap_dis_calc), simps(cha["voltage_V"],grav_cap_cha_calc)
                energy_dis_calc.append(en_dis_calc)
                energy_cha_calc.append(en_cha_calc)

                I_areal.append(round(I/A_el,3))
                I_specific.append(round(I/m_am,3))

                cy_no += 1
                print("Incomplete last cycle - charge added.")

    ce = 100*np.array(cap_dis)/np.array(cap_cha) 
    ce_calc = 100*np.array(cap_dis_calc)/np.array(cap_cha_calc) 
    energy_ef = 100*np.array(energy_dis)/np.array(energy_cha)
    energy_ef_calc = 100*np.array(energy_dis_calc)/np.array(energy_cha_calc)
        
    cap_dis_areal = np.array(cap_dis)*0.001*m_am/A_el
    cap_cha_areal = np.array(cap_cha)*0.001*m_am/A_el

    d = {"Cycle" : range(1,cy_no),
            "Gravimetric Discharge Capacity (mAh/g)" : cap_dis,
            "Gravimetric Discharge Capacity Calculated (mAh/g)" : cap_dis_calc,
            "Areal Discharge Capacity (mAh/cm$^2$)" : cap_dis_areal,
            "Gravimetric Charge Capacity (mAh/g)": cap_cha,
            "Gravimetric Charge Capacity Calculated (mAh/g)" : cap_cha_calc,
            "Areal Charge Capacity (mAh/cm$^2$)" : cap_cha_areal,
            "Coulombic Efficency (%)" : ce,
            "Coulombic Efficency Calculated (%)" : ce_calc,
            "Discharge Energy (mWh/g)": energy_dis,
            "Discharge Energy Calculated (mWh/g)": energy_dis_calc,
            "Charge Energy (mWh/g)": energy_cha,
            "Charge Energy Calculated (mWh/g)": energy_cha_calc,
            "Energy Efficency (%)" : energy_ef,
            "Energy Efficenecy Calculated (%)" : energy_ef_calc,
            "Areal Current (\u03BCA/cm$^2$)" : I_areal,
            "Specific Current (mA/g)" : I_specific
            }
    eva = pd.DataFrame(d)

    meta['eva'] = (eva,galv)

    return(meta)

def eva_folder_nda(folder_p,m_am = 1.5, A_el = 0.785):
    cells = data_set(folder_p)
    d = {}
    for entry in cells:
        pathway = folder_p+"\\"+entry[0]
        name = entry[0].split('.')[0]
        
        eva = eva_nda(pathway,m_am = m_am, A_el = A_el)
        d[name] = eva
        print(name+' evaluated.')
    return(d)

def eva_folder_nda(folder_p,m_am = 1.5, A_el = 0.785):
    cells = data_set(folder_p)
    d = {}
    for entry in cells:
        pathway = folder_p+"\\"+entry[0]
        name = entry[0].split('.')[0]
        
        eva = eva_nda(pathway,m_am = m_am, A_el = A_el)
        d[name] = eva
        print(name+' evaluated.')
    return(d)


def OSCAR_log_extraction(log):
    '''
    Insert panda data frame of the OSCAR data logging function and extract important meta data into a dictionary.
    '''
    d = {}
    
    
    d['Setup'] = log['Cell Setup']
    cathode_string = log['Cathode']
    
    if log['Cell Setup'][0] =='C':
        # Coin Cell
        cathode_A = np.pi*(float(cathode_string.split('*')[0][1:3])/2)**2*0.01
        cathode_A_dens = float(cathode_string.split('*')[1].split('m')[0])
    elif log['Cell Setup'][0] =='A':
        # Pouchiiii
        cathode_A = float(cathode_string.split('*')[0].split('c')[0])*float(cathode_string.split('*')[0].split('c')[0])
        cathode_A_dens = float(cathode_string.split('*')[2].split('m')[0])
    else:
        print('Unknown cell setup. '+str(log['Cell Setup']))
    
    
    
    cathode_thickness = float(log['Thickness after rolling (um)'])
    cathode_m = cathode_A*cathode_A_dens
    
    d['Cathode Surface Area (cm2)'] = cathode_A
    d['Cathode Areal Density (mg/cm2'] = cathode_A_dens
    d['Cathode Thickness (um)'] = cathode_thickness
    d['Cathode Active Material (mg)'] = cathode_m
    return(d)

def OSCAR_data_logging(ID):
    '''
    Insert ID of cell and returns all the meta data stored in the TMF Data Logging spread sheet.
    '''
    sa = gspread.service_account()
    sh = sa.open('TMF Data Logging')
    wks = sh.worksheet('Master Sheet')
    sheet_list = wks.get_all_values()
    meta_data_logging = pd.DataFrame(sheet_list[2:],columns=sheet_list[1])
    index = meta_data_logging.index[meta_data_logging['ID']==ID][0]
    meta_data_ID = meta_data_logging.iloc[index]
    #meta_data_ID = meta_data_logging.loc[meta_data_logging['ID']==ID]
    return(meta_data_ID)

from numpy import diff
def x_der(array):
    '''
    insert array of numbers. Returns an array with all the values in between each value.
    ''' 
    x_mean = []
    i_0 = array.iloc[0]
    for i in array[1:]:
        mean = (i+i_0)/2
        x_mean.append(mean)
        i_0 = i
    return(x_mean)

def derive(x,y):
    dydx = diff(y)/diff(x)
    x_mean = x_der(x)
    return(x_mean,dydx)

def color_gradient(increments, style = cm.viridis):
    cm_subsection = np.linspace(0, 1, increments)
    colors = [ style(x) for x in cm_subsection ]
    return (colors)     

import matplotlib as mpl
def DQDV(file_eva,cy,gradient_len='', save = '', x_lim = "", y_lim = ""):
    '''
    PLot DQDV plot. Insert the ['eva']['X GCPL][1] file.
    '''
    if gradient_len!='':
        gradient_len = gradient_len
    else:
       gradient_len = cy[-1]+1
    color = color_gradient(gradient_len)
    fig, ax = plt.subplots()

    for i,cy_i in enumerate(cy):
        derivation = derive(file_eva[cy_i]['Discharge Potential (V)'],file_eva[cy_i]['Gravimetric Discharge Capacity (mAh/g)'])
        ax.plot(derivation[0][1:-1],derivation[1][1:-1], color = color[cy_i],linewidth=3)
        derivation = derive(file_eva[cy_i]['Charge Potential (V)'],file_eva[cy_i]['Gravimetric Charge Capacity (mAh/g)'])
        ax.plot(derivation[0][1:-1],derivation[1][1:-1], color = color[cy_i],linewidth=3)

    cmap = cm.viridis # this is the colormap used to display the spectrogram
    norm = mpl.colors.Normalize(vmin=0, vmax=gradient_len) # these are the min and max values from the spectrogram

    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label = 'Cycle Number' )

    layout(ax, y_label = r'DQ vs DV ($\mathregular{mAh\,g^{-1}\,V^{-1}}$)', x_label = r'$\mathregular{E\ (V\ vs\ Li^+/Li)}$', size = [8,4], x_lim = x_lim, y_lim = y_lim)
    if save != '':
        plt.savefig(str(save)+'.svg', transparent = True, bbox_inches = 'tight')
