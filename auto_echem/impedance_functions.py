import time
import cmath
import math
# from altair.vegalite.v4.api import value
import numpy as np
from numpy.lib.nanfunctions import _nanquantile_dispatcher
from contextlib import contextmanager
import threading
import _thread

from auto_echem.general_functions import info
from auto_echem.general_functions import tech
from auto_echem.general_functions import time_limit
from auto_echem.general_functions import color_gradient
from auto_echem.general_functions import layout
from auto_echem.general_functions import find_nearest
from auto_echem.general_functions import change_index
from auto_echem.GCPL_functions import cy_index


from impedance.preprocessing import ignoreBelowX
from impedance.models.circuits  import *

from scipy.signal import argrelextrema


def freq_index(freq, f):
    '''
    Insert frequency data and returns index of frequency value f
    '''
    index = abs(freq-f).sort_values(ascending=True).index[0]
    return(index)

def parameter(eva):
    '''
    extract the parameter from the evaluated PEIS data (via Nyquist function) and store them into a dictionary. Assumes a consistent circuit model.  
    '''

    d = {}
    for i,entry in enumerate(eva[2][0][0]):
        parameter = entry
        value_list = []
        for cy in range(len(eva[2])):
            try:
                value_list.append(eva[3][cy][i])
            except IndexError:
                value_list.append(np.nan)
        d[parameter] = value_list
    
    return (d)

def quali_fit(parameter,confidence):
    quality = 0
    for p,c in zip(parameter,confidence):
        quali = c/p
        quality = quality + quali
    return(quality)

def eva_PEIS(df):
    data_PEIS = []
    try:
        cycles_tot = df["cycle number"].iloc[-1] #get the total number of half cycles
    except KeyError:
        cycles_tot = df["z cycle"].iloc[-1]
    except IndexError:
        cycles_tot = 0
    if cycles_tot >= 1:
        # EC Lab counts the cycle number properly.
        for item in range(int(cycles_tot)):
            # cycle number starts with 1
            cycle = item+1
            try:
                data_cycle = df.loc[(df['cycle number'] == cycle)] #only display corresponding to a cycle
            except KeyError:
                data_cycle = df.loc[(df['z cycle'] == cycle)]   
            freq = data_cycle['freq/Hz']
            Re_z = data_cycle['Re(Z)/Ohm']
            Im_z = data_cycle['-Im(Z)/Ohm']
            I = data_cycle['<I>/mA']
            E = data_cycle['<Ewe>/V']
            # check if there is a reference electrode and thus a counter electrode EIS data...
            try:
                Re_z_CE = data_cycle['Re(Zce)/Ohm']
                Im_z_CE = data_cycle['-Im(Zce)/Ohm']
                # WECE is just the sum of both....
                Re_z_WECE = data_cycle['Re(Zwe-ce)/Ohm']
                Im_z_WECE = data_cycle['-Im(Zwe-ce)/Ohm']
                PEIS_CE = [Re_z_CE,Im_z_CE,Re_z_WECE,Im_z_WECE]
            except KeyError:
                PEIS_CE = []

            data_PEIS.append([freq,Re_z,Im_z,E,I,PEIS_CE])
    else:
        # determine the cycle number by the maxima of frequencies. Required for BT lab measurement. 
        idx = argrelextrema(df['freq/Hz'].values, np.greater)
        max_idx = np.append(0,idx[0])
        cycles_tot = len(max_idx)
        for item in range(int(cycles_tot)):
            # cycle number starts with 1
            cycle = item+1
            try:
                data_cycle = df.loc[max_idx[item]:max_idx[item+1]-1] #only display corresponding to a cycle
            except IndexError:
                data_cycle = df.loc[max_idx[item]:]

            freq = data_cycle['freq/Hz']
            Re_z = data_cycle['Re(Z)/Ohm']
            Im_z = data_cycle['-Im(Z)/Ohm']
            I = data_cycle['I/mA']
            E = data_cycle['Ewe/V']
            data_PEIS.append([freq,Re_z,Im_z,E,I])

    return(data_PEIS)

def Nyquist(raw,circ="", fit_para = 0,lf_limit = 0,hf_limit = math.inf):
    '''
    Insert the raw PEIS data and a suitable circuit. Returns a list of the experimental and fitted Nyquist plot and a list of all corresponding charge transfer resistances.
    '''
    evaluated = eva_PEIS(raw)
    counter = 1
    R_ct = []
    Nyn_exp = []
    Nyn_fit = []
    circuit_elements = []
    circuit_values = []
    for entry in evaluated:
        qc = 0
        fit_counter = fit_para
        entry[0] = entry[0].loc[entry[0]!=0]
        with time_limit(15, 'sleep'):
            try:
                try:
                    f = np.array(entry[0])
                    #f_pred = np.geomspace(f[0],f[-1])
                    freq = np.array(entry[0])
                    Re = np.array(entry[1])
                    Im = np.array(entry[2])

                    lf_index = np.argwhere(np.array(freq)>lf_limit)[-1][0]
                    hf_index = np.argwhere(np.array(freq)<hf_limit)[0][0]
                    #f = np.array(freq)

                    freq = freq[hf_index:lf_index]
                    Re = Re[hf_index:lf_index]
                    Im = Im[hf_index:lf_index]

                    while qc == 0:
                        #fitted = fit(entry[0],entry[1],entry[2], circ=circ, fit_counter = fit_counter)
                        try: 
                            fitted = fit(freq,Re,Im, circ=circ, fit_counter = fit_counter)
                        except ValueError:
                            print('Value Error!')
                            break
                        fit_counter += 1
                        f_pred = np.geomspace(freq[0],freq[-1])
                        # Obtain omega max values from the local maxima of -Im(Z)
                        omegas = omega_max(fitted[0].get_param_names(),fitted[0].parameters_.tolist(),-fitted[1].imag,f_pred)
                        qc = omegas[2]

                        R1 = omegas[1][para_index('R1',omegas[0][0])]

                        if circ == 'Rpp' or 'cc':
                            # Quality Control for R1 R2 missmatch, check first if R1 and R2 have been assigned.
                            
                            R2_idx = para_index('R2',omegas[0][0])
                            if np.isnan(R2_idx) == False:

                                if abs(R1-qc_trend('R1',omegas[0][0],circuit_values)) >= abs(R1-qc_trend('R2',omegas[0][0],circuit_values)):
                                    print('Potential R1 and R2 missmatch')
                                    qc = 0
                                else:
                                    pass
                                if qc == 0:
                                    print('Try alternative fitting guesses: '+ str(fit_counter))

                                if fit_counter == 3:
                                    break
                            else:
                                pass
                        if fit_counter==3:
                            break
                    if abs(R1-qc_trend('R1',omegas[0][0],circuit_values)) >= 50000:
                        # if the determine R1 value is so much different than expected the respecitive cycle is fitted with np.nan values
                        nan_lst = []
                        for i in range(len(omegas[1])):
                            nan_lst.append(np.nan)
                        elements = [nan_lst, nan_lst]
                        values = nan_lst
                        print('np.nan added since fit is too bad.')
                    else:
                        elements = omegas[0]
                        values = omegas[1]

                    circuit_elements.append(elements)
                    circuit_values.append(values)
                    print("cycle "+str(counter)+" fitted.")
                    Nyn_exp.append([entry[0],entry[1],entry[2],entry[3],entry[4]])
                    
                    f_pred = f_pred.tolist()
                    fit_real = fitted[1].real.tolist()
                    fit_imag = -fitted[1].imag
                    
                    Nyn_fit.append([f_pred,fit_real,fit_imag.tolist()])
                except IndexError:
                    print('Frequency data is empty. Analysis stopped.')
                    #break
            except KeyboardInterrupt:
                print("cycle "+str(counter)+" fitting timed out.")
                try:
                    para_no = len(elements[0])
                except UnboundLocalError:
                    if circ == 'cc':
                        para_no = 13
                    else:
                        para_no = 5
                lst = []
                for i in range(para_no):
                    lst.append(np.nan)
                circuit_elements.append((lst,lst))
                circuit_values.append(lst)
                R_ct.append(np.nan)
                Nyn_exp.append([entry[0],entry[1],entry[2]])
                Nyn_fit.append([np.nan, np.nan, np.nan])
            counter += 1

    return(Nyn_exp,Nyn_fit,circuit_elements,circuit_values)

def para_index(parameter,circuit_element):
    '''
    Determine the index of a parameter in the element list.
    '''
    for el_idx,entry in enumerate(circuit_element):
        # find the index of the parameter of interest in the circuit element list
        if entry == parameter:
            par_idx = el_idx
    try:
        return(par_idx)
    except UnboundLocalError:
        #print(parameter+' was not found.')
        return(np.nan)

def qc_trend(parameter,circuit_element,circuit_values):
    '''
    Calculate the mean value of a paramenter from previous measurements.
    '''
    value_list = []
    par_idx = para_index(parameter,circuit_element)
    for entry in circuit_values:
        value_list.append(entry[par_idx])
    value_mean = np.nanmean(value_list)

    return(value_mean)





def omega_max(elements,values,Im,f_pred):
    qc = 1

    counter = 0
    idx_CPE = []
    idx_W = []

    for entry in elements[0]:
        des = entry.split('_')
        if 'CPE' in des[0] and '1' in des[1]:
            idx_CPE.append(counter+1)
            #Double coutner increase as later on an additional element will have been added to the elements and values list. So we compensate for that already here. 
            counter += 2
        if 'W' in des[0] and '1' in des[1]:
            idx_W.append(counter+1)

        counter += 1

    idx = argrelextrema(Im, np.greater)[0]
    # sometimes the maximum is not shown since the fit is only partially visible within the given frequency range. Therefore add a 0 to the end to see if there is technically another maximium.
    Im_lst = Im.tolist()
    Im_lst.append(0)
    qc_idx = argrelextrema(np.array(Im_lst), np.greater)[0].tolist()
    omega_local = []
    for index in idx:
        # obtain all the corresponding omega values. Frequencies are in Hz.
        omega = f_pred[index]
        omega_local.append(omega)
    
        #dirty fix: the fit is bad and thus there are not enough maxima... enter np.nan in the file in order to continue with the evaluation. 
    c = 0
    #while len(omega_local) != len(idx_CPE):
    while len(qc_idx) != (len(idx_CPE)+len(idx_W)):
        # compares the number of maximum expected from the number of CPEs (idx_CPE) with the number of found maximum. If they deviate the qc remains 0 and the fit is redone in the fit function.
        omega_local.append(np.nan)
        qc_idx.append(np.nan)
        c+=1
        #print('Bad fit by number of maxima. Dirty fix was applied. ' + str(c)+ ' times.')
        print('Number of maxima in Im deviates from number of parallel R-CPE ' + str(c)+ ' times.')
        #qc = 0
        qc = 1
        if c == 10:
            print('Presumambly more maxima than specified found....')
            break

    CPE_count = 1
    for omega,idx in zip(omega_local,idx_CPE):

        alpha = values[idx-1]
        Q_0 = values[idx-2]
        R_ct = values[idx-3]
        #C_true = Q_0*((omega*2*np.pi)**(alpha-1))
        C_true = 1/(omega*2*np.pi*R_ct) #other formular 

        elements[0].insert(idx,'CPE'+str(CPE_count)+'_2')
        elements[0].insert(idx+1,'CPE'+str(CPE_count)+'_3')
        elements[1].insert(idx,'Hz')
        elements[1].insert(idx+1,'F')

        values.insert(idx,omega)
        values.insert(idx+1,C_true)
        CPE_count += 1
        # insert np nan instead of super high value R2.
        R2_idx = para_index('R2',elements[0])
        if np.isnan(R2_idx) == False:
            if values[para_index('R2',elements[0])] >= 10000000:
                values[para_index('R2',elements[0])] = np.nan
                print('R2 was set to np.nan because it exceeds 10000000')
        else:
            continue
    return(elements,values,qc)


def fit(freq,Re,Im,circ = "",fit_counter = 0, ignore_posIm=True):
    '''
    insert frequency, real, and negative Im of EIS data. Specify the circuit. If none, Randles circuit is used.
    returns the results of the fit and predicted data within the given frequency range. 
    '''
    f = np.array(freq)
    Z = np.complex128(np.array(Re)-1j*(np.array(Im)))

    if ignore_posIm==True:
        f, Z = ignoreBelowX(f, Z) #only EIS data with negative Im and positive Re
        # Ignore all values with a frequency beyon 200 kHz. High frequency artefacts...
        mask = f < 200*1e3
        f = f[mask]
        Z = Z[mask]

    if circ == "Randles":
        circuit = Randles(initial_guess=[10, 200, .000001, .9, .001, 200], CPE=True)
        circuit.fit(f,Z)
        R_ct = circuit.parameters_[1]
        C_CPE = circuit.parameters_[2]
        if not 1e-6<C_CPE<1e-4:
            print('Unexpected CPE Capacitance. Check the fit.')
            
    if circ == "RpW":
        circuit = CustomCircuit(initial_guess=[5, 500, .000001, .85, .001, 200], circuit='R0-p(R1,CPE1)-Wo_1')
        circuit.fit(f,Z)

        circuit.fit(f,Z)
        R_ct = circuit.parameters_[1]
        C_CPE = circuit.parameters_[2]
        
    if circ == "RpC":
        circuit = CustomCircuit(initial_guess=[5, 500, .000001, .85, .001, .85,], circuit='R0-p(R1,CPE1)-CPE2')
        circuit.fit(f,Z)

        circuit.fit(f,Z)
        R_ct = circuit.parameters_[1]
        C_CPE = circuit.parameters_[2]

    elif circ == "cc":
        if fit_counter == 0:
            circuit = CustomCircuit(initial_guess=[10, 20, .00001, .85, 10, .1, .85, .001, 200],
                              circuit='R0-p(R1,CPE1)-p(R2,CPE2)-Wo_1')
        elif fit_counter == 1:
            circuit = CustomCircuit(initial_guess=[2, 2, .00001, .85, 5, .1, .85, .001, 200],
                              circuit='R0-p(R1,CPE1)-p(R2,CPE2)-Wo_1')

        elif fit_counter == 2:
            circuit = CustomCircuit(initial_guess=[2, 0.5, .00001, .85, 50, .1, .85, .001, 200],
                              circuit='R0-p(R1,CPE1)-p(R2,CPE2)-Wo_1')
        circuit.fit(f,Z)
    
    elif circ=='Rpp':
        if fit_counter == 0:
            circuit = CustomCircuit(initial_guess=[5, 20, .000001, .85, 40, .1, .2], circuit='R0-p(R1,CPE1)-p(R2,CPE2)')
        elif fit_counter == 1:
            circuit = CustomCircuit(initial_guess=[5, 200, .000001, .85, 40, .1, .2], circuit='R0-p(R1,CPE1)-p(R2,CPE2)')
        elif fit_counter == 2:
            circuit = CustomCircuit(initial_guess=[5, 200, .000001, .85, 1000, .1, .2], circuit='R0-p(R1,CPE1)-p(R2,CPE2)')
        circuit.fit(f,Z)

    elif circ == 'Rp':
        circuit = CustomCircuit(initial_guess=[5, 500, .000001, .85], circuit='R0-p(R1,CPE1)')
        circuit.fit(f,Z)

    elif circ=='Rppp':
        #if fit_counter == 0:
        circuit = CustomCircuit(initial_guess=[5, 20, .000001, .85, 40, .000001, .85, 40, .000001, .85], circuit='R_0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)')
        circuit.fit(f,Z)
        
    elif circ=='p(RCC-W)':
        #if fit_counter == 0:
        
        initial_guess = [.01, 20, .000001, .000001, .05, 100]
        circuit = CustomCircuit(initial_guess = [.01, 20, .000001, .001, .05, 100], circuit = 'R0-p(R1,C1,C2-Wo1)')
        circuit.fit(f,Z)
        
    elif circ=='pp':
        if fit_counter == 0:
            circuit = CustomCircuit(initial_guess=[800, .00000001, .85, 100, .001, .85], circuit='p(R1,CPE1)-p(R2,CPE2)')
        elif fit_counter == 1:
            circuit = CustomCircuit(initial_guess=[800, .000001, .85, 100, .1, .85], circuit='p(R1,CPE1)-p(R2,CPE2)')
        elif fit_counter == 2:
            circuit = CustomCircuit(initial_guess=[[300, .00000001, .85, 10, .001, .85]], circuit='p(R1,CPE1)-p(R2,CPE2)')
        circuit.fit(f,Z)
    
    else:
        print(str(circ)+' is not defined.')
   
    f_pred = np.geomspace(f[0],f[-1])
    fitted = circuit.predict(f_pred)

    return(circuit,fitted)


def PEIS_analysis(pathway, circ = ['cc', 'cc'], plot = '', resttime = 50, save = '',fit_para = 0):
    meta = info(pathway)
    d_eva = {}
    circ_count = 0
    for entry in meta['data']:
        start = time.time()
        if entry.split(' ')[1] == 'PEIS':
            d = {}
            try:
                print('Fitting with '+circ[circ_count])
            except IndexError:
                print('Please specify the circuit fit for '+str(entry))
                continue
            data_eva = Nyquist(meta['data'][entry], circ = circ[circ_count], fit_para = fit_para)
            d['Nyquist data'] = data_eva
                        
            try:
                data_para = parameter(data_eva)
                d['Nyquist parameter'] = data_para
            except IndexError:
                print('IndexError in parameter function.')
                pass

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
                    plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area']/2, tit = meta['filename'])
                    try:
                        plot_R(meta['waiting time'],data_para, A_el = meta['electrode surface area'], tit = meta['filename'])
                    except KeyError:
                        print('Key Error - could not plot impedance parameter.')

                


            if np.isnan(meta['waiting time']) == False:
                t_rest = meta['waiting time']*np.array(range(len(data_para['R1'])))
                R_ct_rest = find_nearest(t_rest,resttime)
                R_ct = round(data_para['R1'][R_ct_rest[0]]*meta['electrode surface area'],2)
                d['R_'+str(R_ct_rest[1])] = R_ct

                #meta['eva'][entry+'_R_'+str(R_ct_rest[1])] = R_ct
                print(entry+'_R_'+str(R_ct_rest[1])+' : '+str(R_ct))

            d_eva[entry] = d
            end = time.time()
            print(entry + ' evaluated in '+str(round(end - start,2))+' seconds.')
            circ_count += 1
            if circ_count > len(circ):
                print('Please specify the circut model for the '+str(circ_count+1)+' PEIS data set.')
                break

        elif entry.split(' ')[1] == 'GCPL':
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

    meta['eva'] = d_eva

    return(meta)


def stabilize_mean(para_value,cor_fac=3):
    '''
    Calculates the mean value of different Nyquist parameter in a stabilizing PEIS measurement. Ignores the first measurement point. Ignores significantly different values.
    
    para_value: xx['eva']['X PEIS]['Nyquist parameter']
    cor_fac: kicks out all values which exceed cor_fac times the std of the parameter array. Standard: 3
    
    returns a dictionary for all PEIS parameter and a corresponding tuple with (mean, std, evo). Evo is the drift throughout the measurement (last minus first)
    '''
    
    d = {}
    for para in para_value:
        para_data = np.array(para_value[para][1:])
        len_ini = len(para_data)
        correction = abs(para_data-np.nanmean(para_data)) <= np.nanstd(para_data)*cor_fac
        correction_in = []
        for counter,value in enumerate(correction):
            if value == False:
                correction_in.append(counter)

        para_data = para_data[correction]
        if len(para_data)!= len_ini:
            print(para+': '+str(len_ini-len(para_data))+' measurement points were deleted at index '+str(correction_in))
        
        mean = np.nanmean(para_data)
        std = np.nanstd(para_data)
        try:
            dif = para_data[-1]-para_data[0]
        except IndexError:
            dif = np.nan


        values = (mean,std,dif)
        d[para] = values

        print(para+': '+str(mean)+' std: '+str(std)+' evolution: '+str(dif))
    return (d)

# Plotting Functions

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
        plt.savefig(save+'.svg', transparent = True)

def plot_R(t, data_para, A_el, tit = '', save = ''):
    '''
    Plot the restistances of a give PEIS data file. Requires the evaluated data from the parameter function.
    '''
    fig, ax = plt.subplots()
    x = t*np.array(range(len(data_para['R0'])))
    if np.isnan(x[0]) == True:
        plt.scatter(range(len(data_para['R0'])), np.array(data_para['R0'])*A_el, label = 'R0')
        plt.scatter(range(len(data_para['R0'])), np.array(data_para['R1'])*A_el, label = 'R1')
        try :
            plt.scatter(range(len(data_para['R0'])), np.array(data_para['R2'])*A_el, label = 'R2')
        except KeyError:
            pass
        layout(ax, x_label = 'cycle number', y_label = 'Resistance ($\mathregular{\u03A9cm^{2}}$)', title = tit)
    else:    
        plt.scatter(x, np.array(data_para['R0'])*A_el, label = 'R0')
        plt.scatter(x, np.array(data_para['R1'])*A_el, label = 'R1')
        try:
            plt.scatter(x, np.array(data_para['R2'])*A_el, label = 'R2')
        except KeyError:
            pass

        layout(ax, x_label = 'time (h)', y_label = 'Resistance ($\mathregular{\u03A9cm^{2}}$)', title = tit)
    if save != "":
        plt.savefig(str(save)+".svg", transparent = True)

def strip_plate(data, title = ''):
    fig, ax = plt.subplots()
    plt.plot((data['time/s']-data['time/s'].iloc[0])/3600,data['Ewe/V'])
    layout(ax, x_label = 'time (hours)', y_label = r'$\mathregular{E\ (V\ vs\ Li^+/Li)}$', title = title)

def eva_GEIS(PEIS_eva,A_el):
    '''
    Insert the evaluated PEIS file d['eva']['X GEIS] and returns a dictionary with EIS cycles sorted with applied current and EIS fit paramter. 
    '''
    I = []
    for i in range(len(PEIS_eva['Nyquist data'][0])):
        try:
            I_i = PEIS_eva['Nyquist data'][0][i][4].mean()/A_el
        except IndexError:
            I_i = np.nan
        I.append(I_i)
    
    I_index = change_index(I)
    
    I_no = 0
    data_set = []
    d = {}
    for i,index in enumerate(I_index):
        data = []
        Nyquist_data = []
        try:
            for i,cycle in enumerate(range(index,I_index[i+1])):
                try:
                    current = round(PEIS_eva['Nyquist data'][0][cycle][4].mean()/A_el,5)*1000
                    potential = round(PEIS_eva['Nyquist data'][0][cycle][3].mean(),5)
                except IndexError:
                    current = np.nan
                    potential = np.nan
                R_ct = PEIS_eva['Nyquist parameter']['R1'][cycle]
                C = np.array(PEIS_eva['Nyquist parameter']['CPE1_3'][cycle])*1e6
                Nyquist_data_i = PEIS_eva['Nyquist data'][0][cycle]
                data.append([current,R_ct,C,potential])
                Nyquist_data.append(Nyquist_data_i)
            data_set.append(data)
            d[current] = Nyquist_data
        except IndexError:
            for i,cycle in enumerate(range(index,len(I))):
                current = round(PEIS_eva['Nyquist data'][0][cycle][4].mean()/A_el,5)*1000
                potential = round(PEIS_eva['Nyquist data'][0][cycle][3].mean(),5)
                R_ct = PEIS_eva['Nyquist parameter']['R1'][cycle]
                C = np.array(PEIS_eva['Nyquist parameter']['CPE1_3'][cycle])*1e6
                Nyquist_data_i = PEIS_eva['Nyquist data'][0][cycle]
                data.append([current,R_ct,C,potential])
                Nyquist_data.append(Nyquist_data_i)
            data_set.append(data)
            d[current] = Nyquist_data
    return(d,data_set)

def plot_GEIS(data_set, A_el, save = ''):
    fig,ax = plt.subplots()
    #ax2 = ax.twinx()

    for data in data_set:
        colors = color_gradient(len(data))
        for i,data_i in enumerate(data):
            ax.scatter(data_i[0],data_i[1]*A_el, color = colors[i])
            #ax.scatter(data_i[0],data_i[2], color = colors[i], marker = '*')

    layout(ax,  y_label='$\mathregular{R_{Interphase}\,(\u03A9cm^{2}}$)', x_label='$\mathregular{Areal\,Current\,(\u03bcAcm^{-2}}$)')#, y_lim=[180,280])
    #return(fig)
    #layout(ax, y_label = 'Capacitance (nF/cm2)', x_label='Areal Current (uA/cm2)')#, y_lim = [2,3])
    if save != '':
        plt.savefig(save+'.svg', transparent = True)


def EIS_CE(data,sequence,circ='Rp',lf_limit='',hf_limit='', fit_counter = 0,ind=False):
    '''
    Insert data in form of evaluated 3-El data with the eva_threeEl(pathway) function.
    Specify the sequence where the PEIS data is stored.
    Specify the circuit used for fitting.git
    Ind = True means it neglects all measurement points where the imaginary part turns postive (and subsequent ones, i.e. higher frequencies).
    '''
    Nyn_exp = []
    Nyn_fit = []
    circuit_elements = []
    circuit_values = []

    fig, ax = plt.subplots()
    colors = color_gradient(len(data['data'][sequence]))

    for cy in range(len(data['data'][sequence])):
        freq = np.array(data['data'][sequence][cy][0])
        Re_CE = data['data'][sequence][cy][-1][0]
        Im_CE = data['data'][sequence][cy][-1][1]

        if ind == True:
            '''
            Cancel out the positive Im values. 
            '''
            try :
                index = 0
                ind_loop = np.argwhere(np.array(Im_CE)<0)[index][0]
                if ind_loop == 0:
                    # sometimes the high frequency data is also negative. Cancel those out too. 
                    while ind_loop <= 3:
                        ind_loop_i = ind_loop
                        index += 1
                        ind_loop = np.argwhere(np.array(Im_CE)<0)[index][0]
                        freq = freq[ind_loop_i+1:ind_loop]
                        Re_CE = Re_CE[ind_loop_i+1:ind_loop]
                        Im_CE = Im_CE[ind_loop_i+1:ind_loop]
                else:
                    freq = freq[0:ind_loop]
                    Re_CE = Re_CE[0:ind_loop]
                    Im_CE = Im_CE[0:ind_loop]
            except IndexError:
                continue
        
        if lf_limit != '':
            '''
            Cancel out the low frequency values. 
            '''
            try :
                lf_index = np.argwhere(np.array(freq)<lf_limit)[0][0]
                freq = freq[0:lf_index]
                Re_CE = Re_CE[0:lf_index]
                Im_CE = Im_CE[0:lf_index]
            except IndexError:
                print(IndexError)
                continue
        try: 
            fitted = fit(freq,Re_CE,Im_CE, circ = circ, fit_counter=fit_counter)
        except ValueError:
            print('Value Error spotted at cycle number '+str(cy))#
            continue
        f_pred = np.geomspace(freq[0],freq[-1])
        omegas = omega_max(fitted[0].get_param_names(),fitted[0].parameters_.tolist(),-fitted[1].imag,f_pred)

        elements = omegas[0]
        values = omegas[1]
        circuit_elements.append(elements)
        circuit_values.append(values)
        Nyn_exp.append([freq,Re_CE,Im_CE])
        f_pred = f_pred.tolist()
        fit_real = fitted[1].real.tolist()
        fit_imag = -fitted[1].imag
        Nyn_fit.append([f_pred,fit_real,fit_imag.tolist()])

        plt.scatter(Re_CE,Im_CE, color = colors[cy])
        plt.plot(fit_real,fit_imag, color = colors[cy])

    eva_PEIS = (Nyn_exp,Nyn_fit,circuit_elements,circuit_values)
    data_para = parameter(eva_PEIS)
    layout(ax, square = True,x_label = 'Re(Z) (Ohm)', y_label = '-Im(Z) (Ohm)')
    d = {
        'Nyquist data' : eva_PEIS,
        'Nyquist parameter' : data_para,
    }
    return(d)

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg

@contextmanager
def time_limit(seconds, msg=''):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
        
def EIS_WE(data,sequence,circ='Rp',lf_limit='',hf_limit='', fit_counter = 0,ignore_posIm=True):
    '''
    Insert data in form of evaluated 3-El data with the eva_threeEl(pathway) function.
    Specify the sequence where the PEIS data is stored.
    Specify the circuit used for fitting.
    Ind = True means it neglects all measurement points where the imaginary part turns postive (and subsequent ones, i.e. higher frequencies).
    lf_limit: low frequency limit in Hz
    
    '''
    Nyn_exp = []
    Nyn_fit = []
    circuit_elements = []
    circuit_values = []
    fig, ax = plt.subplots()
    colors = color_gradient(len(data['data'][sequence]))
    counter = 0
    for cy in range(len(data['data'][sequence])):
        freq = np.array(data['data'][sequence][cy][0])
        Re = data['data'][sequence][cy][1]
        Im = data['data'][sequence][cy][2]

        if lf_limit != '':
            '''
            Cancel out the low frequency values. 
            '''
            try :
                lf_index = np.argwhere(np.array(freq)<lf_limit)[0][0]
                freq = freq[0:lf_index]
                Re = Re[0:lf_index]
                Im = Im[0:lf_index]
            except IndexError:
                pass
        if hf_limit != '':
            '''
            Cancel out the low frequency values. 
            '''
            try :
                hf_index = np.argwhere(np.array(freq)>hf_limit)[-1][0]
                freq = freq[hf_index:-1]
                Re = Re[hf_index:-1]
                Im = Im[hf_index:-1]
            except IndexError:
                pass

        
        with time_limit(15, 'sleep'):
            try:
                fitted = fit(freq,Re,Im, circ = circ, fit_counter = fit_counter,ignore_posIm=ignore_posIm)
                f_pred = np.geomspace(freq[0],freq[-1])
                omegas = omega_max(fitted[0].get_param_names(),fitted[0].parameters_.tolist(),-fitted[1].imag,f_pred)

                elements = omegas[0]
                values = omegas[1]
                circuit_elements.append(elements)
                circuit_values.append(values)
                Nyn_exp.append([freq,Re,Im])
                f_pred = f_pred.tolist()
                fit_real = fitted[1].real.tolist()
                fit_imag = -fitted[1].imag
                Nyn_fit.append([f_pred,fit_real,fit_imag.tolist()])

                plt.scatter(Re,Im, color = colors[cy])
                plt.plot(fit_real,fit_imag, color = colors[cy])

            except KeyboardInterrupt:
                print("cycle "+str(counter)+" fitting timed out.")
                try:
                    para_no = len(elements[0])
                except UnboundLocalError:
                    para_no = 5
                lst = []
                for i in range(para_no):
                    lst.append(np.nan)
                circuit_elements.append((lst,lst))
                circuit_values.append(lst)
                Nyn_exp.append([entry[0],entry[1],entry[2]])
                Nyn_fit.append([np.nan, np.nan, np.nan])
            counter += 1
    eva_PEIS = (Nyn_exp,Nyn_fit,circuit_elements,circuit_values)
    data_para = parameter(eva_PEIS)
    layout(ax, square = True,x_label = 'Re(Z) (Ohm)', y_label = '-Im(Z) (Ohm)')
    d = {
        'Nyquist data' : eva_PEIS,
        'Nyquist parameter' : data_para,
    }
    return(d)

