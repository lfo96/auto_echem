import matplotlib.pyplot as plt
import numpy as np
import os,sys
import pandas as pd
import threading
import _thread
import time
import numpy as np
import datetime as dt
import cmath
import math
import matplotlib as mpl
from scipy       import integrate, special,optimize
from scipy       import integrate, special,optimize
from auto_echem.auto import auto
import matplotlib.cm     as cm
from lmfit       import Model, Parameters, minimize, fit_report
from auto_echem.general_functions import color_gradient
from auto_echem.general_functions import layout
from auto_echem.general_functions import outliers


from scipy       import integrate
import numpy             as np

import os

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Path '{path}' created successfully.")
        except OSError as e:
            print(f"Error creating path '{path}': {e}")
    else:
        print(f"Path '{path}' already exists.")

### Old Functions which need to be updated...
class Peak():
	def __init__(self, path_to_input, dst_dir):
		self.filename = path_to_input
		self.dst_dir = dst_dir
		self.wavenumber = []
		self.intensity = []
		self.normalized = []
		self.truncated = []
		self.wavenumber2 = []
		self.concentration_m = []
		return
	def run(self):
		self.parse()
		self.normalize()
		self.truncate()
		self.write_results()
		self.area()
		return
	def parse(self):
		with open(self.filename, encoding="utf8", errors="ignore") as f:
			for line in f:
				if line.startswith('#'):
					continue
				parts = line.split()
				w = float(parts[0])
				i = float(parts[1])
				self.wavenumber.append(w)
				self.intensity.append(i)
		return 
	def normalize(self):
		index = 0
		for ind, val in enumerate(self.wavenumber):
			if val > 1400:
				index = ind
		m = max(self.intensity[:index])
		for i in self.intensity:
			self.normalized.append(i/m)
		return
	def truncate(self):
		index1 = 0
		index2 = 0
		for ind1, val1 in enumerate(self.wavenumber):
			if val1 < 762 and val1 > 758:
				index1 = ind1
			if val1 < 682 and val1 > 678:
				index2 = ind1
			self.truncated = self.normalized[index1:index2]
			self.wavenumber2 = self.wavenumber[index1:index2]
		return
	def area(self):
			area_peak = integrate.simps(self.truncated, self.wavenumber2)
			a = -20.46985
			b = -0.6228
			self.concentration_m = (area_peak-b)/a
			return
	def plotting(self):
		plt.plot(self.wavenumber2, self.truncated)
		plt.fill_between(self.wavenumber2, self.truncated, color = "grey",alpha = 0.3, hatch = '|')
		plt.show()
		return 
	def write_results(self):#存好700附近的标准化数据
		with open(self.dst_dir+'/'+self.filename[self.filename.rfind('/'):-4]+"_700PEAK.txt", 'w', encoding="utf8", errors="ignore") as f:
			for i in range(len(self.truncated)):
				f.write(str(self.wavenumber2[i]))
				f.write('\t')
				f.write(str(self.truncated[i]))
				f.write('\n')
		return

def concloading_LFO(raman_p,out_p,fun,meas_index = []):
    '''
    param path1: the input path for the Raman data    
    param path3: the output path for recording ecaluated data
    param x: the number of file you want to analysis
    param fun:#p1 is the function. For detail, pls see "F1"
    ---
    return a dictionary
    '''
    
    e  = {}
    for t in meas_index:
        p4 = raman_p+str(t)+'_BL.txt'
        d  = pd.DataFrame(np.loadtxt(p4), columns = ['Z', 'Wave','Intensity'])
        z  = d['Z'].drop_duplicates().tolist()
        c=[]
        for i in range(len(z)):
            d1 = d.loc[d['Z']==z[i]]
            p5=out_p+r'\raman_linescan.txt'
            with open(p5, 'w', encoding="utf8", errors="ignore") as f:
                for j in range(len(d1['Wave'])):
                    f.write(str(d1['Wave'].iloc[j]))
                    f.write('\t')
                    f.write(str(d1['Intensity'].iloc[j]))
                    f.write('\n')
            n = fun(p5, out_p)
            try:
                n.run()
                c.append(n.concentration_m)
            except ZeroDivisionError:
                c.append(np.nan)       
        print('【Processing '+str(t)+'】')
        e[t] = (z,c)
    print('【All Evaluated】')
    return e,z




## Functions to determine the time for a raman linescan
import os
from datetime import datetime

def list_files_with_last_modified_sorted(directory_path):
    files_info = []

    try:
        # List all files in the directory
        files = os.listdir(directory_path)

        # Get file information and sort by last modified timestamp
        files_info = sorted(
            [(file_name, datetime.fromtimestamp(os.path.getmtime(os.path.join(directory_path, file_name)))) for file_name in files],
            key=lambda x: x[1]
        )

    except OSError as e:
        print(f"Error: {e}")

    return files_info

def scan_wdf(folder_path,filename):
    meas_index = []
    timestamp = []
    direc_list = list_files_with_last_modified_sorted(folder_path)
    for entry in direc_list:
        # Identfiy all the files that have the filename strucutre with increasing integers for measurement number
        filename_i = entry[0].split('_')[:-1]
        filename_i = "_".join(filename_i)
        if filename in filename_i:
            # make sure its the wdf file which is generated right when t he measurement is finished
            format = entry[0].split('.')[-1]
            if format == 'wdf':
                # now disstinguish between _BL and not _BL; _BL indicates manually exported, baseline corrected files. 
                if entry[0][-5].isdigit():
                    meas_index_i = entry[0].split('.wdf')[0].split('_')[-1]
                    meas_index.append(int(meas_index_i))
                    timestamp.append(entry[1])
    return(meas_index,timestamp)


def mean_ramanscan_time(meas_index,timestamp):
    raman_meas_time = []
    for i in range(1,len(timestamp)):
        index_dif = meas_index[i]-meas_index[i-1]
        t_dif = timestamp[i]-timestamp[i-1]
        if index_dif == 1:
            raman_meas_time.append(t_dif.seconds)
    raman_meas_time = outliers(raman_meas_time, 1)
    mean_time = np.nanmean(np.array(raman_meas_time))
    std_time = np.nanstd(np.array(raman_meas_time))
    fig,ax = plt.subplots()
    plt.scatter(range(len(raman_meas_time)),raman_meas_time)
    layout(ax, x_label='Line Scan Index', y_label='Line Scan Time (s)')
    return(mean_time/3600)
                    

    
### Raman Analysis based on eva_concgradient class.

class eva_concgradient():
    def __init__(self, p_echem):
        echem =  auto(p_echem,circ = ['Rp','Rp'],lf_limit=1, hf_limit=10000)
        self.echem = echem
        t_OCV = float(echem['data']['2 OCV']['time/s'].iloc[-1]/3600)
        self.t_OCV = t_OCV
        t_CP = float(echem['data']['5 CP'].loc[echem['data']['5 CP']['half cycle']==1]['time/s']/3600)-t_OCV
        self.t_CP = t_CP
        I_areal = echem['data']['5 CP']['I/mA'].mean()/echem['electrode surface area'] # areal current in mA/cm2
        self.I_areal = I_areal
        return

def evacut_LFO(eva_class):
    '''
    return the cutted eva (remove all OCV scans)
    '''
    eva = eva_class.eva
    t_OCV = eva_class.t_OCV 
    t_linescan= eva_class.t_linescan
    start_index = t_OCV/t_linescan
    print("The index when current applied is "+str(round(start_index,2)))
    time_cp = np.array(list(eva.keys()))-t_OCV/t_linescan # in t_linescan units
    time_cp_h = time_cp * t_linescan
    eva_class.time_CP = time_cp_h
    eva_cut = {}
    c_ini_mean = []
    fig,ax = plt.subplots()
    for i,entry in enumerate(eva):
        time = time_cp_h[i]
        if time >=0:
            eva_cut[entry-int(start_index)] = eva[entry] # Create a new eva_cut with index that correspoinds to timestamps in hour from onset of CC.
        else:
            c_ini_mean.append(np.nanmean(eva[entry][1][3:-3]))
            plt.scatter(np.array(eva[entry][0][3:-3])/1000,eva[entry][1][3:-3])
    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)', title = 'OCV Gradient')
    if len(c_ini_mean)!=0:         
        c_ini = np.array(c_ini_mean).mean()
        c_ini_std = np.array(c_ini_mean).std()
        print('Initial Concentration is: '+str(c_ini)+' -+'+str(c_ini_std))
        eva_class.c_ini = c_ini
    eva_class.eva_cut = eva_cut
    return

def plot_concgradient(eva, time_steps=23/60, list_del =[], save =''):
    '''    

    '''
    
    fig,ax = plt.subplots()
    # colors = color_gradient(len(eva))
    # time = np.arange(0, len(eva)*time_steps, time_steps)
    colors = color_gradient(max(eva.keys())+1)
    time = []
    for entry in eva:
        time.append(entry*time_steps)
    time = np.array(time)
    # delete unwanted time
    keys=list(eva.keys()) #It is a dict_keys object and cannot be used as a dictionary key. So it needs to be converted into a hashable object, such as a list or tuple.
    keys=np.array(keys)
    eva_dict_slice={}
    indices_to_delete = list_del
    keys = np.delete(keys, indices_to_delete)
    for k in list(keys):
        eva_dict_slice[k]=eva[k]

    # plot data
    for i,entry in enumerate(eva_dict_slice):
        z_v = np.array(eva_dict_slice[entry][0])/1000
        c_v = eva_dict_slice[entry][1]
        plt.scatter(z_v,c_v, color = colors[entry])
        
    # add colorbar
    # Create a normalized color map
    norm = plt.Normalize(min(time), max(time))
    cmap = cm.viridis

    # Create a ScalarMappable object
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Plot a colorbar
    cb = plt.colorbar(sm)
    # tick_locator = ticker.MaxNLocator(nbins=len(eva))
    # cb.locator = tick_locator
    # cb.update_ticks()
    cb.set_label('time(h)')
    
    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)')
    #show the figure

    if save !='':
        plt.savefig(save+'.svg',transparent = True)

def eva_cutZ(eva_class, z_del = []):
    '''"Version 8.0 LiFSIG4 50sp 20231109.ipynb"
    Enter eva and the z_value index to remove from z. Returns a trimmed eva. Adjust the z_values if from the head is cut off (i.e. if first point is unsufficient signal, delete that and set the second z value to zero.)
    '''
    eva = eva_class.eva_cut
    
    eva_cutZ = {}
    for entry in eva:
        z = np.delete(eva[entry][0], z_del)
        z = np.array(z)-z[0]
        c = np.delete(eva[entry][1], z_del)
        eva_cutZ[entry] = [z,c]
    eva_class.eva_cutZ = eva_cutZ
    eva_class.z_valuesCut = eva_cutZ[list(dict.keys(eva_cutZ))[0]][0][-1]
    return

from copy import deepcopy

from copy import deepcopy
def removeoutliers_LFO(eva_class, s=0.05, save = ''):
    '''
    Enter eva_class. Returns a 
    ---
    return eva_good 
    '''
    eva = eva_class.eva_cutZ
    c_ini = eva_class.c_ini
    e_good=deepcopy(eva)
    z_max = eva_class.z_valuesCut
    keys = list(dict.keys(eva))
    colors = color_gradient(keys[-1]+1)
    time = eva_class.time_CP[eva_class.time_CP > 0]
    fig,ax = plt.subplots()
    chi_lst = []
    k = 0
    for i in keys:
        x=np.array(eva[i][0])
        ydata=np.array(eva[i][1])
        def func(x, a, b):return  (a)*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(a)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))+c_ini

        #delete "nan" value in the eva_good
        mask=~np.isnan(ydata)
        ydata=ydata[mask]
        x=x[mask]

        #delete negative value
        indices_to_delete= np.argwhere(ydata<0)
        ydata=np.delete(ydata, indices_to_delete)
        x=np.delete(x, indices_to_delete)


        #lmfit function described above
        gmodel = Model(func)
        params = Parameters()
        params.add('a', value=-184000, vary=True)#min=100000, max=300000)
        params.add('b', value=0.004, vary=True, min=1e-6, max=1e-1)
        result = gmodel.fit(ydata, params, x=x, weights=np.sqrt(1.0/ydata))
        
        #plot
        #plt.plot(x, result.best_fit, label = "Fitted Curve"+str(i), color =  colors[i])
        plt.plot(x/1000, result.best_fit, color = colors[i])


        fake_a=result.params['a'].value
        fake_b=result.params['b'].value
        chi_sq = float(result.fit_report().split('chi-square')[1].split('\n')[0].split('=')[1])
        chi_lst.append(chi_sq)  
          
        for j in range(0,len(x),1):
            func_calc = func(eva[i][0][j],fake_a,fake_b)
            if abs(eva[i][1][j]-func_calc)>s:
                e_good[i][1][j]=np.nan
        
         # add colorbar
    # Create a normalized color map
    norm = plt.Normalize(min(time), max(time))
    cmap = cm.viridis

    # Create a ScalarMappable object
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm)
    cb.set_label('time(h)')

    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)')
    if save !='':
        plt.savefig(save+'.svg',transparent = True)   
    fig,ax = plt.subplots()
    plt.scatter(range(len(chi_lst)),chi_lst)
    layout(ax, x_label='Measurement Index', y_label= 'chi squared') 
    eva_class.eva_good = e_good
    return

# Lorenz edited Functions
def noneTOnan(list):
    if None in list:
        modified_list = [np.nan if value is None else value for value in list]
        return(modified_list)
    else:
        return(list)

def curvefitting_LFO(eva_class):
    eva = eva_class.eva_good
    keys = list(dict.keys(eva))
    z_max = eva_class.z_valuesCut
    c_ini = eva_class.c_ini
    chi_lst = []
    report_list = []
    max_list = []
    min_list = []
    a_list = []
    b_list = []
    a_list_err = []
    b_list_err = []
    area_plate_list = []
    area_strip_list = []
    area_plate_err_list = []
    area_strip_err_list = []

    def func(x, a, b):
        return  (a)/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(a/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))+c_ini

    fig,ax = plt.subplots()
    colors = color_gradient(keys[-1]+1)
    for counter,i in enumerate(keys):
        # set x_axis and y_axis value
        x=np.array(eva[i][0])
        ydata=np.array(eva[i][1])

        #delete "nan" value in the eva_good
        mask=~np.isnan(ydata)
        ydata=ydata[mask]
        x=x[mask]

        #delete negative value
        indices_to_delete= np.argwhere(ydata<0)
        ydata=np.delete(ydata, indices_to_delete)
        x=np.delete(x, indices_to_delete)

        #lmfit function described above
        gmodel = Model(func)
        params = Parameters()
        params.add('a', value=-184000, vary=True)#min=100000, max=300000)
        params.add('b', value=0.004, vary=True, min=1e-6, max=1e-1)
        result = gmodel.fit(ydata, params, x=x,weights=np.sqrt(1.0/ydata))#weights=np.sqrt(1.0/ydata)
        
        # color = cm.winter(np.linspace(0,1,totalfilenumber))
        # plt.rc('axes', prop_cycle=(cycler('color', color)))
        chi_sq = float(result.fit_report().split('chi-square')[1].split('\n')[0].split('=')[1])
        #print(chi_sq)
        chi_lst.append(chi_sq)
        
        #taking the max and minimum concentration from the fitting and adding to the list above (used for TDF calculation)
        m = max((result.best_fit))
        mi = min((result.best_fit))
        max_list.append(m)
        min_list.append(mi)

        x_full_cell = np.linspace(0,z_max, 10000)
        ax.scatter(x/10000,ydata, color=colors[i])
        ax.plot(x_full_cell/10000, func(x_full_cell, result.params['a'].value, result.params['b'].value), color = colors[i])


        a,a_err,b,b_err = noneTOnan([result.params['a'].value]),noneTOnan([result.params['a'].stderr]),noneTOnan([result.params['b'].value]),noneTOnan([result.params['b'].stderr])
        
        #adding the 'a' (interfacial conc gradient) and 'b'	(diffusion length) and associated errors to the list above
        a_list.append(a[0])
        b_list.append(b[0])
        a_list_err.append(a_err[0])
        b_list_err.append(b_err[0])


    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)')
    fig,ax = plt.subplots()
    plt.scatter(range(len(chi_lst)),chi_lst)
    layout(ax, x_label='Measurement Index', y_label= 'chi squared') 
    eva_class.a_list = a_list
    eva_class.b_list = b_list
    eva_class.a_list_err = a_list_err
    eva_class.b_list_err = b_list_err
    
    return
        

def detect_outlier(lst, correction_fac = 2):
    '''
    Detect outliers within a list; returns a list with index of those outliers.
    '''
    correction = abs(lst-np.nanmedian(lst)) <= np.nanstd(lst)*correction_fac
    index_lst = []
    for counter,value in enumerate(correction):
        if value == False:
            index_lst.append(counter)
    return(index_lst)

def calc_D(eva_class, list_del = []):
    '''
    param figuretitle
    param totalfilenumber: the number of file (remeber the value should be "n-current_start_index" after cut the eva)
    param figureoutput: the output path to save the plot
    param b_list: b_list record the fit parameter b
    param figurecolor: the color of the figure
    param b_list_err: b_list_err record the error of b
    param list1: indices_to_delete, delete the "b" values corresponding to time that you don't want to see
    param Raman_time_step: the time interval between two Raman depth measurments
    ---
    return D,D_EVF
    '''
    b_list = eva_class.b_list
    b_list_err = eva_class.b_list_err
    time = time = eva_class.time_CP[eva_class.time_CP > 0] # in hour
    time_sq = (3600*time)**0.5 # change unit into s^0.5 for the fitting 

    #copy "b_list", "b_list" is array record all concentration gradient fit parameter "b", and also its error
    Ld_list=np.array(b_list[:])
    b_list_err=np.array(b_list_err[:])
    
    # at time=0 s, ideally there is no concentration gradient. Therefore, set initial data to 0.
    # Ld_list[0]=0
    b_list_err[0]=0

    
    # define the indices to delete
    indices_to_delete = list_del
    # create a new array that excludes the elements to delete, this step is going to delete unstable parts
    Ld_list = np.delete(Ld_list, indices_to_delete)
    time_sq = np.delete(time_sq, indices_to_delete)
    b_list_err=np.delete(b_list_err, indices_to_delete)

    # linear fit
    def F_lin(x,K,B):
        return K*x+B
    K, B = optimize.curve_fit(F_lin, time_sq, Ld_list)[0]
    pcov= optimize.curve_fit(F_lin, time_sq, Ld_list)[1]
    Kstd = np.sqrt(np.diag(pcov))
    

    # calculate the diffusion coefficient and plot the data scatter
    # EVF is the excluded volume factor or say solvent velocity factor
    # D_EVF=D*EVF
    D_EVF=(K/2)**2
    
    
    # serr of D_K
    D_EVF_std=D_EVF*Kstd[0]*2/K # error propagation
    # D_EVF_standard_error=D_EVF_std/(totalfilenumber-len(list1))**0.5
    
    
    # plot
    # add errorbar   fmt="_-k",
    fig,ax = plt.subplots()
    plt.scatter(time_sq, Ld_list, label = 'exp')
    #plt.errorbar(l7, Ld_list, yerr=b_list_err)
    # plt.scatter(l7, Ld_list,color=figurecolor[7], lw=2, marker =",")
    plt.errorbar(time_sq, Ld_list, yerr=b_list_err, alpha=0.5,ecolor ='black',elinewidth=0.5,capsize=5,capthick=0.5,linestyle="none")


    #plot the fit equation on the figure
    x_fit = np.arange(time_sq.min(),time_sq.max(),0.5)
    y_fit = K*x_fit+B
    plt.plot(x_fit,y_fit, label = 'fit')


    
    #add more information
    # plt.xlabel('time/S^0.5')
    # plt.ylabel('Ld/m')
    layout(ax, x_label=r'time ($\mathregular{s^{0.5}}$)', y_label=r'($\mathregular{L_d\,m^{-1}}$)')
        # print D_K
    print('D_app = '+str(D_EVF)+" ± "+str(D_EVF_std)+' m^2s^-1')
    b_out, b_err_out = detect_outlier(b_list), detect_outlier(b_list_err)
    print('Possible outliers detected: '+str(b_out)+' and '+str(b_err_out))
    eva_class.D = D_EVF
    eva_class.D_err = D_EVF_std
    return

#def calc_t(J,Dapp,total_file_number,Raman_time_stepa_list,a_list_err,v_z,solvent_velocity_factor,list1,Raman_time_step):
def calc_t(eva_class, solvent_velocity_factor = 1.08, v_z = 1,list_del = []):
    a_list = eva_class.a_list
    a_list_err = eva_class.a_list_err
    J = eva_class.I_areal
    Dapp = eva_class.D
    F=96485.3321      #unit:s A / mol
    J = J*10 # convert from mA/cm2 to A/m2
    #create time array (in hour unit)
    time_array = time = eva_class.time_CP[eva_class.time_CP > 0] # in hour


    # create "a" array, "a_list" is array record all concentration gradient fit parameter "a"
    a_array =np.array(a_list)
    a_err_array=np.array(a_list_err)

    # define the indices to delete
    indices_to_delete = list_del
    # this step is going to delete bad fits
    a_array = np.delete(a_array, indices_to_delete)
    time_array = np.delete(time_array, indices_to_delete)
    a_err_array= np.delete(a_err_array, indices_to_delete)

    #based on the equation, calculate t+0 at different time 
    t0_array =[]
    
    t0_array = 1+solvent_velocity_factor*(v_z*F*Dapp*a_array)/J
    tplus0_error_indi = solvent_velocity_factor*a_err_array*v_z*F*Dapp

    #calculate the mean and standard diveiation of t0
    tplus0 = np.mean(t0_array)
    tplus0_std = np.std(t0_array)


    #print the result
    fig,ax = plt.subplots()
    plt.scatter(time_array,t0_array)
    plt.errorbar(time_array,t0_array, yerr=tplus0_error_indi, alpha=0.5,ecolor ='black',elinewidth=0.5,capsize=5,capthick=0.5,linestyle="none")
    layout(ax, x_label='time (h)', y_label='transference number')
    a_out, a_err_out = detect_outlier(a_list), detect_outlier(a_list_err)
    print('Possible outliers detected: '+str(a_out)+' and '+str(a_err_out))
    print("Cation Transference number ="+str(tplus0)+" ±",tplus0_std)
    eva_class.t_0 = tplus0
    eva_class.t_0_err = tplus0_std 
    return()

def plot_OP(eva_class):
    echem = eva_class.echem
    J = eva_class.I_areal
    echem_CP = echem['data']['5 CP'] #array
    t_ini_CP = echem_CP['time/s'][0]
    t_CP = eva_class.t_linescan
    counter = 0 # just a parameter to count 
    eta_total = []
    echem_t_linescan = []

    #Every waitting time for raman measurement, extract a value of Ewe from CP data, used as the cell potential
    for i in range(len(echem_CP['time/s'])):
        if echem_CP['time/s'][i]>(t_ini_CP+counter*t_CP*3600):  
            eta_total.append(echem_CP['Ewe/V'][i])
            echem_t_linescan.append(echem_CP['time/s'][i]-t_ini_CP)
            counter += 1

    
    # all data extracted are from the current applied to the finish:         

    # eta_total: the total overpotential
    # R_bulk:    the bulk resistance
    # R_ct：     the charge transfer resistance
    # eta_c:     the concentration overpotential

    eta_total    =np.array(eta_total)                                        #array    unit: V   
    R_bulk       =np.array(echem['eva']["4 PEIS"]['Nyquist parameter']['R0']) #array    unit: Ohm
    R_ct         =np.array(echem['eva']["4 PEIS"]['Nyquist parameter']['R1']) #array    unit: Ohm
    cut_off = np.array([len(eta_total),len(R_bulk),len(R_ct)]).min() # for some reason, sometimes the lenght of OP do not add up. Should be checked no time now
    eta_c = eta_total[:cut_off]-1e-3*J*echem['electrode surface area']*(R_bulk[:cut_off]+R_ct[:cut_off]) 
    eva_class.eta_total, eva_class.eta_c, eva_class.echem_t_linescan = eta_total, eta_c, echem_t_linescan
    #eta_c = eta_c[0:n-current_start_index] # make sure the length of eta_c and eva_cut are same

    fig,ax = plt.subplots(2, 2, figsize=(8, 6))
    parameter = eta_total
    ax[0,0].scatter(range(len(parameter)),parameter, label ='Total Overpotential')
    ax[0, 0].set_title('Total Overpotential')
    parameter = R_bulk
    ax[0,1].scatter(range(len(parameter)),parameter, label ='Total Overpotential')
    ax[0, 1].set_title('Bulk Resistance')
    parameter = R_ct
    ax[1,0].scatter(range(len(parameter)),parameter, label ='Total Overpotential')
    ax[1, 0].set_title('Rct')
    parameter = eta_c
    ax[1,1].scatter(range(len(parameter)),parameter, label ='Total Overpotential')
    ax[1, 1].set_title('Concentration Overpotential')

    plt.tight_layout()


def calc_X(eva_class,list_del = []):
    '''
    param eva_good: the eva dict without outliers
    param T: temperature
    param t: cation transference number
    param t_err: the error of cation transference number
    param totalfile: the number of file (remember the value should be "n-current_start_index" after cut the eva)
    param eta_c: the concentration overpotential 
    param p10: the output path of figure
    param a_list,b_list,a_list_err,b_list_err: just four lists calculated before
    param c7: the converted_initial_concentration   unit: mol/L
    param z7: the distance between two electrode  unit: um
    param list1: indices_to_delete, delete the "b" values corresponding to time that you don't want to see
    ---
    return thermodynamic factor
    '''
    eva = eva_class.eva_good
    T = eva_class.temp
    t = eva_class.t_0
    t_err = eva_class.t_0_err
    eta_c = eva_class.eta_c
    a_list, b_list, a_list_err, b_list_err = eva_class.a_list, eva_class.b_list, eva_class.a_list_err, eva_class.b_list_err
    c_ini = eva_class.c_ini
    z_max = eva_class.z_valuesCut
    keys = list(dict.keys(eva_class.eva_cut))

    F=96485.3321      #unit:s A / mol
    R=8.314           #J/(mol·K）
    


    
    eta_c = eta_c[keys] # just pick the overpotentials where a and b values are avaiable.
    # get 【x-axis】 array: which is ln(Cs,z=L /Cs,z=0)
    x_axis=[0] # The first data is meaningless, set to 0
    x_axis_std=[0] # The first data is meaningless, set to 0
    

    def func(x, a, b):
        return  (a)/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(a/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))+c_ini
    
    def calculate_standard_deviation(x, a, b, aerr, berr):
        # Computes the partial derivatives of func(x, a, b) with respect to the parameters a and b
        df_da = 1/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(1/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))
        df_db = (a / 1000) * ((1 / (3.1415 ** 0.5)) * np.exp(-((((x - z_max) / 1e6) / b) ** 2)) -np.exp(-((((x) / 1e6) / b) ** 2)))
        
        # Calculate standard deviation
        std_dev = np.sqrt((df_da * aerr) ** 2 + (df_db * berr) ** 2)
        return std_dev

    for i in range(1,len(a_list),1):
        func_z_final = func(z_max,a_list[i],b_list[i])
        func_z_0 = func(0,a_list[i],b_list[i])
        x_axis.append(-math.log(func_z_final/func_z_0))
        
        # standard_deviation of x_axis
        sigma_L=calculate_standard_deviation(z_max, a_list[i],b_list[i], a_list_err[i],b_list_err[i])
        sigma_0=calculate_standard_deviation(0, a_list[i],b_list[i], a_list_err[i],b_list_err[i])

        sigma_x=np.sqrt((sigma_L/func(z_max,a_list[i],b_list[i])) ** 2 + (sigma_0/func(0,a_list[i],b_list[i])) ** 2)
        x_axis_std.append(sigma_x)

    


    # get 【y-axis】 array: which is eta_c*F/2/R/T/(1-t) , "t" is cation transferance number 
    y_axis=eta_c*F/(  2*R*T*(1-t)  )

    
    
    # delete the bad fits 
    # this step is going to delete some fits you dont want to see
    x_axis = np.delete(x_axis, list_del)
    y_axis = np.delete(y_axis, list_del)
    x_axis_std = np.delete(x_axis_std, list_del)
    
    # error propagation for y_axis
    y_axis_std = abs(t_err*y_axis/(1-t))
    
    # lineal fit and print X value,and calculate the error of X
    def f10(x,K,B): 
        return K*x+B
    K,B= optimize.curve_fit(f10,x_axis,y_axis)[0]
    pcov= optimize.curve_fit(f10,x_axis,y_axis)[1]
    x_std = np.sqrt(np.diag(pcov))
    x_standard_error=x_std[0]/(len(a_list))**0.5
    
    #plot data point
    fig,ax = plt.subplots()
    plt.scatter(x_axis, y_axis, s=40, marker =",")

    #plot errorbar     (xerr=,_-k)
    plt.errorbar(x_axis, y_axis, yerr=y_axis_std,xerr=x_axis_std,  fmt="",alpha=0.5,ecolor ="black",elinewidth=0.5,capsize=5,capthick=0.5,linestyle="none")
   
    # plot the linear fit
    x_fit = np.arange(x_axis.min(),x_axis.max(),0.02) # x10 and y10 are just used to show the lineal equation
    y_fit = K*x_fit+B
    plt.plot(x_fit, y_fit, label='fit'+'X='+str(round(K,3))+"±"+str(round(x_std[0],3)))
    layout(ax, x_label ='ln(c${_s}_{z=L}$ /c${_s}_{z=0}$)', y_label='$η_{c}$ F/2RT(1-t${_+^0}$)')
    

    #plt.savefig(p10,transparent = True)   
    eva_class.X = K
    eva_class.X_err = x_std[0] 
    print('X = '+str(K)+" ± "+str(x_std[0]))
    return

def calc_cond(eva_class, list_del = [], R_OCV = [], z_cut = True):
    echem = eva_class.echem
    if z_cut == True: 
        z_max = eva_class.z_valuesCut
    else:
        z_max = eva_class.z_values
        
    L = z_max/1000000 # unit: m
    t_OCV = float(echem['data']['2 OCV']['time/s'].iloc[-1]/3600)
    
    R_bulk_OCV = echem['eva']['1 PEIS']['Nyquist parameter']['R0']
    t_bulk_OCV = np.array(range(len(R_bulk_OCV)))/len(R_bulk_OCV)*t_OCV
    R_bulk_CP = echem['eva']['4 PEIS']['Nyquist parameter']['R0']  # unit: Ohm
    t_bulk_CP = np.array(range(len(R_bulk_CP)))*23/60+t_OCV
    
    fig,ax = plt.subplots()
    plt.scatter(t_bulk_OCV, R_bulk_OCV, label ='OCV')
    plt.scatter(t_bulk_CP,R_bulk_CP, label = 'CP')
    
    R_bulk_OCV = np.delete(R_bulk_OCV, R_OCV)
    t_bulk_OCV = np.delete(t_bulk_OCV, R_OCV)
    R_bulk_CP = np.delete(R_bulk_CP, list_del)
    t_bulk_CP = np.delete(t_bulk_CP, list_del)
    plt.scatter(t_bulk_OCV, R_bulk_OCV, label ='selected', color = 'red', facecolor = 'none')
    plt.scatter(t_bulk_CP,R_bulk_CP, color = 'red', facecolor = 'none')
    layout(ax, x_label = 'time (h)', y_label= 'R0 (Ohm)')

    R_bulk_CP = [x for x in R_bulk_CP if not np.isnan(x)]
    R_bulk_OCV = [x for x in R_bulk_OCV if not np.isnan(x)]
    
    R_bulk = np.array((list(R_bulk_OCV)+list(R_bulk_CP))).mean()
    k_ionic=L/R_bulk/(echem['electrode surface area']*1e-4)*10  


    #  error propagation
    R_bulk_standard_deviation=np.std(echem['eva']["4 PEIS"]['Nyquist parameter']['R0'][1:])
    k_ionic_standard_deviation=k_ionic*(R_bulk_standard_deviation/R_bulk)
    k_ionic_standard_error=k_ionic_standard_deviation/(len(echem['eva']["4 PEIS"]['Nyquist parameter']['R0'][1:]))**0.5

    k_ionic_error=k_ionic_standard_deviation
    eva_class.k_ionic = k_ionic
    eva_class.k_ionic_err = k_ionic_error
    print("ionic conductivity is: "+str(k_ionic)+"±"+str(k_ionic_error)+"  mS cm-1")
    return
    