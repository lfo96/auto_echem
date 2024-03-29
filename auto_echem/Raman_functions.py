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
from scipy.integrate import simps
from auto_echem.auto import auto
import matplotlib.cm     as cm
from lmfit       import Model, Parameters, minimize, fit_report
from auto_echem.general_functions import color_gradient
from auto_echem.general_functions import layout
from auto_echem.general_functions import outliers


from scipy       import integrate
import numpy             as np

import os
import copy

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

## External Reference Data Processing
from  BaselineRemoval  import  BaselineRemoval
def BL_removal(y_data):
    baseObj = BaselineRemoval(y_data) 
    # 3 way to do baseline subtraction
    # Modpoly_output = baseObj.ModPoly (polynomial_Degree) 
    # Imodpoly_output = baseObj.IModPoly (polynomial_Degree)
    Zhangfit_output = baseObj.ZhangFit()
    return(Zhangfit_output)

import numpy as np
from scipy.optimize import curve_fit
def gaussian(x, A, mu, sigma):
    """
    Gaussian function.

    Parameters:
    - x: Independent variable.
    - A: Amplitude of the Gaussian.
    - mu: Mean (center) of the Gaussian.
    - sigma: Standard deviation of the Gaussian.

    Returns:
    - Value of the Gaussian at x.
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gaussian_to_data(x_data, y_data):
    """
    Fit a Gaussian function to experimental data.

    Parameters:
    - x_data: Independent variable data.
    - y_data: Dependent variable data.

    Returns:
    - A tuple (A, mu, sigma) representing the parameters of the fitted Gaussian.
    """
    # Initial guesses for the Gaussian parameters
    initial_guess = (max(y_data), np.mean(x_data), np.std(x_data))

    # Use curve_fit to fit the Gaussian function to the data
    params, covariance = curve_fit(gaussian, x_data, y_data, p0=initial_guess)

    # Extract the fitted parameters
    A, mu, sigma = params

    return A, mu, sigma

def peak_ratio(pathway, BLR = False, BL_FSI = [], BL_G4 = [], wn_min = 680,wn_max = 760):
    '''
    Enter pathway of Raman line scan, specify of Baseline removal should be done, provide OCV BL correction data from OCV measurement for FSI (which corresponds to int_norm_FSI) and/or for G4 (which corresponds to int_norm_G4) and wavenumber range for FSI peak integration.
    Returns eva, np.array(z_list)/1000,ref_max_wn, ref_max_lst, int_G4, int_norm_G4, int_norm_FSI, int_normG4_FSI
    '''
    df = pd.read_csv(pathway, sep='\s+')
    eva = {}
    try:
        z_list = list(dict.fromkeys(df['#Z'].tolist()))
    except KeyError:
        df['#Z']=len(df)*[0]
        z_list = [0]

    ref_max_lst = []
    ref_max_wn = []
    int_norm_FSI = []
    int_normG4_FSI = []
    int_G4 = []
    int_norm_G4 = []
    
    for i,z_i in enumerate(z_list):
        if BLR == True:
            df_z = df.loc[df['#Z']==z_i].copy()
            df_z['#Intensity'] = BL_removal(df_z['#Intensity'])
        else:
            df_z = df.loc[df['#Z']==z_i].copy()
        
        # df for the reference peak - integrate or peak max? Also which range for reference peak
        df_z_ref = df_z.loc[(df_z['#Wave']>=407.5) & (df_z['#Wave']<=425)]
        #df_z_ref = df_z.loc[(df_z['#Wave']>=1060) & (df_z['#Wave']<=1080)]
        #df_z_ref = df_z.loc[(df_z['#Wave']>=635) & (df_z['#Wave']<=650)]
        x_data = np.linspace(df_z_ref['#Wave'].iloc[0], df_z_ref['#Wave'].iloc[-1], 1000)
        
        # Use the index to get the corresponding x value from x_data and save the ref max wavenumber (can be used for T determination)
        try:
            A, mu, sigma = fit_gaussian_to_data(df_z_ref['#Wave'],df_z_ref['#Intensity'])
            max_index = np.argmax(gaussian(x_data,A,mu,sigma))
            x_at_max = x_data[max_index]
            ref_max_wn.append(x_at_max)
        except RuntimeError:
            print('Runtime Error detected.')
            ref_max_wn.append(np.nan)

        # Decide which kind of reference max caclulation
        ref_max = gaussian(x_data,A,mu,sigma).max()
        #ref_max = df_z_ref['#Intensity'].max()
        #ref_max = simps(df_z_ref['#Intensity'])
        df_z['#Intensity_norm'] = df_z['#Intensity']/ref_max  
        ref_max_lst.append(ref_max)
        
        # solvent integration
        df_G4 = df_z.loc[df_z['#Wave']>=1400]
        # int = df_G4['#Intensity_norm'].max()
        int = df_G4['#Intensity'].max()
        int_G4.append(int)
        df_z['#Intensity_normG4'] = df_z['#Intensity']/int
        
        int_norm = df_G4['#Intensity_norm'].max()
        if len(BL_G4)!= 0:
            G4_BLcorrection = BL_G4[i]
            factor = int_norm/G4_BLcorrection
            int_norm_G4.append(factor)
        else:
            int_norm_G4.append(int_norm)
        
        # FSI Concentration determination by integrating the Raman spectre in between wn_min and wn_max
        df_FSI = df_z.loc[(df_z['#Wave'] >= wn_min) & (df_z['#Wave'] <= wn_max)]
        int = simps(df_FSI['#Intensity_norm'])
        if len(BL_FSI)!= 0:
            FSI_BLcorrection = BL_FSI[i]
            factor = int/FSI_BLcorrection
            int_norm_FSI.append(factor)
        else:
            int_norm_FSI.append(int)

        int_normG4_FSI.append(simps(df_FSI['#Intensity_normG4']))

        eva[z_i/1000] = df_z
    return(eva, np.array(z_list)/1000,ref_max_wn, ref_max_lst, int_G4, int_norm_G4, int_norm_FSI, int_normG4_FSI)

def eva_load_extref(eva_class):
    '''
    Insert p_raman pathway. Returns a eva dictioniary with the peak_ratio evaluated files.
    '''
    
    p_raman = eva_class.p_raman
    folder = ('\\').join(p_raman.split('\\')[:-1])
    direc_files = list_files_with_last_modified_sorted(folder)
    auto_expo_files = []
    for entry in direc_files:
        if entry[0].split('.')[-1] == 'txt' and entry[0].split('.')[0].split('_')[-1].isdigit() == True:
            auto_expo_files.append(entry[0])
    eva = {}
    for entry in auto_expo_files:
        index = int(entry.split('.')[0].split('_')[-1])
        try:
            eva_i = peak_ratio(folder+'\\'+entry, BLR=True)
        except RuntimeError:
            print(str(index)+' timed out.')
            continue
        eva[index] = eva_i
        print(str(index)+' processed.')
    eva_class.eva_extref_noBL = eva
    eva_class.auto_expo_txt = auto_expo_files
    eva_class.extref = True
    return

def create_BL_OCV(eva_class,start = '', stop = ''):
    '''
    Insert eva_class which has a eva defined from eva_load_extref.
    '''
    t_OCV, t_linescan = eva_class.t_OCV, eva_class.t_linescan
    eva_extref_noBL = eva_class.eva_extref_noBL
    start_index = t_OCV/t_linescan
    print("The index when current applied is "+str(round(start_index,2)))
    
    # Create Baseline of Sapphire Reference Signal from OCV condition (where system is under thermal equilibirum)
    if start =='' and stop =='':
        #start, stop = 3, int(start_index)
        start, stop = 7, int(start_index)
        
    int_FSI_lst = []
    int_G4_lst = []
    for counter,i in enumerate(range(start,stop)):
        int_G4 = pd.Series(np.array(eva_extref_noBL[i][5]))
        int_G4_lst.append(int_G4)
        int_FSI = pd.Series(np.array(eva_extref_noBL[i][6]))
        int_FSI_lst.append(int_FSI)
        
    int_FSI_concat = pd.concat(int_FSI_lst, axis =1)
    int_FSI_mean, int_FSI_std = int_FSI_concat.mean(axis=1),int_FSI_concat.std(axis=1)
    
    int_G4_concat = pd.concat(int_G4_lst, axis =1)
    int_G4_mean, int_G4_std = int_G4_concat.mean(axis=1),int_G4_concat.std(axis=1)
    
    eva_class.start_index = int(start_index)
    eva_class.int_FSI_mean, eva_class.int_FSI_std = int_FSI_mean, int_FSI_std
    eva_class.int_G4_mean, eva_class.int_G4_std = int_G4_mean, int_G4_std
    return

def eva_OCV_BL(eva_class):
    '''
    Insert eva_class and do the baseline correction
    '''
    auto_expo_txt = eva_class.auto_expo_txt
    int_FSI_mean = eva_class.int_FSI_mean
    int_G4_mean = eva_class.int_G4_mean
    p_raman = eva_class.p_raman
    folder = ('\\').join(p_raman.split('\\')[:-1])
    eva_extref = {}
    for entry in auto_expo_txt:
        index = int(entry.split('.')[0].split('_')[-1])
        eva_i = peak_ratio(folder+'\\'+entry, BLR=True, BL_G4 = int_G4_mean, BL_FSI = int_FSI_mean)
        eva_extref[index] = eva_i
        print(str(index)+' processed.')
    eva_class.eva_extref = eva_extref
    return

def plot_gradient_BLcorrected(eva_class):
    keys = list(dict(eva_class.eva_extref).keys())
    start_index = eva_class.start_index
    start, stop = 0, start_index
    fig,ax = plt.subplots()
    colors = color_gradient(stop-start)
    for counter,i in enumerate(range(start,stop)):
        plt.scatter(eva_class.eva_extref[i][1][1:], eva_class.eva_extref[i][6][1:],  color = colors[counter])
    layout(ax, x_label='Z (um)', y_label = 'Normalized Peak Change',  title = 'OCV')
    
    start, stop = start_index,keys[-1]
    fig,ax = plt.subplots()
    colors = color_gradient(stop-start)
    for counter,i in enumerate(range(start,stop)):
        plt.scatter(eva_class.eva_extref[i][1][1:], eva_class.eva_extref[i][6][1:],  color = colors[counter])
    layout(ax, x_label='Z (um)', y_label = 'Normalized Peak Change',  title = 'CP')
    
    return

def signal_integration_map(eva_class, plot = True, eva_cutZ=False):
    '''
    Insert eva_class which contains the BLR corrected Intensity_norm eva's named eva_class.eva_extref and add the intensity map (integrated from 425 till the end) for all z and t values under eva_class.int_map
    '''
    eva = eva_class.eva_extref
    z_lst = list(eva[0][0].keys())
    I_int = []
    for entry in eva:
        I_int_z = []
        for z_i in z_lst:
            df = eva[entry][0][z_i]
            # t ake out the sapphire reference peak
            df_int = df.loc[df['#Wave']>425]
            I_int_i = simps(-df_int['#Intensity_norm'],df_int['#Wave'])
            I_int_z.append(I_int_i) 
        I_int.append(I_int_z)
    
    if eva_cutZ == True:
        z_del = eva_class.eva_cutZ_indices
        z_del_positive = [len(I_int[0]) + x if x < 0 else x for x in z_del]
        z_del_sorted = sorted(z_del_positive, reverse=True)
        for idx in z_del_sorted:
            I_int = np.delete(I_int, idx, axis=1)
            
    I_int_norm = (I_int - np.min(I_int)) / (np.max(I_int) - np.min(I_int))   
    
    eva_class.int_map = I_int
    eva_class.int_map_norm = I_int_norm
     
    if plot == True:
        # Function to show the heat map
        plt.imshow(eva_class.int_map_norm, cmap = 'magma', interpolation='none', aspect = 'auto')
        
        # Adding details to the plot
        plt.xlabel('Z (mm)')
        plt.ylabel('time (h)')

        plt.colorbar()
    return

density_1MLiFSIG4 = {
    273+20 : 1.1201,
    273+30 : 1.1112,
    273+40 : 1.1023,
    273+50 : 1.0934}

G4_1MLiFSIG4_conc = {
    20 : 4.244534124959221,
    30 : 4.21080824895517,
    40 : 4.17708237295112,
    50 : 4.143356496947069,
}

def molal_to_molar(molal,roh, M=0.1871):
    '''
    Convert molality (mol/kg) into molarity (mol/L). 
    Insert the molal value, density and Molar mass.
    Assumes LiFSI as a salt if not specified otherwise (187.1 g/mol)    
    '''
    c = (roh*molal)/(1+molal*M)
    return(c)


def convert_to_eva(eva_class):
    eva_ini = eva_class.eva_extref
    roh = density_1MLiFSIG4[eva_class.temp]
    m_ini = eva_class.m_ini
    # Raman measurement is probing volume not mass; so every change in area corresponds to a change in molar concentration. Thus need to shift everything by the molality to molarity conversion.
    cor_factor = molal_to_molar(m_ini,roh)
    print('Initial concentration has been converted from '+str(m_ini)+' mol/kg to '+str(round(cor_factor,3))+' mol/L')
    eva_class.c_ini = cor_factor
    '''
    Convert the prepared eva_class.eva_extref into a eva_final in the same form as the conventional analysis has happened.
    '''
    eva_final = {}
    for i,entry in enumerate(eva_ini):
        z_lst = list(np.array(eva_ini[entry][1])*1000) # convert back to um
        area_change = eva_ini[entry][6] # FSI intensity change
        FSI_conc = np.array(area_change)*cor_factor # convert the area change to a concentrationFSI_concconcentration
        eva_final[i] = [z_lst,FSI_conc]
    return(eva_final)


# Measured from density measurements; see PaperFigures.ipynb for calculation.
SVF_1MLiFSIG4 = {
    293 : 1.0764343203802311,
    303 : 1.0756299106624074,
    313 : 1.0745979630050677,   
    323: 1.0735633508585876
    }   
    
    
    

## Functions to determine the time for a raman linescan
import os
from datetime import datetime

def find_BL(p_raman):
    '''
    Find the manaually BL corrected files in a directory.
    Returns a list of all the _BL files and a list of the index of the measurements.
    '''
    BL_lst = []
    index_lst = []
    direc = list_files_with_last_modified_sorted('\\'.join(p_raman.split('\\')[:-1]))
    for entry in direc:
        if len(entry[0])>=7 and (entry[0][-6]+entry[0][-5]) == 'BL' and entry[0][-1] == 't':
            BL_lst.append(entry[0])
            index = entry[0].split('_BL')[0]
            index = index.split('_')[-1]
            index_lst.append(int(index))
    return(BL_lst,index_lst)

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
        echem['electrode surface area'] = 0.4*0.4*np.pi
        print('Electrode Surface Area manually set to 0.503 cm2.')
        self.echem = echem
        t_OCV = float(echem['data']['2 OCV']['time/s'].iloc[-1]/3600)
        self.t_OCV = t_OCV
        t_CP = float(echem['data']['5 CP'].loc[echem['data']['5 CP']['half cycle']==1]['time/s']/3600)-t_OCV
        self.t_CP = t_CP
        I_areal = echem['data']['5 CP']['I/mA'].mean()/echem['electrode surface area'] # areal current in mA/cm2
        self.I_areal = I_areal
        return
# old evacut - messes up the c_in; not sure why I did that in the first place
# def evacut_LFO(eva_class):
#     '''
#     return the cutted eva (remove all OCV scans)
#     '''
#     eva = eva_class.eva
#     t_OCV = eva_class.t_OCV 
#     t_linescan= eva_class.t_linescan
#     start_index = t_OCV/t_linescan
#     print("The index when current applied is "+str(round(start_index,2)))
#     time_cp = np.array(list(eva.keys()))-t_OCV/t_linescan # in t_linescan units
#     time_cp_h = time_cp * t_linescan
#     eva_class.time_CP = time_cp_h
#     eva_cut = {}
#     c_ini_mean = []
#     fig,ax = plt.subplots()
#     for i,entry in enumerate(eva):
#         time = time_cp_h[i]
#         if time >=0:
#             eva_cut[entry-int(round(start_index))] = eva[entry] # Create a new eva_cut with index that correspoinds to timestamps in hour from onset of CC.
#         else:
#             c_ini_mean.append(np.nanmean(eva[entry][1][3:-3]))
#             plt.scatter(np.array(eva[entry][0][3:-3])/1000,eva[entry][1][3:-3])
#     layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)', title = 'OCV Gradient')
#     if len(c_ini_mean)!=0:         
#         c_ini = np.array(c_ini_mean).mean()
#         c_ini_std = np.array(c_ini_mean).std()
#         print('Initial Concentration is: '+str(c_ini)+' -+'+str(c_ini_std))
#         eva_class.c_ini = c_ini
#     eva_class.eva_cut = eva_cut
#     return

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
    fig,ax = plt.subplots()
    for i,entry in enumerate(eva):
        time = time_cp_h[i]
        if time >=0:
            eva_cut[entry-int(round(start_index))] = eva[entry] # Create a new eva_cut with index that correspoinds to timestamps in hour from onset of CC.
        else:
            plt.scatter(np.array(eva[entry][0][:])/1000,eva[entry][1][:])
    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)', title = 'OCV Gradient')
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
    '''
    This is purely to remove edge points if the linescan is already on piston height, beyond the electrolyte. Not due to lithium filaments. Z value are being changed!
    Enter eva and the z_value index to remove from z. Returns a trimmed eva. Adjust the z_values if from the head is cut off (i.e. if first point is unsufficient signal, delete that and set the second z value to zero.)
    '''
    eva = eva_class.eva_cut
    
    eva_cutZ = {}
    for entry in eva:
        z = np.delete(eva[entry][0], z_del)
        z = list(np.array(z)-z[0])
        c = list(np.delete(eva[entry][1], z_del))
        eva_cutZ[entry] = [z,c]
    if hasattr(eva_class,'int_map_norm')== True:
        int_map_norm = eva_class.int_map_norm
        int_map_norm_cutZ = []
        for entry in int_map_norm:
            int_map_norm_cutZ.append(np.delete(entry, z_del))
        eva_class.int_map_norm = int_map_norm_cutZ
            
    eva_cutZ_indices = z_del
    eva_class.eva_cutZ_indices = eva_cutZ_indices            
    eva_class.eva_cutZ = eva_cutZ
    eva_class.z_valuesCut = eva_cutZ[list(dict.keys(eva_cutZ))[0]][0]
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
    z_max = eva_class.z_valuesCut[-1]
    keys = list(dict.keys(eva))
    colors = color_gradient(np.array(keys).max()+1)
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
    # Convert to list to make it json serizable
    def convert_np_arrays_to_lists(d):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in d.items()}
    e_good_lst = convert_np_arrays_to_lists(e_good)
    eva_class.eva_good = e_good_lst
    return

# Lorenz edited Functions
def noneTOnan(list):
    if None in list:
        modified_list = [np.nan if value is None else value for value in list]
        return(modified_list)
    else:
        return(list)

def int_loss_detection(int_list_norm, int_thresh = 0.3):
    '''
    Insert the normalized intensity list (derived from the int_norm_map[cy] at a given cycle) and return the x-index where not enough signal was measured.
    '''
    sig_loss = int_list_norm < int_thresh
    sig_loss_ix = np.where(sig_loss)[0]
    return(sig_loss_ix)

def plot_gradient_fit(eva_class, int_loss_filter = 0.3, strip_only = False, index = []):
    if hasattr(eva_class,'eva_good'):
        eva = eva_class.eva_good
    else:
        eva = eva_class.eva_cutZ
    keys = list(dict.keys(eva))
    z_max = eva_class.z_valuesCut[-1]
    c_ini = eva_class.c_ini
    chi_lst = []
    def func(x, a, b):
        return  (a)/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(a/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))+c_ini

    fig,ax = plt.subplots()
    colors = color_gradient(np.array(keys).max()+1)
    for i in index:
        # set x_axis and y_axis value
        x=np.array(eva[i][0])
        ydata=np.array(eva[i][1])
        
        if hasattr(eva_class,'int_map_norm')== True and int_loss_filter is not False: 
            sig_loss_ix = int_loss_detection(eva_class.int_map_norm[i], int_loss_filter)
            for ix in sig_loss_ix:
                ydata[ix] = np.nan
            print('At measurement number '+str(i)+', the z_value index '+str(sig_loss_ix)+' were set to np.nan')
        
        if strip_only != False:
            mid_ix = np.argmin(abs(x-strip_only*z_max))
            ydata[mid_ix:] = np.nan

            
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
        
       

        x_full_cell = np.linspace(0,z_max, 10000)
        ax.scatter(x/10000,ydata, color=colors[i], facecolor = 'none')
        ax.plot(x_full_cell/10000, func(x_full_cell, result.params['a'].value, result.params['b'].value), color = colors[i], label = str(i))


    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)')
    fig,ax = plt.subplots()
    plt.scatter(index,chi_lst)
    layout(ax, x_label='Measurement Index', y_label= 'chi squared') 

def curvefitting_LFO(eva_class, int_loss_filter = 0.3, strip_only = False):
    if hasattr(eva_class,'eva_good'):
        eva = eva_class.eva_good
    else:
        eva = eva_class.eva_cutZ
    keys = list(dict.keys(eva))
    z_max = eva_class.z_valuesCut[-1]
    c_ini = eva_class.c_ini
    chi_lst = []
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
    
    z_fit_lst = []
    conc_fit_lst = []

    def func(x, a, b):
        return  (a)/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(a/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))+c_ini

    fig,ax = plt.subplots()
    colors = color_gradient(np.array(keys).max()+1)
    for i in keys:
        # set x_axis and y_axis value
        x=np.array(eva[i][0])
        ydata=np.array(eva[i][1])
        
        if hasattr(eva_class,'int_map_norm')== True and int_loss_filter is not False: 
            sig_loss_ix = int_loss_detection(eva_class.int_map_norm[i], int_loss_filter)
            for ix in sig_loss_ix:
                ydata[ix] = np.nan
            print('At measurement number '+str(i)+', the z_value index '+str(sig_loss_ix)+' were set to np.nan')
        
        if strip_only != False:
            mid_ix = np.argmin(abs(x-strip_only*z_max))
            ydata[mid_ix:] = np.nan

            
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
        ax.scatter(x/10000,ydata, color=colors[i], facecolor = 'none')
        conc_fit = func(x_full_cell, result.params['a'].value, result.params['b'].value)
        ax.plot(x_full_cell/10000, conc_fit, color = colors[i])
        z_fit_lst.append(x_full_cell/10000)
        conc_fit_lst.append(conc_fit)

        a,a_err,b,b_err = noneTOnan([result.params['a'].value]),noneTOnan([result.params['a'].stderr]),noneTOnan([result.params['b'].value]),noneTOnan([result.params['b'].stderr])
        
        #adding the 'a' (interfacial conc gradient) and 'b'	(diffusion length) and associated errors to the list above
        a_list.append(a[0])
        b_list.append(b[0])
        a_list_err.append(a_err[0])
        b_list_err.append(b_err[0])
        
            
        # Fit integration for transference number calculations.
                #finding the area under the curve for the plating and stripping side
        z_lst = x    
        y_eval = gmodel.eval(result.params, x=z_lst) - c_ini
        y_eval_strip = y_eval[:int(len(y_eval)/2)]
        x_strip = z_lst[:int(len(y_eval)/2)]
        y_eval_plate = y_eval[int(len(y_eval)/2):]
        x_plate = z_lst[int(len(y_eval)/2):]
        
        area_plate = integrate.simps(y_eval_plate, x_plate)
        area_plate_list.append(area_plate)
        area_strip = integrate.simps(y_eval_strip, x_strip)
        area_strip_list.append(area_strip)
        
        # the uncertainty band for the fitting and using this integrl as the error
        dely = result.eval_uncertainty(sigma=1)
        area_plate_err = integrate.simps(dely[int(len(y_eval)/2):], x_plate)
        area_strip_err = integrate.simps(dely[:int(len(y_eval)/2)], x_strip)
        area_plate_err_list.append(area_plate_err)
        area_strip_err_list.append(area_strip_err)

    layout(ax, x_label='Cell length (mm)', y_label=r'Concentration ($\mathregular{mol\,L^{-1}}$)')
    fig,ax = plt.subplots()
    plt.scatter(range(len(chi_lst)),chi_lst)
    layout(ax, x_label='Measurement Index', y_label= 'chi squared') 
    eva_class.a_list = a_list
    eva_class.b_list = b_list
    eva_class.a_list_err = a_list_err
    eva_class.b_list_err = b_list_err
    eva_class.plate_area, eva_class.plate_area_err = area_plate_list, area_plate_err_list
    eva_class.strip_area, eva_class.strip_area_err = area_strip_list, area_strip_err_list
    eva_class.conc_fit_lst = conc_fit_lst
    eva_class.z_fit_lst = z_fit_lst
        
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
    plt.plot(x_fit,y_fit, label = str(round(D_EVF*1e10,2))+'$\mathregular{\cdot 10^{-10} m^2\,s^{-1}}$')


    
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
def calc_t(eva_class, solvent_velocity_factor = 1.08, v_z = 1,list_del = [], error_thresh = 0.2):
    solvent_velocity_factor = SVF_1MLiFSIG4[eva_class.temp]
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

    #based on the equation, calculate t+0 at different time  
    t0_array = 1+solvent_velocity_factor*(v_z*F*Dapp*a_array)/J
    tplus0_error_indi = solvent_velocity_factor*a_err_array*v_z*F*Dapp
    
    #a_out, a_err_out = detect_outlier(a_list), detect_outlier(a_list_err)
    exceeding_indexes = [i for i, value in enumerate(tplus0_error_indi) if value > error_thresh]
    #print('Possible outliers detected: '+str(a_out)+' and '+str(a_err_out))
    print('Index values where error exceeds '+str(error_thresh)+': '+str(exceeding_indexes))
   

    # this step is going to delete bad fits
    t0_array = np.delete(t0_array, list_del)
    time_array = np.delete(time_array, list_del)
    tplus0_error_indi = np.delete(tplus0_error_indi, list_del)
    
    tplus0 = np.mean(t0_array)
    tplus0_std = np.std(t0_array)


    #print the result
    fig,ax = plt.subplots()
    plt.scatter(time_array,t0_array, label = str(round(tplus0,2)))
    plt.errorbar(time_array,t0_array, yerr=tplus0_error_indi, alpha=0.5,ecolor ='black',elinewidth=0.5,capsize=5,capthick=0.5,linestyle="none")
    layout(ax, x_label='time (h)', y_label='transference number')

    eva_class.t_0 = tplus0
    eva_class.t_0_err = tplus0_std 
    print("Cation Transference number ="+str(tplus0)+" ±",tplus0_std)
    return()

def calc_t_hittorf(eva_class, i, f, side = 'strip', t = '', SVF=1.08):
    '''
    Insert eva_class, the start (ini) and final index used for the time difference and the stripping or platting side used.
    Returns tranference number 
    '''
    F=96485.3321      #unit:s A / mol
    SVF = SVF_1MLiFSIG4[eva_class.temp]
    I_areal = eva_class.I_areal
    if np.isscalar(t) and t =='':
        t_linescan = eva_class.time_CP[eva_class.time_CP>0]
    else:
        t_linescan = t
        
    if np.isscalar(side) and side == 'strip':
        A = eva_class.strip_area
    elif np.isscalar(side) and side == 'plate':
        A = eva_class.plate_area
    else:
        # if A is manually supplied i.e. from linear fit.
        A = side    
    delta_t = t_linescan[f]-t_linescan[i]
    # volume_fraction=(1-c_ini*1000*partial_molar_volumes_salt)
    # SVF = 1/volume_fraction

    flux = (A[f]-A[i])/1000 #integration was done in mol/L*um; so this unit is  mol/m^2
    total_charge = 10*I_areal*(delta_t*60*60)    #unit A s /m^2

    t_plus_0_h=1-(flux*F*SVF)/total_charge
    return(t_plus_0_h)

def lin_fit_calct(eva_class,cut_off_time_h = 5, plate = False, list_del = []):
    '''
    Insert eva_class and a cut_off_time_h in hours which sets the cut off time for the linear fit of the integrated concentration values. Linear fit, and the calculate the transference number in hittorf style.
    Returns the transference numbert for the stripping and platting side.
    '''
    
    import scipy
    
    t = eva_class.time_CP[eva_class.time_CP>0]
    cut_off_index = np.argmin(abs(t-cut_off_time_h))
    t = t[:cut_off_index]

    fig,ax = plt.subplots()
    
    x,y = t,eva_class.strip_area[:cut_off_index]
    
    x = np.delete(x, list_del)
    y = np.delete(y, list_del)
    
    plt.scatter(x,y, label = 'strip')
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
    y_fit = slope * np.array(x) + intercept
    t_0_h = calc_t_hittorf(eva_class,0,-1,y_fit,x)
    plt.plot(x,y_fit, label = str(round(t_0_h,2)))
    
    
    if plate != False:
        x,y = t,eva_class.plate_area[:cut_off_index]
        plt.scatter(x,y, label ='plate')
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x,y)
        y_fit = slope * np.array(x) + intercept
        t_0_h_plate = calc_t_hittorf(eva_class,0,-1,y_fit,x)
        plt.plot(x,y_fit,label = str(round(t_0_h_plate,2)))
        layout(ax, y_label = 'Integrated Concentration', x_label='time (h)')
        return(t_0_h,t_0_h_plate)
    
    layout(ax, y_label = 'Integrated Concentration', x_label='time (h)')
    return(t_0_h)


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

def find_time_match(eva_class, time_dif_value = 1):
    '''
    Insert eva_class.
    Finds mathcing time stamps of the Raman linescans and the echem data.
    Returns index of the time_cp (raman) timestamps that should be used an index of the EIS measuremetns that should be used.
    '''
    CP_PEIS = eva_class.echem['data']['4 PEIS']
    cy_no = 0
    CP_PEIS_time = []
    for index in range(len(CP_PEIS)):
        if CP_PEIS['cycle number'].iloc[index]!=cy_no:
            time_EIS = CP_PEIS['time/s'].iloc[index]
            CP_PEIS_time.append(time_EIS)
            cy_no = CP_PEIS['cycle number'].iloc[index]
    CP_PEIS_time = (np.array(CP_PEIS_time)-CP_PEIS_time[0])/3600
    time_CP_pos = eva_class.time_CP[eva_class.time_CP > 0]
    min_index_lst = []
    time_cut = []
    time_cut_index = []
    if len(time_CP_pos)>len(CP_PEIS_time):
        for i,entry in enumerate(CP_PEIS_time):
            time_dif = abs(time_CP_pos-entry).min()
            if time_dif <= time_dif_value:
                # Only return the time stamps if they are within 1h of time difference
                min_index = np.argmin(abs(time_CP_pos-entry))
                min_index_lst.append(min_index)
                time_cut.append(entry)
                time_cut_index.append(i)
            else:
                print('Echem and Raman measurements are too far apart in the '+str(i)+'th EIS measurement: '+ str(round(time_dif,2))+'h')
                continue
        #return(time_CP_pos[min_index_lst],time_cut) #this would reeturn the time arrays
        return(min_index_lst,time_cut_index)
    else:
        for i,entry in enumerate(time_CP_pos):
            time_dif = abs(CP_PEIS_time-entry).min()
            if time_dif <= time_dif_value:
                min_index = np.argmin(abs(CP_PEIS_time-entry))
                min_index_lst.append(min_index)
                time_cut.append(entry)
                time_cut_index.append(i)
            else:
                print('Echem and Raman measurements are too far apart in the '+str(i)+'th Raman linescan: '+ str(round(time_dif,2))+'h')
                continue
        return(time_cut_index,min_index_lst)

def calc_X(eva_class,list_del = [], error_thresh = 1, time_dif_value = 1, max_meas_ix = False):
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
    if hasattr(eva_class,'eva_good'):
        eva = eva_class.eva_good
    else:
        eva = eva_class.eva_cutZ
    
    T = eva_class.temp
    t = eva_class.t_0
    t_err = eva_class.t_0_err
    eta_c = eva_class.eta_c
    c_ini = eva_class.c_ini
    z_max = eva_class.z_valuesCut[-1]
    # a_list, b_list, a_list_err, b_list_err = eva_class.a_list, eva_class.b_list, eva_class.a_list_err, eva_class.b_list_err
 
    F=96485.3321      #unit:s A / mol
    R=8.314           #J/(mol·K）
    t_linescan, t_CP_PEIS = find_time_match(eva_class,time_dif_value)
    a_list, b_list, a_list_err, b_list_err = [eva_class.a_list[i] for i in t_linescan],[eva_class.b_list[i] for i in t_linescan], [eva_class.a_list_err[i] for i in t_linescan], [eva_class.b_list_err[i] for i in t_linescan]
    eta_c = eta_c[t_CP_PEIS]
       
    # get 【x-axis】 array: which is ln(Cs,z=L /Cs,z=0)
    x_axis=[0] # The first data is meaningless, set to 0
    x_axis_std=[0] # The first data is meaningless, set to 0
    
    if max_meas_ix != False:
        a_list, b_list, a_list_err, b_list_err = a_list[:max_meas_ix], b_list[:max_meas_ix], a_list_err[:max_meas_ix], b_list_err[:max_meas_ix]
        eta_c = eta_c[t_CP_PEIS][:max_meas_ix]        

    def func(x, a, b):
        return  (a)/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(a/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))+c_ini
    
    def calculate_standard_deviation(x, a, b, aerr, berr):
        # Computes the partial derivatives of func(x, a, b) with respect to the parameters a and b
        df_da = 1/1000*((b/(3.1415)**0.5)*np.exp(-((((-x+z_max)/1e6)/b)**2))-((((-x+z_max)/1e6))*special.erfc(((-x+z_max)/1e6)/b)))-(1/1000)*((b/(3.1415)**0.5)*np.exp(-(((((x)/1e6))/b)**2))-((((x)/1e6))*special.erfc(((x)/1e6)/b)))
        df_db = (a / 1000) * ((1 / (3.1415 ** 0.5)) * np.exp(-((((x - z_max) / 1e6) / b) ** 2)) -np.exp(-((((x) / 1e6) / b) ** 2)))
        
        # Calculate standard deviation
        std_dev = np.sqrt((df_da * aerr) ** 2 + (df_db * berr) ** 2)
        return std_dev

    for i in range(1,len(a_list)):
        func_z_final = func(z_max,a_list[i],b_list[i])
        func_z_0 = func(0,a_list[i],b_list[i])

        print(i,func_z_final,func_z_0)
        # if func_z_final or func_z_0 <= 0:
        #     func_z_0, func_z_final = np.nan, np.nan
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
    
    x_outlier, y_outlier = [i for i, value in enumerate(x_axis_std) if value > error_thresh], [i for i, value in enumerate(y_axis_std) if value > error_thresh]
    #x_outlier, y_outlier = detect_outlier(x_axis_std), detect_outlier(y_axis_std)
    print('Possible outliers detected where error exceeds '+str(error_thresh)+' for x_error: '+str(x_outlier)+' and y_error: '+str(y_outlier))
    #plt.savefig(p10,transparent = True)   
    eva_class.X = K
    eva_class.X_err = x_std[0] 
    print('X = '+str(K)+" ± "+str(x_std[0]))
    return

def calc_cond(eva_class, list_del = [], R_OCV = [], z_cut = True):
    echem = eva_class.echem
    if z_cut == True: 
        z_max = np.array(eva_class.z_valuesCut).max()
    else:
        z_max = np.array(eva_class.z_values).max()
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

# Export of eva_class as JSON file
import json
import pandas as pd
import numpy as np

def convert_to_json_serializable(value):
    """Converts a value to a JSON serializable format."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, pd.DataFrame):
        return value.to_dict(orient='records')
    elif isinstance(value, pd.Series):
        # Convert Pandas Series to a dictionary with 'values' key
        return {'values': value.values.tolist()}
    elif isinstance(value, dict):
        return {k: convert_to_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [convert_to_json_serializable(item) for item in value]
    else:
        return value

def convert_dict_to_json_serializable(input_dict):
    """Converts a dictionary to a JSON serializable format."""
    return {k: convert_to_json_serializable(v) for k, v in input_dict.items()}

def convert_eva_extref_serializable(eva_extref):
    '''
    Insert eva_extref or eva_extref_noBL and convert into a serializable object to export as json.
    '''
    for entry in eva_extref:
        eva_dict = {}
        # activate this to convert the raman data into serializable objects. Files will be 250MB per set of data frames respecitvly.
        # for z in eva_extref[entry][0]:
        #     eva_dict[z] = eva_extref[entry][0][z].to_dict()
        eva_extref[entry] = (eva_dict,list(eva_extref[entry][1]),eva_extref[entry][2],eva_extref[entry][3],eva_extref[entry][4],eva_extref[entry][5],eva_extref[entry][6],eva_extref[entry][7])
    return(eva_extref)

def save_evaclass_LFO(self, name_add = ''):
    """ Exports a model to JSON

    Parameters
    ----------
    filepath: str
        Destination for exporting model object
    It converts all the entries into JSON serizable objeccts. It struggles with the echem dictinoiary so will leave that out for the moment :(
    """
    eva_class = copy.deepcopy(self)
    members = [attr for attr in dir(eva_class) if not callable(getattr(eva_class, attr)) and not attr.startswith("__")]
   
    exp_name = eva_class.p_raman.split('\\')[-1][:-1]
    pathway = eva_class.p_out+'\\'+exp_name+str(name_add)+'.json'
    eva_dict = {}
    non_serizable = ['echem']
    need_convert = ['eva_extref_noBL', 'eva_extref']
    for entry in members:
        if entry in non_serizable:
            print(str(entry)+' cannot be converted to serizable object and thus exported to JSON.')
            continue
        elif entry in need_convert:
            attr_value = getattr(eva_class, entry)
            print('Converting to serializable object: '+entry)
            eva_dict[entry] = convert_eva_extref_serializable(attr_value)
        else:
            attr_value = getattr(eva_class, entry)
            eva_dict[entry] = convert_to_json_serializable(attr_value)
    
    with open(pathway, 'w') as f:
        json.dump(eva_dict, f)
    print(exp_name+' was exported as a eva_class json object to '+str(pathway))           
    return()

def load_evaclass_JSON(pathway):
    '''
    Open eva_class json file from concentration gradient analysis.  
    '''
    eva = {}
    with open(pathway, 'r') as json_file:
        eva = json.load(json_file)
    return(eva)           