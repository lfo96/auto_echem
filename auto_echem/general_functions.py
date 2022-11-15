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

from scipy.stats import linregress
from stat import ST_CTIME
from matplotlib import cm
from IPython.display import set_matplotlib_formats
from itertools import islice
from galvani import BioLogic as BL
#from galvaniV021 import Biologic as BL
from contextlib import contextmanager

colors = ['#FF1F5B','#00CD6C','#009ADE','#AF58BA','#FFC61E','#F28522','#A0B1BA','#A6761D']
coolors_1 = ['#052F5F','#005377','#06A77D','#D5C67A','#F1A208']
#colors_II = ['#A0B1BA','#A6761D']


def datetime(time_raw):
    '''
    Transforms a time format of h:m:s into hours as float
    '''
    
    start_dt = dt.datetime.strptime("00:00:00", '%H:%M:%S')
    time = time_raw.split('.')[0]
    time_float = float('{:0.3f}'.format((dt.datetime.strptime(time, '%H:%M:%S') - start_dt).seconds/3600))
    
    return(time_float)



def find_nearest(array, value):
    '''
    Insert list or array of numbers and returns the position and value of the closest number to the given value.
    '''
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return (idx,array[idx])


def isclose(a,
            b,
            rel_tol=1e-9,
            abs_tol=0.0,
            method='weak'):
    """
    returns True if a is close in value to b. False otherwise
    :param a: one of the values to be tested
    :param b: the other value to be tested
    :param rel_tol=1e-8: The relative tolerance -- the amount of error
                         allowed, relative to the magnitude of the input
                         values.
    :param abs_tol=0.0: The minimum absolute tolerance level -- useful for
                        comparisons to zero.
    :param method: The method to use. options are:
                  "asymmetric" : the b value is used for scaling the tolerance
                  "strong" : The tolerance is scaled by the smaller of
                             the two values
                  "weak" : The tolerance is scaled by the larger of
                           the two values
                  "average" : The tolerance is scaled by the average of
                              the two values.
    NOTES:
    -inf, inf and NaN behave similar to the IEEE 754 standard. That
    -is, NaN is not close to anything, even itself. inf and -inf are
    -only close to themselves.
    Complex values are compared based on their absolute value.
    The function can be used with Decimal types, if the tolerance(s) are
    specified as Decimals::
      isclose(a, b, rel_tol=Decimal('1e-9'))
    See PEP-0485 for a detailed description
    """
    if method not in ("asymmetric", "strong", "weak", "average"):
        raise ValueError('method must be one of: "asymmetric",'
                         ' "strong", "weak", "average"')

    if rel_tol < 0.0 or abs_tol < 0.0:
        raise ValueError('error tolerances must be non-negative')

    if a == b:  # short-circuit exact equality
        return True
    # use cmath so it will work with complex or float
    if cmath.isinf(a) or cmath.isinf(b):
        # This includes the case of two infinities of opposite sign, or
        # one infinity and one finite number. Two infinities of opposite sign
        # would otherwise have an infinite relative tolerance.
        return False
    diff = abs(b - a)
    if method == "asymmetric":
        return (diff <= abs(rel_tol * b)) or (diff <= abs_tol)
    elif method == "strong":
        return (((diff <= abs(rel_tol * b)) and
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "weak":
        return (((diff <= abs(rel_tol * b)) or
                 (diff <= abs(rel_tol * a))) or
                (diff <= abs_tol))
    elif method == "average":
        return ((diff <= abs(rel_tol * (a + b) / 2) or
                (diff <= abs_tol)))
    else:
        raise ValueError('method must be one of:'
                         ' "asymmetric", "strong", "weak", "average"')


def layout(ax,x_label = "", y_label = "", title = "",  x_lim = "", y_lim = "", square = "", size = []):
    """
    Update plot layout to a standard format.
    Insert ax from matplotlib, x_label, and y_label description and a title.
    """
    
    ax.set_title(title,fontsize = 20)
    ax.set_xlabel(x_label,fontsize = 16)
    ax.set_ylabel(y_label,fontsize = 16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best",frameon = False, fontsize = 12)
    ax.tick_params(direction='in', length=6, width=1.5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
        
    if x_lim != "":
        plt.xlim(x_lim)
    if y_lim != "":
        plt.ylim(y_lim)
    if square != "":
        plt.gca().set_aspect('equal', adjustable='box')
    if len(size)!=0:
        try:
            fig = plt.gcf()
            fig.set_size_inches(size[0], size[1])
        except IndexError:
            print('Please specify size of the figure in inches size = [lenght, height].')
            pass

    return(ax)

def layout_zoom(ax,x_axis = True, y_axis = True, x_label = "", y_label = "", title = "",  x_lim = "", y_lim = "", square = ""):
    """
    Update plot layout to a standard format.
    Insert ax from matplotlib, x_label, and y_label description and a title.
    """
    
    ax.set_title(title,fontsize = 40)
    ax.set_xlabel(x_label,fontsize = 32)
    ax.set_ylabel(y_label,fontsize = 32)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    #plt.legend(loc="best",frameon = False, fontsize = 24)
    ax.tick_params(direction='in', length=12, width=3)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(3)
        
    if x_lim != "":
        plt.xlim(x_lim)
    if y_lim != "":
        plt.ylim(y_lim)
    if square != "":
        plt.gca().set_aspect('equal', adjustable='box')
    if y_axis == False:
        figure = plt.gca()
        y_axis = figure.axes.get_yaxis()
        y_axis.set_visible(False)
    if x_axis == False:
        figure = plt.gca()
        x_axis = figure.axes.get_xaxis()
        x_axis.set_visible(False)


    return(ax)


def manual_header(filename):
    with open(filename,encoding= 'ISO-8859-1') as fin:
        counter = 0
        for line in fin:
            try:
                float(line.split(" ")[0].split("\t")[0])
                break
            except ValueError:
                counter += 1 
    return(counter)



def data_set(pathway):
    '''
    Input folder pathway and return a tuple with the name, time, size, and last modified timing of each file in the folder.
    '''
    nameSet=set()
    for file in os.listdir(pathway):
        fullpath=os.path.join(pathway, file)
        if os.path.isfile(fullpath):
            nameSet.add(file)

    retrievedSet=set()
    for name in nameSet:
        stat = os.stat(os.path.join(pathway, name))
        time = ST_CTIME
        size = stat.st_size 
        modified = stat.st_mtime #Also consider using ST_MTIME to detect last time modified
        retrievedSet.add((name,time,size,modified))

    return(retrievedSet)

def mpr_files(info):
    '''
    Search in the pathway for all corresponding .mpr files. 
    '''
    files = []
    for file in data_set(info["path"]):
        if info["filename"] in file[0] and file[0][-4:]==".mpr":
            files.append(file[0])
    return(files)

def meta(filename):
        #obtain number of header lines in the text file
    with open(filename,encoding= 'ISO-8859-1') as fin:
        head_len = manual_header(filename)   
        t = 0
        techniques = []
        m_am = A_el = cd = t_wait= np.nan
    # head lenght in the beginning of file does not count itself and thus needs to be shifted for 4 rows.
        for line in islice(fin, 0,head_len):
            #print(line.split(":")[0])
            if line.split(":")[0]=="Mass of active material ": #extract the active material mass
                m_am = float(line.split(":")[1].split(" ")[1]) 
            elif line.split(":")[0]=="Electrode surface area ": #extract the electrode surface area
                A_el = float(line.split(":")[1].split(" ")[1]) 
                if A_el == 0.001:
                    print('Electrode Surface Area manually set to 1.13cm2.')
                    A_el = 1.13097
            elif line.split(":")[0]=="Cycle Definition ": #extract the electrode surface area
                cd = line.split(":")[1].split(" ")[1] 
            elif t == 1:  
                technique = line.split("\n")[0]
                if technique == 'Modulo Bat':
                    t=0
                    print('Modulo Bat found.')
                    continue
                try:
                    techniques.append(tech[technique])
                    t = 0
                except KeyError:
                    print('Unknown technique found: '+str(technique))
                    t = 0
            elif line.split(":")[0]=="Technique ":
                t = 1
            elif line.split(":")[0]== 'td (h':
                if np.isnan(t_wait) == False:
                    print('Several waiting times found.')
                    t_wait_2 = datetime(line.split(" ")[11])
                t_wait = datetime(line.split(" ")[11])
            elif line.split(":")[0]== 'tR (h':
                # GCPL and OCV both have a tR entry in the meta file. Necessary to distinguish both
                if techniques[-1] == 'GCPL':
                    t_GCPL = []
                    for entry in line.split('        '):
                        try:
                            t_GCPL.append(datetime(entry.strip()))
                        except ValueError:
                            pass
                    
                else:
                    if np.isnan(t_wait) == False:
                        print('Several waiting times found.')
                        t_wait_2 = datetime(line.split(" ")[11])
                    t_wait = datetime(line.split(" ")[11])
            elif line[0:9]=='ctrl_type':
                # Add all the techniques from the MB setting to techniques list.
                MB_tech = line[9:].split(' ')
                for t_i in MB_tech[:-2]:
                    if t_i != '':
                        techniques.append(t_i)

    #create a dictionary with all the important meta data
    meta_data = {
        'header': head_len,
        'active material mass': m_am,
        'electrode surface area': A_el,
        'protocol': techniques,
        'cd' : cd,
        'waiting time' : t_wait
    }
    try:
         meta_data['waiting time 2'] = t_wait_2
    except NameError:
        pass  
    try:
         meta_data['t_GCPL'] = t_GCPL
    except NameError:
        pass             

    return(meta_data)

def read_in(pathway):
    '''
    Obtain all required meta data from an .mps file. 
    '''
    if pathway[-4:] == '.mps':
        print("Correct file selected.")
    else: 
        print("Please select the .mps setting file.")
    path = pathway.replace(pathway.split('\\')[-1],'')
    name = pathway.split('\\')[-1][0:-4]
    #extract meta data
    meta_data = meta(pathway)
    meta_data['filename'] = name
    meta_data['path'] = path
    return(meta_data)

def info(pathway):
    info = read_in(pathway)
    info['MB'] = False
    files = mpr_files(info)
    info['data'] = {}
    if len(info['protocol']) == 1:
        #if only one technique is used, the data file has no technique specification in the file name.
        if len(files) == 1:
            mpr_file = BL.MPRfile(info['path']+files[0])
            df = pd.DataFrame(mpr_file.data)
            technique = info['protocol'][0]
            info['data']['1 '+technique]= df
            print(technique+" data file found.")
        else:
            print("Error: Inconsistency in setting file and amount of generated .mpr files!")
            
    else:
        counter = 1
        if len(files) == 1:
            MB_df_mpr = BL.MPRfile(info['path']+files[0])
            MB_df = pd.DataFrame(MB_df_mpr.data)
                # One .mpr file while a big list of techniques indicates a modulo bat.
            for NS_i,technique in enumerate(info["protocol"]):
                info['data'][str(NS_i+1)+' '+str(technique)] = MB_df.loc[MB_df['Ns']==NS_i]
                info['MB'] = True
        else:     
            for technique in info["protocol"]:  
                if technique == 'GCPL':
                    for file in files:
                        if 'GCPL' in file:
                            number = int(file.split('_GCPL')[0].split('_')[-1])
                            if counter == number:
                                counter += 1
                                mpr_file = BL.MPRfile(info['path']+file)
                                df_GCPL = pd.DataFrame(mpr_file.data)

                                if len(df_GCPL) != 0:
                                    info['data'][str(number)+' GCPL'] = df_GCPL
                                    print(str(number)+' GCPL data file added.')
                                else:
                                    print(str(number)+' GCPL data file is empty and therefore disregarded.')
                            
                elif technique == 'PEIS':
                    for file in files:
                        if 'PEIS' in file:
                            number = int(file.split('_PEIS')[0].split('_')[-1])
                            if counter == number:
                                counter += 1
                                mpr_file = BL.MPRfile(info['path']+file)
                                df_PEIS = pd.DataFrame(mpr_file.data)

                                if len(df_PEIS) != 0:
                                    info['data'][str(number)+' PEIS'] = df_PEIS
                                    print(str(number)+' PEIS data file added.')
                                else:
                                    print(str(number)+' PEIS data file is empty and therefore disregarded.')

                elif technique == 'OCV':
                    for file in files:
                        if 'OCV' in file:
                            number = int(file.split('_OCV')[0].split('_')[-1])
                            if counter == number:
                                try: 
                                    mpr_file = BL.MPRfile(info['path']+file)
                                    df_OCV = pd.DataFrame(mpr_file.data)
                                except NotImplementedError:
                                    print('Not Implemented Error found in OCV file.')
                                    counter +=1
                                    break
                                                                
                                if len(df_OCV) != 0:
                                    info['data'][str(number)+' OCV'] = df_OCV
                                    print(str(number)+' OCV data file added.')
                                else:
                                    print(str(number)+' OCV data file is empty and therefore disregarded.')
                
                                counter += 1
                                
                elif technique == 'CP':
                    for file in files:
                        if 'CP' in file:
                            number = int(file.split('_CP')[0].split('_')[-1])
                            if counter == number:
                                try: 
                                    mpr_file = BL.MPRfile(info['path']+file)
                                    df_CP = pd.DataFrame(mpr_file.data)
                                except NotImplementedError:
                                    print('Not Implemented Error found in OCV file.')
                                    counter +=1
                                    break
                                                                
                                if len(df_CP) != 0:
                                    info['data'][str(number)+' CP'] = df_CP
                                    print(str(number)+' CP data file added.')
                                else:
                                    print(str(number)+' CP data file is empty and therefore disregarded.')
                
                                counter += 1

                elif technique == 'GEIS':
                    for file in files:
                        if 'GEIS' in file:
                            number = int(file.split('_GEIS')[0].split('_')[-1])
                            if counter == number:
                                counter += 1
                                mpr_file = BL.MPRfile(info['path']+file)
                                df_GEIS = pd.DataFrame(mpr_file.data)

                                if len(df_GEIS) != 0:
                                    info['data'][str(number)+' GEIS'] = df_GEIS
                                    print(str(number)+' GEIS data file added.')
                                else:
                                    print(str(number)+' GEIS data file is empty and therefore disregarded.')
                else:
                    counter += 1
    return(info)


def color_gradient(increments, style = cm.viridis):
    cm_subsection = np.linspace(0, 1, increments)
    colors = [ style(x) for x in cm_subsection ]
    return (colors)     

def quali_fit(parameter,confidence):
    quality = 0
    for p,c in zip(parameter,confidence):
        quali = c/p
        quality = quality + quali
    return(quality)

#Time out manager
# just put a with time_limit(15, 'sleep'): in front of the function which should time out.

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

tech = {
    'Galvanostatic Cycling with Potential Limitation': 'GCPL',
    'Potentio Electrochemical Impedance Spectroscopy': 'PEIS',
    'Open Circuit Voltage' : 'OCV',
    'Loop': 'l',
    'Wait' : 'w',
    'Linear Sweep Voltammetry' : 'LSV',
    'Galvano Electrochemical Impedance Spectroscopy' : 'GEIS',
    'Chronopotentiometry' : 'CP'
    }

def LSV_cond(raw,d=0.14,r_cc = 5):
    '''
    Enter the raw LSV data. Also specify the dimensions in mm of d thickness of sample and r_cc radius of the current collector/electrode for surface area calculation. Returns the conductivity in S/m.
    '''
    A = (r_cc*0.001)*(r_cc*0.001)*3.141592 #m2
    l = d*0.001 #m
    E = raw['Ewe/V'].loc[raw['I/mA']!=0]
    I = raw['I/mA'].loc[raw['I/mA']!=0]*0.001
    fit = linregress(E,I) #x and y are arrays or lists
    if fit.rvalue <= 0.999:
        print('Bad linear fit. Please double check the regression.')
    R = 1/fit.slope # Ohm
    cond = (1/R)*l/A
    print('LSV Conductivity: '+str(round(cond,3))+' S/m')
    return(cond,[E,I])

def change_index(list):
    '''
    Insert list of numbers. Determine if an approximation of this number is repeated or changing. Tracks the index where the number has changed.
    '''
    index = []
    a = 0
    for i,entry in enumerate(list):
        entry_approx = round(entry,3)
        #entry_approx = round(entry,2)
        if a != entry_approx:
            index.append(i)
        else:
            pass
        a = entry_approx
    return(index)

def outliers(lst,cor_fac):
    '''
    Insert list and correction factor. Returns the list with np.nan on outlier that do not lie within multiples of the mean times correction facotor
    '''
    correction = abs(lst-np.nanmedian(lst)) <= np.nanstd(lst)*cor_fac
    i=0
    for counter,value in enumerate(correction):
        if value == False:
            lst[counter] = np.nan
            i += 1
    if i>=1:
        print(str(i)+' values were set to np.nan!')
    return(lst)

def formation_merge(df_1, df_subs):
    '''
    Insert the auto evaluated formation cycle and the auto evaluated subsequent cycles to merge them and return one dictionary with all the data and evaluation frames. 
    '''
    for entry in df_1['data']:
        # add the data
        df_subs['data']['formation '+entry] = df_1['data'][entry]
        print('Formation data of '+str(entry)+' added.')

        if entry.split(' ')[1] == 'GCPL':
            for entry_subs in df_subs['eva']:
                if entry_subs.split(' ')[1] == 'GCPL':
                    # merge the eva files.
                    #print(entry,entry_subs)
                    eva_lst = df_1['eva'][entry][1]+df_subs['eva'][entry_subs][1]

                    df_subs['eva'][entry_subs][0]['Cycle'] = df_subs['eva'][entry_subs][0]['Cycle']+len(df_1['eva'][entry][0])
                    eva_pd = pd.concat([df_1['eva'][entry][0],df_subs['eva'][entry_subs][0]], ignore_index=True)
                    
                    df_subs['eva'][entry_subs] = [eva_pd,eva_lst]
                    print('Formation '+str(entry)+ ' and subsequent '+str(entry_subs)+' evaluation pandas merged.')
    return(df_subs)

def loop_finder(lst):
    '''
    Finds the loop sequence by finding a non consecutive number in the list. If there is none, returns value 1. 
    '''
    for i,j in enumerate(lst,lst[0]):
        if i!=j:
            return (j)
    return(1)


def calc_I(t,Q):
    '''
    Calculate the current based on charge (mAh) and time stemp Returns current in mA.
    '''
    Q_0 = 0
    t_0 = 0
    I = []
    for i,Q_i in enumerate(Q):
        t_i = ((t-t.iloc[0])/3600).iloc[i]
        I.append((Q_i-Q_0)/(t_i-t_0))
        Q_0,t_0 = Q_i,t_i
    return(I)


# Read in .fra files
def fra_headlen(pathway):
    data_index = []
    data_name = []
    with open(pathway,encoding= 'ISO-8859-1') as fin:
    #with open(pathway) as fin:
        for i,line in enumerate(fin):
            #print(line)
            if line.split('=')[0] == 'Serie_name':
                data_index.append(i-2-(len(data_name))) # weird data index number so it works out. not sure why I need thius corerction factor of -2 and len(data)
                data_name.append(line.split('=')[1])
    return(data_index,data_name)

def eva_FRA(pathway):
    '''
    Readin fra files and created dictionary with all sub data sets. 
    '''
    d = {}
    head = fra_headlen(pathway)
    for i,data_index in enumerate(head[0]):
        try:
            dat_len = head[0][i+1] - head[0][i] -2 # correction for lines in between
            df = pd.read_csv(pathway,encoding= 'unicode_escape', header=data_index,nrows=dat_len, delimiter=';', names=['freq(Hz) raw','bias(V)','temp(degC)','Re(Ohm)','Im(Ohm)','current(A)'],usecols=[0,1,2,3,4,5])
        except IndexError:
            df = pd.read_csv(pathway,encoding= 'unicode_escape', header=data_index, delimiter=';', names=['freq(Hz) raw','bias(V)','temp(degC)','Re(Ohm)','Im(Ohm)','current(A)'],usecols=[0,1,2,3,4,5])
        
        freq= []
        for entry in range(len(df)):
            freq.append(float(df['freq(Hz) raw'][entry].split('=')[1]))
        df['freq(Hz)'] = freq
        
        name = head[1][i].split(':')[0].split(' ')[0]
        d[name] = df
    return(d)

def PEIStoTXT(Nyquist_data,filename,pathway):
    '''
    Insert Nyquist data with the respective cycle, and export to the given filename and pathway as .txt.
    '''
    d = {}
    d['freq(Hz)'] = Nyquist_data[0]
    d['Re(Ohm)'] = Nyquist_data[1]
    d['Im(Ohm)'] = -Nyquist_data[2]
    df = pd.DataFrame(d)
    df.to_csv(pathway+'\\'+filename+".txt",
        sep=' ',
        columns=['freq(Hz)','Re(Ohm)','Im(Ohm)'],
        index=False)
    print('Successfully exported: '+str(pathway+'\\'+filename+".txt"))
        

from scipy import sparse
from scipy.sparse.linalg import spsolve
'''
There are two parameters: p for asymmetry and λ for smoothness. Both have to be tuned to the data at hand. We found that generally 0.001 ≤ p ≤ 0.1 is a good choice (for a signal with positive peaks) and 102 ≤ λ ≤ 109, but exceptions may occur. In any case one should vary λ on a grid that is approximately linear for log λ. Often visual inspection is sufficient to get good parameter values.
'''

def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def eva_XRD(pathway):
    '''
    Insert pathway of .ASC file and evaluate the XRD file.
    Returns a dictionioary with the data[deg,int] and eva[normalized, baseline substracted, baseline substracted & normalized]
    '''
    d = {}
    f = open(pathway, "r")
    intensity = []
    for line in f:
        ### get meta data
        key = line.split('=')

        if key[0] == '*START\t\t':
            deg_start = float(key[1])
        if key[0] == '*STOP\t\t':
            deg_end = float(key[1])
        if key[0] == '*STEP\t\t':
            deg_step = float(key[1])


        ### Data Points
        try:
            string = line.split(',')
            float(string[0])
            for entry in string:
                intensity.append(float(entry))
        except ValueError:
            pass
    deg = np.arange(deg_start,deg_end+deg_step,deg_step)
    ### check if the amount of data points agree with the specified degree boundaries and the stepsize.
    if len(deg) != len(intensity):
        print('Something seems off with the data. Different amount of measurement points than specified in meta.')
    
    data = [deg,intensity]
    ### normalized, baseline substracted, baseline substracted & normalized
    eva = [intensity/np.amax(intensity),np.array(intensity)-intensity[-1],(np.array(intensity)-intensity[-1])/np.amax(np.array(intensity)-intensity[-1])]

    d = {
        'deg_start' : deg_start,
        'deg_end' : deg_end,
        'deg_step' : deg_step,
        'data' : data,
        'eva' : eva
    }
    return(d)  