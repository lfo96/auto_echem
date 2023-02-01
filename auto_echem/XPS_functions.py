import pandas as pd
import numpy as np
from scipy.integrate import simps

def eva_XPS(pathway):
    df = pd.read_csv(pathway, sep='\t', header = 5)
    dict = {}
    spectrum_index = []
    for i,entry in enumerate(df):
        # extract index where a new spectrum starts.
        if 'B.E.' in entry:
            spectrum_index.append(i)

    data = []
    # create new DF for each spectrum
    for index_i in range(len(spectrum_index)):
        if index_i != len(spectrum_index)-1:
            df_spec = df[df.columns[spectrum_index[index_i]:spectrum_index[index_i+1]]]
        else:
            df_spec = df[df.columns[spectrum_index[index_i]:]]  
        data.append(df_spec)  

    dict['data'] = data

    # Evaluate the file. To be edited...
    eva = []
    for spectrum in dict['data']:
        dict_s = {}
        dict_data = {}
        areas = {}
        name = spectrum.columns[1]
        cycle = name.split(':')[0]
        #element = name.split(':')[3] more general is -2
        element = name.split(':')[-2]
        BE = spectrum[spectrum.columns[0]]
        raw = spectrum[spectrum.columns[1]]
        raw_norm =  spectrum[spectrum.columns[1]]/spectrum[spectrum.columns[1]].max()
        
        dict_data['Binding Energy (eV)'] = BE
        dict_data['Counts (1/s)'] = raw
        dict_data['Counts norm'] = raw_norm
        curves = []
        #add remaining coloumns which will appear when the file is processed.
        for col in spectrum.columns[2:]:
            col_name = col.split(':')[-1]
            if np.isnan(spectrum[col]).all() == True:
                continue
            elif 'Background' in col_name:
                dict_data['Background'] = spectrum[col]
            elif 'Envelope' in col_name:
                dict_data['Envelope'] = spectrum[col]
                envelope_A = simps((dict_data['Envelope']-dict_data['Background']).dropna(),-dict_data['Binding Energy (eV)'].dropna())
                areas['Envelope'] = envelope_A
            else:
                dict_data[col_name] = spectrum[col]
                curves.append(col_name)
        
        for curve in curves:
            BL_sub = dict_data[curve] - dict_data['Background']
            curve_A = simps(BL_sub.dropna(),-dict_data['Binding Energy (eV)'].dropna())
            areas[curve] = curve_A
            
        dict_s['name'] = name
        dict_s['cycle'] = cycle
        dict_s['element'] = element
        dict_s['data'] = pd.DataFrame(dict_data)
        dict_s['area'] = areas
        

        eva.append(dict_s)
        
    dict['eva'] = eva
    
    return(dict)