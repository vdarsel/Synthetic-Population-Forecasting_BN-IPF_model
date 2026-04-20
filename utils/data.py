import pandas as pd

def force_type_df(dataframe, name_cols_cat, name_cols_float, name_cols_int):
    for col in name_cols_cat:
        dataframe[col] = dataframe[col].astype(str)
    for col in name_cols_float:
        dataframe[col] = dataframe[col].astype(float)
    for col in name_cols_int:
        dataframe[col] = dataframe[col].astype(int)
    dataframe = dataframe.astype(str) # sans le str, on a des scores pourris pas compris
    return dataframe

def load_data(dataname, datapath, filename, vars, type_vars):
    if(type(dataname)==list):
        return [load_data(data, datapath, filename, vars, type_vars) for data in dataname]
    else:
        file = f'{datapath}/{dataname}/{filename}'
        data = pd.read_csv(file, sep=";", low_memory=False, usecols=vars)
        data = data.astype(type_vars)
        return data
    
    
def load_info(folder, info_filename, attribute_set):
    info_path = f"{folder}/{info_filename}"
    info = pd.read_csv(info_path,sep=";")
    info = info[info[attribute_set]]
    return info