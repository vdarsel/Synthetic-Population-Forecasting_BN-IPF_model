import os
import numpy as np
from utils.data import load_data, load_info

from Models.encoding import encoding_coefficient_log, encoding_coefficient_no_embedding, encoding_coefficient_tanh
from Models.encoding import decoding_coefficient_log, decoding_coefficient_no_embedding, decoding_coefficient_tanh

from Models.Projected_Bayesian_Network_hill_climbing import synthetic_population_from_projected_BN_hill
from Models.Projective_Iterative_Proportionnal_Fitting import update_population_with_projective_IPF

import pandas as pd

def projection_process(args):
    if args.encoding == "tanh":
        encoding_coefficient = encoding_coefficient_tanh
        decoding_coefficient = decoding_coefficient_tanh
    elif args.encoding == "None":
        encoding_coefficient = encoding_coefficient_no_embedding
        decoding_coefficient = decoding_coefficient_no_embedding
    elif args.encoding == "log":
        encoding_coefficient = encoding_coefficient_log
        decoding_coefficient = decoding_coefficient_log
    else:
        raise ValueError(f"Unknown encoding: {args.encoding}")
    
    term="_Projection"
    if args.IPF_pre:
        term+="_IPF"
    if args.BN:
        term+="_Bayesian_Network_Hill_Climb"
    if args.IPF_post:
        term+="_IPF"
    
    
    year_prediction = args.year_prediction
    
    year_validation = year_prediction-args.time_horizon
    years_training = np.arange(year_validation-args.n_years+1,year_validation+1, dtype=int)
    years_training_validation  = np.arange(year_validation-args.n_years-args.time_horizon+1,year_validation-args.time_horizon+1, dtype=int)
    
    x_training_values = np.arange(-args.n_years-args.time_horizon+1,-args.time_horizon+1)
    x_target = 0
    
    term+=f"_{args.encoding}_from_{years_training_validation[0]}_Validation_{year_validation}"
    
    
    
    datapath = args.datapath
    dataname_base = args.dataname
    filename = args.filename_training
    filename_test = args.filename_test
    infoname = args.infoname
    sample_folder = args.sample_folder
    attr_setname = args.attributes_setname
    n_sample = args.n_generation
    filename_sampling = (args.sampling_terminaison+"_"+str(n_sample)+term+".").join(filename.split('.'))
    
    folder_sampling = f'{sample_folder}/{args.size_data_str}/{args.folder_save.replace("TTTT",str(args.year_prediction))+term}'
    sampling_file = f'{folder_sampling}/{filename_sampling}'
    if not os.path.isdir(folder_sampling):
        os.makedirs(folder_sampling)
    
    n_alphas = args.n_alpha
    alpha_min = args.alpha_min
    alpha_max = args.alpha_max
    
    n_min = args.n_min
    n_max = args.n_max
    
        
    print("Data Loading...")
    
    datanames_training = [dataname_base.replace("XXXX",str(year)) for year in range(years_training[0],years_training[-1]+1)]
    datanames_training_validation = [dataname_base.replace("XXXX",str(year)) for year in range(years_training_validation[0],years_training_validation[-1]+1)]
    dataname_validation = dataname_base.replace("XXXX",str(year_validation))
    dataname_prediction = dataname_base.replace("XXXX",str(year_prediction))
    
    info = load_info(datapath, infoname, attr_setname)
    
    variables = info["Variable_name"]
    
    dtype_dict = {
        **{col: str for col in info[info["Type"].isin(["category","binary"])]["Variable_name"].to_numpy()},
        **{col: float for col in info[info["Type"].isin(["float"])]["Variable_name"].to_numpy()},
        **{col: 'Int64' for col in info[info["Type"].isin(["int"])]["Variable_name"].to_numpy()}
    }
    
    data_validation = load_data(dataname_validation, datapath, filename, variables, dtype_dict)
    training_data_validation = load_data(datanames_training_validation, datapath, filename, variables, dtype_dict)
    training_data = load_data(datanames_training, datapath, filename, variables, dtype_dict)

    print("Data Loaded.")


    print("Computing unique dictionnary...")
    # compute all modalities in the datasets
    data_concat = pd.concat(training_data_validation+training_data+[data_validation])
    dict_unique_concat = {}

    for col in data_concat.columns:
        dict_unique_concat[col] = np.sort(data_concat[col].astype(str).unique())
        
        
    print("Done")

    if args.IPF_pre:
        print("\n\nIPF premodel...")
        datas = training_data+training_data_validation
        datas = update_population_with_projective_IPF(datas, training_data,
                                                      training_data_validation, data_validation,
                                                      n_alphas, alpha_min, alpha_max,
                                                      n_min, n_max, x_training_values, x_target,
                                                      encoding_coefficient, decoding_coefficient,
                                                      dict_unique_concat, info, n_sample)
        training_data = datas[:len(training_data)]
        training_data_validation = datas[len(training_data):]
        print("End step\n")
    if args.BN:
        print("\n\nProjected BN...")
        synthetic_population = synthetic_population_from_projected_BN_hill(training_data, training_data_validation,
                          data_validation, n_alphas,
                          alpha_min, alpha_max,
                          n_min, n_max,
                          x_training_values, x_target,
                          encoding_coefficient, decoding_coefficient,
                          folder_sampling, n_sample,
                          dict_unique_concat)
        print("End step\n")
    else:
        synthetic_population = data_validation.copy()
        
    if args.IPF_post:
        print("\n\nIPF postsampling...")
        synthetic_population = update_population_with_projective_IPF([synthetic_population], training_data,
                                                      training_data_validation, data_validation,
                                                      n_alphas, alpha_min, alpha_max,
                                                      n_min, n_max, x_training_values, x_target,
                                                      encoding_coefficient, decoding_coefficient,
                                                      dict_unique_concat, info, n_sample)[0]
    
        print("End step\n")
    print("Saving population...")
    synthetic_population.to_csv(sampling_file, sep=";", index=False)
    print("Population saved")
    