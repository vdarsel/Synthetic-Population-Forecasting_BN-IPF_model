import numpy as np
    
from Models.utils_projection import get_coefs_from_regr, get_frequencies_from_df

from Models.Static_Methods.Iterative_Proportional_Fitting import perform_IPF           

def get_parameters_IPF_projection(datas_training, datas_validation, data_validation, info, 
                   projection_x_values, target_x, unique_val_dict, alphas, n_min, n_max, encoding_coefficient, decoding_coefficient):
    vars = info["Variable_name"].to_numpy()
    
    best_alphas = []
    best_ns = []
    best_SRMSE = []

    for var in vars:
        Y_preencoding = np.array([get_frequencies_from_df(data.astype(str), var, unique_val_dict[var]) for data in datas_validation])
        Y = encoding_coefficient(Y_preencoding, Y_preencoding[-1])
        Y_target = get_frequencies_from_df(data_validation.astype(str), var, unique_val_dict[var])
        best_error = 1e5
        for alpha in (alphas):
            for n in range(n_min,n_max+1):
                Y_pred_encoded = get_coefs_from_regr(projection_x_values, Y, target_x, alpha, n)
                Y_pred = decoding_coefficient(Y_pred_encoded, Y_preencoding[-1])
                err = np.sum(np.power(Y_pred-Y_target,2))

                if err<best_error:
                    best_error = err
                    best_n = n
                    best_alpha = alpha
        best_alphas.append(best_alpha)
        best_ns.append(best_n)
        best_SRMSE.append(np.sqrt(best_error*len(Y)))
        
    
    dict_proportions = {}
    for i in range(len(vars)):
        var = vars[i]
        n = best_ns[i]
        alpha = best_alphas[i]
        Y_preencoding = np.array([get_frequencies_from_df(data.astype(str), var, unique_val_dict[var]) for data in datas_training])
        Y = encoding_coefficient(Y_preencoding, Y_preencoding[-1])
        Y_pred = decoding_coefficient(get_coefs_from_regr(projection_x_values, Y, 0, alpha, n), Y_preencoding[-1])
        dict_proportions[var] = Y_pred
    return dict_proportions


def update_population_with_projective_IPF(populations_to_update,
                                          training_data, training_data_validation, 
                      validation_data, n_alphas,
                      alpha_min, alpha_max, 
                      n_min, n_max, 
                      training_values_x, projection_value_x,
                      encoding_function, decoding_function,
                      dict_unique, info,
                      n_sample):
        
    print("Find coefficients for IPF [VALIDATION + PROJECTION]")
    
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)
    
    dict_proportion_projection = get_parameters_IPF_projection(training_data, training_data_validation, 
                                  validation_data, info, 
                                  training_values_x, projection_value_x,
                                  dict_unique, alphas,
                                  n_min, n_max, encoding_function, decoding_function)
    
    print("Coefficients computed")
    
    res = []
    
    print("Performing IPF...")
    for population in populations_to_update:
        res.append(perform_IPF(population,dict_unique, dict_proportion_projection, info, n_sample).reset_index())
    print("IPF done")
    
    return res
        

