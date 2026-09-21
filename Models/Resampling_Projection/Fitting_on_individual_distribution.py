import numpy as np
import pandas as pd

from Models.Static_Methods.Iterative_Proportional_Fitting import perform_IPF
    
variables_resampling = ["Age", "Sex", "Department", "City", "County"]

def get_distributions_from_true_values(target_dataset, variables, dict_unique):
    
    dict_res = {}
    
    for var in variables:
        freq_ser = pd.Series(0.0, index = dict_unique[var])
        freq_ref = target_dataset[var].value_counts(normalize=True)
        freq_ser.loc[freq_ref.index] = freq_ref
        dict_res[var] = freq_ser.to_numpy()    
    
    return dict_res

def get_multivariate_counts_from_true_values(target_dataset, variables, dict_unique):
    
    count_serie = pd.Series(0, index = pd.MultiIndex.from_product(
            [dict_unique[col] for col in variables]
        ))
    
    count_ref = target_dataset[variables].value_counts(normalize=False)
    
    count_serie.loc[count_ref.index] = count_ref
    
    return count_serie


def resampling_using_counts(input_data, counts_ref, variables):
    grouped_data = input_data.groupby(variables)
    
    res = []
    for index, group in grouped_data:
        n_samples_in_group = counts_ref.loc[index]
                
        n_copy = int(n_samples_in_group/len(group))
        
        for _ in range(n_copy):
            res.append(group.copy())
        res.append(group.sample(n_samples_in_group - n_copy*len(group)))
    return pd.concat(res, ignore_index=True)    


def resampling_from_true_marginal_distribution(dataset, reference_dataset, info, unique_val_dict, n_samples):
    vars = info["Variable_name"].to_numpy()

    vars_selection = np.array(list(set(variables_resampling).intersection(set(vars))))
    
    dict_freq_target_variables = get_distributions_from_true_values(reference_dataset, vars_selection,unique_val_dict)
    
    dataset_adjusted_multivariate = perform_IPF(dataset[vars].copy(),unique_val_dict, dict_freq_target_variables, info[info["Variable_name"].isin(vars_selection)], n_samples)
    
    counts = get_multivariate_counts_from_true_values(dataset_adjusted_multivariate, vars_selection, unique_val_dict)
    
    return resampling_using_counts(dataset, counts, list(vars_selection))