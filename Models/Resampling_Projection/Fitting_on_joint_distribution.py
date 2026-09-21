import numpy as np
import pandas as pd
    
variables_resampling = ["Age", "Sex", "Department", "City", "County"]

def get_multivariate_distribution_from_true_values(target_dataset, info, dict_unique):
    vars = info["Variable_name"].to_numpy()

    vars_selection = np.array(list(set(variables_resampling).intersection(set(vars))))
    
    
    freq_serie = pd.Series(0.0, index = pd.MultiIndex.from_product(
            [dict_unique[col] for col in vars_selection]
        ))
    
    freq_ref = target_dataset[vars_selection].value_counts(normalize=True)
    
    freq_serie.loc[freq_ref.index] = freq_ref
    
    return freq_serie


def resampling_using_frequencies(input_data, frequencies_ref, n_samples, dict_unique):
    vars = input_data.columns.to_numpy()

    vars_selection = list(set(variables_resampling).intersection(set(vars)))

    grouped_data = input_data.groupby(vars_selection)
    
    res = []
    for index, group in grouped_data:

        n_samples_in_group = frequencies_ref.loc[index]*n_samples
        
        n_samples_in_group_int = int(np.floor(n_samples_in_group))
        rest = (n_samples_in_group-n_samples_in_group_int)
        n_samples_in_group = (n_samples_in_group_int + np.random.binomial(1,rest))
        
        n_copy = int(n_samples_in_group/len(group))
        
        for _ in range(n_copy):
            res.append(group.copy())
        res.append(group.sample(n_samples_in_group - n_copy*len(group)))

    return pd.concat(res, ignore_index=True)    


def resampling_from_true_joint_distribution(dataset, reference_dataset, info, unique_val_dict, n_samples):
    mutlivariate_freq_target = get_multivariate_distribution_from_true_values(reference_dataset,info,unique_val_dict)
    return resampling_using_frequencies(dataset,mutlivariate_freq_target,n_samples, unique_val_dict)