import numpy as np

import pandas as pd
    
from Models.utils_projection import get_coefs_from_regr, get_frequencies_from_df

import duckdb


def perform_IPF(dataset, unique_val_dict, dict_proportions, info, n_sample):
    ##### INITIALISATION RESAMPLING        
    vars = info["Variable_name"].to_numpy()
    name_cols = ",".join(info["Variable_name"].to_list())
    # Create a temporary table from the query result
    duckdb.sql(f"""
        CREATE TEMP TABLE table_potential_indiv_grouped AS
        SELECT {name_cols}, COUNT(*) AS count
        FROM dataset
        GROUP BY {name_cols}
    """)

    # Now you can alter the temporary table
    duckdb.sql("ALTER TABLE table_potential_indiv_grouped ADD COLUMN proportion FLOAT;")

    # Update the proportion column
    duckdb.sql("""
        UPDATE table_potential_indiv_grouped
        SET proportion = count * 1.0 / (SELECT SUM(count) FROM table_potential_indiv_grouped)
    """)

    duckdb.sql("""
                ALTER TABLE table_potential_indiv_grouped
                DROP COLUMN count; 
            """)
    error = 1
    epoch = 0
    print("IPF")
    while((error>0.001)&(epoch<30)):
        epoch+=1
        vars_random = np.copy(vars)
        np.random.shuffle(vars_random)
        for var in vars_random:
            unique = unique_val_dict[var]
            proportion_target_var = dict_proportions[var]
            for unique, proportion_target in zip(unique, proportion_target_var):
                proportion_res = duckdb.sql(f"SELECT (proportion) FROM table_potential_indiv_grouped WHERE {var}='{unique}'").fetchnumpy()['proportion']
                sum_res = proportion_res.sum()
                if (sum_res>0):
                    coef = proportion_target/sum_res
                    duckdb.sql(f"UPDATE table_potential_indiv_grouped SET (proportion) = {coef}*proportion WHERE {var}='{unique}'")
        
        errors = []
        for var in vars:
            unique = unique_val_dict[var]
            proportion_target_var = dict_proportions[var]
            err = 0
            for unique, proportion_target in zip(unique, proportion_target_var):
                proportion_res = duckdb.sql(f"SELECT sum(proportion) as s FROM table_potential_indiv_grouped WHERE {var}='{unique}'").fetchnumpy()['s'][0]
                if proportion_res>0:
                    err += np.power(proportion_res-proportion_target,2)
            errors.append(np.sqrt(err*len(unique)))
        
        error = np.mean(errors)
        print(error)
    
    
    duckdb.sql("ALTER TABLE table_potential_indiv_grouped ADD COLUMN count FLOAT;")
    duckdb.sql(f"""
        UPDATE table_potential_indiv_grouped
        SET count = proportion*{n_sample}
    """)
    
    rows = duckdb.sql("SELECT * EXCLUDE(proportion, count) FROM table_potential_indiv_grouped").fetchall()
    counts = duckdb.sql("SELECT count FROM table_potential_indiv_grouped").fetchall()

        
    duckdb.sql("""
               DROP TABLE table_potential_indiv_grouped; 
        """)

    floors = np.floor(counts)
    random_decs = np.random.binomial(1, counts - floors)
    counts_final = (floors + random_decs).astype(int)
    
    if ((counts_final<0).any()):
        print("Warning negative count detected")
        counts_final = np.max([counts_final, np.zeros_like(counts_final)], 0)

    # Create a DataFrame from your rows and the final counts
    df_temp = pd.DataFrame(rows, columns=vars)
    df_temp['temp_counts'] = counts_final

    # Use repeat to expand the rows instantly
    return df_temp.loc[df_temp.index.repeat(df_temp['temp_counts'])].drop(columns='temp_counts')
            

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
        res.append(perform_IPF(population,dict_unique, dict_proportion_projection, info, n_sample))
    print("IPF done")
    
    return res
        

