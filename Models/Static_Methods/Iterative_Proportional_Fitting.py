import pandas as pd
import numpy as np 

import duckdb


def compute_error(unique_val_dict, dict_proportions, variables):
    errors = []
    for var in variables:
        uniques = unique_val_dict[var]
        proportion_target_var = dict_proportions[var]
        err = 0
        for unique, proportion_target in zip(uniques, proportion_target_var):
            proportion_res = duckdb.sql(f"SELECT sum(proportion) as s FROM table_potential_indiv_grouped WHERE {var}='{unique}'").fetchnumpy()['s'][0]

            if proportion_res>0:
                err += np.power(proportion_res-proportion_target,2)
            # else:
            #     err+= np.power(proportion_target,2) #not computed as it is impossible to correct it if not in the data
        errors.append(np.sqrt(err*len(uniques)))
    # print(errors)
    error = np.mean(errors)
    return error


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
    print("IPF, initial error:", compute_error(unique_val_dict, dict_proportions, vars))
    while((error>0.0025)&(epoch<30)):   
        # the error threshold seems to be experience dependant. Depending on the training size, number of variables, the good threshold may evolve
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
        error= compute_error(unique_val_dict, dict_proportions, vars)
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
            
