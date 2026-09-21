import os

import numpy as np
from pgmpy.models import BayesianNetwork

from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd

from tqdm import tqdm

from itertools import combinations
    
from Models.utils_projection import get_coefs_from_regr, get_plots_regr, SRMSE_from_freq_series_list

def get_best_PGM_hill_climb(data):
    hc = HillClimbSearch(data)
    best_model = hc.estimate(scoring_method=BicScore(data),)

    return best_model


def freq_list_from_dicts_model_specialized_one_attribute(model, dicts_unique_list,list_other_attributes, specialized_attribute, k):
    assert(k<=len(list_other_attributes))
    
    VE = VariableElimination(model)
    
    combis = (combinations(np.arange(len(list_other_attributes)),k))
    freq_list = []
    for combination in combis:
        cols = np.array([specialized_attribute])            
        if(len(combination))>0:
           cols = np.concat([cols, list_other_attributes[np.array(combination)]])
        freq_serie = pd.Series(0.0, index = pd.MultiIndex.from_product(
            [dicts_unique_list[col] for col in cols]
        ))
        
        t = VE.query(variables = cols, joint = True)
        unique_values_model = [t.state_names[specialized_attribute]]
        if(len(combination))>0:
            for col in  list_other_attributes[np.array(combination)]:
                unique_values_model.append(t.state_names[col])
        freq_serie_extracted = pd.Series(t.values.flatten(), index = pd.MultiIndex.from_product(unique_values_model))
        freq_serie.loc[freq_serie_extracted.index] = freq_serie_extracted
        freq_list.append(freq_serie)
    return freq_list


def fit_model(model, data, dict_unique):
    '''Fit a BN with a given architecture (in model) from a dataset (to implement the transition probabilities)'''
    BayesianModel = BayesianNetwork(model)
    BayesianModel.fit(data.head(3))

    for i in range(len(BayesianModel.get_cpds())):
        (cpd) = BayesianModel.get_cpds()[0]
        BayesianModel.remove_cpds(cpd)
        vars = np.concat([[cpd.variable],cpd.get_evidence()])
        if len(cpd.get_evidence())>1:
            res = pd.crosstab([data[col] for col in cpd.get_evidence()],data[cpd.variable]).reindex(columns=dict_unique[cpd.variable], index=pd.MultiIndex.from_product([dict_unique[var] for var in cpd.get_evidence()]),fill_value=0).sort_index().to_numpy()
        elif (len(cpd.get_evidence())==1):
            res = pd.crosstab([data[col] for col in cpd.get_evidence()],data[cpd.variable]).reindex(columns=dict_unique[cpd.variable], index=dict_unique[cpd.get_evidence()[0]],fill_value=0).sort_index().to_numpy()
        else:
            res = data[cpd.variable].value_counts(normalize=True).reindex(index=dict_unique[cpd.variable],fill_value=0).to_numpy().reshape(1,-1)
        res = np.max([np.ones_like(res)*1e-10,res],0)
        sum_values = np.sum(res,tuple([i for i in range(1,len(res.shape))]), keepdims=True)
        # res = res/np.max([np.ones_like(sum_values),sum_values],0)
        res = res/sum_values
        new_cpd = TabularCPD(cpd.variable, len(dict_unique[cpd.variable]), (res.T), evidence=cpd.get_evidence(), evidence_card=[len(dict_unique[var]) for var in cpd.get_evidence()], 
                             state_names={str(var):list(dict_unique[var]) for var in vars})
        BayesianModel.add_cpds(new_cpd)
    return BayesianModel


def freq_list_from_dicts_specialized_one_attribute(df, dicts_unique_list, list_other_attributes, specialized_attribute, k):
    assert(k<=len(list_other_attributes))
    
    combis = (combinations(np.arange(len(list_other_attributes)),k))
    freq_list = []
    for combination in combis:
        cols = np.array([specialized_attribute])            
        if(len(combination))>0:
           cols = np.concat([cols, list_other_attributes[np.array(combination)]])
        freq_serie = pd.Series(0.0, index = pd.MultiIndex.from_product(
            [dicts_unique_list[col] for col in cols]
        ))
        freq_temp = df[cols].value_counts(normalize=True)
        freq_serie.loc[freq_temp.index] = freq_temp
        freq_list.append(freq_serie)
    return freq_list



def synthetic_population_from_projected_BN_hill(training_data, training_data_validation, 
                      validation_data, n_alphas,
                      alpha_min, alpha_max, 
                      n_min, n_max, 
                      training_values_x, projection_value_x,
                      encoding_function, decoding_function, 
                      folder_sampling, n_generation,
                      dict_unique,
                      print_regression=True):
    
    folder_sampling_graph = f"{folder_sampling}/Graph"
    
    if not os.path.isdir(folder_sampling_graph):
        os.makedirs(folder_sampling_graph)

    
    print("Structure research...")
    # Computing the best sructure for Bayesian Network
    PGM = get_best_PGM_hill_climb(training_data[-1].astype(str))
    
    nx.draw_circular(
        PGM, with_labels=True, arrowsize=30, node_size=800, alpha=0.3, font_weight="bold"
    )
    plt.savefig(f"{folder_sampling}/best_graph.png")
    plt.close()
    print("Structure Found")
    
    print("Analysis of the structure")
    # Create a ref model
    model_ref = fit_model(PGM, training_data[-1].astype(str), dict_unique)
    
    # Constructing step by step PGM
    PGM_copy = PGM.copy()
    missing = len(PGM_copy.nodes)
    list_per_level = []
    while(missing>0):
        roots = PGM_copy.get_roots()
        list_per_level.append(roots)
        missing= missing-len(roots)
        for node in roots:
            PGM_copy.remove_node(node)
    list_order = np.concat(list_per_level)
    
    list_cpds_order = np.array([cpd.variable for cpd in model_ref.get_cpds()])
    
    order_cpds =[]
    for attr in list_order:
        order_cpds.append(np.where(list_cpds_order==attr)[0][0])
    order_cpds = np.array(order_cpds)
    
    order_cpds_cropped =[]
    for i,id in enumerate(order_cpds):
        order_cpds_cropped.append(np.sum(order_cpds[:i]<id)) # count number of elements inferior for previous cpds (get relative position)
    order_cpds_cropped = np.array(order_cpds_cropped)
    print("Done")
    
    print("Find hyperparameters [VALIDATION]")
    
    print("Fit models...")
    # fitting validation models
    models_validation = [fit_model(PGM, data_year.astype(str), dict_unique) for data_year in training_data_validation]
    model_validation_year = fit_model(PGM, validation_data.astype(str), dict_unique)
    print("Models fitted")
    
    print("Grid search...")
    # List best parameters
    model_result_validation = models_validation[-1].copy()
    alphas_res = []
    n_res = []
    n_alpha = n_alphas
    n_max = n_max
    n_min = n_min
    
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)
    
    for i in tqdm(range(len(list_order))):
        model_eval = model_result_validation.copy()
        attributes = list_order[:i+1]
        attributes_to_remove = list_order[i+1:]
        
        freq_validation = freq_list_from_dicts_specialized_one_attribute(validation_data.astype(str), dict_unique,attributes[:i],attributes[i],min(i,2))
        
        cpd_id = order_cpds[i]
        for rem in attributes_to_remove:
            model_eval.remove_node(rem)
            
        assert model_eval.get_cpds()[order_cpds_cropped[i]].variable==attributes[i]
        Y_vals = [model.get_cpds()[cpd_id].values for model in models_validation]
        coefs_last_year = Y_vals[-1].reshape(-1)
        coefs_to_estimate = encoding_function(Y_vals,coefs_last_year)
        
        SRMSE_tab = np.ones((n_alpha, n_max-n_min+1), dtype=float)*1e7
        for j in (range(len(alphas))):
            alpha = alphas[j]
            for k,n in enumerate(range(n_min, n_max+1)):
                coefs_estimated = get_coefs_from_regr(training_values_x, coefs_to_estimate, projection_value_x, alpha, n) 
                coefficients_cpd = decoding_function(coefs_estimated, models_validation[-1].get_cpds()[cpd_id].values)
                model_eval.get_cpds()[order_cpds_cropped[i]].values = coefficients_cpd
                
                freq_eval = freq_list_from_dicts_model_specialized_one_attribute(model_eval, dict_unique,attributes[:i],attributes[i],min(i,2))
    
                SRMSE_specialized = SRMSE_from_freq_series_list(freq_validation, freq_eval)
                SRMSE_tab[j,k] = SRMSE_specialized

        hyperparameters = np.unravel_index(SRMSE_tab.argmin(), SRMSE_tab.shape)

        alpha_best = alphas[hyperparameters[0]]
        n_best = hyperparameters[1] 
        
        alphas_res.append(alpha_best)
        n_res.append(n_best)

        ## Recover the best model
        coefs_estimated = get_coefs_from_regr(training_values_x, coefs_to_estimate, projection_value_x, alpha_best, n_best) 
        coefficients_cpd = decoding_function(coefs_estimated, models_validation[-1].get_cpds()[cpd_id].values)
        model_result_validation.get_cpds()[cpd_id].values = coefficients_cpd
        
        if print_regression:
            name = f"Coefficients for {attributes[-1]}"
            if len(models_validation[-1].get_cpds()[cpd_id].get_evidence())>0:
                 name+=f" conditionnally to {" ".join(models_validation[-1].get_cpds()[cpd_id].get_evidence())}"                
            coefs_to_estimate_val = encoding_function([model_validation_year.get_cpds()[cpd_id].values], coefs_last_year)[0]

            get_plots_regr(training_values_x, coefs_to_estimate, projection_value_x, coefs_to_estimate_val, alpha_best, n_best, folder_sampling_graph,name)
    
    print("Grid search over")
    print("Best parameters computed")
    
    print("Application to the target year [PREDICTION]")
    # Apply best parameters on year to predict
    models_prediction = [fit_model(PGM, data_year.astype(str), dict_unique) for data_year in training_data]
    model_final = models_prediction[-1].copy()
    for i in range(len(list_order)):
        cpd_id = order_cpds[i]
        assert model_final.get_cpds()[cpd_id].variable==attributes[i]
        alpha = alphas_res[i]
        n = n_res[i]
        
        Y_preds = [model.get_cpds()[cpd_id].values for model in models_prediction]
        coefs_last_year = Y_preds[-1]
        coefs = encoding_function(Y_preds,coefs_last_year.reshape(-1))
        coefs_estimated_pred = get_coefs_from_regr(training_values_x, coefs, projection_value_x, alpha, n) 
        coefficients_cpd = decoding_function(coefs_estimated_pred, coefs_last_year)
        model_final.get_cpds()[cpd_id].values = coefficients_cpd
    print("Model prepared")
    np.save(f"{folder_sampling}/n_BN.npy", np.array(n_res))
    
    print("Population generation...")
    synthetic_population = model_final.simulate(n_generation)
    synthetic_population = synthetic_population[validation_data.columns]
    
    
    return synthetic_population
        

