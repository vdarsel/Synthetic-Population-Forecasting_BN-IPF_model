import numpy as np
import pandas as pd
import os
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Evaluation.proportion_sampling import compute_proportion_file_from_unique_array_and_df, recover_lists_from_dictionnary
from Evaluation.metrics_proportion import get_df_scores_by_cat, get_scores_agg
from Evaluation.heatmap import generate_color_map_save, generate_color_map_filter_save
import matplotlib.pyplot as plt
from Evaluation.metrics_originality import get_proportion_from_original_data_df, get_proportion_from_original_data_df_not_in_other_df, get_rate_of_impossible_combinations
from Evaluation.metrics_privacy import Distance_to_Closest_Records,generate_histogram_DCR
import plotly.graph_objects as go

CAT_RARE_VALUE = '__rare__'

def preprocessing_cat_data_dataframe_sampling(dataset_ref: pd.DataFrame, min_count, cols_cat, other_dataset : list[pd.DataFrame] = []):
    if(len(cols_cat)>0):
        X_new = dataset_ref.copy()
        X_new_list = [df.copy() for df in other_dataset]
        print("Initial categories (training):", dataset_ref[cols_cat].nunique().to_list())
        for column_idx in (cols_cat):
            value_counts = dataset_ref[column_idx].value_counts()
            popular_categories = value_counts[value_counts>=min_count].index
            X_new.loc[~X_new[column_idx].isin(popular_categories),column_idx] = CAT_RARE_VALUE
            for df in X_new_list:
                df.loc[~df[column_idx].isin(popular_categories),column_idx] = CAT_RARE_VALUE
        print("Final categories (training):", X_new[cols_cat].nunique().to_list())
        return X_new, X_new_list


def generate_histogram(df, path, i):
    metrics = ["SRMSE","Hellinger","Pearson","R2"]
    for metric in metrics:
        plt.figure(figsize=[7,7])
        plt.hist(df[metric])
        plt.title(f'{metric} {i}')
        plt.savefig(f"{path}/hist_{metric}_{i}.png")
        plt.close()
    
def generate_plot(proportion_sample, proportion_ref, path, i):
    proportion_ref_concat = np.concat(proportion_ref)
    proportion_sample_concat = np.concat(proportion_sample)
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    plt.figure(figsize=[7,7])
    plt.plot([-1,1], [-1,1], color='r', alpha=0.5)
    for p,q in zip(proportion_ref, proportion_sample):
        plt.scatter(p,q)
    plt.xlim(-0.02,np.max([proportion_ref_concat,proportion_sample_concat])+0.02)
    plt.ylim(-0.02,np.max([proportion_ref_concat,proportion_sample_concat])+0.02)
    plt.xlabel("Original data proportion")
    plt.ylabel("Sampled data proportion")
    plt.title(dict_title[i])
    plt.savefig(f"{path}/comparison_{i}.png")
    plt.close()
    generate_color_map_save(proportion_ref_concat,proportion_sample_concat,path,f"Full_Heatmap_{i}",200)
    generate_color_map_filter_save(proportion_ref_concat,proportion_sample_concat,path,f"Full_Heatmap_filter_{i}",2000, min_freq=1)

def generate_plot_plotly(proportion_sample, proportion_ref, combi_names, values, path, i, target=100000):
    coef = max(len(proportion_sample)//target,1)
    keep_idx = np.array([i%coef==0 for i in range(len(proportion_ref))])
    proportion_ref = proportion_ref[keep_idx]
    proportion_sample = proportion_sample[keep_idx]
    combi_names = combi_names[keep_idx]
    values = values[keep_idx]
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    fig = go.Figure(data=[go.Scatter(x=[-1,1], y=[-1,1],mode="lines",line=dict(color='rgba(255, 17, 0, 0.5)'))],
                    layout=go.Layout(title=dict_title[i], showlegend=False,
                                     xaxis_range=[-0.02,np.max([proportion_ref,proportion_sample])+0.02],
                                     yaxis_range=[-0.02,np.max([proportion_ref,proportion_sample])+0.02],
                                     xaxis_title="Original data proportion",
                                     yaxis_title="Sampled data proportion"))
    fig.add_trace(go.Scatter(x=proportion_ref,y=proportion_sample,mode="markers", customdata=np.stack([combi_names,values], axis=1),
                             hovertemplate="<br>".join([
                                "ColX: %{x}",
                                "ColY: %{y}",
                                "Col1: %{customdata[0]}",
                                "Col2: %{customdata[1]}",
                            ])))
    fig.write_html(f"{path}/comparison_{i}.html")
    generate_color_map_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_{i}",2000)
    generate_color_map_filter_save(proportion_ref,proportion_sample,path,f"Full_Heatmap_filter_{i}",2000, min_freq=10)


def output_serie_eval(dict_unique_values, basename_file_save, proportion_test_file, df_for_evaluation, dataset_test_df, dataset_test_df_dcr, dataset_reference_training, df_info, dir_path_save_results=None, save=False):
    Ser_review = pd.Series(index=["SRMSE marginal (cat)","SRMSE bivariate (cat)","SRMSE trivariate (cat)",
                     "Hellinger marginal (cat)","Hellinger bivariate (cat)","Hellinger trivariate (cat)",
                     "SRMSE marginal (global)","SRMSE bivariate (global)","SRMSE trivariate (global)",
                     "Hellinger marginal (global)","Hellinger bivariate (global)","Hellinger trivariate (global)",
                     "Pearson marginal (global)","Pearson bivariate (global)","Pearson trivariate (global)",
                     "R2 marginal (global)","R2 bivariate (global)","R2 trivariate (global)",
                     "Rate of copies (training, with geo)", "Rate of copies (training, without geo)",
                     "Rate of copies (testing not training, with geo)", "Rate of copies (testing not training, without geo)",
                     "Rate of undesired couples (with geo)", "Rate of undesired couples (without geo)"
                     ], dtype=str)

    columns = df_info["Variable_name"].to_numpy()
    columns_without_geo = df_info[(~df_info["Geographical_attribute"])]["Variable_name"]

    
    dict_title = {1:"Marginal", 2:"Bivariate", 3:"Trivariate"}
    df_scores_median = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_mean = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])
    df_scores_agg = pd.DataFrame(columns=["SRMSE","Hellinger","Pearson","R2"])

    
    for i in range(1,4):    
        proportion_file = f"{basename_file_save}_{i}.npy"
        if (not os.path.isfile(proportion_file)):
            print("\nGenerate Proportion file for sample")
            compute_proportion_file_from_unique_array_and_df(dict_unique_values,
                    df_for_evaluation.astype(str),
                    columns,
                    basename_file_save,
                    i,
                    ".")
        combi_test_file = f"{proportion_test_file}_{i}_comb.npy"
        value_test_file = f"{proportion_test_file}_{i}_values.npy"

        proportion_concat = np.load(proportion_file)
        proportion_test_concat = np.load(f"{proportion_test_file}_{i}.npy")
        combi_concat = np.load(combi_test_file).astype(int)
        combi_names_concat = columns[combi_concat]
        values_test_concat = np.load(value_test_file, allow_pickle=True)
        
        proportion = recover_lists_from_dictionnary(columns, dict_unique_values, proportion_concat, i)
        proportion_test = recover_lists_from_dictionnary(columns, dict_unique_values, proportion_test_concat, i)
        combi_list = np.array([a[0] for a in recover_lists_from_dictionnary(columns, dict_unique_values, combi_concat, i)]).astype(int)
        combi_names = columns[combi_list]
        
        df_scores_by_cat = get_df_scores_by_cat(proportion, proportion_test, combi_list, combi_names, i)
        
        if save:
            df_scores_by_cat.to_csv(f"{dir_path_save_results}/scores_by_cat_{i}.csv", sep=";")    
            generate_histogram(df_scores_by_cat, dir_path_save_results, i)
            generate_plot(proportion, proportion_test, dir_path_save_results, i)
            generate_plot_plotly(proportion_concat, proportion_test_concat, combi_names_concat, values_test_concat, dir_path_save_results, i)
        df_scores_median.loc[dict_title[i]] =  df_scores_by_cat[["SRMSE","Hellinger","Pearson","R2"]].median()
        df_scores_mean.loc[dict_title[i]] = df_scores_by_cat[["SRMSE","Hellinger","Pearson","R2"]].mean()
        df_scores_agg.loc[dict_title[i]] = pd.Series(get_scores_agg(proportion_test,proportion),index=["SRMSE","Hellinger","Pearson","R2"])
        
        proportion_file = f"{basename_file_save}_{i}.npy"
        proportion = np.load(proportion_file)

    df_DCR = Distance_to_Closest_Records(df_for_evaluation, dataset_reference_training, dataset_test_df_dcr, df_info.reset_index(),proportion_test_file)

    if save:
        generate_histogram_DCR(df_DCR, dir_path_save_results)
    Wasserstein_distance_DCR = np.sqrt(1/(len(df_DCR)**2)*np.sum(np.power(df_DCR["DCR train"].sort_values().values - df_DCR["DCR test"].sort_values().values,2)))
        
    rate_non_geo_list, rate_geo_list = [],[]
    for i in range(1,4):
        proportion_file = f"{basename_file_save}_{i}.npy"
        proportion = np.load(proportion_file)

        proportion_test = np.load(f"{proportion_test_file}_{i}.npy")
        values = np.load(f"{proportion_test_file}_{i}_values.npy", allow_pickle=True)
        combs = np.load(f"{proportion_test_file}_{i}_comb.npy")
        rate_non_geo, rate_geo = get_rate_of_impossible_combinations(df_for_evaluation,df_info, proportion_test, proportion, columns, combs, values)
        rate_non_geo_list.append(rate_non_geo)
        rate_geo_list.append(rate_geo)

    if save:
        df_scores_agg.to_csv(f"{dir_path_save_results}/scores.csv", sep=";")
        df_scores_mean.to_csv(f"{dir_path_save_results}/scores_mean.csv", sep=";")
        df_scores_median.to_csv(f"{dir_path_save_results}/scores_median.csv", sep=";")

    Ser_review["SRMSE marginal (cat)"] = f"{df_scores_mean["SRMSE"]["Marginal"] :.3g}"
    Ser_review["SRMSE bivariate (cat)"] = f"{df_scores_mean["SRMSE"]["Bivariate"] :.3g}"
    Ser_review["SRMSE trivariate (cat)"] = f"{df_scores_mean["SRMSE"]["Trivariate"] :.3g}"
    Ser_review["Hellinger marginal (cat)"] = f"{df_scores_mean["Hellinger"]["Marginal"] :.3g}"
    Ser_review["Hellinger bivariate (cat)"] = f"{df_scores_mean["Hellinger"]["Bivariate"] :.3g}"
    Ser_review["Hellinger trivariate (cat)"] = f"{df_scores_mean["Hellinger"]["Trivariate"] :.3g}"
    Ser_review["SRMSE marginal (global)"] = f"{df_scores_agg["SRMSE"]["Marginal"] :.3g}"
    Ser_review["SRMSE bivariate (global)"] = f"{df_scores_agg["SRMSE"]["Bivariate"] :.3g}"
    Ser_review["SRMSE trivariate (global)"] = f"{df_scores_agg["SRMSE"]["Trivariate"] :.3g}"
    Ser_review["Hellinger marginal (global)"] = f"{df_scores_agg["Hellinger"]["Marginal"] :.3g}"
    Ser_review["Hellinger bivariate (global)"] = f"{df_scores_agg["Hellinger"]["Bivariate"] :.3g}"
    Ser_review["Hellinger trivariate (global)"] = f"{df_scores_agg["Hellinger"]["Trivariate"] :.3g}"
    Ser_review["Pearson marginal (global)"] = f"{df_scores_agg["Pearson"]["Marginal"] :.3g}"
    Ser_review["Pearson bivariate (global)"] = f"{df_scores_agg["Pearson"]["Bivariate"] :.3g}"
    Ser_review["Pearson trivariate (global)"] = f"{df_scores_agg["Pearson"]["Trivariate"] :.3g}"
    Ser_review["R2 marginal (global)"] = f"{df_scores_agg["R2"]["Marginal"] :.3g}"
    Ser_review["R2 bivariate (global)"] = f"{df_scores_agg["R2"]["Bivariate"] :.3g}"
    Ser_review["R2 trivariate (global)"] = f"{df_scores_agg["R2"]["Trivariate"] :.3g}"
    Ser_review["Rate of undesired couples (with geo)"] = f'{rate_geo_list[1]:.2%}'
    Ser_review["Rate of undesired couples (without geo)"] = f'{rate_non_geo_list[1]:.2%}'
    Ser_review["Rate of copies (training, with geo)"] = f'{np.abs(get_proportion_from_original_data_df(dataset_reference_training,dataset_test_df,columns)-get_proportion_from_original_data_df(dataset_reference_training,df_for_evaluation,columns)):.2%}'
    Ser_review["Rate of copies (training, without geo)"] = f'{np.abs(get_proportion_from_original_data_df(dataset_reference_training,dataset_test_df,columns_without_geo)-get_proportion_from_original_data_df(dataset_reference_training,df_for_evaluation,columns_without_geo)):.2%}'

    val = get_proportion_from_original_data_df_not_in_other_df(dataset_test_df,dataset_reference_training,df_for_evaluation,columns)
    if (val=="NA"):
        Ser_review["Rate of copies (testing not training, with geo)"] = 'NA'
    else:
        Ser_review["Rate of copies (testing not training, with geo)"] = f'{val:.2%}'
    val = get_proportion_from_original_data_df_not_in_other_df(dataset_test_df,dataset_reference_training,df_for_evaluation,columns_without_geo)
    if (val=="NA"):
        Ser_review["Rate of copies (testing not training, without geo)"] = 'NA'
    else:
        Ser_review["Rate of copies (testing not training, without geo)"] = f'{val:.2%}'
    Ser_review["Wasserstein-DCR"] = Wasserstein_distance_DCR
    
    return Ser_review


def evaluation(args, term_0=""):
    term=term_0+"_Projection"
    if args.IPF_pre:
        term+="_IPF"
    if args.BN:
        term+="_Bayesian_Network_Hill_Climb"
    if args.IPF_post:
        term+="_IPF"

    if not ("special_model" in args.__dict__.keys()):
        args.special_model= ""
    
    datapath = args.datapath
    dataname_base = args.dataname
    filename = args.filename_training
    infoname = args.infoname
    sample_folder = args.sample_folder
    attr_setname = args.attributes_setname
    n_sample = args.n_generation
    year_prediction = args.year_prediction
    year_validation = year_prediction-args.time_horizon
    
    term+=f"_{args.encoding}_from_{year_validation-args.n_years-args.time_horizon+1}_Validation_{year_validation}"

    dataname_test = dataname_base.replace("XXXX",str(year_prediction))
    dataname_reference = dataname_base.replace("XXXX",str(year_validation))

    info_path = f'{datapath}/{infoname}'
    filename_test = args.filename_test
    file_test = f'{datapath}/{dataname_test}/{filename_test}'
        
    filename_sampling = (args.sampling_terminaison+"_"+str(n_sample)+term+".").join(args.filename_training.split('.'))
    folder_sampling = f'{sample_folder}/{args.size_data_str}/{args.folder_save.replace("TTTT",str(args.year_prediction))+term}'
    
    dir_path_save_results = f"{folder_sampling}/{dataname_test}"

    if not os.path.isdir(dir_path_save_results):
        os.makedirs(dir_path_save_results)
    n_samples = args.n_generation
    
    filename_sampling = (args.sampling_terminaison+"_"+str(n_samples)+term+".").join(args.filename_training.split('.')).split(".csv")[0]
    basename = f"{folder_sampling}/{filename_sampling.split(".csv")[0]}"
    basename_save = f"{dir_path_save_results}/{filename.split(".csv")[0]+args.sampling_terminaison}_{str(args.n_generation)+term}"

    basename_reference_save = f"{dir_path_save_results}/{filename.split(".csv")[0]+args.sampling_terminaison}_{str(args.n_generation)+term+"_reference_"+str(year_validation)}"
    
    file_reference = f"{datapath}{dataname_reference}/{filename}"

    df_info = pd.read_csv(info_path,sep=";")
    df_info = df_info[df_info[attr_setname]]
    columns = df_info["Variable_name"].to_numpy()
    
    def load_data(filename, df_info):
        data = pd.read_csv(filename, sep=";", low_memory=False,usecols=df_info["Variable_name"])
        for idx in df_info[df_info["Type"].isin(["category","binary"])]["Variable_name"]:
            data[idx] = data[idx].astype(str)
        for idx in df_info[df_info["Type"].isin(["int"])]["Variable_name"]:
            data[idx] = data[idx].astype(int)
        for idx in df_info[df_info["Type"].isin(["float"])]["Variable_name"]:
            data[idx] = data[idx].astype(float)
        return data  

    dataset_reference = load_data(file_reference, df_info)
    df_testing_equalsize = load_data(file_test.replace(".csv",'_equal_size_training.csv'), df_info)
    df_sample = load_data(f'{basename}.csv', df_info)
    dataset_test = load_data(file_test, df_info)
    
    print(f"Datasets size:\tTraining:{len(dataset_reference)}\tTesting:{len(dataset_test)}")
        
    dict_unique_values = {}
    for col in columns:
        unique_values = np.load(f"../../Results/Projection_References/Unique_Values_Rescencement_Projection_2007_2021/unique_values_{col}.npy", allow_pickle=True).astype(str)
        dict_unique_values[col] = np.sort(unique_values)


    proportion_test_file = f"Results/Projection_References/{attr_setname}/{dataname_test}/{dataname_test}/Proportion_reference_data_{dataname_test}_{(filename).split(".")[0]}_{attr_setname}_{filename_test.split(".csv")[0]+args.special_model}"
    folder_test_file = f"Results/Projection_References/{attr_setname}/{dataname_test}/{dataname_test}"
    name_test_file = f"Proportion_reference_data_{dataname_test}_{(filename).split(".")[0]}_{attr_setname}_{filename_test.split(".csv")[0]+args.special_model}"
    
        
    
    for i in range(1,4):    
        if (not os.path.isfile(f"{proportion_test_file}_{i}.npy")):
            if(not os.path.isdir(os.path.dirname(f"{folder_test_file}/{name_test_file}"))):
                os.makedirs(os.path.dirname(f"{folder_test_file}/{name_test_file}"))
            print("\nGenerate Proportion file for test data")
            
            compute_proportion_file_from_unique_array_and_df(dict_unique_values, dataset_test.astype(str), columns, proportion_test_file, i, ".", True)
        else:
            print(f"Shape ({i}):", np.load(f"{proportion_test_file}_{i}.npy").shape)
    
    ser_scores_generated = output_serie_eval(dict_unique_values, basename_save, proportion_test_file, df_sample, dataset_test, df_testing_equalsize, dataset_reference, df_info, dir_path_save_results, True)
    
    ser_scores_reference = output_serie_eval(dict_unique_values, basename_reference_save, proportion_test_file, dataset_reference, dataset_test, df_testing_equalsize, dataset_reference, df_info)
    
    
    res_pandas = pd.DataFrame([ser_scores_generated, ser_scores_reference], index=["Generated","Training data"])    
    res_pandas.transpose().to_csv(f"{dir_path_save_results}/overview_score.csv", sep=";")

