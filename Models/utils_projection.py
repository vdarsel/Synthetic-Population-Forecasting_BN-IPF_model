import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_coefs_from_regr(X,Y, target, alpha, power_max):
    X_poly = np.vander(X, power_max+1, increasing=False).T
    alphas = np.power((1-alpha), -X)

    X_alpha = X_poly*alphas
    
    res = []
    matrix_A = np.linalg.inv((X_alpha).dot(np.transpose(X_poly))).dot(X_alpha)
    for i in range(Y.shape[1]):
        coefs_regr = matrix_A.dot(Y[:,i])
        res.append(np.poly1d(coefs_regr)(target))
    return np.array(res)


def get_plots_regr(X,Y, target_X, target_Y, alpha, power_max, folder_path_graph, name):
    X_poly = np.vander(X, power_max+1, increasing=False).T
    alphas = np.power((1-alpha), -X)

    X_alpha = X_poly*alphas
    
    matrix_A = np.linalg.inv((X_alpha).dot(np.transpose(X_poly))).dot(X_alpha)

    random_index = np.arange(Y.shape[1])
    np.random.shuffle(random_index)
    for j,i in enumerate(random_index[:10]):
        plt.figure()
        coefs_regr = matrix_A.dot(Y[:,i])

        x_plot = np.arange(min(X)-1,target_X+2)
        plt.plot(X,Y[:,i])
        plt.plot(x_plot,np.poly1d(coefs_regr)(x_plot))
        plt.scatter([target_X], [target_Y[i]], c="r")
        plt.title(f"{name} {i}")
        file = f"{folder_path_graph}/{name.replace(" ","_")}_{j}.png"
        plt.savefig(file)
        plt.close()
    

def SRMSE_from_freq_series_list(freq_list_1, freq_list_2):
    res = []
    for (freq_1,freq_2) in zip(freq_list_1, freq_list_2):
        SRMSE = np.sqrt(np.sum(np.power(freq_1-freq_2,2))*len(freq_1))
        res.append(SRMSE)
    return np.mean(res)



def get_frequencies_from_df(df, attr, unique_val):
    res = pd.Series(0.0, index=unique_val)
    ser_temp = df[attr].value_counts(normalize=True)
    res.loc[ser_temp.index] = ser_temp
    return res.to_numpy()
