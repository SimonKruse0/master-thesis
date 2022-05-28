
from cProfile import label
import os
import numpy as np
from src.regression_validation.analysis_helpers import get_data2, get_names

main_data_folder = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data"

folder_name = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data/0522_1435"

import pandas as pd
def plot_means(ax,data_list,name_list, search_name = "Process", color = "blue", modelname = "", type = "mean_rel_error"):
    i = 0
    data_df = pd.DataFrame(index = list(range(10,300)))
    for data, model_name in zip(data_list, name_list):
        if search_name in model_name:
            data_series = pd.Series(data=data[type], index=data["n_train_list"])
            data_df[model_name+f"{i}"]= data_series
            i += 1
    data_df = data_df.ffill()
    data_df.plot(ax=ax,alpha = 0.1, color=color)
    means = data_df.mean(axis=1)
    ax.plot(means.index,means.values,lw=3, color = color, label=f"{modelname}")
    colnames = data_df.columns
    return list(colnames)


def plot_regression_paths(problem,dim, plot_type=0):
    if plot_type == 0:
        type = "mean_rel_error"
    else:
        type = "mean_pred_likelihod"

    fig, ax = plt.subplots()
    problem_dim = f"f{problem}_dim_{dim}"
    ALLdata = get_data2(problem_dim, use_exact_name=True, data_folder ="coco_reg_data")
    data_list,name_list, *_ = ALLdata

    legend_names =  plot_means(ax,data_list,name_list,type=type, search_name = "BOH", color = "blue", modelname = "BOHAMIANN")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "Process", color = "red", modelname = "Gaussian Process")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "emp", color = "green", modelname = "Emperical mean and std")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "numpyro", color = "orange", modelname = "BNN")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "Naive", color = "yellow", modelname = "Naive GMR")

    
    problem_dim = f"f{problem}_DIM{dim}"
    if type==1:
        ax.set_yscale('log')
        ax.set_ylabel("mean relative error")
    else:
        ax.set_ylabel("mean predictive likelihood")

    ax.set_xlabel("Amount of training data")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for name in legend_names:
        try:
            by_label.pop(name)
        except:
            print(f"No data for {name}")
    plt.legend(by_label.values(), by_label.keys())
    ax.set_title(problem_dim)
    plt.show()

import matplotlib.pyplot as plt
if __name__ == "__main__":
    for problem in list(range(1,25)):
        for dim in [2,3,5,10]:
            plot_regression_paths(problem, dim)