
from cProfile import label
import os
import numpy as np
from src.regression_validation.analysis_helpers import get_data2, get_names

#main_data_folder = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data"
#folder_name = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data/0522_1435"

data_folder = "coco_reg_data"
#data_folder = "sklearn_reg_data"
data_folder = "data"
#/home/simon/Documents/MasterThesis/master-thesis/data/0617_1218

import pandas as pd
def plot_means(ax,data_list,name_list, only_means = True,  
        search_name = "GP", 
        use_pred_mass = True,
        color = "blue", modelname = "", type = "mean_rel_error"):
    i = 0
    #data_df = pd.DataFrame(index = list(range(10,300)))
    data_df = pd.DataFrame(index = [int(x) for x in np.logspace(1, 2.5, 9)])
    flag = 0
    for data, model_name in zip(data_list, name_list):
        data_tmp = data[type]
        if type == "mean_pred_likelihod" and use_pred_mass:# and "GP" not in model_name:
            try:
                data_tmp = data["mean_pred_mass"]
            except:
                pass

        if search_name in model_name:
            flag = 1
            data_series = pd.Series(data=data_tmp, index=data["n_train_list"])
            data_df[model_name+f"{i}"]= data_series
            i += 1
    if flag ==1:
        #data_df = data_df.ffill()
        if not only_means:
            data_df.plot(ax=ax,alpha = 0.1, color=color)
        
        means = data_df.mean(axis=1)
        ax.plot(means.index,means.values,lw=3, color = color, label=f"{modelname}")
    colnames = data_df.columns
    return list(colnames)


def plot_regression_paths(ax, problem_name, plot_type=0):
    if plot_type == 0:
        #type = "mean_rel_error"
        type = "mean_rel_error"
    else:
        type = "mean_pred_likelihod"
    
    if data_folder == "coco_reg_data":
        ALLdata = get_data2(problem_name, use_exact_name=True, data_folder =data_folder)
    else:
        ALLdata = get_data2(problem_name, use_exact_name=False, data_folder =data_folder)

    data_list,name_list, *_ = ALLdata

    legend_names =  plot_means(ax,data_list,name_list,type=type, search_name = "BOH",use_pred_mass=False, color = "yellow", modelname = "BOHAMIANN")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "emp", color = "green", modelname = "Emperical mean and std")
    # legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "Process", color = "red", modelname = "Gaussian Process")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "GP",use_pred_mass=False, color = "red", modelname = "Gaussian Process")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "BNN",use_pred_mass=False, color = "orange", modelname = "BNN")
    #legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "numpyro", color = "orange", modelname = "BNN")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "KDE", color = "blue", modelname = "Kernel density")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "GMR", color = "cyan", modelname = "GMR")
    #legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "Naive", color = "yellow", modelname = "Kernel density")
    legend_names += plot_means(ax,data_list,name_list,type=type, search_name = "SPN", color = "black", modelname = "SPN")


    if plot_type==0:
        ax.set_yscale('log')
        ax.set_ylabel("Mean relative error")
    else:
        ax.set_ylabel("Mean log predictive")

    ax.set_xlabel("Size of training data")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for name in legend_names:
        try:
            by_label.pop(name)
        except:
            print(f"No data for {name}")
    plt.legend(by_label.values(), by_label.keys())
    ax.set_title(problem_name)

import matplotlib.pyplot as plt
if __name__ == "__main__":
    ## plotting for thesis ##
    result_folder = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Figures/results_regression/"
    for number in ["1","2","3","4","3b"]:
        for type in ["0","1"]:
            fig, ax = plt.subplots()
            #number = input("what problem 1,2,3,4,3b? ")
            #type = input("what type 0,1? ")
            problem_name = f"Test{number}_dim_1"
            plot_regression_paths(ax, problem_name, plot_type=int(type))
            #plt.ylim(-5,4)
            plt.grid()
            plt.savefig(result_folder+problem_name+f"_{type}"+".pdf")
    
    plt.show()

    if data_folder == "coco_reg_data":
        for problem in list(range(1,25)):
            for dim in [2]:
                fig, ax = plt.subplots()
                problem_name = f"f{problem}_dim_{dim}"
                plot_regression_paths(ax, problem_name, plot_type=2)
    else:
        problem_name = "SimonsTest2_probibalistic_dim_1"
        while True:
            fig, ax = plt.subplots()
            number = input("what problem 1,2,3,4,3b? ")
            type = input("what type 0,1? ")
            problem_name = f"Test{number}_dim_1"
            plot_regression_paths(ax, problem_name, plot_type=int(type))
            plt.ylim(-5,4)
            plt.show()
    