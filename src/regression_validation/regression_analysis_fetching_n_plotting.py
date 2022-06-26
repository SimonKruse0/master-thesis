
from cProfile import label
import os
import numpy as np
from src.regression_validation.analysis_helpers import get_data2, get_names

#main_data_folder = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data"
#folder_name = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data/0522_1435"

#data_folder = "coco_reg_data"
#data_folder = "sklearn_reg_data"

# FOR TEST PROLBEMS
data_folder = "data/0624_20"

#/home/simon/Documents/MasterThesis/master-thesis/data/0617_1218

import pandas as pd
def plot_means(ax,data_list,name_list, only_means = False,  
        search_name = "GP", 
        use_pred_mass = True,
        color = "blue", modelname = "", type = "mean_rel_error"):
    i = 0
    #data_df = pd.DataFrame(index = list(range(10,300)))
    data_df = pd.DataFrame(index = [int(x) for x in np.logspace(1, 2.5, 9)])
    #data_df = data_df.loc[5:]
    flag = 0
    for data, model_name in zip(data_list, name_list):
        data_tmp = data[type]
        #print(type)
        if type == "mean_pred_likelihod" and use_pred_mass:# and "GP" not in model_name:
            try:
                data_tmp = data["mean_pred_mass"]
            except:
                pass
        if type == "mean_pred_likelihod":
            #pass
            data_tmp = np.exp(data_tmp)

        if search_name in model_name:
            flag = 1
            data_series = pd.Series(data=data_tmp, index=data["n_train_list"])
            data_df[model_name+f"{i}"]= data_series
            i += 1

    if flag ==1:
        #data_df = data_df.iloc[:6]
        #data_df = data_df.ffill()
        if not only_means:
            data_df.plot(ax=ax,alpha = 0.1, color=color)
        
        means = data_df.mean(axis=1)
        ax.plot(means.index,means.values,lw=3, color = color, label=f"{modelname}")
    colnames = data_df.columns
    return list(colnames)


def plot_regression_paths(ax, problem_name, plot_type=0, only_means= False):
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

    legend_names =  plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "GP",use_pred_mass=False, color = "red", modelname = "GP")
    legend_names += plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "BNN",use_pred_mass=False, color = "orange", modelname = "BNN")
    legend_names += plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "BOH",use_pred_mass=False, color = "gold", modelname = "BOHAMIANN")
    legend_names += plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "emp", color = "black", modelname = "Emperical mean and std")
    legend_names += plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "KDE", color = "blue", modelname = "KDE")
    legend_names += plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "GMR", color = "cyan", modelname = "GMR")
    legend_names += plot_means(ax,data_list,name_list,only_means=only_means,type=type, search_name = "SPN", color = "purple", modelname = "SPN")


    if plot_type==0:
        ax.set_yscale('log')
        ax.set_ylabel("Mean relative error")
    else:
        ax.set_ylabel("exp(mean log predictive)")

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
    for number in ["1","2","3b","4b"]:
        for type in ["0","1"]:
            fig, ax = plt.subplots()
            #number = input("what problem 1,2,3,4,3b? ")
            #type = input("what type 0,1? ")
            problem_name = f"Test{number}_dim_1"
            if number == "3b" and type=="0":
                plot_regression_paths(ax, problem_name,only_means=True, plot_type=int(type))
            else:
                plot_regression_paths(ax, problem_name,only_means=True, plot_type=int(type))
            plt.xlim(10,316)
            ax.set_xscale("log")
            #ax.set_yscale("log")
            #
            if type == "1":
                if number == "1" or number == "2":
                    ax.set_yscale("log")
                    plt.ylim(1e-4,20)
            #     ax.set_yscale("log")
            #     
            #     
            #         
            #     else:
            #         plt.ylim(-5,1)
            ax.yaxis.label.set_size(13)
            ax.xaxis.label.set_size(13)
            ax.set_title(problem_name[:5])
            ax.title.set_size(15)
            plt.grid()
            plt.tight_layout()
            fig = plt.gcf()
            fig.set_size_inches(6, 4)
            plt.savefig(result_folder+problem_name+f"_{type}"+".pdf")
            #plt.savefig(result_folder+problem_name+f"_{type}_zoom"+".pdf")
    
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
            #plt.ylim(-5,4)
            plt.show()
    