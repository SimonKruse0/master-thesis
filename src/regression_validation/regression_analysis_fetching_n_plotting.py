
from cProfile import label
import os
import numpy as np
from src.regression_validation.analysis_helpers import get_data2, get_names

main_data_folder = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data"

folder_name = "/home/simon/Documents/MasterThesis/master-thesis/coco_reg_data/0522_1435"
def get_optimization_history(coco_folder):
    data = dict()
    coco_folder = f"{main_data_folder}/{coco_folder}"

    for root, dirs, files in os.walk(coco_folder):
        for file in files:
            if file.endswith(".dat"):
                problem = root.split("_")[-1]
                dim = file.split("_")[-2]
                data_path = os.path.join(root, file)
                datContent = [i.strip().split("|") for i in open(data_path).readlines()]
                if datContent == []:
                    continue
                F = [float(l[0].split(" ")[2]) for l in datContent[1:]] #best noise-free fitness - Fopt (7.948000000000e+01)
                f = [float(l[0].split(" ")[3]) for l in datContent[1:]] # measured fitness 
                iter = [int(l[0].split(" ")[0]) for l in datContent[1:]]

                best_f = float(datContent[0][2].split("(")[1].split(")")[0])
                F = np.array(F)
                #print(iter, F/best_f, F)
                data[f"{problem}_{dim}"] ={"iter": iter, 
                                           "F/best_f":F/abs(best_f) }

    return data


def plot_optimization_path(ax, problem,dim, folder_name, end_iter=40, alpha = 0.2):
    data = get_optimization_history(folder_name)
    model = folder_name.split("-")[0].split("_")[-1]
    assert isinstance(dim, int)
    assert isinstance(problem, int)
    problem_dim = f"f{problem}_DIM{dim}"
    try:
        data_problem_dim = data[problem_dim]
        x = data_problem_dim["iter"]
        y =data_problem_dim["F/best_f"]
        x.append(end_iter)
        y = np.append(y, y[-1])
        ax.plot(x,y,color = "blue", alpha = alpha, label=f"{model}")
    except:
        print(f"No data in {folder_name},{problem_dim}")

def plot_multiple_paths_for_model(problem,dim, folder_names= None,search_name = None):
    if folder_names is None:
        folder_names = get_folders_of_similar_BO_runs(search_name)
    fig, ax = plt.subplots()
    for folder_name in folder_names:
        plot_optimization_path(ax,problem,dim,folder_name)
    ax.set_yscale('log')
    problem_dim = f"f{problem}_DIM{dim}"
    ax.set_title(f"{problem_dim}")
    ax.set_xlabel("Budget")
    ax.set_ylabel(r"$\frac{f(x_{best})-f^*}{f^*}$")
    ax.set_ylabel("Relative error")
    #plt.legend()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()


import pandas as pd
def plot_means(ax,data_list,name_list, search_name = "Process", color = "blue", modelname = ""):
    i = 0
    data_df = pd.DataFrame(index = list(range(10,300)))
    for data, model_name in zip(data_list, name_list):
        if "Process" in model_name:
            data_series = pd.Series(data=data["mean_rel_error"], index=data["n_train_list"])
            data_df[model_name+f"{i}"]= data_series
            i += 1
    data_df = data_df.ffill()
    data_df.plot(ax=ax,alpha = 0.1, color=color)
    means = data_df.mean(axis=1)
    ax.plot(means.index,means.values,lw=3, color = color, label=f"{modelname}")
    colnames = data_df.columns
    return colnames

def get_folders_of_similar_BO_runs(search_name):
    sub_folders = [name for name in os.listdir(main_data_folder) if os.path.isdir(os.path.join(main_data_folder, name))]
    folder_names = []
    for folder in sub_folders:
        if search_name in folder:
            folder_names.append(folder)
    return folder_names

def plot_optimization_paths(problem,dim):
    fig, ax = plt.subplots()
    problem_dim = f"f{problem}_dim_{dim}"
    ALLdata = get_data2(problem_dim, use_exact_name=True, data_folder ="coco_reg_data")
    data_list,name_list, *_ = ALLdata
    folder_names_BOHAMIANN = get_folders_of_similar_BO_runs("BOH")
    folder_names_GP = get_folders_of_similar_BO_runs("Process")
    folder_names_RandomSearch = get_folders_of_similar_BO_runs("emp")
    folder_names_BNN = get_folders_of_similar_BO_runs("numpyro")
    folder_names_KernelEstiamtor = get_folders_of_similar_BO_runs("Naive")
    all_folders = folder_names_BOHAMIANN+folder_names_GP+folder_names_RandomSearch
    all_folders += folder_names_BNN+folder_names_KernelEstiamtor

    legend_names = plot_means(ax,data_list,name_list, search_name = "BOH", color = "blue", modelname = "BOHAMIANN")
    legend_names += plot_means(ax,data_list,name_list, search_name = "Process", color = "green", modelname = "Gaussian Process")
    legend_names += plot_means(ax,data_list,name_list, search_name = "emp", color = "yellow", modelname = "Random Search")
    legend_names += plot_means(ax,data_list,name_list, search_name = "Naive", color = "red", modelname = "Naive GMR")

    all_folders = legend_names
    
    problem_dim = f"f{problem}_DIM{dim}"
    ax.set_yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for name in all_folders:
        try:
            by_label.pop(name)
        except:
            print(f"No data for {name}")
    plt.legend(by_label.values(), by_label.keys())
    ax.set_title(problem_dim)
    plt.show()

import matplotlib.pyplot as plt
if __name__ == "__main__":
    problem = 1
    dim = 2
    plot_optimization_paths(problem, dim)
    # fig, ax = plt.subplots()
    # problem_dim = f"f{problem}_dim_{dim}"
    # ALLdata = get_data2(problem_dim, use_exact_name=True, data_folder ="coco_reg_data")
    # data_list,name_list, problem_name, file_path_list, file_path_list2 = ALLdata
    # i = 0
    # data_df = pd.DataFrame(index = list(range(10,300)))
    # for data, model_name in zip(data_list, name_list):
    #     if "Process" in model_name:
    #         data_series = pd.Series(data=data["mean_rel_error"], index=data["n_train_list"])
    #         data_df[model_name+f"{i}"]= data_series
    #         i += 1
    # data_df = data_df.ffill()
    # data_df.plot(ax=ax,alpha = 0.1, color="red")
    # means = data_df.mean(axis=1)
    # ax.plot(means.index,means.values,lw=3, color = "blue", label=f"HEJ")
    # colnames = data_df.columns
    
    
    # ax.set_yscale('log')
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = dict(zip(labels, handles))
    # for name in colnames:
    #     try:
    #         by_label.pop(name)
    #     except:
    #         print(f"No data for {name}")
    # plt.legend(by_label.values(), by_label.keys())
    # ax.set_title(problem_dim)
    # plt.show()