
from cProfile import label
import os
from tkinter import Y
import numpy as np
import pandas as pd

main_data_folder = "/home/simon/Documents/MasterThesis/master-thesis/exdata"

folder_name = "/home/simon/Documents/MasterThesis/master-thesis/exdata/BO_40_empirical_mean_and_std_regression-002"
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

def get_problem_name(file):
    problem = file[:5]
    if problem == "Test3" or problem == "Test4":
        if file[:6] == "Test3c":
            problem = "Test3c"
        elif file[:6] == "Test4c":
            problem = "Test4c"
        else:
           problem = ".." 
    return problem
    
def get_optimization_history_TESTs(sklearn_folder, model="BNN", extra= ""):
    data = dict()
    #sklearn_folder = f"{main_data_folder}/{sklearn_folder}"
    sklearn_folder = f"{sklearn_folder}"

    for root, dirs, files in os.walk(sklearn_folder):
        for file in files:
            if model not in file or not file.endswith(".txt"):
                continue
                #print(model,file)
            if file.endswith(".txt"):
                problem_name = get_problem_name(file)
                #seed = root.split("_")[-1] #HACK
                seed = root.split("_")[-2]
                dat = np.loadtxt(root+"/"+file)
                y_data = dat[:,1]
                y_min = 1000000.
                Y = []
                for y in y_data:

                    if y<y_min:
                        y_min = y
                    Y.append(y_min)
                name = problem_name+"_"+seed
                data[name] ={"iter": list(range(1,len(y_data)+1)), "Y": Y }
    
    return data

def get_mean_of_data(data, problem="Test3"):
    data_df = pd.DataFrame(index = list(range(1,36)))
    for problem_name,dat in data.items():
        if problem == problem_name.split("_")[0]:
            data_series = pd.Series(data=dat["Y"], index=dat["iter"])
            data_df[problem_name]= data_series
    seeds = [c.split("_")[-1] for c in data_df.columns]
    return data_df, seeds




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


def plot_means(ax,problem,dim, folder_names=None, search_name = None, color = "blue", modelname = ""):
    if folder_names is None:
        folder_names = get_folders_of_similar_BO_runs(search_name)
    problem_dim = f"f{problem}_DIM{dim}"
    data_list = []
    data_df = pd.DataFrame(index = list(range(1,31)))
    for folder_name in folder_names:
        try:
            data = get_optimization_history(folder_name)[problem_dim]
            data_series = pd.Series(data=data["F/best_f"], index=data["iter"])
            data_df[folder_name]= data_series
        except:
            print(f"No data in {folder_name},{problem_dim}")
        #data_list.append(data)
    try:
        data_df = data_df.ffill()
        #data_df.plot(ax=ax,alpha = 0.4, color=color)#,  legend = False)
        means = data_df.mean(axis=1)
        ax.plot(means.index,means.values,lw=3, color = color, label=f"{modelname}")
        return data_df.columns
    except:
        print(f"couldn't get data from {modelname} for {problem_dim}")

def plot_means_TESTs(ax,data_model,problem="Test3", color = "blue", modelname = "", seeds_list = ["1", "2"]):
    data_df, seeds = get_mean_of_data(data_model, problem=problem)

    flitered_columns = [col.split("_")[-1] in seeds_list for col in data_df.columns ]
    data_df = data_df.iloc[4:,flitered_columns]
    print(modelname, data_df.shape, seeds)
    try:
        #data_df = data_df.ffill()
        #data_df.plot(ax=ax,alpha = 0.01, color=color)#,  legend = False)
        means = data_df.mean(axis=1)
        ax.plot(means.index,means.values,lw=3, color = color, label=f"{modelname}")
    except:
        print(f"couldn't get data from {modelname} for {problem}")


def get_folders_of_similar_BO_runs(search_name):
    sub_folders = [name for name in os.listdir(main_data_folder) if os.path.isdir(os.path.join(main_data_folder, name))]
    folder_names = []
    for folder in sub_folders:
        if search_name in folder:
            folder_names.append(folder)
    return folder_names

def plot_optimization_paths(ax, problem,dim):
    folder_names_BOHAMIANN = get_folders_of_similar_BO_runs("BOH")
    folder_names_SPN = get_folders_of_similar_BO_runs("SPN")
    folder_names_GMR = get_folders_of_similar_BO_runs("GMR")
    folder_names_GP = get_folders_of_similar_BO_runs("GP")
    folder_names_RandomSearch = get_folders_of_similar_BO_runs("emp")
    folder_names_BNN = get_folders_of_similar_BO_runs("BNN")
    folder_names_KDE = get_folders_of_similar_BO_runs("KDE")
    all_folders = folder_names_BOHAMIANN+folder_names_GP+folder_names_RandomSearch
    all_folders += folder_names_BNN+folder_names_KDE+folder_names_SPN+folder_names_GMR

    # seeds_list = set([f"{x}" for x in range(25)])
    # for data in [data_BNN,data_GP,data_BOHAMIANN, data_KDE,data_GMR, data_SPN,data_emp]:
    #     _, seeds = get_mean_of_data(data, problem=problem)
    #     seeds_list = seeds_list & set(seeds)


    hej = plot_means(ax,problem,dim, folder_names_GP, color = "red", modelname = "GP")
    plot_means(ax,problem,dim,folder_names_BNN, color = "orange", modelname = "BNN")
    plot_means(ax,problem,dim,folder_names_BOHAMIANN, color="gold", modelname = "BOHAMIANN")
    plot_means(ax,problem,dim,folder_names_RandomSearch, color = "black", modelname = "Random Search")
    plot_means(ax,problem,dim,folder_names_KDE, color = "blue", modelname = "KDE")
    plot_means(ax,problem,dim, folder_names_GMR, color = "cyan", modelname = "GMR")
    plot_means(ax,problem,dim,folder_names_SPN,color = "purple" ,modelname = "SPN")

    # plot_means_TESTs(ax,data_GP,problem, color = "red", modelname = "GP", seeds_list =seeds_list)
    # plot_means_TESTs(ax,data_BNN,problem,color = "orange", modelname = "BNN", seeds_list =seeds_list)
    # plot_means_TESTs(ax,data_BOHAMIANN,problem, color = "gold", modelname = "BOHAMIANN", seeds_list =seeds_list)
    # plot_means_TESTs(ax,data_emp,problem, color = "black", modelname = "Random Search", seeds_list =seeds_list)
    # plot_means_TESTs(ax,data_KDE,problem, color = "blue", modelname = "KDE", seeds_list =seeds_list)
    # plot_means_TESTs(ax,data_GMR,problem, color = "cyan", modelname = "GMR", seeds_list =seeds_list)
    # plot_means_TESTs(ax,data_SPN,problem,color = "purple" ,modelname = "SPN", seeds_list =seeds_list)


    problem_dim = f"f{problem}_DIM{dim}"
    ax.set_yscale('log')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    for name in all_folders:
        try:
            by_label.pop(name)
        except:
            print(f"No data for {name}")
    ax.legend(by_label.values(), by_label.keys())
    ax.set_title(problem_dim)
    ax.set_xlabel("Budget")
    ax.set_ylabel("rel error")

def plot_optimization_paths_TESTs(ax, problem, folder):
    data_BNN = get_optimization_history_TESTs(folder, model = "BNN")
    data_GP = get_optimization_history_TESTs(folder, model = "GP")
    data_BOHAMIANN = get_optimization_history_TESTs(folder, model = "BOHAMIANN")
    
    data_KDE = get_optimization_history_TESTs(folder, model = "KDE", extra="")
    data_GMR = get_optimization_history_TESTs(folder, model = "GMR", extra="")
    data_SPN = get_optimization_history_TESTs(folder, model = "SPN", extra="")
    data_emp = get_optimization_history_TESTs(folder, model = "emp", extra="")
    # data_KDE = get_optimization_history_TESTs(folder, model = "KDE", extra="sig10_correct")
    # data_GMR = get_optimization_history_TESTs(folder, model = "GMR", extra="sig10_correct")
    # data_SPN = get_optimization_history_TESTs(folder, model = "SPN", extra="sig10_correct")

    seeds_list = set([f"{x}" for x in range(25)])
    for data in [data_BNN,data_GP,data_BOHAMIANN, data_KDE,data_GMR, data_SPN,data_emp]:
        _, seeds = get_mean_of_data(data, problem=problem)
        seeds_list = seeds_list & set(seeds)
    
    plot_means_TESTs(ax,data_GP,problem, color = "red", modelname = "GP", seeds_list =seeds_list)
    plot_means_TESTs(ax,data_BNN,problem,color = "orange", modelname = "BNN", seeds_list =seeds_list)
    plot_means_TESTs(ax,data_BOHAMIANN,problem, color = "gold", modelname = "BOHAMIANN", seeds_list =seeds_list)
    plot_means_TESTs(ax,data_emp,problem, color = "black", modelname = "Random Search", seeds_list =seeds_list)
    plot_means_TESTs(ax,data_KDE,problem, color = "blue", modelname = "KDE", seeds_list =seeds_list)
    plot_means_TESTs(ax,data_GMR,problem, color = "cyan", modelname = "GMR", seeds_list =seeds_list)
    plot_means_TESTs(ax,data_SPN,problem,color = "purple" ,modelname = "SPN", seeds_list =seeds_list)

    # lesearch_name = "GP",use_pred_mass=False, color = "red", modelname = "GP")
    # lesearch_name = "BNN",use_pred_mass=False, color = "orange", modelname = "BNN")
    # lesearch_name = "BOH",use_pred_mass=False, color = "gold", modelname = "BOHAMIANN")
    # lesearch_name = "emp", color = "black", modelname = "Emperical mean and std")
    # lesearch_name = "KDE", color = "blue", modelname = "KDE")
    # lesearch_name = "GMR", color = "cyan", modelname = "GMR")
    # lesearch_name = "SPN", color = "purple", modelname = "SPN")
    plt.grid()
    ax.legend()
    ax.set_title(problem)
    ax.set_xlabel("Budget")
    ax.set_ylabel("y_min")




import matplotlib.pyplot as plt
if __name__ == "__main__":
    
    plot_folder = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Figures/results_baysopt"
    # sklearn_folder = "/home/simon/Documents/MasterThesis/master-thesis/bayes_opt_experiments/1D_figures_cluster"
    # #plot_optimization_path(ax,1,2,folder_name)
    # for problem in ["Test1","Test2","Test3c","Test4c"]:
    #     fig, ax = plt.subplots()
    #     plot_optimization_paths_TESTs(ax, problem, sklearn_folder)
    #     #ax.set_yscale("log")
    #     ax.set_xlim(5,35)
    #     plt.savefig(plot_folder+"/"+problem+".pdf")
    
    # plt.show()
    
    
    ## COCO ##
    
    for problem in list(range(3,25,6)):
        #fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="all")
        #for ax,dim in zip([ax1,ax2,ax3,ax4],[3,5,10,2]):
        for dim in [2,3,5,10,20]:
            fig, ax = plt.subplots()
            plot_optimization_paths(ax,problem,dim)
            plt.show()