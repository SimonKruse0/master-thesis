import os
import json
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
if __name__ == "__main__":
    from utils import RegressionValidation
else:
    from .utils import RegressionValidation

BASE_DIRECTORY = '/home/simon/Documents/MasterThesis/master-thesis'


def redefine_data_names():
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(BASE_DIRECTORY, "data")):
        for filename in filenames:
            if "noise" in filename:
                file_path = os.path.join(dirpath, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                os.remove(file_path)
                file_path = file_path.replace("noise_", "noise-")
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)



def include_true_values(Problems, min_n_test_points=9999, remove_min_n_test = False):
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(BASE_DIRECTORY, "data")):
        
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            #print(filename)
            flag = 0
            with open(file_path, 'r') as f:
                data = json.load(f)
                if "y_test" in data:
                    continue
                n_test_points = data["n_test_points"]
                if n_test_points < min_n_test_points:
                    print("n_test_points", n_test_points,filename)
                    if remove_min_n_test:
                        print(f"removes: {filename}")
                        os.remove(file_path)
                    continue
                random_seed = int(file_path.split("seed_")[1].split("_")[0])
                problem_name = filename.split("_")[1]
                dim = int(filename.split("_")[3])
                for problem in Problems:
                    if problem_name == type(problem).__name__:
                        if dim == problem.N:
                            RV = RegressionValidation(problem, None, random_seed)
                            RV.data_generator( 0, n_test_points)
                            data["y_test"] = [a.astype(float) for a in RV.test_y]
                            flag = 1

            if flag == 1:
                print(f"redefining: {filename}")
                os.remove(file_path)

                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)

def get_data2(target_name, use_exact_name=False):
    data_list = []
    name_list = []
    file_path_list = set()
    problem_name = None
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(BASE_DIRECTORY, "data")):
        # Opening JSON file
        #print(dirpath, filenames)
        for filename in filenames:
            if target_name not in filename:
                continue
            if (target_name != "_".join(filename.split("_")[1:4])) and use_exact_name:
                continue
            try:
                file_path = os.path.join(dirpath, filename)
                with open(file_path) as json_file:
                    data_list.append(json.load(json_file))
                    name_list.append(filename.split("_")[0])
                    file_path_list.add(dirpath.split("/")[-1])
                    tmp =  " ".join(filename.split("_")[1:4])
                    if problem_name is None:
                        problem_name = tmp
                    else:
                        if tmp != problem_name:
                            print("target", target_name)
                            print("More problems in same target..", problem_name,"and", tmp)
                            return 0
            except:
                print(filename, "could not be read")

    file_path_list2 = [f"{x[2:4]}/{x[:2]} {x[-4:-2]}:{x[-2:]}" for x in file_path_list]
    return data_list, name_list, problem_name, file_path_list, file_path_list2

def get_names():
    names = set()
    for (dirpath, dirnames, filenames) in os.walk(os.path.join(BASE_DIRECTORY, "data")):
        
        # Opening JSON file
        #print(dirpath, filenames)
        for filename in filenames:
            if "noise" in filename:
                filename = filename.replace("noise_", "noise-")
            names.add("_".join(filename.split("_")[1:4]))

    return sorted(names)




def color(name):
    if "Mixture" in name:
        return "orange"
    if "Gaussian" in name:
        return "blue"
    if "numpyro" in name:
        return "red"
    if "BOHAMIANN" in name:
        return "green"
    return "black"

def ls(name):
    if "-" in name:
        if "learn"in name:
            return "dashdot"
        else:
            return "dash"

    else:
        return "solid"

#data
import pandas as pd
def data_to_pandas(data_list, name_list):
    indexes = dict()
    name_visted = []
    df_dict = dict()
    for data, name in zip(data_list,name_list):
        assert data["n_test_points"]>9999 #we only want proper tests 
        if name not in name_visted:
            indexes[name] = set(data["n_train_points_list"])
            name_visted.append(name)
            df_dict[name] = pd.DataFrame(index=indexes[name]).sort_index()
        else:
            if sorted(data["n_train_points_list"]) != sorted(indexes[name]):
                temp_indexes = set(data["n_train_points_list"])
                indexes[name] = indexes[name].union(temp_indexes)
                df_dict[name] = df_dict[name].reindex(indexes[name]).sort_index()
        
        d_temp = dict()
        d_temp["mean_abs_pred_error"] = pd.Series(data["mean_abs_pred_error"], index=data["n_train_points_list"])
        d_temp["mean_uncertainty_quantification"] = pd.Series(data["mean_uncertainty_quantification"], index=data["n_train_points_list"])
        try:
            d_temp["mean_rel_pred_error"] = pd.Series(data["mean_rel_pred_error"], index=data["n_train_points_list"])
        except:
            #print("no rel_pred_error")
            pass
        
        df_new_data = pd.DataFrame(data=d_temp,index=indexes[name])
        df_dict[name] = pd.concat([df_dict[name], df_new_data], axis=1, join="outer")
    return df_dict,name_visted

def all_data_prepros_plotly(data_list, name_list, type= "mean_abs_pred_error"):
    dataX = dict()
    dataY = dict()
    name_visted = []
    for data, name in zip(data_list,name_list):
        if name in name_visted:
            dataX[name]+=["None"]
            dataY[name]+=["None"]
            dataX[name]+=(data["n_train_points_list"])
            dataY[name]+=(data[type])
        else:
            dataX[name] = data["n_train_points_list"]
            dataY[name] = data[type]
            name_visted.append(name)
    return dataX, dataY, name_visted

def mean_prepros_plotly(data_list, name_list, type= "mean_abs_pred_error"):
    dataX = dict()
    dataY = dict()
    df_dict,name_visted = data_to_pandas(data_list, name_list)
    for name in name_visted:
        df = df_dict[name][type].mean(axis=1)
        dataX[name] = df.index.values
        dataY[name] = df.values
    return dataX, dataY, name_visted

def analysis_regression_performance_plotly(problem,type = "mean_abs_pred_error",  means = False, print_file_paths = False):
    data_list,name_list, problem_name, file_path_list, file_path_list2 = get_data2(problem, use_exact_name=True)

    if means:
        dataX, dataY, name_visted = mean_prepros_plotly(data_list, name_list, type= "mean_abs_pred_error")
    else:
        dataX, dataY, name_visted = all_data_prepros_plotly(data_list, name_list, type= "mean_abs_pred_error")

    #fig = go.Figure()
    fig = make_subplots(rows=1, cols=2)
    for name in name_visted:
        fig.add_trace(go.Scatter(mode='lines+markers', x=dataX[name], y=dataY[name], name=name,
                            line=dict(color=color(name),dash = ls(name), width=2),showlegend=True),
                            row=1, col=1
                            )
    if means:
        dataX, dataY, name_visted = mean_prepros_plotly(data_list, name_list, type= "mean_uncertainty_quantification")
    else:
        dataX, dataY, name_visted = all_data_prepros_plotly(data_list, name_list, type= "mean_uncertainty_quantification")

    for name in name_visted:
        fig.add_trace(go.Scatter(mode='lines+markers', x=dataX[name], y=dataY[name], name=name,
                            line=dict(color=color(name),dash = ls(name), width=2),showlegend=False),
                            row=1, col=2
                            )

    fig.update_layout(title=problem_name,
                    #xaxis_title='n_train_points',
                    yaxis_title=type,
                    margin=go.layout.Margin(
                            l=0, #left margin
                            r=0, #right margin
                            b=0, #bottom margin
                            t=30, #top margin
                        )
                    )
    #fig.update_xaxes(visible=False, showticklabels=True)
    text = "Data collected from: "
    text += ", ".join(file_path_list2)
    text_raw = " --- ".join(file_path_list)
    if print_file_paths:
        print(file_path_list)
        print(text, text_raw)
    fig.show() 

def analysis_regression_rel_error_plotly(problem,  means = False):
    print_file_paths = True
    data_list,name_list, problem_name, file_path_list, file_path_list2 = get_data2(problem, use_exact_name=True)

    data2 = dict()
    data3 = dict()
    name_visted = []

    if means:
        for data, name in zip(data_list,name_list):
            if name in name_visted:
                data3[name]+=(get_relative_error(data))
                
            else:
                data2[name] = data["n_train_points_list"]
                data3[name] = get_relative_error(data)
                name_visted.append(name)
        
        for name in name_visted:
            tmp = np.atleast_2d(np.array(data3[name]))
            data3[name] = np.mean(tmp, axis=0)
            #print(data3[name])
            #print(name,data3[name])
    else:
        for data, name in zip(data_list,name_list):
            if name in name_visted:
                data2[name]+=["None"]
                data3[name]+=["None"]
                data2[name]+=(data["n_train_points_list"])
                data3[name]+=(get_relative_error(data))
            else:
                data2[name] = data["n_train_points_list"]
                data3[name] = get_relative_error(data)
                name_visted.append(name)

    fig = go.Figure()
    for name in name_visted:
        fig.add_trace(go.Scatter(mode='lines+markers', x=data2[name], y=data3[name], name=name,
                            line=dict(color=color(name),dash = ls(name), width=2),showlegend=True))

    fig.update_layout(title=problem_name,
                    #xaxis_title='n_train_points',
                    yaxis_title="mean_relative error",
                    margin=go.layout.Margin(
                            l=0, #left margin
                            r=0, #right margin
                            b=0, #bottom margin
                            t=30, #top margin
                        )
                    )
    #fig.update_xaxes(visible=False, showticklabels=True)
    if print_file_paths:
        print(file_path_list)
    text = "Data collected from: "
    text += ", ".join(file_path_list2)
    text_raw = " --- ".join(file_path_list)
    print(text, text_raw)
    fig.show() 

def get_relative_error(data):
    return data["mean_abs_pred_error"]/np.mean(data["mean_abs_pred_error"])
    print_file_paths = True
    data_list,name_list, problem_name, file_path_list, file_path_list2 = get_data2(problem, use_exact_name=True)



print(__name__)
if __name__ == "__main__":
    analysis_regression_performance_plotly("Schwefel26_dim_5",type = "mean_abs_pred_error",  means = False)
    
    # data_list,name_list, problem_name, file_path_list, file_path_list2 = get_data2("Schwefel26_dim_5", use_exact_name=True)
    # data_to_pandas(data_list,name_list)
    #analysis_regression_rel_error_plotly("Rastrigin_dim_5", means=True)
    #redefine_data_names()
    #include_true_values(None)
    pass