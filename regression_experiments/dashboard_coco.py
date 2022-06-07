import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
from src.regression_validation.analysis_helpers import get_data2, get_names
import numpy as np

data_folder = "sklearn_reg_data" #"coco_reg_data"

app = dash.Dash()

names = [{"label":x, 'value': x} for x in get_names(data_folder=data_folder)]
LETTER = "W"
app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'Regression Analysis', style = {'textAlign':'center',\
                                            'marginTop':10,'marginBottom':10}),
        dcc.RadioItems(
            ['means', 'all data'],
            'means',
            inline=True,
            id ="means"
        ),
        dcc.RadioItems(
            ['mean_abs_error', 'mean_rel_error'],
            'mean_abs_error',
            inline=True,
            id ="rel_abs"
        ),
        dcc.Dropdown( id = 'dropdown',
        options = names,
        value = 'Step2_dim_5'),
        dcc.Graph(id = 'bar_plot'),
        dcc.Graph(id = 'bar_plot2'),
        html.Div(id="link"),
        html.Div(id="link2"),
        #html.A(children='link_replace_letter',href=f"https://infinity77.net/global_optimization/test_functions_nd_{LETTER}.html"),
        html.H2(id = "data_origin", children = 'Data collected from:', style = {"color":"red"}),
        html.H5(id = "data_origin2",style = {'textAlign':'left',\
                                            'marginTop':10,'marginBottom':10})
    ])

    

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
    if "-" in name or "mean" in name:
        return "dash"
    else:
        return "solid"

#data

@app.callback([Output(component_id='bar_plot', component_property= 'figure'),
                Output(component_id='data_origin', component_property = "children"),
                Output(component_id='data_origin2', component_property = "children")],
              [Input(component_id='dropdown', component_property= 'value', 
              ),Input(component_id='means', component_property= 'value',
              ),Input(component_id='rel_abs', component_property= 'value',
              )])
def analysis_regression_performance_plotly(problem, means, rel_abs):
    print_file_paths = True
    data_list,name_list, problem_name, file_path_list, file_path_list2 = get_data2(problem, use_exact_name=True, data_folder=data_folder)

    type = rel_abs
    #type ="mean_abs_pred_error"
    #type ="mean_rel_pred_error"

    data2 = dict()
    data3 = dict()
    name_visted = []

    if means == "means":
        print("bling")
        for data, name in zip(data_list,name_list):
            try:
                len(data[type])
            except:
                continue
            #print(len(data[type]) ,len(data["n_train_list"]))
            if len(data[type]) != 9:
                continue
            if len(data[type])!= len(data["n_train_list"]):
                print("error: not same sizes")
                return
            if name in name_visted:
                data3[name]+=(data[type])
                
            else:
                data2[name] = data["n_train_list"]
                data3[name] = data[type]
                name_visted.append(name)
        
        for name in name_visted:
            tmp = np.atleast_2d(np.array(data3[name]))
            data3[name] = np.mean(tmp, axis=0)
            #print(data3[name])
            #print(name,data3[name])

    else:
        for data, name in zip(data_list,name_list):
            try:
                len(data[type])
            except:
                continue
            if name in name_visted:
                data2[name]+=["None"]
                data3[name]+=["None"]
                data2[name]+=(data["n_train_list"])
                data3[name]+=(data[type])
            else:
                data2[name] = data["n_train_list"]
                data3[name] = data[type]
                name_visted.append(name)

    fig = go.Figure()
    for name in name_visted:
        fig.add_trace(go.Scatter(mode='lines+markers', x=data2[name], y=data3[name], name=name,
                            line=dict(color=color(name),dash = ls(name), width=2),showlegend=True))

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
    if print_file_paths:
        print(file_path_list)
    text = "Data collected from: "
    text += ", ".join(file_path_list2)
    text_raw = " --- ".join(file_path_list)
    return fig, text, text_raw

@app.callback(Output(component_id='bar_plot2', component_property= 'figure'),
              [Input(component_id='dropdown', component_property= 'value', 
              ),Input(component_id='means', component_property= 'value',
              )])
def analysis_regression_performance_plotly(problem,means):
    data_list,name_list, problem_name, *_ = get_data2(problem, use_exact_name=True,data_folder=data_folder)

    type ="mean_pred_likelihod"

    data2 = dict()
    data3 = dict()
    name_visted = []

    if means == "means":

        for data, name in zip(data_list,name_list):
            try:
                data_inst = data["mean_pred_mass"]
                assert data_inst is not None
            except:
                data_inst = data[type]
            if len(data_inst) != 9: #KÆMPE HACK
                print("#KÆMPE HACK")
                continue
            if name in name_visted:
                data3[name]+=(data_inst)
                
            else:
                data2[name] = data["n_train_list"]
                data3[name] = data_inst
                name_visted.append(name)
        
        for name in name_visted:
            tmp = np.atleast_2d(np.array(data3[name]))
            data3[name] = np.mean(tmp, axis=0)
            #print(data3[name])
            #print(name,data3[name])

    else:

        for data, name in zip(data_list,name_list):
            try:
                data_inst = data["mean_pred_mass"]
                assert data_inst is not None
            except:
                data_inst = data[type]

            if name in name_visted:
                data2[name]+=["None"]
                data3[name]+=["None"]
                data2[name]+=(data["n_train_list"])
                data3[name]+=(data_inst)
            else:
                data2[name] = data["n_train_list"]
                data3[name] = data_inst
                name_visted.append(name)

    fig = go.Figure()
    for name in name_visted:
        fig.add_trace(go.Scatter(mode='lines+markers', x=data2[name], y=data3[name], name=name,
                            line=dict(color=color(name),dash = ls(name), width=2),showlegend=True))

    fig.update_layout(#title=problem_name,
                    xaxis_title='n_train_points',
                    yaxis_title=type,
                    margin=go.layout.Margin(
                            l=0, #left margin
                            r=0, #right margin
                            b=0, #bottom margin
                            t=0, #top margin
                        ))

    return fig


@app.callback([Output(component_id='link', component_property= 'children'),
                Output(component_id='link2', component_property= 'children')],
              [Input(component_id='dropdown', component_property= 'value', 
              )])
def functionLINK(name):
    print(name)
    link1 = html.A(children='link_letter',href=f"https://infinity77.net/global_optimization/test_functions_nd_{name[0]}.html")
    fullname = name.split("_")[0]
    link2 = html.A(children='link_function',
    href=f"https://infinity77.net/global_optimization/test_functions_nd_{name[0]}.html#go_benchmark.{fullname}",
    target='_blank')
    return link1,link2

if __name__ == '__main__': 
    app.run_server()