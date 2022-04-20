import dash
import dash_html_components as html
import plotly.graph_objects as go
import dash_core_components as dcc
import plotly.express as px
from dash.dependencies import Input, Output
from src.analysis_helpers import get_data2, get_names



app = dash.Dash()

names = [{"label":x, 'value': x} for x in get_names()]

app.layout = html.Div(id = 'parent', children = [
    html.H1(id = 'H1', children = 'Regression Analysis', style = {'textAlign':'center',\
                                            'marginTop':10,'marginBottom':10}),

        dcc.Dropdown( id = 'dropdown',
        options = names,
        value = 'Step2_dim_5'),
        dcc.Graph(id = 'bar_plot'),
        dcc.Graph(id = 'bar_plot2'),
        html.H2(id = "data_origin", children = 'Data collected from:', style = {"color":"red"}),
        html.H5(id = "data_origin2",style = {'textAlign':'left',\
                                            'marginTop':10,'marginBottom':10})
    ])

    
# @app.callback(Output(component_id='bar_plot', component_property= 'figure'),
#               [Input(component_id='dropdown', component_property= 'value')])
# def graph_update(dropdown_value):
#     print(dropdown_value)
#     fig = go.Figure([go.Scatter(x = df['date'], y = df['{}'.format(dropdown_value)],\
#                      line = dict(color = 'firebrick', width = 4))
#                      ])
    
#     fig.update_layout(title = 'Stock prices over time',
#                       xaxis_title = 'Dates',
#                       yaxis_title = 'Prices'
#                       )
#     return fig  


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
        return "dash"
    else:
        return "solid"

#data

@app.callback([Output(component_id='bar_plot', component_property= 'figure'),
                Output(component_id='data_origin', component_property = "children"),
                Output(component_id='data_origin2', component_property = "children")],
              [Input(component_id='dropdown', component_property= 'value')])
def analysis_regression_performance_plotly(problem):
    print_file_paths = True
    data_list,name_list, problem_name, file_path_list, file_path_list2 = get_data2(problem, use_exact_name=True)

    type ="mean_abs_pred_error"

    data2 = dict()
    data3 = dict()
    name_visted = []
    for data, name in zip(data_list,name_list):
        if name in name_visted:
            data2[name]+=["None"]
            data3[name]+=["None"]
            data2[name]+=(data["n_train_points_list"])
            data3[name]+=(data[type])
        else:
            data2[name] = data["n_train_points_list"]
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
              [Input(component_id='dropdown', component_property= 'value')])
def analysis_regression_performance_plotly(problem):
    data_list,name_list, problem_name, *_ = get_data2(problem, use_exact_name=True)

    type ="mean_uncertainty_quantification"

    data2 = dict()
    data3 = dict()
    name_visted = []
    for data, name in zip(data_list,name_list):
        if name in name_visted:
            data2[name]+=["None"]
            data3[name]+=["None"]
            data2[name]+=(data["n_train_points_list"])
            data3[name]+=(data[type])
        else:
            data2[name] = data["n_train_points_list"]
            data3[name] = data[type]
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


if __name__ == '__main__': 
    app.run_server()