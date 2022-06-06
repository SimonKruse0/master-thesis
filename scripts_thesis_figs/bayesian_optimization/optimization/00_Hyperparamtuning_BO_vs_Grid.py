# multimodal test function
from unittest import case
from numpy import arange
from numpy import meshgrid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.optimization.bayesian_optimization import BayesianOptimization
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.benchmarks.custom_test_functions.problems import general_setup

# objective function
def objective(z):
    try:
        x = z[:,0]
        y = z[:,1]
    except:
        x = z[0]
        y = z[1]
    return (x**2 + y - 11)**2 + (x + y**2 -7)**2
# define range for input
r_max = 5.0
plot_navigator = 2 #1,2,3
n_points=23
if plot_navigator == 1:
    problem = general_setup(bounds=(0,r_max),objective_function=objective)   
    X_init = np.array([[1,2], [2,3],[4.9,2]])
    Y_init = problem.fun(X_init)[:,None]
    BO = BayesianOptimization(problem,GaussianProcess_sklearn(), X_init, Y_init )
    BO.optimize(num_steps=n_points-3,plot_steps=False, type="grid")
    hist = BO.get_optimization_hist()

    # create a mesh from the axis
    xaxis = arange(0, r_max, 0.01)
    yaxis = arange(0, r_max, 0.01)
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    results,_ = BO.predict(np.array([x.flatten(), y.flatten()]).T)
    xs,ys,zs = hist[0][:,0], hist[0][:,1], hist[1][:,0]
elif plot_navigator == 2:
    xaxis = arange(0, r_max, .5)
    yaxis = arange(0, r_max, 0.5)
    x, y = meshgrid(xaxis, yaxis)

    results = objective(np.array([x.flatten(), y.flatten()]).T)
    results[:-n_points] = np.nan
    xs,ys,zs = x.flatten(), y.flatten(), results.flatten()
elif plot_navigator == 3:
    xaxis = arange(0, r_max, 0.01)
    yaxis = arange(0, r_max, 0.01)
    x, y = meshgrid(xaxis, yaxis)

    results = objective(np.array([x.flatten(), y.flatten()]).T)
    xs,ys,zs = x.flatten(), y.flatten(), results.flatten()

results = results.reshape(x.shape)


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, results, cmap='jet')

ax.scatter(xs,ys,zs, color="tab:orange")

ax.set_xlabel('sigma')
ax.set_ylabel('lambda')
ax.set_zlabel('Prediction error')
# show the plot
#plt.show()
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/BO_vs_Grid{plot_navigator}.eps', bbox_inches='tight',format='eps')


# #from hyperopt import fmin, hp, space_eval, tpe, STATUS_OK, Trials
# #from hyperopt.pyll import scope, stochastic
# from plotly import express as px
# from plotly import graph_objects as go
# from plotly import offline as pyo
# from sklearn.datasets import load_boston
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.metrics import make_scorer, mean_squared_error
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.utils import check_random_state

# MEDIAN_HOME_VALUE = "median_home_value"

# # Load the boston dataset using sklearn's helper function
# boston_dataset = load_boston()
# # Convert the data into a Pandas dataframe
# data = np.concatenate(
#     [boston_dataset["data"], boston_dataset["target"][..., np.newaxis]],
#     axis=1,
# )
# features, target = boston_dataset["feature_names"], MEDIAN_HOME_VALUE
# columns = np.concatenate([features, [target]])
# boston_dataset_df = pd.DataFrame(data, columns=columns)

# model = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=100)
# #100-150 n_est
# #0.01-0.15 LR

# model.fit(X)