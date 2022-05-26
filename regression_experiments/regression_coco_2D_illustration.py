import numpy as np

from src.regression_models.SPN_regression2 import SumProductNetworkRegression

import random
from src.regression_validation.reg_validation import RegressionTest

import cocoex #the 24 functions i defined from here
import os
from datetime import datetime

from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot2D(RV:RegressionTest,n_train, output_path, plot_type = 1):

    DATA =  RV.data_generator(n_train, 0)
    X_train = DATA[2] 
    Y_train = DATA[3]

    RV.fit(X_train,Y_train)


    xbounds = (-5, 5) #improvmenet
    ybounds = (-5, 5) #variance
    x1_grid = np.linspace(*xbounds, 100, dtype=np.float)
    x2_grid = np.linspace(*ybounds, 90,dtype=np.float)

    x1,x2 = np.meshgrid(x1_grid, x2_grid)
    X = np.hstack([x1.flatten()[:,None],x2.flatten()[:,None]])

    mu_pred,sigma_pred = RV.predict(X)
    #y = np.array([RV.obj_fun(x) for x in X])
    mu_pred = mu_pred.reshape(x1.shape)
    sigma_pred = sigma_pred.reshape(x1.shape)

    for plot_type in [2,3]:
    
        if plot_type == 1:
            f, (ax,ax2) = plt.subplots(1, 2, sharey=True,  sharex=True)
        if plot_type == 2:
            f, (ax) = plt.subplots(1, 1, sharey=True,  sharex=True)
        if plot_type == 3:
            f, (ax2) = plt.subplots(1, 1, sharey=True,  sharex=True)

        # cbar.set_ticks(list(range(6)))
        # cbar.set_ticklabels(list(range(6)))
        if plot_type == 2 or plot_type == 1:
            c = ax.contourf(x1,x2, mu_pred, np.linspace(mu_pred.min(), mu_pred.max(),40), cmap="twilight_shifted")
            cbar = plt.colorbar(c,ax=ax)
            ax.plot(*X_train.T,'*', color='black', alpha=0.5,
                        markersize=10, markeredgewidth=0, label="data")
            ax.set_title(r"$\mu_{\mathcal{D}}(x)$")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")

        if plot_type == 3 or plot_type == 1:
            #c2 = ax2.contourf(x1,x2, sigma_pred, np.linspace(sigma_pred.min(), sigma_pred.max(),40), cmap="Reds_r")
            c2 = ax2.contourf(x1,x2, sigma_pred, np.linspace(0, sigma_pred.max(),40), cmap="Reds_r")
            cbar2 = plt.colorbar(c2, ax=ax2)
            #ax2.plot(*X_train.T,'*', color='black', alpha=0.5,
            #        markersize=10, markeredgewidth=0, label="data")
            ax2.set_title(r"$\sigma_{\mathcal{D}}(x)$")
            ax2.set_xlabel(r"$x_1$")

        #ax.set_ylim(0,10)
        #plt.show()
        plt.savefig(output_path+f"{plot_type}.pdf")

# reg_models = [#MeanRegression(), 
#             NaiveGMRegression(optimize=True), 
#             GaussianProcess_sklearn(), 
#             BOHAMIANN(num_keep_samples=500), 
#             NumpyroNeuralNetwork(num_warmup=200,num_samples=200,
#                                 num_chains=4, num_keep_samples=200, 
#                                 extra_name="200-200"),
#             SumProductNetworkRegression(optimize=True)]
            

reg_models = [SumProductNetworkRegression(optimize=True)]

### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
#n_train_array = [10000]
n_test = 10000

if os.path.expanduser('~') == "/home/simon":
    if input("plot in result folder? y = yes ") != "y":
        run_name = datetime.today().strftime('%m%d_%H%M')
        dirname=os.path.dirname
        path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"coco_reg_data/{run_name}")
        try:
            os.mkdir(path)
        except:
            print(f"Couldn't create {path}")
    else:
        path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Figures/coco_reg"
else:
    run_name = datetime.today().strftime('%m%d_%H%M')
    dirname=os.path.dirname
    path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"coco_reg_data/{run_name}")
    try:
        os.mkdir(path)
    except:
        print(f"Couldn't create {path}")

suite = cocoex.Suite("bbob", "", "dimensions:2 instance_indices:1")

for problem in suite:
    name_problem = problem.name.split(" ")[3]
    dim = problem.name.split(" ")[-1]
    print(name_problem)
    # if int(name_problem[1:]) < 23:
    #     continue
    np.random.seed()
    for random_seed in np.random.randint(99999, size=1):
        for regression_model in reg_models:
            print(regression_model.name, f"{name_problem} in {dim}")
            RV = RegressionTest(regression_model,problem, random_seed)
            output_path=path+f"/{name_problem}_{regression_model.name}"
            plot2D(RV,100, output_path, plot_type=2)
            #RV.train_test_loop(n_train_array, n_test, output_path = f"{path}")