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

def plot2D(RV:RegressionTest,n_train, output_path):

    DATA =  RV.data_generator(n_train, 0)
    X_train = DATA[2] 
    Y_train = DATA[3]

    RV.fit(X_train,Y_train)

    f, (ax) = plt.subplots(1, 1, sharey=True,  sharex=True)

    xbounds = (-5, 5) #improvmenet
    ybounds = (-5, 5) #variance
    x1_grid = np.linspace(*xbounds, 100, dtype=np.float)
    x2_grid = np.linspace(*ybounds, 90,dtype=np.float)

    x1,x2 = np.meshgrid(x1_grid, x2_grid)
    X = np.hstack([x1.flatten()[:,None],x2.flatten()[:,None]])
    mu_pred,sigma_pred = RV.predict(X)
    #y = np.array([RV.obj_fun(x) for x in X])
    mu_pred = mu_pred.reshape(x1.shape)
    c = ax.contourf(x1,x2, mu_pred, np.linspace(mu_pred.min(), mu_pred.max(),40), cmap="twilight_shifted")
    cbar = plt.colorbar(c)
    # cbar.set_ticks(list(range(6)))
    # cbar.set_ticklabels(list(range(6)))
    ax.plot(*X_train.T,'*', color='black', alpha=0.5,
                markersize=10, markeredgewidth=0, label="data")
    ax.set_title(r"$\mu_{\mathcal{D}}(x)$")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    #ax.set_ylim(0,10)
    plt.savefig(output_path)

reg_models = [#MeanRegression(), 
            NaiveGMRegression(optimize=True), 
            GaussianProcess_sklearn(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=200,num_samples=200,
                                num_chains=4, num_keep_samples=200, 
                                extra_name="200-200"),
            SumProductNetworkRegression(optimize=True)]
            
random.seed()
random.shuffle(reg_models)
#reg_models = [NaiveGMRegression()]
### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
#n_train_array = [10000]
n_test = 10000

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
    # if int(name_problem[1:]) < 19:
    #     continue
    np.random.seed()
    for random_seed in np.random.randint(99999, size=1):
        for regression_model in reg_models:
            print(regression_model.name, f"{name_problem} in {dim}")
            RV = RegressionTest(regression_model,problem, random_seed)
            output_path=path+f"/{name_problem}_{regression_model.name}"
            plot2D(RV,100, output_path)
            #RV.train_test_loop(n_train_array, n_test, output_path = f"{path}")