# Notes:
# This illustrates how exploitation is really favoured by EI
# it is a problem.

import numpy as np
from src.utils import OptimizationStruct
from src.optimization.bayesian_optimization import BayesianOptimization
from src.optimization.bayesopt_solver import BayesOptSolver_sklearn

from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
#from src.regression_models.stan_neural_network import StanNeuralNetwork
#from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.bohamiann import BOHAMIANN

from src.benchmarks.go_benchmark_functions.go_funcs_S import Step, Schwefel26
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.benchmarks.custom_test_functions.problems import SimonsTest,SimonsTest0,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction

problem = SimonsTest4_cosine_fuction()
problem = SimonsTest()
problem = SimonsTest0()
#problem = Schwefel26(dimensions=1)
#problem = Step(dimensions=1)
np.random.seed(2)

BOHAMIANN_regression = BOHAMIANN(num_warmup= 800, num_samples=800, num_keep_samples= 800)
GP_regression = GaussianProcess_sklearn()
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 2000, num_samples=2000, num_keep_samples= 300, extra_name="2000-2000")
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=200, num_keep_samples= 200, extra_name="200-200")
#BNN_regression = StanNeuralNetwork()
SPN_reg = SumProductNetworkRegression(optimize=True, manipulate_variance=True)
SPN_reg2 = SumProductNetworkRegression(optimize=False, manipulate_variance=True)
GM_reg = NaiveGMRegression(optimize=False, manipulate_variance=False)

regression_models = [GP_regression,NNN_regression, GM_reg,BOHAMIANN_regression ]
regression_models = [GP_regression,GP_regression, GM_reg,GP_regression ]
plt.figure(figsize=(12, 8))
outer_gs = gridspec.GridSpec(2, 2)

assert outer_gs.ncols+outer_gs.nrows == len(regression_models)

BOs = []
for i in range(len(regression_models)):
    BOs.append(BayesOptSolver_sklearn(regression_models[i],problem,acquisition="LCB", budget=20, n_init_samples=5))


import os
fig_name = "4_methods"
#path = f"/home/simon/Documents/MasterThesis/master-thesis/bayes_opt_experiments/figures_{fig_name}"
# path = f"/zhome/17/6/118138/master-thesis/bayes_opt_experiments/figures_{fig_name}"
# try:
#     os.mkdir(path)
# except:
#     a = input("folder already exists, ok? (y=yes) ")
#     assert a == "y"

for iter in range(50):
    for i in range(len(regression_models)):
        BO = BOs[i]
        ax = outer_gs[i]
        BO.optimization_step()
        BO.plot_surrogate_and_expected_improvement(ax, show_name=True)
    plt.show()
    number = f"{iter}".zfill(3)
    #plt.savefig(f"{path}/{fig_name}_{number}.jpg")

x_hist,y_hist = BO.get_optimization_hist()
print(x_hist)