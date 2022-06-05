from src.optimization.bayesopt_solver import PlotBayesOpt1D
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.SPN_regression2 import SumProductNetworkRegression

from src.benchmarks.go_benchmark_functions.go_funcs_S import Step
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass
from src.benchmarks.custom_test_functions.problems import Step,SimonsTest,SimonsTestStep, SimonsTest0,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction
import numpy as np
import os

## INPUT ##
acquisition = "EI" #LCB, EI
n_init_samples = 3
samples = 30

## main ##
#reg_model = GaussianProcess_sklearn()
#reg_model = NaiveGMRegression(optimize=True, opt_n_iter=10,opt_cv=10)
reg_model = SumProductNetworkRegression()
# reg_model = SumProductNetworkRegression(optimize=False,
#                         alpha0_x=6.188900996704582, alpha0_y=7.544489215988586, 
#                         beta0_x=0.7, beta0_y=0.14627658075704839, 
#                         train_epochs=1000)

#reg_model = BOHAMIANN()
problem_sklearn = SimonsTestStep(dimensions=1)
plot_BO = PlotBayesOpt1D(reg_model, problem_sklearn, acquisition=acquisition,budget=n_init_samples+samples, n_init_samples=n_init_samples,disp=True)
plot_BO.beta = 2.5

#np.random.seed(2)
extra_name = "score_MAE"
fig_name = f"{plot_BO.problem_name[:20]}_{plot_BO.model.name[:10]}_{acquisition}_{extra_name}"
path = f"/home/simon/Documents/MasterThesis/master-thesis/bayes_opt_experiments/1D_figures/{fig_name}/"
path = path.replace(" ","_")
pathnumber = 0
while True:
    try:
        if pathnumber>0:
            path_more_runs = path+f"{pathnumber}/"
        else:
            path_more_runs = path
        os.mkdir(path_more_runs)
        break
    except:
        #a = input("folder already exists, ok? (y=yes) ")    
        pathnumber += 1
plot_BO.optimize(path_more_runs)