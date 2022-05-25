# Notes:
# This illustrates how exploitation is really favoured by EI
# it is a problem.

from src.optimization.bayesopt_solver import PlotBayesOpt1D
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.benchmarks.go_benchmark_functions.go_funcs_S import Step
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass
from src.benchmarks.custom_test_functions.problems import Step,SimonsTest, SimonsTest0,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction
import numpy as np
import os

## INPUT ##
acquisition = "LCB"
n_init_samples = 3
samples = 30

## main ##
reg_model = GaussianProcess_sklearn()
problem_sklearn = SimonsTest(dimensions=1)
plot_BO = PlotBayesOpt1D(reg_model, problem_sklearn, acquisition=acquisition,budget=n_init_samples+samples, n_init_samples=n_init_samples,disp=True)
plot_BO.beta = 2.5

#np.random.seed(2)
fig_name = f"{plot_BO.problem_name[:20]}_{plot_BO.model.name[:10]}_{acquisition}"
path = f"/home/simon/Documents/MasterThesis/master-thesis/bayes_opt_experiments/1D_figures/{fig_name}/"
try:
    os.mkdir(path)
except:
    a = input("folder already exists, ok? (y=yes) ")    
    assert a == "y"    

plot_BO.optimize(path)