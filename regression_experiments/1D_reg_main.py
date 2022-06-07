from src.regression_validation.reg_validation import PlotReg1D_mixturemodel
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.SPN_regression2 import SumProductNetworkRegression

from src.benchmarks.go_benchmark_functions.go_funcs_S import Step
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass
from src.benchmarks.custom_test_functions.problems import Step,SimonsTest,SimonsTestStep,SimonsTest2, SimonsTest0,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction
from src.benchmarks.custom_test_functions.problems import SimonsTest2_probibalistic, Step_random
import numpy as np
import os

## INPUT ##
samples = 20

## main ##
#reg_model = NumpyroNeuralNetwork()
reg_model = NaiveGMRegression(optimize=False,Ndx=1e-6, opt_n_iter=25,opt_cv=10)
#reg_model = SumProductNetworkRegression(prior_settings={ "Ndx": 1e-2,"sig_prior": 1.2 })
# reg_model = SumProductNetworkRegression(optimize=False,
#                         alpha0_x=6.188900996704582, alpha0_y=7.544489215988586, 
#                         beta0_x=0.7, beta0_y=0.14627658075704839, 
#                         train_epochs=1000)

#reg_model = BOHAMIANN()
#problem_sklearn = SimonsTest2_probibalistic(dimensions=1)
#problem_sklearn = SimonsTestStep(dimensions=1)
problem_sklearn = SimonsTest2(dimensions=1)
#problem_sklearn = Step_random(dimensions=1)
plot_reg = PlotReg1D_mixturemodel(reg_model, problem_sklearn, disp=True)
plot_reg(samples, show_name=True)
