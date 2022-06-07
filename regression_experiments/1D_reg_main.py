from src.regression_validation.reg_validation import PlotReg1D_mixturemodel
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.SPN_regression2 import SumProductNetworkRegression

from src.benchmarks.go_benchmark_functions.go_funcs_S import Step
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass
from src.benchmarks.custom_test_functions.problems import Step,SimonsTest,SimonsTestStep,SimonsTest2, SimonsTest0,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction
from src.benchmarks.custom_test_functions.problems import SimonsTest2_probibalistic, Step_random, SimonsTest2_probibalistic2
import numpy as np
import os

## INPUT ##
samples = 200

## main ##
reg_models = [NaiveGMRegression(optimize=True), 
            GaussianProcess_sklearn(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=200,num_samples=200,
                                num_chains=4, num_keep_samples=200, 
                                extra_name="200-200"),
            SumProductNetworkRegression(optimize=False)]
# for problem_sklearn in [SimonsTest2_probibalistic(dimensions=1), SimonsTest2(dimensions=1)]:
#problem_sklearn = SimonsTestStep(dimensions=1)
#problem_sklearn = SimonsTest2(dimensions=1)
problem_sklearn = SimonsTest2_probibalistic2(dimensions=1)

for reg_model in reg_models:
    for samples in [10,20,40]:
        plot_reg = PlotReg1D_mixturemodel(reg_model, problem_sklearn, disp=True)
        plot_reg(samples,path= "master-thesis/regression_experiments/1D_reg_plots/2/",show_name=True)
    