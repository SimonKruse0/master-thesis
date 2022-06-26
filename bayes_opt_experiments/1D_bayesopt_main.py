from src.optimization.bayesopt_solver import PlotBayesOpt1D
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.mean_regression import MeanRegression

from src.benchmarks.custom_test_functions.problems import Test1,Test2, Test3, Test4b,Test4c,Test3c, Test3b
import numpy as np
import os

## INPUT ##
acquisition = "EI" #LCB, EI
n_init_samples = 5
samples = 30

## main ##
reg_models = [GMRegression(optimize=True),
            NaiveGMRegression(optimize=True), 
            GaussianProcess_GPy(), 
            BOHAMIANN(num_keep_samples=100), 
            NumpyroNeuralNetwork(hidden_units = 50,num_warmup=300,num_samples=300,
                                num_chains=4,alpha=1000),
            SumProductNetworkRegression(optimize=True, sig_prior = 10)]

reg_models = [MeanRegression()]
# import random
# random.seed()
# random.shuffle(reg_models)
#problems =  [Test1(),Test2(),Test4b(),Test3b()]
problems =  [Test1(),Test2(),Test4c(),Test3c()]
DATA_path = "/zhome/17/6/118138/master-thesis/bayes_opt_experiments/1D_figures_cluster"
DATA_path = "/home/simon/Documents/MasterThesis/master-thesis/bayes_opt_experiments/1D_figures_cluster"

for seed in range(20):
    for reg_model in reg_models:
        for problem in problems:
            np.random.seed(seed)
            plot_BO = PlotBayesOpt1D(reg_model, problem, acquisition=acquisition,
            budget=n_init_samples+samples, n_init_samples=n_init_samples,disp=True, show_name=False)
            #plot_BO.beta = 2.5

            extra_name = f"{seed}_2406"
            fig_name = f"{plot_BO.problem_name[:20]}_{plot_BO.model.name[:10]}_{acquisition}_{extra_name}"
            path = f"{DATA_path}/{fig_name}/"
            path = path.replace(" ","_")
            #pathnumber = 0
            #while True:
            try:
                # if pathnumber>0:
                #     path_more_runs = path+f"{pathnumber}/"
                # else:
                #     path_more_runs = path
                os.mkdir(path)
            except:
                pass
                #continue
                #a = input("folder already exists, ok? (y=yes) ")    
                #pathnumber += 1
            #plot_BO.optimize(path, extension="jpg")
            plot_BO.optimize(path, extension=None)