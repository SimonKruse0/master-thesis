from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import seed  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
from src.optimization.bayesopt_solver import BayesOptSolver
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random

reg_models = [MeanRegression(), 
            NaiveGMRegression(), 
            GaussianProcess_sklearn(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=200,num_samples=200,num_chains=4, num_keep_samples=200, extra_name="200-200")
]
random.seed()
random.shuffle(reg_models)
reg_models = [MeanRegression()]
def fmin(problem, budget):
    BO = BayesOptSolver(reg_model,problem, budget, disp=False)
    return BO()

budget_multiplier = 40  # increase to 10, 100, ...
seed()
for reg_model in reg_models:

    name = reg_model.name.replace(" ", "_")
    ### input
    suite_name = "bbob"
    output_folder = f"BO_test_{budget_multiplier}_{name}"
    #fmin = scipy.optimize.fmin


    ### prepare
    suite = cocoex.Suite(suite_name, "", "dimensions:2 instance_indices:2")
    observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        name_problem = problem.name.split(" ")[3]
        if name_problem != "f1":
            continue
        problem.observe_with(observer)  # generates the data for cocopp post-processing
        #fmin(problem, problem.dimension * budget_multiplier)  # here we assume that `fmin` evaluates the final/returned solution
        fmin(problem, budget_multiplier)  # here we assume that `fmin` evaluates the final/returned solution
        minimal_print(problem, final=problem.index == len(suite) - 1)

    ### post-process data
    #cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc


    #webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
