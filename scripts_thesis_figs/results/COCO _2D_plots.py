from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
from src.optimization.bayesopt_solver import BayesOptSolver
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random
import numpy as np
import matplotlib.pyplot as plt

reg_models = [MeanRegression()]
random.seed()
random.shuffle(reg_models)

def fmin(problem, budget):
    BO = BayesOptSolver(reg_model,problem, budget, disp=False)
    return BO()

budget_multiplier = 1  # increase to 10, 100, ...

for reg_model in reg_models:
    name = reg_model.name.replace(" ", "_")
    ### input
    suite_name = "bbob"
    output_folder = f"plotting"
    #fmin = scipy.optimize.fmin


    ### prepare
    suite = cocoex.Suite(suite_name, "", "dimensions:2 instance_indices:1")
    #observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
    minimal_print = cocoex.utilities.MiniPrint()

    ### go
    for problem in suite:  # this loop will take several minutes or longer
        name_problem = problem.name.split(" ")[3]
        print(name_problem)
        # if int(name_problem[1:])<= 10:
        #     continue
        x = np.linspace(problem.lower_bounds[0], problem.upper_bounds[0], 300)
        y = np.linspace(problem.lower_bounds[1], problem.upper_bounds[1], 300)
        
        X,Y = np.meshgrid(x,y)
        XY = np.hstack([X.flatten()[:,None], Y.flatten()[:,None]])

        f = np.array([problem(xy) for xy in XY])
        id_min = f.argmin()
        f = f.reshape(X.shape)#-f.min()
        plt.figure()
        plt.contourf(X,Y,f,np.linspace(f.min(), f.max(),40), cmap="twilight_shifted")
        plt.colorbar()
        plt.plot(X.flatten()[id_min],Y.flatten()[id_min], "*r")

        fig_path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Figures/coco"
        plt.savefig(f"{fig_path}/{name_problem}.pdf")
        

        #problem.observe_with(observer)  # generates the data for cocopp post-processing
        #fmin(problem, problem.dimension * budget_multiplier)  # here we assume that `fmin` evaluates the final/returned solution
        #fmin(problem, budget_multiplier)  # here we assume that `fmin` evaluates the final/returned solution
        minimal_print(problem, final=problem.index == len(suite) - 1)

    ### post-process data
    #cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc


    #webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
