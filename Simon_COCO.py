from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
from numpy.random import rand  # for randomised restarts
import os, webbrowser  # to show post-processed results in the browser
from src.optimization.bayesopt_solver import BayesOptSolver_GP
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression

#reg_model = NaiveGMRegression(optimize=False, manipulate_variance=False)
reg_model = MeanRegression()
name = reg_model.name.replace(" ", "_")

### input
suite_name = "bbob"
output_folder = f"BO_{name}"
#fmin = scipy.optimize.fmin
budget_multiplier = 30  # increase to 10, 100, ...


def fmin(problem, budget):
    BO = BayesOptSolver_GP(reg_model,problem, budget)
    return BO()


### prepare
suite = cocoex.Suite(suite_name, "", "instance_indices:1")
observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
minimal_print = cocoex.utilities.MiniPrint()

### go
for problem in suite:  # this loop will take several minutes or longer
    problem.observe_with(observer)  # generates the data for cocopp post-processing
    fmin(problem, problem.dimension * budget_multiplier)  # here we assume that `fmin` evaluates the final/returned solution
    #minimal_print(problem, final=problem.index == len(suite) - 1)

### post-process data
cocopp.main(observer.result_folder)  # re-run folders look like "...-001" etc
webbrowser.open("file://" + os.getcwd() + "/ppdata/index.html")
