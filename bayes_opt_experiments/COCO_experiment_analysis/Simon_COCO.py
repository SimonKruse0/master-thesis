from __future__ import division, print_function
import cocoex, cocopp  # experimentation and post-processing modules
import scipy.optimize  # to define the solver to be benchmarked
import numpy as np
import os, webbrowser  # to show post-processed results in the browser
from src.optimization.bayesopt_solver import BayesOptSolver_coco
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.SPN_regression2 import SumProductNetworkRegression
import random

reg_models = [MeanRegression(), 
            GaussianProcess_GPy(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=500,num_samples=500,
                                num_chains=4),
            NaiveGMRegression(optimize=True), 
            SumProductNetworkRegression(optimize=True),
            GMRegression(optimize=True)]

# def fmin(reg_model,problem, budget, seed_number):
#     np.random.seed(seed_number)
#     BO = BayesOptSolver_coco(reg_model,problem, budget=budget , disp=True)
#     return BO()

budget_multiplier = 35  # increase to 10, 100, ...
suite_name = "bbob"
# for reg_model in reg_models:

#     name = reg_model.name.replace(" ", "_")
#     ### input
#     suite_name = "bbob"
#     output_folder = f"BO_25_june_all_{budget_multiplier}_{name}"
#     #fmin = scipy.optimize.fmin


#     ### prepare
#     suite = cocoex.Suite(suite_name, "", "dimensions:2,3,5,10,20 instance_indices:2")
#     observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
#     minimal_print = cocoex.utilities.MiniPrint()

#     ### go
#     for problem in suite:  # this loop will take several minutes or longer
#         name_problem = problem.name.split(" ")[3]
#         # if name_problem != "f1":
#         #     continue
#         if (int(name_problem[1:])-1)%6 != 2:
#             continue

#         problem.observe_with(observer)  # generates the data for cocopp post-processing
#         #fmin(problem, problem.dimension * budget_multiplier)  # here we assume that `fmin` evaluates the final/returned solution
#         fmin(reg_model,problem, budget_multiplier, 42)  # here we assume that `fmin` evaluates the final/returned solution
#         minimal_print(problem, final=problem.index == len(suite) - 1)

#run_name = datetime.today().strftime('%m%d_%H%M')
run_name = "COCO_run2"
dirname=os.path.dirname
path = os.path.join(dirname(dirname(dirname(os.path.abspath(__file__)))),f"data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

### prepare
suite = cocoex.Suite(suite_name, "", "dimensions:2,3,5,10,20 instance_indices:1")
#observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
# minimal_print = cocoex.utilities.MiniPrint()

minimal_print = cocoex.utilities.MiniPrint()
K  =2
# for K in [2,1,3,4,5,0]:
for problemnumber in range(2,200,6):
    for random_seed in range(20): #10 runs
        try:
            path2 = f"{path}/seed_{random_seed}"
            os.mkdir(path2)
        except:
            pass
        for reg_model in reg_models:
            try:
                reg_name = reg_model.name[:6]
                path3 = f"{path2}/{reg_name}/"
                os.mkdir(path3)
            except:
                pass
            output_folder = f"BOO2_{budget_multiplier}_{reg_name}_{random_seed}"
            observer = cocoex.Observer(suite_name, "result_folder: " + output_folder)
            for id, problem in enumerate(suite):
                if id != problemnumber:
                    continue
                problem.observe_with(observer) 
                name_problem = problem.name.split(" ")[3]
                dim = problem.name.split(" ")[-1]
                print(name_problem)
                # if (int(name_problem[1:])-1)%6 != K: #ONLY 3, 9,18,24
                #     continue
                try:
                    path4 = f"{path3}/{name_problem}-{dim}"
                    os.mkdir(path4)
                except:
                    continue

                np.random.seed(random_seed)
                BO = BayesOptSolver_coco(reg_model,problem, budget=budget_multiplier, disp=True)
                BO()
                minimal_print(problem, final=problem.index == len(suite) - 1)
                #problem.free()
