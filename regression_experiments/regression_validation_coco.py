import numpy as np


import random
from src.regression_validation.reg_validation import RegressionTest

import cocoex #the 24 functions i defined from here
import os
from datetime import datetime

from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random

reg_models = [MeanRegression(), 
            GaussianProcess_GPy(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=500,num_samples=500,
                                num_chains=4),
            NaiveGMRegression(optimize=True), 
            SumProductNetworkRegression(optimize=True),
            GMRegression(optimize=True)]

### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
#n_train_array = [10000]
n_test = 10000

run_name = datetime.today().strftime('%m%d_%H%M')
run_name = datetime.today().strftime('%m%d_%H')
dirname=os.path.dirname
path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"coco_reg_data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

suite = cocoex.Suite("bbob", "", "dimensions:2,3,5,10,20 instance_indices:1")

for K in [2,1,3,4,5,0]:
    for problem in suite:
        name_problem = problem.name.split(" ")[3]
        dim = problem.name.split(" ")[-1]
        print(name_problem)
        if (int(name_problem[1:])-1)%6 != K: #ONLY 3, 9,18,24
            continue
        try:
            path2 = f"{path}/{name_problem}-{dim}"
            os.mkdir(path2)
        except:
            pass

        for random_seed in range(10): #10 runs
            try:
                path3 = f"{path2}/seed_{random_seed}"
                os.mkdir(path3)
            except:
                pass

            for regression_model in reg_models:
                RV = RegressionTest(regression_model,problem, random_seed)
                try:
                    path4 = f"{path3}/{RV.model.name[:6]}/"
                    os.mkdir(path4)
                except:
                    continue
                RV.train_test_loop(n_train_array, n_test, output_path = f"{path4}")

