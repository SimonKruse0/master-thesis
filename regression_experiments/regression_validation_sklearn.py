import numpy as np

from src.regression_models.SPN_regression2 import SumProductNetworkRegression

import random
from src.regression_validation.reg_validation import RegressionTest_sklearn

from src.benchmarks.custom_test_functions.problems import Test1,Test2,Test3,Test4b,Test3b
import os
from datetime import datetime

from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
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
            
random.seed()
random.shuffle(reg_models)


### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
n_test = 10000

#run_name2 = datetime.today().strftime('%m%d_%H%M')
run_name2 = datetime.today().strftime('%m%d_%H')
dirname=os.path.dirname
path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"data/{run_name2}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")


problems = [Test1(), Test2(), Test4b(), Test3b()]

for random_seed in range(10): #20 runs
    try:
        path2 = f"{path}/seed_{random_seed}"
        os.mkdir(path2)
    except:
        pass
    for problem in problems:
        try:
            path3 = f"{path2}/{type(problem).__name__}"
            os.mkdir(path3)
        except:
            pass

        for regression_model in reg_models:
            RV = RegressionTest_sklearn(regression_model,problem, random_seed)
            try:
                path4 = f"{path3}/{RV.model.name[:6]}/"
                os.mkdir(path4)
            except:
                continue
            RV.train_test_loop(n_train_array, n_test, output_path = f"{path4}")