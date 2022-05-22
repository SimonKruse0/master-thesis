import numpy as np

from src.regression_models.SPN_regression2 import SumProductNetworkRegression

import random
from src.regression_validation.reg_validation import RegressionTest

import cocoex #the 24 functions i defined from here
import os
from datetime import datetime

from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random

reg_models = [MeanRegression(), 
            NaiveGMRegression(optimize=True), 
            GaussianProcess_sklearn(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=200,num_samples=200,
                                num_chains=4, num_keep_samples=200, 
                                extra_name="200-200"),
            SumProductNetworkRegression(optimize=True)]
            
random.seed()
random.shuffle(reg_models)
### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
#n_train_array = [10000]
n_test = 10000

run_name = datetime.today().strftime('%m%d_%H%M')
dirname=os.path.dirname
path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"coco_reg_data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

suite = cocoex.Suite("bbob", "", "dimensions:5,6,7,8,9,10 instance_indices:1")

for problem in suite:
    name_problem = problem.name.split(" ")[3]
    dim = problem.name.split(" ")[-1]
    print(name_problem)
    # if int(name_problem[1:]) < 19:
    #     continue
    np.random.seed()
    for random_seed in np.random.randint(99999, size=1):
        for regression_model in reg_models:
            print(regression_model.name, f"{name_problem} in {dim}")
            RV = RegressionTest(regression_model,problem, random_seed)
            RV.train_test_loop(n_train_array, n_test, output_path = f"{path}")