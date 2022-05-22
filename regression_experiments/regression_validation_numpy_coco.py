import numpy as np

from src.regression_models.SPN_regression2 import SumProductNetworkRegression

import random
from src.regression_validation.reg_validation import RegressionTest_numpycoco

import cocoex #the 24 functions i defined from here

from src.bbob_numpy.suite import Suite

import os
from datetime import datetime

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
            NumpyroNeuralNetwork(num_warmup=200,num_samples=200,
                                num_chains=4, num_keep_samples=200, 
                                extra_name="200-200"),
            SumProductNetworkRegression(optimize=True,manipulate_variance=False)]
            
random.seed()
random.shuffle(reg_models)
reg_models = [GaussianProcess_sklearn()]

### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
n_test = 10000

# print("no real test!")
# n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
# n_test = 100

run_name = datetime.today().strftime('%m%d_%H%M')
dirname=os.path.dirname
path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"coco_reg_data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

#np.random.seed()
#suite = cocoex.Suite("bbob", "", "dimensions:2,3,5,10 instance_indices:1")
#suite = cocoex.Suite("bbob", "", "dimensions:2 instance_indices:1")
test_suite_options = {'name': 'full',
                        'dim': [2,4],
                        'n_instances': 1}
test_suite = Suite(test_suite_options)


for problem in test_suite:
    name_problem = problem.name
    print(name_problem)
    if name_problem.lower() != "sphere":
        continue
    dim = problem.f_obj.d
    for random_seed in np.random.randint(9999, size=1):
        for regression_model in reg_models:
            print(regression_model.name, f"{name_problem} in {dim}")
            RV = RegressionTest_numpycoco(regression_model,problem, random_seed)
            RV.train_test_loop(n_train_array, n_test, output_path = f"{path}")