import numpy as np

from src.regression_models.SPN_regression2 import SumProductNetworkRegression

import random
from src.regression_validation.reg_validation import RegressionTest_sklearn

from src.benchmarks.custom_test_functions.problems import Test1,Test2,Test3,Test4,Test3b
import os
from datetime import datetime

from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random


reg_models = [MeanRegression(), 
            GMRegression(optimize=True),
            NaiveGMRegression(optimize=True), 
            GaussianProcess_sklearn(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=500,num_samples=500,
                                num_chains=4),
            SumProductNetworkRegression(optimize=True)]
            
random.seed()
random.shuffle(reg_models)
#reg_models = [NaiveGMRegression(optimize=False), SumProductNetworkRegression(optimize=False, opt_n_iter=5)]

### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
n_test = 10000

run_name = datetime.today().strftime('%m%d_%H%M')
dirname=os.path.dirname
path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")


#problems = [SimonsTest2_probibalistic()]
# problems = [Step_random()]
# problems += [SimonsTest2_probibalistic2()]
problems = [Test1(),Test2(),Test3(),Test4(),Test3b()]
for regression_model in reg_models:
    # name_problem = problem.name.split(" ")[3]
    # dim = problem.name.split(" ")[-1]
    # print(name_problem)
    # if int(name_problem[1:])%3 != 2:
    #     continue
    np.random.seed()
    for random_seed in np.random.randint(99999, size=3):
        for problem in problems:
            #print(regression_model.name, f"{name_problem} in {dim}")
            
            RV = RegressionTest_sklearn(regression_model,problem, random_seed)
            RV.train_test_loop(n_train_array, n_test, output_path = f"{path}")