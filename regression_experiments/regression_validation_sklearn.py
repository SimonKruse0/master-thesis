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
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import random


reg_models = [MeanRegression(), 
            NaiveGMRegression(optimize=True), 
            GaussianProcess_GPy(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=500,num_samples=500,
                                num_chains=4),
            SumProductNetworkRegression(optimize=True),
            GMRegression(optimize=True)]
            
# random.seed()
# random.shuffle(reg_models)
#reg_models = [SumProductNetworkRegression(optimize=False)]#, 
#reg_models = [SumProductNetworkRegression(optimize=False, opt_n_iter=10, extra_name="no-optimized-low-test")]
#reg_models = [GMRegression(optimize=False,opt_n_iter=10, extra_name="no-optimized-low-test")]
### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
n_test = 10000
#n_test = 1000

n_train_array = [10]


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
#for regression_model in reg_models:
for problem in problems:
    np.random.seed()
    for random_seed in np.random.randint(99999, size=1):
        #for problem in problems:
        for regression_model in reg_models:
            #print(regression_model.name, f"{name_problem} in {dim}")
            
            RV = RegressionTest_sklearn(regression_model,problem, random_seed)
            RV.train_test_loop(n_train_array, n_test, output_path = f"{path}")