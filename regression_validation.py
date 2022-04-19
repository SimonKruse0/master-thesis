import numpy as np
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.gaussian_process_regression import GaussianProcess
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_mixture_regression2 import GMRegression

import random
import numpy as np
from src.utils import RegressionValidation
from src.benchmark_problems import Zirilli, Weierstrass, Rosenbrock

from src.go_benchmark_functions.go_funcs_S import Step, Step2

import os
from datetime import datetime

#prob = Zirilli(dimensions = 2)
#problems = [Weierstrass(dimensions = 2), Zirilli(dimensions = 2), Rosenbrock(dimensions=2), Rosenbrock(dimensions=10)]
problems = [Step2(dimensions = 1), Step2(dimensions = 3), Step2(dimensions = 5), Step2(dimensions=10)]
random.seed()
random.shuffle(problems)

BOHAMIANN_regression_fast = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 100)
BOHAMIANN_regression = BOHAMIANN(num_warmup = 2000, num_samples = 2000, num_keep_samples= 100,  extra_name="2000-2000")
GP_regression = GaussianProcess(noise = 0)
NNN_regression_fast = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 100)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 2000, num_samples=2000, num_keep_samples= 100, extra_name="2000-2000")
mixture_regression = GMRegression()

regression_models = [mixture_regression,BOHAMIANN_regression_fast,GP_regression,BOHAMIANN_regression,NNN_regression_fast, NNN_regression]


### unit test ###
# for problem in problems:
#     print(regression_models[0].name, f"{type(problem).__name__} in dim {problem.N}")
#     RV = RegressionValidation(problem, regression_models[0], 2)
#     RV.train_test_loop([10], 1)
# print("\n-- TEST: all problems are defined --\n")

# for regression_model in regression_models:
#     print(regression_model.name, f"{type(problem).__name__}")
#     RV = RegressionValidation(problems[0], regression_model, 2)
#     RV.train_test_loop([10], 1)
# print("\n-- TEST: all models are defined --\n")

### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
n_test = 10000

run_name = datetime.today().strftime('%m%d_%H%M')

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

np.random.seed()
for problem in problems:
    for random_seed in np.random.randint(9999, size=1):
        for regression_model in regression_models:
            print(regression_model.name, f"{type(problem).__name__} in dim {problem.N}")
            RV = RegressionValidation(problem, regression_model, random_seed)
            RV.train_test_loop(n_train_array, n_test)
            RV.save_regression_validation_results(f"{path}")
            #print(RV.mean_abs_pred_error, RV.mean_uncertainty_quantification)

