import numpy as np
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.gaussian_process_regression import GaussianProcess
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_mixture_regression2 import GMRegression

import numpy as np
from src.utils import RegressionValidation
from src.benchmark_problems import Zirilli, Weierstrass, Rosenbrock

from src.go_benchmark_functions.go_funcs_S import Step, Step2

import os
from datetime import datetime

#prob = Zirilli(dimensions = 2)
problems = [Weierstrass(dimensions = 2), Zirilli(dimensions = 2), Rosenbrock(dimensions=2), Rosenbrock(dimensions=10)]
problems = [Step2(dimensions=10)]

BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 100)
GP_regression = GaussianProcess(noise = 0)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 100)
mixture_regression = GMRegression()

regression_model = [mixture_regression,BOHAMIANN_regression,GP_regression, NNN_regression]

#test:
for problem in problems:
    print(regression_model[0].name, f"{type(problem).__name__}")
    RV = RegressionValidation(problem, regression_model[0], 2)
    RV.train_test_loop([10], 1)
print("test passed")

n_train_array = [int(x) for x in np.logspace(1, 2.5, 5)]
n_train_array = [int(x) for x in np.logspace(1, 2.5, 1)]
n_test = 100 #10000

run_name = datetime.today().strftime('%m%d_%H%M')

path = os.path.join(os.path.dirname(os.path.abspath(__file__)),f"data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

np.random.seed()
for problem in problems:
    for random_seed in np.random.randint(9999, size=1):
        for i in range(4):
            print(regression_model[i].name, f"{type(problem).__name__} in dim {problem.N}")
            RV = RegressionValidation(problem, regression_model[i], random_seed)
            RV.train_test_loop(n_train_array, n_test)
            RV.save_regression_validation_results(f"{path}")
            #print(RV.mean_abs_pred_error, RV.mean_uncertainty_quantification)

