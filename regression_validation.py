import numpy as np
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.gaussian_process_regression import GaussianProcess
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.gaussian_mixture_regression2 import GMRegression

import numpy as np
from src.utils import RegressionValidation
from src.benchmark_problems import Zirilli, Weierstrass, Rosenbrock

import os
from datetime import datetime

#prob = Zirilli(dimensions = 2)
problems = [Weierstrass(dimensions = 2), Zirilli(dimensions = 2), Rosenbrock(dimensions=2), Rosenbrock(dimensions=10)]

BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 100)
GP_regression = GaussianProcess(noise = 0)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 100)
mixture_regression = GMRegression()

regression_model = [mixture_regression,BOHAMIANN_regression,GP_regression, NNN_regression]

n_train_array = [int(x) for x in np.logspace(1, 2.5, 5)]
n_test = 100

run_name = datetime.today().strftime('%m%d_%H%M')

try:
    path = os.path.join(os.getcwd(),f"master-thesis/data/{run_name}")
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

for problem in problems[2:]:
    for random_seed in np.random.randint(9999, size=2):
        for i in range(3):
            print(regression_model[i].name, f"{type(problem).__name__}")
            RV = RegressionValidation(problem, regression_model[i], random_seed)
            RV.train_test_loop(n_train_array, n_test)
            RV.save_regression_validation_results(f"master-thesis/data/{run_name}")
            #print(RV.mean_abs_pred_error, RV.mean_uncertainty_quantification)

