import numpy as np
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.bohamiann import BOHAMIANN
#from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.mean_regression import MeanRegression
from src.regression_models.SPN_regression2 import SumProductNetworkRegression

import random
import numpy as np
from src.utils import RegressionValidation

from src.benchmarks.custom_test_functions.problems import SimonsTest,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction
from src.benchmarks.go_benchmark_functions.go_funcs_S import Step, Step2, Schwefel26
from src.benchmarks.go_benchmark_functions.go_funcs_R import Rastrigin, Rosenbrock
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass
from src.benchmarks.go_benchmark_functions.go_funcs_K import Katsuura

#from src.analysis_helpers import include_true_values

import os
from datetime import datetime

#prob = Zirilli(dimensions = 2)
dims = [2,3,5,10]
#dims = [1]

problems = [Step(dimensions = x) for x in dims]
problems += [Rastrigin(dimensions = x) for x in dims]
problems += [Weierstrass(dimensions = x) for x in dims]
#problems += [Katsuura(dimensions = x) for x in dims]
problems += [Schwefel26(dimensions = x) for x in dims]
#problems += [Rosenbrock(dimensions = x) for x in dims]

#problems = [SimonsTest4_cosine_fuction()]

random.seed()
random.shuffle(problems)

# BOHAMIANN_regression_fast = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 100)
# BOHAMIANN_regression = BOHAMIANN(num_warmup = 2000, num_samples = 2000, num_keep_samples= 300,  extra_name="2000-2000")
# BOHAMIANN_regression_slow = BOHAMIANN(num_warmup = 4000, num_samples = 4000, num_keep_samples= 300,  extra_name="4000-4000")
# NNN_regression_fast = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 100)
# NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 2000, num_samples=2000, num_keep_samples= 300, extra_name="2000-2000")
# # Slow NNN with 3*50 hidden units takes 15 min per. training 
# # -> way to slow for these experiements. 

# regression_models = [BOHAMIANN_regression_fast,BOHAMIANN_regression,BOHAMIANN_regression_slow,NNN_regression_fast]#, NNN_regression]
# regression_models += [MeanRegression()]
# regression_models += [GMRegression()] #Gaussian Mixture
# random.shuffle(regression_models)
# #regression_models = [MeanRegression()]
# regression_models = [GaussianProcess_sklearn()]
regression_models = [SumProductNetworkRegression(manipulate_variance=False, optimize=True, tracks=2, channels=50)]
## Data enrichment ##
#include_true_values(problems, remove_min_n_test=True)

# unit test ###
test_model = MeanRegression()
for problem in problems:
    print(test_model.name, f"{type(problem).__name__} in dim {problem.N}")
    RV = RegressionValidation(problem, test_model, 2)
    RV.train_test_loop([10], 1)
print("\n-- TEST: all problems are defined --\n")

for regression_model in regression_models:
    print(regression_model.name, f"{type(problem).__name__}")
    RV = RegressionValidation(problems[0], regression_model, 2)
    RV.train_test_loop([10], 1)
print("\n-- TEST: all models are defined --\n")

### main ###
n_train_array = [int(x) for x in np.logspace(1, 2.5, 9)]
# n_train_array = [10]
# n_train_array = [10,20,30]
#n_train_array = [int(x) for x in np.logspace(1, 1.8, 9)]
n_test = 1000#10000

run_name = datetime.today().strftime('%m%d_%H%M')
dirname=os.path.dirname

path = os.path.join(dirname(dirname(dirname(os.path.abspath(__file__)))),f"data/{run_name}")
try:
    os.mkdir(path)
except:
    print(f"Couldn't create {path}")

np.random.seed()
for problem in problems:
    for random_seed in np.random.randint(9999, size=1):
        for regression_model in regression_models:
            #try:
            print(regression_model.name, f"{type(problem).__name__} in dim {problem.N}")
            RV = RegressionValidation(problem, regression_model, random_seed)
            RV.train_test_loop(n_train_array, n_test,path =  f"{path}")
            RV.save_regression_validation_results(f"{path}")
            # except:
            #     print(f"ERROR: Could not train {regression_model.name} on {type(problem).__name__} in dim {problem.N}")


