import numpy as np
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.gaussian_process_regression import GaussianProcess
from src.regression_models.bohamiann import BOHAMIANN
import numpy as np
from src.utils import RegressionValidation
from src.benchmark_problems import Zirilli, Weierstrass, Rosen


#prob = Zirilli(dimensions = 2)
problem = Weierstrass(dimensions = 2)

BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 100)
GP_regression = GaussianProcess(noise = 0)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 100)

regression_model = [BOHAMIANN_regression,GP_regression, NNN_regression]

n_train_array = [int(x) for x in np.logspace(1, 2.5, 5)]
n_test = 100

for random_seed in np.random.randint(9999, size=1):
    for i in range(3):
        print(regression_model[i].name, f"{type(problem).__name__}")
        RV = RegressionValidation(problem, regression_model[i], random_seed)
        RV.train_test_loop(n_train_array, n_test)
        RV.save_regression_validation_results("master-thesis/data")
        #print(RV.mean_abs_pred_error, RV.mean_uncertainty_quantification)

