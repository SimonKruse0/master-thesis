from atexit import register
import numpy as np
from src.utils import OptimizationStruct
from src.optimization.bayesian_optimization import BayesianOptimization

from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.bohamiann import BOHAMIANN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.benchmarks.custom_test_functions.problems import SimonsTest,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction

problem = SimonsTest4_cosine_fuction()

np.random.seed(2)
X_init =  np.random.uniform(*problem.bounds[0],size = (50,problem.N))
Y_init = problem.fun(X_init)



BOHAMIANN_regression = BOHAMIANN(num_warmup = 2000, num_samples = 3000, num_keep_samples= 1000)
# GP_regression = GaussianProcess(noise = 0)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 50)
SPN_reg = SumProductNetworkRegression(optimize=True, manipulate_variance=True)
GM_reg = GMRegression()

regression_models = [NNN_regression,BOHAMIANN_regression, SPN_reg, GM_reg ]
plt.figure(figsize=(12, 8))
outer_gs = gridspec.GridSpec(2, len(regression_models)//2+1)

for i in range(len(regression_models)):
    BO = BayesianOptimization(problem, regression_models[i],X_init,Y_init )
    #BO.optimize(4, type = "numeric", n_restarts=10)
    ax = outer_gs[i]
    opt = BO.optimization_step(type="grid")
    BO.plot_surrogate_and_expected_improvement(ax, opt, show_name=True)

plt.show()
x_hist,y_hist = BO.get_optimization_hist()
print(x_hist)