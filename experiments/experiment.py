from atexit import register
import numpy as np
from pymoo.factory import get_problem, get_visualization
from ..src.utils import OptimizationStruct
from bayesian_optimization import BayesianOptimization
from ..src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from ..src.regression_models.gaussian_process_regression import GaussianProcess
from ..src.regression_models.bohamiann import BOHAMIANN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



n_var = 1
problem = get_problem("rosenbrock", n_var=n_var)

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1
    #return problem.evaluate(x)

bounds = [-2,2]
np.random.seed(2)
X_init =  np.random.uniform(*bounds,size = (5,n_var))
Y_init = obj_fun(X_init)

BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 50)
GP_regression = GaussianProcess(noise = 0)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=300, num_keep_samples= 50)

regression_model = [BOHAMIANN_regression, GP_regression, NNN_regression]

for i in range(3):
    BO = BayesianOptimization(obj_fun, regression_model[i],bounds,X_init,Y_init )
    #BO.optimize(4, type = "numeric", n_restarts=10)
    ax = plt.subplot()
    opt = BO.optimization_step(type="numeric")
    BO.plot_surrogate_and_expected_improvement(ax, opt)
    plt.show()
    x_hist,y_hist = BO.get_optimization_hist()
    print(x_hist)