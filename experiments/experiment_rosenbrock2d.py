from atexit import register
import numpy as np
from pymoo.factory import get_problem
from bayesian_optimization import BayesianOptimization
from ..src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from ..src.regression_models.gaussian_process_regression import GaussianProcess
from ..src.regression_models.bohamiann import BOHAMIANN
import numpy as np
import matplotlib.pyplot as plt

n_var = 2
problem = get_problem("rosenbrock", n_var=n_var)

def obj_fun(x): 
    return np.log(problem.evaluate(x))#/1000

bounds = [-2,2]
np.random.seed(2)
X_init =  np.random.uniform(*bounds,size = (10,n_var))
Y_init = obj_fun(X_init)

BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 300, num_keep_samples= 100)
GP_regression = GaussianProcess(noise = 0)
NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 2000, num_samples=3000, num_keep_samples= 100)

regression_model = [BOHAMIANN_regression, GP_regression, NNN_regression]

for i in range(3):
    BO = BayesianOptimization(obj_fun, regression_model[i],bounds,X_init,Y_init )
    opt = BO.optimization_step()
    BO.plot_2d(opt, plot_obj=False)
    #plt.savefig(f"master-thesis/figures2d/true_log.png")
    #plt.savefig(f"master-thesis/figures2d/{BO.model.name}_log.png")
    plt.show()
    # BO.optimize(10, type = "grid")
    # x_hist,y_hist = BO.get_optimization_hist()
    # print(x_hist)
    # print(y_hist)
    # with open(f'{BO.model.name}_rosenbrock2d.npy', 'wb') as f:
    #     np.save(f, x_hist)
    #     np.save(f, y_hist)

for i in range(3):
    name = regression_model[i].name
    with open(f'{name}_rosenbrock2d.npy', 'rb') as f:
        x_hist = np.load(f,allow_pickle=True)
        y_hist = np.load(f,allow_pickle=True)

        print(x_hist,y_hist)
