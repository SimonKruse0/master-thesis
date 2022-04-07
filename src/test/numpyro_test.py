# import os
# print(os.getcwd())

from regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import numpy as np

NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=200, num_keep_samples= 50)

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

X_sample =  np.random.uniform(0,1,size = (10,1))
Y_sample = obj_fun(X_sample)

NNN_regression.fit(X_sample, Y_sample)