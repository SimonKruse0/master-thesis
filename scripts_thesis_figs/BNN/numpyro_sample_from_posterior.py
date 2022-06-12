
# import os
# print(os.getcwd())

from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import jax.random as random

def obj_fun(x): 
    x = x/5
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

X_train =  np.random.uniform(-10,10,size = (10,1))
X_train =  np.random.uniform(-10,10,size = (2,1))
Y_train = obj_fun(X_train)

fig, ax = plt.subplots(nrows=2, ncols=2, sharex="all")#, sharey="row")

X_sample =  np.linspace(-10,10,500)[:,None]
b_var = [1,5]
w_var = [1,5]
for b, row in zip(b_var, ax):
    for w, ax in zip(w_var,row):

        NNN_regression = NumpyroNeuralNetwork(
            num_chains = 4, num_warmup= 100, num_samples=30, 
            hidden_units_bias_variance=b, 
            hidden_units_variance=w)

        NNN_regression.fit(X_train, Y_train, do_normalize = False)
        Y_max = 0
        n_samples = 100
        np.random.seed(1)
        Y = NNN_regression.predict(X_sample, get_y_pred=True)
        for y in Y:
            ax.plot(X_sample,y, color = "Blue", alpha=0.2)
            #plt.title(NNN_regression.text_priors)
            #Y_max =max(Y_max,Y.max())
        #plt.text(-8,Y_max-4,NNN_regression.text_priors)
        ax.set_title(NNN_regression.text_priors)
        ax.plot(X_train, Y_train, "*")
        plt.show()
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
# plt.savefig(f'{path}/bayesian_nn_posterior_samples.pdf', bbox_inches='tight')

