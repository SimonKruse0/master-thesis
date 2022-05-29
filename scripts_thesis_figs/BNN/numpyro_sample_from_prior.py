
# import os
# print(os.getcwd())

from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import jax.random as random

X_sample =  np.linspace(-10,10,500)[:,None]
fig, ax = plt.subplots(nrows=2, ncols=2, sharex="all")#, sharey="row")

b_var = [1,5]
w_var = [1,5]
for b, row in zip(b_var, ax):
    for w, ax in zip(w_var,row):

        NNN_regression = NumpyroNeuralNetwork(
            num_chains = 4, num_warmup= 200, num_samples=200, 
            num_keep_samples= 50, hidden_units_bias_variance=b, 
            hidden_units_variance=w)

        #NNN_regression.fit(X_sample, Y_sample)
        NNN_regression.x_mean = 0
        NNN_regression.x_std = 1
        Y_max = 0
        n_samples = 100
        np.random.seed(1)
        for _ in range(n_samples):
            r = np.random.randint(1000000,size = 1)[0]
            NNN_regression.rng_key, NNN_regression.rng_key_predict = random.split(random.PRNGKey(r))
            Y = NNN_regression.model_sample(X_sample)
            ax.plot(X_sample,Y, color = "Blue", alpha=0.2)
            #plt.title(NNN_regression.text_priors)
            Y_max =max(Y_max,Y.max())
        #plt.text(-8,Y_max-4,NNN_regression.text_priors)
        ax.set_title(NNN_regression.text_priors)

path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/bayesian_nn_prior_samples.pdf', bbox_inches='tight')

