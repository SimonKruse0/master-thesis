
# import os
# print(os.getcwd())

from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import jax.random as random

X_sample =  np.linspace(-10,10,500)[:,None]
#fig, axs = plt.subplots(nrows=2, ncols=2, sharex="all")#, sharey="row")

font = {'family' : 'normal',
    'weight' : 'bold',
    'size'   : 20}
import matplotlib
matplotlib.rc('font', **font)

hidden_units = [1,5,20,100]
for i,hu in enumerate(hidden_units):
# for b, row in zip([], ax):
#     for w, ax in zip(w_var,row):
    print(i%3,i//3)
    #ax0 = axs[i%2,i//2]
    fig,ax0 = plt.subplots()
    NNN_regression = NumpyroNeuralNetwork(
        hidden_units=hu,
        num_chains = 4, num_warmup= 200, num_samples=10, 
        hidden_units_bias_variance=1, 
        hidden_units_variance=1)

    #NNN_regression.fit(X_sample, Y_sample)
    NNN_regression.x_mean = 0
    NNN_regression.x_std = 1
    Y_max = 0
    n_samples = 50
    np.random.seed(1)
    for _ in range(n_samples):
        r = np.random.randint(1000000,size = 1)[0]
        NNN_regression.rng_key, NNN_regression.rng_key_predict = random.split(random.PRNGKey(r))
        Y = NNN_regression.model_sample(X_sample)
        ax0.plot(X_sample,Y, color = "Black", alpha=0.4, lw=1.5)
        #plt.title(NNN_regression.text_priors)
        Y_max =max(Y_max,Y.max())
    #plt.text(-8,Y_max-4,NNN_regression.text_priors)
    ax0.set_title(f"hidden units = {NNN_regression.hidden_units}")
    path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
    plt.savefig(f'{path}/bayesian_nn_prior_samples_hidden_units_{i}.pdf', bbox_inches='tight')
plt.show()

