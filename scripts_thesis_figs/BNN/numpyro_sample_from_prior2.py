
# import os
# print(os.getcwd())

from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import jax.random as random
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

X_sample =  np.linspace(-10,10,500)[:,None]
fig, ax = plt.subplots(nrows=1, ncols=1, sharex="all")#, sharey="row")



NNN_regression = NumpyroNeuralNetwork(
    hidden_units=300,
    num_chains = 4, num_warmup= 200, num_samples=10, 
    hidden_units_bias_variance=1, 
    hidden_units_variance=1)

#NNN_regression.fit(X_sample, Y_sample)
NNN_regression.x_mean = 0
NNN_regression.x_std = 1
n_samples = 1000
np.random.seed(1)
Y_all = []
for i in range(n_samples):
    r = np.random.randint(1000000,size = 1)[0]
    NNN_regression.rng_key, NNN_regression.rng_key_predict = random.split(random.PRNGKey(r))
    Y = NNN_regression.model_sample(X_sample)
    #plt.title(NNN_regression.text_priors)
    Y_all.append(Y)
y_pred = np.array(Y_all).squeeze()
y_mean = np.mean(y_pred, axis=0)
y_std = np.std(y_pred , axis=0)

ax.fill_between(
    X_sample.squeeze(),
    y_mean - 1.96*y_std,
    y_mean + 1.96*y_std,
    alpha=0.5,
    color="tab:blue",
    label=r"$\pm$ 1 std. dev.",
)
ax.plot(X_sample,y_pred[40:60,:].T, color = "Black", alpha=1)
ax.plot(X_sample.squeeze(), y_mean, color="red", label="Mean")

#plt.text(-8,Y_max-4,NNN_regression.text_priors)
ax.set_title("BNN(50x50x50) prior")
ax.set_xlabel("x")
ax.set_ylabel("y")
#plt.show()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(4, 3)
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/BNN_prior.pdf', bbox_inches='tight')
