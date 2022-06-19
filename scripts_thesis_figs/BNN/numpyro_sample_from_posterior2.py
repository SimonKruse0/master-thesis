
# import os
# print(os.getcwd())

from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import jax.random as random
from src.utils import denormalize 
def obj_fun(x): 
    x = x/10
    return 50*x*np.sin(x * (2 * np.pi)) + 5*x + np.sin(100*x)

# # X_train =  np.random.uniform(-10,10,size = (10,1))
# X_train =  np.random.uniform(-10,10,size = (4,1))
# Y_train = obj_fun(X_train)

rng = np.random.RandomState(4)
X_train = rng.uniform(-5, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)

Y_train = y_train[:,None]

fig, ax = plt.subplots(nrows=1, ncols=1, sharex="all")#, sharey="row")

X_sample =  np.linspace(-10,10,500)[:,None]


NNN_regression = NumpyroNeuralNetwork(
    num_chains = 4, num_warmup= 300, num_samples=200, 
    hidden_units_bias_variance=1, 
    hidden_units_variance=1)

NNN_regression.fit(X_train, Y_train, do_normalize = True)
Y_max = 0
np.random.seed(1)
Y = NNN_regression.predict(X_sample, get_y_pred=True)

y_pred = np.array(Y).squeeze()
y_pred = denormalize(y_pred, NNN_regression.y_mean, NNN_regression.y_std)
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
ax.plot(X_sample,y_pred[::20,:].T, color = "Black", alpha=1)
ax.plot(X_sample.squeeze(), y_mean, color="red", label="Mean")
ax.scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
#plt.text(-8,Y_max-4,NNN_regression.text_priors)
ax.set_title("BNN(50x50x50) posterior")
ax.set_xlabel("x")
ax.set_ylabel("y")
#plt.show()
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches(4, 3)

path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/BNN_posterior.pdf', bbox_inches='tight')

