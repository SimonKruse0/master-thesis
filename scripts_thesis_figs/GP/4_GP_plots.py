from cmath import nan
from matplotlib.markers import MarkerStyle
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2
import numpy as np

rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)


def plot_gpr_samples(gpr_model, n_samples, ax, plot_type = 0):
    """Plot samples drawn from the Gaussian process model.

    If the Gaussian process model is not trained then the drawn samples are
    drawn from the prior distribution. Otherwise, the samples are drawn from
    the posterior distribution. Be aware that a sample here corresponds to a
    function.

    Parameters
    ----------
    gpr_model : `GaussianProcessRegressor`
        A :class:`~sklearn.gaussian_process.GaussianProcessRegressor` model.
    n_samples : int
        The number of samples to draw from the Gaussian process distribution.
    ax : matplotlib axis
        The matplotlib axis where to plot the samples.
    """
    if plot_type == 1:
        x = np.linspace(0, 5, 6)
    else:
        x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        if plot_type == 1:
            ax.plot(
            x,
            single_prior,
            linestyle="--",
            marker="*",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
            )
        else:
            ax.plot(
            x,
            single_prior,
            linestyle="--",
            #marker="*",
            alpha=0.7,
            label=f"Sampled function #{idx + 1}",
            )
    if plot_type == 1:
        xx = list(x)+list(x+0.05) +list(x-0.05) +list(x+0.5)[:-1]
        xx.sort()
        xx = np.array(xx)
        y_mean, y_std = gpr_model.predict(xx[:,None], return_std=True)
        xx[[abs(k.round()-k) == 0.5 for k in xx]] = np.NaN
        x = xx
        #x+(x+0.01
        ax.plot(x, y_mean, color="black", label="Mean")
        ax.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.5,
            color="tab:blue",
            label=r"$\pm$ 1 std. dev.",
        )
    else:
        ax.plot(x, y_mean, color="black", label="Mean")
        ax.fill_between(
            x,
            y_mean - y_std,
            y_mean + y_std,
            alpha=0.5,
            color="tab:blue",
            label=r"$\pm$ 1 std. dev.",
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])

from sklearn.gaussian_process.kernels import Matern

kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)#, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=10, ax=axs[0], plot_type=1)
#axs[0].set_title(r"$\mathcal{N}([f(x_1), \dots, f(x_6)]|0, \kappa([x_1,\dots, x_6]^2))$")
axs[0].set_title(r"$\mathcal{N}([f(0), \dots, f(5)]|0, \kappa([0,\dots, 5]^2))$")

# plot posterior
#gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=20, ax=axs[1], plot_type=2)
# axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="Observations")
# #axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("$\mathcal{N}(f(X)|0, \kappa(X,X))$")

#fig.suptitle("Mattern kernel", fontsize=18)
plt.tight_layout()
#plt.show()
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/GP_samples_mattern.pdf', bbox_inches='tight')
