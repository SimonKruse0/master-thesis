import numpy as np
import matplotlib.pyplot as plt 
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy

from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork

from src.benchmarks.custom_test_functions.problems import SimonsTest0
def f(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(x*100)*0.01

rng = np.random.RandomState(2)
x = rng.rand(20)
x = x[x>0.1]
y = f(x)

grid = np.linspace(0, 1, 200)
fvals = f(grid)

# plt.plot(grid, fvals, "k--")
# plt.plot(x, y, "ro")
# plt.grid()
# plt.xlim(0, 1)

# plt.show()
plot_type = 1
gp_local_min = 2
while plot_type == 1:
    plt.figure()
    GP_reg = GaussianProcess_GPy(bounds=False )
    GP_reg.fit(x[:, None],y[:, None])
    m, sd, _ = GP_reg.predict(grid[:,None])

    plt.plot(x, y, "ro")
    #plt.grid()
    plt.plot(grid, fvals, "k--")
    plt.plot(grid, m, "tab:orange")
    plt.fill_between(grid, m + sd, m - sd, color="tab:blue", alpha=0.8)
    plt.fill_between(grid, m + 2 * sd, m - 2 * sd, color="tab:blue", alpha=0.6)
    plt.fill_between(grid, m + 3 * sd, m - 3 * sd, color="tab:blue", alpha=0.4)
    plt.xlim(0, 1)
    plt.xlabel(r"Input $x$")
    plt.ylabel(r"Output $f(x)$")
    plt.title(f"{GP_reg.name}({GP_reg.params})")
    #plt.show()
    if GP_reg.model.log_likelihood()>-15 and gp_local_min==1:
        path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
        #plt.savefig(f'{path}/GP_vs_BNN1_b.pdf', bbox_inches='tight')
        plt.show()
        break
    if GP_reg.model.log_likelihood()<-15 and gp_local_min!=1:
        path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
        #plt.savefig(f'{path}/GP_vs_BNN1.pdf', bbox_inches='tight')
        plt.show()
        break

if plot_type == 2:
    plt.figure()
    bo_reg = NumpyroNeuralNetwork(num_warmup=1000, num_samples = 1500)
    bo_reg.fit(x[:, None],y[:, None])
    m, sd,_ = bo_reg.predict(grid[:, None])

    # bo_reg = BOHAMIANN()
    # bo_reg.fit(x[:, None],y[:, None])
    # m, sd,_ = bo_reg.predict(grid[:, None])

    plt.plot(x, y, "ro")
    #plt.grid()
    plt.plot(grid, fvals, "k--")
    plt.plot(grid, m, "tab:orange")
    plt.fill_between(grid, m + sd, m - sd, color="tab:blue", alpha=0.8)
    plt.fill_between(grid, m + 2 * sd, m - 2 * sd, color="tab:blue", alpha=0.6)
    plt.fill_between(grid, m + 3 * sd, m - 3 * sd, color="tab:blue", alpha=0.4)
    plt.xlim(0, 1)
    plt.xlabel(r"Input $x$")
    plt.ylabel(r"Output $f(x)$")
    plt.title(f"BNN")#({bo_reg.params})")
    #plt.show()
    path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
    plt.savefig(f'{path}/GP_vs_BNN2.pdf', bbox_inches='tight')

