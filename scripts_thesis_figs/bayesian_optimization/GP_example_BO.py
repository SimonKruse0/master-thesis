from src.optimization.bayesopt_solver import PlotBayesOpt1D
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
import matplotlib.pyplot as plt
from src.benchmarks.custom_test_functions.problems import Test1b
import numpy as np
import os

## INPUT ##
acquisition = "EI" #LCB, EI
n_init_samples = 10
samples = 0
## main ##
reg_model = GaussianProcess_GPy()

problem_sklearn = Test1b()
BOs = []
acquisitions = ["EI", "LCB", "LCB"]
for i,acquisition in enumerate(acquisitions):
    np.random.seed(21)
    plot_BO = PlotBayesOpt1D(reg_model, problem_sklearn, deterministic = True,
                acquisition=acquisition,budget=n_init_samples+samples,show_name=False, n_init_samples=n_init_samples,disp=True)
    plot_BO.beta = 1
    if i==2:
        plot_BO.beta = 3
    BOs.append(plot_BO)

import matplotlib.gridspec as gridspec

fig,ax = plt.subplots(1,1,sharex="all")

ax1 = ax#[0]
#ax2 = ax[1]

for iter in range(1):
    plot_BO = BOs[0]
    plot_BO.optimization_step(update_y_min=False)
    plot_BO.plot_surrogate_and_acquisition_function(ax1)
    #plt.show()
#     path_data = "/home/simon/Documents/MasterThesis/master-thesis/scripts_thesis_figs/bayesian_optimization"
#     np.savetxt(path_data+"/GP_imp_sig_data.txt", np.array([imp,sigma]))

path  = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f"{path}/BO_example.pdf")
