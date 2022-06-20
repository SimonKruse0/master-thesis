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
    plot_BO = PlotBayesOpt1D(reg_model, problem_sklearn, 
                acquisition=acquisition,budget=n_init_samples+samples,show_name=False, n_init_samples=n_init_samples,disp=True)
    plot_BO.beta = 1
    if i==2:
        plot_BO.beta = 3
    BOs.append(plot_BO)

import matplotlib.gridspec as gridspec

fig,ax = plt.subplots(3,1,sharex="all")

ax1 = ax[0]
ax2 = ax[1]
ax3 = ax[2]

for iter in range(1):
    for i, plot_BO in enumerate(BOs):
        plot_BO.optimization_step(update_y_min=False)
        
        if i == 0:
            plot_BO.plot_regression_gaussian_approx(ax1, show_name =False)
            ax1.plot(plot_BO._X[:-1],plot_BO._Y[:-1], ".", markersize = 10, color="black")
            ax = ax2
            imp, sigma = plot_BO.plot_acquisition_function(ax, color = f"C{i}", show_y_label = False, return_path = True)
        elif i==2:
            ax = ax3
            imp2, sigma2 = plot_BO.plot_acquisition_function(ax, color = f"C{i}", show_y_label = False,  return_path = True)
        else:
            ax = ax3
            imp3, sigma3 = plot_BO.plot_acquisition_function(ax, color = f"C{i}", show_y_label = False,  return_path = True)

        x_next = plot_BO.opt.x_next[:,None]
        max_AQ= plot_BO.acquisition_function(x_next)
        ax.plot(x_next, max_AQ, "^", markersize=10,color="tab:orange")#, label=f"x_next = {plot_BO.opt.x_next[0]:.2f}")
        #ax.plot(x_next, max_AQ, "^", markersize=10,color="tab:orange",label=f"x_next = {plot_BO.opt.x_next[0]:.2f}")
        ax.legend(loc=1)
        
        #plt.show()
        number = f"{iter}".zfill(3)
    ax1.set_ylabel("y")
    ax3.set_xlabel("x")
    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    #plt.show()
    path_data = "/home/simon/Documents/MasterThesis/master-thesis/scripts_thesis_figs/bayesian_optimization"
#     np.savetxt(path_data+"/GP_imp_sig_data.txt", np.array([imp,sigma]))
#     np.savetxt(path_data+"/GP_imp_sig_data2.txt", np.array([imp2,sigma2]))
#     np.savetxt(path_data+"/GP_imp_sig_data3.txt", np.array([imp3,sigma3]))

path  = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f"{path}/illustration_AQs.pdf")
