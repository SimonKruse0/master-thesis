# Notes:
# This illustrates how exploitation is really favoured by EI
# it is a problem.

import numpy as np
from src.utils import OptimizationStruct
from src.optimization.bayesian_optimization import BayesianOptimization

from src.regression_models.naive_GMR import NaiveGMRegression

#from src.benchmarks.go_benchmark_functions.go_funcs_S import Step
from src.benchmarks.go_benchmark_functions.go_funcs_W import Weierstrass

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.benchmarks.custom_test_functions.problems import Step,SimonsTest,SimonsTest3_cosine_fuction, SimonsTest4_cosine_fuction

problem = Step(dimensions=1)
problem = SimonsTest(dimensions=1)

np.random.seed(2)

GM_reg = NaiveGMRegression(optimize=False, manipulate_variance=True)

regression_models = [GM_reg]
plt.figure(figsize=(12, 8))
outer_gs = gridspec.GridSpec(1, 1)

BOs = []
opts = []
for i in range(len(regression_models)):
    BOs.append(BayesianOptimization(problem, regression_models[i]))
    opts.append(OptimizationStruct())

import os
fig_name = "alternative_EI_naive_GMR_manipulated_variance"
path = "/home/simon/Documents/MasterThesis/master-thesis/bayes_opt_experiments/figures"
#os.mkdir(path)
for iter in range(50):
    for i in range(len(regression_models)):
        BO = BOs[i]
        opt = opts[i]
        #BO.optimize(4, type = "numeric", n_restarts=10)
        ax = outer_gs[i]
        opt = BO.optimization_step(opt, type="grid")
        BO.plot_surrogate_and_expected_improvement(ax, opt, show_name=True)
        opts[i] = opt
        number = f"{iter}".zfill(3)
    plt.savefig(f"{path}/{fig_name}_{number}.jpg")
    #plt.show()

x_hist,y_hist = BO.get_optimization_hist()
print(x_hist)