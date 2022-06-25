from src.regression_validation.reg_validation import PlotReg1D_mixturemodel
from src.benchmarks.custom_test_functions.problems import Test1,Test2, Test3c, Test4c,Test4b, Test3b
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
# run_name = datetime.today().strftime('%m%d_%H%M')
# dirname=os.path.dirname
# path = os.path.join(dirname(dirname(os.path.abspath(__file__))),f"data/{run_name}")
# try:
#     os.mkdir(path)
# except:
#     print(f"Couldn't create {path}")

def find_minima(plot_reg):
    X = np.linspace(-100,100,10000)[:,None]
    y = plot_reg.obj_fun(X)
    return X[np.argmin(y)][0], np.min(y)

for problem_sklearn in [Test1(),Test2(),Test3b(),Test4b(), Test3c(),Test4c()]:
    fig, ax = plt.subplots()
    plot_reg = PlotReg1D_mixturemodel(None, problem_sklearn, disp=False)

    plot_reg.bounds = (plot_reg.bounds[0]-10, plot_reg.bounds[1]+10)
    plt.ylim(0,300)
    plt.xlim(-110,110)
    try:
        plot_reg.plot_true_function2(ax)
    except:
        plot_reg.plot_true_function(ax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    prob_name = type(problem_sklearn).__name__[:6]
    ax.set_title(prob_name[:5])
    print(find_minima(plot_reg))
    ax.plot(*find_minima(plot_reg),".", markersize = 10, color="red")
    path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Figures/reg_illustrations/all_reg_figures"

    plt.tight_layout()
    # plt.show()
    fig = plt.gcf()
    fig.set_size_inches(3, 3)
    plt.savefig(f"{path}/{prob_name}.pdf")