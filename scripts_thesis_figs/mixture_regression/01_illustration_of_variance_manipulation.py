# Notes:
# Illustration why EI can get stucked 

from src.regression_models.naive_GMR import NaiveGMRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.benchmarks.custom_test_functions.problems import SimonsTest
problem = SimonsTest(dimensions=1)

np.random.seed(2)

GM_reg = NaiveGMRegression(optimize=False, manipulate_variance=False, 
                    x_component_std=0.1,
                    y_component_std=0.1,
                    prior_settings={"Ndx":0.1,"sig_prior":1, "prior_type":1})

# X_init = np.random.uniform(-.5,.5,size = (3,1))
# Y_init = [problem.fun(x) for x in X_init]


plot_navigator = input("1 or 2 or 3? ")#1#1,2

if plot_navigator == "1":
    extra_points = 0
elif plot_navigator == "2":
    extra_points = 3
elif plot_navigator == "3":
    extra_points = 10
else:
    assert False


X_init = np.array([[-.5,0,.5, *(0.01*np.random.randn(extra_points))]]).T
Y_init = np.array([[0.5,0,0.5, *(0.01*np.random.randn(extra_points))]]).T

GM_reg.fit(X_init,Y_init)
f, (ax1) = plt.subplots(1, 1, sharey=True,  sharex=True)
GM_reg.plot(ax1, xbounds=(-1,1), ybounds=(-0.5,1), plot_credible_set =False)
x_grid =np.linspace(-1,1,1000)[:,None]
mean,std, px =  GM_reg.predict(x_grid)
ax1.plot(x_grid, mean, color="red")
ax1.plot(x_grid, mean+1.96*std,"--", color="tab:red")
ax1.plot(x_grid, mean-1.96*std,"--", color="tab:red")
ax1.plot(X_init, Y_init, '.', markersize=10, color="orange")
plt.show()

# path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
# plt.savefig(f'{path}/mixture_regression_variance_problem{plot_navigator}.pdf', bbox_inches='tight',format='pdf')
