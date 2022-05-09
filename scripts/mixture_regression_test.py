
from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
import matplotlib.pyplot as plt
import numpy as np

def obj_fun(x): 
    x= x/100
    return 50 * (np.sign(x-0.5) + 1)+np.sin(100*x)*10

bounds = [0,100]
#datasize = int(input("Enter number of datapoints: "))
datasize = 10
np.random.seed(20)
X =  np.random.uniform(*bounds,size = (datasize,1))
Y = obj_fun(X)
y = Y

SPN_regression = GMRegression()
SPN_regression = SumProductNetworkRegression(
                    tracks=5,
                    channels = 50, train_epochs= 1000,
                    manipulate_varance = False, 
                    optimize=False, opt_n_iter=2,opt_cv=2)
SPN_regression.fit(X, y)

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,  sharex=True)
SPN_regression.plot(ax1, xbounds=bounds, ybounds=(-100,200))
#SPN_regression.plot(ax2, xbounds=bounds, ybounds=(-100,200))
X_test = np.linspace(0,100,1000)[:,None]
mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
ax2.plot(X_test, mean, "--", color="red")
mean = mean.squeeze()
std_deviation = std_deviation.squeeze()
ax2.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,facecolor="orange",
                            edgecolor="orange", alpha=0.9, label=r"90\% credible interval") 

ax2.set_ylim([-100,200])
ax2.plot(X, y, "*", color="grey")
ax1.plot(X, y, "*", color="grey")
plt.show()