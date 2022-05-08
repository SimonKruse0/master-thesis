
from regression_models.SPN_regression2 import SumProductNetworkRegression
from regression_models.gaussian_mixture_regression2 import GMRegression
import matplotlib.pyplot as plt
import numpy as np

def obj_fun(x): 
    x= x/100
    return 50 * (np.sign(x-0.5) + 1)+np.sin(100*x)*10

bounds = [0,100]
#datasize = int(input("Enter number of datapoints: "))
datasize = 100
np.random.seed(20)
X =  np.random.uniform(*bounds,size = (datasize,1))
Y = obj_fun(X)
y = Y.squeeze()

# SPN_regression = GMRegression()
SPN_regression = SumProductNetworkRegression(
                    tracks=5,
                    channels = 50, train_epochs= 1000,
                    manipulate_varance = True)
SPN_regression.fit(X, y, optimize=True, opt_n_iter=4,opt_cv=3)
#SPN_regression.optimize_and_fit(X, y, n_iter=20,cv=3)

fig, ax = plt.subplots()
SPN_regression.plot(ax, xbounds=bounds, ybounds=(-100,200))

X_test = np.linspace(0,100,100)[:,None]
mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
ax.plot(X_test, mean, "--", color="black")
mean = mean.squeeze()
std_deviation = std_deviation.squeeze()
ax.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,
                            color="black", alpha=0.3, label=r"90\% credible interval") 
ax.plot(X, y, "*")
plt.show()