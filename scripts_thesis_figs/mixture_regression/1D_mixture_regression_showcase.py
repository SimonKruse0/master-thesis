from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.naive_GMR import NaiveGMRegression
#from src.regression_models.gaussian_mixture_regression2 import GMRegression
import matplotlib.pyplot as plt
import numpy as np

##
mixture_model = "Naive GMR"#SPN, Naive GMR
mixture_model = "SPN"#SPN, Naive GMR
N = 100
N = 10

# %% Dataset
x = np.linspace(0, 1, N)
y = 1 - 2*x + (np.random.rand(N) > 0.5)*(x > 0.5) + np.random.randn(N)*0.1
x[x > 0.5] += 0.25
x[x < 0.5] -= 0.25

x[0] = -1.
y[0] = -0.5

x_min, x_max = -2, 2
y_min, y_max = -2, 2


bounds = [x_min, x_max]

X = x[:,None]
Y = y[:,None]

prior_settings={ "Ndx": 1,"sig_prior": 1.2, "prior_type":1}

if  mixture_model =="Naive GMR":
    mixture_regression = NaiveGMRegression(manipulate_variance=False, optimize=True, 
                    opt_n_iter=30,opt_cv=min(N, 20), 
                    prior_settings = prior_settings)

elif mixture_model =="SPN":
    mixture_regression = SumProductNetworkRegression(
                    tracks=2,
                    channels = 20, train_epochs= 1000,
                    manipulate_variance = False, 
                    optimize=True, opt_n_iter=30,opt_cv=N, 
                    prior_settings=prior_settings)
else:
    assert False

mixture_regression.fit(X, Y)


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,  sharex=True)
xbounds = (x_min, x_max)
ybounds = (y_min, y_max)
mixture_regression.plot(ax1, xbounds=xbounds, ybounds=ybounds)
X_test = np.linspace(*xbounds,100)[:,None]
mean,std_deviation,Y_CI = mixture_regression.predict(X_test)
ax2.plot(X_test, mean, "--", color="red")
mean = mean.squeeze()
std_deviation = std_deviation.squeeze()
ax2.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,facecolor="orange",
                            edgecolor="orange", alpha=0.9, label=r"90\% credible interval") 

# ax2.plot(X, y, "*", color="grey")
# ax1.plot(X, y, "*", color="grey")
ax1.plot(X, y, '.', color='black', alpha=.5,
                 markersize=10, markeredgewidth=0)
ax2.plot(X, y, '.', color='black', alpha=.5,
                 markersize=10, markeredgewidth=0)
ax1.set_ylim(-3,2)
ax1.set_title("Predictive distribution")
ax2.set_title("Gaussian approximation")

plt.suptitle(f"{mixture_regression.name}")
#plt.suptitle(f"{mixture_regression.name}({mixture_regression.params})")
plt.show()
# path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
# plt.savefig(f'{path}/1D_mixture_regression_{mixture_model}_N_{N}.pdf', bbox_inches='tight',format='pdf')
