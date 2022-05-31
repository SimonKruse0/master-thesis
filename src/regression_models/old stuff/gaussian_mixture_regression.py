from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
#from gmr import GMM
from gmr import GMM, plot_error_ellipses
#OBS changes were done in self.condition in GMM. !!! 

import numpy as np

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

N = 20
np.random.seed(20)
x =  np.random.uniform(0,1,size = (N,1))

XY_train = np.column_stack((x, y))

n_components = 10

#gmm_sklearn = GaussianMixture(n_components=n_components, covariance_type="full")
gmm_sklearn = BayesianGaussianMixture(n_components=n_components,
                                        covariance_type = "full",
                                        weight_concentration_prior = 1. / n_components / 10000, 
                                        mean_precision_prior = 0.001, 
                                        covariance_prior =np.array([[0.001,0.001],[0.001,0.002]]))
                                        #covariance_prior =np.diag(np.diag(np.cov(XY_train.T)))/1000)
                                        #weight_concentration_prior_type="dirichlet_distribution")
gmm_sklearn.fit(XY_train)


gmm = GMM(
    n_components=n_components, priors=gmm_sklearn.weights_, means=gmm_sklearn.means_,
    covariances=gmm_sklearn.covariances_)#np.array([np.diag(c) for c in gmm_sklearn.covariances_]))

# gmm = GMM(
#     n_components=N, priors=np.repeat(1/N, N), means=XY_train,
#     covariances=np.repeat([np.eye(2)/10000], N, axis=0))#np.array([np.diag(c) for c in gmm_sklearn.covariances_]))


# gmm.condition()




"""
=====================
Multimodal Regression
=====================
In multimodal regression we do not try to fit a function f(x) = y but a
probability distribution p(y|x) with more than one peak in the probability
density function.
The dataset that we use to illustrate multimodal regression by Gaussian
mixture regression is from Section 5 of
C. M. Bishop, "Mixture Density Networks", 1994,
https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
On the left side of the figure you see the training data and the fitted
GMM indicated by ellipses corresponding to its components. On the right
side you see the predicted probability density p(y|x=0.5). There are
three peaks that correspond to three different valid predictions. Each
peak is represented by at least one of the Gaussians of the GMM.
"""
print(__doc__)

#import numpy as np
import matplotlib.pyplot as plt


# def f(y, random_state):
#     eps = random_state.rand(*y.shape) * 0.2 - 0.1
#     return y + 0.3 * np.sin(2.0 * np.pi * y) + eps


# y = np.linspace(0, 1, 1000)
# random_state = np.random.RandomState(3)
# x = f(y, random_state)

# XY_train = np.column_stack((x, y))
# gmm = GMM(n_components=4, random_state=random_state)
# gmm.from_samples(XY_train)

plt.figure(figsize=(10, 5))

ax = plt.subplot(121)
ax.set_title("Dataset and GMM")
colors = ["r", "g", "b", "orange"]*100
#colors = ["r"]*100
plot_error_ellipses(ax, gmm, colors=colors, alpha = 0.1)

Y = np.linspace(-0.1, 1.2, 1000)
Y_test = Y[:, np.newaxis]
X_test = 0.5
conditional_gmm = gmm.condition([0], [X_test])
p_of_Y = conditional_gmm.to_probability_density(Y_test)/20
ax.plot(X_test+p_of_Y, Y_test)

x_test_list = np.linspace(-1,2,500)
CI = []
CI1 = []
CI2 = []
for x_test in x_test_list:
    conditional_gmm = gmm.condition([0], [x_test], manipulate_test_bounds = True)
    y_given_x_samples = conditional_gmm.sample(n_samples=2000)
    CI.append(np.quantile(y_given_x_samples, [0.05, 0.95]))
    CI1.append(np.quantile(y_given_x_samples, [0.15, 0.85]))
    CI2.append(np.quantile(y_given_x_samples, [0.35, 0.65]))
CI = np.array(CI)
CI1 = np.array(CI1)
CI2 = np.array(CI2)
ax.plot(X_test+p_of_Y, Y_test)
plt.fill_between(x_test_list, *CI.T,color="black", alpha=0.2)
plt.fill_between(x_test_list, *CI1.T,color="black", alpha=0.2)
plt.fill_between(x_test_list, *CI2.T,color="black",alpha=0.2)

ax.scatter(x, y, s=3)
ax.set_xlabel("x")
ax.set_ylabel("y")

# ax = plt.subplot(122)
# ax.set_title("Conditional Distribution")
# Y = np.linspace(0, 1.1, 1000)
# Y_test = Y[:, np.newaxis]
# X_test = 0.5
# conditional_gmm = gmm.condition([0], [X_test])
# p_of_Y = conditional_gmm.to_probability_density(Y_test)
# ax.plot(Y, p_of_Y, color="k", label="GMR", lw=3)
# for component_idx in range(conditional_gmm.n_components):
#     p_of_Y = (conditional_gmm.priors[component_idx]
#               * conditional_gmm.extract_mvn(
#                 component_idx).to_probability_density(Y_test))
#     ax.plot(Y, p_of_Y, color=colors[component_idx],
#             label="Component %d" % (component_idx + 1))
# ax.set_xlabel("y")
# ax.set_ylabel("$p(y|x=%.1f)$" % X_test)
# #ax.legend(loc="best")

plt.tight_layout()
plt.show()