#from statistics import covariance
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2

y_grid = np.linspace(-2,2,1000)

rv_y = multivariate_normal([-0.5], [[0.1]])
rv_y2 = multivariate_normal([+0.3], [[0.1]])

p_y = rv_y.pdf(y_grid)+rv_y2.pdf(y_grid)
p_y *= 0.5
p_prior_y = multivariate_normal([0], [[2]]).pdf(y_grid)

p_x = 0.1
p_x2 = 0.5
p_x3 = 2
N = 1


p_predictive = (p_x*N*p_y+p_prior_y)/(p_x*N+1)
p_predictive2 = (p_x2*N*p_y+p_prior_y)/(p_x2*N+1)
p_predictive3 = (p_x3*N*p_y+p_prior_y)/(p_x3*N+1)

# p_predictive *= np.gradient(y_grid)[:,None]
# p_predictive2 *= np.gradient(y_grid)[:,None]

fig = plt.figure()
gs = fig.add_gridspec(3,1, hspace=0)
(ax1,ax2,ax3) = gs.subplots(sharex='col', sharey='col')

ax1.plot(y_grid, p_y, color = "black")
ax2.plot(y_grid, p_prior_y,color = "black")
ax3.plot(y_grid, p_predictive, label=r"$p(x) = 0.1$")
ax3.plot(y_grid, p_predictive2,label=r"$p(x) = 0.5$")
ax3.plot(y_grid, p_predictive3, label=r"$p(x) = 2$")
ax1.text(-1.9, 0.5, r"$p(y|x)$", fontsize=12)
ax2.text(-1.9, 0.5, r"$p_{prior}(y)$", fontsize=12)
ax3.text(-1.9, 0.5,r"$\hat p(y|x)$", fontsize=12)

# ax1.set_title(r"$p(y|x)$")
# ax2.set_title(r"$p_{prior}(y)$")
# ax3.set_title(r"$\hat p(y|x)$")
ax3.legend()

#plt.setp(plt.gcf().get_axes(),grid = True, xticks=[], yticks=[]);
ax1.grid(True)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.show()
#plt.savefig(f'{path}/mixture_predictive_bayesian.eps', bbox_inches='tight',format='eps')