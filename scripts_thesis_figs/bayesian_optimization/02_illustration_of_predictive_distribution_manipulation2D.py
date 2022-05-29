#from statistics import covariance
import matplotlib.pyplot as plt
#plt.style.use('seaborn-whitegrid')
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


mpl.rcParams['lines.linewidth'] = 2
x_grid = np.linspace(-2,2,1000)
y_grid = np.linspace(-1,1,200)
dx = (x_grid[1] - x_grid[0]) / 2.0
dy = (y_grid[1] - y_grid[0]) / 2.0

extent = [
    x_grid[0] - dx,
    x_grid[-1] + dx,
    y_grid[0] - dy,
    y_grid[-1] + dy,
]

x, y = np.meshgrid(x_grid, y_grid)
pos = np.dstack((x, y))
covariance = -0.02
rv = multivariate_normal([-0.5, +0.5], [[0.1, covariance], [covariance, 0.04]])
rv_x = multivariate_normal([-0.5], [[0.1]])
rv2 = multivariate_normal([0.5, -0.5], [[0.1, covariance], [covariance, 0.04]])
rv2_x = multivariate_normal([0.5], [[0.1]])

weigth1 = 0.75
weigth2 = 1-weigth1

p_xy = weigth1*rv.pdf(pos)+weigth2*rv2.pdf(pos)
p_x = weigth1*rv_x.pdf(x_grid)+weigth2*rv2_x.pdf(x_grid)
p_prior_y = multivariate_normal([0], [[0.2]]).pdf(y_grid)

N = 2

mixture_mean_y = weigth1*0.5+weigth2*(-0.5)
mean = N*mixture_mean_y/(N*p_x+1)

p_predictive3 = np.ones_like(p_xy)*p_prior_y[:,None] #(1*p_xy+p_prior_y[:,None])/(1*p_x[:,None].T+1)
p_predictive = p_xy/p_x[:,None].T #(1*p_xy+p_prior_y[:,None])/(1*p_x[:,None].T+1)
p_predictive2 = (N*p_xy+p_prior_y[:,None])/(N*p_x[:,None].T+1)

# p_predictive *= np.gradient(y_grid)[:,None]
# p_predictive2 *= np.gradient(y_grid)[:,None]

fig = plt.figure()
gs = fig.add_gridspec(3,1, hspace=0.3)
(ax2,ax3, ax4) = gs.subplots(sharex='col', sharey='row')
ax2.imshow(np.log(p_xy),extent=extent,    aspect="auto",
            origin="lower", cmap="Blues",  vmin=-2, vmax=1)
ax1 = ax2.twinx()  # instantiate a second axes that shares the same x-axis
color = 'red'
ax1.plot(x_grid, p_x, color = color)
#ax1.set_yticklabels([])
ax1.set_ylabel('p(x)', color=color)
ax1.grid(color=color, alpha = 0.2)
# ax1.spines['top'].set_visible(False)
# ax1.spines['right'].set_visible(False)
# ax1.spines['bottom'].set_visible(False)
# ax1.spines['left'].set_visible(False)

ax1.tick_params(axis='y', labelcolor=color)

#ax2.contourf(x, y,p_xy, cmap="Blues")
img = ax3.imshow(#(x, y, p_predictive, cmap="Blues")
            #p_predictive,
            np.log(p_predictive),
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues',
            vmin=-2, vmax=1,
            label=r"$p(x,y)$"
        )
#ax2.colorbar()
img2 = ax4.imshow(#(x, y, p_predictive, cmap="Blues")
            #p_predictive2,
            np.log(p_predictive2),
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues',
            vmin=-2, vmax=1
        )
# ax2.set_yticklabels([])
# ax3.set_yticklabels([])
# ax4.set_yticklabels([])
ax2.set_ylabel("y")
ax3.set_ylabel("y")
ax4.set_ylabel("y")
ax4.set_xlabel("x")

#ax2.set_title(r"$p(x,y)$")
# ax3.set_title(r"$p(y|x)$")
# ax4.set_title(r"$\hat p(y|x)$")

# plt.colorbar(img, ax=ax2)
# plt.colorbar(img2, ax=ax4)
cmap = plt.cm.Blues
legend_elements = [Patch(facecolor=cmap(0.6), edgecolor=cmap(0.6),
                         label='p(x,y)')]
ax2.legend(handles=legend_elements)
legend_elements = [Patch(facecolor=cmap(0.6), edgecolor=cmap(0.6),
                         label='p(y|x)')]
ax3.legend(handles=legend_elements)
legend_elements = [Patch(facecolor=cmap(0.6), edgecolor=cmap(0.6),
                         label=r'$\hat p(y|x)$')]
ax4.legend(handles=legend_elements)
#plt.show()
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/mixture_predictive_bayesian2D.pdf', bbox_inches='tight',format='pdf')



# x = np.linspace(0, 10, 50)
# x2 = np.linspace(0, 10, 100)
# y = np.sin(x)
# y2 = np.sin(x2)



# model = GMRegression(manipulate_variance=True)
# model.fit(x[:,None],y[:,None])
# ax = plt.subplot()
# model.plot(ax,  xbounds=(0,10),ybounds=(-1.5,1.5))
# ax.plot(x, y, 'o', color='black');
# ax.plot(x2, y2, '-', color='red');
# plt.show()

# plt.plot(x, y, 'o', color='black');
# plt.plot(x2, y2, '-', color='red');
# plt.show()

# model = SumProductNetworkRegression(manipulate_varance=True)
# model.fit(x[:,None],y[:,None])
# ax = plt.subplot()
# model.plot(ax,  xbounds=(0,10),ybounds=(-1.5,1.5))
# ax.plot(x, y, 'o', color='black');
# ax.plot(x2, y2, '-', color='red');
# plt.show()