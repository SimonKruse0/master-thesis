#from statistics import covariance
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from scipy.stats import multivariate_normal

x_grid = np.linspace(-2,2,1000)
y_grid = np.linspace(-1,1,200)
dx = (x_grid[1] - x_grid[0]) / 2.0
dy = (y_grid[1] - y_grid[0]) / 2.0
x, y = np.meshgrid(x_grid, y_grid)
pos = np.dstack((x, y))
covariance = 0# -0.008
rv = multivariate_normal([-0.5, +0.5], [[0.1, covariance], [covariance, 0.02]])
rv_x = multivariate_normal([-0.5], [[0.1]])
rv2 = multivariate_normal([0.5, -0.5], [[0.1, covariance], [covariance, 0.02]])
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
gs = fig.add_gridspec(4,1, hspace=0.3)
(ax1, ax2,ax3, ax4) = gs.subplots(sharex='col', sharey='row')
ax1.plot(x_grid, p_x)
#ax1.set_yticklabels([])
ax1.grid(color='k', alpha = 0.2)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
extent = [
    x_grid[0] - dx,
    x_grid[-1] + dx,
    y_grid[0] - dy,
    y_grid[-1] + dy,
]
#ax2.contourf(x, y,p_xy, cmap="Blues")
ax2.imshow(np.log(p_xy),extent=extent,    aspect="auto",
            origin="lower", cmap="Blues",  vmin=-2, vmax=1)
img = ax3.imshow(#(x, y, p_predictive, cmap="Blues")
            #p_predictive,
            np.log(p_predictive),
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues',
            vmin=-2, vmax=1
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
ax2.set_yticklabels([])
ax3.set_yticklabels([])
ax4.set_yticklabels([])
ax2.set_ylabel("0")
ax3.set_ylabel("0")
ax4.set_ylabel("0")
ax4.set_xlabel("x")
# plt.colorbar(img, ax=ax2)
# plt.colorbar(img2, ax=ax4)
plt.show()



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