import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

f, (ax) = plt.subplots(1, 1, sharey=True,  sharex=True)

xbounds = (-3, 1) #improvmenet
ybounds = (0.00001, 2) #variance
x_grid = np.linspace(*xbounds, 1000, dtype=np.float)
y_grid = np.linspace(*ybounds, 900,dtype=np.float)

imp,sigma = np.meshgrid(x_grid, y_grid)

Z = imp/sigma
exploitation = imp*norm.cdf(Z)
exploration = sigma*norm.pdf(Z)
EI = exploitation + exploration

dx = (x_grid[1] - x_grid[0]) / 2.0
dy = (y_grid[1] - y_grid[0]) / 2.0
extent = [
    x_grid[0] - dx,
    x_grid[-1] + dx,
    y_grid[0] - dy,
    y_grid[-1] + dy,
]
# ax.imshow(
#     EI,
#     extent=extent,
#     aspect="auto",
#     origin="lower",
#     cmap='Blues',
#     vmin=0, vmax=10
# )  # , vmin=-3, vmax=1)

#c = ax.contourf(imp,sigma, EI, np.linspace(EI.min(), EI.max(),40), cmap="twilight_shifted")
c = ax.contourf(imp,sigma, np.log(EI), np.linspace(-7, np.max(np.log(EI)),10), cmap="Greys")
cbar = plt.colorbar(c)
cbar.set_ticks(list(cbar.get_ticks()))
cbar.set_ticklabels(list(np.round(np.exp(cbar.get_ticks()),3)))

ax.set_title(r"$\mathbb{EI}(x)$")
ax.set_xlabel(r"$y_{\min}-\mu_x$")
#ax.set_ylabel(r"$\sqrt{\mathbb{V}_{p(y|x,\mathcal{D})}[y]}$")
ax.set_ylabel(r"$\sigma_x$")
ax.set_ylim(0,ybounds[1])

path_data = "/home/simon/Documents/MasterThesis/master-thesis/scripts_thesis_figs/bayesian_optimization/GP_imp_sig_data.txt"
data = np.loadtxt(path_data)
plt.plot(data[0], data[1],"-",lw=2,color="tab:blue", label = "AQ path")
opt_location = int(len(data[0])*(126.06/200)) #optimum at 26.06
plt.plot(data[0][opt_location], data[1][opt_location],".",markersize = 10,color="tab:orange",label="x=-100")
plt.plot(data[0][-1], data[1][-1],".",markersize = 10,color="tab:blue",label="x=100")
plt.plot(data[0][0], data[1][0],".",markersize = 10,color="tab:blue",label="x=-100")
#plt.legend()

#
#plt.show()
fig = plt.gcf()
fig.set_size_inches(8, 3)
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/expected_improvement_illustration.pdf', bbox_inches='tight',format='pdf')
