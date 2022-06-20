import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

f, (ax) = plt.subplots(1, 1, sharey=True,  sharex=True)

xbounds = (-3, 1) #improvmenet
ybounds = (0.00001, 2) #variance
x_grid = np.linspace(*xbounds, 1000, dtype=np.float)
y_grid = np.linspace(*ybounds, 900,dtype=np.float)

imp,sigma = np.meshgrid(x_grid, y_grid)

beta = 1# 1,3
nLCB =-( -imp - beta*sigma) #assumin imp = -mu which is ok. 

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

c = ax.contourf(imp,sigma, nLCB,  cmap="Greys")
#c = ax.contourf(imp,sigma, nLCB, np.linspace(min(nLCB), max(nLCB),10), cmap="Greys")
#c = ax.contourf(imp,sigma, nLCB, np.linspace(nLCB.min(), nLCB.max(),40), cmap="twilight_shifted")
# cbar = plt.colorbar(c)
# cbar.set_ticks(list(range(6)))
# cbar.set_ticklabels(list(range(6)))

if beta == 1:
    ax.set_title(r"$nLCB_{0.841}(x)$")
else:
    ax.set_title(r"$nLCB_{0.999}(x)$")
ax.set_xlabel(r"$y_{\min}-\mu_x$")
#ax.set_ylabel(r"$\sqrt{\mathbb{V}_{p(y|x,\mathcal{D})}[y]}$")
ax.set_ylabel(r"$\sigma_x$")
ax.set_ylim(0,ybounds[1])

path_data = "/home/simon/Documents/MasterThesis/master-thesis/scripts_thesis_figs/bayesian_optimization/GP_imp_sig_data3.txt"
data = np.loadtxt(path_data)
if beta == 1:
    plt.plot(data[0], data[1],"-",lw=2,color="tab:orange", label = "AQ path")
else:
    plt.plot(data[0], data[1],"-",lw=2,color="tab:green", label = "AQ path")

if beta == 1:
    opt_location = int(len(data[0])*(125.40/200)) #optimum at 25.40 and -13.29
else:
    opt_location = int(len(data[0])*((100-13.29)/200)) #optimum at 25.40 and -13.29

plt.plot(data[0][opt_location], data[1][opt_location],".",markersize = 10,color="tab:orange",label="x=-100")
# plt.plot(data[0][-1], data[1][-1],".",markersize = 10,color="tab:blue",label="x=100")
# plt.plot(data[0][0], data[1][0],".",markersize = 10,color="tab:blue",label="x=-100")

ax.set_yticks([0,1,2])
ax.set_xticks([-3,0,1])

fig = plt.gcf()
fig.set_size_inches(3, 3)
#plt.show()
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/neg_lower_confidence_illustration_{beta}.pdf', bbox_inches='tight',format='pdf')
