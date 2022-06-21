# from src.regression_models.SPN_regression2 import SumProductNetworkRegression
# import numpy as np
# import matplotlib.pyplot as plt

# X = np.array([1,1,1,2,2,2,3,3,3])
# Y = np.array([1,2,3,1,2,3,1,2,3])

# SPN = SumProductNetworkRegression(tracks=1, channels=6, alpha0_x=1, alpha0_y=1, beta0_x=0.01, beta0_y=0.01)
# SPN.fit(X[:,None], Y[:,None])
# f, (ax1) = plt.subplots(1, 1, sharey=True,  sharex=True)
# SPN.plot(ax1,  xbounds=(-0.5,3.5),ybounds=(-0.5,3.5))
# plt.show()

# %% Import libraries
from cProfile import label
import enum
import matplotlib.pyplot as plt
import torch
import src.regression_models.supr as supr
from src.regression_models.supr.utils import drawnow
from scipy.stats import norm
from math import sqrt
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib

cmap = matplotlib.cm.Blues
cmap.set_bad('w',1.)


plot_navigator = 3#1,2,3,4
Nc = 30
#Nc = 20
K = 3
N = K**2*Nc

if plot_navigator == 1 or plot_navigator > 2:
    sigma_x = torch.tensor([[[.01], [.01], [.01]], [[.04], [.04], [.04]],[[.08], [.08], [.08]]])
    sigma_y = torch.tensor([[[.01], [.04], [.08]], [[.01], [.04], [.08]],[[.01], [.04], [.08]]])
elif plot_navigator == 2:
# sigma_x = torch.tensor([[[.01], [.04], [.08]], [[.01], [.04], [.08]],[[.01], [.04], [.08]]])
# sigma_y = torch.tensor([[[.01], [.04], [.08]], [[.01], [.04], [.08]],[[.01], [.04], [.08]]])

    sigma_y = torch.tensor([[[.01], [.01], [.01]], [[.04], [.04], [.04]],[[.08], [.08], [.08]]])
    sigma_x = torch.tensor([[[.01], [.04], [.08]], [[.01], [.04], [.08]],[[.01], [.04], [.08]]])

mu_x = torch.linspace(0, 1, K)
mu_y = torch.linspace(0, 1, K)
x, y = torch.meshgrid((mu_x, mu_y))
x = torch.flatten(x[:,:,None] + torch.randn((K, K, Nc))*sigma_x)
y = torch.flatten(y[:,:,None] + torch.randn((K, K, Nc))*sigma_y)

if plot_navigator == 3:
    x=x[30:-57]
    y=y[30:-57]
elif plot_navigator==4:
    mask = np.array(np.random.randint(0,2,size=len(x)), dtype=bool)
    mask[:60] = False 
    x=x[mask]
    y=y[mask]

X = torch.stack((x, y), dim=1)
# %% Grid to evaluate predictive distribution
x_res, y_res = 400, 400
x_min, x_max = -.25, 1.35
y_min, y_max = -.25, 1.35
x_grid = torch.linspace(x_min, x_max, x_res)
y_grid = torch.linspace(y_min, y_max, y_res)
XY_grid = torch.stack([x.flatten() for x in torch.meshgrid(
    x_grid, y_grid, indexing='ij')], dim=1)
X_grid = torch.stack([x_grid, torch.zeros(x_res)]).T

# %% Sum-product network
# Parameters
tracks = 1
variables = 2
channels = 3

# Priors for variance of x and y
alpha0 = torch.tensor([[[1], [1]]])
beta0 = torch.tensor([[[.01], [.01]]])

# Construct SPN model
model = supr.Sequential(
    supr.NormalLeaf(tracks, variables, channels, n=N, mu0=0.,
                    nu0=0, alpha0=alpha0, beta0=beta0),
    supr.Einsum(tracks, variables, channels, 1),
    supr.Weightsum(tracks, variables, 1)
)

# %% Fit model and display results
epochs = 200

for epoch in range(epochs):
    model.train()
    model[0].marginalize = torch.zeros(variables, dtype=torch.bool)
    logp = model(X).sum()
    print(f"Log-posterior ∝ {logp:.2f} ")
    model.zero_grad(True)
    logp.backward()

    model.eval()  # swap?
    model.em_batch_update()

#Check if fittet correctly
if plot_navigator == 2:
    assert logp>200
if plot_navigator == 1:
    assert logp>400
if plot_navigator == 3:
    assert logp>300

# Plot data and model
# -------------------------------------------------------------------------
# Evaluate joint distribution on grid
with torch.no_grad():
    log_p_xy = model(XY_grid)
    p_xy = torch.exp(log_p_xy).reshape(x_res, y_res)

    # Plot posterior
    #plt.figure(1).clf()
    plt.title('SPN with 3 channels')
    dx = (x_max-x_min)/x_res/2
    dy = (y_max-y_min)/y_res/2
    extent = [x_grid[0]-dx, x_grid[-1]+dx, y_grid[0]-dy, y_grid[-1]+dy]
    plt.imshow(torch.log(p_xy).T, extent=extent,
                aspect='auto', origin='lower',
                vmin=-4, vmax=1, cmap=cmap)
    # plt.plot(x, y, '*', color='black', alpha=0.5,
    #             markersize=10, markeredgewidth=0, label="data")
    plt.plot(x, y, '.', color='tab:orange', alpha=.5,
                 markersize=10, markeredgewidth=0)
                #markersize=0, markeredgewidth=10, label="data")
    plt.axis('square')
    off_set = 0
    plt.xlim([x_min+off_set, x_max+off_set])
    plt.ylim([y_min+off_set, y_max+off_set])
    plt.xticks(mu_x,[f"{x:.1f}" for x in mu_x.numpy()])
    plt.yticks(mu_y,[f"{x:.1f}" for x in mu_y.numpy()])
    #plt.legend()

x_grid = np.linspace(-2,2,1000)
y_grid = np.linspace(-2,2,1000)
with torch.no_grad():
    mu = model[0].mu.numpy()
    sig = model[0].sig.numpy()
    weights = model[1].weights.numpy().squeeze()
mu = mu.squeeze()
sig = sig.squeeze()

print("sig", sig)
print("mu", mu.round(2))

if plot_navigator >2:
    for i,m_x1 in enumerate(mu.round(2)[0]):
        for j, m_x2 in enumerate(mu.round(2)[1]):
            plt.text(m_x1+0.1,m_x2+0.1,weights[i][j].round(2))

p_x = np.zeros_like(x_grid)
for i in range(len(mu[0])): 
    rv_x = multivariate_normal([mu[0][i]], [[sig[0][i]]])
    p_x += 1/len(mu[0])*rv_x.pdf(x_grid)

p_y = np.zeros_like(y_grid)
for i in range(len(mu[1])): 
    rv_y = multivariate_normal([mu[1][i]], [[sig[1][i]]])
    p_y += 1/len(mu[1])*rv_y.pdf(y_grid)

# plt.plot(x_grid, p_x/50+x_min, "-",color = "green")
# plt.plot(p_y/50+x_min,y_grid, "-",color = "green")
off_set = 0
#plt.fill_betweenx(y_grid, 0*p_y+y_min+off_set, p_y/30+y_min+off_set, color = "#D0021B")
## OG!
# plt.fill_between(x_grid, 0*p_x+x_min+off_set, (4+np.log(p_x))/40+x_min+off_set, color = "green", alpha = 0.2,label="x-leaf dists")
# plt.fill_betweenx(y_grid, 0*p_y+y_min+off_set, (4+np.log(p_y))/40+y_min+off_set, color = "red", alpha = 0.2, label="y-leaf dists")

plt.plot(x_grid,(4+np.log(p_x))/40+x_min+off_set, color = "green", alpha = 1,label="x-leaf dists")
plt.plot((4+np.log(p_y))/40+y_min+off_set,y_grid, color = "red", alpha = 1, label="y-leaf dists")


#plt.legend()

#plt.

#plt.show()
path = "/home/simon/Documents/MasterThesis/master-thesis/thesis/Pictures"
plt.savefig(f'{path}/SPN_illustration{plot_navigator}.pdf', bbox_inches='tight',format='pdf')
