# %% Import libraries
import matplotlib.pyplot as plt
import torch
import supr
from supr.utils import drawnow
from scipy.stats import norm
from math import sqrt
import numpy as np

# %% Dataset
N = 10
x = torch.linspace(0, 1, N)
y = 1 - 2*x + (torch.rand(N) > 0.5)*(x > 0.5) + torch.randn(N)*0.1
x[x > 0.5] += 0.25
x[x < 0.5] -= 0.25

x[0] = -1.
y[0] = -0.5

X = torch.stack((x, y), dim=1)

# %% Grid to evaluate predictive distribution
x_res, y_res = 800, 1000
x_min, x_max = -2, 2
y_min, y_max = -2, 2
x_grid = torch.linspace(x_min, x_max, x_res)
y_grid = torch.linspace(y_min, y_max, y_res)
XY_grid = torch.stack([x.flatten() for x in torch.meshgrid(
    x_grid, y_grid, indexing='ij')], dim=1)
X_grid = torch.stack([x_grid, torch.zeros(x_res)]).T

# %% Sum-product network
# Parameters
tracks = 1
variables = 2
channels = 50

# Priors for variance of x and y
alpha0 = torch.tensor([[[1], [1]]])
beta0 = torch.tensor([[[.01], [.01]]])

# Construct SPN model
model = supr.Sequential(
    supr.NormalLeaf(tracks, variables, channels, n=N, mu0=0.,
                    nu0=0, alpha0=alpha0, beta0=beta0),
    supr.Weightsum(tracks, variables, channels)
)

# Marginalization query
marginalize_y = torch.tensor([False, True])

# %% Fit model and display results
epochs = 30

for epoch in range(epochs):
    model.train()
    model[0].marginalize = torch.zeros(variables, dtype=torch.bool)
    logp = model(X).sum()
    print(f"Log-posterior ∝ {logp:.2f} ")
    model.zero_grad(True)
    logp.backward()

    model.eval()  # swap?
    model.em_batch_update()

    # Plot data and model
    # -------------------------------------------------------------------------
    # Evaluate joint distribution on grid
    with torch.no_grad():
        log_p_xy = model(XY_grid)
        p_xy = torch.exp(log_p_xy).reshape(x_res, y_res)

    # Evaluate marginal distribution on x-grid
    log_p_x = model(X_grid, marginalize=marginalize_y)
    p_x = torch.exp(log_p_x)
    model.zero_grad(True)
    log_p_x.sum().backward()

    with torch.no_grad():
        # Define prior conditional p(y|x)
        Ndx = 0.9
        sig_prior = 1
        p_y = norm(0, sqrt(sig_prior)).pdf(y_grid)

        # Compute normal approximation
        m_pred = (N*(model.mean())[:, 1]*p_x + Ndx*0)/(N*p_x+Ndx)
        v_pred = (N*p_x*(model.var()[:, 1]+model.mean()[:, 1]
                  ** 2) + Ndx*sig_prior)/(N*p_x+Ndx) - m_pred**2
        v_pred2 = v_pred.clone()
        v_pred /= torch.clamp(p_x*40, 1,40)
        std_pred2 = torch.sqrt(v_pred2)
        std_pred = torch.sqrt(v_pred)

        # Compute predictive distribution
        p_predictive = (N*p_xy + Ndx*p_y[None, :]) / (N*p_x[:, None] + Ndx)

        # Compute 95% highest-posterior region
        hpr = torch.ones((x_res, y_res), dtype=torch.bool)
        for k in range(x_res):
            p_sorted = -np.sort(-(p_predictive[k] * np.gradient(y_grid)))
            i = np.searchsorted(np.cumsum(p_sorted), 0.95)
            idx = (p_predictive[k]*np.gradient(y_grid)) < p_sorted[i]
            hpr[k, idx] = False

        # Plot posterior
        plt.figure(1).clf()
        plt.title('Predictive distribution')
        dx = (x_max-x_min)/x_res/2
        dy = (y_max-y_min)/y_res/2
        extent = [x_grid[0]-dx, x_grid[-1]+dx, y_grid[0]-dy, y_grid[-1]+dy]
        plt.imshow(torch.log(p_predictive).T, extent=extent,
                   aspect='auto', origin='lower',
                   vmin=-4, vmax=1, cmap='Blues')
        plt.contour(hpr.T, levels=1, extent=extent)
        plt.plot(x, y, '.', color='tab:orange', alpha=.5,
                 markersize=15, markeredgewidth=0)
        plt.axis('square')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        drawnow()

        # Plot normal approximation to posterior
        plt.figure(2).clf()
        plt.title('Predictive Normal approximation')
        plt.plot(x, y, '.', color='tab:orange', alpha=.5,
                 markersize=15, markeredgewidth=0)
        plt.plot(x_grid, m_pred, color='tab:orange')
        plt.fill_between(x_grid, m_pred+1.96*std_pred, m_pred -
                         1.96*std_pred, color='tab:purple', alpha=0.1)
        plt.fill_between(x_grid, m_pred+1.96*std_pred2, m_pred -
                         1.96*std_pred2, color='tab:orange', alpha=0.1)
        plt.axis('square')
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        drawnow()
plt.show()