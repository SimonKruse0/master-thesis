#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import supr
from supr.utils import drawnow
from scipy.stats import norm

#%% Dataset
N = 10
x = torch.linspace(0, 1, N)
# y = -1*x + (torch.rand(N)>0.5)*(x>0.5) + torch.randn(N)*0.1
x[x>0.5] += 0.25
x[x<0.5] -= 0.25
def obj_fun(x): 
    return x*0.5 * (torch.sign(x-0.5) + 1)+torch.sin(100*x)*0.1
y = obj_fun(x)

X = torch.stack((x,y), dim=1)

#%% Grid to evaluate predictive distribution
x_grid = torch.linspace(-1, 2, 300)
y_grid = torch.linspace(-2, 2, 300)
X_grid = torch.stack([x.flatten() for x in torch.meshgrid(x_grid, y_grid, indexing='ij')], dim=1)

#%% Sum-product network
tracks = 3
variables = 2
channels = 20

# Priors for variance of x and y
alpha0 = torch.tensor([[[0.02], [0.01]]])
beta0 =  torch.tensor([[[0.02], [0.01]]])

model = supr.Sequential(
    supr.NormalLeaf(tracks, variables, channels, n=N, mu0=0., nu0=0, alpha0=alpha0, beta0=beta0),
    supr.Weightsum(tracks, variables, channels)
    )

#%% Fit model and display results
epochs = 10

for epoch in range(epochs):
    model.train()
    model[0].marginalize = torch.zeros(variables, dtype=torch.bool)    
    loss = model(X).sum()
    print(f"Loss = {loss}")
    loss.backward()
    
    with torch.no_grad():
        model.eval()
        model.em_batch_update()
        model.zero_grad(True)
        
        p_xy = torch.exp(model(X_grid).reshape(len(x_grid), len(y_grid)).T)

        model[0].marginalize = torch.tensor([False, True])
        p_x = torch.exp(model(X_grid).reshape(len(x_grid), len(y_grid)).T)
        
        p_prior = norm(0, 0.5).pdf(y_grid)[:,None]
        
        #p_predictive = (N*p_xy + p_prior)/(N*p_x+1)
        p_predictive = (p_xy/p_x + p_prior/N)
        bound = torch.FloatTensor([0.01])
        p_predictive *= 0.0001/torch.max(bound.expand_as(p_x),p_x)


        plt.figure(1).clf()
        dx = (x_grid[1]-x_grid[0])/2.
        dy = (y_grid[1]-y_grid[0])/2.
        extent = [x_grid[0]-dx, x_grid[-1]+dx, y_grid[0]-dy, y_grid[-1]+dy]
        plt.imshow(torch.log(p_predictive), extent=extent, aspect='auto', origin='lower')#, vmin=-3, vmax=1)
        plt.plot(x, y, '.')
        drawnow()
plt.show()

