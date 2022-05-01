from tokenize import Double
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.distributions import constraints, transform_to

import pyro
import pyro.contrib.gp as gp

assert pyro.__version__.startswith('1.8.0')
pyro.set_rng_seed(1)

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

class GaussianProcess_sklearn:
    def __init__(self, extra_name="") -> None:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=1.5)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, 
                            optimizer='fmin_l_bfgs_b', 
                            n_restarts_optimizer=200, 
                            normalize_y=True)
    
        self.name = f"Gaussian Process{extra_name} - sklearn"
        self.latex_architecture = r"gp.kernels.Matern52"

    def fit(self, X,Y):
        self.model.fit(X,Y)
        self.params = f"noise = {self.model.get_params()}, only 1 dim params!"
    
    def predict(self, X_test):
        mu, sigma = self.model.predict(X_test, return_std=True)
        return mu.squeeze(), sigma, None
    


class GaussianProcess_pyro:
    def __init__(self,noise=0.01, X=None, y=None, extra_name="") -> None:
        self.name = f"Gaussian Process{extra_name} - pyro"
        self.latex_architecture = r"gp.kernels.Matern52"
        self.gpmodel = None
        self.noise = noise
        self.params = f"noise = {noise}, only 1 dim params!"

    def model(self,input_dim):
        X,y = None, None
        self.gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim = input_dim), #input_dim = input_dim
                                 noise=torch.tensor(self.noise), jitter=1.0e-4)

    def fit(self, X, Y):
        if self.gpmodel is None:
            input_dim = X.shape[1]
            self.model(input_dim)
        X = torch.tensor(X) # incorporate new evaluation
        Y = torch.tensor(Y.squeeze())
        self.gpmodel.set_data(X, Y)
        # optimize the GP hyperparameters using Adam with lr=0.001
        optimizer = torch.optim.Adam(self.gpmodel.parameters(), lr=0.001)
        gp.util.train(self.gpmodel, optimizer)
        self._update_latex()
        

    def _update_latex(self):
        lower_bound, upper_bound = 0,1
        constraint = constraints.interval(lower_bound, upper_bound)
        lengthscale_unconstrained = self.gpmodel.kernel.lengthscale_unconstrained
        variance_unconstrained = self.gpmodel.kernel.variance_unconstrained
        lengthscale = transform_to(constraint)(lengthscale_unconstrained).detach().numpy()
        variance = transform_to(constraint)(variance_unconstrained).detach().numpy()
        self.latex_architecture = f"gp.kernels.Matern52, lengthscale = {lengthscale:.2f}, var = {variance:.2f}"
        self.params = ", ".join([self.params,self.latex_architecture])

    def predict(self, X_test):
        X_test = torch.tensor(X_test)
        mu, variance = self.gpmodel(X_test, full_cov=False, noiseless=False)
        sigma = variance.sqrt()
        return mu.detach().numpy(),sigma.detach().numpy(),None



