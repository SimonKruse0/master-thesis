import GPy
#GPy.plotting.change_plotting_library('plotly')

import numpy as np

from src.utils import normalize, denormalize


class GaussianProcess_GPy:
    def __init__(self, extra_name="") -> None:
        self.name = f"Gaussian Process{extra_name} - GPy"
        self.latex_architecture = r"gp.kernels.Matern52"

    def fit(self, X,Y):
        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)
        kernel = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=1.)
        self.model = GPy.models.GPRegression(X,Y,kernel)
        self.model.optimize_restarts(num_restarts = 10)
        #self.model.parameters[0].lengthscale
        #self.model.parameters[0].variance
        #self.model.parameters[1].variance #noise
        #self.params = f"noise = {self.model.alpha:0.2e}, length_scale = {self.model.kernel_.length_scale:0.2f}"
    
    def predict(self, X_test):
        X_test, *_ = normalize(X_test,self.x_mean, self.x_std)
        mu, sigma = self.model.predict(X_test)
        #transform back to original space #obs validate this!
        mu = denormalize(mu, self.y_mean, self.y_std)
        sigma = sigma*self.y_std
        return mu.squeeze(), sigma.squeeze(), None
    
def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

def obj_fun_nd(x): 
    return np.sum(0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1, axis = 1)


if __name__ == "__main__":
    bounds = [0,1]
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 200
    np.random.seed(20)
    xdim = 1
    X_sample =  np.random.uniform(*bounds,size = (datasize,xdim))
    Y_sample = obj_fun_nd(X_sample)[:,None]

    reg_model = GaussianProcess_GPy()
    reg_model.fit(X_sample, Y_sample)