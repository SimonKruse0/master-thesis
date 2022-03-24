import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import PlottingClass
from numpyro_neural_network import NumpyroNeuralNetwork #JAX
from bohamiann import BOHAMIANN #Torch


class BayesianOptimization(PlottingClass):
    def __init__(self, objectivefunction, regression_model,X_init,Y_init) -> None:
        self.obj_fun = objectivefunction
        self.model = regression_model
        self._X = X_init #OBS: should X be stored here or in the model?!
        self._Y = Y_init
        self.f_best = np.min(Y_init) # incumbent np.min(Y) or np.min(E[Y]) ?? BOHAMIANN does this
        self.bounds = ((0,1),)
        super().__init__() #initalize plotting functions

        print(f"\n-- initial training -- \t {self.model.name}")
        self.model.fit(X_init,Y_init)

    # OBS put plot functioner i plot_class!

    def predict(self,X, gaussian_approx = True):
        if gaussian_approx:
            Y_mu,Y_sigma,_ = self.model.predict(X)
            return Y_mu,Y_sigma
        else:
            Y_mu,_,Y_CI = self.model.predict(X)
            return Y_mu,Y_CI

    def expected_improvement(self,X,xi=0.01):
        #print("OBS X[:,None] might fail in largers dims!")
        mu, sigma = self.predict(X[:,None]) #Partial afledt. Pytorch. 
        imp = -mu - self.f_best - xi
        Z = imp/sigma
        EI = (imp*norm.cdf(Z) + sigma*norm.pdf(Z))
        return EI

    def find_a_candidate(self):
        n_restarts = 1
        x_next = np.nan
        max_EI = 1e-5
        _,nx = self._X.shape

        def min_obj(x):
            EI = self.expected_improvement(x)
            return -EI

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[0][0], self.bounds[0][1],
                                    size=(n_restarts, nx)):
            res = minimize(min_obj, x0=x0,bounds=self.bounds, method='Nelder-Mead')        
            if -res.fun > max_EI:
                max_EI = res.fun
                x_next = res.x
        return x_next,max_EI

    def optimization_step(self,x_next = None):
        if x_next is not None:
            y_next = self.obj_fun(x_next)
            self._X = np.vstack((self._X, x_next))
            self._Y = np.vstack((self._Y, y_next))
            self.model.fit(self._X,self._Y)
        x_next, max_EI = self.find_a_candidate()
        return x_next,max_EI

    def optimize(self,num_steps:int):
        x_next = None
        for i in range(num_steps):
            print(f"-- finding x{i+1} --",end="\n")
            x_next,_ = self.optimization_step(x_next)
            print(f"-- x{i+1} = {x_next[0]:.2f} --")
        Best_Y_id = np.argmin(self._Y)
        Best_X = self._X[Best_Y_id]
        Best_Y = self._Y[Best_Y_id]
        print(f"-- End of optimization -- best x = {Best_X} with objective y = {Best_Y}")


def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

#import pickle

if __name__ == "__main__":
    plt.figure(figsize=(12, 8))
    outer_gs = gridspec.GridSpec(2, 1)

    bounds = np.array([[0,1]])
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 20
    np.random.seed(2)
    X_sample =  np.random.uniform(*bounds[0],size = datasize)
    X_sample = X_sample[:,None]
    Y_sample = obj_fun(X_sample)
    
    BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 300)
    BO_BOHAMIANN = BayesianOptimization(obj_fun, BOHAMIANN_regression,X_sample,Y_sample)
    #x_next = BO_BOHAMIANN.plot_surrogate_and_expected_improvement(outer_gs[0], name = True)

    NNN = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=100, keep_every = 50)
    BO_BNN = BayesianOptimization(obj_fun, NNN,X_sample,Y_sample)
    #x_next = BO_BNN.plot_surrogate_and_expected_improvement(outer_gs[1], name = True)
    #plt.show()
    
    BO_BNN.optimize(10)
    BO_BOHAMIANN.optimize(10)


