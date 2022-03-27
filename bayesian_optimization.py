import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import PlottingClass, OptimizationStruct
from numpyro_neural_network import NumpyroNeuralNetwork #JAX
from bohamiann import BOHAMIANN #Torch

PLOT_NR = 0

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

    def find_a_candidate_on_grid(self, Xgrid):
        EI = self.expected_improvement(Xgrid)
        x_id = np.argmax(EI)

        opt = OptimizationStruct()  #insert results in struct
        opt.x_next           = Xgrid[x_id]
        opt.max_EI          = EI[x_id]
        opt.EI_of_Xgrid     = EI
        opt.Xgrid           = Xgrid

        return opt

    def find_a_candidate(self):
        n_restarts = 1
        x_next = None
        max_EI = 0
        _,nx = self._X.shape

        def min_obj(x):
            EI = self.expected_improvement(x)
            return -EI

        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[0][0], self.bounds[0][1],
                                    size=(n_restarts, nx)):
            res = minimize(min_obj, x0=x0,bounds=self.bounds, method='Nelder-Mead')        
            max_EI_temp = -res.fun #negating since it is minimizing
            
            if max_EI_temp > max_EI:
                max_EI = max_EI_temp
                x_next = res.x
        
        assert x_next == None

        opt = OptimizationStruct() #insert results in struct
        opt.x_next = x_next[0]
        opt.max_EI = max_EI

        return opt

    def optimization_step(self,opt:OptimizationStruct,
                                use_grid_optimization=False,
                                plot_step = False):
        if opt.x_next is not None:
            y_next = self.obj_fun(opt.x_next)
            self._X = np.vstack((self._X, opt.x_next))
            self._Y = np.vstack((self._Y, y_next))
            self.model.fit(self._X,self._Y)
        if use_grid_optimization:
            if opt.Xgrid is None:
                opt.Xgrid = np.linspace(*self.bounds[0], 1000)
            opt = self.find_a_candidate_on_grid(opt.Xgrid)
        else:
            opt = self.find_a_candidate()

        if plot_step:
            global PLOT_NR
            fig = plt.figure()
            ax = plt.subplot()
            self.plot_surrogate_and_expected_improvement(ax,opt)
            plt.savefig(f"master-thesis/figures/{self.model.name}_x{PLOT_NR}.png")
            PLOT_NR = PLOT_NR+1
            fig.close()
        return opt

    def optimize(self,num_steps:int, 
                        use_grid_optimization = False,
                        plot_steps = False):
        opt = OptimizationStruct()
        for i in range(num_steps):
            print(f"-- finding x{i+1} --",end="\n")
            opt = self.optimization_step(opt,use_grid_optimization = use_grid_optimization,
                                plot_step = plot_steps)
            print(f"-- x{i+1} = {opt.x_next:.2f} --")
        
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
    datasize = 5
    np.random.seed(2)
    X_sample =  np.random.uniform(*bounds[0],size = datasize)
    X_sample = X_sample[:,None]
    Y_sample = obj_fun(X_sample)

    BOHAMIANN_regression = BOHAMIANN(num_warmup = 5000, num_samples = 5000)
    BO_BOHAMIANN = BayesianOptimization(obj_fun, BOHAMIANN_regression,X_sample,Y_sample)
    BO_BOHAMIANN.optimize(10, plot_steps = True,use_grid_optimization=True)
    
    NNN = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 2000, num_samples=1000, keep_every = 50)
    BO_BNN = BayesianOptimization(obj_fun, NNN,X_sample,Y_sample)
    BO_BNN.optimize(10, plot_steps = True, use_grid_optimization=True)
    

