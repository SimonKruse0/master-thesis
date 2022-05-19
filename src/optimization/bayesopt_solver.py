from ast import Not
from src.optimization.bayesian_optimization import BayesianOptimization
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
import numpy as np
from datetime import datetime
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import random
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import PlottingClass, OptimizationStruct, uniform_grid
import sys

class BayesOptSolver():
    def __init__(self, reg_model, problem, budget,n_init_samples = 2, disp = False) -> None:
        self.problem = problem #problem.best_observed_fvalue1
        self.problem_name = problem.name.split(" ")[3]
        self.budget = budget
        self.obj_fun = lambda x: problem(x)
        self.model = reg_model
        self.problem_dim = problem.dimension
        self.bounds = [problem.lower_bounds, problem.upper_bounds]
        
        if n_init_samples >0:
            self._X, self._Y = self._init_XY(n_init_samples)
            print(f"\n-- initial training -- \t {self.model.name}")
            self.fit()
            self.y_best = np.min(self._Y)
        else:
            self._X, self._Y, self.y_best = None,None,None
        self.x_best = None
        #assert id(self.y_best) == id(problem.best_observed_fvalue1) #should be pointers!
        self.opt = OptimizationStruct()
        self.disp = disp

    def __call__(self):
        if self.disp == False:
            sys.stdout = open(os.devnull, 'w')
        self.optimize()
        sys.stdout = sys.__stdout__
        x = self.x_best
        #print(self.get_optimization_hist())
        return x

    def _init_XY(self, sample_size):
        X_init = next(self._randomgrid(1, sample_size))
        Y_init = np.array([self.obj_fun(x) for x in X_init])[:,None]
        self.budget -= sample_size
        return X_init, Y_init

    def predict(self,X, gaussian_approx = True, get_px = False):
        if get_px:
            Y_mu,Y_sigma,p_x = self.model.predict(X)
            return Y_mu,Y_sigma, p_x
        assert gaussian_approx == True
        Y_mu,Y_sigma,_ = self.model.predict(X)
        return Y_mu,Y_sigma

    def expected_improvement(self,X,xi=0, return_analysis = False):
        assert X.ndim == 2
        if self.model.name == "Naive Gaussian Mixture Regression":
            mu, sigma, p_x = self.predict(X, get_px=True) 
        else:
            mu, sigma = self.predict(X)
        imp = -mu - self.y_best - xi # xi ??
        Z = imp/sigma
        exploitation = imp*norm.cdf(Z)
        exploration = sigma*norm.pdf(Z)
        EI = exploitation + exploration
        if self.model.name == "Naive Gaussian Mixture Regression":
            N = self._X.shape[0]
            factor =  np.clip(1/(N*p_x),1e-8,100)
            EI *= factor
            EI[factor > 99] = EI.max() #For at undgå inanpropiate bump.!
        if return_analysis:
            return EI, exploitation, exploration
        else:
            return EI, None, None

    def _budget_is_fine(self):
        return self.budget >= self.problem.evaluations

    def _randomgrid(self,n_batches,n=10000):
        for _ in range(n_batches):
            yield np.random.uniform(*self.bounds , size=(n,self.problem_dim))

    def find_a_candidate_on_randomgrid(self, n_batches  = 1):
        if self.model.name == "empirical mean and std regression": #random search
            opt = OptimizationStruct()
            opt.x_next = next(self._randomgrid(1,n=1)).squeeze()
            self.opt = opt
            return
        
        max_EI = -1
        for Xgrid_batch in self._randomgrid(n_batches):
            EI, _exploitation, _exploration = self.expected_improvement(Xgrid_batch, return_analysis=False)

            max_id = np.argwhere(EI == np.amax(EI)).flatten()
            if len(max_id) > 1: #multiple maxima -> pick one at random
                x_id = random.choice(max_id)
            else:
                x_id = max_id[0]

            max_EI_batch = EI[x_id]
            if max_EI_batch > max_EI:
                max_EI = max_EI_batch
                x_next = Xgrid_batch[x_id]

        opt = OptimizationStruct()  #insert results in struct
        opt.x_next          = x_next
        opt.max_EI          = max_EI
        opt.EI_of_Xgrid     = EI #OBS maxlength = batch len
        opt._exploitation   = _exploitation #OBS maxlength = batch len
        opt._exploration    = _exploration #OBS maxlength = batch len
        opt.Xgrid           = Xgrid_batch #OBS maxlength = batch len

        self.opt = opt #redefines self.opt!

    def observe(self,x_next):
        assert x_next is not None
        #assert self._budget_is_fine()
        y_next = self.obj_fun(x_next)
        self._X = np.vstack((self._X, x_next))
        self._Y = np.vstack((self._Y, np.array([[y_next]])))
        self.y_best = np.min(self._Y)

    def fit(self, X = None, Y = None):
        if X is None:
            self.model.fit(self._X,self._Y)
        else:
            self.model.fit(X,Y)

    def optimization_step(self):
        x_next = self.opt.x_next
        if x_next is not None:
            self.observe(x_next)
            self.fit()
        n_batches = min(self.problem_dim,20)
        self.find_a_candidate_on_randomgrid(n_batches = n_batches)

    def optimize(self):
        for i in range(self.budget-1):
            print(f"-- finding x{i+1} --",end="\n")
            self.optimization_step()
            x_next = self.opt.x_next
            x_next_text = " , ".join([f"{x:.2f}" for x in x_next])
            print(f"-- x{i+1} = {x_next_text} --")
        x_next = self.opt.x_next
        if x_next is not None:
            self.observe(x_next) #last evaluation

        #Define best x!
        Best_Y_id = np.argmin(self._Y)
        self.x_best = self._X[Best_Y_id]
        #assert self.y_best == self._Y[Best_Y_id]
        print(f"-- End of optimization -- best objective y = {self.y_best:0.2f}")

    def get_optimization_hist(self):
        return self._X, self._Y



if __name__ == "__main__":
    #BO = BayesOptSolver_GP()
    pass
