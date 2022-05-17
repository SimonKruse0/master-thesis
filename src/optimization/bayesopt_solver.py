from src.optimization.bayesian_optimization import BayesianOptimization
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
import numpy as np
from datetime import datetime
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import PlottingClass, OptimizationStruct, uniform_grid


class BayesOptSolver_GP():
    def __init__(self, problem, x0, budget) -> None:
        init_n_samples = 2
        self.obj_fun = lambda x: problem(x)
        self.model = GaussianProcess_sklearn()
        self.problem_dim = problem.dimension
        self.bounds = [problem.lower_bounds, problem.upper_bounds]
        self._X, self._Y = self._init_XY(init_n_samples)
        self.fit()
        self.y_best = problem.best_observed_fvalue1
        #assert id(self.y_best) == id(problem.best_observed_fvalue1) #should be pointers!
        self.opt = OptimizationStruct()
        self.budget = budget
        self.problem = problem
        self.initialize = True


    def __call__(self):
        if self.initialize:
            x = self._X[0,:]
        else:
            self.optimization_step()
            x = self.opt.x_next
            self.initialize = False
        return x

    def _init_XY(self, sample_size):
        #np.random.seed(2)
        #X_init = np.random.uniform(*self.bounds , size=(sample_size,self.problem_dim))
        X_init = next(self._randomgrid(1, sample_size))
        Y_init = np.array([self.obj_fun(x) for x in X_init])[:,None]
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
        #print("self.y_best", self.y_best)
        assert self.problem.best_observed_fvalue1 == self.y_best
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

    # def _batch(iterable, n=1000):
    #     l = len(iterable)
    #     for ndx in range(0, l, n):
    #         yield iterable[ndx:min(ndx + n, l)]

    def _budget_is_fine(self):
        return self.budget > self.problem.evaluations

    def _randomgrid(self,n_batches,n=1000):
        for _ in range(n_batches):
            yield np.random.uniform(*self.bounds , size=(n,self.problem_dim))

    def find_a_candidate_on_randomgrid(self, n_batches  = 10):
        # for Xgrid_batch in self._batch(Xgrid):
        max_EI = -1
        for Xgrid_batch in self._randomgrid(n_batches):
            EI, _exploitation, _exploration = self.expected_improvement(Xgrid_batch, return_analysis=True)

            max_id = np.argwhere(EI == np.amax(EI)).flatten()
            if len(max_id) > 1: #Very relevant for Naive GMR
                x_id = random.choice(max_id)
            else:
                x_id = max_id[0]

            max_batch = EI[x_id]
            if max_batch > max_EI:
                max_EI = EI[x_id]
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
        assert self._budget_is_fine()
        y_next = self.obj_fun(x_next)
        self._X = np.vstack((self._X, x_next))
        self._Y = np.vstack((self._Y, y_next[:,None]))

    def fit(self):
        self.model.fit(self._X,self._Y)

    def optimization_step(self, plot_step = False):
        x_next = self.opt.x_next
        if x_next is not None:
            self.observe(x_next)
            self.fit()
        self.find_a_candidate_on_randomgrid(n_batches = 10)

    def optimize(self,num_steps:int, 
                        type,
                        n_restarts = 1,
                        plot_steps = False):
        opt = OptimizationStruct()
        for i in range(num_steps):
            print(f"-- finding x{i+1} --",end="\n")
            opt = self.optimization_step(opt,type = type,
                                n_restarts = n_restarts,
                                plot_step = plot_steps)
            if opt.x_next is not None:
                x_next_text = " , ".join([f"{x:.2f}" for x in opt.x_next])
            else:
                x_next_text = "None"
            print(f"-- x{i+1} = {x_next_text} --")
        
        Best_Y_id = np.argmin(self._Y)
        Best_X = self._X[Best_Y_id]
        Best_Y = self._Y[Best_Y_id]
        print(f"-- End of optimization -- best x = {Best_X} with objective y = {Best_Y}")

    def get_optimization_hist(self):
        y_next = np.nan
        self._X = np.vstack((self._X, self.opt.x_next))
        self._Y = np.vstack((self._Y, y_next))
        return self._X[self.n_initial_points:], self._Y[self.n_initial_points:]



if __name__ == "__main__":
    #BO = BayesOptSolver_GP()
    pass
