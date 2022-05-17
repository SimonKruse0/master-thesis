from datetime import datetime
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import PlottingClass, OptimizationStruct, uniform_grid

if __name__=="__main__":
    from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork #JAX
    from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn, GaussianProcess_pyro
    from src.regression_models.bohamiann import BOHAMIANN #Torch
    from src.regression_models.naive_GMR import NaiveGMRegression
    from src.regression_models.SPN_regression2 import SumProductNetworkRegression
    from src.regression_models.mean_regression import MeanRegression
    from src.benchmarks.custom_test_functions.problems import SimonsTest3_cosine_fuction

PLOT_NR = 0

class BayesianOptimization(PlottingClass):
    def __init__(self, problem, regression_model, X_init=None,Y_init=None, init_n_samples = 5) -> None:
        super().__init__() #initalize plotting functions
        self.problem_name = type(problem).__name__
        self.problem_dim = problem.N
        self.bounds = problem.bounds[0] #OBS problem if higher dim have different bounds?
        self.obj_fun = lambda x: np.array([problem.fun(xi) for xi in x])[:,None] 
        self.model = regression_model
        if X_init is None:
            X_init,Y_init = self._initXY(init_n_samples)
        assert X_init.shape[0] == Y_init.shape[0]
        self._X = X_init
        self._Y = Y_init
        self.f_best = np.min(Y_init) # incumbent np.min(Y) or np.min(E[Y]) ?? BOHAMIANN does this
        #self.bounds = bounds
        self.n_initial_points , self.nX = X_init.shape
        self.opt: OptimizationStruct = None
        self.fig_folder = None

        print(f"\n-- initial training -- \t {self.model.name}")
        self.model.fit(X_init,Y_init)

    def _initXY(self, sample_size):
        np.random.seed(2)
        X_init = np.random.uniform(*self.bounds,size = (sample_size,self.problem_dim))
        try:
            Y_init = self.obj_fun(X_init)
        except:
            assert False
            # y = []
            # for x in X_init:
            #     y.append(self.obj_fun(x))
            # Y_init = np.array(y)[:,None]
        return X_init,Y_init
    def predict(self,X, gaussian_approx = True, get_px = False):
        if get_px:
            Y_mu,Y_sigma,p_x = self.model.predict(X)
            return Y_mu,Y_sigma, p_x
        if gaussian_approx:
            Y_mu,Y_sigma,_ = self.model.predict(X)
            return Y_mu,Y_sigma
        else:
            Y_mu,_,Y_CI = self.model.predict(X)
            return Y_mu,Y_CI

    def expected_improvement(self,X,xi=0, return_analysis = False):
        assert X.ndim == 2
        #print("OBS X[:,None] might fail in largers dims!")
        if self.model.name == "Naive Gaussian Mixture Regression":
            mu, sigma, p_x = self.predict(X, get_px=True) #Partial afledt. Pytorch. 
        else:
            mu, sigma = self.predict(X) #Partial afledt. Pytorch. 

        imp = -mu - self.f_best - xi
        Z = imp/sigma
        exploitation = imp*norm.cdf(Z)
        exploration = sigma*norm.pdf(Z)
        EI = exploitation + exploration
        if self.model.name == "Naive Gaussian Mixture Regression":
            N = self._X.shape[0]
            factor =  np.clip(1/(N*p_x),1e-8,100)
            EI *= factor
            EI[factor > 99] = EI.max() #For at undgå inanpropiate bump.!

        #EI = exploitation/10 + exploration
        if return_analysis:
            return EI, exploitation, exploration
        else:
            return EI


    def expected_improvement2(self,X,xi=0, return_analysis = False):
        assert X.ndim == 2
        #print("OBS X[:,None] might fail in largers dims!")
        mu, sigma = self.predict(X) #Partial afledt. Pytorch. 
        imp = -mu - self.f_best - xi
        Z = imp/sigma
        exploitation = imp*norm.cdf(Z)
        exploration = sigma*norm.pdf(Z)
        EI = exploitation + exploration
        #EI = exploitation/10 + exploration
        if return_analysis:
            return EI, exploitation, exploration
        else:
            return EI


    def find_a_candidate_on_grid(self, Xgrid): #TODO: lav adaptiv grid search. Og batches! 
        EI, _exploitation, _exploration = self.expected_improvement(Xgrid, return_analysis=True)
        #x_id = np.argmax(EI)
        max_id = np.argwhere(EI == np.amax(EI)).flatten()
        
        if len(max_id) > 1: #Very relevant for Naive GMR
            x_id = random.choice(max_id)
        else:
            x_id = max_id[0]

        opt = OptimizationStruct()  #insert results in struct
        opt.x_next          = Xgrid[x_id]
        opt.max_EI          = EI[x_id]
        opt.EI_of_Xgrid     = EI
        opt._exploitation   = _exploitation
        opt._exploration    = _exploration
        opt.Xgrid           = Xgrid

        return opt

    def find_a_candidate(self,n_restarts=1, x0 = None): #TODO: implementer global optimiering
        x_next = None
        max_EI = 0
        _,nx = self._X.shape

        def min_obj(x):
            EI = self.expected_improvement(x[None,:])
            return -EI

        # Find the best optimum by starting from n_restart different random points.
        lb,ub = self.bounds

        if x0 is None:
            for x0 in np.random.uniform(*self.bounds,
                                        size=(n_restarts, nx)):
                res = minimize(min_obj, x0=x0,bounds = ((lb,ub),), method='Nelder-Mead')        
                max_EI_temp = -res.fun #negating since it is minimizing
                if max_EI_temp > max_EI:
                    max_EI = max_EI_temp
                    x_next = res.x
        else:
            res = minimize(min_obj, x0=x0,bounds = ((lb,ub),), method='Nelder-Mead')        
            max_EI_temp = -res.fun
            if max_EI_temp > max_EI:
                    max_EI = max_EI_temp
                    x_next = res.x
        #assert x_next is not None

        opt = OptimizationStruct() #insert results in struct
        opt.x_next = x_next
        opt.max_EI = max_EI

        return opt

    def optimization_step(self,opt:OptimizationStruct = OptimizationStruct(),
                                n_restarts = 1,
                                type="grid",
                                plot_step = False):
        if opt.x_next is not None:
            y_next = self.obj_fun(opt.x_next)
            self._X = np.vstack((self._X, opt.x_next))
            self._Y = np.vstack((self._Y, y_next))
            self.f_best = np.min(self._Y) #WOW!!
            self.model.fit(self._X,self._Y)
        if type == "grid":
            if opt.Xgrid is None:
                opt.Xgrid = uniform_grid(self.bounds, n_var=self.nX)
            opt = self.find_a_candidate_on_grid(opt.Xgrid)
        elif type == "numeric":
            opt = self.find_a_candidate(n_restarts = n_restarts)
        elif type == "both":
            if opt.Xgrid is None:
                opt.Xgrid = uniform_grid(self.bounds, n_var=self.nX)
            opt_grid = self.find_a_candidate_on_grid(opt.Xgrid)
            opt_num = self.find_a_candidate()
            print(f"\nEI_gridsearch = {opt_grid.max_EI:0.3f}\n EI_numsearch = {opt_num.max_EI:0.3f}")
            if opt_grid.max_EI > opt_num.max_EI:
                opt_num2 = self.find_a_candidate(x0 = opt_grid.x_next)
                print(f"EI_numsearch2 = {opt_num2.max_EI:0.3f} (with grid search starting point)")
                if opt_grid.max_EI > opt_num2.max_EI:
                    #print("--2 grid search of surrogate")
                    opt = opt_grid
                else:
                    #print("--2 num search in surrogate")
                    opt = opt_num2
            else:
                #print("num search in surrogate")
                opt = opt_num
        else:
            print("ERROR: Choose a optimization type")
            assert False

        if plot_step:
            if self.nX != 1:
                print(f"ERROR: plotting is not available for dim = {self.nX}")
            else:
                if self.fig_folder is None:
                    self.fig_folder = input("Folder name: ")
                    try:
                        path = os.path.join(os.getcwd(),f"master-thesis/figures/{fig_folder}")
                        os.mkdir(path)
                    except:
                        print(f"{self.fig_folder} already exists")

                global PLOT_NR
                fig = plt.figure()
                ax = plt.subplot()
                self.plot_surrogate_and_expected_improvement(ax,opt, show_name=True)
                Timestamp = datetime.today().strftime('%m%d_%H%M')
                plt.savefig(f"master-thesis/figures/{self.fig_folder}/{self.model.name}_x{PLOT_NR}_{Timestamp}.png")
                PLOT_NR = PLOT_NR+1
                plt.close(fig)
        
        self.opt = opt
        return opt

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


    # def save_output(self, it):

    #     data = dict()
    #     data["optimization_overhead"] = self.time_overhead[it]
    #     data["runtime"] = self.runtime[it]
    #     data["incumbent"] = self.incumbents[it]
    #     data["incumbents_value"] = self.incumbents_values[it]
    #     data["time_func_eval"] = self.time_func_evals[it]
    #     data["iteration"] = it

    #     json.dump(data, open(os.path.join(self.outpulikelihood

#import pickle

if __name__ == "__main__":

    bounds = [0,1]
    #datasize = int(input("Enter number of datapoints: "))
    # datasize = 10
    # np.random.seed(2)
    # X_sample =  np.random.uniform(*bounds,size = (datasize,1))
    # Y_sample = obj_fun(X_sample)
    problem = SimonsTest3_cosine_fuction()
    mean_regression = MeanRegression()
    SPN_regression = SumProductNetworkRegression(manipulate_variance=False, optimize=True)
    # GP_regression = GaussianProcess_sklearn()
    # GP_regression2 = GaussianProcess_pyro(noise=0)
    BOHAMIANN_regression = BOHAMIANN(num_warmup = 200, num_samples = 400)  
    NNN_regression = NumpyroNeuralNetwork(num_chains = 4, num_warmup= 200, num_samples=200, num_keep_samples= 50)
    mixture_regression = NaiveGMRegression()
    
    #regression_model = [mixture_regression, GP_regression,BOHAMIANN_regression,NNN_regression]
    regression_models = [SPN_regression, mixture_regression,BOHAMIANN_regression, mean_regression]
    # BO_BNN = BayesianOptimization(obj_fun, regression_model[1],bounds,X_sample,Y_sample)
    # BO_BNN.optimize(10, plot_steps = True, type="grid")
    # print(BO_BNN.get_optimization_hist())

    ### plot all regressions next to each other. ###
    plt.figure(figsize=(12, 8))
    outer_gs = gridspec.GridSpec(2, len(regression_models)//2+1)
    for i in range( len(regression_models)):
        BO_BNN = BayesianOptimization(problem, regression_models[i])
        opt = BO_BNN.optimization_step()
        BO_BNN.plot_surrogate_and_expected_improvement(outer_gs[i],opt, show_name=True)
    plt.show()
