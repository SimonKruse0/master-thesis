from ast import Not
from sklearn.neighbors import KernelDensity
from src.optimization.bayesian_optimization import BayesianOptimization
from src.regression_models.gaussian_process_regression import GaussianProcess_sklearn
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.SPN_regression2 import SumProductNetworkRegression
from src.regression_models.naive_GMR import NaiveGMRegression
import numpy as np
from datetime import datetime
import os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import random
from matplotlib.patches import Patch

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.utils import PlottingClass2, PlottingClass, OptimizationStruct, uniform_grid
import sys

#class BayesOptSolverBase(PlottingClass):
class BayesOptSolverBase(PlottingClass2):
    def __init__(self, reg_model, problem,acquisition, budget,disp) -> None:
        self.acquisition = acquisition
        self.problem = problem #problem.best_observed_fvalue1
        self.budget = budget
        self.model = reg_model
        self._X, self._Y, self.y_min, self.x_best = None, None,None,None
        
        #assert id(self.y_min) == id(problem.best_observed_fvalue1) #should be pointers!
        self.opt = OptimizationStruct()
        self.disp = disp
        self.beta = None #for LCB
        self.init_train = False

    def problem_init(self):
        raise NotImplementedError

    def obj_fun(self,x):
        raise NotImplementedError

    def init_XY_and_fit(self, n_init_samples):
        if n_init_samples >0:
            self._X, self._Y = self._init_XY(n_init_samples)
            self.y_min = np.min(self._Y)
            self.init_train = True
            # print(f"\n-- initial training -- \t {self.model.name}")
            # self.fit()
        else:
            self._X, self._Y = None,None


    def __call__(self):
        if self.disp == False:
            sys.stdout = open(os.devnull, 'w')
        self.optimize()
        sys.stdout = sys.__stdout__
        x = self.x_best
        #print(self.get_optimization_hist())
        return x

    def _init_XY(self, sample_size, vectorized = False):
        #X_init = self._lineargrid1D(n=sample_size)
        X_init = next(self._randomgrid(1, sample_size))
        #print(X_init)
        if vectorized:
            Y_init = self.obj_fun(X_init)[:,None]
        else:
            Y_init = np.array([self.obj_fun(x) for x in X_init])[:,None]
        self.budget -= sample_size
        
        assert Y_init.ndim == 2
        return X_init, Y_init

    def batch(self,iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    def _batched_predictive_pdf(self,X_test):
        Y_mu_list = []
        Y_sigma_list = []
        for X_batch in batch(X_test, 1000):
            Y_mu,Y_sigma,_ = self.predict(X_batch)
            Y_mu_list.append(Y_mu)
            Y_sigma_list.append(Y_sigma)
        return np.array(Y_mu_list).flatten(),np.array(Y_sigma_list).flatten()

    def predictive_pdf(self, X,Y, return_px = False, grid1D = False):
        
        if grid1D: #making grid prediction.
            x_res = X.shape[0]
            y_res = Y.shape[0]
            XY_grid = [x.flatten() for x in np.meshgrid(X.flatten(), Y.flatten(), indexing="ij")]
            X,Y = XY_grid[0][:,None],  XY_grid[1][:,None]
        try:
            predictive_pdfs = []
            p_x = []
            for X_batch, Y_batch in zip(self.batch(X, 1000), self.batch(Y, 1000)):
                temp = self.model.predictive_pdf(X_batch,Y_batch) 
                predictive_pdfs.append(temp[0])
                p_x.append(temp[1])

            predictive_pdfs = np.array(predictive_pdfs)
            p_x = np.array(p_x)
        except:
            print(f"{self.model.name} didn't have a predictive_pdf - using Gaussian")

            Y = Y.squeeze()
            if grid1D:
                Y_mu,Y_sigma = self.predict(X[::y_res]) 
                Y_mu = Y_mu.squeeze()
                Y_sigma = Y_sigma.squeeze()
                Y = Y.reshape(x_res,y_res)
                Z_pred = (Y-Y_mu[:,None])/Y_sigma[:,None] #std. normal distributed. 
                Z_pred = Z_pred.flatten()
            else:
                Y_mu,Y_sigma = self.predict(X)
                Z_pred = (Y-Y_mu)/Y_sigma #std. normal distributed. 
            
            predictive_pdfs = norm.pdf(Z_pred)
            p_x = None
        if return_px:
            return predictive_pdfs, p_x
        return predictive_pdfs

    def predictive_logpdf(self, X,Y, return_px = False, grid1D = False):
        
        if grid1D: #making grid prediction.
            x_res = X.shape[0]
            y_res = Y.shape[0]
            XY_grid = [x.flatten() for x in np.meshgrid(X.flatten(), Y.flatten(), indexing="ij")]
            X,Y = XY_grid[0][:,None],  XY_grid[1][:,None]
        try:
            predictive_logpdfs = []
            p_x = []
            for X_batch, Y_batch in zip(self.batch(X, 1000), self.batch(Y, 1000)):
                temp = self.model.predictive_pdf(X_batch,Y_batch) 
                predictive_logpdfs.append(np.log(temp[0]))
                p_x.append(temp[1])

            predictive_logpdfs = np.array(predictive_logpdfs)
            p_x = np.array(p_x)
        except:
            print(f"{self.model.name} didn't have a predictive_pdf - using Gaussian")

            Y = Y.squeeze()
            if grid1D:
                Y_mu,Y_sigma = self.predict(X[::y_res]) 
                Y_mu = Y_mu.squeeze()
                Y_sigma = Y_sigma.squeeze()
                Y = Y.reshape(x_res,y_res)
                Z_pred = (Y-Y_mu[:,None])/Y_sigma[:,None] #std. normal distributed. 
                Z_pred = Z_pred.flatten()
            else:
                Y_mu,Y_sigma = self.predict(X)
                Z_pred = (Y-Y_mu)/Y_sigma #std. normal distributed. 
            
            predictive_logpdfs = norm.logpdf(Z_pred)
            p_x = None
        if return_px:
            return predictive_logpdfs, p_x
        return predictive_logpdfs

    def predict(self,X, gaussian_approx = True, get_px = False):
        # if X.shape[0] > 1000:
        if X.ndim == 1:
            X = X[:,None]
        assert X.ndim == 2
        if get_px:
            Y_mu,Y_sigma,p_x = self.model.predict(X)
            return Y_mu,Y_sigma, p_x
        Y_mu_list = []
        Y_sigma_list = []
        assert gaussian_approx == True
        for X_batch in self.batch(X, 1000):
            Y_mu,Y_sigma,_ = self.model.predict(X_batch)
            Y_mu_list.append(Y_mu)
            Y_sigma_list.append(Y_sigma)
        return np.array(Y_mu_list).flatten(),np.array(Y_sigma_list).flatten()

    def predictive_samples(self, X, n_samples  =9):
        return self.model.predictive_samples(X, n_samples = n_samples)
        # try:
        #     return self.model.predictive_samples(X)
        # except:
        #     pass

    def nlower_confidense_bound(self,X, beta = 2,  return_analysis = False):
        mu, sigma = self.predict(X)
        if self.beta is None:
            self.beta = beta
        imp = self.y_min - mu
        lcb = -imp - self.beta*sigma
        if return_analysis:
            return -lcb,imp, sigma  #note since symetric around mu, minimizing the LCB is the same as maximizing the UCB
        else:
            return -lcb

    def approx_expected_improvement(self,X):
        predictive_samples = self.predictive_samples(X, n_samples=1000)
        approx_EI = np.mean(np.maximum(0,self.y_min-predictive_samples), axis=1)
        return approx_EI

    def expected_improvement(self,X, return_analysis = False):
        assert self.y_min is not None
        assert X.ndim == 2
        if self.model.name == "Naive Gaussian Mixture Regression":
            mu, sigma, p_x = self.predict(X, get_px=True) 
        else:
            mu, sigma = self.predict(X)
        imp = self.y_min - mu
        Z = imp/sigma
        exploitation = imp*norm.cdf(Z)
        exploration = sigma*norm.pdf(Z)
        EI = exploitation + exploration
        # if self.model.name == "Naive Gaussian Mixture Regression":
        #     N = self._X.shape[0]
        #     factor =  np.clip(1/(N*p_x),1e-8,100)
        #     EI *= factor
        #     EI[factor > 99] = EI.max() #For at undgå inanpropiate bump.!
        if return_analysis:
            #return EI, exploitation, exploration
            return EI, imp, sigma
        else:
            return EI, None, None

    def _budget_is_fine(self):
        return self.budget >= self.problem.evaluations

    def _randomgrid(self,n_batches,n=5000):
        for _ in range(n_batches):
            yield np.random.uniform(*self.bounds , size=(n,self.problem_dim))

    def _lineargrid1D(self,n=1000):
        return np.linspace(*self.bounds, n)

    def acquisition_function(self,x):
        if self.acquisition=='LCB':
            return self.nlower_confidense_bound(x)
        if self.acquisition=='EI':
            EI, exploitation, exploration = self.expected_improvement(x, return_analysis=False)
            return EI
        if self.acquisition=='aEI':
            EI = self.approx_expected_improvement(x)
            return EI

    def find_a_candidate_on_randomgrid(self, n_batches  = 1):
        if self.model.name == "empirical mean and std regression": #random search
            opt = OptimizationStruct()
            opt.x_next = next(self._randomgrid(1,n=1))[0]
            self.opt = opt
            return
        
        max_AQ = -np.Inf
        for Xgrid_batch in self._randomgrid(n_batches):
            AQ= self.acquisition_function(Xgrid_batch)

            max_id = np.argwhere(AQ == np.amax(AQ)).flatten()
            if len(max_id) > 1: #multiple maxima -> pick one at random
                x_id = random.choice(max_id)
            else:
                x_id = max_id[0]

            max_AQ_batch = AQ[x_id]
            print(max_AQ_batch)
            if max_AQ_batch > max_AQ:
                max_AQ = max_AQ_batch
                x_next = Xgrid_batch[x_id]

        opt = OptimizationStruct()  #insert results in struct
        opt.x_next          = x_next
        opt.max_AQ          = max_AQ
        print(x_next,"x_next")
        self.opt = opt #redefines self.opt!

    def update_y_min(self):
        self.y_min = np.min(self._Y)

    def observe(self,x_next):
        assert x_next is not None
        #assert self._budget_is_fine()
        y_next = self.obj_fun(x_next)
        self._X = np.vstack((self._X, x_next))
        self._Y = np.vstack((self._Y, np.array([[y_next]])))

    def fit(self, X = None, Y = None):
        #try:
        if X is None:
            self.model.fit(self._X,self._Y)
        else:
            self.model.fit(X,Y)
        # except:
        #     raise Exception("No data to fit")

    def optimization_step(self, update_y_min = True):
        if self.init_train:
            print(f"-- training on +{self._X.shape[0]} data points, nettobudget {self.budget} -- \t {self.model.name}")
            self.init_train = False
        self.fit()
        n_batches = min(self.problem_dim,20)
        self.find_a_candidate_on_randomgrid(n_batches = n_batches)
        x_next = self.opt.x_next
        self.observe(x_next)
        if update_y_min:
            self.update_y_min()

    def optimize(self):
        for i in range(self.budget):
            print(f"-- finding x{i+1} --",end="\n")
            self.optimization_step()
            x_next = self.opt.x_next
            x_next_text = " , ".join([f"{x:.2f}" for x in x_next])
            print(f"-- x{i+1} = {x_next_text} --")

        #Define best x!
        # Best_Y_id = np.argmin(self._Y)
        # self.x_best = self._X[Best_Y_id]
        #assert self.y_min == self._Y[Best_Y_id]
        print(f"-- End of optimization -- best objective y = {self.y_min:0.2f}\n")

    def get_optimization_hist(self):
        return np.hstack([ self._X, self._Y])

class BayesOptSolver_coco(BayesOptSolverBase):
    def __init__(self, reg_model, problem, acquisition="EI",budget = 35, n_init_samples=5, disp=False) -> None:
        super().__init__(reg_model,problem,acquisition, budget, disp)
        self.problem_init()
        self.init_XY_and_fit(n_init_samples)

    def problem_init(self):
        problem = self.problem
        self.problem_name = problem.name.split(" ")[3]
        self.problem_dim = problem.dimension
        self.bounds = [problem.lower_bounds, problem.upper_bounds]

    def obj_fun(self,x):
        return self.problem(x)

class BayesOptSolver_sklearn(BayesOptSolverBase):
    def __init__(self, reg_model, problem,acquisition="EI", budget = 5, n_init_samples=2, disp=False) -> None:
        super().__init__(reg_model,problem,acquisition, budget, disp)
        self.problem_init()
        self.init_XY_and_fit(n_init_samples)

    def problem_init(self):
        problem = self.problem
        self.problem_name = type(problem).__name__
        self.problem_dim = problem.N
        lower_bounds = [ulb[0] for ulb in problem.bounds]
        upper_bounds = [ulb[1] for ulb in problem.bounds]
        self.bounds = [lower_bounds, upper_bounds]

    def obj_fun(self,x):
        if x.ndim == 1:
            return self.problem.fun(x)
        else:
            return np.array([self.problem.fun(xi) for xi in x])


class PlotBayesOpt1D(BayesOptSolver_sklearn):
    def __init__(self, reg_model, problem, acquisition="EI", budget=5,deterministic = False, n_init_samples=2, disp=False, show_name = True) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        assert self.problem_dim == 1
        self.bounds = (self.bounds[0][0], self.bounds[1][0])
        self.Xgrid = np.linspace(*self.bounds, 1000)[:, None]
        self.show_name = show_name
        self.deterministic = deterministic
    def optimize(self, path="", extension= "jpg"):
        for i in range(self.budget):
            print(f"-- finding x{i+1} --",end="\n")

            #self.fit()
            self.optimization_step(update_y_min=False)
            #self.opt.x_next = np.array([0.])
            x_next = self.opt.x_next
            x_next_text = " , ".join([f"{x:.2f}" for x in x_next])
            print(f"-- x{i+1} = {x_next_text} --")
            outer_gs = gridspec.GridSpec(1, 1)
            self.plot_surrogate_and_acquisition_function(outer_gs[0])

            self.update_y_min()
            number = f"{i}".zfill(3)
            if extension is None:
                continue
            if path == "":
                plt.show()
            else:
                fig_path = path+f"{self.problem_name}{self.model.name}{number}.{extension}"
                fig_path = fig_path.replace(" ", "_")
                plt.savefig(fig_path)
        print(f"-- End of optimization -- best objective y = {self.y_min:0.2f}\n")
        opt_hist = self.get_optimization_hist()
        txt_path =  path+f"{self.problem_name}{self.model.name}.txt"
        np.savetxt(txt_path,opt_hist)

    def plot_surrogate_and_acquisition_function(self,subplot_spec):
        opt = self.opt
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        self.plot_true_function(ax1)
        if self.acquisition == "aEI":
            self.plot_predictive_dist(ax1)
            cmap = plt.cm.Blues
            legend_elements = [Patch(facecolor=cmap(0.6), edgecolor=cmap(0.6),
                        label="predictive distribution")]
                        #label=r'$\hat p(y|x)$')]
            ax1.legend(handles=legend_elements)
        else:
            self.plot_regression_gaussian_approx(ax1, show_name =self.show_name)
            ax1.legend(loc=2)
        
        self.plot_train_data(ax1, self._X[:-1],self._Y[:-1], size =10)
        ax1.set_title(f"{self.model.name}({self.model.params})")
        
        self.plot_acquisition_function(ax2)

        ax2.set_xlabel("x")
        x_next = opt.x_next[:,None]
        max_AQ= self.acquisition_function(x_next)
        ax2.plot(x_next, max_AQ, "^", markersize=10,color="tab:orange", label=f"x_next = {opt.x_next[0]:.2f}")
        ax2.legend(loc=1)

    def plot_regression_gaussian_approx(self,ax,show_name = False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions

        Ymu, Ysigma = self.predict(self.Xgrid)
        Xgrid = self.Xgrid.squeeze()
        ax.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        ax.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma,
                        color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        ax.set_xlim(*self.bounds)
        #ax.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if show_name:
            ax.set_title(f"{self.model.name}")#({self.model.params})")
        

    def plot_acquisition_function(self,ax, color = "tab:blue", show_y_label = True, return_path = False):
        Xgrid = self.Xgrid

        if self.acquisition == "EI":
            #EI_of_Xgrid, exploitation, exploration  = self.expected_improvement(Xgrid, return_analysis = True)
            EI_of_Xgrid, imp, sigma  = self.expected_improvement(Xgrid, return_analysis = True)
            # plot the acquisition function ##
            ax.plot(Xgrid, EI_of_Xgrid, color=color, label = "EI") 
            # ax.plot(Xgrid, exploitation, "--", color = "cyan", label="exploitation") 
            # ax.plot(Xgrid, exploration, "--", color = "red", label="exploration") 
        if self.acquisition == "LCB":
            LCB_of_Xgrid, imp, sigma = self.nlower_confidense_bound(Xgrid, return_analysis = True)
            # plot the acquisition function ##
            lvl = norm.cdf(self.beta)
            ax.plot(Xgrid, LCB_of_Xgrid, color=color, label = f"LCB({lvl:0.3f})") 

        if self.acquisition == "aEI":
            Xgrid = np.linspace(*self.bounds, 100)[:, None]
            aEI_of_Xgrid = self.approx_expected_improvement(Xgrid)
            ax.plot(Xgrid, aEI_of_Xgrid, color=color, label = "aEI") 

        ax.set_xlim(*self.bounds)
        if show_y_label:
            ax.set_ylabel("Acquisition Function")
        ax.legend(loc=1)
        if return_path:
            return imp, sigma


class PlotBayesOpt2D(BayesOptSolver_sklearn):
    #raise NotImplementedError

    def __init__(self, reg_model, problem, acquisition="EI", budget=5, n_init_samples=2, disp=False) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        assert self.problem_dim == 2
        self.bounds = (self.bounds[0][0], self.bounds[1][0])
        self.Xgrid = np.linspace(*self.bounds, 1000)[:, None]

    def optimize(self):
        for i in range(self.budget):
            print(f"-- finding x{i+1} --",end="\n")
            self.optimization_step()
            x_next = self.opt.x_next
            x_next_text = " , ".join([f"{x:.2f}" for x in x_next])
            print(f"-- x{i+1} = {x_next_text} --")
            outer_gs = gridspec.GridSpec(1, 1)
            self.plot_surrogate_and_acquisition_function(outer_gs[0])
            plt.show()
        print(f"-- End of optimization -- best objective y = {self.y_min:0.2f}\n")

    def plot_surrogate_and_acquisition_function(self,subplot_spec, show_name = False):
        opt = self.opt
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        self.plot_regression_gaussian_approx(ax1, show_name = show_name)
        X_true =  np.linspace(*self.bounds,1000)[:,None]
        Y_true = self.obj_fun(X_true)
        ax1.plot(X_true, Y_true, "--", color="Black")
        ax1.plot(self._X[:-1],self._Y[:-1], ".", markersize = 10, color="black")  # plot all observed data
        #ax1.plot(self._X[-1],self._Y[-1], ".", markersize = 10, color="tab:orange")  # plot

        self.plot_acquisition_function(ax2)
        x_next = opt.x_next[:,None]
        max_AQ= self.acquisition_function(x_next)
        ax2.plot(x_next, max_AQ, "^", markersize=10,color="tab:orange", label=f"x_next = {opt.x_next[0]:.2f}")


    def plot_regression_gaussian_approx(self,ax,show_name = False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions

        Ymu, Ysigma = self.predict(self.Xgrid)
        Xgrid = self.Xgrid.squeeze()
        ax.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        ax.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma,
                        color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        ax.set_xlim(*self.bounds)
        #ax.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if show_name:
            ax.set_title(f"{self.model.name}({self.model.params})")
        ax.legend(loc=2)

    def plot_acquisition_function(self,ax):
        Xgrid = self.Xgrid

        if self.acquisition == "EI":
            EI_of_Xgrid, exploitation, exploration  = self.expected_improvement(Xgrid, return_analysis = True)
            # plot the acquisition function ##
            ax.plot(Xgrid, EI_of_Xgrid, color="tab:blue", label = "EI") 
            ax.plot(Xgrid, exploitation, "--", color = "cyan", label="exploitation") 
            ax.plot(Xgrid, exploration, "--", color = "red", label="exploration") 
        if self.acquisition == "LCB":
            LCB_of_Xgrid = self.nlower_confidense_bound(Xgrid)
            # plot the acquisition function ##
            ax.plot(Xgrid, LCB_of_Xgrid, color="tab:blue", label = "LCB") 

        ax.set_xlim(*self.bounds)
        ax.set_ylabel("Acquisition Function")
        ax.legend(loc=1)


if __name__ == "__main__":
    from src.benchmarks.custom_test_functions.problems import SimonsTest2, Test3b
    from src.benchmarks.go_benchmark_functions.go_funcs_K import Katsuura
    import cocoex
    suite = iter(cocoex.Suite("bbob", "", "dimensions:2 instance_indices:1"))
    problem_sklearn = SimonsTest2()
    problem_sklearn = Test3b()
    problem_coco = next(suite)
    regression_model = GaussianProcess_GPy()
    regression_model = NaiveGMRegression()
    regression_model = SumProductNetworkRegression(optimize=False, opt_n_iter=40)
    plot_BO = PlotBayesOpt1D(regression_model, problem_sklearn, acquisition="EI",budget=54, n_init_samples=50,disp=True)
    plot_BO()
    #plot_BO = PlotBayesOpt1D(regression_model, problem_sklearn, acquisition="EI",budget=14,n_init_samples=10,disp=True)

    # BO = BayesOptSolver_coco(regression_model, problem_coco, acquisition="LCB",disp=True)
    # BO2 = BayesOptSolver_sklearn(regression_model, problem_sklearn,acquisition="EI", disp=True)
    # BO()
    # BO2()