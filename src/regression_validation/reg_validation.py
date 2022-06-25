from turtle import shape
from src.optimization.bayesopt_solver import BayesOptSolverBase, BayesOptSolver_coco, BayesOptSolver_sklearn
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}
import matplotlib
matplotlib.rc('font', **font)
from src.utils import normalize, PlottingClass2

def jsonize_array(array):
    if array is None or array[0] is None:
        return None
    return [a.astype(float) for a in array]

class PlotReg1D_mixturemodel(BayesOptSolver_sklearn):
    def __init__(self, reg_model, problem,random_seed = 42, acquisition="EI", budget=5, n_init_samples=2, disp=False) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        self.seednr = random_seed
        assert self.problem_dim == 1
        self.bounds = (self.bounds[0][0], self.bounds[1][0]) #redefine bounds. 
        self.show_name = True
        self.deterministic = False

    def __call__(self, n, grid_points=2000,show_gauss= False, show_pred= True, show_name = False, path=""):
        fig,ax = plt.subplots()
        
        np.random.seed(self.seednr)
        self.init_XY_and_fit(n)
        self.fit()
        self.bounds = (self.bounds[0]-10, self.bounds[1]+10)
        self.Xgrid = np.linspace(*self.bounds, grid_points)[:, None]
        self.ygrid = np.linspace(0,300, 1000)
        plt.ylim(0,300)
        try:
            self.plot_true_function2(ax)
        except:
            self.plot_true_function(ax)

        if show_pred:
            self.plot_predictive_dist(ax)
            cmap = plt.cm.Blues
            legend_elements = [Patch(facecolor=cmap(0.6), edgecolor=cmap(0.6),
                         label="predictive distribution")]
                         #label=r'$\hat p(y|x)$')]
            ax.legend(handles=legend_elements)
        if show_gauss and show_pred:
            self.plot_gaussian_approximation(ax, only_mean = True)
            #self.plot_gaussian_approximation(ax, only_mean = False)
        if show_gauss and not show_pred:
            self.plot_gaussian_approximation(ax)
        if show_name:
            name = self.model.name
            if "BNN" in name:
                name = "BNN"
            ax.set_title(f"{name}({self.model.params})")
        self.plot_train_data(ax)

        
        if path == "":
            plt.show()

        else:
            number = f"{n}"
            fig_path = path+f"{self.problem_name}_{self.model.name}_n_{number}_seed_{self.seednr}.pdf"
            fig_path = fig_path.replace(" ", "_")
            plt.savefig(fig_path)

    def plot_true_function(self,ax):
        X_true =  np.linspace(*self.bounds,1000)[:,None]
        Y_true = self.obj_fun(X_true)
        ax.plot(X_true, Y_true, "-",lw = 1, color="Black", zorder=0)
    
    def plot_true_function2(self,ax):
        X_true =  np.linspace(*self.bounds,1000)[:,None]
        Y_true = [self.problem.plot_objectiv_function(x,nr=0) for x in X_true]
        ax.plot(X_true, Y_true, "-",lw = 1, color="Black", zorder=0)
        Y_true = [self.problem.plot_objectiv_function(x,nr=1) for x in X_true]
        ax.plot(X_true, Y_true, "-",lw = 1, color="Black", zorder=0)
        Y_true = [self.problem.plot_objectiv_function(x,nr=2) for x in X_true]
        ax.plot(X_true, Y_true, "-",lw = 1, color="Black", zorder=0)
        Y_true = [self.problem.plot_objectiv_function(x,nr=3) for x in X_true]
        ax.plot(X_true, Y_true, "-",lw = 1, color="Black", zorder=0)
    
    def plot_train_data(self,ax):
        ax.plot(self._X,self._Y, ".", markersize = 5, color="black")  # plot all observed data
        #ax.plot(self._X,self._Y, ".", markersize = 5, color="tab:orange")  # plot all observed data

    def y_gradient(self,y_grid):
        y_grid, *_ = normalize(y_grid, self.model.y_mean, self.model.y_std)
        return np.gradient(y_grid)

    def plot_credible_interval(self, ax, p_predictive,y_grid,x_res,y_res , extent):
        # Compute 95% highest-posterior region
        hpr = np.ones((x_res, y_res))
        for k in range(x_res):
            p_sorted = -np.sort(-(p_predictive[k] * self.y_gradient(y_grid)))
            total_p = (p_predictive[k] * self.y_gradient(y_grid)).sum()
            if total_p<0.01:
                hpr[k, :] = np.nan
                continue
            i = np.searchsorted(np.cumsum(p_sorted/total_p), 0.95)
            idx = (p_predictive[k]*self.y_gradient(y_grid)) < p_sorted[i]
            hpr[k, idx] = 0

        ax.contour(hpr.T, levels=[1], colors="tab:blue", extent=extent , zorder=2)

    def plot_gaussian_approximation(self,ax,show_name = False, only_mean=False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions

        Ymu, Ysigma = self.predict(self.Xgrid)
        Xgrid = self.Xgrid.squeeze()
        ax.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        if not only_mean:
            ax.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma, #alpha = 0.3
                            color="C0", alpha=0.9, label=r"$E[y]\pm 2  \sqrt{Var[y]}$",
                            zorder=1 )  # plot uncertainty intervals
        ax.set_xlim(*self.bounds)
        #ax.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if show_name:
            ax.set_title(f"{self.model.name}({self.model.params})")
        if not only_mean:
            ax.legend(loc=2)

    def plot_predictive_dist(self,ax,show_name = False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions
        x_grid = self.Xgrid.squeeze()
        y_grid = self.ygrid
        predictive_pdf, p_x = self.predictive_pdf(x_grid[:,None], y_grid[:,None], return_px=True, grid1D = True)
        
        dx = (x_grid[1] - x_grid[0]) / 2.0
        dy = (y_grid[1] - y_grid[0]) / 2.0

        extent = [
            x_grid[0] - dx,
            x_grid[-1] + dx,
            y_grid[0] - dy,
            y_grid[-1] + dy,
        ]

        x_res = len(x_grid)
        y_res = len(y_grid)
        
        picture = np.log(predictive_pdf.reshape(x_res, y_res)).T
        picture[picture<-5] = np.nan
        ax.imshow(
            picture,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues',
            vmin=-5, vmax=1, 
            alpha = 0.9
        )
        self.plot_credible_interval( ax, predictive_pdf.reshape(x_res, y_res),y_grid,x_res,y_res , extent )
        
        if p_x is not None:
            p_x = p_x.reshape(x_res, y_res)[:,0]
            ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            prior_weight = self.model.prior_weight
            a = self.model.N*p_x/prior_weight
            ax1.plot(x_grid, a/(a+1), color = color)
            #ax1.set_ylabel(r'$\alpha_x$', color=color)
            ax1.set_ylim(0,5)
            ax1.grid(color=color, alpha = 0.2)
            ticks = [0,0.2,0.4,0.6,0.8,1.0]
            ax1.set_yticks(ticks)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.text(x_grid[len(x_grid)//2],1.1,r"$\alpha(x)$", color=color, size="large")
        


class RegressionTestBase(PlottingClass2):
    def data_generator(self, n_train, n_test):
        #obs, important that already test on the same test-data
        if self.Y_test is None:
            np.random.seed(self.seednr+1)
            X_test, Y_test = self._init_XY(n_test)
            self.X_test, self.Y_test = X_test, Y_test
        else:
            X_test, Y_test = self.X_test, self.Y_test
        np.random.seed(self.seednr)
        X_train, Y_train = self._init_XY(n_train)
        return X_test, Y_test, X_train, Y_train

    def mean_abs_error(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred-y_true))

    def mean_rel_error(self,y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred-y_true)/(np.abs(y_true)+0.000001))

    def mean_pred_log_gaussian(self, mu_pred,sigma_pred,y_true):
        assert mu_pred.shape == y_true.shape
        assert sigma_pred.shape == y_true.shape
        Z_pred = (y_true-mu_pred)/sigma_pred #std. normal distributed. 
        return np.mean(norm.logpdf(Z_pred))
    
    def mean_pred_log_mass(self, x_true,y_true):
        try:
            return np.mean(self.predictive_logpdf(x_true,y_true))
        except:
            return None

    def plot(self,n,output_path):
        assert self.problem_dim==1
        fig,ax = plt.subplots()
        name = self.model.name
        if "BNN" in name:
            name = "BNN"


        np.random.seed(self.seednr)
        bounds_tmp = self.bounds
        self.bounds = (self.bounds[0][0]-10, self.bounds[1][0]+10)
        if "GMR" in name:
            x_res = 100
            y_res = 100
        else:    
            x_res = 500
            y_res = 1000
        self.Xgrid = np.linspace(*self.bounds, x_res)[:, None]
        self.ygrid = np.linspace(0,300, y_res)
        plt.ylim(0,300)
        try:
            self.plot_true_function2(ax)
        except:
            self.plot_true_function(ax)

        self.plot_predictive_dist(ax)
        cmap = plt.cm.Blues
        legend_elements = [Patch(facecolor=cmap(0.6), edgecolor=cmap(0.6),
                        label="predictive distribution")]
                        #label=r'$\hat p(y|x)$')]
        ax.legend(handles=legend_elements)

        self.plot_gaussian_approximation(ax, only_mean = True)

   
        ax.set_title(f"{name}({self.model.params})")
        self.plot_train_data(ax)

        

        number = f"{n}"
        fig_path = output_path+"/"+f"{self.problem_name}_{self.model.name}_n_{number}_seed_{self.seednr}.pdf"
        fig_path = fig_path.replace(" ", "_")
        plt.savefig(fig_path)
        
        self.bounds = bounds_tmp

    def train_test_loop(self, n_train_list, n_test, output_path):
        assert isinstance(n_train_list, list)
        assert isinstance(n_train_list[0], int)
        assert isinstance(n_test, int)

        mean_abs_error = []
        mean_rel_error = []
        mean_pred_likelihod = []
        mean_pred_mass = []
        for n_train in n_train_list:
            DATA =  self.data_generator(n_train, n_test)
            X_test = DATA[0] 
            Y_test = DATA[1]
            X_train = DATA[2]
            Y_train = DATA[3]
            #print(Y_test[:10], Y_train) #Checking if random seed works!
            
            self.fit(X_train,Y_train)
            mu_pred,sigma_pred = self.predict(X_test)
            y_test = Y_test.squeeze()
            mean_abs_error.append(self.mean_abs_error(mu_pred, y_test))
            mean_rel_error.append(self.mean_rel_error(mu_pred, y_test))
            mean_pred_likelihod.append(
                self.mean_pred_log_gaussian(mu_pred,sigma_pred,y_test))
            mean_pred_mass.append(
                self.mean_pred_log_mass(X_test,Y_test))
            print(n_train)
            self._X = X_train
            self._Y = Y_train
            if self.problem_dim==1:
                self.plot(n_train,output_path)


        
        # save data
        data = dict()
        data["n_train_list"]        = n_train_list
        data["n_test"]              = n_test
        data["mean_abs_error"]      = jsonize_array(mean_abs_error)
        data["mean_rel_error"]      = jsonize_array(mean_rel_error)
        data["mean_pred_likelihod"] = jsonize_array(mean_pred_likelihod)
        data["mean_pred_mass"] = jsonize_array(mean_pred_mass)

        time = datetime.today().strftime('%Y-%m-%d-%H_%M')
        filename = f"{self.model.name}_{self.problem_name}_dim_{self.problem_dim}_seed_{self.seednr}_time_{time}.json"
        json.dump(data, open(os.path.join(output_path, filename), "w"))

class RegressionTest_sklearn(BayesOptSolver_sklearn, RegressionTestBase):
    def __init__(self, reg_model, problem, random_seed,acquisition="EI", budget=5, n_init_samples=2, disp=False) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        self.seednr = random_seed
        self.Y_test = None
        self.X_test = None

class RegressionTest(BayesOptSolver_coco, RegressionTestBase):
    def __init__(self, reg_model, problem, random_seed, acquisition="EI", budget=5, n_init_samples=0, disp=False) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        self.seednr = random_seed
        self.Y_test = None
        self.X_test = None

# if __name__ == "__main__":

#     plot_reg = PlotReg1D_mixturemodel(reg_model, problem)
#     plot_reg(20)