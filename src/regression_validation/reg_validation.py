from turtle import shape
from src.optimization.bayesopt_solver import BayesOptSolverBase, BayesOptSolver_coco, BayesOptSolver_sklearn
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        self.Xgrid = np.linspace(*self.bounds, 300)[:, None]
        self.show_name = True
        self.deterministic = False

    def __call__(self, n,show_gauss= False, show_name = False, path=""):
        fig,ax = plt.subplots()
        self.init_XY_and_fit(n)
        self.fit()
        self.plot_predictive_dist(ax, show_name=show_name)
        if show_gauss:
            self.plot_gaussian_approximation(ax)
        self.plot_true_function(ax)
        self.plot_train_data(ax)

        number = f"{n}"
        fig_path = path+f"{self.problem_name}{self.model.name}_n_{number}.jpg"
        fig_path = fig_path.replace(" ", "_")
        plt.show()
        #plt.savefig(fig_path)

    def plot_true_function(self,ax):
        X_true =  np.linspace(*self.bounds,10000)[:,None]
        Y_true = self.obj_fun(X_true)
        ax.plot(X_true, Y_true, ".", markersize = 1, color="Black")
    
    def plot_train_data(self,ax):
        ax.plot(self._X,self._Y, ".", markersize = 10, color="black")  # plot all observed data


    def plot_gaussian_approximation(self,ax,show_name = False):
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

    def plot_predictive_dist(self,ax,show_name = False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions
        x_grid = self.Xgrid.squeeze()
        y_grid = np.linspace(0,300, 300)
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
        

        ax.imshow(
            np.log(predictive_pdf.reshape(x_res, y_res)).T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues',
            vmin=-5, vmax=1
        )

        Ymu, Ysigma = self.predict(self.Xgrid)
        Xgrid = self.Xgrid.squeeze()
        # ax.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        # ax.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma,
        #                 color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        # ax.set_xlim(*self.bounds)
        # #ax.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if show_name:
            ax.set_title(f"{self.model.name}({self.model.params})")
        
        if p_x is not None:
            p_x = p_x.reshape(x_res, y_res)[:,0]
            ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            Ndx = self.model.Ndx
            a = self.model.N*p_x/Ndx
            ax1.plot(x_grid, a/(a+1), color = color)
            #ax1.set_ylabel(r'$\alpha_x$', color=color)
            ax1.set_ylim(0,5)
            ax1.grid(color=color, alpha = 0.2)
            ticks = [0,0.2,0.4,0.6,0.8,1.0]
            ax1.set_yticks(ticks)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.text(x_grid[len(x_grid)//2],1.1,r"$\alpha(x)$", color=color, size="large")
        
        ax.legend(loc=2)


class RegressionTestBase():
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

    def mean_pred_gaussian(self, mu_pred,sigma_pred,y_true):
        assert mu_pred.shape == y_true.shape
        assert sigma_pred.shape == y_true.shape
        Z_pred = (y_true-mu_pred)/sigma_pred #std. normal distributed. 
        return np.mean(norm.pdf(Z_pred))
    
    def mean_pred_mass(self, x_true,y_true):
        try:
            return np.mean(self.predictive_pdf(x_true,y_true))
        except:
            return None

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
                self.mean_pred_gaussian(mu_pred,sigma_pred,y_test))
            mean_pred_mass.append(
                self.mean_pred_mass(X_test,Y_test))
            print(n_train)
        
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