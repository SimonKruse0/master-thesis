from cProfile import label
from turtle import color
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.stats import norm
import json
import os
import numpy as np
from datetime import datetime

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

class OptimizationStruct:
    def __init__(self) -> None:
        self.x_next = None
        self.max_EI = None
        self.Xgrid = None
        self.EI_of_Xgrid = None
        self.bounds = None

def uniform_grid(bound, n_var, points_pr_dim=100):# -> np.ndarray():
    all_axis = np.repeat(np.linspace(*bound, points_pr_dim)[None,:],n_var, axis=0)
    return np.array([x_i.flatten() for x_i in np.meshgrid(*all_axis)]).T




class RegressionValidation():
    def __init__(self,problem, regression_model, random_seed) -> None:
        self.model = regression_model
        self.problem = problem
        self.problem_name = type(problem).__name__
        self.problem_size = problem.N
        self.bounds = problem.bounds
        self.mean_abs_pred_error = []
        self.mean_uncertainty_quantification = []
        self.seednr = random_seed

    def data_generator(self, n_train, n_test, use_random_seed = True):
        if use_random_seed:
            np.random.seed(self.seednr)
        n = n_train+n_test
        X = np.random.uniform(*self.bounds[0], size=[n,self.problem_size]) #OBS small hack, fails if bounds are not the same
        y = []
        for i in range(n):
            y.append(self.problem.fun(X[i,:]))
        y=np.array(y)
        self.test_X, self.test_y = X[:n_test], y[:n_test]
        self.train_X, self.train_y = X[n_test:], y[n_test:]
        
    def train_test_loop(self,n_train_points_list, n_test_points):
        self.n_train_points_list = n_train_points_list
        self.n_test_points = n_test_points

        for n_train in n_train_points_list:
            self.data_generator(n_train, n_test_points)
            X,y=self.train_X, self.train_y
            X_test,y_test = self.test_X, self.test_y 
            self.model.fit(X,y)
            Y_mu,Y_sigma,_ = self.model.predict(X_test)
            self.mean_abs_pred_error.append(np.mean(np.abs(y_test-Y_mu)))
            
            Z_pred = (y_test-Y_mu)/Y_sigma #std. normal distributed. 
            self.mean_uncertainty_quantification.append(np.mean(norm.pdf(Z_pred)))
            print(n_train)

    def save_regression_validation_results(self, output_path):

        data = dict()
        data["n_train_points_list"] = self.n_train_points_list
        data["n_test_points"] = self.n_test_points
        data["mean_uncertainty_quantification"] = self.mean_uncertainty_quantification
        data["mean_abs_pred_error"] = [a.astype(float) for a in self.mean_abs_pred_error] #Num pyro gave float32
        time = datetime.today().strftime('%Y-%m-%d-%H_%M')
        filename = f"{self.model.name}_{self.problem_name}_dim_{self.problem_size}_seed_{self.seednr}_time_{time}.json"
        json.dump(data, open(os.path.join(output_path, filename), "w"))


class PlottingClass:
    def __init__(self) -> None:
        pass

    def plot_regression_gaussian_approx(self,ax,Xgrid,show_name = False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions

        #Xgrid = np.linspace(*self.bounds[0], num_grid_points)
        Ymu, Ysigma = self.predict(Xgrid)

        ax.plot(self._X,self._Y, "kx")  # plot all observed data
        ax.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        ax.fill_between(Xgrid.squeeze(), Ymu - 2*Ysigma, Ymu + 2*Ysigma,
                        color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        ax.set_xlim(*self.bounds)
        ax.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if show_name:
            ax.set_title(self.model.name)
        else:
            ax.set_title(self.model.latex_architecture)
        ax.legend(loc=2)

    def plot_regression_credible_interval(self,ax,Xgrid):
        if self.model.name != "numpyro neural network":
            return
        #Ymu, Y_CI = self.predict(Xgrid[:,None], gaussian_approx = False)
        Ymu, Y_CI = self.predict(Xgrid, gaussian_approx = False)

        ax.plot(self._X,self._Y, "kx")  # plot all observed data
        ax.fill_between(Xgrid.squeeze(), Y_CI[0], Y_CI[1],
                                color="black", alpha=0.3, label=r"90\% credible interval")  # plot uncertainty intervals
        ax.legend(loc=2)

    def plot_expected_improvement(self,ax,opt:OptimizationStruct):
        if opt.EI_of_Xgrid is None:
            opt.EI_of_Xgrid = self.expected_improvement(opt.Xgrid)
        ## plot the acquisition function ##
        ax.plot(opt.Xgrid, opt.EI_of_Xgrid) 

        ## plot the new candidate point ##
        ax.plot(opt.x_next[0], opt.max_EI, "^", markersize=10,label=f"x_next = {opt.x_next[0]:.2f}")
        ax.set_xlim(*self.bounds)
        ax.set_ylabel("Acquisition Function")
        ax.legend(loc=1)

    def plot_surrogate_and_expected_improvement(self,subplot_spec,opt:OptimizationStruct, show_name = False):
        if opt.Xgrid is None:
            opt.Xgrid = uniform_grid(self.bounds,1, points_pr_dim=1000)
            #opt.Xgrid = np.linspace(*self.bounds, 1000)
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        self.plot_regression_gaussian_approx(ax1,opt.Xgrid, show_name = show_name)
        self.plot_regression_credible_interval(ax1,opt.Xgrid)
        self.plot_expected_improvement(ax2,opt)

    def plot_2d(self, opt, plot_obj = False, fig_name = "hej"):
        X_next,_ = self.get_optimization_hist()
        X = opt.Xgrid[:,0].reshape(100,100)
        Y = opt.Xgrid[:,1].reshape(100,100)
        if plot_obj:
            Z = self.obj_fun(opt.Xgrid).reshape(100,100)  # Ground truth
        else:
            Ymu, Ysigma = self.predict(opt.Xgrid)
            Z = Ymu.reshape(100,100)
        

        # fig = plt.figure()
        # 
        # if plot_ideal:
        #         ax.plot_surface(X,Y,Z0,rstride=8, cstride=8, alpha=0.1,label='Ground Truth')
        # 
        cs = plt.contourf(X,Y,Z, cmap=cm.coolwarm)
        plt.colorbar()
        #plt.plot(self._X[:,0], self._X[:,1],'ro', label="data")
        cs.changed()
        #ax.scatter(X_next[:,0], X_next[:,1], 0,'rx')
        # ax.plot_surface(X,Y,Z,rstride=8, cstride=8, alpha=0.3,label='Posterior mean')
        # cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z)-1, cmap=cm.coolwarm)
        # #cset = ax.contour(X, Y, Z, zdir='x', offset=lb[0], cmap=cm.coolwarm)
        # #cset = ax.contour(X, Y, Z, zdir='y', offset=ub[1], cmap=cm.coolwarm)
        # ax.set_zlabel('$f(x1,x2)$')

        Y_mu, Y_sigma = self.predict(self._X)
        z = [0 if error > std/10 else 1 for error,std in zip(np.abs(self._Y[:-1,0]-Y_mu[:-1]),Y_sigma[:-1])]
        z = np.array(z)
        colors = np.array(["black", "green"])


        ax = plt.gca()
        ax.scatter(self._X[:-1,0], self._X[:-1,1],c=colors[z],edgecolor="k", label="data")
        for i, txt in enumerate(self._Y):
            if txt[0] is None:
                continue
            ax.annotate(f"{txt[0]:0.2f}", (self._X[i,0], self._X[i,1]))
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(self.model.latex_architecture)
        plt.legend()
        #plt.show()
        # if fig_name:
        #     fig.savefig(f"{fig_name}.pdf")


def normalize(X, mean=None, std=None):
    #zero_mean_unit_var_normalization
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def denormalize(X_normalized, mean, std):
    #zero_mean_unit_var_denormalization
    return X_normalized * std + mean