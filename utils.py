import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
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
    def her():
        pass

def uniform_grid(bound, n_var, points_pr_dim=100):# -> np.ndarray():
    all_axis = np.repeat(np.linspace(*bound, points_pr_dim)[None,:],n_var, axis=0)
    return np.array([x_i.flatten() for x_i in np.meshgrid(*all_axis)]).T

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

    def plot_surrogate_and_expected_improvement(self,subplot_spec,opt:OptimizationStruct, show_name = False, return_x_next=True):
        if opt.Xgrid is None:
            opt.Xgrid = uniform_grid(self.bounds,1, points_pr_dim=1000)
            #opt.Xgrid = np.linspace(*self.bounds, 1000)
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        self.plot_regression_gaussian_approx(ax1,opt.Xgrid, show_name = show_name)
        self.plot_regression_credible_interval(ax1,opt.Xgrid)
        self.plot_expected_improvement(ax2,opt)