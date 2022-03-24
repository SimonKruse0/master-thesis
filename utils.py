import matplotlib.pyplot as plt
import numpy as np


class plot_surrogate:
    def __init__(self, ax) -> None:
        self.ax = ax

    def plot_regression_gaussian_approx(self,gs,name = False, num_grid_points = 1000):
        assert self._X.shape[1] == 1   #Can only plot 1D functions

        Xgrid = np.linspace(*self.bounds[0], num_grid_points)
        Ymu, Ysigma = self.predict(Xgrid[:,None])

        ax1 = plt.subplot(gs[0])
        ax1.plot(self._X,self._Y, "kx")  # plot all observed data
        ax1.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        ax1.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma,
                        color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        ax1.set_xlim(*self.bounds[0])
        ax1.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if name:
            ax1.set_title(self.model.name)
        else:
            ax1.set_title(self.model.latex_architecture)
        ax1.legend(loc=2)

    def plot_regression_credible_interval(self,gs,num_grid_points = 1000):
        if self.model.name != "numpyro neural network":
            return
        Xgrid = np.linspace(*self.bounds[0], num_grid_points)
        Ymu, Y_CI = self.predict(Xgrid, gaussian_approx = False)
        ax1 = plt.subplot(gs[0])
        ax1.plot(self._X,self._Y, "kx")  # plot all observed data
        ax1.fill_between(Xgrid, Y_CI[0], Y_CI[1],
                                color="black", alpha=0.3, label=r"90\% credible interval")  # plot uncertainty intervals
        ax1.legend(loc=2)

    def plot_expected_improvement(self,gs,num_grid_points = 1000):
        ax2 = plt.subplot(gs[1])
        Xgrid = np.linspace(*self.bounds[0], num_grid_points)
        
        ## plot the acquisition function ##
        EI = self.expected_improvement(Xgrid)
        ax2.plot(Xgrid, EI) 

        ## plot the new candidate point ##
        
        #x_max,max_EI = find_a_candidate(model,f_best) #slow way
        x_id = np.argmax(EI) #fast way
        x_max = Xgrid[x_id]
        max_EI = EI[x_id]
        ax2.plot(x_max, max_EI, "^", markersize=10,label=f"x_max = {x_max:.2f}")
        ax2.set_xlim(*self.bounds[0])
        ax2.set_ylabel("Acquisition Function")
        ax2.legend(loc=1)
        return x_max

    def plot_surrogate_and_expected_improvement(self,subplot_spec,name = False, return_x_next=True):
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
        self.plot_regression_gaussian_approx(gs, name = name)
        self.plot_regression_credible_interval(gs)
        x_next = self.plot_expected_improvement(gs)
        if return_x_next:
            return x_next