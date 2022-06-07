import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from scipy.stats import norm, invgamma
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
        self.y_next = None
        self.max_EI = None
        self.Xgrid = None
        self.EI_of_Xgrid = None
        self.bounds = None

def uniform_grid(bound, n_var, points_pr_dim=1000):# -> np.ndarray():
    all_axis = np.repeat(np.linspace(*bound, points_pr_dim)[None,:],n_var, axis=0)
    return np.array([x_i.flatten() for x_i in np.meshgrid(*all_axis)]).T



def jsonize_array(array):
    return [a.astype(float) for a in array]



def plot_inv_gamma(ax,alpha, beta):
    x = np.linspace(0, 3, 300)
    ax.plot(x, invgamma.pdf(x, alpha, beta))
    return ax


def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

class PlottingClass:
    def __init__(self) -> None:
        pass

    def plot_regression_gaussian_approx(self,ax,Xgrid,show_name = False):
        assert self._X.shape[1] == 1   #Can only plot 1D functions

        #Xgrid = np.linspace(*self.bounds[0], num_grid_points)
        Ymu, Ysigma = self.predict(Xgrid)
        Xgrid = Xgrid.squeeze()
        #ax.plot(self._X,self._Y, "kx", lw=2)  # plot all observed data
        ax.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        ax.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma,
                        color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
        #ax.set_ylim(-0.7+np.min(self._Y), 0.5+0.7+np.max(self._Y))
        if show_name:
            ax.set_title(f"{self.model.name}({self.model.params})")
        else:
            #ax.set_title(self.model.latex_architecture)
            pass
        ax.legend(loc=2)

    def plot_regression_credible_interval(self,ax,Xgrid):
        if self.model.name != "numpyro neural network":
            return
        #Ymu, Y_CI = self.predict(Xgrid[:,None], gaussian_approx = False)
        Ymu, Y_CI = self.predict(Xgrid, gaussian_approx = False)

        ax.fill_between(Xgrid.squeeze(), Y_CI[0], Y_CI[1],
                                color="black", alpha=0.3, label=r"90\% credible interval")  # plot uncertainty intervals
        ax.legend(loc=2)

    def plot_expected_improvement(self,ax,Xgrid):
        opt = self.opt
        EI_of_Xgrid, exploitation, exploration  = self.expected_improvement(Xgrid, return_analysis = True)
        # plot the acquisition function ##
        ax.plot(Xgrid, EI_of_Xgrid, color="tab:blue", label = "EI") 
        ax.plot(Xgrid, exploitation, "--", color = "cyan", label="exploitation") 
        ax.plot(Xgrid, exploration, "--", color = "red", label="exploration") 
        # max_EI_id = np.argmax(EI_of_Xgrid)
        # max_EI = EI_of_Xgrid[max_EI_id]
        # x_next = Xgrid[max_EI_id]
        # ax.plot(x_next, max_EI, "^", markersize=10,color="tab:orange", label=f"x_next = {opt.x_next[0]:.2f}")

        # ax.plot(opt.Xgrid, opt.EI_of_Xgrid, color="tab:blue", label = "EI") 
        # ax.plot(opt.Xgrid, opt._exploitation, "--", color = "cyan", label="exploitation") 
        # ax.plot(opt.Xgrid, opt._exploration, "--", color = "red", label="exploration") 


        #EI_of_Xgrid 
        ## plot the new candidate point ##
        #ax.plot(opt.x_next[0], opt.max_EI, "^", markersize=10,color="tab:orange", label=f"x_next = {opt.x_next[0]:.2f}")
        ax.set_xlim(self.bounds[0][0], self.bounds[1][0])
        ax.set_ylabel("Acquisition Function")
        ax.legend(loc=1)

    # def plot_expected_improvement_parts(self,ax,opt:OptimizationStruct):
    #     EI, Epl = self.expected_improvement(opt.Xgrid, )

    def plot_surrogate_and_expected_improvement(self,subplot_spec,opt:OptimizationStruct = None, show_name = False):
        if opt is None: #HACK
            opt = self.opt
        bounds = (self.bounds[0][0], self.bounds[1][0])
        Xgrid = uniform_grid(bounds,1, points_pr_dim=1000)
        gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot_spec)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])

        
        self.plot_regression_gaussian_approx(ax1,Xgrid, show_name = show_name)
        #self.plot_regression_credible_interval(ax1,opt.Xgrid)
        X_true =  np.linspace(*self.bounds,1000)
        Y_true = self.obj_fun(X_true)
        ax1.plot(X_true, Y_true, "--", color="Black")
        ax1.plot(self._X,self._Y, ".", markersize = 10, color="black")  # plot all observed data
        ax1.plot(self._X[-1],self._Y[-1], ".", markersize = 10, color="tab:orange")  # plot all observed data
        self.plot_expected_improvement(ax2,Xgrid)
        x_next = opt.x_next[:,None]
        max_EI, *_ = self.expected_improvement(x_next, return_analysis = False)
        ax2.plot(x_next, max_EI, "^", markersize=10,color="tab:orange", label=f"x_next = {opt.x_next[0]:.2f}")


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


class RegressionValidation(PlottingClass):
    def __init__(self,problem, regression_model, random_seed) -> None:
        self.model = regression_model
        self.problem = problem
        self.problem_name = type(problem).__name__
        self.problem_dim = problem.N
        self.bounds = problem.bounds
        self.mean_abs_pred_error = []
        self.mean_rel_pred_error = []
        self.mean_uncertainty_quantification = []
        self.seednr = random_seed

    def data_generator(self, n_train, n_test, use_random_seed = True):
        if use_random_seed:
            np.random.seed(self.seednr)
        n = n_train+n_test
        X = np.random.uniform(*self.bounds[0], size=[n,self.problem_dim]) #OBS small hack, fails if bounds are not the same
        y = []
        for i in range(n):
            y.append(self.problem.fun(X[i,:]))
        y=np.array(y)
        self.test_X, self.test_y = X[:n_test], y[:n_test]
        self.train_X, self.train_y = X[n_test:], y[n_test:]
    
    def predict(self, X_test):
        Y_mu,Y_sigma,_  = self.model.predict(X_test)
        return Y_mu, Y_sigma

    def train_test_loop(self,n_train_points_list, n_test_points, path = None):
        self.n_train_points_list = n_train_points_list
        self.n_test_points = n_test_points

        for n_train in n_train_points_list:
            self.data_generator(n_train, n_test_points)
            X,y=self.train_X, self.train_y
            X_test,y_test = self.test_X, self.test_y 
            self.model.fit(X,y[:,None])
            Y_mu,Y_sigma,_ = self.model.predict(X_test)
            self.mean_abs_pred_error.append(np.mean(np.abs(y_test-Y_mu)))
            self.mean_rel_pred_error.append(
                np.mean(np.abs(y_test-Y_mu)/(np.abs(y_test)+0.001)))
            
            Z_pred = (y_test-Y_mu)/Y_sigma #std. normal distributed. 
            self.mean_uncertainty_quantification.append(np.mean(norm.pdf(Z_pred)))
            print(n_train)
            if self.problem_dim == 1:
                self._save_plot(X, y, path)

    def save_regression_validation_results(self, output_path):

        data = dict()
        data["n_train_points_list"] = self.n_train_points_list
        data["n_test_points"] = self.n_test_points
        #data["y_test"] = jsonize_array(self.test_y)
        #data["y_pred"] = jsonize_array(self.Y_mu)
        #data["y_sigma"] = jsonize_array(self.Y_sigma)
        data["mean_uncertainty_quantification"] = self.mean_uncertainty_quantification
        data["mean_abs_pred_error"] = jsonize_array(self.mean_abs_pred_error) #Num pyro gave float32
        data["mean_rel_pred_error"] = jsonize_array(self.mean_rel_pred_error) #Num pyro gave float32
        data["problem_name"] = self.problem_name
        data["problem dim"] = self.problem_dim
        data["model_name"] = self.model.name
        data["params"] = self.model.params
        time = datetime.today().strftime('%Y-%m-%d-%H_%M')
        filename = f"{self.model.name}_{self.problem_name}_dim_{self.problem_dim}_seed_{self.seednr}_time_{time}.json"
        json.dump(data, open(os.path.join(output_path, filename), "w"))

    def _save_plot(self, X, Y, output_path):
        assert self.problem_dim == 1
        self._X,self._Y = X, Y
        fig, ax = plt.subplots()
        self.plot_regression_gaussian_approx(ax, np.linspace(*self.bounds[0], 200)[:,None], show_name=True)
        ax.plot(self.test_X, self.test_y, ".", color="blue")
        ax.plot(self._X,self._Y,'.', markersize=10, color="yellow")
        time = datetime.today().strftime('%Y-%m-%d-%H_%M')
        n_train = self.train_X.shape[0]
        filename_png = f"{self.model.name}_{self.problem_name}_dim_{self.problem_dim}_seed_{self.seednr}_time_{time}_n_{n_train}.png"
        plt.savefig(os.path.join(output_path, filename_png))


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

print(__name__)
if __name__ == "__main__":
    fig, ax = plt.subplots()
    plot_inv_gamma(ax, 5,0.01)
    plt.show()