import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from src.utils import normalize, denormalize
from math import sqrt
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
from scipy.special import logsumexp
#logsumexp(a,b)
#np.log(np.sum(b*np.exp(a))) is returned
#However in this implementation, 
# np.log(b*np.sum(np.exp(a))) = np.log(b)+np.log(np.sum(np.exp(a))) #for equal b. 

class naive_GMR:
    #GMM_regression with NO correlation
    def p_xy(self,x,y):
        shape = y.shape
        assert shape == x.shape
        y = y.flatten()[:,None]
        x = x.flatten()[:,None]
        p_xy_all=norm.pdf(x, loc = self.means[:,0], scale = self.x_component_std)*norm.pdf(y, loc=self.means[:,1], scale = self.y_component_std) #(xy.len, components)
        p_xy = np.sum(p_xy_all, axis=1)/self.n_components #same weight on all components. 
        return p_xy.reshape(shape)
        #loop version!
        # p_xy = 0
        # for i in self.n_components:
        #     p_xy +=self.priors[i]*multivariate_normal(self.means[i], self.variances[i]).pdf(x,y)

    def lp_x_all(self, x):
        assert x.ndim == 2
        return norm.logpdf(x, loc = self.means[:,0], scale = self.x_component_std)
         
    def lp_x(self,x, lp_x_all= None):
        if lp_x_all is None:
            lp_x_all = self.lp_x_all(x)
        #p_x = np.sum(p_x_all, axis=1)*self.prior #same weight on all components. 
        lp_x = logsumexp(lp_x_all, axis=1)+np.log(self.prior) #same weight on all components. 
        return lp_x

    def E_predictive(self, X_test): #predictive mean E_{p(y|x)[y]}
        E_y_all =self.means[:,1]
        lp_x_all = self.lp_x_all(X_test) #shape = (X_test,N_components)
        lp_x = self.lp_x(X_test, lp_x_all=lp_x_all)
        a = lp_x_all-lp_x[:,None]
        E_predictive = np.dot(np.exp(a),E_y_all.T)*self.prior
        #E_predictive = np.dot(lp_x_all, E_y_all.T)*self.prior/self.p_x(X_test, lp_x_all=lp_x_all)
        return E_predictive

    def E2_predictive(self, X_test): #predictive second moment E_{p(y|x)[y^2]}
        E_y2_all =self.means[:,1]**2+self.y_component_std**2 #E[y]²+V[y]
        lp_x_all = self.lp_x_all(X_test) #shape = (X_test,N_components)
        lp_x = self.lp_x(X_test, lp_x_all=lp_x_all)
        a = lp_x_all-lp_x[:,None]
        E2_predictive = np.dot(np.exp(a),E_y2_all.T)*self.prior
        #E2_predictive = np.dot(p_x_all, E_y2_all.T)/self.p_x(X_test, p_x_all=p_x_all)*self.prior
        return E2_predictive

class NaiveGMRegression(naive_GMR, BaseEstimator):
    def __init__(self,x_component_std = 5e-2,
                    y_component_std= 5e-2, 
                    prior_settings = {"Ndx": 1,"v_prior":5},
                    manipulate_variance = True, 
                    optimize=False, opt_n_iter=10, opt_cv = 3
                    ):
        self.model = None
        self.name = "Gaussian Mixture Regression"
        self.x_component_std = x_component_std
        self.y_component_std = y_component_std
        self.prior_settings = prior_settings
        self.manipulate_variance = manipulate_variance
        self.optimize_hyperparams = optimize
        self.opt_n_iter, self.opt_cv = opt_n_iter, opt_cv

    def fit(self, X, Y):
        if self.optimize_hyperparams:
            self._optimize(X,Y)
            print("-- Fitted with optimized hyperparams --")
            return

        self.N, self.nX = X.shape
        #nXY = self.nX+1        
        self.n_components = self.N

        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)
        self.means = np.column_stack((X, Y))
        self.prior = 1/self.n_components
        self.params = f"x_k_std = {self.x_component_std}, y_k_std= {self.y_component_std}"

    def score(self, X_test, y_test):
        m_pred, sd_pred, _ = self.predict(X_test)
        Z_pred = (y_test-m_pred)/sd_pred #std. normal distributed. 
        score = np.mean(norm.pdf(Z_pred))
        print(f"mean pred likelihood = {score:0.3f}")
        return score

    def _optimize(self, X, y):
        #OBS! BayesSearchCV only look at the init params! if they are not decleared in params!
        opt = BayesSearchCV(
            self,
            {
                'x_component_std': (1e-3, 3e-1, 'uniform'),
                'y_component_std': (1e-3, 3e-1, 'uniform'),
            },
            n_iter=self.opt_n_iter,
            cv=self.opt_cv
        )
        opt.fit(X, y)
        print(" ")
        print(f"best score = {opt.best_score_}")
        print("best params",opt.best_params_)

        self.__dict__.update(opt.best_estimator_.__dict__)
        #self.set_params(**opt.best_estimator_.get_params())
        # self.fit(X,y) #Not nessesary done by opt.fit
        self.optimize_hyperparams = True

    def get_params(self, deep=False):
        out = dict()      
        out["x_component_std"] = self.x_component_std
        out["y_component_std"] = self.y_component_std
        out["manipulate_variance"] = self.manipulate_variance 
        out["prior_settings"] = self.prior_settings
        #out["optimize"] = self.optimize_hyperparams #gets into trouble with the CV code
        return out
        
    def predict(self, X_test):
        X_test,*_ = normalize(X_test,self.x_mean, self.x_std)
        Ndx = self.prior_settings["Ndx"]
        v_prior = self.prior_settings["v_prior"]

        # likelihood
        m_pred = self.E_predictive(X_test)
        E2_pred = self.E2_predictive(X_test)
        v_pred = E2_pred-m_pred**2 #Var[x] = Ex²-(Ex)²
        
        # evidens
        p_x = np.exp(self.lp_x(X_test))

        # posterior 
        m_pred_bayes = (self.N*p_x*m_pred + Ndx*0)/(self.N*p_x+Ndx)
        E2_pred_bayes = (self.N*p_x*(v_pred+m_pred**2) + Ndx*v_prior)/(self.N*p_x+Ndx) 
        v_pred_bayes = E2_pred_bayes - m_pred_bayes**2

        std_pred_bayes = np.sqrt(v_pred_bayes)
        m_pred_bayes = denormalize(m_pred_bayes, self.y_mean, self.y_std)
        std_pred_bayes *= self.y_std

        return np.array(m_pred_bayes),np.array(std_pred_bayes).T,None

    def _bayesian_conditional_pdf(self,x_grid,y_grid):
        x_grid, *_ = normalize(x_grid, self.x_mean, self.x_std)
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)

        x_res, y_res = x_grid.shape[0], y_grid.shape[0]
        Ndx = self.prior_settings["Ndx"]
        v_prior = self.prior_settings["v_prior"]
        N = self.N


        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        p_xy = self.p_xy(X, Y)
        p_x = np.exp(self.lp_x(x_grid[:,None]))

        p_prior_y = norm(0, np.sqrt(v_prior)).pdf(y_grid)
        p_predictive = (N*p_xy + Ndx*p_prior_y[None, :]) / (N*p_x[:, None] + Ndx)
        return p_predictive, p_x

    def y_gradient(self,y_grid):
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)
        return np.gradient(y_grid)

    def plot(self, ax, xbounds=(0,1),ybounds=(-2.5,2.5)):
        self.x_res, self.y_res  = 500, 800
        x_res, y_res = self.x_res, self.y_res
        x_grid = np.linspace(*xbounds, self.x_res, dtype=np.float)
        y_grid = np.linspace(*ybounds, self.y_res,dtype=np.float)

        p_predictive, p_x = self._bayesian_conditional_pdf(x_grid,y_grid)

         # Compute 95% highest-posterior region
        hpr = np.ones((x_res, y_res), dtype=np.bool)
        for k in range(x_res):
            p_sorted = -np.sort(-(p_predictive[k] * self.y_gradient(y_grid)))
            i = np.searchsorted(np.cumsum(p_sorted), 0.95)
            if i == y_res:
                i = y_res-1
            idx = (p_predictive[k]*self.y_gradient(y_grid)) < p_sorted[i]
            hpr[k, idx] = False

        dx = (x_grid[1] - x_grid[0]) / 2.0
        dy = (y_grid[1] - y_grid[0]) / 2.0
        extent = [
            x_grid[0] - dx,
            x_grid[-1] + dx,
            y_grid[0] - dy,
            y_grid[-1] + dy,
        ]
        ax.imshow(
            np.log(p_predictive).T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues',
            vmin=-5, vmax=1
        )  # , vmin=-3, vmax=1)
        ax.contour(hpr.T, levels=1, extent=extent )
        #mean = self.predict(x_grid[:,None], only_mean = True)
        # if mean is not None:
        #     ax.plot(x_grid,mean,"--", color="red")
        if p_x is not None:
            ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            ax1.plot(x_grid, p_x, color = color)
            ax1.set_ylabel('p(x)', color=color)
            ax1.set_ylim(0,30)
            ax1.grid(color=color, alpha = 0.2)
            ax1.tick_params(axis='y', labelcolor=color)
