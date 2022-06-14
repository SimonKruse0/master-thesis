from turtle import Turtle
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
from src.utils import normalize, denormalize
from math import sqrt
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from scipy.special import logsumexp
#logsumexp(a,b)
#np.log(np.sum(b*np.exp(a))) is returned
#However in this implementation, 
# np.log(b*np.sum(np.exp(a))) = np.log(b)+np.log(np.sum(np.exp(a))) #for equal b. 

class naive_GMR:
    #GMM_regression with NO correlation
    def p_xy(self,x,y):
        # ONly works for 1D x...!
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

    def p_xy_nd(self,x,y):
        assert x.shape[1] == self.nX
        assert y.shape[1] == 1
        lp_xy_all=np.sum(norm.logpdf(x, loc = self.means[:,None,:-1], scale = self.x_component_std), axis = 2)
        lp_xy_all+= norm.logpdf(y, loc=self.means[:,None,-1:], scale = self.y_component_std).squeeze()
        #assert lp_xy_all.shape[0] == self.means.shape[0]
        p_xy = np.sum(np.exp(lp_xy_all), axis=0)/self.n_components 
        return p_xy

    def lp_x_all(self, x):
        assert x.ndim == 2
        #return norm.logpdf(x, loc = self.means[:,0], scale = self.x_component_std)
        return norm.logpdf(x, loc = self.means[:,None,:-1], scale = self.x_component_std).sum(axis=2).T
        #return norm.logpdf(x, loc = self.means[:,:-1].T.flatten(), scale = self.x_component_std)
        #Ok summes sammen!
         
    def lp_x(self,x, lp_x_all= None):
        if lp_x_all is None:
            lp_x_all = self.lp_x_all(x)
        #p_x = np.sum(p_x_all, axis=1)*self.prior #same weight on all components. 
        lp_x = logsumexp(lp_x_all, axis=1)+np.log(self.prior) #same weight on all components. 
        return lp_x

    def E_predictive(self, X_test): #predictive mean E_{p(y|x)[y]}
        E_y_all =self.means[:,-1]
        lp_x_all = self.lp_x_all(X_test) #shape = (X_test,N_components)
        lp_x = self.lp_x(X_test, lp_x_all=lp_x_all)
        a = lp_x_all-lp_x[:,None]
        E_predictive = np.dot(np.exp(a),E_y_all.T)*self.prior
        #E_predictive = np.dot(lp_x_all, E_y_all.T)*self.prior/self.p_x(X_test, lp_x_all=lp_x_all)
        return E_predictive

    def E2_predictive(self, X_test): #predictive second moment E_{p(y|x)[y^2]}
        E_y2_all =self.means[:,-1]**2+self.y_component_std**2 #E[y]²+V[y]
        lp_x_all = self.lp_x_all(X_test) #shape = (X_test,N_components)
        lp_x = self.lp_x(X_test, lp_x_all=lp_x_all)
        a = lp_x_all-lp_x[:,None]
        E2_predictive = np.dot(np.exp(a),E_y2_all.T)*self.prior
        #E2_predictive = np.dot(p_x_all, E_y2_all.T)/self.p_x(X_test, p_x_all=p_x_all)*self.prior
        return E2_predictive

class NaiveGMRegression(naive_GMR, BaseEstimator):
    def __init__(self,x_component_std = 5e-2,
                    y_component_std= 5e-2, 
                    prior_weight =  1e-6,
                    manipulate_variance = False, 
                    optimize=False, opt_n_iter=40, opt_cv = 10, 
                    predictive_score = False):
        self.name = f"KDE"
        self.x_component_std = x_component_std
        self.y_component_std = y_component_std
        self.sig_prior = 1.1
        self.prior_weight = prior_weight
        self.manipulate_variance = manipulate_variance
        self.optimize_hyperparams = optimize
        self.opt_n_iter, self.opt_cv = opt_n_iter, opt_cv
        self.predictive_score = predictive_score

    def fit(self, X, Y):
        if self.optimize_hyperparams:
            if X.shape[0] >= self.opt_cv:
                #self.opt_cv = min(30,X.shape[0]) #leave one out!!
                self._optimize( X, Y)
                print("-- Fitted with optimized hyperparams --")
                return
            else:
                print("-- Fitting with default hyperparams since too little data for CV-- ")

        self.N, self.nX = X.shape     
        self.n_components = self.N

        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)
        self.means = np.column_stack((X, Y))
        self.prior = 1/self.n_components
        self.params = f"sig_x = {self.x_component_std:0.2e}, sig_y = {self.y_component_std:0.2e}, w_prior = {self.prior_weight:0.2e}"
        print(self.params)

    def score(self, X_test, y_test):
        y_test = y_test.squeeze()
        assert y_test.ndim <= 1 
        m_pred, sd_pred, _ = self.predict(X_test)
        assert m_pred.ndim == 1
        assert sd_pred.ndim == 1
        if self.predictive_score:
            score = -np.mean(abs(y_test-m_pred))
            print(f"negative mean pred error = {score:0.3f}")
        else:
            #score = -np.mean(abs(y_test-m_pred))/10
            if y_test.ndim == 0:
                y_test = np.array([y_test, y_test])
                X_test = X_test.repeat(2)[:,None]
            p_predictive, p_x = self.predictive_pdf(X_test, y_test[:,None])
            score = np.mean(p_predictive)

            print(f"mean pred dist = {score:0.3f}")

            # Z_pred = (y_test-m_pred)/sd_pred #std. normal distributed. 
            # score = np.mean(norm.pdf(Z_pred))
            # print(f"mean pred likelihood = {score:0.3f}")
        return score

    def _optimize(self, X, y):
        #OBS! BayesSearchCV only look at the init params! if they are not decleared in params!
        opt = BayesSearchCV(
            self,
            {
                'x_component_std': (1e-3, 3e-1, 'uniform'),
                'y_component_std': (1e-3, 3e-1, 'uniform'),
                #'prior_weight' : (1e-6, 1., 'uniform')
            },
            n_iter=self.opt_n_iter,
            cv=self.opt_cv,
            #n_points = 2,
            n_jobs = 4,
           #optimizer_kwargs={'base_estimator': 'RF'} #33 vs 35.5
        )
        # parameters = {'x_component_std':np.linspace(1e-3, 3e-1,30), 
        #                 'y_component_std':np.linspace(1e-3, 3e-1,30),
        #                 'prior_weight':np.linspace(1e-6, 1,20)}
        # opt = GridSearchCV(self, parameters,cv=self.opt_cv)

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
        out["opt_n_iter"] = self.opt_n_iter
        out["opt_cv"] = self.opt_cv
        out["prior_weight"] = self.prior_weight
        #out["optimize"] = self.optimize_hyperparams #gets into trouble with the CV code
        return out

    def predictive_pdf(self,X,Y):
        X,*_ = normalize(X,self.x_mean, self.x_std)
        Y,*_ = normalize(Y,self.y_mean, self.y_std)
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior
        N = self.N
        assert X.ndim ==2
        assert Y.ndim ==2
        p_xy = self.p_xy_nd(X, Y)
        #print("p_xy", p_xy)
        p_x = np.exp(self.lp_x(X))
        p_prior_y = norm(0, sig_prior).pdf(Y)
        p_predictive = (N*p_xy + prior_weight*p_prior_y.squeeze()) / (N*p_x + prior_weight)    
        return p_predictive, p_x

    def predict(self, X_test):
        X_test,*_ = normalize(X_test,self.x_mean, self.x_std)
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior

        # likelihood
        m_pred = self.E_predictive(X_test)
        E2_pred = self.E2_predictive(X_test)
        v_pred = E2_pred-m_pred**2 #Var[x] = Ex²-(Ex)²
        assert not any(v_pred<0) 
        
        # evidens
        p_x = np.exp(self.lp_x(X_test))

        # posterior 
        m_pred_bayes = (self.N*p_x*m_pred + prior_weight*0)/(self.N*p_x+prior_weight)
        E2_pred_bayes = (self.N*p_x*(v_pred+m_pred**2) + prior_weight*sig_prior**2)/(self.N*p_x+prior_weight) 
        v_pred_bayes = E2_pred_bayes - m_pred_bayes**2

        std_pred_bayes = np.sqrt(v_pred_bayes)

        m_pred_bayes = denormalize(m_pred_bayes, self.y_mean, self.y_std)
        std_pred_bayes *= self.y_std

        return m_pred_bayes,std_pred_bayes,p_x #HACK

    def _bayesian_conditional_pdf(self,x_grid,y_grid):
        x_grid, *_ = normalize(x_grid, self.x_mean, self.x_std)
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)

        x_res, y_res = x_grid.shape[0], y_grid.shape[0]
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior
        N = self.N

        X, Y = np.meshgrid(x_grid, y_grid, indexing="ij")
        p_xy = self.p_xy(X, Y)
        p_x = np.exp(self.lp_x(x_grid[:,None]))
        p_prior_y = norm(0, sig_prior).pdf(y_grid)
        p_predictive = (N*p_xy + prior_weight*p_prior_y[None, :]) / (N*p_x[:, None] + prior_weight)
        return p_predictive, p_x

    def y_gradient(self,y_grid):
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)
        return np.gradient(y_grid)

    def plot(self, ax, xbounds=(0,1),ybounds=(-2.5,2.5), plot_credible_set = False):
        self.x_res, self.y_res  = 300, 200
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
        
        if plot_credible_set:
            ax.contour(hpr.T, levels=1, extent=extent )
        #mean = self.predict(x_grid[:,None], only_mean = True)
        # if mean is not None:
        #     ax.plot(x_grid,mean,"--", color="red")
        if p_x is not None:
            ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            prior_weight = self.prior_weight
            a = self.N*p_x/prior_weight
            ax1.plot(x_grid, a/(a+1), color = color)
            #ax1.set_ylabel(r'$\alpha_x$', color=color)
            ax1.set_ylim(0,5)
            ax1.grid(color=color, alpha = 0.2)
            ticks = [0,0.2,0.4,0.6,0.8,1.0]
            ax1.set_yticks(ticks)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.text(x_grid[len(x_grid)//2],1.1,r"$\alpha(x)$", color=color, size="large")

def obj_fun_nd(x): 
    return np.sum(0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1, axis = 1)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    bounds = [0,1]
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 200
    np.random.seed(20)
    X_sample =  np.random.uniform(*bounds,size = (datasize,1))
    Y_sample = obj_fun_nd(X_sample)[:,None]

    reg_model = NaiveGMRegression(optimize=True, opt_n_iter=5)
    reg_model.fit(X_sample, Y_sample)
    fig,ax = plt.subplots()
    reg_model.plot(ax)
    plt.show()
    print(reg_model.predictive_pdf(X_sample[:10,:]+0.01*np.random.rand(10)[:, None], Y_sample[:10,:]))
