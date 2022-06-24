from pickletools import optimize
from gmr import GMM, plot_error_ellipses
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from gmr.mvn import MVN
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
from scipy.stats import norm
from src.utils import normalize, denormalize
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.base import BaseEstimator

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class GMM_bayesian(GMM):
    def __init__(self, n_components,N, prior_weight, sig_prior, priors=None, means=None, covariances=None, verbose=0, random_state=None):
        super().__init__(n_components, priors, means, covariances, verbose, random_state)
        self.N = N
        self.prior_weight = prior_weight
        self.sig_prior = sig_prior
        #Manipulation of gmr.GMM functions
        # def predict():
        #     raise "don't use this"

    def predict(self, X_test , manipulate_variance = False):
        n_data = self.N
        m_preds,std_preds  = [], []
        for i,x in enumerate(X_test):
            if i%10 ==0:
                print(f"Points tested {100*i/X_test.shape[0]:0.1f}%", end="\r")
            conditional_gmm = self.condition(x)
            p_x = self.marginalize(x) #probability of data at the x. 
            prior_weight = self.prior_weight 
            sig_prior = self.sig_prior

            m_pred = (p_x*n_data*conditional_gmm.mean() + prior_weight*0)/(n_data*p_x+prior_weight)
            v_pred = (p_x*n_data*(conditional_gmm.variance()+conditional_gmm.mean()**2)+
                        prior_weight*sig_prior**2)/(n_data*p_x+prior_weight) - m_pred**2

            m_preds.append(m_pred)
            if manipulate_variance:
                v_pred /= np.clip(p_x*50,1,40) 
            std_preds.append(sqrt(v_pred))
        return np.array(m_preds), np.array(std_preds)
    
    def predictive_pdf(self,X,Y):
        p_predictive = []
        p_x_list = []
        N = self.N
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior

        #sort() ??

        for x,y in zip(X,Y):
            conditional_gmm = self.condition(x)
            p_x = self.marginalize(x)[0]
            p_conditional_gmm = conditional_gmm.to_probability_density(y)[0]
            p_prior_y = norm(0, sig_prior).pdf(y)[0]
            p_predictive_tmp = (p_x*N*p_conditional_gmm+ prior_weight*p_prior_y)/(p_x*N+prior_weight)
            p_predictive.append(p_predictive_tmp)
            p_x_list.append(p_x)

        return np.array(p_predictive), np.array(p_x_list)

    def _bayesian_conditional_pdf(self, x_grid,y_grid , manipulate_variance = False):
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior
        n_data = self.N
        p_predictive = np.zeros((len(x_grid),len(y_grid)))
        p_prior_y = norm(0, sig_prior).pdf(y_grid)
        p_x_list = []
        for i,x in enumerate(x_grid):
            if i%10 ==0:
                print(f"Points evaluated {100*i/x_grid.shape[0]:0.1f}%", end="\r")
            conditional_gmm = self.condition(x)
            p_x = self.marginalize(x) #probability of data at the x. 
            for j,y in enumerate(y_grid):
                p_conditional_gmm = conditional_gmm.to_probability_density(y)
                p_predictive[i,j] = (p_x*n_data*p_conditional_gmm)/(p_x*n_data+prior_weight)
                #p_predictive[i,j] = (p_x*n_data*p_conditional_gmm + prior_weight*p_prior_y[j])/(p_x*n_data+prior_weight)
            p_predictive[i,:] += prior_weight*p_prior_y/(p_x*n_data+prior_weight)
            p_x_list.append(p_x)
        return p_predictive,np.array(p_x_list)

    def mean(self):
        mean = 0 #only 1D since we only need E[y|x]
        for k in range(self.n_components):
            mean += self.priors[k]*self.means[k]
        return mean

    def variance(self):
        second_moment = 0 
        for k in range(self.n_components):
            second_moment += self.priors[k]*(self.covariances[k]+self.means[k]**2) #E[y^2]
        variance =  second_moment - self.mean()**2 
        return variance

    def marginalize(self,x, indices= None):
        if indices is None:
            indices = np.arange(self.means.shape[1]-1)
        # marginalizes over the variables NOT in indices, i.e. y
        p_x = 0
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],random_state=self.random_state)
            p_x += self.priors[k]*mvn.marginalize(indices).to_probability_density(x)
        return p_x

    #Changing the conditional function in GMM
    def condition(self, x):
        """Conditional distribution over given indices.
        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        self._check_initialized()
        
        indices = np.arange(self.means.shape[1]-1, dtype=int)
        x = np.asarray([x])

        means = np.empty((self.n_components, 1))
        covariances = np.empty((self.n_components, 1, 1))

        marginal_norm_factors = np.empty(self.n_components)
        marginal_prior_exponents = np.empty(self.n_components)

        # calculate the margianl p(x) = sum pi_k*p_k(x)
        #p_x = self.marginalize(indices)

        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                        random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance
            marginal_norm_factors[k], marginal_prior_exponents[k] = \
                mvn.marginalize(indices).to_norm_factor_and_exponents(x) # These values can be used to compute the probability density function
                                                           # of this Gaussian: p(x) = norm_factor * np.exp(exponents).
        
        priors = _safe_probability_density(
            self.priors * marginal_norm_factors,
            marginal_prior_exponents[np.newaxis])[0]

        return GMM_bayesian(self.n_components, self.N, self.prior_weight, self.sig_prior,priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

def _safe_probability_density(norm_factors, exponents):
    m = np.max(exponents, axis=1)[:, np.newaxis]
    p = norm_factors[np.newaxis] * np.exp(exponents - m)
    p /= np.sum(p, axis=1)[:, np.newaxis]
    return p

class GMRegression(BaseEstimator):
    def __init__(self,optimize=False, manipulate_variance = False, 
                    n_components = 2,
                    prior_weight = 1e-6,
                    opt_n_iter  =40, opt_cv = 10,
                    train_epochs = 1000, 
                    sig_prior = 10,
                    extra_name=""):
        self.model = None
        self.name = f"GMR-{extra_name}"
        self.prior_weight, self.sig_prior = prior_weight,sig_prior
        self.manipulate_variance = manipulate_variance
        self.n_components = n_components
        self.predictive_score = False
        self.optimize_hyperparams = optimize
        self.opt_n_iter = opt_n_iter
        self.opt_cv = opt_cv
        self.train_epochs = train_epochs
        

    def fit(self, X, Y):
        self.N, self.nX = X.shape
        assert Y.ndim == 2
        if self.optimize_hyperparams:
            if X.shape[0] >= 10:
                self.opt_cv = min(30,X.shape[0])
                self._optimize( X, Y, int(self.N * (self.opt_cv-1)/(self.opt_cv)))
                print("-- Fitted with optimized hyperparams --")
                return
            else:
                print("-- Fitting with default hyperparams since too little data for CV-- ")
        self.params = f"n_components = {self.n_components}, prior_w = {self.prior_weight}"
        print(self.params)

        X_, self.x_mean, self.x_std = normalize(X)
        Y_, self.y_mean, self.y_std = normalize(Y)
        XY_train = np.column_stack((X_, Y_))
        # gmm_sklearn = BayesianGaussianMixture(n_components=self.n_components,
        #                                 covariance_type = "full",
        #                                 weight_concentration_prior =  1e-9,#1. / n_components, 
        #                                 mean_precision_prior = 1e-9,#0.001, 
        #                                 #covariance_prior =np.array([[0.001,0],[0,0.001]]),
        #                                 n_init = 10, init_params="kmeans", 
        #                                 verbose=2
        #                                 )
        #                                 #covariance_prior =np.diag(np.diag(np.cov(XY_train.T)))/1000)
        #                                 #weight_concentration_prior_type="dirichlet_distribution")
        gmm_sklearn = GaussianMixture(n_components=self.n_components, 
                                    covariance_type="full", 
                                    max_iter=self.train_epochs, 
                                    verbose = 0,
                                    tol = 1e-6,
                                    init_params="kmeans", 
                                    n_init=20)
        gmm_sklearn.fit(XY_train)

        self.model = GMM_bayesian(
        self.n_components, self.N, self.prior_weight, self.sig_prior, priors=gmm_sklearn.weights_, means=gmm_sklearn.means_,
        covariances=gmm_sklearn.covariances_)
    def score(self, X_test, y_test):
        y_test = y_test.squeeze()
        assert y_test.ndim <= 1 
        if self.predictive_score:
            m_pred, sd_pred = self.predict(X_test)
            assert m_pred.ndim == 1
            assert sd_pred.ndim == 1
            score = -np.mean(abs(y_test-m_pred))
            print(f"negative mean pred error = {score:0.3f}")
        else:
            if y_test.ndim == 0:
                y_test = np.array([y_test, y_test])
                X_test = X_test.repeat(2)[:,None]
            p_predictive, p_x = self.predictive_pdf(X_test, y_test[:,None])
            score = np.mean(np.log(p_predictive))
            # Z_pred = (y_test-m_pred)/sd_pred #std. normal distributed. 
            # score = np.mean(norm.pdf(Z_pred))
            print(f"mean log predictive = {score:0.3f}")
        print(" ")
        return score

    def get_params(self, deep=False):
        out = dict()
        out["n_components"] = self.n_components
        out["prior_weight"] = self.prior_weight
        #out["optimize"] = self.optimize_hyperparams #gets into trouble with the CV code
        return out

    def _optimize(self, X, y, n):
        #OBS! BayesSearchCV only look at the init params! if they are not decleared in params!
        opt = BayesSearchCV(
            self,
            {
                'n_components': Integer(1,n),
                'prior_weight' : Real(1e-6, 1., 'uniform'),
            },
            n_iter=self.opt_n_iter,
            cv=self.opt_cv, 
            n_jobs=4
        )
        opt.fit(X, y)
        print(" ")
        print(f"best score = {opt.best_score_}")
        print("best params",opt.best_params_)

        self.__dict__.update(opt.best_estimator_.__dict__)
        #self.set_params(**opt.best_estimator_.get_params())
        # self.fit(X,y) #Not nessesary done by opt.fit
        self.optimize_hyperparams = True

    def predictive_pdf(self,X,Y):
        X,*_ = normalize(X,self.x_mean, self.x_std)
        Y,*_ = normalize(Y,self.y_mean, self.y_std)
        return self.model.predictive_pdf(X,Y)

    def predict(self,X_test, CI=[0.05,0.95]):
        #print(X_test.shape)
        X_test,*_  = normalize(X_test,self.x_mean, self.x_std)
        mean_preds, std_preds = self.model.predict(X_test, manipulate_variance=self.manipulate_variance)
        
        mean_preds = denormalize(mean_preds, self.y_mean, self.y_std)
        std_preds *= self.y_std

        return mean_preds.squeeze(),std_preds,None

    def _bayesian_conditional_pdf(self,x_grid,y_grid):
        x_grid, *_ = normalize(x_grid, self.x_mean, self.x_std)
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)
        return self.model._bayesian_conditional_pdf(x_grid, y_grid)

    def plot(self, ax, xbounds=(-0.1,1.1),ybounds=(-2.5,2.5)):
        x_res, y_res  = 100, 100
        x_grid = np.linspace(*xbounds, x_res, dtype=float)
        y_grid = np.linspace(*ybounds, y_res,dtype=float)

        p_predictive,p_x = self._bayesian_conditional_pdf(x_grid,y_grid)

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

        if p_x is not None:
            ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            prior_weight = 1 #self.prior_settings["prior_weight"]
            a = self.N*p_x/prior_weight
            ax1.plot(x_grid, a/(a+1), color = color)
            #ax1.set_ylabel(r'$\alpha_x$', color=color)
            ax1.set_ylim(0,5)
            ax1.grid(color=color, alpha = 0.2)
            ticks = [0,0.2,0.4,0.6,0.8,1.0]
            ax1.set_yticks(ticks)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.text(x_grid[len(x_grid)//2],1.1,r"$\alpha(x)$", color=color, size="large")


def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1
if __name__ == "__main__":

    N = 5
    np.random.seed(20) #weird thing happening!
    np.random.seed(1)
    X =  np.random.uniform(0,1,size = (N,1))
    y = obj_fun(X)


    if True:
        GMR = GMRegression(optimize=False, n_components=3)
        GMR.fit(X,y)
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        x_test_list = np.linspace(-0.1,1.1,500)
        mu, sd, _ = GMR.predict(x_test_list)
        mu =mu.squeeze()
        plt.plot(x_test_list, mu, color = "red")
        #plt.fill_between(x_test_list, mu-2*sd,mu+2*sd ,color="orange",alpha=0.4)
        ax.scatter(X, y, s=3, color="black")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.legend()
        # plt.show()
        # ax = plt.subplot(111)
        GMR.plot(ax)


        # plot_error_ellipses(ax, GMR.model, alpha = 1)
        # ax.set_ylim([-1,2])
        # ax.set_xlim([-1,1.2])
        plt.show()
    else:
        XY_train = np.column_stack((x, y))
        gmm = GMM_bayesian(
            n_components=N, priors=np.repeat(1/N, N), means=XY_train,
            covariances=np.repeat([np.eye(2)/10000], N, axis=0))#np.array([np.diag(c) for c in gmm_sklearn.covariances_]))
        
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        ax.set_title("Dataset and GMM")
        colors = ["r", "g", "b", "orange"]*100
        #colors = ["r"]*100
        plot_error_ellipses(ax, gmm, colors=colors, alpha = 0.1)
        x_test_list = np.linspace(-0.1,1.1,1000)
        CI = []
        CI1 = []
        CI2 = []

        mu = []
        sd = []

        alpha_list = []
        for x_test in x_test_list:
            conditional_gmm, alpha = gmm.condition([0], [x_test], manipulate_test_bounds = False)

            mu.append(conditional_gmm.mean())
            var = conditional_gmm.variance()
            var /= np.clip(alpha*20,1,20) 
            sd.append(sqrt(var))

            alpha_list.append(alpha)
            y_given_x_samples = conditional_gmm.sample(n_samples=20)
            CI.append(np.quantile(y_given_x_samples, [0.05, 0.95]))
            CI1.append(np.quantile(y_given_x_samples, [0.15, 0.85]))
            CI2.append(np.quantile(y_given_x_samples, [0.35, 0.65]))
            
        sd = np.array(sd).squeeze()
        mu = np.array(mu).squeeze()
        CI = np.array(CI)
        CI1 = np.array(CI1)
        CI2 = np.array(CI2)
        plt.fill_between(x_test_list, *CI.T,color="blue", alpha=0.2)
        plt.fill_between(x_test_list, *CI1.T,color="blue", alpha=0.2)
        plt.fill_between(x_test_list, *CI2.T,color="blue",alpha=0.2)
        plt.plot(x_test_list, alpha_list, label="alpha")
        plt.plot(x_test_list, mu, color = "red")
        plt.fill_between(x_test_list, mu-2*sd,mu+2*sd ,color="orange",alpha=0.4)
        ax.scatter(x, y, s=3, color="black")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.legend()
        plt.show()