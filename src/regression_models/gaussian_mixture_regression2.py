from gmr import GMM, plot_error_ellipses
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from gmr.mvn import MVN
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
from scipy.stats import norm
from src.utils import normalize, denormalize

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class GMM_bayesian(GMM):
    def __init__(self, n_components,N, Ndx, sig_prior, priors=None, means=None, covariances=None, verbose=0, random_state=None):
        super().__init__(n_components, priors, means, covariances, verbose, random_state)
        self.N = N
        self.Ndx = Ndx
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
            Ndx = self.Ndx 
            sig_prior = self.sig_prior

            m_pred = (p_x*n_data*conditional_gmm.mean() + Ndx*0)/(n_data*p_x+Ndx)
            v_pred = (p_x*n_data*(conditional_gmm.variance()+conditional_gmm.mean()**2)+
                        Ndx*sig_prior**2)/(n_data*p_x+Ndx) - m_pred**2

            m_preds.append(m_pred)
            if manipulate_variance:
                v_pred /= np.clip(p_x*50,1,40) 
            std_preds.append(sqrt(v_pred))
        return np.array(m_preds), np.array(std_preds)
    
    def _bayesian_conditional_pdf(self, x_grid,y_grid , manipulate_variance = False):
        Ndx = self.Ndx
        sig_prior = self.sig_prior
        n_data = self.N
        p_predictive = np.zeros((len(x_grid),len(y_grid)))
        p_prior_y = norm(0, sqrt(sig_prior)).pdf(y_grid)
        p_x_list = []
        for i,x in enumerate(x_grid):
            if i%10 ==0:
                print(f"Points evaluated {100*i/x_grid.shape[0]:0.1f}%", end="\r")
            conditional_gmm = self.condition(x)
            p_x = self.marginalize(x) #probability of data at the x. 
            for j,y in enumerate(y_grid):
                p_conditional_gmm = conditional_gmm.to_probability_density(y)
                p_predictive[i,j] = (p_x*n_data*p_conditional_gmm)/(p_x*n_data+Ndx)
                #p_predictive[i,j] = (p_x*n_data*p_conditional_gmm + Ndx*p_prior_y)/(p_x*n_data+Ndx)
            p_predictive[i,:] += Ndx*p_prior_y/(p_x*n_data+Ndx)
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

        return GMM_bayesian(self.n_components, self.N, self.Ndx, self.sig_prior,priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state)

def _safe_probability_density(norm_factors, exponents):
    m = np.max(exponents, axis=1)[:, np.newaxis]
    p = norm_factors[np.newaxis] * np.exp(exponents - m)
    p /= np.sum(p, axis=1)[:, np.newaxis]
    return p

class GMRegression():
    def __init__(self,component_variance = 1e-2, manipulate_variance = False) -> None:
        self.model = None
        self.name = "Gaussian Mixture Regression"
        self.Ndx, self.sig_prior = 1e-2,1.1
        #self.n_components = 
        self.component_variance = component_variance
        self.manipulate_variance = manipulate_variance

    def fit(self, X, Y):
        self.N, self.nX = X.shape
        nXY = self.nX+1
        X_, self.x_mean, self.x_std = normalize(X)
        Y_, self.y_mean, self.y_std = normalize(Y)
        XY_train = np.column_stack((X_, Y_))
        n_components = self.N//5
        gmm_sklearn = BayesianGaussianMixture(n_components=n_components,
                                        covariance_type = "full",
                                        weight_concentration_prior =  1e-9,#1. / n_components, 
                                        mean_precision_prior = 1e-9,#0.001, 
                                        covariance_prior =np.array([[0.01,0],[0,0.01]]),
                                        n_init = 5, init_params="kmeans")
                                        #covariance_prior =np.diag(np.diag(np.cov(XY_train.T)))/1000)
                                        #weight_concentration_prior_type="dirichlet_distribution")
        #gmm_sklearn = GaussianMixture(n_components=n_components, covariance_type="full", max_iter=100)
        gmm_sklearn.fit(XY_train)

        self.model = GMM_bayesian(
        n_components, self.N, self.Ndx, self.sig_prior, priors=gmm_sklearn.weights_, means=gmm_sklearn.means_,
        covariances=gmm_sklearn.covariances_)
        self.params = f"component_variance = {self.component_variance}, manipulate_variance = {self.manipulate_variance}"

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
        x_res, y_res  = 100, 300
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
            Ndx = 1 #self.prior_settings["Ndx"]
            a = self.N*p_x/Ndx
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

    N = 200
    np.random.seed(20) #weird thing happening!
    np.random.seed(1)
    X =  np.random.uniform(0,1,size = (N,1))
    y = obj_fun(X)


    if True:
        GMR = GMRegression()
        GMR.fit(X,y)
        plt.figure(figsize=(10, 5))
        ax = plt.subplot(111)
        x_test_list = np.linspace(-0.1,1.1,500)
        mu, sd, _ = GMR.predict(x_test_list)
        mu =mu.squeeze()
        plt.plot(x_test_list, mu, color = "red")
        plt.fill_between(x_test_list, mu-2*sd,mu+2*sd ,color="orange",alpha=0.4)
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