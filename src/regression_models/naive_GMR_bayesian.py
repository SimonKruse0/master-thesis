import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm
#from src.utils import normalize, denormalize
from math import sqrt
from skopt import BayesSearchCV
from sklearn.base import BaseEstimator
#from scipy.special import logsumexp
import pyro
import pyro.distributions as dist
from pyro.infer.mcmc import MCMC, NUTS
import torch

#logsumexp(a,b)
#torch.log(torch.sum(b*torch.exp(a))) is returned
#However in this implementation, 
# torch.log(b*torch.sum(torch.exp(a))) = torch.log(b)+torch.log(torch.sum(torch.exp(a))) #for equal b. 

class naive_GMR:
    #GMM_regression with NO correlation
    def p_xy(self,x,y):
        # ONly works for 1D x...!
        shape = y.shape
        assert shape == x.shape
        y = y.flatten()[:,None]
        x = x.flatten()[:,None]
        p_xy_all=norm.pdf(x, loc = self.means[:,0], scale = self.x_component_std)*norm.pdf(y, loc=self.means[:,1], scale = self.y_component_std) #(xy.len, components)
        p_xy = torch.sum(p_xy_all, axis=1)/self.n_components #same weight on all components. 
        return p_xy.reshape(shape)
        #loop version!
        # p_xy = 0
        # for i in self.n_components:
        #     p_xy +=self.priors[i]*multivariate_normal(self.means[i], self.variances[i]).pdf(x,y)

    def lp_x_all(self, x):
        assert x.ndim == 2
        #return norm.logpdf(x, loc = self.means[:,0], scale = self.x_component_std)
        d = dist.Normal(self.means[:,None,:-1],self.x_component_std)
        lp_x_all =  d.log_prob(x).sum(axis=2).T
        #lp_x_all = norm.logpdf(x, loc = self.means[:,None,:-1], scale = self.x_component_std).sum(axis=2).T
        return lp_x_all
        #return norm.logpdf(x, loc = self.means[:,:-1].T.flatten(), scale = self.x_component_std)
        #Ok summes sammen!
         
    def lp_x(self,x, lp_x_all= None):
        if lp_x_all is None:
            lp_x_all = self.lp_x_all(x)
        
        #p_x = torch.sum(p_x_all, axis=1)*self.prior #same weight on all components. 
        lp_x = torch.logsumexp(lp_x_all, axis=1)+torch.log(self.prior) #same weight on all components. 
        return lp_x

    def E_predictive(self, X_test): #predictive mean E_{p(y|x)[y]}
        E_y_all =self.means[:,-1]
        lp_x_all = self.lp_x_all(X_test) #shape = (X_test,N_components)
        lp_x = self.lp_x(X_test, lp_x_all=lp_x_all)
        a = lp_x_all-lp_x[:,None]
        E_predictive = torch.matmul(torch.exp(a),E_y_all.T)*self.prior
        #E_predictive = torch.dot(lp_x_all, E_y_all.T)*self.prior/self.p_x(X_test, lp_x_all=lp_x_all)
        return E_predictive

    def E2_predictive(self, X_test): #predictive second moment E_{p(y|x)[y^2]}
        E_y2_all =self.means[:,-1]**2+self.y_component_std**2 #E[y]²+V[y]
        lp_x_all = self.lp_x_all(X_test) #shape = (X_test,N_components)
        lp_x = self.lp_x(X_test, lp_x_all=lp_x_all)
        a = lp_x_all-lp_x[:,None]
        E2_predictive = torch.matmul(torch.exp(a),E_y2_all.T)*self.prior
        #E2_predictive = torch.dot(p_x_all, E_y2_all.T)/self.p_x(X_test, p_x_all=p_x_all)*self.prior
        return E2_predictive

class NaiveGMRegressionBayesian(naive_GMR, BaseEstimator):
    def __init__(self,x_component_std = 5e-2,
                    y_component_std= 5e-2, 
                    prior_settings = {"Ndx": 1,"v_prior":1.2},
                    manipulate_variance = False, 
                    extra_name = ""
                    ):
        #self.model = None
        self.name = f"Naive Gaussian Mixture Regression{extra_name}"
        self.x_component_std = x_component_std
        self.y_component_std = y_component_std
        self.prior_settings = prior_settings
        self.manipulate_variance = manipulate_variance

    def fit(self, X, Y):
        self.N, self.nX = X.shape  
        self.n_components = self.N

        X_, self.x_mean, self.x_std = normalize(X)
        Y_, self.y_mean, self.y_std = normalize(Y)
        self.means = torch.column_stack((X_, Y_))
        self.prior = torch.tensor(1/self.n_components)
        self.params = f"x_k_std = {self.x_component_std}, y_k_std= {self.y_component_std}"

    def model(self, X,Y_obs):
        alpha = 1
        beta = 0.1

        self.x_component_std = pyro.sample("x_component_std",dist.InverseGamma(alpha, beta)) 
        self.y_component_std = pyro.sample("y_component_std",dist.InverseGamma(alpha, beta))

        #with torch.no_grad():
        m_pred_bayes,std_pred_bayes,_ = self.predict(X)

        with pyro.plate("plate"):
            Y = pyro.sample("Y", dist.Normal(m_pred_bayes,std_pred_bayes), obs=Y_obs)
        
    def predict(self, X_test):
        X_test_,*_ = normalize(X_test,self.x_mean, self.x_std)
        Ndx = self.prior_settings["Ndx"]
        sigma_prior = self.prior_settings["v_prior"]

        # likelihood
        m_pred = self.E_predictive(X_test_)
        E2_pred = self.E2_predictive(X_test_)
        v_pred = E2_pred-m_pred**2 #Var[x] = Ex²-(Ex)²
        print(v_pred)
        assert not any(v_pred<0)
        
        # evidens
        p_x = torch.exp(self.lp_x(X_test_))

        # posterior 
        m_pred_bayes = (self.N*p_x*m_pred + Ndx*0)/(self.N*p_x+Ndx)
        E2_pred_bayes = (self.N*p_x*(v_pred+m_pred**2) + Ndx*sigma_prior**2)/(self.N*p_x+Ndx) 
        v_pred_bayes = E2_pred_bayes - m_pred_bayes**2

        assert not any(v_pred_bayes<0)

        std_pred_bayes = torch.sqrt(v_pred_bayes)

        if self.manipulate_variance:
            factor = (1/torch.clip(self.N*p_x, 0.9, torch.inf))
            std_pred_bayes*=factor

        m_pred_bayes_ = denormalize(m_pred_bayes, self.y_mean, self.y_std)
        std_pred_bayes *= self.y_std

        return m_pred_bayes_,std_pred_bayes,p_x #HACK


def normalize(X, mean=None, std=None):
    #zero_mean_unit_var_normalization
    if mean is None:
        mean = torch.mean(X, axis=0)
    if std is None:
        std = torch.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def denormalize(X_normalized, mean, std):
    #zero_mean_unit_var_denormalization
    return X_normalized * std + mean

def obj_fun(x): 
    return 0.5 * (torch.sign(x-0.5) + 1)+torch.sin(100*x)*0.1

if __name__ == "__main__":
    bounds = [0,1]
    #datasize = int(itorchut("Enter number of datapoints: "))
    datasize = 200
    #np.random.seed(20)
    d1 = dist.Uniform(*bounds)
    X_sample = d1.sample((datasize,1))
    X_sample = torch.tensor(X_sample)
    Y_sample = obj_fun(X_sample)
    torch.autograd.set_detect_anomaly(True)

    reg_model = NaiveGMRegressionBayesian()
    reg_model.fit(X_sample,Y_sample)
    kernel = NUTS(reg_model.model)
    m8_1stan_4chains = MCMC(kernel, num_samples=100, num_chains=1)
    m8_1stan_4chains.run(X_sample, Y_sample)
    #precis(m8_1stan_4chains)

    # SPN_regression = SumProductNetworkRegression(
    #                 tracks=5,
    #                 channels = 50, train_epochs= 1000,
    #                 manipulate_variance = True)
    # SPN_regression.fit(X_sample, Y_sample.squeeze())
    
    # fig, ax = plt.subplots()
    # SPN_regression.plot(ax)
    
    # X_test = torch.linspace(0,1,100)[:,None]
    # mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
    # ax.plot(X_test, mean, "--", color="black")
    # ax.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,
    #                             color="black", alpha=0.3, label=r"90\% credible interval") 
    # ax.plot(X_sample, Y_sample, "*")
    # plt.show()