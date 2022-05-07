from cProfile import label
from gmr import GMM, plot_error_ellipses
from gmr.mvn import MVN
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt


def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class GMM_bayesian(GMM):
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

    #Changing the conditional function in GMM
    def condition(self, indices, x,manipulate_test_bounds = []):
        """Conditional distribution over given indices.
        Returns
        -------
        conditional : GMM
            Conditional GMM distribution p(Y | X=x).
        """
        self._check_initialized()

        indices = np.asarray(indices, dtype=int)
        x = np.asarray(x)

        n_features = self.means.shape[1] - len(indices)
        # means = np.empty((self.n_components+1, n_features))
        # covariances = np.empty((self.n_components+1, n_features, n_features))

        # marginal_norm_factors = np.empty(self.n_components+1)
        # marginal_prior_exponents = np.empty(self.n_components+1)
        means = np.empty((self.n_components, n_features))
        covariances = np.empty((self.n_components, n_features, n_features))

        marginal_norm_factors = np.empty(self.n_components)
        marginal_prior_exponents = np.empty(self.n_components)

        ## Simon Change
        # calculate the margianl p(x) = sum pi_k*p_k(x)
        p_x = 0
        for k in range(self.n_components):
            mvn = MVN(mean=self.means[k], covariance=self.covariances[k],random_state=self.random_state)
            p_x += self.priors[k]*mvn.marginalize(indices).to_probability_density(x)
        
        #for k in range(self.n_components+1):
        for k in range(self.n_components):
            if k == self.n_components: #last element, i.e. prior dist. 
                mvn = MVN(mean=np.array([x[0],0]), covariance=np.eye(x.shape[1]+1)/10)
            else:
                mvn = MVN(mean=self.means[k], covariance=self.covariances[k],
                        random_state=self.random_state)
            conditioned = mvn.condition(indices, x)
            means[k] = conditioned.mean
            covariances[k] = conditioned.covariance

            ## Simon Change
            if manipulate_test_bounds is not None:
                #if covariances[k] < 0.01 : 
                    #print(covariances[k])
                #    covariances[k] = 0.01
                #covariances[k] = covariances[k]*1/np.max([0.01,weight])
                covariances[k] = covariances[k]*1/np.max([0.001,p_x])
                #covariances[k] = covariances[k]*1/weight

            marginal_norm_factors[k], marginal_prior_exponents[k] = \
                mvn.marginalize(indices).to_norm_factor_and_exponents(x) # These values can be used to compute the probability density function
                                                           # of this Gaussian: p(x) = norm_factor * np.exp(exponents).
        
        #priors2 = np.append(self.priors*alpha/len(self.priors), (1-alpha))
        #priors2 = priors2/np.sum(priors2)
        priors2 = self.priors
        
        priors = _safe_probability_density(
            priors2 * marginal_norm_factors,
            marginal_prior_exponents[np.newaxis])[0]

        return GMM_bayesian(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state), p_x

def _safe_probability_density(norm_factors, exponents):
    """Compute numerically safe probability densities of a GMM.

    The probability density of individual Gaussians in a GMM can be computed
    from a formula of the form
    q_k(X=x) = p_k(X=x) / sum_l p_l(X=x)
    where p_k(X=x) = c_k * exp(exponent_k) so that
    q_k(X=x) = c_k * exp(exponent_k) / sum_l c_l * exp(exponent_l)
    Instead of using computing this directly, we implement it in a numerically
    more stable version that works better for very small or large exponents
    that would otherwise lead to NaN or division by 0.
    The following expression is mathematically equal for any constant m:
    q_k(X=x) = c_k * exp(exponent_k - m) / sum_l c_l * exp(exponent_l - m),
    where we set m = max_l exponents_l.

    Parameters
    ----------
    norm_factors : array, shape (n_components,)
        Normalization factors of individual Gaussians

    exponents : array, shape (n_samples, n_components)
        Exponents of each combination of Gaussian and sample

    Returns
    -------
    p : array, shape (n_samples, n_components)
        Probability density of each sample
    """
    m = np.max(exponents, axis=1)[:, np.newaxis]
    p = norm_factors[np.newaxis] * np.exp(exponents - m)
    p /= np.sum(p, axis=1)[:, np.newaxis]
    return p

class GMRegression():
    def __init__(self) -> None:
        self.model = None
        self.name = "Gaussian Mixture Regression"
        self.params = ""

    def fit(self, X, Y):
        N, self.nX = X.shape
        nXY = self.nX+1
        X_, self.x_mean, self.x_std = normalize(X)
        Y_, self.y_mean, self.y_std = normalize(Y)
        XY_train = np.column_stack((X_, Y_))

        self.model = GMM_bayesian(
        n_components=N, priors=np.repeat(1/N, N), means=XY_train,
        covariances=np.repeat([np.eye(nXY)/10000], N, axis=0))

    def predict(self,X_test, CI=[0.05,0.95]):
        #print(X_test.shape)
        X_test_, *_ = normalize(X_test, self.x_mean, self.x_std)
        mean, percentiles,std_deviation  = [], [], []
        test_bounds = 0+np.array([-2,2])*1
        for i,x in enumerate(X_test_):
            if i%10 ==0:
                print(f"Points tested {100*i/X_test_.shape[0]:0.1f}%", end="\r")
            
            conditional_gmm, alpha = self.model.condition(np.arange(self.nX), [x], manipulate_test_bounds = test_bounds)
            mean.append(conditional_gmm.mean())
            var = conditional_gmm.variance()
            var /= np.clip(alpha*20,1,20) 
            std_deviation.append(sqrt(var))
        
        mean += self.y_mean
        std_deviation *= self.y_std
            # y_given_x_samples = conditional_gmm.sample(n_samples=200)
            # y_given_x_samples = denormalize(y_given_x_samples, self.y_mean, self.y_std)
            # mean.append(np.mean(y_given_x_samples))
            # percentiles.append(np.quantile(y_given_x_samples,CI ))
            # std_deviation.append(np.std(y_given_x_samples))
        print("")
        return np.array(mean),np.array(std_deviation).T,None#np.array(percentiles).T

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

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1
if __name__ == "__main__":

    N = 20
    np.random.seed(20) #weird thing happening!
    np.random.seed(1)
    x =  np.random.uniform(0,1,size = (N,1))
    y = obj_fun(x)

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
        conditional_gmm, alpha = gmm.condition([0], [x_test], manipulate_test_bounds = True)

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