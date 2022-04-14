from gmr import GMM, plot_error_ellipses
from gmr.mvn import MVN
import numpy as np
import matplotlib.pyplot as plt
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

class GMM_bayesian(GMM):
    def condition(self, indices, x,manipulate_test_bounds = None):
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
        if manipulate_test_bounds is not None:
            #OK alt det her er ligegyldigt man kan bare sætte alpha = 0.999?
            test_bound = (-0.2,1.2) #OBS!!
            X = np.vstack([np.repeat(x,100),np.linspace(*test_bound, 100)]).T
            alpha = np.sum(self.to_probability_density(X))*(test_bound[1]-test_bound[0])/100
            #alpha = sigmoid(-0.5+alpha*10000)
            #alpha = 0.999999999999999
            if alpha>1:
                print(alpha)
            #     print("HEj", alpha)
            #     alpha = 1
        else:
            alpha = 1
            #print(weight)
        
        #for k in range(self.n_components+1):
        for k in range(self.n_components):
            if k == self.n_components: #last element, i.e. prior dist. 
                mvn = MVN(mean=np.array([x[0],0]), covariance=np.eye(2)/10)
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
                covariances[k] = covariances[k]*1/np.max([0.001,alpha])
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

        return GMM(n_components=self.n_components, priors=priors, means=means,
                   covariances=covariances, random_state=self.random_state), alpha

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

    def fit(self, X, Y):
        N = X.shape[0]
        XY_train = np.column_stack((X, Y))
        self.model = GMM_bayesian(
        n_components=N, priors=np.repeat(1/N, N), means=XY_train,
        covariances=np.repeat([np.eye(2)/10000], N, axis=0))

    def predict(self,X_test, CI=[0.05,0.95]):
        print(X_test.shape)
        mean, percentiles,std_deviation  = [], [], []
        for x in X_test:
            conditional_gmm, alpha = self.model.condition([0], [x], manipulate_test_bounds = True)
            y_given_x_samples = conditional_gmm.sample(n_samples=2000)
            mean.append(np.mean(y_given_x_samples))
            percentiles.append(np.quantile(y_given_x_samples,CI ))
            std_deviation.append(np.std(y_given_x_samples))
        
        return np.array(mean),np.array(std_deviation).T,np.array(percentiles).T

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1
if __name__ == "__main__":

    N = 40
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
    alpha_list = []
    for x_test in x_test_list:
        conditional_gmm, alpha = gmm.condition([0], [x_test], manipulate_test_bounds = True)
        alpha_list.append(alpha)
        y_given_x_samples = conditional_gmm.sample(n_samples=2000)
        CI.append(np.quantile(y_given_x_samples, [0.05, 0.95]))
        CI1.append(np.quantile(y_given_x_samples, [0.15, 0.85]))
        CI2.append(np.quantile(y_given_x_samples, [0.35, 0.65]))
    CI = np.array(CI)
    CI1 = np.array(CI1)
    CI2 = np.array(CI2)
    plt.fill_between(x_test_list, *CI.T,color="blue", alpha=0.2)
    plt.fill_between(x_test_list, *CI1.T,color="blue", alpha=0.2)
    plt.fill_between(x_test_list, *CI2.T,color="blue",alpha=0.2)
    #plt.plot(x_test_list, alpha_list)
    ax.scatter(x, y, s=3, color="black")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()