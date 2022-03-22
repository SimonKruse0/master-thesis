import numpy as np
from scipy.optimize import minimize

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.gridspec as gridspec
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro import sample
from numpyro.infer import MCMC, NUTS,Predictive

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import time
import os


class bayesian_optimization:
    def __init__(self, objectivefunction, regression_model,X_init,Y_init, gridspec = None) -> None:
        self.obj_fun = objectivefunction
        self.model = regression_model
        self._X = X_init #OBS: should X be stored here or in the model?!
        self._Y = Y_init
        self.f_best = np.min(Y_init) # incumbent np.min(Y) or np.min(E[Y]) ??
        self.model.fit(X_init,Y_init) #fit the model
        self.bounds = ((0,1),)
        self.gs = gridspec #for plotting

    # OBS put plot functioner i plot_class!
    def plot_regression_gaussian_approx(self,gs,num_grid_points = 1000):
        assert self.model.X.shape[1] == 1   #Can only plot 1D functions

        Xgrid = np.linspace(*self.bounds[0], num_grid_points)
        Ymu, Ysigma = self.predict(Xgrid)

        ax1 = plt.subplot(gs[0])
        ax1.plot(self._X,self._Y, "kx")  # plot all observed data
        ax1.plot(Xgrid, Ymu, "red", lw=2)  # plot predictive mean
        ax1.fill_between(Xgrid, Ymu - 2*Ysigma, Ymu + 2*Ysigma,
                        color="C0", alpha=0.3, label=r"$E[y]\pm 2  \sqrt{Var[y]}$")  # plot uncertainty intervals
        ax1.set_xlim(*self.bounds[0])
        #ax1.set_ylim(-0.5+np.min(self._Y), 0.5+0.5+np.max(self._Y))
        ax1.set_title(self.model.latex_architecture)
        ax1.legend(loc=2)

    def plot_regression_credible_interval(self,gs,num_grid_points = 1000):
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

    def predict(self,X, gaussian_approx = True):
        if gaussian_approx:
            Y_mu,Y_sigma,_ = self.model.predict4(X)
            return Y_mu,Y_sigma
        else:
            Y_mu,_,Y_CI = self.model.predict4(X)
            return Y_mu,Y_CI

    def expected_improvement(self,X,xi=0.01):
        mu, sigma = self.predict(X)
        imp = -mu - self.f_best - xi
        Z = imp/sigma
        EI = (imp*norm.cdf(Z) + sigma*norm.pdf(Z))
        return EI

    def find_a_candidate(self):
        n_restarts = 1
        x_next = np.nan
        max_EI = 1e-5
        _,nx = self.model.X.shape

        def min_obj(x):
           # x = x[:,None]
            EI = self.expected_improvement(x)
            return -EI
        # Find the best optimum by starting from n_restart different random points.
        for x0 in np.random.uniform(self.bounds[0][0], self.bounds[0][0],
                                    size=(n_restarts, nx)):
            res = minimize(min_obj, x0=x0,bounds=self.bounds, method='Nelder-Mead')        
            if -res.fun > max_EI:
                max_EI = res.fun
                x_next = res.x
        return x_next,max_EI

    def optimization_step(self,x_next):
        y_next = self.obj_fun(x_next)
        self._X = np.append(self._X,x_next)
        self._Y = np.append(self._Y,y_next)
        self.model.fit(self._X,self._Y)
        x_next, max_EI = self.find_a_candidate()
        return x_next,max_EI


class numpyro_neural_network:
    def __init__(self, hidden_units = 10, num_warmup=1000, num_samples = 2000, num_chains=1):
        self.kernel = None 
        #self.nonlin = lambda x: jnp.tanh(x)
        self.hidden_units = hidden_units
        self.hidden_units_variance = 2
        self.hidden_units_bias_variance = 1 
        self.obs_variance = 0.01
        self.obs_variance_prior = 0 #Obs E[sigma] = 1/lambda
        self.target_accept_prob = 0.8
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        #self.rng_key = rng_key
        #self.rng_key_predict = rng_key_predict
        self.name = f"BNN_{self.hidden_units_variance}_{self.hidden_units_bias_variance}"
        self.latex_architecture = r"$\theta_{\mu} \sim \mathcal{N}(0,{self.hidden_units_variance})$"
        self.samples = None
        # self.data = 

        text_observation = r"$y \sim \mathcal{N}(f_{\theta}(x),\sigma),$"
        text_prior = r" $\theta_{w} \sim \mathcal{N}(0,$"+f"{self.hidden_units_variance}"+r"$I_{30}),$"
        text_prior_bias = r" $\theta_{bias} \sim \mathcal{N}(0,$"+f"{self.hidden_units_bias_variance}"+r"$I_{3}),$"
        if self.obs_variance_prior<1e-9:
            text_obs_prior = r" $\sigma = $"+f"{self.obs_variance},"
        else:
            text_obs_prior = r" $\sigma \sim Exp($"+f"{self.obs_variance_prior}"+r"$)$,"
        text_f = r" $f_{\theta} = NN("+f"{hidden_units},{hidden_units},{hidden_units}"+r")$"
        self.latex_architecture = text_observation + text_prior + text_prior_bias+ text_obs_prior+text_f

        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(num_chains)
        self.rng_key, self.rng_key_predict = random.split(random.PRNGKey(0))


    def model(self, X, Y=None):
        n_output = 1 #output dim
        N, n_input = X.shape
        
        n_hidden = self.hidden_units
        sigma_w = self.hidden_units_variance
        sigma_b = self.hidden_units_bias_variance

        #prior dist
        w1 = sample("w1", dist.Normal(jnp.zeros((n_input, n_hidden)), jnp.ones((n_input, n_hidden))*sigma_w))
        w2 = sample("w2", dist.Normal(jnp.zeros((n_hidden, n_hidden)), jnp.ones((n_hidden, n_hidden))*sigma_w))
        w3 = sample("w3", dist.Normal(jnp.zeros((n_hidden, n_output)), jnp.ones((n_hidden, n_output))*sigma_w))

        b1 = sample("b1", dist.Normal(jnp.zeros((1,1)), sigma_b*jnp.ones((1,1))))
        b2 = sample("b2", dist.Normal(jnp.zeros((1,1)), sigma_b*jnp.ones((1,1))))
        b3 = sample("b3", dist.Normal(jnp.zeros((1,1)), sigma_b*jnp.ones((1,1))))

        # we put a prior on the observation noise
        #prec_obs = sample("prec_obs", dist.Gamma(3.0, 1.0))
        #sigma_obs = 0.1 / jnp.sqrt(prec_obs)
        
        if self.obs_variance_prior<1e-9:
            sigma_obs = self.obs_variance
        else:
            sigma_obs = sample("sigma", dist.Exponential(jnp.ones((1,1))*self.obs_variance_prior))+0.00001

        #likelihood 
        z1 = jnp.tanh(b1+jnp.matmul(X, w1))  # <= first layer of activations
        z2 = jnp.tanh(b2+jnp.matmul(z1, w2))  # <= second layer of activations
        z3 = b3+jnp.matmul(z2, w3)  # <= output of the neural network
        with numpyro.plate("data", N):
            # note we use to_event(1) because each observation has shape (1,)
            numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)

        return z3, sigma_obs
    
    def fit(self, X, Y): #run_inference
        try:
            X.shape[1]
        except:
            X = X[:,None]
        if self.samples is None:
            print("-- initial fitting --")
        start = time.time()
        kernel = NUTS(self.model, target_accept_prob=self.target_accept_prob)
        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
        )
        mcmc.run(self.rng_key, X, Y)
        mcmc.print_summary()
        print("\nMCMC elapsed time:", time.time() - start)
        self.samples = mcmc.get_samples()
        self.X = X
        self.y = Y

    def predict4(self,X_test,CI=[5.0, 95.0]):
        try:
            X_test.shape[1]
        except:
            X_test = X_test[:,None]
        predictive = Predictive(self.model, posterior_samples=self.samples, return_sites = ["Y"])
        y_pred = predictive(self.rng_key_predict, X_test, Y=None)["Y"]
        y_pred = y_pred.squeeze()
        mean_prediction = jnp.mean(y_pred, axis=0)
        percentiles = np.percentile(y_pred,CI , axis=0)
        std_deviation = np.std(y_pred , axis=0)
        return mean_prediction, std_deviation,percentiles

    # helper function for prediction
    def predict3(self, X, CI=[5.0, 95.0]): #Denne metode er 
        self.model = handlers.substitute(handlers.seed(self.model, self.rng_key), self.samples)
        # note that Y will be sampled in the model because we pass Y=None here
        model_trace = handlers.trace(self.model).get_trace(X=X, Y=None)
        
        predictions = model_trace["Y"]["value"]
        predictions = predictions[..., 0] #HVORFOR??!

        mean_prediction = jnp.mean(predictions, axis=0)
        percentiles = np.percentile(predictions,CI , axis=0)
        return mean_prediction, percentiles

    def predict2(self,rng_key,sample,X_test):
        self.model = handlers.substitute(handlers.seed(self.model, rng_key), sample)
        # note that Y will be sampled in the model because we pass Y=None here
        model_trace = handlers.trace(self.model).get_trace(X=X_test, Y=None)
        
        return model_trace["Y"]["value"]

    def predict(self,X_test,rng_key_predict,CI=[5.0, 95.0]): #obs check om dette kan implementeres med predictive class
        vmap_args = (
            self.samples,
            random.split(rng_key_predict, self.num_samples * self.num_chains),
        )
        predictions = vmap(
            lambda sample, rng_key: self.predict2(rng_key, sample,X_test)
        )(*vmap_args)
        predictions = predictions[..., 0]
        percentiles = np.percentile(predictions,CI , axis=0)
        return mean_prediction, percentiles


def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

#import pickle

if __name__ == "__main__":

    bounds = np.array([[0,1]])
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 20
    np.random.seed(2)
    X_sample =  np.random.uniform(*bounds[0],size = datasize)
    Y_sample = obj_fun(X_sample)

    plt.figure(figsize=(12, 8))
    outer_gs = gridspec.GridSpec(1, 1)
    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[0])

    BNN = numpyro_neural_network(num_chains = 4, num_warmup= 1000, num_samples=2000)
    BO_BNN = bayesian_optimization(obj_fun, BNN,X_sample,Y_sample, gridspec = gs)

    BO_BNN.plot_regression_gaussian_approx(gs)
    BO_BNN.plot_regression_credible_interval(gs)
    x_next = BO_BNN.plot_expected_improvement(gs)
    plt.show()
    
    # print(x_next)
    # for i in range(4):
    #     x_next,_ = BO_BNN.optimization_step(x_next)
    #     print(x_next)


