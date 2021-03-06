import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro import sample
from numpyro.infer import MCMC, NUTS,Predictive
import numpy as np

from src.utils import normalize, denormalize

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import time
import os

class NumpyroNeuralNetwork:
    def __init__(self, hidden_units = 50, num_warmup=500, num_samples = 500, num_chains=4, 
                     obs_variance_prior = 10000,
                     hidden_units_variance = 1, 
                     hidden_units_bias_variance = 1,
                     alpha = 1000,
                     extra_name=""):
        self.kernel = None 
        self.hidden_units = hidden_units
        self.hidden_units_variance = hidden_units_variance
        self.hidden_units_bias_variance = hidden_units_bias_variance
        self.obs_variance_prior = obs_variance_prior #Obs E[sigma] = 1/lambda
        #self.obs_variance_prior = 10 #Obs E[sigma] = 1/lambda
        self.alpha = alpha
        self.target_accept_prob = 0.6
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        if extra_name == "":
            extra_name = f"{num_warmup}-{num_samples}-hu-{hidden_units}-alpha-{alpha}"
        self.name = f"BNN{extra_name}"
        self.latex_architecture = r"$\theta_{\mu} \sim \mathcal{N}(0,{self.hidden_units_variance})$"
        self.samples = None
        self.params = f"n_warmup = {num_warmup},n_samples = {num_samples}, n_chains = {num_chains}"
        text_observation = r"$y \sim \mathcal{N}(f_{\theta}(x),\sigma)$"
        text_prior = r" $\theta_{w}   \sim \mathcal{N}(0,$"+f"{self.hidden_units_variance}"+r"$I_{30})$"
        text_prior_bias = r" $\theta_{b} \sim \mathcal{N}(0,$"+f"{self.hidden_units_bias_variance}"+r"$I_{3})$"
        text_obs_prior = r" $\sigma \sim Exp($"+f"{self.obs_variance_prior}"+r"$)$,"
        text_f = r" $f_{\theta} = NN("+f"{hidden_units},{hidden_units},{hidden_units}"+r")$"
        self.latex_architecture = text_observation + text_prior + text_prior_bias+ text_obs_prior+text_f
        self.text_priors = text_prior + ", "+text_prior_bias


        numpyro.set_platform("cpu")
        numpyro.set_host_device_count(num_chains)
        r = np.random.randint(1000000,size = 1)[0]
        self.rng_key, self.rng_key_predict = random.split(random.PRNGKey(r))

    def model_sample(self,X, Y=None):

        #OBOSOBSOBSOSBOBSOBS NOT THE USED MODEL!!!

        n_output = 1 #output dim
        N, n_input = X.shape
        
        n_hidden = self.hidden_units
        sigma_w = self.hidden_units_variance
        sigma_b = self.hidden_units_bias_variance

        #prior dist
        w1 = sample("w1" ,dist.Normal(jnp.zeros((n_input, n_hidden)), jnp.ones((n_input, n_hidden))*sigma_w),rng_key=self.rng_key)
        w2 = sample("w2", dist.Normal(jnp.zeros((n_hidden, n_hidden)), jnp.ones((n_hidden, n_hidden))*sigma_w),rng_key=self.rng_key)
        w3 = sample("w3", dist.Normal(jnp.zeros((n_hidden, n_output)), jnp.ones((n_hidden, n_output))*sigma_w),rng_key=self.rng_key)

        b1 = sample("b1", dist.Normal(jnp.zeros((1,1)), sigma_b*jnp.ones((1,1))),rng_key=self.rng_key)
        b2 = sample("b2", dist.Normal(jnp.zeros((1,1)), sigma_b*jnp.ones((1,1))),rng_key=self.rng_key)
        b3 = sample("b3", dist.Normal(jnp.zeros((1,1)), sigma_b*jnp.ones((1,1))),rng_key=self.rng_key)

        # we put a prior on the observation noise
        sigma_obs = sample("sigma_obs", dist.InverseGamma(self.alpha, rate=1),rng_key=self.rng_key)
        
        #sigma_obs = sample("sigma", dist.Exponential(jnp.ones((1,1))*self.obs_variance_prior),rng_key=self.rng_key)+0.00001
        
        #sigma_obs = 0.000001
        #likelihood 
        z1 = jnp.tanh(b1+jnp.matmul(X, w1))  # <= first layer of activations
        z2 = jnp.tanh(b2+jnp.matmul(z1, w2))  # <= second layer of activations
        z3 = b3+jnp.matmul(z2, w3)  # <= output of the neural network
        with numpyro.plate("data", N):
            # note we use to_event(1) because each observation has shape (1,)
            Y = numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y,rng_key=self.rng_key)
            #Y = numpyro.sample("Y", z3, obs=Y,rng_key=self.rng_key)
            #Y = numpyro.sample("Y", dist.Delta(z3).to_event(1), obs=Y)
        return Y
        
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
        sigma_obs = sample("sigma_obs", dist.InverseGamma(self.alpha, rate=1))
        
        # if self.obs_variance_prior<1e-9:
        #     sigma_obs = self.obs_variance
        # else:
        #     sigma_obs = sample("sigma", dist.Exponential(jnp.ones((1,1))*self.obs_variance_prior))+0.00001

        #sigma_obs = 0.000001
        #likelihood 
        z1 = jnp.tanh(b1+jnp.matmul(X, w1))  # <= first layer of activations
        z2 = jnp.tanh(b2+jnp.matmul(z1, w2))  # <= second layer of activations
        z3 = b3+jnp.matmul(z2, w3)  # <= output of the neural network
        with numpyro.plate("data", N):
            # note we use to_event(1) because each observation has shape (1,)
            numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)
            #numpyro.sample("Y", dist.Delta(z3).to_event(1), obs=Y)

        return z3, sigma_obs
    
    def fit(self, X, Y, verbose = True, do_normalize = True): #run_inference
        assert X.ndim == 2
        assert Y.ndim == 2

        if do_normalize:
            X, self.x_mean, self.x_std = normalize(X)
            Y, self.y_mean, self.y_std = normalize(Y)

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
        if verbose:
            mcmc.print_summary()
            print("\nMCMC elapsed time:", time.time() - start)
        samples = mcmc.get_samples()
        # for random_variable in samples: #THIS DOESN'T WORK!
        #     samples[random_variable] = samples[random_variable][::self.keep_every]
        self.samples = samples
        # self.X = X
        # self.y = Y

    def predict(self,X_test,CI=[5.0, 95.0], get_y_pred = False): #TODO: Make a numpy model, which should be easier to compute predictions!!?
        assert X_test.ndim == 2

        # if get_y_pred:
        #     print("OBS not normalized")
        #     X_test_ = X_test
        # else:
        X_test_, *_ = normalize(X_test, self.x_mean, self.x_std)

        predictive = Predictive(self.model, posterior_samples=self.samples, return_sites = ["Y"])
        y_pred_ = predictive(self.rng_key_predict, X_test_, Y=None)["Y"]
        y_pred_ = y_pred_.squeeze()
        if get_y_pred:
            return y_pred_
        y_pred = denormalize(y_pred_, self.y_mean, self.y_std)


        mean_prediction = jnp.mean(y_pred, axis=0)
        percentiles = np.percentile(y_pred,CI , axis=0)
        std_deviation = np.std(y_pred , axis=0)
        return mean_prediction, std_deviation,percentiles

