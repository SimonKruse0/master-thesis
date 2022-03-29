import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro import sample
from numpyro.infer import MCMC, NUTS,Predictive
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import time
import os

class NumpyroNeuralNetwork:
    def __init__(self, hidden_units = 10, num_warmup=1000, num_samples = 2000, num_chains=1, num_keep_samples = 50):
        self.kernel = None 
        #self.nonlin = lambda x: jnp.tanh(x)
        self.hidden_units = hidden_units
        self.hidden_units_variance = 2
        self.hidden_units_bias_variance = 1 
        self.obs_variance = 0.01
        self.obs_variance_prior = 0 #Obs E[sigma] = 1/lambda
        self.target_accept_prob = 0.6
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.num_keep_samples = num_keep_samples
        #self.keep_every = keep_every
        #self.rng_key = rng_key
        #self.rng_key_predict = rng_key_predict
        self.name = f"numpyro neural network"
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
    
    def fit(self, X, Y, verbose = False): #run_inference
        assert X.ndim == 2
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
        keep_every = self.num_samples//self.num_keep_samples
        for random_variable in samples:
            samples[random_variable] = samples[random_variable][::keep_every]
        self.samples = samples
        self.X = X
        self.y = Y

    def predict(self,X_test,CI=[5.0, 95.0]):
        assert X_test.ndim == 2
        predictive = Predictive(self.model, posterior_samples=self.samples, return_sites = ["Y"])
        y_pred = predictive(self.rng_key_predict, X_test, Y=None)["Y"]
        y_pred = y_pred.squeeze()
        mean_prediction = jnp.mean(y_pred, axis=0)
        percentiles = np.percentile(y_pred,CI , axis=0)
        std_deviation = np.std(y_pred , axis=0)
        return mean_prediction, std_deviation,percentiles

