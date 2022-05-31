
import numpy as np
import pystan
from src.regression_models.stan_helpers import model_definition
from src.utils import normalize, denormalize
# prepare data for Stan model
import pickle
from hashlib import md5

def StanModel_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

class StanNeuralNetwork:
    def __init__(self, hidden_units = 10, num_warmup=1000, num_samples = 2000, num_chains=1, num_keep_samples = 50, extra_name=""):
        self.name = f"stan neural network{extra_name}"
        self.samples = None
        self.hidden_units = hidden_units
        self.params = f"layers = 3, hidden_units = {hidden_units}, num_warmup = {num_warmup},\
                            num_samples = {num_samples}, num_chains = {num_chains}"
        self.model = StanModel_cache(model_code=model_definition)
    
    def fit(self, X, Y, verbose = False): #run_inference
        assert X.ndim == 2

        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)

        self.X = X
        self.y = Y.squeeze()
        self.N, self.D = X.shape
        
        # self.X = X
        # self.y = Y

    def predict(self,X_test,CI=[5.0, 95.0]): #TODO: Make a numpy model, which should be easier to compute predictions!!?
        assert X_test.ndim == 2
        # Kæmpe hack - fit and prediction right now...!
        N_test= X_test.shape[0]
        X_test, *_ = normalize(X_test, self.x_mean, self.x_std)

        data = {'N': self.N, 'D': self.D, 'num_neurons':self.hidden_units, 'num_hidden_layers':3, 
        'X': self.X, 'y':self.y, 'Ntest': N_test, 'Xtest':X_test}

        fit = self.model.sampling(data=data, iter=1000, chains=1, algorithm="NUTS", seed=42, verbose=True,control=dict(max_treedepth=10))
        y_pred_ = fit["predictions"]

        y_pred_ = y_pred_.squeeze()
        y_pred = denormalize(y_pred_, self.y_mean, self.y_std)

        mean_prediction = np.mean(y_pred, axis=0)
        percentiles = np.percentile(y_pred,CI , axis=0)
        std_deviation = np.std(y_pred , axis=0)
        return mean_prediction, std_deviation,percentiles
