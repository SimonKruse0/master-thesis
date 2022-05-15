import numpy as np
from matplotlib import pyplot as plt
from pystan.external.pymc import plots
from scipy.stats import kde
import sys

model_definition = """
functions {
  vector NNet(matrix X, vector bias_first, vector[] bias_hidden, real bias_output, 
                matrix w_first, matrix[] w_hidden, vector w_output, int num_hidden_layers) {
    int N = rows(X);
    int num_neurons = rows(w_first);
    matrix[N, num_neurons] layer_values[num_hidden_layers];
    vector[N] nnet_output;

    layer_values[1] = tanh(rep_matrix(bias_first',N) + X * w_first');   
    for(i in 2:(num_hidden_layers)) 
      layer_values[i] = tanh(rep_matrix(bias_hidden[i-1]',N) + layer_values[i-1] * w_hidden[i-1]');
    nnet_output = bias_output + layer_values[num_hidden_layers] * w_output;

    return nnet_output;
  }
}
data {
  int<lower=0> N;
  int<lower=0> D;
  int<lower=0> num_neurons;
  int<lower=0> num_hidden_layers;
  matrix[N,D] X;
  real y[N];
  int<lower=0> Ntest;
  matrix[Ntest,D] Xtest;
}
parameters {
  // linear coefficients
  real beta;
  
  // neural network parameters
  vector[num_neurons] bias_first;
  vector[num_neurons] bias_hidden[num_hidden_layers-1];
  real bias_output;
  matrix[num_neurons, D] w_first;
  matrix[num_neurons, num_neurons] w_hidden[num_hidden_layers-1];
  vector[num_neurons] w_output;
}
model{
  vector[N] nnet_output;
  
  //prior over linear coefficients delete
  //beta ~ normal(0, 1); delete
  
  // priors over neural network biases
  bias_first ~ normal(0, 1);
  for (i in 1:(num_hidden_layers-1)) {
    bias_hidden[i] ~ normal(0, 1);
  }
  bias_output ~ normal(0, 1);

  // priors over neural network weights
  to_vector(w_first) ~ normal(0, 1);
  for (i in 1:(num_hidden_layers-1)) {
    to_vector(w_hidden[i]) ~ normal(0, 1);
  }
  w_output ~ normal(0, 1);
  
  // likelihood
  nnet_output = NNet(X, bias_first, bias_hidden, bias_output,
                      w_first, w_hidden, w_output, num_hidden_layers);
  y ~ normal(nnet_output, 0.001);
}
generated quantities{
  vector[Ntest] predictions;
  {
    vector[Ntest] nnet_output;
    nnet_output = NNet(Xtest, bias_first, bias_hidden, bias_output,
                      w_first, w_hidden, w_output, num_hidden_layers);
    for(i in 1:Ntest) 
      predictions[i] = nnet_output[i];
  }
}
"""

if sys.version_info[0] == 3:
    def xrange(i):
        return range(i)

def vb_extract(fit):
    var_names = fit["sampler_param_names"]
    samples = np.array([x for x in fit["sampler_params"]])
    
    samples_dict = {}
    means_dict = {}
    for i in xrange(len(var_names)-1):
        samples_dict[var_names[i]] = samples[i,:]
        means_dict[var_names[i]] = fit["mean_pars"][i]
        
    return samples_dict, means_dict, var_names


def vb_extract_variable(fit, var_name, var_type="real", dims=None):
    if var_type == "real":
        return fit["mean_pars"][fit["sampler_param_names"].index(var_name)]
    elif var_type == "vector":
        vec = []
        for i in xrange(len(fit["sampler_param_names"])):
            #if var_name+"." in fit["sampler_param_names"][i]:
            if var_name in fit["sampler_param_names"][i]:
                vec.append(fit["mean_pars"][i])
        return np.array(vec)
    elif var_type == "matrix":
        if dims == None:
            raise Exception("For matrix variables, you must specify a 'dims' parameter")
        C, D = dims
        mat = []
        for i in xrange(len(fit["sampler_param_names"])):
            #if var_name+"." in fit["sampler_param_names"][i]:
            if var_name in fit["sampler_param_names"][i]:
                mat.append(fit["mean_pars"][i])
        mat = np.array(mat).reshape(C, D, order='F')
        return mat
    else:
        raise Exception("Unknown variable type: %s. Valid types are: real, vector and matrix" % (var_type,))


def vb_plot_variables(fit, var_names):
    samples, means, names = vb_extract(fit)

    if type(var_names) == str:
        var_names = [var_names]
    elif type(var_names) != list:
        raise Exception("Invalid argument type for var_names")

    to_plot = []
    for var in var_names:
        for i in xrange(len(fit["sampler_param_names"])):
            if var in fit["sampler_param_names"][i]: 
                to_plot.append(fit["sampler_param_names"][i])

    for var in to_plot:
        plots.kdeplot_op(plt, samples[var])
    plt.legend(to_plot)
    plt.show()


def plot_kde(samples):
    plots.kdeplot_op(plt, samples)


def posterior_mode(samples):
	density = kde.gaussian_kde(samples)
	xs = np.linspace(np.min(samples),np.max(samples),1000)
	return xs[np.argmax([density.pdf(x) for x in xs])]
	    

def report(fit, prefix=''):
    for param in fit['sampler_param_names']:
        if param.startswith(prefix):
            print(param, "=", vb_extract_variable(fit, var_name=param))
        