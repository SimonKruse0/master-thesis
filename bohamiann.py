try:
    from pybnn import bohamiann
except ImportError:
    raise ImportError(
        """
        This module is missing required dependencies. Try running
        pip install git+https://github.com/automl/pybnn.git
        Refer to https://github.com/automl/pybnn for further information.
    """
    )
import torch
import torch.nn as nn
import numpy as np

def get_default_network(input_dimensionality: int) -> torch.nn.Module:
    class AppendLayer(nn.Module):
        def __init__(self, bias=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if bias:
                self.bias = nn.Parameter(torch.Tensor(1, 1))
            else:
                self.register_parameter("bias", None)

        def forward(self, x):
            return torch.cat((x, self.bias * torch.ones_like(x)), dim=1)

    def init_weights(module):
        if type(module) == AppendLayer:
            nn.init.constant_(module.bias, val=np.log(1e-3))
        elif type(module) == nn.Linear:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="linear")
            nn.init.constant_(module.bias, val=0.0)

    return nn.Sequential(
        nn.Linear(input_dimensionality, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1), AppendLayer()
    ).apply(init_weights)


class BOHAMIANN:
    def __init__(self,num_warmup=1000, num_samples = 2000, keep_every=50, lr=1e-2) -> None:
        self.model = bohamiann.Bohamiann(get_network=get_default_network)
        #assert num_steps>num_burnin #since num_burnin
        self.num_steps = num_samples+num_warmup
        self.num_burnin = num_warmup
        self.keep_every=keep_every 
        self.lr=lr
        self.name = "BOHAMIANN"
        self.latex_architecture = "nn.Linear(input_dimensionality, 50), nn.Tanh(), nn.Linear(50, 50), nn.Tanh(), nn.Linear(50, 1), AppendLayer()"
    
    def fit(self, X, Y):
        self.model.train(
            X, Y, num_steps=self.num_steps, num_burn_in_steps=self.num_burnin, 
            lr= self.lr, keep_every=self.keep_every, verbose=True
        )

    def predict(self,X_test):
        assert X_test.ndim == 2
        m, v = self.model.predict(X_test)
        return m, np.sqrt(v), None
