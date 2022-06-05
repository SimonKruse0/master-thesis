import torch
import torch.nn as nn
from torch.nn.functional import pad
import math
#from supr.utils import discrete_rand, local_scramble_2d
from .utils import discrete_rand, local_scramble_2d
from typing import List


# Data:
# N x V x D
# └───│──│─ N: Data points
#     └──│─ V: Variables
#        └─ D: Dimensions
#
# Probability:
# N x T x V x C
# └───│───│───│─ N: Data points
#     └───│───│─ T: Tracks
#         └───│─ V: Variables
#             └─ C: Channels

class Supr(nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self):
        pass


class SuprLayer(nn.Module):
    epsilon = 1e-12

    def __init__(self):
        super().__init__()

    def em_batch(self):
        pass

    def em_update(self, *args, **kwargs):
        pass


class Sequential(nn.Sequential):
    def __init__(self, *args: object):
        super().__init__(*args)

    def em_batch_update(self):
        with torch.no_grad():
            for module in self:
                module.em_batch()
                module.em_update()

    def em_batch(self):
        with torch.no_grad():
            for module in self:
                module.em_batch()

    def em_update(self):
        with torch.no_grad():
            for module in self:
                module.em_update()
                
    def sample(self):
        value = []
        for module in reversed(self):
            value = module.sample(*value)
        return value

    def mean(self):
        return self[0].mean()

    def var(self):
        return self[0].var()
    
    def forward(self, value, marginalize=None):
        for module in self:
            if isinstance(module, SuprLeaf):
                value = module(value, marginalize=marginalize)
            else:
                value = module(value)
        return value



class Parallel(SuprLayer):
    def __init__(self, nets: List[SuprLayer]):
        super().__init__()
        self.nets = nets

    def forward(self, x: List[torch.Tensor]):
        return [n(x) for n, x in zip(self.nets, x)]


class ScrambleTracks(SuprLayer):
    """ Scrambles the variables in each track """

    def __init__(self, tracks: int, variables: int):
        super().__init__()
        # Permutation for each track
        perm = torch.stack([torch.randperm(variables) for _ in range(tracks)])
        self.register_buffer('perm', perm)

    def sample(self, track, channel_per_variable):
        return track, torch.scatter(channel_per_variable, 0, self.perm[track], channel_per_variable)

    def forward(self, x):
        return x[:, torch.arange(x.shape[1])[:, None], self.perm, :]


class ScrambleTracks2d(SuprLayer):
    """ Scrambles the variables in each track """

    def __init__(self, tracks: int, variables: int, distance: float, dims: tuple):
        super().__init__()
        # Permutation for each track
        perm = torch.stack([local_scramble_2d(distance, dims) for _ in range(tracks)])
        self.register_buffer('perm', perm)

    def sample(self, track, channel_per_variable):
        return track, torch.scatter(channel_per_variable, 0, self.perm[track], channel_per_variable)

    def forward(self, x):
        return x[:, torch.arange(x.shape[1])[:, None], self.perm, :]


class VariablesProduct(SuprLayer):
    """ Product over all variables """

    def __init(self):
        super().__init__()
        self.variables = None

    def sample(self, track, channel_per_variable):
        return track, torch.full((self.variables,), channel_per_variable[0]).to(channel_per_variable.device)

    def forward(self, x):
        if not self.training:
            self.variables = x.shape[2]
        return torch.sum(x, dim=2, keepdim=True)


class ProductSumLayer(SuprLayer):
    """ Base class for product-sum layers """

    def __init__(self, weight_shape, normalize_dims):
        super().__init__()
        # Parameters
        self.weights = nn.Parameter(torch.rand(*weight_shape))
        self.weights.data /= torch.clamp(self.weights.sum(dim=normalize_dims, keepdim=True), self.epsilon)
        # Normalize dimensions
        self.normalize_dims = normalize_dims
        # EM accumulator
        self.register_buffer('weights_acc', torch.zeros(*weight_shape))

    def em_batch(self):
        self.weights_acc.data += self.weights * self.weights.grad

    def em_update(self, learning_rate: float = 1.):
        weights_grad = torch.clamp(self.weights_acc, self.epsilon)
        weights_grad /= torch.clamp(weights_grad.sum(dim=self.normalize_dims, keepdim=True), self.epsilon)
        if learning_rate < 1.:
            self.weights.data *= 1. - learning_rate
            self.weights.data += learning_rate * weights_grad
        else:
            self.weights.data = weights_grad
        # Reset accumulator
        self.weights_acc.zero_()


class Einsum(ProductSumLayer):
    """ Einsum layer """

    def __init__(self, tracks: int, variables: int, channels: int, channels_out: int = None):
        # Dimensions
        variables_out = math.ceil(variables / 2)
        if channels_out is None:
            channels_out = channels
        # Initialize super
        super().__init__((tracks, variables_out, channels_out, channels, channels), (3, 4))
        # Padding
        self.x1_pad = torch.zeros(variables_out, dtype=torch.bool)
        self.x2_pad = torch.zeros(variables_out, dtype=torch.bool)
        # Zero-pad if necessary
        if variables % 2 == 1:
            # Pad on the right
            self.pad = True
            self.x2_padding = [0, 0, 0, 1]
            self.x2_pad[-1] = True
        else:
            self.pad = False
        # TODO: Implement choice of left, right, or both augmentation. Both returns 2 times the number of tracks

    def sample(self, track: int, channel_per_variable: torch.Tensor):
        r = []
        for v, c in enumerate(channel_per_variable):
            # Probability matrix
            px1 = torch.exp(self.x1[0, track, v, :][:, None])
            px2 = torch.exp(self.x2[0, track, v, :][None, :])
            prob = self.weights[track, v, c] * px1 * px2
            # Sample
            idx = discrete_rand(prob)[0]
            # Remove indices of padding
            idx_valid = idx[[not self.x1_pad[v], not self.x2_pad[v]]]
            # Store on list
            r.append(idx_valid)
        # Concatenate and return indices
        return track, torch.cat(r)

    def forward(self, x: torch.Tensor):
        # Split the input variables in two and apply padding if necessary
        x1 = x[:, :, 0::2, :]
        x2 = x[:, :, 1::2, :]
        if self.pad:
            x2 = pad(x2, self.x2_padding)
        # Store the inputs for use in sampling routine
        if not self.training:
            self.x1, self.x2 = x1, x2
        # Compute maximum
        a1, a2 = [torch.max(x, dim=3, keepdim=True)[0] for x in [x1, x2]]
        # Subtract maximum and compute exponential
        exa1, exa2 = [torch.clamp(torch.exp(x - a), self.epsilon) for x, a in [(x1, a1), (x2, a2)]]
        # Compute the contraction
        y = a1 + a2 + torch.log(torch.einsum('ntva,ntvb,tvcab->ntvc', exa1, exa2, self.weights))
        return y


class Weightsum(ProductSumLayer):
    """ Weightsum layer """

    # Product over all variables and weighted sum over tracks and channels
    def __init__(self, tracks: int, variables: int, channels: int):
        # Initialize super
        super().__init__((tracks, channels), (0, 1))

    def sample(self):
        prob = self.weights * torch.exp(self.x_sum[0] - torch.max(self.x_sum[0]))
        s = discrete_rand(prob)[0]
        return s[0], torch.full((self.variables,), s[1]).to(self.weights.device)

    def forward(self, x: torch.Tensor):
        # Product over variables
        x_sum = torch.sum(x, 2)
        # Store the inputs for use in sampling routine
        if not self.training:
            self.x_sum = x_sum
            self.variables = x.shape[2]
        # Compute maximum
        a = torch.max(torch.max(x_sum, dim=1)[0], dim=1)[0]
        # Subtract maximum and compute exponential
        exa = torch.clamp(torch.exp(x_sum - a[:, None, None]), self.epsilon)
        # Compute the contraction
        y = a + torch.log(torch.einsum('ntc,tc->n', exa, self.weights))
        return y


class TrackSum(ProductSumLayer):
    """ TrackSum layer """

    # Weighted sum over tracks
    def __init__(self, tracks: int, channels: int):
        # Initialize super
        super().__init__((tracks, channels), (0,))

    def sample(self, track: int, channel_per_variable: torch.Tensor):
        prob = self.weights[:, None] * torch.exp(self.x[0] - torch.max(self.x[0], dim=0)[0])
        s = discrete_rand(prob)[0]
        return s[0], channel_per_variable

    def forward(self, x: torch.Tensor):
        # Module is only valid when number of variables is 1
        assert x.shape[2] == 1
        # Store the inputs for use in sampling routine
        if not self.training:
            self.x = x
        # Compute maximum
        a = torch.max(x, dim=1)[0]
        # Subtract maximum and compute exponential
        exa = torch.clamp(torch.exp(x - a[:, None]), self.epsilon)
        # Compute the contraction
        y = a + torch.log(torch.einsum('ntvc,tc->nvc', exa, self.weights))
        # Insert track dimension
        y = y[:, None]
        return y

class SuprLeaf(SuprLayer):
    def __init__(self):
        super().__init__()

class NormalLeaf(SuprLeaf):
    """ NormalLeaf layer """

    def __init__(self, tracks: int, variables: int, channels: int, n: int = 1, mu0: torch.tensor = 0.,
                 nu0: torch.tensor = 0., alpha0: torch.tensor = 0., beta0: torch.tensor = 0.):
        super().__init__()
        # Dimensions
        self.T, self.V, self.C = tracks, variables, channels
        # Number of data points
        self.n = n
        # Prior
        self.mu0, self.nu0, self.alpha0, self.beta0 = mu0, nu0, alpha0, beta0
        # Parametes
        # self.mu = nn.Parameter(torch.randn(self.T, self.V, self.C))
        # self.mu = nn.Parameter(torch.linspace(0, 1, self.C)[None, None, :].repeat((self.T, self.V, 1)))
        self.mu = nn.Parameter(torch.randn(self.T, self.V, self.C))
        self.sig = nn.Parameter(torch.ones(self.T, self.V, self.C) * 0.1)
        # Which variables to marginalized
        self.register_buffer('marginalize', torch.zeros(variables, dtype=torch.bool))
        # Input
        self.register_buffer('x', torch.Tensor())
        # Output
        self.register_buffer('z', torch.Tensor())
        # EM accumulator
        self.register_buffer('z_acc', torch.zeros(self.T, self.V, self.C))
        self.register_buffer('z_x_acc', torch.zeros(self.T, self.V, self.C))
        self.register_buffer('z_x_sq_acc', torch.zeros(self.T, self.V, self.C))

    def em_batch(self):
        self.z_acc.data += torch.clamp(torch.sum(self.z.grad, dim=0), self.epsilon)
        self.z_x_acc.data += torch.sum(self.z.grad * self.x[:, None, :, None], dim=0)
        self.z_x_sq_acc.data += torch.sum(self.z.grad * self.x[:, None, :, None] ** 2, dim=0)

    def em_update(self, learning_rate: float = 1.):
        # Sum of weights
        sum_z = torch.clamp(self.z_acc, self.epsilon)
        # Mean
        # mu_update = (self.nu0 * self.mu0 + self.n * (self.z_x_acc / sum_z)) / (self.nu0 + self.n)
        mu_update = (self.nu0 * self.mu0 + self.z_acc * (self.z_x_acc / sum_z)) / (self.nu0 + self.z_acc)
        self.mu.data *= 1. - learning_rate
        self.mu.data += learning_rate * mu_update
        # Standard deviation
        # sig_update = (self.n * (self.z_x_sq_acc / sum_z - self.mu ** 2) + 2 * self.beta0 + self.nu0 * (
        #           self.mu0 - self.mu) ** 2) / (self.n + 2 * self.alpha0 + 3)
        sig_update = (self.z_x_sq_acc - self.z_acc * self.mu ** 2 + 2 * self.beta0 + self.nu0 * (
                  self.mu0 - self.mu) ** 2) / (self.z_acc + 2 * self.alpha0 + 3)
        self.sig.data *= 1 - learning_rate
        self.sig.data += learning_rate * sig_update
        # Reset accumulators
        self.z_acc.zero_()
        self.z_x_acc.zero_()
        self.z_x_sq_acc.zero_()

    def sample(self, track: int, channel_per_variable: torch.Tensor):
        variables_marginalize = torch.sum(self.marginalize).int()
        mu_marginalize = self.mu[track, self.marginalize, channel_per_variable[self.marginalize]]
        sig_marginalize = self.sig[track, self.marginalize, channel_per_variable[self.marginalize]]
        r = torch.empty_like(self.x[0])
        r[self.marginalize] = mu_marginalize + torch.randn(variables_marginalize).to(self.x.device) * torch.sqrt(
            torch.clamp(sig_marginalize, self.epsilon))
        r[~self.marginalize] = self.x[0][~self.marginalize]
        return r
    
    def mean(self):
        return (torch.clamp(self.z.grad, self.epsilon) * self.mu).sum([1, 3])
    
    def var(self):
        return (torch.clamp(self.z.grad, self.epsilon) * (self.mu**2 + self.sig)).sum([1, 3]) - self.mean()**2

    def forward(self, x: torch.Tensor, marginalize=None):
        # Get shape
        batch_size = x.shape[0]
        # Marginalize variables
        if marginalize is not None:
            self.marginalize = marginalize
        # Store the data
        self.x = x
        # Compute the probability
        self.z = torch.zeros(batch_size, self.T, self.V, self.C, requires_grad=True, device=x.device)
        # Get non-marginalized parameters and data
        mu_valid = self.mu[None, :, ~self.marginalize, :]
        sig_valid = self.sig[None, :, ~self.marginalize, :]
        x_valid = self.x[:, None, ~self.marginalize, None]
        # Evaluate log probability
        self.z.data[:, :, ~self.marginalize, :] = \
            torch.distributions.Normal(mu_valid, torch.sqrt(torch.clamp(sig_valid, self.epsilon))).log_prob(
                x_valid).float()
        return self.z


class BernoulliLeaf(SuprLeaf):
    """ BernoulliLeaf layer """

    def __init__(self, tracks: int, variables: int, channels: int, n: int = 1,
                 alpha0: float = 1., beta0: float = 1.):
        super().__init__()
        # Dimensions
        self.T, self.V, self.C = tracks, variables, channels
        # Number of data points
        self.n = n
        # Prior
        self.alpha0, self.beta0 = alpha0, beta0
        # Parametes
        self.p = nn.Parameter(torch.rand(self.T, self.V, self.C))
        # Which variables to marginalized
        self.register_buffer('marginalize', torch.zeros(variables, dtype=torch.bool))
        # Input
        self.register_buffer('x', torch.Tensor())
        # Output
        self.register_buffer('z', torch.Tensor())
        # EM accumulator
        self.register_buffer('z_acc', torch.zeros(self.T, self.V, self.C))
        self.register_buffer('z_x_acc', torch.zeros(self.T, self.V, self.C))

    def em_batch(self):
        self.z_acc.data += torch.sum(self.z.grad, dim=0)
        self.z_x_acc.data += torch.sum(self.z.grad * self.x[:, None, :, None], dim=0)

    def em_update(self, learning_rate: float = 1.):
        # Probability
        sum_z = torch.clamp(self.z_acc, self.epsilon)
        # p_update = (self.n * self.z_x_acc / sum_z + self.alpha0 - 1) / (self.n + self.alpha0 + self.beta0 - 2)
        p_update = (self.z_x_acc + self.alpha0 - 1) / (self.z_acc + self.alpha0 + self.beta0 - 2)
        self.p.data *= 1. - learning_rate
        self.p.data += learning_rate * p_update
        # Reset accumulators
        self.z_acc.zero_()
        self.z_x_acc.zero_()

    def sample(self, track: int, channel_per_variable: torch.Tensor):
        variables_marginalize = torch.sum(self.marginalize).int()
        p_marginalize = self.p[track, self.marginalize, channel_per_variable[self.marginalize]]
        r = torch.empty_like(self.x[0])
        r[self.marginalize] = (torch.rand(variables_marginalize).to(self.x.device) < p_marginalize).float()
        r[~self.marginalize] = self.x[0][~self.marginalize]
        return r
    
    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size = x.shape[0]
        # Store the data
        self.x = x
        # Compute the probability
        self.z = torch.zeros(batch_size, self.T, self.V, self.C, requires_grad=True, device=x.device)
        # Get non-marginalized parameters and data
        p_valid = self.p[None, :, ~self.marginalize, :]
        x_valid = self.x[:, None, ~self.marginalize, None]
        # Evaluate log probability
        self.z.data[:, :, ~self.marginalize, :] = \
            torch.distributions.Bernoulli(probs=p_valid).log_prob(x_valid).float()
        return self.z


class CategoricalLeaf(SuprLeaf):
    """ CategoricalLeaf layer """

    def __init__(self, tracks: int, variables: int, channels: int, dimensions: int, n: int = 1,
                 alpha0: float = 1.):
        super().__init__()
        # Dimensions
        self.T, self.V, self.C, self.D = tracks, variables, channels, dimensions
        # Number of data points
        self.n = n
        # Prior
        self.alpha0 = alpha0
        # Parametes
        self.p = nn.Parameter(torch.rand(self.T, self.V, self.C, self.D))
        # Which variables to marginalized
        self.register_buffer('marginalize', torch.zeros(variables, dtype=torch.bool))
        # Input
        self.register_buffer('x', torch.Tensor())
        # Output
        self.register_buffer('z', torch.Tensor())
        # EM accumulator
        self.register_buffer('z_acc', torch.zeros(self.T, self.V, self.C))
        self.register_buffer('z_x_acc', torch.zeros(self.T, self.V, self.C, self.D))

    def em_batch(self):
        self.z_acc.data += torch.sum(self.z.grad, dim=0)
        x_onehot = torch.eye(self.D, dtype=bool)[self.x]
        self.z_x_acc.data += torch.sum(self.z.grad[:, :, :, :, None] * x_onehot[:, None, :, None, :], dim=0)

    def em_update(self, learning_rate: float = 1.):
        # Probability
        sum_z = torch.clamp(self.z_acc, self.epsilon)
        # p_update = (self.n * self.z_x_acc / sum_z[:, :, :, None] + self.alpha0 - 1) / (
                    # self.n + self.D * (self.alpha0 - 1))
        p_update = (self.z_x_acc + self.alpha0 - 1) / (
                    self.z_acc[:,:,:,None] + self.D * (self.alpha0 - 1))
        self.p.data *= 1. - learning_rate
        self.p.data += learning_rate * p_update
        # Reset accumulators
        self.z_acc.zero_()
        self.z_x_acc.zero_()

    def sample(self, track: int, channel_per_variable: torch.Tensor):
        p_marginalize = self.p[track, self.marginalize, channel_per_variable[self.marginalize], :]
        r = torch.empty_like(self.x[0])
        r_sample = torch.distributions.Categorical(probs=p_marginalize).sample()
        r[self.marginalize] = r_sample
        r[~self.marginalize] = self.x[0][~self.marginalize]
        return r

    def forward(self, x: torch.Tensor):
        # Get shape
        batch_size = x.shape[0]
        # Store the data
        self.x = x
        # Compute the probability
        self.z = torch.zeros(batch_size, self.T, self.V, self.C, requires_grad=True, device=x.device)
        # Get non-marginalized parameters and data
        p_valid = self.p[None, :, ~self.marginalize, :, :]
        x_valid = self.x[:, None, ~self.marginalize, None]
        # Evaluate log probability
        self.z.data[:, :, ~self.marginalize, :] = \
            torch.distributions.Categorical(probs=p_valid).log_prob(x_valid).float()
        return self.z
