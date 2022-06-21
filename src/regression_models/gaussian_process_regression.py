from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor
from src.utils import normalize, denormalize
import GPy

class GaussianProcess_GPy:
    def __init__(self, extra_name="") -> None:
        self.name = f"GP"
        self.latex_architecture = r"gp.kernels.Matern52"

    def fit(self, X,Y):
        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)
        dim = X.shape[1]
        kernel = GPy.kern.Matern52(input_dim=dim, variance=1., lengthscale=1.)
        self.model = GPy.models.GPRegression(X,Y,kernel)
        self.model.optimize_restarts(num_restarts = 20)
        kernel_ls = self.model.parameters[0].lengthscale[0]
        kernel_var = self.model.parameters[0].variance[0]
        noise_var = self.model.parameters[1].variance[0] #noise
        self.params = f"noise = {noise_var:0.2e}, kernel_ls = {kernel_ls:0.2f}, kernel_var = {kernel_var:0.2f}"
    
    def predict(self, X_test):
        X_test, *_ = normalize(X_test,self.x_mean, self.x_std)
        mu, sigma = self.model.predict(X_test)
        #transform back to original space #obs validate this!
        mu = denormalize(mu, self.y_mean, self.y_std)
        sigma = sigma*self.y_std
        return mu.squeeze(), sigma.squeeze(), None


class GaussianProcess_sklearn:
    def __init__(self, extra_name="") -> None:
        kernel = Matern(length_scale=0.1, length_scale_bounds=(1e-02, 2.0), nu=1.5)
        self.model = GaussianProcessRegressor(kernel=kernel,alpha=1e-10, #alpha=1e-10, 
                            optimizer='fmin_l_bfgs_b', 
                            n_restarts_optimizer=200, 
                            normalize_y=False) #Implicite giver dette variance på prior!
    
        self.name = f"Gaussian Process{extra_name} - sklearn"
        self.latex_architecture = r"gp.kernels.Matern52"

    def fit(self, X,Y):
        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)
        self.model.fit(X,Y)
        self.params = f"noise = {self.model.alpha:0.2e}, length_scale = {self.model.kernel_.length_scale:0.2f}"
    
    def predict(self, X_test):
        X_test, *_ = normalize(X_test,self.x_mean, self.x_std)
        mu, sigma = self.model.predict(X_test, return_std=True)
        #transform back to original space #obs validate this!
        mu = denormalize(mu, self.y_mean, self.y_std)
        sigma = sigma*self.y_std
        return mu.squeeze(), sigma, None
    


