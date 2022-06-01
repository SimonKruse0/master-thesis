from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

class GaussianProcess_sklearn:
    def __init__(self, extra_name="") -> None:
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-05, 100000.0), nu=1.5)
        self.model = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, 
                            optimizer='fmin_l_bfgs_b', 
                            n_restarts_optimizer=200, 
                            normalize_y=True) #Implicite giver dette variance på prior!
    
        self.name = f"Gaussian Process{extra_name} - sklearn"
        self.latex_architecture = r"gp.kernels.Matern52"

    def fit(self, X,Y):
        self.model.fit(X,Y)
        self.params = f"noise = {self.model.alpha:0.2e}, length_scale = {self.model.kernel_.length_scale:0.2f}"
    
    def predict(self, X_test):
        mu, sigma = self.model.predict(X_test, return_std=True)
        return mu.squeeze(), sigma, None
    


