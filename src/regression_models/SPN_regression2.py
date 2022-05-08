import torch
if __name__ == "__main__":
    import supr
    from supr.utils import drawnow
else:
    from . import supr
    from .supr.utils import drawnow
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from skopt import BayesSearchCV
from skopt.space import Real
# Marginalization query
from sklearn.base import BaseEstimator, RegressorMixin
#from ..utils import normalize, denormalize

def normalize(X, mean=None, std=None):
    #zero_mean_unit_var_normalization
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)

    X_normalized = (X - mean) / std

    return X_normalized, mean, std


def denormalize(X_normalized, mean, std):
    #zero_mean_unit_var_denormalization
    return X_normalized * std + mean

class SumProductNetworkRegression(BaseEstimator):
    def __init__(self,
                tracks=4, channels=100,
                manipulate_varance = True
                , train_epochs = 500,
                alpha0_x=10,alpha0_y=10, 
                beta0_x = 0.01,beta0_y = 0.01):
        self.epochs = train_epochs
        # Priors for variance of x and y
        self.alpha0_x = alpha0_x#invers gamma
        self.alpha0_y = alpha0_y#invers gamma
        self.beta0_x = beta0_x
        self.beta0_y = beta0_y
        self.name = "SPN regression"
        self.params = f"tracks = {tracks}, channels = {channels}"
        self.tracks = tracks
        self.channels = channels
        # Define prior conditional p(y|x)
        self.prior_settings = {"Ndx": 0.1,"sig_prior":1}
        self.manipulate_varance = manipulate_varance
        
    
    # def _train_all_parmeters(self):
    #     self.model[0].marginalize = torch.zeros(self.xy_variables , dtype=torch.bool)

    def fit(self, X, Y, show_plot=False, optimize=False, opt_n_iter  =2, opt_cv = 3):
        if optimize:
            self._optimize( X, Y, n_iter  =opt_n_iter, cv = opt_cv)
            print("Fitting with optimized hyperparams")
            return
        print("params= ",self.get_params())
        print("X.shape", X.shape)
        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)

        self.N,x_variables = X.shape
        self.xy_variables = x_variables+1
        self.marginalize_y = torch.cat([torch.zeros(x_variables, dtype=torch.bool), torch.tensor([1],dtype=torch.bool)])
        
        print(f"mean variance_x = {self.beta0_x/(self.alpha0_x+1):0.6f}")
        print(f"mean variance_y = {self.beta0_y/(self.alpha0_y+1):0.6f}")
        #Should be optimized!
        alpha_x = torch.tensor(self.alpha0_x)
        alpha_x = alpha_x.repeat_interleave(x_variables)
        alpha_y = torch.tensor([self.alpha0_y])
        alpha = torch.cat([alpha_x,alpha_y])[None, :,None ]
        
        beta_x = torch.tensor(self.beta0_x)
        beta_x = beta_x.repeat_interleave(x_variables)
        beta_y = torch.tensor([self.beta0_y])
        beta =torch.cat([beta_x, beta_y])[None,:,None]

        self.model = supr.Sequential(
            supr.NormalLeaf(
                self.tracks,
                self.xy_variables ,
                self.channels,
                n=self.N,
                mu0=0.0,
                nu0=0,
                alpha0=alpha,
                beta0=beta,
            ),
            supr.Weightsum(self.tracks, self.xy_variables , self.channels),
        )
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        #XY = torch.stack((X, Y[:,None]), dim=1).squeeze()
        XY = torch.hstack((X, Y[:,None]))
        
        #make sure we train on all parameters. 
        #self._train_all_parmeters()
        logp_tmp = 0
        for epoch in range(self.epochs):
            self.model.train()
            logp = self.model(XY).sum()
            #print(f"Log-posterior ∝ {logp:.2f} ")#, end="\r")
            self.model.zero_grad(True)
            logp.backward()
            self.model.eval()  # swap?
            self.model.em_batch_update()
            if abs(logp-logp_tmp) <1e-7:
                print(f"stopped training after {epoch} epochs")
                break
            else:
                logp_tmp = logp
        print(f"-- stopped training -- max iterations = {self.epochs}")

        if show_plot:
            fig, ax = plt.subplots()
            self.plot(ax)
            
            X_test = np.linspace(0,1,100)[:,None]
            mean,std_deviation,Y_CI = self.predict(X_test)
            ax.plot(X_test, mean, "--", color="black")
            ax.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,
                                        color="black", alpha=0.3, label=r"90\% credible interval") 
            ax.plot(X, Y, "*")
            plt.show()

    
    def score(self, X_test, y_test):
        m_pred, sd_pred, _ = self.predict(X_test)
        Z_pred = (y_test-m_pred)/sd_pred #std. normal distributed. 
        score = np.mean(norm.pdf(Z_pred))
        print(f"mean pred likelihood = {score:0.3f}")
        print(" ")
        return score

    def get_params(self, deep=False):
        out = dict()
        out["alpha0_x"] = self.alpha0_x
        out["alpha0_y"] = self.alpha0_y
        out["beta0_x"] = self.beta0_x
        out["beta0_y"] = self.beta0_y
        out["train_epochs"] = self.epochs
        out["tracks"] = self.tracks
        out["channels"] = self.channels 
        return out

    def _optimize(self, X, y, n_iter  =2, cv = 3):
        opt = BayesSearchCV(
            self,
            {
                'alpha0_x': (2e+0, 5e1, 'uniform'), #inversGamma params. 
                'alpha0_y': (2e+0, 5e1, 'uniform'),
                'beta0_x': (1e-4, 1e-1, 'uniform'),
                'beta0_y': (1e-4, 1e-1, 'uniform'),
            },
            n_iter=n_iter,
            cv=cv
        )
        opt.fit(X, y)
        print(" ")
        print(f"best score = {opt.best_score_}")
        print("best params",opt.best_params_)
        #self = opt.best_estimator_.copy()
        self.__dict__.update(opt.best_estimator_.__dict__)
        #self.set_params(**opt.best_estimator_.get_params())
        # self.fit(X,y)
        

    def predict(self,X_test):
        X_test, *_ = normalize(X_test, self.x_mean, self.x_std)
        Ndx = self.prior_settings["Ndx"]
        sig_prior = self.prior_settings["sig_prior"]
        x_grid = torch.tensor(X_test, dtype=torch.double)
        XX_grid = torch.hstack([x_grid, torch.zeros(len(x_grid),1)])
                # Evaluate marginal distribution on x-grid
        log_p_x = self.model(XX_grid, marginalize=self.marginalize_y)
        p_x = torch.exp(log_p_x)
        self.model.zero_grad(True)
        log_p_x.sum().backward()

        with torch.no_grad():
            # Compute normal approximation
            m_pred = (self.N*(self.model.mean())[:, 1]*p_x + Ndx*0)/(self.N*p_x+Ndx)
            v_pred = (self.N*p_x*(self.model.var()[:, 1]+self.model.mean()[:, 1]
                    ** 2) + Ndx*sig_prior)/(self.N*p_x+Ndx) - m_pred**2
            
            # Compute normal approximation
            # m_pred = (self.N*(self.model.mean())[:, 1] + Ndx*0)/(self.N*p_x+Ndx)
            # v_pred = (self.N*(self.model.var()[:, 1]+self.model.mean()[:, 1]
            #         ** 2) + Ndx*sig_prior)/(self.N*p_x+Ndx) - m_pred**2

            #v_pred2 = v_pred.clone()
            if self.manipulate_varance:
                v_pred /= torch.clamp(p_x*50, 1,40)
            #std_pred2 = torch.sqrt(v_pred2)
            std_pred = torch.sqrt(v_pred)
        #transform back to original space #obs validate this!
        m_pred = denormalize(m_pred, self.y_mean, self.y_std)
        std_pred = std_pred*self.y_std

        return np.array(m_pred),np.array(std_pred).T,None
    
    
    def _bayesian_conditional_pdf(self,x_grid,y_grid): #FAILS!!
        x_grid, *_ = normalize(x_grid, self.x_mean, self.x_std)
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)
        x_grid = torch.tensor(x_grid,dtype=torch.float)
        y_grid = torch.tensor(y_grid,dtype=torch.float)
        x_res, y_res = self.x_res, self.y_res
        Ndx = self.prior_settings["Ndx"]
        sig_prior = self.prior_settings["sig_prior"]
        N = self.N

        XY_grid = torch.stack(
            [x.flatten() for x in torch.meshgrid(x_grid, y_grid, indexing="ij")], dim=1
        )
        XX_grid = torch.stack([x_grid, torch.zeros(x_res)]).T
        
        with torch.no_grad():
            log_p_xy = self.model(XY_grid)
            p_xy = torch.exp(log_p_xy).reshape(x_res, y_res)
        
        # Evaluate marginal distribution on x-grid
        log_p_x = self.model(XX_grid, marginalize=self.marginalize_y)
        p_x = torch.exp(log_p_x)
        self.model.zero_grad(True) #Hvad sker der her??
        log_p_x.sum().backward()
        with torch.no_grad():
            p_prior_y = norm(0, sqrt(sig_prior)).pdf(y_grid)
            p_predictive = (N*p_xy + Ndx*p_prior_y[None, :]) / (N*p_x[:, None] + Ndx)
        return p_predictive

    def plot(self, ax, xbounds=(0,1),ybounds=(-2.5,2.5)):
        self.x_res, self.y_res  = 300, 400
        x_grid = torch.linspace(*xbounds, self.x_res, dtype=torch.float)
        y_grid = torch.linspace(*ybounds, self.y_res,dtype=torch.float)

        p_predictive = self._bayesian_conditional_pdf(x_grid,y_grid)

        dx = (x_grid[1] - x_grid[0]) / 2.0
        dy = (y_grid[1] - y_grid[0]) / 2.0
        extent = [
            x_grid[0] - dx,
            x_grid[-1] + dx,
            y_grid[0] - dy,
            y_grid[-1] + dy,
        ]
        ax.imshow(
            torch.log(p_predictive).T,
            extent=extent,
            aspect="auto",
            origin="lower",
            cmap='Blues'
        )  # , vmin=-3, vmax=1)

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

if __name__ == "__main__":
    bounds = [0,1]
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 20
    np.random.seed(20)
    X_sample =  np.random.uniform(*bounds,size = (datasize,1))
    Y_sample = obj_fun(X_sample)

    SPN_regression = SumProductNetworkRegression(
                    tracks=5,
                    channels = 50, train_epochs= 1000,
                    manipulate_varance = True)
    SPN_regression.fit(X_sample, Y_sample.squeeze())
    
    fig, ax = plt.subplots()
    SPN_regression.plot(ax)
    
    X_test = np.linspace(0,1,100)[:,None]
    mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
    ax.plot(X_test, mean, "--", color="black")
    ax.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,
                                color="black", alpha=0.3, label=r"90\% credible interval") 
    ax.plot(X_sample, Y_sample, "*")
    plt.show()
    