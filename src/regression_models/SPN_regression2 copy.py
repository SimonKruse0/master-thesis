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


class SumProductNetworkRegression:
    def __init__(self, tracks=2, channels=50,
                manipulate_varance=False, train_epochs = 500,
                alpha0=10, beta0 = 0.01):
        self.epochs = train_epochs
        # Priors for variance of x and y
        self.alpha0 = torch.tensor(alpha0)#invers gamma
        self.beta0 = torch.tensor(beta0)
        self.name = "SPN regression"
        self.params = f"tracks = {tracks}, channels = {channels}"
        self.tracks = tracks
        self.channels = channels
        # Define prior conditional p(y|x)
        self.prior_settings = {"Ndx": 0.9,"sig_prior":1}
        self.manipulate_varance = manipulate_varance
        
    
    # def _train_all_parmeters(self):
    #     self.model[0].marginalize = torch.zeros(self.xy_variables , dtype=torch.bool)

    def fit(self, X, Y):
        self.N,variables = X.shape
        self.xy_variables = variables+1
        self.marginalize_y = torch.cat([torch.zeros(variables, dtype=torch.bool), torch.tensor([1],dtype=torch.bool)])
        #Should be optimized!
        alpha = self.alpha0.repeat_interleave(self.xy_variables)[None,:,None]
        beta = self.beta0.repeat_interleave(self.xy_variables)[None,:,None]

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
            print(f"Log-posterior ∝ {logp:.2f} ", sep="\r")
            self.model.zero_grad(True)
            logp.backward()
            self.model.eval()  # swap?
            self.model.em_batch_update()
            if abs(logp-logp_tmp) <1e-7:
                print(f"stopped training after {epoch} epochs")
                break
            else:
                logp_tmp = logp
        print("")
    
    def score(self, X_test, y_test):
        m_pred, sd_pred, _ = self.predict(X_test)
        Z_pred = (y_test-m_pred)/sd_pred #std. normal distributed. 
        return np.mean(norm.pdf(Z_pred))

    def get_params(self, deep=False):
        out = dict()
        out["alpha0"] = self.alpha0
        out["beta0"] = self.beta0
        return out

    def optimize():
        pass

    def predict(self,X_test):
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
            #v_pred2 = v_pred.clone()
            if self.manipulate_varance:
                v_pred /= torch.clamp(p_x*50, 1,40)
            #std_pred2 = torch.sqrt(v_pred2)
            std_pred = torch.sqrt(v_pred)

        return np.array(m_pred),np.array(std_pred).T,None
    
    
    def _bayesian_conditional_pdf(self,x_grid,y_grid): #FAILS!!
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

    def plot(self, ax):
        self.x_res, self.y_res  = 300, 400
        x_grid = torch.linspace(0, 1, self.x_res)
        y_grid = torch.linspace(-2.5, 2.5, self.y_res)

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
    SPN_regression.fit(X_sample, Y_sample)
    
    fig, ax = plt.subplots()
    SPN_regression.plot(ax)
    
    X_test = np.linspace(0,1,100)[:,None]
    mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
    ax.plot(X_test, mean, "--", color="black")
    ax.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,
                                color="black", alpha=0.3, label=r"90\% credible interval") 
    ax.plot(X_sample, Y_sample, "*")
    plt.show()
    