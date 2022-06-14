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
from src.utils import batch
# Marginalization query
from sklearn.base import BaseEstimator, RegressorMixin
#from ..utils import normalize, denormalize
import copy

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
                tracks=10, channels=30,
                manipulate_variance = False
                , train_epochs = 10000,
                alpha0_x=5,alpha0_y=5, 
                beta0_x = 1,beta0_y = 1, 
                prior_weight = 1,
                sig_prior = 1.1,
                optimize=False, opt_n_iter  =40, opt_cv = 3,
                predictive_score = False):
        self.epochs = train_epochs
        # Priors for variance of x and y
        self.alpha0_x = alpha0_x#invers gamma
        self.alpha0_y = alpha0_y#invers gamma
        self.beta0_x = beta0_x
        self.beta0_y = beta0_y

        self.name = "SPN"

        #self.params = f"manipulate_variance = {manipulate_variance}, optimize = {optimize}, tracks = {tracks}, channels = {channels}"
        self.tracks = tracks
        self.channels = channels
        self.model = None
        # Define prior conditional p(y|x)
        self.sig_prior = sig_prior#
        self.prior_weight = prior_weight
        self.manipulate_variance = manipulate_variance
        self.optimize_hyperparams = optimize #important it is initialised as false
        self.opt_n_iter, self.opt_cv = opt_n_iter, opt_cv
        self.verbose = True
        self.predictive_score = predictive_score
    
    # def _train_all_parmeters(self):
    #     self.model[0].marginalize = torch.zeros(self.xy_variables , dtype=torch.bool)

    def _model(self,X, Y,alpha, beta):
        model = supr.Sequential(
            supr.NormalLeaf(
                self.tracks,
                self.xy_variables ,
                self.channels,
                n=self.N,
                mu0=0.0,
                nu0=0,#beta[0][0][0]/(alpha[0][0][0]-1), #var(x)=1
                alpha0=alpha,
                beta0=beta,
            ),
            supr.Einsum(self.tracks, self.xy_variables , self.channels, 1),
            supr.Weightsum(self.tracks, self.xy_variables, 1)
            #supr.Weightsum(self.tracks, self.xy_variables , self.channels),
        )
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        XY = torch.hstack((X, Y))
        
        #make sure we train on all parameters. 
        #self._train_all_parmeters()
        logp_tmp = 0
        counter = 0
        for epoch in range(self.epochs):
            model.train()
            logp = model(XY).sum()
            if epoch%10==0:
                print(f"Log-posterior ∝ {logp:.2f} ", end="\r")
            model.zero_grad(True)
            logp.backward()
            model.eval()  # swap?
            model.em_batch_update()
            #print(abs(logp-logp_tmp))
            #if abs(logp-logp_tmp) <1e-7:
            if abs(logp-logp_tmp) <1e-2:
                counter += 1
            else:
                counter = 0
                logp_tmp = logp
            if counter > 5:
                print(f"stopped training after {epoch} epochs")
                break

        return logp.detach().numpy(), model

    def fit(self, X, Y, show_plot=False):
        assert Y.ndim == 2
        if self.optimize_hyperparams:
            if X.shape[0] >= 10:
                self.opt_cv =min(30,X.shape[0])
                self._optimize( X, Y)
                print("-- Fitted with optimized hyperparams --")
                return
            else:
                print("-- Fitting with default hyperparams since too little data for CV-- ")
        #print("params= ",self.get_params())
        #print("X.shape", X.shape)
        X, self.x_mean, self.x_std = normalize(X)
        Y, self.y_mean, self.y_std = normalize(Y)

        self.N,x_variables = X.shape
        self.xy_variables = x_variables+1
        self.marginalize_y = torch.cat([torch.zeros(x_variables, dtype=torch.bool), torch.tensor([1],dtype=torch.bool)])
        
        if self.verbose:
            print(f"-- traning -- max iterations = {self.epochs}")
        print(f"var(x-leafs) ~ InvGa({self.alpha0_x:0.2f}, {self.beta0_x:0.2f}), var(y-leafs)~ InvGa({self.alpha0_y:0.2f}, {self.beta0_y:0.2f})")

        alpha_x = torch.tensor(self.alpha0_x)
        alpha_x = alpha_x.repeat_interleave(x_variables)
        alpha_y = torch.tensor([self.alpha0_y])
        alpha = torch.cat([alpha_x,alpha_y])[None, :,None ]
        
        beta_x = torch.tensor(self.beta0_x)
        beta_x = beta_x.repeat_interleave(x_variables)
        beta_y = torch.tensor([self.beta0_y])
        beta =torch.cat([beta_x, beta_y])[None,:,None]

        bestlogp = -np.inf
        n_trainings = 2
        for i in range(n_trainings):
            if self.verbose:
                print(f"training {i+1} out of {n_trainings}",end="\r")
            logp, model = self._model(X, Y,alpha, beta)
            if logp > bestlogp:
                bestlogp = logp
                self.model = model
            if self.verbose:
                print(f"logp = {logp:0.3f}, best logp = {bestlogp:0.3f}")

        if self.verbose:
            print(f"-- stopped training --")
            self.params = f"sig_x = InvGa({self.alpha0_x:0.0f},{self.beta0_x:0.1e})"
            self.params += f", sig_y =InvGa({self.alpha0_y:0.0f},{self.beta0_y:0.1e}), prior_w = {self.prior_weight:0.2e}"
            #self.params += f",\n channels={self.channels}, tracks={self.tracks}"
            # prior_weight = self.prior_weight
            # sig_prior = self.sig_prior
            #self.params += f",\n likelihood:prior weight = p(x){self.N/prior_weight}:1,\nprior_std= {sig_prior}"


    def score(self, X_test, y_test):
        y_test = y_test.squeeze()
        assert y_test.ndim <= 1 
        if self.predictive_score:
            m_pred, sd_pred = self._batched_predict(X_test)
            assert m_pred.ndim == 1
            assert sd_pred.ndim == 1
            score = -np.mean(abs(y_test-m_pred))
            print(f"negative mean pred error = {score:0.3f}")
        else:
            if y_test.ndim == 0:
                y_test = np.array([y_test, y_test])
                X_test = X_test.repeat(2)[:,None]
            p_predictive, p_x = self.predictive_pdf(X_test, y_test[:,None])
            score = np.mean(p_predictive)
            # Z_pred = (y_test-m_pred)/sd_pred #std. normal distributed. 
            # score = np.mean(norm.pdf(Z_pred))
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
        out["manipulate_variance"] = self.manipulate_variance 
        out["prior_weight"] = self.prior_weight
        #out["optimize"] = self.optimize_hyperparams #gets into trouble with the CV code
        return out

    def _optimize(self, X, y):
        #OBS! BayesSearchCV only look at the init params! if they are not decleared in params!
        opt = BayesSearchCV(
            self,
            {
                'alpha0_x': (1e0, 2e2, 'uniform'), #inversGamma params. E[var_x] = beta/(1+alpha)
                'alpha0_y': (1e0, 2e2, 'uniform'),
                #'beta0_x': (1e-2, 7e-1, 'log-uniform'),
                #'beta0_x': (0.1, 2, 'log-uniform'),
                #'beta0_y': (1e-3, 7e-1, 'log-uniform'),
                'prior_weight' : (1e-6, 1., 'uniform'),
            },
            n_iter=self.opt_n_iter,
            cv=self.opt_cv, 
            n_jobs=4
        )
        opt.fit(X, y)
        print(" ")
        print(f"best score = {opt.best_score_}")
        print("best params",opt.best_params_)

        self.__dict__.update(opt.best_estimator_.__dict__)
        #self.set_params(**opt.best_estimator_.get_params())
        # self.fit(X,y) #Not nessesary done by opt.fit
        self.optimize_hyperparams = True #important. 

    def _batched_predict(self,X_test):
        Y_mu_list = []
        Y_sigma_list = []
        for X_batch in batch(X_test, 1000):
            Y_mu,Y_sigma,_ = self.predict(X_batch)
            Y_mu_list.append(Y_mu)
            Y_sigma_list.append(Y_sigma)
        return np.array(Y_mu_list).flatten(),np.array(Y_sigma_list).flatten()

    def predict(self,X_test):
        model = copy.deepcopy(self.model)
        X_test, *_ = normalize(X_test, self.x_mean, self.x_std)
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior
        x_grid = torch.tensor(X_test, dtype=torch.double)
        XX_grid = torch.hstack([x_grid, torch.zeros(len(x_grid),1)])
        
        # Evaluate marginal distribution on x-grid
        log_p_x = model(XX_grid, marginalize=self.marginalize_y)
        p_x = torch.exp(log_p_x)
        model.zero_grad(True)
        log_p_x.sum().backward()

        with torch.no_grad():
            # Compute normal approximation
            mean_prior =0
            m_pred = (self.N*(model.mean())[:, -1]*p_x + prior_weight*mean_prior)/(self.N*p_x+prior_weight)
            v_pred = (self.N*p_x*(model.var()[:, -1]+model.mean()[:, -1]
                    ** 2) + prior_weight*sig_prior**2)/(self.N*p_x+prior_weight) - m_pred**2
            assert not any(v_pred<0) 
            if self.manipulate_variance:
                v_pred /= torch.clamp(p_x*50, 1,40)
            std_pred = torch.sqrt(v_pred)
        #transform back to original space #obs validate this!
        m_pred = denormalize(m_pred, self.y_mean, self.y_std)
        std_pred = std_pred*self.y_std
        return np.array(m_pred),np.array(std_pred).T,None
    
    def predictive_pdf(self,X,Y):
        X,*_ = normalize(X,self.x_mean, self.x_std)
        Y,*_ = normalize(Y,self.y_mean, self.y_std)
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        model = copy.deepcopy(self.model)

        XY = torch.hstack((X, Y))
        XX = torch.hstack((X, torch.zeros(X.shape[0],1)))
        N = self.N
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior
        assert X.ndim ==2
        assert Y.ndim ==2

        with torch.no_grad():
            log_p_xy = model(XY)
            p_xy = torch.exp(log_p_xy)
            #print("p_xy",p_xy)
            log_p_x = model(XX, marginalize=self.marginalize_y)
            p_x = torch.exp(log_p_x)
            normal_sig_prior = torch.distributions.Normal(0,sig_prior)
            p_prior_y= torch.exp(normal_sig_prior.log_prob(Y))
            p_predictive = (N*p_xy + prior_weight*p_prior_y.squeeze()) / (N*p_x + prior_weight)
            return p_predictive.detach().numpy(), p_x.detach().numpy()

    def _bayesian_conditional_pdf(self,x_grid,y_grid): #FAILS!!
        x_grid, *_ = normalize(x_grid, self.x_mean, self.x_std)
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)
        x_grid = torch.tensor(x_grid,dtype=torch.float)
        y_grid = torch.tensor(y_grid,dtype=torch.float)
        x_res, y_res = self.x_res, self.y_res
        prior_weight = self.prior_weight
        sig_prior = self.sig_prior
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
        # self.model.zero_grad(True) #Hvad sker der her??
        # log_p_x.sum().backward()
        mean = None
        with torch.no_grad():
            p_prior_y = norm(0, sig_prior).pdf(y_grid)
            print((p_prior_y[None, :]).shape)
            p_predictive = (N*p_xy + prior_weight*p_prior_y[None, :]) / (N*p_x[:, None] + prior_weight)
            return p_predictive, mean, p_x

    def y_gradient(self,y_grid):
        y_grid, *_ = normalize(y_grid, self.y_mean, self.y_std)
        return np.gradient(y_grid)

    def plot(self, ax, xbounds=(0,1),ybounds=(-2.5,2.5)):
        assert self.xy_variables == 2
        self.x_res, self.y_res  = 500, 800
        x_res, y_res = self.x_res, self.y_res
        x_grid = torch.linspace(*xbounds, self.x_res, dtype=torch.float)
        y_grid = torch.linspace(*ybounds, self.y_res,dtype=torch.float)

        p_predictive, mean, p_x = self._bayesian_conditional_pdf(x_grid,y_grid)



         # Compute 95% highest-posterior region
        hpr = torch.ones((x_res, y_res), dtype=torch.bool)
        for k in range(x_res):
            p_sorted = -np.sort(-(p_predictive[k] * self.y_gradient(y_grid)))
            i = np.searchsorted(np.cumsum(p_sorted), 0.95)
            if i == y_res:
                i = y_res-1
            idx = (p_predictive[k]*self.y_gradient(y_grid)) < p_sorted[i]
            hpr[k, idx] = False

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
            cmap='Blues',
            vmin=-5, vmax=1
        )  # , vmin=-3, vmax=1)
        ax.contour(hpr.T, levels=1, extent=extent )
        #mean = self.predict(x_grid[:,None], only_mean = True)
        if mean is not None:
            ax.plot(x_grid,mean,"--", color="red")
        if p_x is not None:
            ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:green'
            prior_weight = self.prior_weight
            a = self.N*p_x.detach().numpy()/prior_weight
            ax1.plot(x_grid, a/(a+1), color = color)
            #ax1.set_ylabel(r'$\alpha_x$', color=color)
            ax1.set_ylim(0,5)
            ax1.grid(color=color, alpha = 0.2)
            ticks = [0,0.2,0.4,0.6,0.8,1.0]
            ax1.set_yticks(ticks)
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.text(x_grid[len(x_grid)//2],1.1,r"$\alpha(x)$", color=color, size="large")
            
def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

def obj_fun_nd(x): 
    return np.sum(0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1, axis = 1)


if __name__ == "__main__":
    bounds = [0,1]
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 200
    np.random.seed(20)
    xdim = 1
    X_sample =  np.random.uniform(*bounds,size = (datasize,xdim))
    Y_sample = obj_fun_nd(X_sample)[:,None]

    SPN_regression = SumProductNetworkRegression(
                    tracks=1,
                    channels = 30, train_epochs= 1000,
                    optimize=True)
    SPN_regression.fit(X_sample, Y_sample)

    X = X_sample[:5,:]+np.random.randn(5,xdim)*0.001
    Y = Y_sample[:5,:]+np.random.randn(5,1)*0.001
    print("result 1 = ", SPN_regression.predictive_pdf(X, Y)) 
    print("result 2 = ", SPN_regression.predictive_pdf(X, Y)) 
    fig, ax = plt.subplots()
    SPN_regression.plot(ax)
    
    X_test = np.linspace(0,1,100)[:,None]
    mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
    ax.plot(X_test, mean, "--", color="black")
    # ax.fill_between(X_test.squeeze(), mean-2*std_deviation, mean+2*std_deviation,
    #                             color="black", alpha=0.3, label=r"90\% credible interval") 
    ax.plot(X_sample, Y_sample, "*")
    plt.show()
    