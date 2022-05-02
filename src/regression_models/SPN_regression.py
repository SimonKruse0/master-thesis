#%% Import libraries
from timeit import repeat
import numpy as np
import matplotlib.pyplot as plt
import torch
if __name__ == "__main__":
    import supr
    from supr.utils import drawnow
else:
    from . import supr
    from .supr.utils import drawnow
from scipy.stats import norm



class SumProductNetworkRegression:
    def __init__(self, tracks=5, channels=20, train_iter = 1000) -> None:
        self.epochs = train_iter
        # Priors for variance of x and y
        self.alpha0 = torch.tensor([[[0.001], [0.001]]])
        self.beta0 = torch.tensor([[[0.001], [0.001]]])
        self.name = "SPN regression"
        self.params = f"tracks = {tracks}, channels = {channels}"
        self.tracks = tracks
        self.channels = channels
    
    def fit(self, X, Y):
        self.N,variables = X.shape
        xy_variables = variables+1
        self.model = supr.Sequential(
            supr.NormalLeaf(
                self.tracks,
                xy_variables ,
                self.channels,
                n=self.N,
                mu0=0.0,
                nu0=0,
                alpha0=self.alpha0,
                beta0=self.beta0,
            ),
            supr.Weightsum(self.tracks, xy_variables , self.channels),
        )
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        XY = torch.stack((X, Y), dim=1).squeeze()
        
        #make sure we train on all parameters. 
        self.model[0].marginalize = torch.zeros(xy_variables , dtype=torch.bool)

        for epoch in range(self.epochs):
            self.model.train()
            loss = self.model(XY).sum()
            print(f"Loss = {loss}", sep="\r")
            loss.backward()
            
            #Tror det her skal være her!
            with torch.no_grad():
                self.model.eval()
                self.model.em_batch_update()
                self.model.zero_grad(True)
        print("")
    
    def predict(self, X_test):
        mean, percentiles,std_deviation  = [], [], []
        n_rejection_samples = 10000

        #%% Grid to evaluate predictive distribution
        x_grid = torch.tensor(X_test, dtype=torch.double).flatten()
        #y_grid = torch.linspace(-2, 2, 300, dtype=torch.float32).flatten()
        y_random_grid = torch.DoubleTensor(n_rejection_samples).uniform_(-1.5, 1.5)
        XY_grid = torch.stack(
            [x.flatten() for x in torch.meshgrid(x_grid, y_random_grid, indexing="ij")], dim=1
        )
        
        with torch.no_grad():
            # Calculate p(x,y)
            self.model[0].marginalize = torch.tensor([False, False])
            p_xy = torch.exp(self.model(XY_grid).reshape(len(x_grid), len(y_random_grid)).T)

            # Calculate p(x)
            self.model[0].marginalize = torch.tensor([False, True]) # Skal de her ikke ombyttes?
            p_x = torch.exp(self.model(XY_grid).reshape(len(x_grid), len(y_random_grid)).T)

            # Calculate p(x|y) with a prior. 
            p_prior = norm(0, 5).pdf(y_random_grid)[:, None]
            p_predictive = ((self.N) * p_xy + p_prior) / ((self.N) * p_x + 1)
            #print(p_predictive)
            #p_predictive = p_xy/p_x
        repeat = 0
        for i in range(x_grid.shape[0]):
            pmax = p_predictive[:,i].max().numpy()
            samples = y_random_grid[np.random.rand(n_rejection_samples)*pmax<p_predictive[:,i].numpy()]
            if len(samples) > 20: 
                print( f"num samples = {len(samples)}")
                mu = samples.mean().clone()
                sigma = samples.std().clone()
                CI = (samples.quantile(0.05).clone(), samples.quantile(0.95).clone())
            else:
                repeat = 1
                print("Not enough samples!")
                # mu = None
                # sigma = None
                # CI = None
            mean.append(mu)
            std_deviation.append(sigma)
            percentiles.append(CI)



        return np.array(mean),np.array(std_deviation).T,np.array(percentiles).T
    
    def plot(self, ax):
        x_grid = torch.linspace(0, 1, 300)
        y_grid = torch.linspace(-1.5, 1.5, 300)
        X_grid = torch.stack(
            [x.flatten() for x in torch.meshgrid(x_grid, y_grid, indexing="ij")], dim=1
        )
        with torch.no_grad():
            p_xy = torch.exp(self.model(X_grid).reshape(len(x_grid), len(y_grid)).T)

            self.model[0].marginalize = torch.tensor([False, True])
            p_x = torch.exp(self.model(X_grid).reshape(len(x_grid), len(y_grid)).T)

            p_prior = norm(0, 5).pdf(y_grid)[:, None]

            p_predictive = ((self.N) * p_xy + p_prior) / ((self.N) * p_x + 1)
            #p_predictive = p_xy/p_x+0.001
            # p_predictive = (p_xy/p_x + p_prior/N)
            # bound = torch.FloatTensor([0.01])
            # p_predictive *= 0.0001/torch.max(bound.expand_as(p_x),p_x)
            
            dx = (x_grid[1] - x_grid[0]) / 2.0
            dy = (y_grid[1] - y_grid[0]) / 2.0
            extent = [
                x_grid[0] - dx,
                x_grid[-1] + dx,
                y_grid[0] - dy,
                y_grid[-1] + dy,
            ]
            ax.imshow(
                torch.log(p_predictive),
                extent=extent,
                aspect="auto",
                origin="lower",
            )  # , vmin=-3, vmax=1)
            #plt.plot(x, y, ".")

def rejection_sampler(p,xbounds = [-1,2],pmax = 2):
    while True:
        x = np.random.rand(1)*(xbounds[1]-xbounds[0])+xbounds[0]
        y = np.random.rand(1)*pmax
        px = p(x)
        if y<=px:
            return x

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

def obj_fun_torch(x):
    return x * 0.5 * (torch.sign(x - 0.5) + 1) + torch.sin(100 * x) * 0.1

RUN_mikkels_test = 0
if __name__ == "__main__":
    bounds = [0,1]
    #datasize = int(input("Enter number of datapoints: "))
    datasize = 50
    np.random.seed(20)
    X_sample =  np.random.uniform(*bounds,size = (datasize,1))
    Y_sample = obj_fun(X_sample)

    SPN_regression = SumProductNetworkRegression(train_iter = 100)
    SPN_regression.fit(X_sample, Y_sample)
    
    fig, ax = plt.subplots()
    SPN_regression.plot(ax)
    
    X_test = np.linspace(0,1,100)[:,None]
    mean,std_deviation,Y_CI = SPN_regression.predict(X_test)
    ax.plot(X_test, mean, "--", color="black")
    ax.fill_between(X_test.squeeze(), Y_CI[0], Y_CI[1],
                                color="black", alpha=0.3, label=r"90\% credible interval") 
    ax.plot(X_sample, Y_sample, "*")
    plt.show()
    


    if RUN_mikkels_test:
        N = 10
        torch.manual_seed(2)
        x = torch.FloatTensor(N).uniform_(0, 1)
        #x = torch.linspace(0, 1, N)
        # y = -1*x + (torch.rand(N)>0.5)*(x>0.5) + torch.randn(N)*0.1
        x[x > 0.5] += 0.25
        x[x < 0.5] -= 0.25



        y = obj_fun_torch(x)
        #%% Dataset
        X = torch.stack((x, y), dim=1)

        #%% Grid to evaluate predictive distribution
        x_grid = torch.linspace(-1, 2, 300)
        y_grid = torch.linspace(-2, 2, 300)
        X_grid = torch.stack(
            [x.flatten() for x in torch.meshgrid(x_grid, y_grid, indexing="ij")], dim=1
        )

        #%% Sum-product network
        tracks = 5
        variables = 2
        channels = 20

        # Priors for variance of x and y
        alpha0 = torch.tensor([[[0.01], [0.01]]])
        beta0 = torch.tensor([[[0.01], [0.01]]])

        model = supr.Sequential(
            supr.NormalLeaf(
                tracks,
                variables,
                channels,
                n=N,
                mu0=0.0,
                nu0=0,
                alpha0=alpha0,
                beta0=beta0,
            ),
            supr.Weightsum(tracks, variables, channels),
        )

        #%% Fit model and display results
        epochs = 10

        for epoch in range(epochs):
            model.train()
            model[0].marginalize = torch.zeros(variables, dtype=torch.bool)
            loss = model(X).sum()
            print(f"Loss = {loss}")
            loss.backward()

            with torch.no_grad():
                model.eval()
                model.em_batch_update()
                model.zero_grad(True)

                p_xy = torch.exp(model(X_grid).reshape(len(x_grid), len(y_grid)).T)

                model[0].marginalize = torch.tensor([False, True])
                p_x = torch.exp(model(X_grid).reshape(len(x_grid), len(y_grid)).T)

                p_prior = norm(0, 0.5).pdf(y_grid)[:, None]

                p_predictive = ((N+10000) * p_xy + p_prior) / ((N+10000) * p_x + 1)
                # p_predictive = (p_xy/p_x + p_prior/N)
                # bound = torch.FloatTensor([0.01])
                # p_predictive *= 0.0001/torch.max(bound.expand_as(p_x),p_x)
                
                plt.figure(1).clf()
                dx = (x_grid[1] - x_grid[0]) / 2.0
                dy = (y_grid[1] - y_grid[0]) / 2.0
                extent = [
                    x_grid[0] - dx,
                    x_grid[-1] + dx,
                    y_grid[0] - dy,
                    y_grid[-1] + dy,
                ]
                plt.imshow(
                    torch.log(p_predictive),
                    extent=extent,
                    aspect="auto",
                    origin="lower",
                )  # , vmin=-3, vmax=1)
                plt.plot(x, y, ".")
                drawnow()
        plt.show()
