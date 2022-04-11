from timeit import repeat
from tkinter import Y
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

import matplotlib.pyplot as plt
import numpy as np

def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

# np.random.seed(2)
# X_sample =  np.random.uniform(0,1,size = (30,1))
# X_sample = np.vstack([X_sample, np.ones_like(X_sample)])
# X_sample = np.vstack([X_sample, np.zeros_like(X_sample)])
# X_sample = np.vstack([X_sample, np.ones_like(X_sample)])
# X_sample = np.repeat(X_sample, 100, axis=0)
# #X_sample = np.vstack([np.zeros_like(X_sample), np.ones_like(X_sample)])
# Y_sample = obj_fun(X_sample)

# mask = Y_sample.squeeze()<0.5
# mask1 = Y_sample.squeeze()<0
# X0 = X_sample[mask1]
# X1 = X_sample[mask & ~mask1]
# X2 = X_sample[~mask]


# print(X0, X1,X2)
# bgm0 = BayesianGaussianMixture(n_components=4, random_state=42,
# covariance_type = "tied",
# weight_concentration_prior = 100000, weight_concentration_prior_type="dirichlet_distribution").fit(X0)
# bgm1 = BayesianGaussianMixture(n_components=4, random_state=42,
# covariance_type = "tied",
# weight_concentration_prior = 100000, weight_concentration_prior_type="dirichlet_distribution").fit(X1)
# bgm2 = BayesianGaussianMixture(n_components=4, random_state=42,
#                         covariance_type = "tied", 
#                         weight_concentration_prior = 100000, 
#                         weight_concentration_prior_type="dirichlet_distribution").fit(X2)


# x = np.linspace(0,1,100)
# plt.plot(x, np.exp(bgm0.score_samples(x.reshape(-1, 1))/10))
# plt.plot(x, np.exp(bgm1.score_samples(x.reshape(-1, 1))/10))
# plt.plot(x, np.exp(bgm2.score_samples(x.reshape(-1, 1))/10))
# plt.plot(X_sample, Y_sample , '*')
# plt.show()
#print(sum(bgm.weights_), bgm.weights_)

class DiscretizedRegression():
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y

    def _find_levels(self):
        # Find the bins for regression. 
        self.levels = np.quantile(self.y,[0,0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8, 0.9, 1])
        #self.levels = np.array([-0.1,0,0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8, 0.9, 1])

    def train(self):
        self.bgm_list = []
        self._find_levels()
        for i, lvl in enumerate(self.levels):
            #distance for y to lvl. weight X. 
            distance = abs(self.y-lvl)
            #distance[distance>0.1] = 2
            try:
                maxdist = np.abs(self.levels[i+1]-self.levels[i-1])/2
            except:
                maxdist = 0.1
            distance[distance>maxdist] = 2
            eps =0.1# 0.9#0.5 #max is 1
            rep = np.floor(1/(eps+distance)).squeeze().astype("int")
            print(rep)
            X_lvl = np.repeat(self.X,rep*100, axis=0)
            if (X_lvl.shape[0]<4):
                self.bgm_list.append(None)
                continue
            bgm = BayesianGaussianMixture(n_components=4,
                                    covariance_type = "tied",
                                    weight_concentration_prior = 100000, 
                                    weight_concentration_prior_type="dirichlet_distribution").fit(X_lvl)
            self.bgm_list.append(bgm)

    def plot(self):
        ax = plt.subplot()
        plt.plot(self.X, self.y , '*')
        x = np.linspace(0,1,1000)
        pdf_data = np.zeros((len(self.bgm_list),len(x)))
        for i,bgm in enumerate(self.bgm_list):
            if bgm is None:
                continue
            pdf_data[i,:] = np.exp(bgm.score_samples(x.reshape(-1, 1)))
        
        normlization = np.sum(pdf_data, axis=0)
        masking = np.argmax(pdf_data, axis=0)
        pdf_data_none = np.zeros_like(pdf_data)
        for i in range(pdf_data.shape[0]):
            x_none = np.ones_like(x)*np.nan
            mask = masking==i
            pdf_data_none[i,mask] = pdf_data[i,mask]
            x_none[mask] = x[mask]
            plt.fill_between(x, pdf_data_none[i]/normlization+self.levels[i], self.levels[i], color="blue",  alpha = 0.5)
            plt.plot(x_none,np.repeat(self.levels[i], len(x_none)), color="red")
        
        for i in range(pdf_data.shape[0]):
            plt.fill_between(x, pdf_data[i]/normlization+self.levels[i], self.levels[i],color="gray", alpha = 0.3)

        plt.show()

if __name__ == "__main__":
    X_sample =  np.random.uniform(0,1,size = (40,1))
    Y_sample = obj_fun(X_sample)
    DR = DiscretizedRegression(X_sample, Y_sample)
    DR.train()
    DR.plot()