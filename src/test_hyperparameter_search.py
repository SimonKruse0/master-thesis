from skopt import BayesSearchCV
from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from regression_models.SPN_regression2 import SumProductNetworkRegression
from sklearn.datasets import load_iris
import numpy as np
def obj_fun(x): 
    return 0.5 * (np.sign(x-0.5) + 1)+np.sin(100*x)*0.1

bounds = [0,1]
#datasize = int(input("Enter number of datapoints: "))
datasize = 20
np.random.seed(20)
X =  np.random.uniform(*bounds,size = (datasize,1))
Y = obj_fun(X)
y = Y.squeeze()
# X, y = load_iris(return_X_y=True)

#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=.25, random_state=0)

# log-uniform: understand as search over p = exp(x) by varying x
opt = BayesSearchCV(
    SumProductNetworkRegression(),
    {
        'alpha0_x': (2e+0, 5e1, 'uniform'), #inversGamma params. 
        'alpha0_y': (2e+0, 5e1, 'uniform'),
        'beta0_x': (1e-4, 1e-1, 'uniform'),
        'beta0_y': (1e-4, 1e-1, 'uniform'),
        
        # 'degree': (1, 8),  # integer valued parameter
        # 'kernel': ['linear', 'poly', 'rbf'],  # categorical parameter
    },
    n_iter=2,
    cv=3
)

opt.fit(X, y)
print("val. score: %s" % opt.best_score_)
print(opt.best_params_)
opt.best_estimator_.fit(X,y, show_plot = True)
