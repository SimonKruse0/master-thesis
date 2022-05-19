from src.optimization.bayesopt_solver import BayesOptSolver
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import os

def jsonize_array(array):
    return [a.astype(float) for a in array]


class RegressionTest(BayesOptSolver):
    def __init__(self, reg_model, problem, random_seed = 42) -> None:
        super().__init__(reg_model, problem, 0, n_init_samples= 0)
        self.seednr = random_seed
        self.Y_test = None
        self.X_test = None

    def data_generator(self, n_train, n_test):
        #obs, important that already test on the same test-data
        if self.Y_test is None:
            np.random.seed(self.seednr+1)
            X_test, Y_test = self._init_XY(n_test)
            self.X_test, self.Y_test = X_test, Y_test
        else:
            X_test, Y_test = self.X_test, self.Y_test
        np.random.seed(self.seednr)
        X_train, Y_train = self._init_XY(n_train)
        return X_test, Y_test, X_train, Y_train

    def mean_abs_error(self, y_pred, y_true):
        return np.mean(np.abs(y_pred-y_true))

    def mean_rel_error(self,y_pred, y_true):
        return np.mean(np.abs(y_pred-y_true)/(np.abs(y_true)+0.001))

    def mean_pred_likehood(self, mu_pred,sigma_pred,y_true):
        Z_pred = (y_true-mu_pred)/sigma_pred #std. normal distributed. 
        return np.mean(norm.pdf(Z_pred))

    def train_test_loop(self, n_train_list, n_test, output_path):
        assert isinstance(n_train_list, list)
        assert isinstance(n_train_list[0], int)
        assert isinstance(n_test, int)

        mean_abs_error = []
        mean_rel_error = []
        mean_pred_likelihod = []
        for n_train in n_train_list:
            DATA =  self.data_generator(n_train, n_test)
            X_test = DATA[0] 
            Y_test = DATA[1]
            X_train = DATA[2]
            Y_train = DATA[3]
            #print(Y_test[:10], Y_train) #Checking if random seed works!
            
            self.fit(X_train,Y_train)
            mu_pred,sigma_pred = self.predict(X_test)
            mean_abs_error.append(self.mean_abs_error(mu_pred, Y_test))
            mean_rel_error.append(self.mean_rel_error(mu_pred, Y_test))
            mean_pred_likelihod.append(
                self.mean_pred_likehood(mu_pred,sigma_pred,Y_test))
            print(n_train)
        
        # save data
        data = dict()
        data["n_train_list"]        = n_train_list
        data["n_test"]              = n_test
        data["mean_abs_error"]      = jsonize_array(mean_abs_error)
        data["mean_rel_error"]      = jsonize_array(mean_rel_error)
        data["mean_pred_likelihod"] = jsonize_array(mean_pred_likelihod)

        time = datetime.today().strftime('%Y-%m-%d-%H_%M')
        filename = f"{self.model.name}_{self.problem_name}_dim_{self.problem_dim}_seed_{self.seednr}_time_{time}.json"
        json.dump(data, open(os.path.join(output_path, filename), "w"))

