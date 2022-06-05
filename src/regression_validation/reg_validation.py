from turtle import shape
from src.optimization.bayesopt_solver import BayesOptSolverBase, BayesOptSolver_coco, BayesOptSolver_sklearn
import numpy as np
from scipy.stats import norm
from datetime import datetime
import json
import os

def jsonize_array(array):
    return [a.astype(float) for a in array]

class RegressionTest_numpycoco(BayesOptSolverBase):
    def __init__(self, reg_model, problem, random_seed = 42) -> None:
        self.seednr = random_seed
        self.Y_test = None
        self.X_test = None
        self.problem_name = problem.name
        self.budget = 0
        self.obj_fun = lambda x: problem.f_obj(x) #Evt. bedre at vectorisere?
        self.model = reg_model
        self.problem_dim = problem.f_obj.d
        self.bounds = [[-5 for _ in range(self.problem_dim)], [5 for _ in range(self.problem_dim)]]

    def data_generator(self, n_train, n_test):
        #obs, important that already test on the same test-data
        if self.Y_test is None:
            np.random.seed(self.seednr+1)
            X_test, Y_test = self._init_XY(n_test, vectorized = True)
            self.X_test, self.Y_test = X_test, Y_test
        else:
            X_test, Y_test = self.X_test, self.Y_test
        np.random.seed(self.seednr)
        X_train, Y_train = self._init_XY(n_train, vectorized = True)
        return X_test, Y_test, X_train, Y_train

    def mean_abs_error(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred-y_true))

    def mean_rel_error(self,y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred-y_true)/(np.abs(y_true)+0.000001))

    def mean_pred_likehood(self, mu_pred,sigma_pred,y_true):
        assert mu_pred.shape == y_true.shape
        assert sigma_pred.shape == y_true.shape
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
            y_test = Y_test.squeeze()
            mean_abs_error.append(self.mean_abs_error(mu_pred, y_test))
            mean_rel_error.append(self.mean_rel_error(mu_pred, y_test))
            mean_pred_likelihod.append(
                self.mean_pred_likehood(mu_pred,sigma_pred,y_test))
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


class RegressionTestBase():
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
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred-y_true))

    def mean_rel_error(self,y_pred, y_true):
        assert y_pred.shape == y_true.shape
        return np.mean(np.abs(y_pred-y_true)/(np.abs(y_true)+0.000001))

    def mean_pred_gaussian(self, mu_pred,sigma_pred,y_true):
        assert mu_pred.shape == y_true.shape
        assert sigma_pred.shape == y_true.shape
        Z_pred = (y_true-mu_pred)/sigma_pred #std. normal distributed. 
        return np.mean(norm.pdf(Z_pred))
    
    def mean_pred_mass(self, x_true,y_true):
        return np.mean(self.predictive_pdf(x_true,y_true))

    def train_test_loop(self, n_train_list, n_test, output_path):
        assert isinstance(n_train_list, list)
        assert isinstance(n_train_list[0], int)
        assert isinstance(n_test, int)

        mean_abs_error = []
        mean_rel_error = []
        mean_pred_likelihod = []
        mean_pred_mass = []
        for n_train in n_train_list:
            DATA =  self.data_generator(n_train, n_test)
            X_test = DATA[0] 
            Y_test = DATA[1]
            X_train = DATA[2]
            Y_train = DATA[3]
            #print(Y_test[:10], Y_train) #Checking if random seed works!
            
            self.fit(X_train,Y_train)
            mu_pred,sigma_pred = self.predict(X_test)
            y_test = Y_test.squeeze()
            mean_abs_error.append(self.mean_abs_error(mu_pred, y_test))
            mean_rel_error.append(self.mean_rel_error(mu_pred, y_test))
            mean_pred_likelihod.append(
                self.mean_pred_gaussian(mu_pred,sigma_pred,y_test))
            mean_pred_mass.append(
                self.mean_pred_mass(X_test,Y_test))
            print(n_train)
        
        # save data
        data = dict()
        data["n_train_list"]        = n_train_list
        data["n_test"]              = n_test
        data["mean_abs_error"]      = jsonize_array(mean_abs_error)
        data["mean_rel_error"]      = jsonize_array(mean_rel_error)
        data["mean_pred_likelihod"] = jsonize_array(mean_pred_likelihod)
        data["mean_pred_mass"] = jsonize_array(mean_pred_mass)

        time = datetime.today().strftime('%Y-%m-%d-%H_%M')
        filename = f"{self.model.name}_{self.problem_name}_dim_{self.problem_dim}_seed_{self.seednr}_time_{time}.json"
        json.dump(data, open(os.path.join(output_path, filename), "w"))

class RegressionTest_sklearn(BayesOptSolver_sklearn, RegressionTestBase):
    def __init__(self, reg_model, problem, random_seed,acquisition="EI", budget=5, n_init_samples=2, disp=False) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        self.seednr = random_seed
        self.Y_test = None
        self.X_test = None

class RegressionTest(BayesOptSolver_coco, RegressionTestBase):
    def __init__(self, reg_model, problem, random_seed, acquisition="EI", budget=5, n_init_samples=0, disp=False) -> None:
        super().__init__(reg_model, problem, acquisition, budget, n_init_samples, disp)
        self.seednr = random_seed
        self.Y_test = None
        self.X_test = None

    # def data_generator(self, n_train, n_test):
    #     #obs, important that already test on the same test-data
    #     if self.Y_test is None:
    #         np.random.seed(self.seednr+1)
    #         X_test, Y_test = self._init_XY(n_test)
    #         self.X_test, self.Y_test = X_test, Y_test
    #     else:
    #         X_test, Y_test = self.X_test, self.Y_test
    #     np.random.seed(self.seednr)
    #     X_train, Y_train = self._init_XY(n_train)
    #     return X_test, Y_test, X_train, Y_train

    # def mean_abs_error(self, y_pred, y_true):
    #     assert y_pred.shape == y_true.shape
    #     return np.mean(np.abs(y_pred-y_true))

    # def mean_rel_error(self,y_pred, y_true):
    #     assert y_pred.shape == y_true.shape
    #     return np.mean(np.abs(y_pred-y_true)/(np.abs(y_true)+0.000001))

    # def mean_pred_likehood(self, mu_pred,sigma_pred,y_true):
    #     assert mu_pred.shape == y_true.shape
    #     assert sigma_pred.shape == y_true.shape
    #     Z_pred = (y_true-mu_pred)/sigma_pred #std. normal distributed. 
    #     return np.mean(norm.pdf(Z_pred))
        

    # def train_test_loop(self, n_train_list, n_test, output_path):
    #     assert isinstance(n_train_list, list)
    #     assert isinstance(n_train_list[0], int)
    #     assert isinstance(n_test, int)

    #     mean_abs_error = []
    #     mean_rel_error = []
    #     mean_pred_likelihod = []
    #     for n_train in n_train_list:
    #         DATA =  self.data_generator(n_train, n_test)
    #         X_test = DATA[0] 
    #         Y_test = DATA[1]
    #         X_train = DATA[2]
    #         Y_train = DATA[3]
    #         #print(Y_test[:10], Y_train) #Checking if random seed works!
            
    #         self.fit(X_train,Y_train)
    #         mu_pred,sigma_pred = self.predict(X_test)
    #         y_test = Y_test.squeeze()
    #         mean_abs_error.append(self.mean_abs_error(mu_pred, y_test))
    #         mean_rel_error.append(self.mean_rel_error(mu_pred, y_test))
    #         mean_pred_likelihod.append(
    #             self.mean_pred_likehood(mu_pred,sigma_pred,y_test))
    #         print(n_train)
        
    #     # save data
    #     data = dict()
    #     data["n_train_list"]        = n_train_list
    #     data["n_test"]              = n_test
    #     data["mean_abs_error"]      = jsonize_array(mean_abs_error)
    #     data["mean_rel_error"]      = jsonize_array(mean_rel_error)
    #     data["mean_pred_likelihod"] = jsonize_array(mean_pred_likelihod)

    #     time = datetime.today().strftime('%Y-%m-%d-%H_%M')
    #     filename = f"{self.model.name}_{self.problem_name}_dim_{self.problem_dim}_seed_{self.seednr}_time_{time}.json"
    #     json.dump(data, open(os.path.join(output_path, filename), "w"))


