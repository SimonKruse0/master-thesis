from src.regression_validation.reg_validation import PlotReg1D_mixturemodel
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.SPN_regression2 import SumProductNetworkRegression
import os

from src.benchmarks.custom_test_functions.problems import Test1,Test2, Test3, Test4, Test3b

## main ##
reg_models = [NaiveGMRegression(optimize=False), 
            GaussianProcess_GPy(), 
            BOHAMIANN(num_keep_samples=500), 
            NumpyroNeuralNetwork(num_warmup=200,num_samples=200,
                                num_chains=4, num_keep_samples=200),
            SumProductNetworkRegression(optimize=False)]
reg_models = [NumpyroNeuralNetwork(hidden_units = 100, num_warmup=200,num_samples=200,
                                num_chains=4, num_keep_samples=200)]
#reg_models = [BOHAMIANN(num_warmup=3000, num_samples=10000,num_keep_samples=500)]
for problem_sklearn in [Test4()]:
    for reg_model in reg_models:
        path = f"master-thesis/thesis/Figures/reg_illustrations/{reg_model.name}".replace(" ", "")
        try:
            os.mkdir(path)
        except:
            pass
        for samples in [200]:
            plot_reg = PlotReg1D_mixturemodel(reg_model, problem_sklearn, disp=True)
            plot_reg(samples,show_gauss = True,path= "",show_name=True)
            #plot_reg(samples,show_gauss = True,path= "master-thesis/regression_experiments/1D_reg_plots/GP/",show_name=True)


# for problem_sklearn in [Test1(),Test2(),Test3(),Test4(), Test3b()]:
#     for reg_model in reg_models:
#         path = f"master-thesis/thesis/Figures/reg_illustrations/{reg_model.name}".replace(" ", "")
#         try:
#             os.mkdir(path)
#         except:
#             pass
#         for samples in [10,20,40]:
#             plot_reg = PlotReg1D_mixturemodel(reg_model, problem_sklearn, disp=True)
#             #plot_reg(samples,show_gauss = True,path= "",show_name=True)
#             #plot_reg(samples,show_gauss = True,path= "master-thesis/regression_experiments/1D_reg_plots/GP/",show_name=True)
#             if "SPN" in reg_model.name or "KDE" in reg_model.name:
#                 plot_reg(samples,show_pred = True,show_gauss = False,path= path+"/",show_name=False)
#             else:
#                 plot_reg(samples,show_pred = False,show_gauss = True,path= path+"/",show_name=False)