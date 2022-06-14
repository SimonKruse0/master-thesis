from src.regression_validation.reg_validation import PlotReg1D_mixturemodel
from src.regression_models.gaussian_process_regression import GaussianProcess_GPy
from src.regression_models.naive_GMR import NaiveGMRegression
from src.regression_models.gaussian_mixture_regression2 import GMRegression
from src.regression_models.numpyro_neural_network import NumpyroNeuralNetwork
from src.regression_models.bohamiann import BOHAMIANN
from src.regression_models.SPN_regression2 import SumProductNetworkRegression
import os

from src.benchmarks.custom_test_functions.problems import Test1,Test2, Test3, Test4, Test3b

## main ##
reg_models = [GMRegression(optimize=True),
            NaiveGMRegression(optimize=True), 
            GaussianProcess_GPy(), 
            BOHAMIANN(num_keep_samples=100), 
            NumpyroNeuralNetwork(hidden_units = 50,num_warmup=300,num_samples=300,
                                num_chains=4,alpha=1000),
            SumProductNetworkRegression(optimize=True)]


# reg_models = [NumpyroNeuralNetwork(hidden_units = 50, num_warmup=200,num_samples=200,
#                                 num_chains=4, alpha=1000)]
#reg_models = [GaussianProcess_GPy()]
#reg_models = [BOHAMIANN(num_warmup=3000, num_samples=10000,num_keep_samples=500)]

# #SINGLE TEST
# for problem_sklearn in [Test2()]:
#     for reg_model in reg_models:
#         path = f"master-thesis/thesis/Figures/reg_illustrations/{reg_model.name}".replace(" ", "")
#         try:
#             os.mkdir(path)
#         except:
#             pass
#         for samples in [200]:
#             plot_reg = PlotReg1D_mixturemodel(reg_model, problem_sklearn, disp=True)
#             plot_reg(samples,show_pred=False,show_gauss = True,path= path+"/",show_name=True)


for samples,problem_sklearn in zip([20,50,100, 200, 200], [Test1(),Test2(),Test3(),Test4(),Test3b]):
    for reg_model in reg_models:
        try:
            name = reg_model.name.replace(" ", "")
            if "BNN" in name:
                name = "BNN"
            path = f"master-thesis/thesis/Figures/reg_illustrations/{name}"
            try:
                os.mkdir(path)
            except:
                pass

            plot_reg = PlotReg1D_mixturemodel(reg_model, problem_sklearn, disp=True)
            #plot_reg(samples,show_gauss = True,path= "",show_name=True)
            #plot_reg(samples,show_gauss = True,path= "master-thesis/regression_experiments/1D_reg_plots/GP/",show_name=True)
            if "SPN" in reg_model.name or "KDE" in reg_model.name:
                plot_reg(samples,grid_points = 100, show_pred = True,show_gauss = False,path= path+"/",show_name=False)
            else:
                plot_reg(samples,show_pred = False,show_gauss = True,path= path+"/",show_name=False)
        except:
            print("OMGOMGOMG")
            pass