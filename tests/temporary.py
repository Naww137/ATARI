#%%
import numpy as np
import pandas as pd
from scipy.stats import normaltest, chi2, ks_1samp, norm
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI

from ATARI.syndat.control import syndatOPT, Syndat_Model
from matplotlib.pyplot import *
from ATARI.syndat.tests import noise_distribution_test, noise_distribution_test2

#%%
pair = Particle_Pair()


par = {'bkg_func': 'exp'
#         'trigs' :   (1e10,  0),
#         'trigo' :   (1e10,  0),
#         'm1'    :   (1,     0.016) ,#0.016
#         'm2'    :   (1,     0.008) ,#0.008
#         'm3'    :   (1,     0.018) ,#0.018
#         'm4'    :   (1,     0.005) ,#0.005
#         'ks'    :   (0.563, 0.0240), #0.02402339737495515
#         'ko'    :   (1.471, 0.0557), #0.05576763648617445
#         'b0s'   :   (9.9,   0.1) ,#0.1
#         'b0o'   :   (13.4,  0.7) ,#0.7
        }

generative_measurement_model = Transmission_RPI(**par)
reductive_measurement_model = Transmission_RPI(**par)

# return [{true_parameters}, {true_parameters}]
model_correlations = {  
                        'b0o' : (13.4, 0.7),
                        
                        }


#%%

# energy_grid = np.sort(np.random.default_rng().uniform(10,3000,10)) #np.linspace(min(energy_range),max(energy_range),10) # energy below 10 has very low counts due to approximate open spectrum
# df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.01,1.0,10)})
# df_true.sort_values('E', ascending=True, inplace=True)

# energy_grid = np.array([20,30,40])
# df_true = pd.DataFrame({'E':energy_grid, 'true':np.array([0.6,0.6,0.6])})
# df_true.sort_values('E', ascending=True, inplace=True)
# options = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 

# for i in range(1):
     
#     # raw_data = self.generate_raw_observables(pw_true, true_model_parameters={})
#     # def generate_raw_observables(self, pw_true, true_model_parameters: dict):
#     if options.sampleTMP:
#         true_model_parameters = generative_measurement_model.sample_true_model_parameters(true_model_parameters)
#     else:
#         true_model_parameters = generative_measurement_model.model_parameters

#     generative_measurement_model.true_model_parameters = true_model_parameters

#     ### generate raw count data from generative reduction model
#     raw_data = generative_measurement_model.generate_raw_data(df_true, 
#                                                                 true_model_parameters, 
#                                                                 options)

    
#     # reduced_data, covariance_data, raw_data = self.reduce_raw_observables(raw_data)
#     # def reduce_raw_observables(self, raw_data):
#     red_data, covariance_data, raw_data = reductive_measurement_model.reduce_raw_data(raw_data, options)
#     # self.covariance_data = self.reductive_measurement_model.covariance_data



#%%

# generative_model = Transmission_RPI()
# reductive_model = Transmission_RPI()

# synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
# exp_model = Experimental_Model(energy_grid=energy_grid, energy_range = [min(energy_grid), max(energy_grid)])
# SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)


# mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynMod, df_true = df_true, ipert=250, print_out=True) #noise_distribution_test(SynMod, print_out=True)
# print("Mean of residuals is not 0",  np.isclose(mean_of_residual, 0, atol=1e-1))
# print("Normalized residuals are standard normal", norm_test_on_residual.pvalue>1e-5)
# print()
# print("Chi2 of data", kstest_on_chi2.pvalue>1e-5, )

# #%%

# generative_model = Capture_Yield_RPI()
# reductive_model = Capture_Yield_RPI()

# synOPT = syndatOPT(smoothTNCS = True) 
# SynY = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
# mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynY, df_true=df_true, ipert=250, print_out=True)
