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
# pair = Particle_Pair()


# par = {'bkg_func': 'exp'
# #         'trigs' :   (1e10,  0),
# #         'trigo' :   (1e10,  0),
# #         'm1'    :   (1,     0.016) ,#0.016
# #         'm2'    :   (1,     0.008) ,#0.008
# #         'm3'    :   (1,     0.018) ,#0.018
# #         'm4'    :   (1,     0.005) ,#0.005
# #         'ks'    :   (0.563, 0.0240), #0.02402339737495515
# #         'ko'    :   (1.471, 0.0557), #0.05576763648617445
# #         'b0s'   :   (9.9,   0.1) ,#0.1
# #         'b0o'   :   (13.4,  0.7) ,#0.7
#         }

# generative_measurement_model = Transmission_RPI(**par)
# reductive_measurement_model = Transmission_RPI(**par)

# # return [{true_parameters}, {true_parameters}]
# model_correlations = {  
#                         'b0o' : (13.4, 0.7),
                        
#                         }


# import os
# from ATARI.sammy_interface import sammy_functions, sammy_classes


# sammyexe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'

# exp_model = Experimental_Model(channel_widths={"maxE": [250],"chw": [100.0],"dchw": [0.8]})
# # template_creator.make_input_template('samtemplate.inp', Ta_pair, exp_model, rto)
# exp_model.template = os.path.realpath('samtemplate.inp')

# rto = sammy_classes.SammyRunTimeOptions(sammyexe,
#                                         bayes = True,
#                                         get_ECSCM=True,
#                                         )

# Ta_pair = Particle_Pair()
# Ta_pair.add_spin_group(Jpi='3.0', J_ID=1, D=8.79, gn2_avg=46.5, gn2_dof=1, gg2_avg=64.0, gg2_dof=1000)
# pair = Ta_pair
# resonance_ladder = Ta_pair.sample_resonance_ladder()
# from copy import copy
# resonance_ladder_fit = copy(Ta_pair.resonance_ladder)
# resonance_ladder_fit["varyE"] = np.ones(len(resonance_ladder_fit))
# resonance_ladder_fit["varyGg"] = np.ones(len(resonance_ladder_fit))
# resonance_ladder_fit["varyGn1"] = np.ones(len(resonance_ladder_fit))

# sammyINP = sammy_classes.SammyInputData(
#         pair,
#         # resonance_ladder,
#         resonance_ladder_fit,
#         os.path.realpath('samtemplate.inp'),
#         exp_model,
#         energy_grid=exp_model.energy_grid,
# )

# samout = sammy_functions.run_sammy(sammyINP, rto)

# data_unc = np.sqrt(samout.pw['theo_trans'])/10
# data = np.random.default_rng().normal(samout.pw['theo_trans'], data_unc)

# data_df = pd.DataFrame({'E':samout.pw['E'],
#                         'exp': data,
#                         'exp_unc': data_unc})

# print(data_df)
# print(samout.est_df)

#%%
from ATARI.syndat.data_classes import syndatOUT
import os
from ATARI.utils import atario
import h5py
from ATARI.PiTFAll.performance_test import PerformanceTest
import ATARI.PiTFAll.file_handler as pfh
from ATARI.syndat.control import Syndat_Control
import ATARI.utils.hdf5 as h5io


### define paths
sammy_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'
sample_file = "/Users/noahwalton/Documents/GitHub/ATARI/tests/test.hdf5"
perf_test_file = "/Users/noahwalton/Documents/GitHub/ATARI/tests/perf_test.hdf5"

### make a syndat hdf5 samples file
pair = Particle_Pair()
pair.resonance_ladder = pd.DataFrame({"E":[210,220], "Gg": [60,60], "Gn1":[1,4]})
print_out = False
exp_model = Experimental_Model()
syndat_models = [Syndat_Model(generative_experimental_model=exp_model, title="1"), 
                        Syndat_Model(generative_experimental_model=exp_model, title="2"),
                        Syndat_Model(generative_experimental_model=exp_model, title="3")]
df_true_1 = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })
df_true_2 = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })
df_true_3 = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })
save_file = sample_file #os.path.realpath("./test.hdf5")
if os.path.isfile(save_file):
    os.remove(save_file)
syndat = Syndat_Control(pair, syndat_models = syndat_models, model_correlations=[], sampleRES=False)
syndat.sample(num_samples=3, pw_true_list=[df_true_1, df_true_2, df_true_3],  save_samples_to_hdf5=True, hdf5_file=save_file)
for isample in range(3):
    read_par = h5io.read_par(save_file, isample, 'true')
    pw_reduced_df, cov_data = h5io.read_pw_reduced(save_file, isample, "1")
    

### delete perf test file if it exists
if os.path.isfile(perf_test_file):
    os.remove(perf_test_file)

### create from synthetic data samples
pfh.add_synthetic_data_samples(perf_test_file,
                                sample_file,
                                isample_max=3, 
                                energy_range=(200,225))
### could also create from fits


### read out sample 0 to make a "fit"
par_fit = pd.read_hdf(perf_test_file, "sample_0/par_true")
exp_df_dict = {}
for exp in ["1", "2", "3"]:
    exp_df_dict[exp] = pd.read_hdf(perf_test_file, f"sample_0/exp_dat_{exp}/pw_reduced")[["E", "true"]]

### add fit for isample 0
pfh.add_model_fit(perf_test_file,
                   0,
                   "fit_1",
                   par_fit,
                   exp_df_dict)

### overwrite the fit
for exp in ["1"]:#, "2", "3"]:
    print(pd.read_hdf(perf_test_file, f"sample_0/exp_dat_{exp}/pw_reduced"))
    exp_df_dict[exp]['true']=np.ones(len(exp_df_dict[exp]))
pfh.add_model_fit(perf_test_file,
                   0,
                   "fit_1",
                   par_fit,
                   exp_df_dict, overwrite=True)
for exp in ["1"]:#, "2", "3"]:
    print(pd.read_hdf(perf_test_file, f"sample_0/exp_dat_{exp}/pw_reduced"))




# pfh.add_fine_grid_doppler_only(perf_test_file,
#                                5,
#                                (200,225),
#                                sammy_exe,
#                                particle_pair,
#                                model_title = "true"
#                                )

# print(pd.read_hdf(perf_test_file, "sample_0/theo_pw"))

# pfh.add_fine_grid_doppler_only(perf_test_file,
#                                1,
#                                (200,225),
#                                sammy_exe,
#                                particle_pair,
#                                model_title = "fit_1"
#                                )
# print(pd.read_hdf(perf_test_file, "sample_0/theo_pw"))


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
