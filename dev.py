# %%
import numpy as np
import syndat
import pandas as pd
import os
from matplotlib.pyplot import *
import PiTFAll as pf
import h5py
# h5py.enable_ipython_completer()


# %%
# %matplotlib widget

# %%
# Peformance Test for Fitting ALgorithm (PiTFALl)

# %%
# Initialize particle pair and quantup spin groups of interest

ac = 0.81271    # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity [sic: perhaps, angular momentum?]
l_max = 1       # highest order l-wave to consider

spin_groups = [ (3.0,1,0) ] # (4.0,1,0) 
average_parameters = pd.DataFrame({ 'dE'    :   {'3.0':8.79, '4.0':4.99},
                                    'Gg'    :   {'3.0':46.4, '4.0':35.5},
                                    'gn2'    :   {'3.0':64.0, '4.0':64.0}  })

Ta_pair = syndat.particle_pair( ac, M, m, I, i, l_max,
                                spin_groups=spin_groups, average_parameters=average_parameters,
                                input_options={})


# %%

# initialize experimental setup
E_min_max = [100, 120]
input_options = { 'Add Noise': True,
            'Sample TURP':True,
            'Sample TOCS':True, 
            'Calculate Covariance': False,
            'Compression Points':[],
            'Grouping Factors':None}

experiment_parameters = {'bw': {'val':0.3,    'unc'   :   0}}

exp = syndat.experiment(E_min_max, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)
len(exp.energy_domain)

# %%
# run a performance test with the PiTFAll module

case_file = './perf_test_baron'
# case_file = './perf_test_baron.hdf5'
number_of_datasets = 50

# case_file = './perf_test_baron_rev2.hdf5'
# number_of_datasets = 27

path_to_application_exe = '/Applications/MATLAB_R2021b.app/bin/matlab'
path_to_fitting_script = "/Users/noahwalton/Documents/GitHub/ATARI/baron_fit_rev1.m"

input_options = {   'Overwrite Syndats'    :   False, 
                    'Overwrite Fits'       :   False,
                    'Use HDF5'             :   False    } 

perf_test = pf.performance_test(number_of_datasets,
                                case_file,
                                input_options=input_options)

sample_data_df = perf_test.generate_syndats(Ta_pair, exp, 
                                                solver='syndat_SLBW')

# %%
# test accessing generated data
# sample_0 = pd.read_hdf(case_file, 'sample_0/syndat_par')
# sample_data_df = pd.read_hdf(case_file, 'test_stats/sample_data')

print(f"Average # Resonances: {np.mean(sample_data_df.NumRes)}")
print(f"Min/Max # Resonances: {np.min(sample_data_df.NumRes)}/{np.max(sample_data_df.NumRes)}")
print(f"Energy Points (constant): {np.mean(sample_data_df.NumEpts)}")
print(f"Min theoretical SE: {np.min(sample_data_df.theo_exp_SE)}")
# sample_0

# %%
out = perf_test.generate_fits(False)
print(out)

# %%
# f = h5py.File(case_file, 'r+')
# # del f['sample_0']['fit_par']
# # del f['sample_0']['fit_pw']
# print(f['sample_9'].keys())
# # type(f['sample_9'])
# # print(f['test_stats/sample_data'].keys())
# f.close()


# %%
integral_FoMs, sample_data = perf_test.analyze()
# pd.read_hdf(case_file, 'integral_FoMs')
integral_FoMs

# %%
figure()
scatter(integral_FoMs.fit_theo_SE, sample_data.NumRes, marker='.', s=35)
ylim([-0.1,np.max(sample_data_df.NumRes)+1])
# xlim([0-np.max(integral_FoMs.fit_theo_SE)/3,np.max(integral_FoMs.fit_theo_SE)*1.1])

# %%
figure()
bins = hist(integral_FoMs.fit_theo_SE, bins=20)



# %%
pf.sample_case.plot_trans(case_file, 2, True)

# %%
pw_data, syndat_par_df, fit_par_df = pf.sample_case.read_sample_case_data(case_file,2)
# figure()
# plot(pw_data.E, pw_data.theo_trans)
# plot(pw_data.E, pw_data.est_trans)
# plot(pw_data.E, pw_data.exp_trans)
print(np.sum((pw_data.theo_trans-pw_data.est_trans)**2))

# %%



