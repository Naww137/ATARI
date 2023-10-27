#%%
import numpy as np
import pandas as pd
import os
# import ATARI
from ATARI.PiTFAll.performance_test import Performance_Test
from ATARI.models.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
import h5py
import scipy.stats as sts

import ATARI.utils.io.hdf5 as h5io

#%% Initialize particle pair and quantup spin groups of interest

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

Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                spin_groups=spin_groups, average_parameters=average_parameters,
                                input_options={})

#%%

# initialize experimental setup
E_min_max = [75, 125]
input_options = { 'Add Noise': True,
            'Sample TURP':True,
            'Sample TOCS':True, 
            'Calculate Covariance': True,
            'Compression Points':[],
            'Grouping Factors':None}

experiment_parameters = {'bw': {'val':0.1024,    'unc'   :   0}}

exp = Experiment(E_min_max, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)

# %%
# run a performance test with the PiTFAll module

case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125/Ta181_500samples_E75_125.hdf5'

dataset_range = (0, 10)
input_options = {   'Overwrite Syndats'    :   False, 
                    'Overwrite Fits'       :   False,
                    'Use HDF5'             :   True,
                    'Vary Erange'          :   None} 


from ATARI.utils.io.experimental_parameters import BuildExperimentalParameters_fromDIRECT

# build experimental parameters
builder_exppar = BuildExperimentalParameters_fromDIRECT(0.067166, 0, 1e-2)
exppar = builder_exppar.construct()

import sample_case
isample = 2

# could add code to get all exsisting models in the performance test
model_labels = ['true']
# siglevels = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5]
siglevels = np.linspace(0.001, 0.9, 50)
for sig in siglevels:
    model_labels.append(f"par_est_{isample}_pv_{str(sig).split('.')[1]}")
    # model_labels.append(f"est_iFB_{isample}_pv_{str(sig).split('.')[1]}")

dc = sample_case.get_dc_for_isample_fromHDF5(case_file, isample, model_labels, Ta_pair, exppar)


print(dc.pw.exp_columns)

# analyze syndat


# analyze fits

# write finepw or results to hdf5 if desired

# %%
