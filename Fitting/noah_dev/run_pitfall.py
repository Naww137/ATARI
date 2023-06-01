#%%

import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
# import ATARI
from ATARI.PiTFAll.performance_test import Performance_Test
from ATARI.syndat.particle_pair import Particle_Pair
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

case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125_1.hdf5'

dataset_range = (0, 480)
input_options = {   'Overwrite Syndats'    :   False, 
                    'Overwrite Fits'       :   False,
                    'Use HDF5'             :   True,
                    'Vary Erange'          :   None} 

perf_test = Performance_Test(dataset_range,
                                case_file,
                                input_options=input_options)

# sample_data_df, out = perf_test.generate_syndats(Ta_pair, exp, 
#                                                     solver='syndat_SLBW')
# print(out)

#%%

import ATARI.atari_io.hdf5 as io
from ATARI.utils.misc import fine_egrid 
from ATARI.utils.io.datacontainer import DataContainer
from ATARI.utils.io.pointwise import PointwiseContainer
from ATARI.utils.io.parameters import TheoreticalParameters, ExperimentalParameters

residual_list = []

for isample in range(min(dataset_range), max(dataset_range)):

    pw_exp, CovT = h5io.read_pw_exp(case_file, isample)
    ladder_true = h5io.read_par(case_file, isample, 'true')
    ladder_fit = h5io.read_par(case_file, isample, 'fit')

    threshold_0T = 1e-2
    exppar = ExperimentalParameters(0.067166, 0, threshold_0T)
    theopar_true = TheoreticalParameters(Ta_pair, ladder_true, 'true')
    theopar_fit = TheoreticalParameters(Ta_pair, ladder_fit, 'fit_pv_0p07')

    pwfine = pd.DataFrame({'E':fine_egrid(pw_exp.E,100)})
    pw = PointwiseContainer(pw_exp, pwfine)

    pw.add_experimental(pw_exp, CovT, exppar)
    pw.add_model(theopar_true, exppar)
    pw.add_model(theopar_fit, exppar)

    dc = DataContainer(pw, exppar, theopar_true, {'fit_pv_0p07':theopar_fit})

    residual_list.append( dc.pw.fine.fit_pv_0p07_xs-dc.pw.fine.true_xs )

print(residual_list)
residual_matrix = np.array(residual_list)
print(np.shape(residual_matrix))

# %%

fnorm = np.linalg.norm(residual_matrix)
fnorm_normed = fnorm/residual_matrix.size

print(fnorm)
print(fnorm_normed)
# %%

# test = np.load('/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/pvals_pv_0p03.npy')

# figure()
# hist(np.log10(test[test >0]), bins=50)

#%%