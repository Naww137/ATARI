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

from scipy import integrate
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

case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125/Ta181_500samples_E75_125_1.hdf5'
# case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/TestFMReduction2.hdf5'

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
from ATARI.utils.io.experimental_parameters import BuildExperimentalParameters_fromDIRECT, DirectExperimentalParameters
from ATARI.utils.io.theoretical_parameters import BuildTheoreticalParameters_fromHDF5, BuildTheoreticalParameters_fromATARI, DirectTheoreticalParameters
from ATARI.utils.io.pointwise_container import BuildPointwiseContainer_fromHDF5, BuildPointwiseContainer_fromATARI, DirectPointwiseContainer
from ATARI.utils.io.data_container import BuildDataContainer_fromBUILDERS, BuildDataContainer_fromOBJECTS, DirectDataContainer

# build experimental parameters
builder_exppar = BuildExperimentalParameters_fromDIRECT(0.067166, 0, 1e-2)
exppar = builder_exppar.construct()

# siglevels = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5]
# siglevels = np.linspace(0.001, 0.9, 50)
siglevels = np.logspace(-4,-2, 10)
siglevels[0] = np.round(siglevels[0], 4)
# siglevels = np.concatenate( [siglevels, np.linspace(0.001, 0.9, 50)] )

residual_dict = {}
MSE_dict = {}
for sig in siglevels:
    residual_dict[sig]= []
    MSE_dict[sig] = []

for isample in range(0,480):

    # build true model parameter object
    builder_theopar = BuildTheoreticalParameters_fromHDF5('true', case_file, isample, Ta_pair)
    truepar = builder_theopar.construct()

    # build pointwise data 
    builder_pw = BuildPointwiseContainer_fromHDF5(case_file, isample)
    pw = builder_pw.construct_full()

    builder_dc = BuildDataContainer_fromOBJECTS( pw, exppar, [truepar])
    dc = builder_dc.construct()

    for sig in siglevels:
        sig_str = str(sig).split('.')[1][0:7]
        est_par_builder = BuildTheoreticalParameters_fromHDF5(f'par_est_iFB_{isample}_pv_{sig_str}', case_file, isample, Ta_pair)
        est_par = est_par_builder.construct()
        dc.add_theoretical_parameters(est_par)

    dc.models_to_pw()

    for sig in siglevels:
        sig_str = str(sig).split('.')[1][0:7]
        residual_dict[sig].append( np.array(dc.pw.fine[f'par_est_iFB_{isample}_pv_{sig_str}_xs']-dc.pw.fine.true_xs) )

        # print(np.sum( (dc.pw.fine[f'par_est_iFB_{isample}_pv_{sig_str}_xs']-dc.pw.fine.true_xs)**2))
        MSE = integrate.trapezoid((dc.pw.fine.true_xs-dc.pw.fine[f'par_est_iFB_{isample}_pv_{sig_str}_xs'])**2, dc.pw.fine.E)
        print(MSE)
        MSE_dict[sig].append(MSE)
    
    print(f"Completed sample: {isample}")
    # print(dc.pw.fine_columns)
#%%
fnns = []
for sig in siglevels:
    residual_matrix = np.array(residual_dict[sig])

    fnorm = np.linalg.norm(residual_matrix)
    fnorm_normed = fnorm/residual_matrix.size
    fnns.append(fnorm_normed)

    print()
    print(sig)
    print(fnorm)
    print(fnorm_normed)

#%%
MSEs = []
for residual in residual_dict[0.0001]:
    MSEs.append(integrate.trapezoid(residual**2, dc.pw.fine.E))


# %%

figure()
plot(siglevels, fnns, 'o')
xlabel('Significance Level')
ylabel(r'$\frac{||R||_F}{R.size}$')
title('Informed Feature Bank')


# %%

# test = np.load('/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/pvals_pv_0p07.npy')

# figure()
# hist(np.log10(test[test >0]), bins=100)
# xlim([-4,0])
