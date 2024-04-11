import sys
sys.path.append('../')
# import ATARI.utils.hdf5 as h5io
# from matplotlib.pyplot import *
import numpy as np
import pandas as pd
# import importlib
import os
from copy import copy
from ATARI.sammy_interface import sammy_classes, sammy_functions, template_creator

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.particle import Particle, Neutron
from ATARI.theory.resonance_statistics import log_likelihood

from ATARI.PiTFAll import fnorm

sammypath = "/Users/colefritsch/GitLab/Sammy/sammy/build/bin/sammy"


num_samples = 5 # number of samples
num_sg_samples = 50 # number of spingroup samples
bounds = (200.0, 250.0) # energy limits for window
ladder_bounds = [180,270] # energy limits for resonance ladder

#%% Defining Particle Pair:

Ta181 = Particle(Z=73, A=181, I=3.5, mass=180.94803, name='Ta181')
Ta_pair = Particle_Pair(isotope = "Ta181",
                        resonance_ladder = pd.DataFrame(),
                        formalism = "XCT",
                        energy_range = ladder_bounds,
                        ac = 0.8127,
                        target=Ta181,
                        projectile=Neutron,
                        l_max = 1
)

# print quant number map up to l_max
Ta_pair.map_quantum_numbers(print_out=True)

# add spin group information for both s-wave resonances
Ta_pair.add_spin_group(Jpi='3.0',
                       J_ID=1,
                       D=9.0030,
                       gn2_avg=452.56615, #46.5,
                       gn2_dof=1,
                       gg2_avg=32.0,
                       gg2_dof=1000)

Ta_pair.add_spin_group(Jpi='4.0',
                       J_ID=2,
                       D=8.3031,
                       gn2_avg=332.24347, #35.5,
                       gn2_dof=1,
                       gg2_avg=32.0,
                       gg2_dof=1000)

Ta_pair.sample_resonance_ladder()
Ta_pair.expand_ladder()

#%% Experiments:

trans12mm_gen_exp = Experimental_Model(title = "T12mm", 
                                 reaction = "transmission", 
                                 energy_range = [200, 250], 
                                 template = None, 
                                 energy_grid = None, 
                                 n = (0.067166, 0.0), 
                                 FP = (35.185, 0.0), 
                                 t0 = (3326.0, 0.0), 
                                 burst = (10, 1.0), 
                                 temp = (300, 0.0), 
                                 channel_widths = { 
                                     "maxE": [300],
                                     "chw": [100.0],
                                     "dchw": [0.8]
                                 }
)

cap12mm_gen_exp = Experimental_Model(title = "Y12mm", 
                                 reaction = "capture", 
                                 energy_range = [200, 250], 
                                 template = None,
                                 energy_grid = None, 
                                 n = (0.067166, 0.0), 
                                 FP = (35.185, 0.0), 
                                 t0 = (3326.0, 0.0), 
                                 burst = (10, 1.0), 
                                 temp = (300, 0.0), 
                                 channel_widths = { 
                                     "maxE": [300],
                                     "chw": [100.0],
                                     "dchw": [0.8]
                                 }
)


sammy_rto = sammy_classes.SammyRunTimeOptions(sammypath,
                                              Print        = True,
                                              bayes        = False,
                                              keep_runDIR  = True,
                                              sammy_runDIR = "sammy_runDIR_1")

template_creator.make_input_template(
    'template_T.inp', Ta_pair, trans12mm_gen_exp, sammy_rto)

template_creator.make_input_template(
    'template_Y.inp', Ta_pair, cap12mm_gen_exp, sammy_rto)


cap12mm_gen_exp.template = os.path.realpath('template_Y.inp')
trans12mm_gen_exp.template = os.path.realpath('template_T.inp')

#%% RPIs:

from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI
from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.syndat.control import syndatOPT

# trans12mm_gen_meas = Transmission_RPI(trigo=trigo, trigs=trigs)
# trans12mm_red_meas = Transmission_RPI(trigo=trigo, trigs=trigs)
trans12mm_gen_meas = Transmission_RPI()
trans12mm_red_meas = Transmission_RPI()
trans12mm_red_meas.approximate_unknown_data(exp_model=trans12mm_gen_exp, smooth=False, check_trig=True)
trans12mm_gen_meas.approximate_unknown_data(exp_model=trans12mm_gen_exp, smooth=False, check_trig=True)

cap12mm_gen_meas = Capture_Yield_RPI()
cap12mm_red_meas = Capture_Yield_RPI()
cap12mm_red_meas.approximate_unknown_data(exp_model=cap12mm_gen_exp, smooth=False, check_trig=True)
cap12mm_gen_meas.approximate_unknown_data(exp_model=cap12mm_gen_exp, smooth=False, check_trig=True)

#%% Syndat Model:

synOPT = syndatOPT(calculate_covariance=True)

synOPT.calculate_covariance = True
syndat_trans12mm = Syndat_Model(trans12mm_gen_exp,
                            trans12mm_gen_meas,
                            trans12mm_red_meas,
                            options = synOPT,
                            title='trans12mm')
syndat_trans12mm.sample(Ta_pair, 
                    sammyRTO=sammy_rto,
                    num_samples=num_samples)

# synOPT.calculate_covariance = False
# syndat_cap12mm = Syndat_Model(cap12mm_gen_exp,
#                             cap12mm_gen_meas,
#                             cap12mm_red_meas,
#                             options = synOPT,
#                             title='cap12mm')
# syndat_cap12mm.sample(Ta_pair, 
#                     sammyRTO=sammy_rto,
#                     num_samples=num_samples)

#%% Optimization:

sammy_outs_chi2 = []
sammy_outs_F    = []
for sample, syndat_sampleT in enumerate(syndat_trans12mm.samples):
    print(f'Running sample #{sample}.')

    dataT = syndat_sampleT.pw_reduced
    # dataY = syndat_sampleY.pw_reduced

    F_min    = np.inf
    chi2_min = np.inf
    sammy_out_chi2_min = None
    sammy_out_F_min    = None
    for iter in range(num_sg_samples):
        # Changing spingroups inside the ladder:
        res_ladder_fit = Ta_pair.resonance_ladder
        res_ladder_fit[['varyE','varyGg','varyGn1']] = 1
        subset_bools = (res_ladder_fit['E'] > bounds[0]) & (res_ladder_fit['E'] < bounds[1])
        sgs = np.random.randint(0, 2, size=sum(subset_bools))
        res_ladder_fit.loc[subset_bools, 'J_ID'] = np.array([1.0, 2.0])[sgs]
        res_ladder_fit.loc[subset_bools, 'Jpi' ] = np.array([3.0, 4.0])[sgs]

        sammyINP = sammy_classes.SammyInputData(Ta_pair,
                                                res_ladder_fit,
                                                os.path.realpath('template_T.inp'),
                                                trans12mm_gen_exp,
                                                experimental_data = dataT,
                                                energy_grid = trans12mm_gen_exp.energy_grid,
                                                initial_parameter_uncertainty=1.0)

        sammy_rto = sammy_classes.SammyRunTimeOptions(sammypath,
                                                    Print        = True,
                                                    bayes        = True,
                                                    keep_runDIR  = True,
                                                    iterations   = 10,
                                                    sammy_runDIR = "sammy_runDIR")
        sammy_out = sammy_functions.run_sammy(sammyINP, sammy_rto)
        
        chi2_out = sammy_out.chi2_post
        if chi2_out < chi2_min:
            chi2_min = chi2_out
            sammy_out_chi2_min = copy(sammy_out)

        F_out = chi2_out + 2.0*log_likelihood(Ta_pair, res_ladder_fit, bounds)
        if F_out < F_min:
            F_min = F_out
            sammy_out_F_min    = copy(sammy_out)

    sammy_outs_chi2.append(sammy_out_chi2_min)
    sammy_outs_F.append(sammy_out_F_min)

#%% Performance:

print('Finding Performances:')
reactions = ['transmission']
template  = sammyINP.template
temp = 300.0

metrics_chi2 = []
metrics_F    = []
for sample, (sammy_out_chi2, sammy_out_F) in enumerate(zip(sammy_outs_chi2, sammy_outs_F)):
    # Chi2:
    print(f'Finding performance of sample #{sample}.')
    ResidualMatrixDict, _ = fnorm.build_residual_matrix_dict([sammy_out_chi2.par_post], [Ta_pair.resonance_ladder], sammypath, Ta_pair, (200.0, 250.0), temp, template, reactions)
    metric_chi2 = fnorm.calculate_fnorms(ResidualMatrixDict, reactions)
    metrics_chi2.append(metric_chi2['transmission'])

    # F:
    ResidualMatrixDict, _ = fnorm.build_residual_matrix_dict([sammy_out_F.par_post], [Ta_pair.resonance_ladder], sammypath, Ta_pair, (200.0, 250.0), temp, template, reactions)
    metric_F = fnorm.calculate_fnorms(ResidualMatrixDict, reactions)
    metrics_F.append(metric_F['transmission'])

    print(f'Performance (chi/F) = ({metric_chi2['transmission']:.8f} / {metric_F['transmission']:.8f})')

print()
print(f'Metric Chi2 = {np.mean(metrics_chi2):.8f} ± {np.std(metrics_chi2)/len(metrics_chi2):.8f}')
print(f'Metric F    = {np.mean(metrics_F):.8f} ± {np.std(metrics_F)/len(metrics_F):.8f}')

import matplotlib.pyplot as plt
plt.figure()
plt.clf()
plt.hist(metrics_chi2, 20, density=True, label='Chi2')
plt.hist(metrics_F   , 20, density=True, label='F'   )
plt.xlabel('SSE Performance', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('ATARI_results.png')

