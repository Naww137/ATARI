
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from copy import copy, deepcopy
import pickle

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI

from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.syndat.data_classes import syndatOPT
from ATARI.syndat.control import Syndat_Control

from ATARI.sammy_interface import sammy_classes, sammy_functions, template_creator
from ATARI.AutoFit.functions import update_vary_resonance_ladder, separate_external_resonance_ladder
from ATARI.AutoFit.initial_FB_solve import InitialFBOPT
import ATARI.utils.plotting as myplot


import sys
import ATARI.utils.plotting as myplot
from ATARI.utils import atario
import ATARI.utils.hdf5 as h5io

import ATARI.AutoFit.functions as fn
from ATARI.utils.datacontainers import Evaluation_Data, Evaluation
from ATARI.sammy_interface.sammy_classes import SolverOPTs_YW
from ATARI.AutoFit import sammy_interface_bindings, fit_and_eliminate, auto_fit

from ATARI.AutoFit import window_functions
from ATARI.utils.atario import save_general_object, load_general_object
from ATARI.AutoFit import fit_and_eliminate
from ATARI.AutoFit import auto_fit

from ATARI.utils.file_handling import clean_and_make_directory
import time

os.chdir(os.path.dirname(__file__))

isamples = 2
sammypath = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'
rto = sammy_classes.SammyRunTimeOptions(sammypath,
                                        **{"Print":   True,
                                         "bayes":   True,
                                         "keep_runDIR": True,
                                         "sammy_runDIR": "sammy_runDIR"
                                         })

energy_range = (200,207)
Ta_pair = Particle_Pair(energy_range = energy_range)      
Ta_pair.add_spin_group(Jpi='3.0', J_ID=1, D=2.0, gn2_avg=465, gn2_dof=1, gg2_avg=32, gg2_dof=1000)
# Ta_pair.add_spin_group(Jpi='3.0', J_ID=1, D=8.79, gn2_avg=465, gn2_dof=1, gg2_avg=32, gg2_dof=1000)
# Ta_pair.add_spin_group(Jpi='4.0', J_ID=2, D=8.3031, gn2_avg=332.24347, gn2_dof=1, gg2_avg=32.0, gg2_dof=1000)

# setup experimental models
exp_model_T = Experimental_Model(reaction='transmission', title='trans', 
                                 energy_range=energy_range)
exp_model_Y = Experimental_Model(reaction='capture', title='cap', 
                                 n = (0.0056, 0.0),
                                 FP = (45.27, 0.05),
                                 energy_range=energy_range)
template_creator.make_input_template('template_T.inp', Ta_pair, exp_model_T, rto)
template_creator.make_input_template('template_Y.inp', Ta_pair, exp_model_Y, rto)
exp_model_T.template = os.path.realpath('template_T.inp')
exp_model_Y.template = os.path.realpath('template_Y.inp')


# Syndat models
syndat_T = Syndat_Model(exp_model_T, Transmission_RPI(), Transmission_RPI(), options = syndatOPT(calculate_covariance=True), title='trans')
for measurement_model in [syndat_T.generative_measurement_model, syndat_T.reductive_measurement_model]:
    measurement_model.approximate_unknown_data(exp_model=exp_model_T, smooth=False, check_trig=True)
syndat_Y = Syndat_Model(exp_model_Y, Capture_Yield_RPI(), Capture_Yield_RPI(), options = syndatOPT(calculate_covariance=False), title='cap')
for measurement_model in [syndat_Y.generative_measurement_model, syndat_Y.reductive_measurement_model]:
    measurement_model.approximate_unknown_data(exp_model=exp_model_Y, smooth=False, check_trig=True)
syndat = Syndat_Control(particle_pair= Ta_pair, syndat_models= [syndat_T, syndat_Y], model_correlations=[])

# draw sample(s) and save
if os.path.isfile("./syndat_samples.hdf5"):
    os.remove("./syndat_samples.hdf5")
syndat.sample(rto, num_samples=isamples, save_samples_to_hdf5=True, hdf5_file="./syndat_samples.hdf5", overwrite=True)

#%%

#%% Options

sammy_rto_fit = sammy_classes.SammyRunTimeOptions(sammypath, **{"Print":True, "bayes":True, "keep_runDIR":False, "sammy_runDIR":f"sammy_runDIR_test_FE"})

initialFBopt = InitialFBOPT(starting_Gn1_multiplier = 100,
                            starting_Gg_multiplier = 1,
                            fit_all_spin_groups=False,
                            spin_group_keys = ['3.0'],
                            external_resonances=True
                            )

solver_options = SolverOPTs_YW(max_steps = 50,
                            step_threshold=0.1,
                            LevMar=True, LevMarV=1.5,LevMarVd=5,minF = 1e-4,maxF = 1.5,
                            initial_parameter_uncertainty=0.1, iterations=1,
                            idc_at_theory=False)

fit_and_elim_options = fit_and_eliminate.FitAndEliminateOPT(chi2_allowed=0.001,
                                                            width_elimination=False, 
                                                            greedy_mode=True,
                                                            deep_fit_max_iter = 25,
                                                            deep_fit_step_thr = 0.01,
                                                            LevMarV0_priorpassed = 0.1)



### get data
eval_data = Evaluation_Data.from_hdf5(experimental_titles=['trans','cap'], experimental_models=[exp_model_T, exp_model_Y], sample_file="./syndat_samples.hdf5", isample=0)


initial_feature_bank, fixed_resonance_ladder = fn.get_initial_resonance_ladder(initialFBopt, Ta_pair, energy_range, external_resonance_ladder=None)
# initial_resonance_ladder.sort_values('E', inplace=True)
# initial_resonance_ladder.reset_index(drop=True, inplace=True)
print(initial_feature_bank)
print(fixed_resonance_ladder)

#### Fit and eliminate
# solver_initial = sammy_interface_bindings.Solver_factory(sammy_rto_fit, solver_options._solver, solver_options, Ta_pair, eval_data) 
# solver_elim = sammy_interface_bindings.Solver_factory(sammy_rto_fit, solver_options._solver, solver_options, Ta_pair, eval_data)

# fe = fit_and_eliminate.FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=fit_and_elim_options, particle_pair=Ta_pair)
# initial_samout = fe.initial_fit(initial_feature_bank, fixed_resonance_ladder=fixed_resonance_ladder)

# initial_feature_bank, fixed_resonance_ladder = separate_external_resonance_ladder(initial_samout.par_post, fe.output.external_resonance_indices)
# print(initial_feature_bank)
# print(fixed_resonance_ladder)

# from ATARI.AutoFit.functions import objective_func
# chi2 = np.sum(initial_samout.chi2_post)
# obj_train = objective_func(chi2, initial_samout.par_post, Ta_pair, None, Wigner_informed=True, PorterThomas_informed=True)
# print(chi2)
# print(obj_train)

# elimination_history = fe.eliminate(initial_feature_bank, target_ires=0, fixed_resonance_ladder=fixed_resonance_ladder, )

# print(elimination_history[4]['selected_ladder_chars'].par_post)

# print(elimination_history[0]['selected_ladder_chars'].par_post)





### Instead, do full autofit
autofit_options = auto_fit.AutoFitOPT(save_elimination_history   = True,
                                    save_CV_elimination_history  = True,
                                    parallel_CV                  = False,
                                    parallel_processes           = 5,
                                    final_fit_to_0_res           = False)
af = auto_fit.AutoFit(sammy_rto_fit, Ta_pair, solver_options, solver_options, AutoFit_options=autofit_options, fit_and_elim_options=fit_and_elim_options)
autofit_out = af.fit(eval_data, initial_feature_bank, fixed_resonance_indices=[])

### Save
# file = open(os.path.realpath(f'./fits_{iw}/Out_{isample}.pkl'), 'wb')
# pickle.dump(autofit_out, file)
# file.close()



