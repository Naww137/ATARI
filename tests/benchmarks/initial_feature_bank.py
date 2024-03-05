# %%

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
from ATARI.syndat.control import syndatOPT, Syndat_Control

from ATARI.sammy_interface import sammy_classes, sammy_functions, template_creator
from ATARI.AutoFit.functions import update_vary_resonance_ladder
from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT
import ATARI.utils.plotting as myplot

os.chdir(os.path.dirname(__file__))

isamples = 2
sammypath = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'
rto = sammy_classes.SammyRunTimeOptions(sammypath,
                                        {"Print":   True,
                                         "bayes":   True,
                                         "keep_runDIR": True,
                                         "sammy_runDIR": "sammy_runDIR"
                                         })

#%%
energy_range = (200,225)

Ta_pair = Particle_Pair(energy_range = energy_range)      
Ta_pair.add_spin_group(Jpi='3.0', J_ID=1, D=8.79, gn2_avg=465, gn2_dof=1, gg2_avg=32, gg2_dof=1000)
Ta_pair.add_spin_group(Jpi='4.0', J_ID=2, D=8.3031, gn2_avg=332.24347, gn2_dof=1, gg2_avg=32.0, gg2_dof=1000)

# setup experimental models
exp_model_T = Experimental_Model(reaction='transmission', title='trans', 
                                #  n = (0.0056, 0.0),
                                #  FP = (100.14, 0.05),
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
syndat_Y = Syndat_Model(exp_model_Y, Capture_Yield_RPI(), Capture_Yield_RPI(), options = syndatOPT(calculate_covariance=False), title='cap')
syndat = Syndat_Control(particle_pair= Ta_pair, syndat_models= [syndat_T, syndat_Y], model_correlations=None, options=syndatOPT() )

# draw sample(s) and save
syndat.sample(rto, num_samples=isamples)
syndat_file = open("syndat.pkl", 'wb')
pickle.dump(syndat, syndat_file)
syndat_file.close()

#%%

chi2_fit = []
chi2_true = []

for i in range(isamples):
    
    sample1 = syndat.get_sample(i)

    # settup datasets
    datasets = [val.pw_reduced for key, val in sample1.items()]
    experiments = [synmod.generative_experimental_model for synmod in syndat.syndat_models]
    covariance_data = [val.covariance_data for key, val in sample1.items()]


    print("\nFit From True: \n")
    true_par = deepcopy(sample1['trans'].par_true)
    true_par = update_vary_resonance_ladder(true_par, varyE=1, varyGg=1, varyGn1=1)
    print(true_par)

    sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = Ta_pair,
            resonance_ladder = true_par,  

            datasets= datasets,
            experiments = experiments,
            experimental_covariance= covariance_data, 

            max_steps = 10,
            iterations = 2,
            step_threshold = 0.001,
            initial_parameter_uncertainty = 0.1,
            )

    true_out = sammy_functions.run_sammy_YW(sammyINPyw, rto)
    chi2_true.append(true_out.chi2_post)

    print("Fit from initial FB")
    options = InitialFBOPT(Gn_threshold=1e-2,
                        iterations=3,
                        max_steps = 50,
                        LevMarV0=0.05,
                        LevMarV = 1.25,
                        # LevMarVd = 2,
                        batch_fitpar = False,
                        fit_all_spin_groups=False,
                        spin_group_keys = ['3.0'],
                        step_threshold=0.001,
                        starting_Gn1_multiplier = 50,
                        starting_Gg_multiplier = 0.01,
                        fitpar1= [0,1,1], 
                        fitpar2=[1,1,1],
                        # num_Elam=100,
                        external_resonances=True)

    autofit_initial = InitialFB(options)
    outs = autofit_initial.fit(Ta_pair,
                                energy_range,
                                datasets,
                                experiments,
                                covariance_data,
                                rto)


    chi2_fit.append(outs.sammy_outs_fit_2[-1].chi2_post)
    
    if np.sum(outs.sammy_outs_fit_2[-1].chi2_post) > np.sum(true_out.chi2_post):

        fig = myplot.plot_reduced_data_TY(datasets=datasets,
                                    experiments=experiments,
                                    priors= outs.sammy_outs_fit_1[0].pw,
                                    fits=outs.sammy_outs_fit_2[-1].pw_post,
                                    fits_chi2= outs.sammy_outs_fit_2[-1].chi2_post,
                                    fit_pars= outs.sammy_outs_fit_2[-1].par_post,
                                    true= true_out.pw_post,
                                    true_chi2= true_out.chi2,
                                    true_pars= true_out.par_post, 
                                    xlim=energy_range)
        fig.savefig(f"badfit_{i}.png")
        fig.close()

np.save("chi2fit.npy", chi2_fit)
np.save("chi2true.npy", chi2_true)


# %%
