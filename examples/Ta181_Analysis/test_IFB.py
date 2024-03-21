# %%

from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import os
from copy import copy, deepcopy
from ATARI.sammy_interface import sammy_classes, sammy_functions, template_creator
import plotting as myplot
from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model

from ATARI.AutoFit import elim_addit_funcs

# %%


sammypath = "/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy"


bad_case_dir = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/IFB_failed_cases/bad_cases_for_Noah_check"
bad_cases_ids = [190, 268, 105, 372, 127]

for case_id in bad_cases_ids:

    case_filename = f'sample_{case_id}.pkl'
    gen_params_filename =  f'params_gen.pkl'

    # loading case
    sample_data = elim_addit_funcs.load_obj_from_pkl(folder_name = bad_case_dir,
                                                        pkl_fname = case_filename )

    params_loaded = elim_addit_funcs.load_obj_from_pkl(folder_name = bad_case_dir,
                                                        pkl_fname= gen_params_filename)

    ### define new particle pair for updated version
    syndat_loaded = params_loaded
    # Ta_pair_loaded = syndat_loaded.particle_pair
    # Ta_pair = Ta_pair_loaded
    Ta_pair = Particle_Pair()
    Ta_pair.add_spin_group(Jpi='3.0',
                        J_ID=1,
                        D=9.0030,
                        gn2_avg=452.56615,
                        gn2_dof=1,
                        gg2_avg=32.0,
                        gg2_dof=1000)

    Ta_pair.add_spin_group(Jpi='4.0',
                        J_ID=2,
                        D=8.3031,
                        gn2_avg=332.24347, 
                        gn2_dof=1,
                        gg2_avg=32.0,
                        gg2_dof=1000)


    ### udpate experiment template
    for synmod in syndat_loaded.syndat_models:
        exp = synmod.generative_experimental_model
        filepath = os.path.realpath(f'./data/template_{exp.title}_edited_oldres')
        exp.template = os.path.realpath(filepath)


    sammy_rto_fit = sammy_classes.SammyRunTimeOptions(sammypath,
                                            {"Print":   True,
                                            "bayes":   True,
                                            "keep_runDIR": True,
                                            "sammy_runDIR": "sammy_runDIR"
                                            })


    options = InitialFBOPT(Gn_threshold=1e-1,
                        iterations=2,
                        max_steps = 30,
                        LevMarV0=0.05,
                        batch_fitpar = True,
                        fit_all_spin_groups=False,
                        spin_group_keys = ['3.0'],
                        step_threshold=0.001,
                        starting_Gn1_multiplier = 20,
                        num_Elam=50,
                        external_resonances=False)
    autofit_initial = InitialFB(options)



    datasets = [val.pw_reduced for key, val in sample_data.items()]
    experiments = [synmod.generative_experimental_model for synmod in syndat_loaded.syndat_models]
    covariance_data = [{} for key, val in sample_data.items()]


    # outs = autofit_initial.fit(Ta_pair,
    #                             [202, 227],
    #                             datasets,
    #                             experiments,
    #                             covariance_data,
    #                             sammy_rto_fit)

    filehandler = open(f"/Users/noahwalton/Documents/GitHub/ATARI/examples/Ta181_Analysis/initialFB_{case_id}.pkl","rb")
    outs = pickle.load(filehandler)
    filehandler.close()


    true_par = deepcopy(sample_data['trans1mm'].par_true)
    true_par['varyE'] = np.ones(len(true_par))
    true_par['varyGg'] = np.ones(len(true_par))
    true_par['varyGn1'] = np.ones(len(true_par))

    sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = Ta_pair,
            resonance_ladder = true_par,  

            datasets= datasets,
            experiments = experiments,
            experimental_covariance= covariance_data, 
            
            max_steps = 5,
            iterations = 2,
            step_threshold = 0.001,
            LevMar = True,
            LevMarV = 1.5,
            LevMarVd = 5,
            minF = 1e-5,
            maxF = 2.0,
            initial_parameter_uncertainty = 0.1,
            
            autoelim_threshold = None,
            LS = False,
            )

    true_out = sammy_functions.run_sammy_YW(sammyINPyw, sammy_rto_fit)

    # # vary1 = np.tile([0,0,1,1], int(len(outs.sammy_outs_fit_2[-1].par_post)/4))
    # # vary2 = np.tile([1,1,0,0], int(len(outs.sammy_outs_fit_2[-1].par_post)/4))
    # pattern1= [0,1]
    # vary1 = np.tile(pattern1, (len(outs.sammy_outs_fit_2[-1].par_post) // len(pattern1)) + 1)[:len(outs.sammy_outs_fit_2[-1].par_post)]
    # pattern2= [1,0]
    # vary2 = np.tile(pattern2, (len(outs.sammy_outs_fit_2[-1].par_post) // len(pattern2)) + 1)[:len(outs.sammy_outs_fit_2[-1].par_post)]

    # chi2_per_ndat = np.sum(outs.sammy_outs_fit_2[-1].chi2_post) / np.sum([len(each) for each in outs.sammy_outs_fit_2[-1].pw_post])
    # parameters = deepcopy(outs.sammy_outs_fit_2[-1].par_post)
    # tracker = 1
    # while chi2_per_ndat > np.sum(true_out.chi2_post)/np.sum([len(each) for each in true_out.pw_post]):

    #     # if tracker % 2 == 0:
    #     parameters['varyE'] = vary1
    #     parameters['varyGg'] = vary1
    #     parameters['varyGn1'] = vary1
    #     print("vary1")
    #     # else:
    #     sammyINPyw.resonance_ladder = parameters
    #     sammyout = sammy_functions.run_sammy_YW(sammyINPyw, sammy_rto_fit)
    #     chi2_per_ndat = np.sum(sammyout.chi2_post)/np.sum([len(each) for each in sammyout.pw_post])
    #     parameters = sammyout.par_post


    #     parameters['varyE'] = vary2
    #     parameters['varyGg'] = vary2
    #     parameters['varyGn1'] = vary2
    #     print("vary2")
    #     sammyINPyw.resonance_ladder = parameters
    #     sammyout = sammy_functions.run_sammy_YW(sammyINPyw, sammy_rto_fit)
    #     chi2_per_ndat = np.sum(sammyout.chi2_post)/np.sum([len(each) for each in sammyout.pw_post])
    #     parameters = sammyout.par_post

    #     tracker += 1
    #     if tracker > 10:
    #         break
    #     if chi2_per_ndat >= chi2_per_ndat_next:
    #         break
    #     chi2_per_ndat_next = chi2_per_ndat
        

    import pickle
    # filehandler = open(f"/Users/noahwalton/Documents/GitHub/ATARI/examples/Ta181_Analysis/initialFB_{case_id}.pkl","wb")
    # pickle.dump(outs, filehandler)
    # filehandler.close()
    # filehandler = open(f"/Users/noahwalton/Documents/GitHub/ATARI/examples/Ta181_Analysis/true_{case_id}.pkl", 'wb')
    # pickle.dump(true_out, filehandler)
    # filehandler.close()
    # filehandler = open(f"/Users/noahwalton/Documents/GitHub/ATARI/examples/Ta181_Analysis/after_01_{case_id}.pkl", 'wb')
    # pickle.dump(sammyout, filehandler)
    # filehandler.close()
    filehandler = open(f"/Users/noahwalton/Documents/GitHub/ATARI/examples/Ta181_Analysis/initialFB_{case_id}_50p_inLM.pkl","wb")
    pickle.dump(outs, filehandler)
    filehandler.close()