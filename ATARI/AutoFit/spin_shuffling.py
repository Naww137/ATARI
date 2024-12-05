import numpy as np
import pandas as pd
from copy import copy

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyOutputData
from ATARI.sammy_interface.sammy_functions import run_sammy_YW
from ATARI.AutoFit.functions import objective_func
from ATARI.AutoFit.sammy_interface_bindings import Solver

from ATARI.TAZ.RunMaster import RunMaster
from ATARI.TAZ.PTBayes import PTBayes
from ATARI.TAZ.ATARI_interface import ATARI_to_TAZ
from ATARI.ASTERIODS.empirical_false_width_distribution import estimate_false_missing_density_from_spacing, empirical_false_distribution

def assign_spingroups(respar:pd.DataFrame, spingroups:np.ndarray, particle_pair:Particle_Pair):
    """
    ...
    """

    respar_post = copy(respar)
    for TAZ_spin_id, (Jpi, spingroup_params) in enumerate(particle_pair.spin_groups.items()):
        respar_post.loc[spingroups == TAZ_spin_id, 'J_ID'] = spingroup_params['J_ID']
        respar_post.loc[spingroups == TAZ_spin_id, 'L'   ] = spingroup_params['Ls']
        respar_post.loc[spingroups == TAZ_spin_id, 'Jpi' ] = Jpi
    return respar_post

def shuffle_spingroups(respar:pd.DataFrame, particle_pair:Particle_Pair,
                       num_shuffles:float, false_resonances:bool=False,
                       rng:np.random.Generator=None, seed:int=None):
    """
    ...
    """

    respar = copy(respar.sort_values(by=['E'], ignore_index=True, inplace=False))
    num_res = len(respar)
    reaction_TAZ, _, spingroup_IDs_TAZ = ATARI_to_TAZ(particle_pair)
    # J_IDs = [spingroup['J_ID'] for spingroup in particle_pair.spin_groups.values()]
    # respar, _ = ATARI_to_TAZ_resonances(respar, J_IDs)
    if false_resonances:
        lvl_dens_exp = sum(reaction_TAZ.lvl_dens_all)
        ladder_size = reaction_TAZ.EB[1] - reaction_TAZ.EB[0]
        false_dens = estimate_false_missing_density_from_spacing(num_res, lvl_dens_exp, ladder_size)
        false_width_dist, _ = empirical_false_distribution(respar['Gn1'], false_frac, )
    else:
        false_dens = 0.0
        false_width_dist = None
    prior, log_likelihood_prior = PTBayes(respar, reaction_TAZ, false_width_dist=false_width_dist.pdf)
    run_master = RunMaster(respar['E'], reaction_TAZ.EB, level_spacing_dists=reaction_TAZ.distributions('Wigner'), false_dens=false_dens, prior=prior, log_likelihood_prior=log_likelihood_prior)
    spin_shuffles = run_master.WigSample(num_trials=num_shuffles, rng=rng, seed=seed)
    shuffle_respars = []
    for shuffle_id in range(num_shuffles):
        spin_shuffle = spin_shuffles[shuffle_id,:]
        shuffle_respar = assign_spingroups(respar, spin_shuffle)
        shuffle_respars.append(shuffle_respar)
    return shuffle_respars

def minimize_spingroup_shuffling(elimination_data:dict, solver:Solver,
                                 num_shuffles:float, false_resonances:bool=False,
                                 model_selection:str='chi2',
                                 external_resonance_indices = [],
                                 rng:np.random.Generator=None, seed:int=None):
    """
    ...
    """
    
    if   model_selection == 'chi2':
        Wig_informed = False
        PT_informed  = False
    elif model_selection == 'chi2+PT':
        Wig_informed = False
        PT_informed  = True
    elif model_selection == 'chi2+Wig':
        Wig_informed = True
        PT_informed  = False
    elif model_selection == 'chi2+Wig+PT':
        Wig_informed = True
        PT_informed  = True
    else:
        raise ValueError(f'Unknown model selection criteria, "{model_selection}".')

    particle_pair = solver.sammyINP.particle_pair
    respar = elimination_data['selected_ladder_chars'].par_post
    shuffled_respars = shuffle_spingroups(respar=respar, particle_pair=particle_pair,
                                          num_shuffles=num_shuffles, false_resonances=false_resonances,
                                          rng=rng, seed=seed)
    
    spin_shuffle_cases = []
    for shuffled_respar in shuffled_respars:
        shuffled_respar['Gn1'] *= np.random.choice([-1, 1], size=(len(shuffled_respar))) # also shuffle sign of resonance widths
        sammy_out = solver.fit(shuffled_respar, external_resonance_indices=external_resonance_indices)
        chi2 = sammy_out.chi2_post
        obj_value = objective_func(chi2=chi2, res_ladder=sammy_out.par_post, particle_pair=particle_pair, fixed_resonances_indices=[],
                                   Wigner_informed=Wig_informed, PorterThomas_informed=PT_informed)
        
        spin_shuffle_case = {'sammy_out': sammy_out,
                             'obj_value': obj_value}
        spin_shuffle_cases.append(spin_shuffle_case)

    return spin_shuffle_cases