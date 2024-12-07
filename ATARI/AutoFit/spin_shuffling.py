from typing import Union, Tuple
import numpy as np
import pandas as pd
from copy import copy
from scipy.stats import rv_continuous

from ATARI.ModelData.particle_pair import Particle_Pair
# # from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyOutputData
# # from ATARI.sammy_interface.sammy_functions import run_sammy_YW
from ATARI.AutoFit.functions import objective_func
from ATARI.AutoFit.sammy_interface_bindings import Solver
from ATARI.AutoFit.functions import separate_external_resonance_ladder

from ATARI.TAZ.RunMaster import RunMaster
from ATARI.TAZ.PTBayes import PTBayes
from ATARI.TAZ.ATARI_interface import ATARI_to_TAZ

def assign_spingroups(respar:pd.DataFrame, spingroups:np.ndarray, particle_pair:Particle_Pair, respar_mask_in_window:pd.Series, num_spingroups:int):
    """
    ...
    """

    respar_post = copy(respar)
    for TAZ_spin_id, (Jpi, spingroup_params) in enumerate(particle_pair.spin_groups.items()):
        respar_post.loc[respar_mask_in_window].loc[spingroups == TAZ_spin_id, 'J_ID'] = spingroup_params['J_ID']
        respar_post.loc[respar_mask_in_window].loc[spingroups == TAZ_spin_id, 'L'   ] = spingroup_params['Ls'][0] # Assume largest L (NOTE: handling for larger L not considered yet)
        respar_post.loc[respar_mask_in_window].loc[spingroups == TAZ_spin_id, 'Jpi' ] = Jpi
    respar_post = respar_post.loc[spingroups != num_spingroups] # removing false resonances
    return respar_post

def shuffle_spingroups(respar:pd.DataFrame, particle_pair:Particle_Pair,
                       num_shuffles:float,
                       window_E_bounds:Tuple[float,float],
                       false_dens:float=0.0, false_width_dist:Union[rv_continuous,None]=None,
                       rng:np.random.Generator=None, seed:int=None):
    """
    ...
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    respar = copy(respar.sort_values(by=['E'], ignore_index=True, inplace=False))
    # Neglecting external resonances:
    respar_mask_in_window = (respar['E'] > window_E_bounds[0]) & (respar['E'] < window_E_bounds[1])
    respar_window = respar.loc[respar_mask_in_window]

    num_res = len(respar_window)
    reaction_TAZ, _, spingroup_IDs_TAZ = ATARI_to_TAZ(particle_pair)
    num_spingroups = reaction_TAZ.num_groups
    # J_IDs = [spingroup['J_ID'] for spingroup in particle_pair.spin_groups.values()]
    # respar, _ = ATARI_to_TAZ_resonances(respar, J_IDs)
    reaction_TAZ.false_dens = false_dens
    prior, log_likelihood_prior = PTBayes(respar_window, reaction_TAZ, false_width_dist=false_width_pdf)
    print('Porter-Thomas Prior:')
    print(prior)
    print()
    print('Reaction:')
    print(reaction_TAZ)
    print()
    print('Resonance Parameters:')
    print(respar_window)

    run_master = RunMaster(E=respar_window['E'], energy_range=window_E_bounds, level_spacing_dists=reaction_TAZ.distributions('Wigner'), false_dens=false_dens, prior=prior, log_likelihood_prior=log_likelihood_prior)
    spin_shuffles = run_master.WigSample(num_trials=num_shuffles, rng=rng, seed=seed)
    width_signs = rng.choice([-1, 1], size=spin_shuffles.shape) # shuffling width sign
    
    print('Spin Shuffles:')
    print(spin_shuffles.T)

    shuffled_respars = []
    for shuffle_id in range(num_shuffles):
        spin_shuffle = spin_shuffles[:,shuffle_id]
        shuffled_respar = assign_spingroups(respar=respar, spingroups=spin_shuffle, particle_pair=particle_pair, respar_mask_in_window=respar_mask_in_window)
        shuffled_respar.loc[respar_mask_in_window, 'Gn1'] *= width_signs[:,shuffle_id] # also shuffle sign of resonance widths
        for shuffled_respar_previous_case in shuffled_respars: # if this case was already added to the list, there is no point in retesting this case
            if shuffled_respar.equals(shuffled_respar_previous_case):
                break
        else: # else, add to list
            shuffled_respars.append(shuffled_respar)
    return shuffled_respars

def minimize_spingroup_shuffling(respar:pd.DataFrame, solver:Solver,
                                 num_shuffles:float,
                                 window_E_bounds:Tuple[float,float],
                                 false_dens:float=0.0, false_width_dist:Union[rv_continuous,None]=None,
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
    shuffled_respars = shuffle_spingroups(respar=respar, particle_pair=particle_pair,
                                          num_shuffles=num_shuffles,
                                          window_E_bounds=window_E_bounds,
                                          false_dens=false_dens, false_width_dist=false_width_dist,
                                          rng=rng, seed=seed) # shuffle around spingroups, weighted by their likelihood
    
    spin_shuffle_cases = []
    for shuffled_respar in shuffled_respars:
        sammy_out = solver.fit(shuffled_respar, external_resonance_indices=external_resonance_indices)
        chi2 = sum(sammy_out.chi2_post)
        obj_value = objective_func(chi2=chi2, res_ladder=sammy_out.par_post, particle_pair=particle_pair, fixed_resonances_indices=[],
                                   Wigner_informed=Wig_informed, PorterThomas_informed=PT_informed)
        
        spin_shuffle_case = {'sammy_out': sammy_out,
                             'obj_value': obj_value}
        spin_shuffle_cases.append(spin_shuffle_case)

    return spin_shuffle_cases