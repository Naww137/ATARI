from typing import Union, Tuple
import numpy as np
import pandas as pd
from copy import copy
from scipy.stats import rv_continuous

from ATARI.ModelData.particle_pair import Particle_Pair
# from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyOutputData
# from ATARI.sammy_interface.sammy_functions import run_sammy_YW
from ATARI.AutoFit.functions import objective_func
from ATARI.AutoFit.sammy_interface_bindings import Solver
# from ATARI.AutoFit.functions import separate_external_resonance_ladder

from ATARI.TAZ.RunMaster import RunMaster
from ATARI.TAZ.PTBayes import PTBayes
from ATARI.TAZ.ATARI_interface import ATARI_to_TAZ

def assign_spingroups(respar:pd.DataFrame, spingroups:np.ndarray, neutron_width_signs:np.ndarray, capture_width_signs:np.ndarray, particle_pair:Particle_Pair, respar_mask_in_window:pd.Series, num_spingroups:int):
    """
    ...
    """

    respar_post = copy(respar)
    respar_post.loc[respar_mask_in_window, 'Gn1'] *= neutron_width_signs # shuffle sign of neutron widths
    respar_post.loc[respar_mask_in_window, 'Gg' ] *= capture_width_signs # shuffle sign of capture widths
    for TAZ_spin_id, (Jpi, spingroup_params) in enumerate(particle_pair.spin_groups.items()):

        # Finally, assign the value to the relevant rows in the original DataFrame
        selected_rows = respar_post.loc[respar_mask_in_window]
        selected_index = selected_rows.loc[spingroups == TAZ_spin_id].index
        respar_post.loc[selected_index, 'J_ID'] = spingroup_params['J_ID']
        respar_post.loc[selected_index, 'L'   ] = spingroup_params['Ls'][0] # Assume largest L (NOTE: handling for larger L not considered yet)
        respar_post.loc[selected_index, 'Jpi' ] = Jpi
        # respar_post.loc[respar_mask_in_window].loc[spingroups == TAZ_spin_id, 'J_ID'] = spingroup_params['J_ID']
        # respar_post.loc[respar_mask_in_window].loc[spingroups == TAZ_spin_id, 'L'   ] = spingroup_params['Ls'][0] # Assume largest L (NOTE: handling for larger L not considered yet)
        # respar_post.loc[respar_mask_in_window].loc[spingroups == TAZ_spin_id, 'Jpi' ] = Jpi

    selected_rows = respar_post.loc[respar_mask_in_window]
    selected_index = selected_rows.loc[spingroups == num_spingroups].index
    respar_post.drop(selected_index, inplace=True)
    # respar_post = respar_post.loc[spingroups != num_spingroups] # removing false resonances
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

    sort_indices = respar['E'].argsort()
    unsort_indices = np.argsort(sort_indices.values)
    respar_sorted = respar.iloc[sort_indices].reset_index(drop=True)
    # Neglecting external resonances:
    respar_mask_in_window = (respar_sorted['E'] > window_E_bounds[0]) & (respar_sorted['E'] < window_E_bounds[1])
    respar_window = respar_sorted.loc[respar_mask_in_window]

    # num_res = len(respar_window)
    reaction_TAZ, _, spingroup_IDs_TAZ = ATARI_to_TAZ(particle_pair)
    num_spingroups = reaction_TAZ.num_groups
    # J_IDs = [spingroup['J_ID'] for spingroup in particle_pair.spin_groups.values()]
    # respar, _ = ATARI_to_TAZ_resonances(respar, J_IDs)
    reaction_TAZ.false_dens = false_dens
    prior, log_likelihood_prior = PTBayes(respar_window, reaction_TAZ, false_width_dist=false_width_dist)
    print('Porter-Thomas Prior:')
    print(prior)
    print()
    print('Reaction:')
    print(reaction_TAZ)
    print()
    print('Resonance Parameters:')
    print(respar_sorted)

    run_master = RunMaster(E=respar_window['E'], energy_range=window_E_bounds, level_spacing_dists=reaction_TAZ.distributions('Wigner'), false_dens=false_dens, prior=prior, log_likelihood_prior=log_likelihood_prior)
    spin_shuffles = run_master.WigSample(num_trials=num_shuffles, rng=rng, seed=seed)
    neutron_width_signs = rng.choice([-1, 1], size=spin_shuffles.shape) # shuffling neutron width sign
    capture_width_signs = rng.choice([-1, 1], size=spin_shuffles.shape) # shuffling capture width sign
    
    print('Spin Shuffles:')

    shuffled_respars = []
    for shuffle_id in range(num_shuffles):
        spin_shuffle = spin_shuffles[:,shuffle_id]
        print(f'{shuffle_id}: {spin_shuffle.tolist()} | {neutron_width_signs[:,shuffle_id].tolist()} | {capture_width_signs[:,shuffle_id].tolist()}')
        shuffled_respar = assign_spingroups(respar=respar_sorted, spingroups=spin_shuffle, neutron_width_signs=neutron_width_signs[:,shuffle_id], capture_width_signs=capture_width_signs[:,shuffle_id], particle_pair=particle_pair, respar_mask_in_window=respar_mask_in_window, num_spingroups=num_spingroups)
        shuffled_respar = shuffled_respar.iloc[unsort_indices].reset_index(drop=True) # unsorting resonance parameters (so that fixed resonance indices are consistent)
        shuffled_respars.append(shuffled_respar)
    return shuffled_respars

def minimize_spingroup_shuffling(respar_prior:pd.DataFrame, solver:Solver,
                                 num_shuffles:float,
                                 window_E_bounds:Tuple[float,float],
                                 false_dens:float=0.0, false_width_dist:Union[rv_continuous,None]=None,
                                 model_selection:str='chi2',
                                 fixed_resonance_indices = [],
                                 target_Nres:int=None, unique_cases_only:bool=False,
                                 rng:np.random.Generator=None, seed:int=None,
                                 verbose:bool=False):
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

    # Check if there is enough resonances to generate the desired number of shuffles:
    respar_prior_mask_in_window = (respar_prior['E'] > window_E_bounds[0]) & (respar_prior['E'] < window_E_bounds[1])
    num_res_prior = len(respar_prior[respar_prior_mask_in_window])
    if unique_cases_only and (num_shuffles > len(particle_pair.spin_groups)**num_res_prior):
        raise RuntimeError(f'With {num_res_prior} resonances in the prior window, there are not enough unique cases to generate {num_shuffles} shuffles.')
    
    # Special case with no resonances:
    if num_res_prior == 0:
        sammy_out = solver.fit(respar_prior, external_resonance_indices=fixed_resonance_indices)
        chi2 = np.sum(sammy_out.chi2)
        obj_value = objective_func(chi2=chi2, res_ladder=sammy_out.par, particle_pair=particle_pair, fixed_resonances_indices=fixed_resonance_indices,
                                    Wigner_informed=Wig_informed, PorterThomas_informed=PT_informed)
        spin_shuffle_cases = [{'sammy_out': sammy_out,
                               'obj_value': obj_value}]*num_shuffles
        return spin_shuffle_cases

    # Running shuffling algorithm:
    num_shuffles_per_attempt = num_shuffles # FIXME: THIS VALUE IS ARBITRARY! MAKE IT REASONABLY CONSERVATIVE! THIS COULD BE DONE BETTER!
    shuffled_respars_filtered = []
    if verbose:
        print('Shuffling Spingroups')
    for attempt in range(1000):
        shuffled_respars = shuffle_spingroups(respar=respar_prior, particle_pair=particle_pair,
                                            num_shuffles=num_shuffles_per_attempt,
                                            window_E_bounds=window_E_bounds,
                                            false_dens=false_dens, false_width_dist=false_width_dist,
                                            rng=rng, seed=seed) # shuffle around spingroups, weighted by their likelihood
        
        # Only accept shuffles that have the desired size:
        for shuffled_respar in shuffled_respars:
            
            # Checking if shuffle meets target number of resonances:
            respar_mask_in_window = (shuffled_respar['E'] > window_E_bounds[0]) & (shuffled_respar['E'] < window_E_bounds[1])
            shuffled_respar_in_window = shuffled_respar.loc[respar_mask_in_window]
            num_res = len(shuffled_respar_in_window)
            print(num_res, target_Nres)
            if (target_Nres is not None) and (num_res != target_Nres):
                continue # if shuffle does not have target number of resonances, don't add to list
            add_shuffle_to_list = True
            
            # Checking if shuffle meets the unique requirement:
            if unique_cases_only:
                for shuffle_id_prev, shuffled_respar_previous_case in enumerate(shuffled_respars_filtered):
                    # Checking if it has already been added to list
                    if shuffled_respar.equals(shuffled_respar_previous_case):
                        add_shuffle_to_list = False
                        break

            if add_shuffle_to_list:
                shuffled_respars_filtered.append(shuffled_respar)
        # Check if we have enough shuffles:
        if len(shuffled_respars_filtered) >= num_shuffles:
            shuffled_respars = shuffled_respars_filtered[:num_shuffles]
            break
        else:
            if verbose:
                print(f'Attempt {attempt}: number of shuffles with correct target number of resonances was {len(shuffled_respars_filtered)} which is less than the number of shuffles, {num_shuffles}. Trying again.')
    else:
        raise RuntimeError(f'Not enough shuffles were accepted: {len(shuffled_respars_filtered)} accepted, but {num_shuffles} requested.')
    
    if verbose:
        print(f'Accepted shuffles on attempt {attempt}.')
        print('Optimizing fit on spin shuffles')

    spin_shuffle_cases = []
    shuffled_respars_already_listed = []
    for shuffled_respar in shuffled_respars:
        # If this case was already added to the list, there is no point in reoptimizing:
        for shuffle_id_prev, shuffled_respar_previous_case in enumerate(shuffled_respars_already_listed):
            # Checking if it has already been added to list. If so, add results of optimization:
            if shuffled_respar.equals(shuffled_respar_previous_case):
                assert not unique_cases_only
                spin_shuffle_case = copy(spin_shuffle_cases[shuffle_id_prev])
                spin_shuffle_cases.append(spin_shuffle_case)
                break
        else:
            # Else, optimize and add to the list:
            shuffled_respars_already_listed.append(shuffled_respar)
            sammy_out = solver.fit(shuffled_respar, external_resonance_indices=fixed_resonance_indices)
            if len(shuffled_respar) == len(fixed_resonance_indices): # FIXME: DOES THIS WORK?
                chi2 = np.sum(sammy_out.chi2)
            else:
                chi2 = np.sum(sammy_out.chi2_post)
            obj_value = objective_func(chi2=chi2, res_ladder=sammy_out.par_post, particle_pair=particle_pair, fixed_resonances_indices=fixed_resonance_indices,
                                    Wigner_informed=Wig_informed, PorterThomas_informed=PT_informed)
            
            spin_shuffle_case = {'sammy_out': sammy_out,
                                 'obj_value': obj_value}
            spin_shuffle_cases.append(spin_shuffle_case)

    return spin_shuffle_cases