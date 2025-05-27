from typing import Tuple, Union
from copy import copy
import numpy as np

from ATARI.theory.level_spacing_distributions import SpacingDistributionBase, merge
from ATARI.TAZ.RunMaster import RunMaster

__doc__ = """
This module merges 
"""

def tree_search_generator(branch:Union[tuple,int]):
    """
    A greedy tree search for spin group combination.

    Parameters
    ----------
    branch : tuple or int
        A branch of the tree
    
    Yields
    ------
    branch : tuple or int
        The next branch according to a greedy search
    """
    if isinstance(branch, tuple):
        yield branch
        for subbranch in branch:
            yield from tree_search_generator(subbranch)

def validate_tree(tree:tuple, num_spingroups:int):
    """
    ...
    """
    # Checking first branch point:
    if not isinstance(tree, tuple):
        raise TypeError('The partition structure must be embedded tuples of integer spingroup IDs.')
    # Checking each branch:
    valid_leaves = range(num_spingroups)
    seen_leaves  = set()
    for branch in tree_search_generator(tree):
        if isinstance(branch, tuple): # if tuple, make sure it is not empty or has 1 element.
            if   len(branch) == 0:  raise ValueError('A tuple in the partition structure does not have any elements.')
            elif len(branch) == 1:  raise ValueError('A tuple in the partition structure only has one element. This can and should be simplified.')
        elif isinstance(branch, int):
            leaf = branch
            if leaf not in valid_leaves:   raise ValueError(f'Spingroup ID, "{leaf}" is not valid. Spingroup IDs range from 0 to {num_spingroups-1}')
            if leaf in seen_leaves:        raise ValueError(f'Spingroup ID, "{leaf}" appears multiple times in the partition structure.')
            else:
                seen_leaves.add(leaf)
        else:
            raise ValueError('The partition structure can only contain embedded tuples of integer spingroup IDs.')
            
def find_branch_inclusions(branch:Union[tuple,int]):
    """
    Finds all spingroup IDs that exist within the branch.
    """
    def _inspect_structure(branch):
        if isinstance(branch, tuple):
            spingroups = []
            for subbranch in branch:
                spingroups += _inspect_structure(subbranch)
        else:
            spingroups = [branch]
        return spingroups
    return _inspect_structure(branch)

def merge_groups(prior:np.ndarray, level_spacing_dists:Tuple[SpacingDistributionBase],
                 merge_group_states:Tuple[Tuple[int]], false_group:bool):
    """
    ...
    """
    merged_prior = np.zeros((prior.shape[0],len(merge_group_states)+1))
    merged_distributions = []
    for i, group in enumerate(merge_group_states):
        merged_prior[:,i] = np.sum(prior[:,group], axis=1)
        merged_distribution = merge(*[level_spacing_dists[spingroup_ID] for spingroup_ID in group])
        merged_distributions.append(merged_distribution)
    if false_group:
        merged_prior[:,-1] = prior[:,-1] # false group
    return merged_prior, tuple(merged_distributions)

def sample_group_partitioner(E, energy_range:tuple,
                 level_spacing_dists:Tuple[SpacingDistributionBase], false_dens:float=0.0,
                 prior=None,
                 err:float=1e-9, partition_structure:tuple=None,
                 verbose:bool=False,
                 rng:np.random.Generator=None, seed:int=None):
    """
    Samples spingroups in a partitioned structure. ...

    Parameters
    ----------
    E                    : array-like of float
        Resonance energies for the ladder.
    energy_range         : (float, float)
        The ladder energy boundaries.
    level_spacing_dists  : ndarray[SpacingDistribution]
        The level-spacing distributions object.
    false_dens           : float
        The false level-density. Default = 0.0.
    prior                : array-like of float, optional
        The prior probabilitiy distribution for each spingroup.
    err                  : float
        A level-spacing probability threshold at which resonances are considered to be too far
        apart to be nearest neighbors.
    partition_structure  : tuple
        Imbedded tupled ints that determine the structure of merging for spingroup sampling.
    verbose              : bool
        The verbosity controller. Default is False.
    rng                  : np.random.Generator
        The random number generator for random sampling. Default is None.
    seed                 : int
        The random number seed for random sampling. Default is None.

    Returns
    -------
    samples : array of ints
        The sampled IDs for each resonance.
    """

    num_energies = len(E)
    num_spingroups = len(level_spacing_dists)
    lvl_dens = np.array([lvl_spacing_dist.lvl_dens for lvl_spacing_dist in level_spacing_dists] + [false_dens])
    if prior is None:
        prior = np.tile(lvl_dens/np.sum(lvl_dens), (num_energies,1))

    if partition_structure is None:
        partition_structure = tuple(range(num_spingroups))
    else:
        validate_tree(partition_structure, num_spingroups)

    # This is essentially the wavefunction collapse algorithm:
    spin_states = [tuple(range(num_spingroups+1)) for ires in range(len(E))]
    for partition in tree_search_generator(partition_structure):
        merge_group_states = [find_branch_inclusions(p) for p in partition]
        false_group = (partition == partition_structure) # only consider false groups in the outermost group
        merged_prior, merged_distributions = merge_groups(prior, level_spacing_dists, merge_group_states, false_group=false_group)
        group_resonances = np.array([state_is_in_spin_states(partition, spin_state) for spin_state in spin_states])
        group_res_indices = [ires for ires, group_res in enumerate(group_resonances) if group_res]
        E_case = E[group_resonances] # get the energies valid for this partition
        merged_prior = merged_prior[group_resonances,:]
        runmaster = RunMaster(E_case, energy_range, merged_distributions, false_dens=false_dens, prior=merged_prior, err=err, verbose=verbose)
        sample = runmaster.WigSample(1, rng, seed)[:,0]

        for ires, group in zip(group_res_indices, sample):
            if group == len(merge_group_states): # for false group
                spin_states[ires] = (num_spingroups,)
            else:
                spin_states[ires] = partition[group]
    
    # Extracting final spin state for each resonance:
    spingroups = []
    for res_spin_state in spin_states:
        if not isinstance(res_spin_state, int):
            if len(res_spin_state) > 1:
                raise RuntimeError('The spingroup states have not fully collapsed.')
            res_spin_state = res_spin_state[0]
        spingroups.append(res_spin_state)
    return np.array(spingroups)

def state_is_in_spin_states(partition, spin_states):
    a_partition_state = copy(partition)
    while not isinstance(a_partition_state, int):
        a_partition_state = a_partition_state[0]
    if isinstance(spin_states, int):
        return (a_partition_state == spin_states)
    else:
        return a_partition_state in spin_states