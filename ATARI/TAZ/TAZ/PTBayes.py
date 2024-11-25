from copy import copy
from pandas import DataFrame
import numpy as np
from numpy import newaxis as NA

from TAZ.Theory import porter_thomas_dist
from TAZ import Reaction

__doc__ = """
This module contains Bayes' update for the probabilistic distribution on the neutron widths (and
gamma widths if specified).
"""

def PTBayes(resonances:DataFrame, reaction:Reaction, false_width_dist=None, prior=None, gamma_width_on:bool=False):
    """
    Performs a Bayesian update on the spingroup probabilities of resonances, based on Porter-Thomas
    Distribution on the neutron widths (and gamma widths if specified).

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Parameters
    ----------
    resonances       : DataFrame
        The dataframe of resonance energies, widths, etc.
    reaction         : Reaction
        A Reaction object that holds the mean parameters for the reaction.
    false_width_dist : function, optional
        The PDF for the neutron widths of false resonances. If none are given, false widths are
        sampled from the joint neutron width PDF of all spingroups.
    prior            : float [L,G+1], optional
        Optional prior spingroup probabilities. Such probabilities may include information on
        statistical fits. However, the probabilities must be independent of the information
        provided by the width distributions.
    gamma_width_on   : bool
        Determines whether the gamma-width probabilities are calculated based on the theoretical
        distribution. Many RRR evaluations assume the gamma widths are more or less constant. This
        theory is controversial, which is why the gamma width distribution is not considered by
        default. Default is False.

    Returns
    -------
    posterior        : float [L,G+1]
        The posterior spingroup probabilities.
    log_likelihood   : float
        Calculated log-likelihood.
    """

    # Error Checking:
    if type(resonances) is not DataFrame:
        raise TypeError('The "resonances" argument must be a DataFrame')
    if type(reaction) is not Reaction:
        raise TypeError('The "mean_param" argument must be a "Reaction" object.')
    
    E  = resonances['E'].to_numpy()
    Gg = resonances['Gg'].to_numpy()
    Gn = resonances['Gn1'].to_numpy()
    
    # Setting prior:
    if prior == None:
        prob = reaction.lvl_dens_all / np.sum(reaction.lvl_dens_all)
        prior = np.tile(prob, (E.size,1))
    posterior = prior

    # Neutron widths:
    Gnms = reaction.Gnm
    for g, Gnm in enumerate(Gnms):
        posterior[:,g] *= porter_thomas_dist.pdf(Gn, mean=Gnm(E), df=reaction.nDOF[g], trunc=0.0)

    # Gamma widths: (if gamma_width_on is True)
    if gamma_width_on:
        for g, Gnm in enumerate(Gnms):
            posterior[:,g] *= porter_thomas_dist.pdf(Gg, mean=reaction.Ggm[g], df=reaction.gDOF[g], trunc=0.0)

    # False distribution:
    if (reaction.false_dens != 0.0) and (false_width_dist is not None):
        posterior[:,-1] *= false_width_dist(E, Gn, Gg)
    else:
        posterior[:,-1] *= np.sum(posterior[:,:-1], axis=1) / np.sum(prob[:-1])

    # Normalization:
    total_probability = np.sum(posterior, axis=1)
    posterior /= total_probability[:,NA]

    # Log likelihood:
    log_likelihood = np.sum(np.log(total_probability))

    return posterior, log_likelihood

def PTMaxLogLikelihoods(probs, num_best:int):
    """
    Finds the best ladder given independent probabilities for each spingroup of each resonance.

    Let `L` be the number of resonances and `G` be the number of (true) spingroups.

    Parameters
    ----------
    probs    : float [L,G+1]
        Independent probabilities on spingroup assignment for each resonance.
    num_best : int
        The number of maximal likelihood solutions to return.

    Returns
    -------
    best_spingroup_ladders : list[list[int]]
        A list of the highest likelihood spingroup ladders, represented by a list of spingroup IDs.
    best_log_likelihoods   : list[float]
        Log-likelihoods for the best spingroup assignments.
    """

    L = probs.shape[0]
    G = probs.shape[1] - 1

    # Logification:
    with np.errstate(divide='ignore'):
        log_probs = np.log(probs)

    # Finding best spingroups:
    best_spingroup_ladders = [[]]
    best_log_likelihoods = [0.0]
    for i in range(L):
        new_best_spingroup_ladders = []
        new_best_log_likelihoods = []
        for old_spingroup_ladder, old_likelihood in zip(best_spingroup_ladders, best_log_likelihoods):
            for g in range(G+1):
                new_likelihood = old_likelihood + log_probs[i,g]
                new_spingroup_ladder = old_spingroup_ladder + [g]
                idx = np.searchsorted(new_best_log_likelihoods, new_likelihood)
                new_best_log_likelihoods.insert(idx, new_likelihood)
                new_best_spingroup_ladders.insert(idx, new_spingroup_ladder)
                if len(new_best_log_likelihoods) > num_best:
                    del new_best_log_likelihoods[0], new_best_spingroup_ladders[0]
        best_spingroup_ladders = copy(new_best_spingroup_ladders)
        best_log_likelihoods = copy(new_best_log_likelihoods)
    best_spingroup_ladders = np.array(best_spingroup_ladders, dtype=np.int8)
    # Provide in decreasing order:
    best_spingroup_ladders = best_spingroup_ladders[::-1]
    best_log_likelihoods   = best_log_likelihoods  [::-1]
    return best_spingroup_ladders, best_log_likelihoods