from typing import Tuple
import numpy as np
from numpy import newaxis as NA
from scipy.stats import rv_continuous, gaussian_kde
from copy import copy

from ATARI.theory.distributions import porter_thomas_dist
from ATARI.TAZ.TAZ.DataClasses.Reaction import Reaction

__doc__ = """
This file contains tools for estimating the distribution of false widths.
"""

# def estimate_false_missing_fraction_from_widths(widths, width_dist:rv_continuous, trunc:float):
#     """
#     ...
#     """

#     widths = abs(np.array(widths, dtype=float))
#     widths_above_trunc = widths[widths >= trunc]
#     frac_widths_above_trunc = len(widths_above_trunc)/len(widths)
#     frac_widths_above_trunc_true = width_dist.isf(trunc)
#     frac_false_minus_missing = frac_widths_above_trunc/frac_widths_above_trunc_true
#     return frac_false_minus_missing

# def estimate_false_missing_fraction_from_spacing(num_res:int, lvl_dens_exp:float, ladder_size:float):
#     """
#     ...
#     """

#     num_res_exp = lvl_dens_exp * ladder_size
#     frac_false_minus_missing = (num_res - num_res_exp) / num_res
#     return frac_false_minus_missing

# =================================================================================================
#    False/Missing Level-Density Estimation:
# =================================================================================================

def estimate_false_missing_density_from_spacing(num_res:int, window_E_bounds:Tuple[float,float], lvl_dens_exp:float):
    """
    This function estimates the false minus missing level density from the expected level density
    and observed number of resonances in a window. It is difficult to separate the level densities
    for false and missing resonances due to the ambiguity of what can be considered a false and
    missing resonance.

    Parameters
    ----------
    num_res : int
        The number of resonances in the window.
    window_E_bounds : (float, float)
        The energy boundaries for the window.
    level_dens_exp : float
        The expected level density.

    Returns
    -------
    lvl_dens_false_minus_missing : float
        The estimated level density for false resonances minus the expected level density for
        missing resonances.
    """

    window_size = window_E_bounds[1] - window_E_bounds[0]
    lvl_dens_obs = num_res / window_size
    lvl_dens_false_minus_missing = lvl_dens_obs - lvl_dens_exp
    return lvl_dens_false_minus_missing

# def estimate_false_missing_density_from_widths(Gn, window_E_bounds:Tuple[float,float], trunc:float, reaction:Reaction):
#     """
#     This function estimates the false minus missing level density from the neutron width
#     distributions and the observed neutron widths. All widths above a certain threshold, `trunc`,
#     are assumed to be true, found resonances. 

#     Parameters
#     ----------
#     Gn : array-like of float
#         The partial neutron widths in a window.
#     window_E_bounds : (float, float)
#         The energy boundaries for the window.
#     trunc : float
#         A threshold where all widths above this value will be considered true, found resonances.
#     reaction : float
#         A TAZ Reaction object for mean parameters.

#     Returns
#     -------
#     lvl_dens_false_minus_missing : float
#         The estimated level density for false resonances minus the expected level density for
#         missing resonances.
#     """

#     E_eval = (window_E_bounds[0] + window_E_bounds[1]) / 2
#     window_size = window_E_bounds[1] - window_E_bounds[0]
#     level_dens_tot = sum(reaction.lvl_dens)

#     # Expected fraction of widths above truncation width:
#     lvl_dens_widths_above_trunc_exp = 0.0
#     for lvl_dens, Gnm, nDOF in zip(reaction.lvl_dens, reaction.Gnm, reaction.nDOF):
#         lvl_dens_widths_above_trunc_exp += lvl_dens * porter_thomas_dist.sf(trunc, mean=Gnm(E_eval), df=nDOF, trunc=0.0)

#     # Observed fraction of widths above truncation width:
#     Gn = abs(np.array(Gn, dtype=float))
#     Gn_above_trunc = Gn[Gn >= trunc]
#     lvl_dens_widths_below_trunc_obs = len(Gn_above_trunc)/window_size

#     # Returning false minus missing level-density function:
#     frac_false_minus_missing = frac_widths_above_trunc_obs/frac_widths_above_trunc_true
#     level_dens_false_minus_missing = frac_false_minus_missing * level_dens_tot
#     return level_dens_false_minus_missing

# NOTE: Energy Dependent Version Below
# def estimate_false_missing_density_from_widths(E, Gn, reaction:Reaction, trunc:float):
#     """
#     ...
#     """

#     level_dens_tot = sum(reaction.lvl_dens)

#     # True fraction of widths above truncation width:
#     NUM_ENERGY_POINTS = 10_000
#     E_grid = np.linspace(*reaction.EB, NUM_ENERGY_POINTS)
#     prob_above_trunc_true = np.zeros((NUM_ENERGY_POINTS,), dtype=float)
#     prob_spingroup_priors = np.array(reaction.lvl_dens) / level_dens_tot
#     for prob_spingroup_prior, Gnm, nDOF in zip(prob_spingroup_priors, reaction.Gnm, reaction.nDOF):
#         prob_above_trunc_true += prob_spingroup_prior * porter_thomas_dist.isf(trunc*np.ones((NUM_ENERGY_POINTS,)), mean=Gnm(E_grid), df=nDOF, trunc=0.0)
#     # FIXME: NOW MAKE A CONTINUOUS FUNCTION WITH THIS!

#     # Observed fraction of widths above truncation width:
#     widths = abs(np.array(widths, dtype=float))
#     widths_are_above_trunc_obs = (widths >= trunc)
    
#     # FIXME: NOW MAKE A CONTINUOUS FUNCTION WITH THIS!
#     # FIXME: USE INFORMATION FROM LEVEL SPACING TO INFORM THE ESTIMATE!

#     # Returning false minus missing level-density function:
#     def level_dens_false_minus_missing(E):
#         frac_false_minus_missing_E = frac_widths_above_trunc_obs(E)/frac_widths_above_trunc_true(E)
#         level_dens_false_minus_missing_E = frac_false_minus_missing_E * level_dens_tot
#         return level_dens_false_minus_missing_E
#     return level_dens_false_minus_missing

# =================================================================================================
#    Empirical False Width Distribution:
# =================================================================================================

# def empirical_false_distribution(widths, prob_false:float, true_dist:rv_continuous, trunc:float=np.inf, err_thres:float=1e-5):
#     """
#     ...
#     """

#     widths = np.array(widths, dtype=float)
#     P_F = prob_false
#     P_T = 1.0 - P_F

#     P_W_gvn_T = true_dist.pdf(widths)

#     # Arbitrary guess for the false distribution to start.
#     # We guess the false distribution is the same as the true distribution
#     # which results in an uninformed false probability.
#     P_F_gvn_W = P_F*np.ones_like(widths)

#     # Iterating over false distribution estimation until convergence
#     for it in range(1000):
#         P_F_gvn_W_prev = P_F_gvn_W
#         P_W_gvn_F_func = limited_gaussian_kde(widths, weights=P_F_gvn_W_prev, trunc=trunc) # estimating false distribution
#         P_W_gvn_F = P_W_gvn_F_func.pdf(widths)
#         P_F_gvn_W = (P_F*P_W_gvn_F)/(P_F*P_W_gvn_F + P_T*P_W_gvn_T)
#         if all(abs(P_F_gvn_W - P_F_gvn_W_prev) < err_thres):
#             break
#     else:
#         raise RuntimeError('The false distribution never converged.')
#     return P_W_gvn_F_func, P_F_gvn_W

# def limited_gaussian_kde(widths, weights, trunc:float=np.inf):
#     """
#     ...
#     """

#     # if trunc != np.inf:
#     #     raise NotImplementedError('Limited distribution fitting has not been implemented yet.')

#     weights = copy(weights)
#     weights[abs(widths) > trunc] = 0.0
#     widths_sym  = np.concatenate((widths, -widths))
#     weights_sym = np.concatenate((weights, weights))
#     distribution = gaussian_kde(widths_sym, weights=weights_sym)
#     return distribution

def empirical_false_distribution(Gn, E_eval:float,
                                 prob_false_prior:float, reaction:Reaction,
                                 trunc:float=np.inf, err_thres:float=1e-5):
    """
    ...
    """

    Gn = np.array(Gn, dtype=float)
    P_F = prob_false_prior
    P_T = 1.0 - P_F

    # Calculating probabilities for widths given true:
    P_Gn_gvn_T = np.zeros_like(Gn, dtype=float)
    P_SG_gvn_T = np.array(reaction.lvl_dens) / np.sum(reaction.lvl_dens)
    P_W_gvn_SG_pdf = porter_thomas_dist.pdf
    for p_SG_gvn_T, Gnm, nDOF in zip(P_SG_gvn_T, reaction.Gnm, reaction.nDOF):
        P_Gn_gvn_T += p_SG_gvn_T * P_W_gvn_SG_pdf(Gn, mean=Gnm(E_eval), df=nDOF, trunc=0.0)

    # Arbitrary guess for the false distribution to start.
    # We guess the false distribution is the same as the true distribution
    # which results in an uninformed false probability.
    P_F_gvn_Gn = P_F*np.ones_like(Gn)

    # Iterating over false distribution estimation until convergence:
    for iter in range(1000):
        P_F_gvn_Gn_prev = P_F_gvn_Gn
        P_Gn_gvn_F_func = _fitting_false_width_dist(Gn, weights=P_F_gvn_Gn_prev, trunc=trunc) # estimating false distribution
        P_Gn_gvn_F = P_Gn_gvn_F_func.pdf(Gn)
        P_F_gvn_Gn = (P_F*P_Gn_gvn_F)/(P_F*P_Gn_gvn_F + P_T*P_Gn_gvn_T)
        if all(abs(P_F_gvn_Gn - P_F_gvn_Gn_prev) < err_thres):
            break
    else:
        raise RuntimeError('The false distribution never converged.')
    return P_Gn_gvn_F_func, P_F_gvn_Gn

def _fitting_false_width_dist(Gn, weights, trunc:float=np.inf):
    """
    ...
    """

    # If widths are beyond truncation threshold, set likelihood to zero:
    weights = copy(weights)
    weights[abs(Gn) > trunc] = 0.0
    # Make the distribution symmetric:
    Gn_sym      = np.concatenate((Gn, -Gn))
    weights_sym = np.concatenate((weights, weights))
    # Returning the false width distribution:
    false_width_dist = gaussian_kde(Gn_sym, weights=weights_sym)
    return false_width_dist

# NOTE: Energy-Dependent Version Below
# def empirical_false_distribution(energies, Gn,
#                                  prob_false_prior:Union[Callable, float], reaction:Reaction,
#                                  trunc:float=np.inf, bandwidth_scale:float=1e-2, err_thres:float=1e-5):
#     """
#     ...
#     """

#     Gn = np.array(Gn, dtype=float)
#     if callable(prob_false_prior):      P_F = prob_false_prior(energies)
#     else:                               P_F = prob_false_prior
#     P_T = 1.0 - P_F

#     # Calculating probabilities for widths given true:
#     P_Gn_gvn_T = np.zeros_like(Gn, dtype=float)
#     P_SG_gvn_T = np.array(reaction.lvl_dens) / np.sum(reaction.lvl_dens)
#     P_W_gvn_SG_pdf = porter_thomas_dist.pdf
#     for p_SG_gvn_T, Gnm, nDOF in zip(P_SG_gvn_T, reaction.Gnm, reaction.nDOF):
#         P_Gn_gvn_T += p_SG_gvn_T * P_W_gvn_SG_pdf(Gn, mean=Gnm(energies), df=nDOF, trunc=0.0)

#     # Arbitrary guess for the false distribution to start.
#     # We guess the false distribution is the same as the true distribution
#     # which results in an uninformed false probability.
#     P_F_gvn_Gn = P_F*np.ones_like(Gn)

#     # Iterating over false distribution estimation until convergence:
#     for iter in range(1000):
#         P_F_gvn_Gn_prev = P_F_gvn_Gn
#         P_Gn_gvn_F_func = _fitting_false_width_dist(energies, Gn, weights=P_F_gvn_Gn_prev, trunc=trunc, bandwidth_scale=bandwidth_scale) # estimating false distribution
#         P_Gn_gvn_F = P_Gn_gvn_F_func.pdf(Gn)
#         P_F_gvn_Gn = (P_F*P_Gn_gvn_F)/(P_F*P_Gn_gvn_F + P_T*P_Gn_gvn_T)
#         if all(abs(P_F_gvn_Gn - P_F_gvn_Gn_prev) < err_thres):
#             break
#     else:
#         raise RuntimeError('The false distribution never converged.')
#     return P_Gn_gvn_F_func, P_F_gvn_Gn

# def _fitting_false_width_dist(energies, Gn, weights, trunc:float=np.inf, bandwidth_scale:float=1e-2):
#     """
#     ...
#     """

#     # If widths are beyond truncation threshold, set likelihood to zero:
#     weights = copy(weights)
#     weights[abs(Gn) > trunc] = 0.0
#     # Make the distribution symmetric:
#     energies_sym = np.concatenate((energies, energies))
#     Gn_sym   = np.concatenate((Gn, -Gn))
#     weights_sym  = np.concatenate((weights, weights))
#     # Combine the data and scaling:
#     data_sym = np.concatenate((energies_sym[NA,:]*bandwidth_scale, Gn_sym[NA,:]), axis=0)
#     # Returning the false width dsitribution:
#     false_width_dist = gaussian_kde(data_sym, weights=weights_sym)
#     return false_width_dist



# #%% Testing:

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     from scipy.stats import norm

#     trunc = 0.5
#     PF = 0.5
#     PT = 1-PF
#     N = 100
#     NF = int(PF*N)
#     NT = N-NF
#     WF = np.random.normal(0.0, 0.1, size=(NF,))
#     WT = np.random.normal(0.0, 1.0, size=(NT,))
#     W = np.concatenate((WF, WT))
#     np.random.shuffle(W)
#     true_dist = norm(loc=0.0, scale=1.0)
#     false_dist = norm(loc=0.0, scale=0.1)
#     false_dist_est, false_probs = empirical_false_distribution(W, PF, true_dist, trunc=trunc)

#     X = np.linspace(-3, 3, 1000)
#     comb_dist = PT*true_dist.pdf(X) + PF*false_dist.pdf(X)
#     comb_dist_est = PT*true_dist.pdf(X) + PF*false_dist_est.pdf(X)
#     print('Finished Calculations!')

#     plt.figure(1)
#     plt.clf()
#     plt.hist(W, bins=25, density=True, label='Data')
#     plt.plot(X, true_dist.pdf(X), '-b', label='True Distribution')
#     plt.plot(X, false_dist.pdf(X), '-r', label='False Distribution')
#     plt.plot(X, false_dist_est.pdf(X), '--r', label='False Distribution Estimate')
#     plt.plot(X, comb_dist, '-k', label='Combined Distribution')
#     plt.plot(X, comb_dist_est, '--k', label='Combined Distribution Estimate')
#     plt.xlabel('Reduced Width ($\\sqrt{eV}$)', fontsize=16)
#     plt.ylabel('Probability Density', fontsize=16)
#     plt.legend(fontsize=10)
#     plt.tight_layout()
#     # plt.savefig('./false_width_dist')
#     plt.show()