import numpy as np
from numpy import newaxis as NA
from scipy.stats import rv_continuous, gaussian_kde
from copy import copy

__doc__ = """
This file contains tools for estimating the distribution of false widths.
"""

def estimate_false_missing_fraction_from_widths(widths, width_dist:rv_continuous, trunc:float):
    """
    ...
    """

    widths = abs(np.array(widths, dtype=float))
    widths_above_trunc = widths[widths >= trunc]
    frac_widths_above_trunc = len(widths_above_trunc)/len(widths)
    frac_widths_above_trunc_true = width_dist.isf(trunc)
    frac_false_minus_missing = frac_widths_above_trunc/frac_widths_above_trunc_true
    return frac_false_minus_missing

def estimate_false_missing_fraction_from_spacing(num_res:int, lvl_dens_exp:float, ladder_size:float):
    """
    ...
    """

    num_res_exp = lvl_dens_exp * ladder_size
    frac_false_minus_missing = (num_res - num_res_exp) / num_res
    return frac_false_minus_missing

def empirical_false_distribution(widths, prob_false:float, true_dist:rv_continuous, trunc:float=np.inf, err_thres:float=1e-5):
    """
    ...
    """

    widths = np.array(widths, dtype=float)
    P_F = prob_false
    P_T = 1.0 - P_F

    P_W_gvn_T = true_dist.pdf(widths)

    # Arbitrary guess for the false distribution to start.
    # We guess the false distribution is the same as the true distribution
    # which results in an uninformed false probability.
    P_F_gvn_W = P_F*np.ones_like(widths)

    # Iterating over false distribution estimation until convergence
    for it in range(1000):
        P_F_gvn_W_prev = P_F_gvn_W
        P_W_gvn_F_func = limited_gaussian_kde(widths, weights=P_F_gvn_W_prev, trunc=trunc) # estimating false distribution
        P_W_gvn_F = P_W_gvn_F_func.pdf(widths)
        P_F_gvn_W = (P_F*P_W_gvn_F)/(P_F*P_W_gvn_F + P_T*P_W_gvn_T)
        if all(abs(P_F_gvn_W - P_F_gvn_W_prev) < err_thres):
            break
    else:
        raise RuntimeError('The false distribution never converged.')
    return P_W_gvn_F_func, P_F_gvn_W

def limited_gaussian_kde(widths, weights, trunc:float=np.inf):
    """
    ...
    """

    # if trunc != np.inf:
    #     raise NotImplementedError('Limited distribution fitting has not been implemented yet.')

    weights = copy(weights)
    weights[abs(widths) > trunc] = 0.0
    widths_sym  = np.concatenate((widths, -widths))
    weights_sym = np.concatenate((weights, weights))
    distribution = gaussian_kde(widths_sym, weights=weights_sym)
    return distribution

# #%% Energy-dependent false distribution:

# def estimate_false_missing_fraction(widths, width_dist:rv_continuous, trunc:float):
#     """
#     ...
#     """

#     widths = abs(np.array(widths, dtype=float))
#     widths_above_trunc = widths[widths >= trunc]
#     frac_widths_above_trunc = len(widths_above_trunc)/len(widths)
#     frac_widths_above_trunc_true = width_dist.isf(trunc)
#     frac_false_minus_missing = frac_widths_above_trunc/frac_widths_above_trunc_true
#     return frac_false_minus_missing

# def empirical_false_distribution(energies, widths, prob_false, true_dist:rv_continuous, trunc:float=np.inf, err_thres:float=1e-5):
#     """
#     ...
#     """

#     widths = np.array(widths, dtype=float)
#     P_F = prob_false(energies)
#     P_T = 1.0 - P_F

#     P_W_gvn_T = true_dist.pdf(widths)

#     # Arbitrary guess for the false distribution to start.
#     # We guess the false distribution is the same as the true distribution
#     # which results in an uninformed false probability.
#     P_F_gvn_W = P_F*np.ones_like(widths)

#     # Iterating over false distribution estimation until convergence
#     for it in range(1000):
#         P_F_gvn_W_prev = P_F_gvn_W
#         P_W_gvn_F_func = _fitting_false_dist(energies, widths, weights=P_F_gvn_W_prev, trunc=trunc) # estimating false distribution
#         P_W_gvn_F = P_W_gvn_F_func.pdf(widths)
#         P_F_gvn_W = (P_F*P_W_gvn_F)/(P_F*P_W_gvn_F + P_T*P_W_gvn_T)
#         if all(abs(P_F_gvn_W - P_F_gvn_W_prev) < err_thres):
#             break
#     else:
#         raise RuntimeError('The false distribution never converged.')
#     return P_W_gvn_F_func, P_F_gvn_W

# def _fitting_false_dist(energies, widths, weights, trunc:float=np.inf):
#     """
#     ...
#     """

#     weights = copy(weights)
#     weights[abs(widths) > trunc] = 0.0
#     energies_sym = np.concatenate((energies, energies))
#     widths_sym   = np.concatenate((widths, -widths))
#     weights_sym  = np.concatenate((weights, weights))
#     data_sym = np.concatenate((energies_sym[NA,:], widths_sym[NA,:]), axis=0)
#     distribution = gaussian_kde(data_sym, weights=weights_sym)
#     return distribution



#%% Testing:

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    trunc = 0.5
    PF = 0.5
    PT = 1-PF
    N = 100
    NF = int(PF*N)
    NT = N-NF
    WF = np.random.normal(0.0, 0.1, size=(NF,))
    WT = np.random.normal(0.0, 1.0, size=(NT,))
    W = np.concatenate((WF, WT))
    np.random.shuffle(W)
    true_dist = norm(loc=0.0, scale=1.0)
    false_dist = norm(loc=0.0, scale=0.1)
    false_dist_est, false_probs = empirical_false_distribution(W, PF, true_dist, trunc=trunc)

    X = np.linspace(-3, 3, 1000)
    comb_dist = PT*true_dist.pdf(X) + PF*false_dist.pdf(X)
    comb_dist_est = PT*true_dist.pdf(X) + PF*false_dist_est.pdf(X)
    print('Finished Calculations!')

    plt.figure(1)
    plt.clf()
    plt.hist(W, bins=25, density=True, label='Data')
    plt.plot(X, true_dist.pdf(X), '-b', label='True Distribution')
    plt.plot(X, false_dist.pdf(X), '-r', label='False Distribution')
    plt.plot(X, false_dist_est.pdf(X), '--r', label='False Distribution Estimate')
    plt.plot(X, comb_dist, '-k', label='Combined Distribution')
    plt.plot(X, comb_dist_est, '--k', label='Combined Distribution Estimate')
    plt.xlabel('Reduced Width ($\\sqrt{eV}$)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.legend(fontsize=10)
    plt.tight_layout()
    # plt.savefig('./false_width_dist')
    plt.show()