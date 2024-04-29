import numpy as np
from scipy.stats import chisquare

import os
import matplotlib.pyplot as plt

DIRECTORY = os.path.dirname(__file__)

__doc__ = """
A file containing utility functions for unit tests.
"""

def chi2_test(dist, data, num_bins:int,
              test_obj, threshold:float,
              quantity_name:str, dist_name:str=None,
              p_or_chi2='p'):
    """
    Performs a Pearson's Chi-squared test with the provided distribution and data.
    """

    data_len = len(data)
    quantiles = np.linspace(0.0, 1.0, num_bins+1)
    with np.errstate(divide='ignore'):
        edges = dist.ppf(quantiles)
    obs_counts, edges = np.histogram(data, edges)
    exp_counts = (data_len / num_bins) * np.ones((num_bins,))
    chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    chi2_bar = chi2 / num_bins

    if   p_or_chi2 == 'p':
        if p > threshold:
            return chi2_bar, p
    elif p_or_chi2 == 'chi2':
        if chi2_bar < threshold:
            return chi2_bar, p
    else:
        raise ValueError('"p_or_chi2" can only be "p" or "chi2".')

    sample_distribution_error_plots(dist.pdf, data, edges, f'{test_obj.__class__.__name__}_error.png', quantity_name)
    if dist_name is None:
        dist_name = 'the expected distribution'
    message = f'\nThe {quantity_name} samples do not follow {dist_name} according to the null hypothesis.\n' \
            + f'Calculated χ² / dof = {chi2_bar:.5f}; p = {p:.5f}\n'
    test_obj.assertTrue(False, message)
    return chi2_bar, p

def chi2_uniform_test(data, num_bins:int):
    """
    Performs a Pearson's Chi-squared test on the data, assuming that the underlying distribution
    is uniform.
    """

    obs_counts, bin_edges = np.histogram(data, num_bins)
    exp_counts = (len(data)/num_bins) * np.ones((num_bins,))
    chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
    chi2_bar = chi2 / num_bins
    return chi2_bar, p

def sample_distribution_error_plots(pdf, data, bins, image_name, quantity_name):
    """
    Creates a plot of the histogram and distribution for analysis after an error is thrown.
    """

    X = np.linspace(min(data), max(data), 1000)
    Y = pdf(X)
    plt.figure(1)
    plt.clf()
    plt.hist(data, bins, density=True)
    plt.plot(X, Y, '-k')
    plt.ylabel('Probability Density', fontsize=16)
    plt.xlabel(f'{quantity_name}'.title(), fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{DIRECTORY}/error_plots/{image_name}')