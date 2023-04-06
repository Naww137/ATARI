#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:08:47 2022

@author: noahwalton
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
from syndat import scattering_theory


def sample_chisquare(N_samples, DOF):
    """
    Sample the chi-squared distribution.

    This function simply samples from the chi-square distribution and is used
    by other functions for generating reduced resonance width samples.

    Parameters
    ----------
    N_samples : int
        Number of samples and/or length of sample vector.
    DOF : float
        Degrees of freedom for the chi-squared distribution.

    Returns
    -------
    numpy.ndarray or float
        Array of i.i.d. samples from chi-squared distribution.

    See Also
    --------
    chisquare_PDF : Calculate probability density function for chi-squared distribution.

    Notes
    -----
    A more robust/recomended way to set seed for examples would be to create a
    random number generator and pass it to the function. The example included
    in this documentation sets a global random number seed. See this article
    for more information on why this could be improved:
    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f.
    
    Examples
    --------
    >>> from sample_resparm import sample_widths
    >>> np.random.seed(7)
    >>> sample_widths.sample_chisquare(2,10)
    array([18.7081546 ,  7.46151704])
    """
    samples = np.random.chisquare(DOF, size=N_samples)
    # if N_samples == 1:
    #     samples = samples.item()
    return samples
    
def chisquare_PDF(x, DOF, avg_reduced_width_square):
    """
    Calculate probability density function for chi-squared distribution.

    This function simply houses the probaility density function for the 
    chi-squared distribution and allows for a rescaling factor to be applied.
    The rescaling factor represents the average resonance width value s.t.
    this PDF will represent the distribution of widths for a specific isotope.

    Parameters
    ----------
    x : numpy.ndarray
        Values at which to evaluation the PDF.
    DOF : float or int
        Degrees of freedom for the chi-squared distribution.
    avg_reduced_width_square : float or int
        Re-scaling factor for isotope specific distribution.

    Returns
    -------
    numpy.ndarray
        Pointwise function evaluated at x, given DOF and rescaling factor.

    See Also
    --------
    sample_chisquare : Sample the chi-squared distribution.
    
    Examples
    --------
    >>> from sample_resparm import sample_widths
    >>> import scipy.stats as stats
    >>> sample_widths.chisquare_PDF(np.array([1.0, 2.5, 3.0]), 2, 1)
    array([0.30326533, 0.1432524 , 0.11156508])
    """
    x = x/avg_reduced_width_square
    y = stats.chi2.pdf(x, DOF)
    y_norm = y/avg_reduced_width_square
    return y_norm

def sample_resonance_widths(DOF, N_levels, avg_reduced_width_square):
    
    reduced_widths_square = avg_reduced_width_square*sample_chisquare(N_levels, DOF)
    partial_widths = 0  # add function with penetrability =2*P(E)*red_wid_sqr
    
    return reduced_widths_square, partial_widths


def sample_RRR_widths(level_vector, avg_reduced_width_square, DOF):
    """
    Samples resonance widths corresponding to a vector of resonance energies.

    This function uses the porter thomas distribution to sample a vector of
    reduced width amplitudes (gn^2) corresponding to a vector of resonance level energies.

    Parameters
    ----------
    level_vector : numpy.ndarray
        Ladder of resonance energy levels.
    avg_reduced_width_square : float or int
        Average value for the reduced width for rescale of the PT distribution.
    DOF : float or int
        Degrees of freedom applied to the PT distiribution (chi-square).

    Returns
    -------
    reduced_widths_square : numpy.ndarray
        Array of reduced widths squared, this is what is sampled directly.
    """
    reduced_widths_square = avg_reduced_width_square*sample_chisquare(len(level_vector), DOF)
    
    return np.array(reduced_widths_square)


def compare_pdf_to_samples(reduced_widths_square_vector, avg_reduced_width_square, dof):
    """
    Compare samples to pdf (re-scaled).

    This function plots a histogram of the parameter samples with an
    overlaid probability density function of the distribution from which the 
    samples were drawn. In the limit that sample size approaches infinity, the
    PDF and histogram should line up exactly, acting as visual verification 
    for the sampling methods.

    Parameters
    ----------
    reduced_widths_square_vector : numpy.ndarray
        Array of reduced widths/decay amplitudes squared (little gamma squared).
    avg_reduced_width_square : float or int
        Isotope/spin group specific average reduced width/decay amplitude squared.
    dof : float or int
        Degrees of freedom for the chi-squared distribution.

    Returns
    -------
    
    Notes
    -----
    Showing the example with a plot included is not working for this docstring.
    """
    fig = plt.figure(num=1,frameon=True); ax = fig.gca()
    
    x = np.linspace(0,max(reduced_widths_square_vector),10000)
    plt.plot(x, chisquare_PDF(x,dof,avg_reduced_width_square), color='r', label='$\chi^2$ PDF', zorder=10)
        
    plt.hist(reduced_widths_square_vector, bins=75, density=True, ec='k', linewidth=0.75,color='cornflowerblue', zorder=2, label='samples')
    
    ax.set_facecolor('whitesmoke'); ax.grid(color='w', linestyle='-', linewidth=2, zorder=1, alpha=1)
    ax.set_xlabel('Reduced Widths Squared ($\gamma^2$)'); ax.set_ylabel('Normalized Frequency'); plt.title('Reduced Widths Squared ($\gamma^2$)')
    plt.legend()
    plt.show(); plt.close()
    
    return
