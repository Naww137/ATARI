import numpy as np
from scipy.linalg import eigvalsh_tridiagonal
from typing import Union, Optional
from scipy.stats.distributions import chi2

from ATARI.theory.distributions import wigner_dist, porter_thomas_dist, semicircle_dist

def getD(xi,res_par_avg):    
    return res_par_avg['<D>']*2*np.sqrt(np.log(1/(1-xi))/np.pi)


def make_res_par_avg(Jpi, J_ID, D_avg, gn_avg, n_dof, gg_avg, g_dof):

    res_par_avg = {'Jpi':Jpi, 'J_ID':J_ID, '<D>':D_avg, '<gn2>':gn_avg, 'n_dof':n_dof, '<gg2>':gg_avg, 'g_dof':g_dof}

    quantiles = {}
    quantiles['D01']  = getD(0.01,res_par_avg)
    quantiles['D99']  = getD(0.99,res_par_avg)
    quantiles['gn01'] = res_par_avg['<gn2>']*chi2.ppf(0.01, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    quantiles['gn99'] = res_par_avg['<gn2>']*chi2.ppf(0.99, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    quantiles['gg01'] = res_par_avg['<gg2>']*chi2.ppf(0.01, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    quantiles['gg99'] = res_par_avg['<gg2>']*chi2.ppf(0.99, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    quantiles['gt01'] = quantiles['gn01'] + quantiles['gg01']
    quantiles['gt99'] = quantiles['gn99'] + quantiles['gg99']

    res_par_avg['quantiles'] = quantiles

    return res_par_avg

def wigner_LL(resonance_levels  : Union[np.ndarray, list], 
              average_spacing   : float                    ) -> float:
    resonance_levels = np.sort(list(resonance_levels))
    Di = np.diff(resonance_levels)
    probs = wigner_PDF(Di, average_spacing)
    return np.sum(np.log(probs))


def width_LL(resonance_widths   : Union[np.ndarray, list],
                  average_width : float,
                  dof           :float                      ) -> float:
    probs = chisquare_PDF(resonance_widths, dof, average_width)
    return np.sum(np.log(probs))


# =====================================================================
# Resonance level sampling
# =====================================================================

def sample_wigner_invCDF(N_samples:int,
                         rng=None, seed=None):
    """
    Sample the wigner distribution using inverse CDF sampling.

    This function simply samples from the wigner distribution using inverse
    CDF sampling and is used by other functions for generating resonance level spacing.

    Parameters
    ----------
    N_samples : int
        Number of samples and/or length of sample vector.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    numpy.ndarray or float
        Array of i.i.d. samples from wigner distribution.

    Notes
    -----
    
    
    Examples
    --------
    >>> from theory import resonance_statistics
    >>> np.random.seed(7)
    >>> resonance_statistics.sample_wigner_invCDF(2,10)
    array([1.7214878 , 1.31941784])
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    samples = np.sqrt(-4/np.pi*np.log(rng.uniform(low=0.0,high=1.0,size=N_samples)))
    if N_samples == 1:
        samples = np.ndarray.item(samples)
    return samples

def sample_NNE_energies(E_range, avg_level_spacing:float,
                        rng=None, seed=None):
    """
    Sample resonance energies for the ladder using inverse CDF sampling.

    Parameters
    ----------
    E_range : array-like
        The energy range for sampling.
    avg_level_spacing : float
        The mean level spacing.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    np.ndarray
        Array of resonance energies sampled from the Wigner distribution.

    Notes
    -----
    See sample_GE_energies for sampling energies from Gaussian Ensembles.
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed
    
    E_limits = (min(E_range), max(E_range))
    num_res_est = round((E_limits[1] - E_limits[0]) / avg_level_spacing)
    num_res_tot = num_res_est + round(3.3252*np.sqrt(num_res_est)) # there will only be more resonances than this once in 1e10 samples.
    
    level_spacings = np.zeros((num_res_tot+1,))
    level_spacings[0] = avg_level_spacing * np.sqrt(2/np.pi) * np.abs(rng.normal())
    level_spacings[1:] = avg_level_spacing * sample_wigner_invCDF(num_res_tot, rng=rng)
    res_E = E_limits[0] + np.cumsum(level_spacings)
    res_E = res_E[res_E < E_limits[1]]
    return res_E

def sample_GE_eigs(num_eigs:int, beta:int=1,
                   rng=None, seed:Optional[int]=None):
    """
    Samples the eigenvalues of n by n Gaussian Ensemble random matrices efficiently using the
    tridiagonal representation. The time complexity of this method is O( n**2 ) using scipy's
    eigvalsh_tridiagonal function. However, there exist O( n log(n) ) algorithms that have more
    low n cost and higher error. Unfortunately, no implementation of that algorithm has been made
    in Python.

    Parameters
    ----------
    num_eigs : int
        The rank of the random matrix. This is also the number of eigenvalues to sample.
    beta : 1, 2, or 4
        The ensemble to consider, corresponding to GOE, GUE, and GSE respectively.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    np.ndarray
        The eigenvalues of the random matrix.

    Notes
    -----
    The theory for the tridiagonal eigenvalue sampling can be found in
    https://people.math.wisc.edu/~valko/courses/833/2009f/lec_8_9.pdf.
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    # Tridiagonal Matrix Coefficients:
    # Using Householder transformations (orthogonal transformations), we can
    # transform GOE-sampled matrices into a tridiagonal symmetric matrix.
    # Instead of performing the transformation, we can sample the transformed matrix directly.
    # Let `a` be the central diagonal elements and `b` be the offdiagonal diagonal elements.
    # We can define the following:
    a = np.sqrt(2) * rng.normal(size=(num_eigs,))
    b = np.sqrt(rng.chisquare(beta*np.arange(1,num_eigs)))

    # Now we sample the eigenvalues of the tridiagonal symmetric matrix:
    eigs = eigvalsh_tridiagonal(a, b)
    eigs /= np.sqrt(beta)
    eigs.sort()
    return eigs

def sample_GE_energies(E_range, avg_level_spacing:float=1.0, beta:int=1,
                       rng=None, seed:Optional[int]=None):
    """
    Samples GOE (β = 1), GUE (β = 2), or GSE (β = 4) resonance energies within a given energy
    range, `EB` and with a specified mean level-density, `freq`.

    Parameters
    ----------
    E_range : array-like
        The energy range for sampling.
    avg_level_spacing : float
        The mean level spacing.
    beta : 1, 2, or 4
        The ensemble parameter, where β = 1 is GOE, β = 2 is GUE, and β = 4 is GSE.
        Default is β = 1.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    np.ndarray
        The sampled resonance energies.

    Notes
    -----
    The semicircle law inverse CDF method was found by David Brown. The original source code can be
    found at https://github.com/LLNL/fudge/blob/master/brownies/BNL/restools/level_generator.py.
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    margin = 0.1 # a margin of safety where we consider the GOE samples to properly follow the semicircle law. This removes tails that diverge from the semicircle law.
    E_limits = (min(E_range), max(E_range))
    num_res_est = (E_limits[1]-E_limits[0]) / avg_level_spacing # estimate number of resonances
    num_res_tot = round((1 + 2*margin) * num_res_est) # number of resonances sampled (estimate + buffer)

    eigs = sample_GE_eigs(num_res_tot, beta=beta, rng=rng)
    eigs /= 2*np.sqrt(num_res_tot)
    eigs = eigs[eigs > -1.0+margin]
    eigs = eigs[eigs <  1.0-margin]

    # Using semicircle law CDF to make the resonances uniformly spaced:
    # Source: https://github.com/LLNL/fudge/blob/master/brownies/BNL/restools/level_generator.py
    E_res = E_limits[0] + (num_res_tot * avg_level_spacing) * (semicircle_dist.cdf(eigs) - semicircle_dist.cdf(-1.0+margin))
    E_res = E_res[E_res < E_limits[1]]
    E_res = np.sort(E_res)
    return E_res

def sample_RRR_levels(E_range, avg_level_spacing:float, ensemble:str='NNE',
                      rng=None, seed=None):
    """
    Sample the resonance energy levels.

    This function samples the wigner distribution using invCDF method in order 
    to get a ladder of resonance energy levels within a specified range. The energy range given
    is expanded by 5-6 times the average level spacing, a resonance ladder is sampled over that, 
    then the ladder is filtered to the energy range of interest.

    Parameters
    ----------
    E_range : array-like
        Array energies in RRR, only need min/max.
    avg_level_spacing : float
        Average level spacing value to scale wigner distribution.
    ensemble : 'NNE', 'GOE', 'GUE', 'GSE', or 'Poisson'
        The level-spacing distribution to sample from:
        NNE : Nearest Neighbor Ensemble
        GOE : Gaussian Orthogonal Ensemble
        GUE : Gaussian Unitary Ensemble
        GSE : Gaussian Symplectic Ensemble
        Poisson : Poisson Ensemble
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    levels : numpy.ndarray
        Array of resonance energy levels.
    spacings : numpy.ndarray
        Array of resonance level spacing samples.

    See Also
    --------
    sample_resonance_levels : Samples a specified number of resonance energy levels.

    Notes
    -----
    
    
    Examples
    --------
    >>> from theory import resonance_statistics
    >>> np.random.seed(7)
    >>> resonance_statistics.sample_RRR_levels([0.1,10], 2)
    ([array([6.31322239]),
      array([6.56223504]),
      array([8.65279185]),
      array([10.27692974])],
     [array([6.21322239]),
      array([0.24901265]),
      array([2.09055681]),
      array([1.62413789])])
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    if ensemble == 'NNE': # Nearest Neighbor Ensemble
        levels = sample_NNE_energies(E_range, avg_level_spacing, rng=rng)
    elif ensemble == 'GOE': # Gaussian Orthogonal Ensemble
        levels = sample_GE_energies(E_range, avg_level_spacing, beta=1, rng=rng)
    elif ensemble == 'GUE': # Gaussian Unitary Ensemble
        levels = sample_GE_energies(E_range, avg_level_spacing, beta=2, rng=rng)
    elif ensemble == 'GSE': # Gaussian Symplectic Ensemble
        levels = sample_GE_energies(E_range, avg_level_spacing, beta=4, rng=rng)
    elif ensemble == 'Poisson': # Poisson Ensemble (i.i.d. resonances)
        E_limits = (min(E_range), max(E_range))
        num_samples = rng.poisson((E_limits[1] - E_limits[0]) / avg_level_spacing)
        levels = rng.uniform(*E_limits, size=num_samples)
    else:
        raise NotImplementedError(f'The {ensemble} ensemble has not been implemented yet.')
            
    spacings = np.diff(levels) # I do not think this output is used anywhere
    return levels, spacings

def wigner_PDF(x, avg_level_spacing:float, beta:int=1):
    return wigner_dist.pdf(x, scale=avg_level_spacing, beta=beta)

# =====================================================================
# Resonance width sampling
# =====================================================================

def sample_RRR_widths(N_levels, 
                      avg_reduced_width_square, 
                      DOF:int=1, trunc:float=0.0,
                      rng=None, seed=None):
    """
    Samples resonance widths corresponding to a vector of resonance energies.

    This function uses the porter thomas distribution to sample a vector of
    reduced widths (gn^2) corresponding to a vector of resonance level energies.

    Parameters
    ----------
    N_levels : numpy.ndarray
        Ladder of resonance energy levels.
    avg_reduced_width_square : float or int
        Average value for the reduced width for rescale of the PT distribution.
    DOF : int
        Degrees of freedom applied to the PT distiribution (chi-square).
    trunc : float
        All reduced widths below this value are ignored. Default = 0.0.
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    reduced_widths_square : numpy.ndarray
        Array of reduced widths squared, this is what is sampled directly.
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    reduced_widths_square = porter_thomas_dist.rvs(mean=avg_reduced_width_square, df=int(DOF), trunc=trunc, size=N_levels, random_state=rng)
    return np.array(reduced_widths_square)

def chisquare_PDF(x, DOF:int=1, avg_reduced_width_square:float=1.0, trunc:float=0.0):
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
    DOF : int
        Degrees of freedom for the chi-squared distribution.
    avg_reduced_width_square : float or int
        Re-scaling factor for isotope specific distribution.
    trunc : float
        All reduced widths below this value are ignored. Default = 0.0.

    Returns
    -------
    numpy.ndarray
        Pointwise function evaluated at x, given DOF and rescaling factor.

    See Also
    --------
    sample_RRR_widths : Sample the widths according to Porter-Thomas distribution.
    
    Examples
    --------
    >>> from theory import resonance_statistics
    >>> import scipy.stats as stats
    >>> resonance_statistics.chisquare_PDF(np.array([1.0, 2.5, 3.0]), 2, 1)
    array([0.30326533, 0.1432524 , 0.11156508])
    """
    return porter_thomas_dist.pdf(x=x, mean=avg_reduced_width_square, df=DOF, trunc=trunc)

# =====================================================================
# GOE, GUE, and GSE distributions
# =====================================================================

def general_wigner_pdf(x, mean_level_spacing:float=1.0, beta:int=1):
    """
    Wigner Distribution PDF for GOE, GUE, and GSE.

    Parameters
    ----------
    x                  : float or float array
        The nearest level-spacing.
    mean_level_spacing : float
        The mean level spacing of the distribution.
    beta               : 1, 2, or 4
        The parameter that determines the assumed ensemble. For GOE, GUE, and GSE, `beta` = 1, 2,
        or 4, respectively. The default is 1 (GOE).

    Returns
    -------
    prob_dens : float or float array
        The probability density for the distribution evaluated at each level-spacing.
    """
    if   beta == 1:
        coef = np.pi/(4*mean_level_spacing**2)
        return 2*coef * x * np.exp(-coef*x**2)
    elif beta == 2:
        coef1 = 4/(np.pi*mean_level_spacing**2)
        coef2 = coef1 * (8/(np.pi*mean_level_spacing))
        return coef2 * x**2 * np.exp(-coef1*x**2)
    elif beta == 4:
        coef1 = 64/(9*np.pi*mean_level_spacing**2)
        coef2 = 262144/(729*np.pi**3*mean_level_spacing**5)
        return coef2 * x**4 * np.exp(-coef1*x**2)
    else:
        raise NotImplementedError(f'beta = {beta} has not been implemented. Choose beta = 1, 2, or 4.')
    
def level_spacing_ratio_pdf(ratio:float, beta:int=1):
    """
    This function returns the probability density on the ensemble's nearest level-spacing ratio,
    evaluated at `ratio`. The ensemble can be chosen from GOE, GUE, and GSE for `beta` = 1, 2, or
    4, respectively.

    Source: https://arxiv.org/pdf/1806.05958.pdf (Eq. 1)

    Parameters
    ----------
    ratio : float or float array
        The nearest level-spacing ratio(s).
    beta  : 1, 2, or 4
        The parameter that determines the assumed ensemble. For GOE, GUE, and GSE, `beta` = 1, 2,
        or 4, respectively. The default is 1 (GOE).

    Returns
    -------
    level_spacing_ratio_pdf : float or float array
        The probability density (or densities) evaluated at the the provided level-spacing
        ratio(s).
    """
    if   beta == 1:     C_beta = 27/8
    elif beta == 2:     C_beta = 81*np.sqrt(3)/(4*np.pi)
    elif beta == 4:     C_beta = 729*np.sqrt(3)/(4*np.pi)
    else:               raise ValueError('"beta" can only be 1, 2, or 4.')
    level_spacing_ratio_pdf = C_beta * (ratio+ratio**2)**beta / (1+ratio+ratio**2)**(1+(3/2)*beta)
    return level_spacing_ratio_pdf
    
def dyson_mehta_delta_3(E, EB:tuple):
    """
    Finds the Dyson-Mehta ∆3 metric for the given data.

    Source: https://arxiv.org/pdf/2011.04633.pdf (Eq. 21 & 22)

    Parameters
    ----------
    E  : float, array-like
        The recorded resonance energies.
    EB : float [2]
        The lower and upper energies for the resonance ladder.

    Returns
    -------
    delta3 : float
        The Dyson-Mehta ∆3 metric.
    """

    E = np.sort(E) # sort energies if not already sorted
    z = (E-EB[0])/(EB[1]-EB[0]) # renormalize energies
    s1 = np.sum(z)
    s2 = np.sum(z**2)
    a = np.arange( len(z)-1, -1, -1 )
    s3 = np.sum((2*a+1)*z)
    delta3 = 6*s1*s2 - 4*s1**2 - 3*s2**2 + s3
    return delta3

def dyson_mehta_delta_3_predict(L:int, ensemble:str='GOE'):
    """
    A function that predicts the value of the Dyson-Mehta ∆3 metric based on the number of
    observed resonances and type of ensemble.

    Source: https://www.osti.gov/servlets/purl/1478482 (Eq. 31 & 32 & 33)

    Parameters
    ----------
    L        : int
        The expected number of resonances.
    ensemble : 'GOE', 'Poisson', or 'picket'
        The ensemble to assumed under the calculation of the Dyson-Mehta ∆3 metric.

    Returns
    -------
    delta_3 : float
        The prediction on the Dyson-Mehta ∆3 metric.
    """

    if   ensemble.lower() == 'goe':
        delta_3 = np.pi**(-2) * (np.log(L) - 0.0687)
    elif ensemble.lower() == 'poisson':
        delta_3 = L/15
    elif ensemble.lower() == 'picket':
        delta_3 = 1/12   # "picket" refers to "picket fence", where the levels are uniformly distributed
    else:
        raise ValueError(f'Unknown ensemble, {ensemble}. Please choose from "GOE", "Poisson" or "picket".')
    return delta_3

# def compare_pdf_to_samples(reduced_widths_square_vector, avg_reduced_width_square, dof):
#     """
#     Compare samples to pdf (re-scaled).

#     This function plots a histogram of the parameter samples with an
#     overlaid probability density function of the distribution from which the 
#     samples were drawn. In the limit that sample size approaches infinity, the
#     PDF and histogram should line up exactly, acting as visual verification 
#     for the sampling methods.

#     Parameters
#     ----------
#     reduced_widths_square_vector : numpy.ndarray
#         Array of reduced widths/decay amplitudes squared (little gamma squared).
#     avg_reduced_width_square : float or int
#         Isotope/spin group specific average reduced width/decay amplitude squared.
#     dof : float or int
#         Degrees of freedom for the chi-squared distribution.

#     Returns
#     -------
    
#     Notes
#     -----
#     Showing the example with a plot included is not working for this docstring.
#     """
#     fig = plt.figure(num=1,frameon=True); ax = fig.gca()
    
#     x = np.linspace(0,max(reduced_widths_square_vector),10000)
#     plt.plot(x, chisquare_PDF(x,dof,avg_reduced_width_square), color='r', label='$\chi^2$ PDF', zorder=10)
        
#     plt.hist(reduced_widths_square_vector, bins=75, density=True, ec='k', linewidth=0.75,color='cornflowerblue', zorder=2, label='samples')
    
#     ax.set_facecolor('whitesmoke'); ax.grid(color='w', linestyle='-', linewidth=2, zorder=1, alpha=1)
#     ax.set_xlabel('Reduced Widths Squared ($\gamma^2$)'); ax.set_ylabel('Normalized Frequency'); plt.title('Reduced Widths Squared ($\gamma^2$)')
#     plt.legend()
#     plt.show(); plt.close()
    
#     return
