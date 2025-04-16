import numpy as np
import pandas as pd
from scipy.linalg import eigvalsh_tridiagonal
from typing import Union, Optional
from scipy.stats.distributions import chi2

from ATARI.theory.distributions import wigner_dist, porter_thomas_dist, semicircle_dist
from ATARI.theory.scattering_params import gstat, k_wavenumber

def getD(xi,res_par_avg):    
    return res_par_avg['<D>']*2*np.sqrt(np.log(1/(1-xi))/np.pi)


def make_res_par_avg(Jpi, J_ID, D_avg, gn_avg, n_dof, gg_avg, g_dof):

    res_par_avg = {'Jpi':Jpi, 'J_ID':J_ID, '<D>':D_avg, '<gn2>':gn_avg, 'n_dof':n_dof, '<gg2>':gg_avg, 'g_dof':g_dof}

    quantiles = {}
    quantiles['D01']  = getD(0.01,res_par_avg)
    quantiles['D99']  = getD(0.99,res_par_avg)
    quantiles['gn01'] = res_par_avg['<gn2>']*chi2.ppf(0.01, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    quantiles['gn99'] = res_par_avg['<gn2>']*chi2.ppf(0.99, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    if (g_dof is None) or (g_dof == np.inf):
        quantiles['gg01'] = res_par_avg['<gg2>']
        quantiles['gg99'] = res_par_avg['<gg2>']
    else:
        quantiles['gg01'] = res_par_avg['<gg2>']*chi2.ppf(0.01, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
        quantiles['gg99'] = res_par_avg['<gg2>']*chi2.ppf(0.99, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    quantiles['gt01'] = quantiles['gn01'] + quantiles['gg01']
    quantiles['gt99'] = quantiles['gn99'] + quantiles['gg99']

    res_par_avg['quantiles'] = quantiles

    return res_par_avg

def expected_strength(particle_pair):
    """
    ...
    """

    Sns = {}
    for Jpi, spingroup in particle_pair.spin_groups.items():
        l = spingroup['Ls'][0]
        gJ = gstat(abs(Jpi), particle_pair.I, particle_pair.i)
        gn2m = spingroup['<gn2>']
        Dm = spingroup['<D>']
        Sn_Jpi = gJ/(2*l+1) * (gn2m*1e-3/Dm)
        Sns[Jpi] = Sn_Jpi
    return Sns

def find_external_levels(particle_pair, energy_bounds:tuple, return_reduced:bool=True):
    """
    From F. Frohner and Olivier Bouland, "Treatment of External Levels in Neutron Resonance
    Fitting: Applications to the Nonfissile Nuclide Cr-52".
    URL: https://doi.org/10.13182/NSE01-A2176
    """
    energy_bounds = (min(energy_bounds), max(energy_bounds))
    Eb = (energy_bounds[0] + energy_bounds[1])/2
    I  = energy_bounds[1] - energy_bounds[0]

    spingroups = particle_pair.spin_groups
    Ls    = [spingroup['Ls']    for spingroup in spingroups.values()]
    Jpis  = [spingroup['Jpi']    for spingroup in spingroups.values()]
    gn2ms = [spingroup['<gn2>'] for spingroup in spingroups.values()]
    Dms   = [spingroup['<D>']   for spingroup in spingroups.values()]
    
    gg2m = max([spingroup['<gg2>'] for spingroup in spingroups.values()])

    E_low  = Eb - (np.sqrt(3)/2) * I
    E_high = Eb + (np.sqrt(3)/2) * I

    s = 1e-3 * np.sum(gn2ms) * np.sum(1/np.array(Dms))
    gn2 = 1e3 * (3/2) * I * s

    Ggm = particle_pair.gg2_to_Gg(gg2m)
    # Gns = particle_pair.gn2_to_Gn(gn2, np.array([E_low,E_high]), 0)
    Gn_low  = particle_pair.gn2_to_Gn(gn2, np.array([E_low ]), 0)[0]
    Gn_high = particle_pair.gn2_to_Gn(gn2, np.array([E_high]), 0)[0]
    res_ext = pd.DataFrame({'E':[E_low,E_high], 'Gg':[Ggm,Ggm], 'Gn1':[Gn_low,Gn_high], 'J_ID':[1,1]})
    if return_reduced:
        res_ext['gg2'] = gg2m
        res_ext['gn2'] = gn2
        res_ext['Jpi'] = Jpis[0]
        res_ext['L']   = 0
    # Gnms_low = [particle_pair.gn2_to_Gn(gn2m, np.array([E_low]), l[0]) for gn2m, l in zip(gn2ms, Ls)]
    # str_low = np.sum(Gnms_low)*1e-3 * np.sum(1/np.array(Dms))
    # Gn_low  = 1e3 * (3/2) * I * str_low
    # Gnms_high = [particle_pair.gn2_to_Gn(gn2m, np.array([E_high]), l[0]) for gn2m, l in zip(gn2ms, Ls)]
    # str_high = np.sum(Gnms_high)*1e-3 * np.sum(1/np.array(Dms))
    # Gn_high = 1e3 * (3/2) * I * str_high
    # s = 1e-3 * np.sum(gn2ms) * np.sum(1/np.array(Dms))
    # a = particle_pair.ac
    # k_low  = k_wavenumber(E_low, particle_pair.M, particle_pair.m)
    # Gn_low  = 1e3 * (3/2) * I * (2*s*k_low*a)
    # k_high = k_wavenumber(E_high, particle_pair.M, particle_pair.m)
    # Gn_high = 1e3 * (3/2) * I * (2*s*k_high*a)
    return res_ext

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
    # wigner = wigner_dist(scale=avg_level_spacing, beta=beta)
    # return wigner.pdf(x)

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

    if DOF in (None, np.inf):
        reduced_widths_square = np.repeat(avg_reduced_width_square, N_levels)
    else:
        reduced_widths_square = porter_thomas_dist.rvs(mean=avg_reduced_width_square, df=int(DOF), trunc=trunc, size=N_levels, random_state=rng)
    sign = 2*rng.randint(0, 2, size=N_levels)-1
    return np.array(reduced_widths_square * sign, dtype=float)

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
# Dyson Mehta ∆3 Metric
# =====================================================================
    
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

# =====================================================================
# Missing Fraction Estimation
# =====================================================================

def fraction_missing_gn2(gn2_trunc:float, gn2m:float=1.0, dof:int=1):
    """
    Gives the fraction of missing resonances due to the truncation in reduced neutron width.

    Parameters
    ----------
    trunc : float
        The lower limit on the reduced neutron width.
    gn2m  : float
        The mean reduced neutron width. Default = 1.0.
    dof   : int
        The number of degrees of freedom for the chi-squared distribution.

    Returns
    -------
    fraction_missing : float
        The fraction of missing resonances within the spingroup.
    """
    fraction_missing = porter_thomas_dist(mean=gn2m, df=dof, trunc=0.0).cdf(gn2_trunc)
    return fraction_missing
  
# =====================================================================
# Log Likelihood
# =====================================================================

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

def log_likelihood(particle_pair, resonance_ladder, energy_range=None):
    """
    ...
    """

    if energy_range is not None:
        resonance_ladder = resonance_ladder[(resonance_ladder['E'] > energy_range[0]) & (resonance_ladder['E'] < energy_range[1])]

    log_likelihood = 0.0
    for jpi in resonance_ladder['Jpi'].unique():
        mean_parameters = particle_pair.spin_groups[jpi]
        resonance_ladder_sg = resonance_ladder[resonance_ladder['Jpi'] == jpi]
        E  = resonance_ladder_sg['E'  ].to_numpy()
        Gn = resonance_ladder_sg['Gn1'].to_numpy()
        L  = resonance_ladder_sg['L'].to_numpy()
        
        log_likelihood += wigner_LL(E , mean_parameters['<D>'])
        
        gn2  = particle_pair.Gn_to_gn2(Gn, E, L[0])
        gn2m = mean_parameters['<gn2>']
        log_likelihood += np.sum(-abs(gn2)/(2*gn2m) - 0.5*np.log(2*np.pi*gn2m)) # little gamma width LL
        # log_likelihood +=  width_LL(Gn, mean_parameters['<gn2>'], mean_parameters['n_dof']) # big gamma width LL
        
    return log_likelihood



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
