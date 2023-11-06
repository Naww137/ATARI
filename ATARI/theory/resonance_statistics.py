import numpy as np
from scipy.linalg import eigvalsh_tridiagonal

from scipy.stats.distributions import chi2


def getD(xi,res_par_avg):    
    return res_par_avg['<D>']*2*np.sqrt(np.log(1/(1-xi))/np.pi)

def make_res_par_avg(J_ID, D_avg, Gn_avg, n_dof, Gg_avg, g_dof, print):

    res_par_avg = {'J_ID':J_ID, '<D>':D_avg, '<Gn>':Gn_avg, 'n_dof':n_dof, '<Gg>':Gg_avg, 'g_dof':g_dof}

    res_par_avg['D01']  = getD(0.01,res_par_avg)
    res_par_avg['D99']  = getD(0.99,res_par_avg)
    res_par_avg['Gn01'] = res_par_avg['<Gn>']*chi2.ppf(0.01, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    res_par_avg['Gn99'] = res_par_avg['<Gn>']*chi2.ppf(0.99, df=res_par_avg['n_dof'])/res_par_avg['n_dof']
    res_par_avg['Gg01'] = res_par_avg['<Gg>']*chi2.ppf(0.01, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    res_par_avg['Gg99'] = res_par_avg['<Gg>']*chi2.ppf(0.99, df=res_par_avg['g_dof'])/res_par_avg['g_dof']
    res_par_avg['Gt01'] = res_par_avg['Gn01'] + res_par_avg['Gg01']
    res_par_avg['Gt99'] = res_par_avg['Gn99'] + res_par_avg['Gg99']

    if print:
        print('D99  =',res_par_avg['D99'])
        print('Gn01 =',res_par_avg['Gn01'])
        print('Gn99 =',res_par_avg['Gn99'])
        print('Gg01 =',res_par_avg['Gg01'])
        print('Gg99 =',res_par_avg['Gg99'])
        print('Gt01 =',res_par_avg['Gt01'])
        print('Gt99 =',res_par_avg['Gt99'])

    return res_par_avg





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

def wigner_semicircle_CDF(x):
    """
    CDF of Wigner's semicircle law distribution.
    """
    return (x/np.pi) * np.sqrt(1.0 - x**2) + np.arcsin(x)/np.pi + 0.5

def sample_GE_eigs(num_eigs:int, beta:int=1,
                   rng=None, seed:int=None):
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
                       rng=None, seed:int=None):
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
    E_res = E_limits[0] + (num_res_tot * avg_level_spacing) * (wigner_semicircle_CDF(eigs) - wigner_semicircle_CDF(-1.0+margin))
    E_res = E_res[E_res < E_limits[1]]
    E_res = np.sort(E_res)
    return E_res

def wigner_PDF(x, avg_level_spacing):
    x = x/avg_level_spacing
    y = (np.pi/2) * x * np.exp(-np.pi*(x**2)/4)
    y = y/avg_level_spacing
    return y

# def generate_GOE(N):
#     A = np.random.default_rng().standard_normal(size=(N,N))/np.sqrt(2*N)
#     X = A + np.transpose(A)
#     return X

# def sample_resonance_levels(E0, N_levels, avg_level_spacing, method):
    
#     if method == 'invCDF':
#         level_spacing = avg_level_spacing*sample_wigner_invCDF(N_levels)
            
#     elif method == 'GOE':
#         level_spacing = []
#         for ilevel in range(N_levels):
#             X = generate_GOE(2)
#             eigenvalues = np.linalg.eigvals(X)
#             spacing = avg_level_spacing*abs(np.diff(eigenvalues))/(np.pi/2)
#             level_spacing.append(spacing.item())
#     else:
#         print('method for sampling resonance levels is not recognized')
#         os.sys.exit()
            
#     E0 = E0+avg_level_spacing*np.random.default_rng().uniform(low=0.0,high=1.0) # offset starting point so we are not just finding the distribution each time
#     levels = [E0+level_spacing[0]]
    
#     for ilevel in range(1,N_levels):
#         levels.append(levels[ilevel-1]+level_spacing[ilevel])
            
#     return levels, level_spacing



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


# def compare_pdf_to_samples(level_spacing_vector, avg_level_spacing, method):
    
#     fig = plt.figure(num=1,frameon=True); ax = fig.gca()
    
#     x = np.linspace(0,max(level_spacing_vector),10000)
#     plt.plot(x, wigner_PDF(x,avg_level_spacing), color='r', label='Wigner PDF', zorder=10)
        
#     if method == 'GOE':
#         print(); print('WARNING: ')
#         print('GOE sampling does not match wigner pdf exactly'); print()
#         plt.hist(level_spacing_vector, bins=75, density=True, ec='k', linewidth=0.75,color='cornflowerblue', zorder=2, label='GOE')

#     elif method == 'invCDF':
#         plt.hist(level_spacing_vector, bins=75, density=True, ec='k', linewidth=0.75,color='cornflowerblue', zorder=2, label='invCDF')
        
#     else:
#         print(); print('WARNING: ')
#         print('no appropriate method selected for pdf comparison'); print()
    
#     ax.set_facecolor('whitesmoke'); ax.grid(color='w', linestyle='-', linewidth=2, zorder=1, alpha=1)
#     ax.set_xlabel('Level Spacing'); ax.set_ylabel('Normalized Frequency'); plt.title('Distribution of Level Spacing Samples')
#     plt.legend()
#     plt.show(); plt.close()
    
#     return












# =====================================================================
# Resonance width sampling
# =====================================================================


def sample_chisquare(N_samples, DOF,
                     rng=None, seed=None):
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
    rng : np.random.Generator or None
        Numpy random number generator object. Default is None.
    seed : int or None
        Random number generator seed. Only used when rng is None. Default is None.

    Returns
    -------
    numpy.ndarray or float
        Array of i.i.d. samples from chi-squared distribution.

    See Also
    --------
    chisquare_PDF : Calculate probability density function for chi-squared distribution.

    Notes
    -----
    
    
    Examples
    --------
    >>> from theory import resonance_statistics
    >>> np.random.seed(7)
    >>> resonance_statistics.sample_chisquare(2,10)
    array([18.7081546 ,  7.46151704])
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

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
    >>> from theory import resonance_statistics
    >>> import scipy.stats as stats
    >>> resonance_statistics.chisquare_PDF(np.array([1.0, 2.5, 3.0]), 2, 1)
    array([0.30326533, 0.1432524 , 0.11156508])
    """
    x = x/avg_reduced_width_square
    y = chi2.pdf(x, DOF)
    y_norm = y/avg_reduced_width_square
    return y_norm


# def sample_resonance_widths(DOF, N_levels, avg_reduced_width_square):
    
#     reduced_widths_square = avg_reduced_width_square*sample_chisquare(N_levels, DOF)
#     partial_widths = 0  # add function with penetrability =2*P(E)*red_wid_sqr
    
#     return reduced_widths_square, partial_widths


def sample_RRR_widths(level_vector, avg_reduced_width_square, DOF,
                      rng=None, seed=None):
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

    reduced_widths_square = avg_reduced_width_square*sample_chisquare(len(level_vector), DOF, rng=rng)/DOF
    
    # vlads code:
    # res_par_avg['<Gg>']*np.random.chisquare(df=res_par_avg['g_dof'],size=sample_size)/res_par_avg['g_dof'] 
    
    return np.array(reduced_widths_square)


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
