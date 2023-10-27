import numpy as np

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

def sample_wigner_invCDF(N_samples):
    """
    Sample the wigner distribution.

    This function simply samples from the wigner distribution using inverse
    CDF sampling and is used by other functions for generating resonance level spacing.

    Parameters
    ----------
    N_samples : int
        Number of samples and/or length of sample vector.

    Returns
    -------
    numpy.ndarray or float
        Array of i.i.d. samples from wigner distribution.

    Notes
    -----
    See preliminary methods for sampling resonance level spacing from GOE.
    
    Examples
    --------
    >>> from sample_resparm import sample_levels
    >>> np.random.seed(7)
    >>> sample_levels.sample_wigner_invCDF(2,10)
    array([1.7214878 , 1.31941784])
    """
    samples = np.sqrt(-4/np.pi*np.log(np.random.default_rng().uniform(low=0.0,high=1.0,size=N_samples)))
    #samples = np.sqrt(-4*np.log(np.random.default_rng().uniform(low=0.0,high=1.0,size=N_samples)))      # remove the pi terms to match GOE
    if N_samples == 1:
        samples = np.ndarray.item(samples)
    return samples

def generate_GOE(N):
    A = np.random.default_rng().standard_normal(size=(N,N))/np.sqrt(2*N)
    X = A + np.transpose(A)
    return X

def wigner_PDF(x, avg_level_spacing):
    x = x/avg_level_spacing
    y = (np.pi/2) * x * np.exp(-np.pi*(x**2)/4)
    #y = (1/2) * x * np.exp(-(x**2)/4)   # remove the pi terms to match GOE
    y = y/avg_level_spacing
    return y


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



def sample_RRR_levels(E_range, avg_level_spacing):
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

    Returns
    -------
    levels : numpy.ndarray
        Array of resonance energy levels.
    spacings : numpy.ndarray
        Array of i.i.d. resonance level spacing samples.

    See Also
    --------
    sample_resonance_levels : Samples a specified number of resonance energy levels.

    Notes
    -----
    
    
    Examples
    --------
    >>> from sample_resparm import sample_levels
    >>> np.random.seed(7)
    >>> sample_levels.sample_RRR_levels([0.1,10], 2)
    ([array([6.31322239]),
      array([6.56223504]),
      array([8.65279185]),
      array([10.27692974])],
     [array([6.21322239]),
      array([0.24901265]),
      array([2.09055681]),
      array([1.62413789])])
    """
    # randomly offset starting point so we are not just finding the distribution fixed to this point with ML
    # is this necessary?
    # sample a ladder 5-6 average level spacings before and 5-6 average level spacings after window
    E0 = min(E_range)-avg_level_spacing*np.random.default_rng().uniform(low=5.0,high=6.0)     
    E_end = max(E_range)+avg_level_spacing*np.random.default_rng().uniform(low=5.0,high=6.0)   
    
    levels = []; spacings = []
    spacing = avg_level_spacing*sample_wigner_invCDF(1)
    spacings.append(spacing)
    level = E0+spacing
    
    while level < E_end:
        levels.append(level)
        spacing = avg_level_spacing*sample_wigner_invCDF(1)
        spacings.append(spacing)
        level = levels[-1] + spacing

    levels = list(filter(lambda l: l<max(E_range) and l>min(E_range), levels))
            
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
    y = chi2.pdf(x, DOF)
    y_norm = y/avg_reduced_width_square
    return y_norm


# def sample_resonance_widths(DOF, N_levels, avg_reduced_width_square):
    
#     reduced_widths_square = avg_reduced_width_square*sample_chisquare(N_levels, DOF)
#     partial_widths = 0  # add function with penetrability =2*P(E)*red_wid_sqr
    
#     return reduced_widths_square, partial_widths


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
    reduced_widths_square = avg_reduced_width_square*sample_chisquare(len(level_vector), DOF)/DOF
    
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
