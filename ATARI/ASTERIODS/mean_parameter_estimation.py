import numpy as np
from scipy.optimize import curve_fit

from ATARI.theory.distributions import porter_thomas_dist, fraction_missing_gn2
from ATARI.ModelData.spingroups import HalfInt

__doc__ = """
This module compiles mean parameter estimation methods.
"""


"""
Methods for mean-level spacing estimation:
    1. Bethe formula for level-densities.
    2. Ladder size over number of levels.
    3. Linear regression on the cumulative level function.
  / 4. Fitting Wigner PDF.
  / 5. Fitting Wigner CDF.

Methods for mean width estimation:
    1. Mean of reduced widths.
    2. Porter-Thomas CDF regression.
  / 3. Porter-Thomas PDF regression.
"""


# =================================================================================================
#    Mean Level-Spacing Estimation:
# =================================================================================================

def mean_spacing_Bethe(J:HalfInt, A:int, E:float, E0:float=0.0):
    """
    Finds the mean level-spacing using the Bethe formula.

    Parameters
    ----------
    J  : HalfInt
        Total angular momentum.
    A  : int
        Atomic mass number.
    E  : float
        The energy to find the mean level-spacing.
    E0 : float
        Threshold energy for the reaction channel.

    Returns
    -------
    mean_lvl_spacing : float
        The mean level-spacing.
    """

    a = A / 11 # 1 / MeV
    s2c = 0.0888 * A**(2/3) * np.sqrt(a*(E-E0))
    fJ = np.exp(-(J**2 + (J+1)**2)/(2*s2c))
    c = np.exp(2*np.sqrt(a*(E-E0))) / (12 * np.sqrt(2*s2c) * (a*(E-E0)**5)**(1/4))
    return c * fJ

def mean_spacing_averaging(E):
    """
    Finds the mean level-spacing by taking the average of the level-spacings. Also returns the
    standard deviation of the mean level-spacing.

    Parameters
    ----------
    E : ndarray[float]
        Resonance energies.

    Returns
    -------
    mean_lvl_spacing     : float
        The mean level-spacing of the given energies.
    mean_lvl_spacing_std : float
        The standard deviation of the mean level-spacing for the given energies.
    """

    E = np.sort(E)
    N = len(E) - 1
    lvl_spacings = np.diff(E)
    mean_lvl_spacing = np.mean(lvl_spacings)
    # mean_lvl_spacing = (E[-1] - E[0]) / N # alternative
    mean_lvl_spacing_std = np.sqrt( np.mean((lvl_spacings - mean_lvl_spacing)**2) / (N-1) )
    return mean_lvl_spacing, mean_lvl_spacing_std

def mean_spacing_regression(E, EB:tuple):
    """
    Finds the mean level-spacing of the given energies by taking the slope of the empirical CDF
    of the energy level distribution.

    Parameters
    ----------
    E  : ndarray[float]
        Resonance energies.
    EB : tuple[float]
        Resonance ladder boundaries.
    
    Returns
    -------
    mean_lvl_spacing     : float
        The mean level-spacing of the given energies.
    """
    
    N = len(E)
    x = np.concatenate(([EB[0]], E, [EB[1]]))
    dx = np.diff(x)
    dx2 = np.diff(x**2)
    y = np.arange(N+1)
    # Delta  = EB[1] - EB[0]
    Delta2 = EB[1]**2 - EB[0]**2
    Delta3 = EB[1]**3 - EB[0]**3

    a = np.sum(y*dx)
    b = np.sum(y*dx2)
    A = 3*(b - a*(EB[1]+EB[0])) / (2*Delta3-(3/2)*(EB[0]+EB[1])*Delta2)
    mean_lvl_spacing = 1 / A
    return mean_lvl_spacing

# =================================================================================================
#    Mean Partial Widths:
# =================================================================================================

def mean_width_averaging(widths):
    """
    Finds the mean partial widths by taking the average of the widths. Also returns the standard
    deviation of the mean partial widths.

    Parameters
    ----------
    widths : ndarray[float]
        Resonance partial widths.

    Returns
    -------
    mean_width     : float
        The mean width of the given the partial widths.
    mean_width_std : float
        The standard deviation of the mean width given the partial widths.
    """

    mean_width = np.mean(abs(np.array(widths)))
    mean_width_std = np.sqrt( np.mean((widths - mean_width)**2) / (len(widths)-1) )
    return mean_width, mean_width_std

def mean_width_CDF_regression(widths, dof:int=1, thres:float=0.0):
    """
    Finds the mean partial widths by performing a regression on the Porter-Thomas CDF distribution.
    A truncation on the widths can be provided.

    Parameters
    ----------
    widths : ndarray[float]
        Resonance partial widths.
    dof    : int
        Porter-Thomas degrees of freedom. Default = 1.
    thres  : float
        Truncates all widths below this value. Default = 0.0.

    Returns
    -------
    mean_width       : float
        The mean width of the given the partial widths.
    mean_width_std   : float
        The standard deviation of the mean width given the partial widths.
    frac_missing     : float
        The fraction of missing resonances, estimated using Porter-Thomas distribution.
    frac_missing_std : float
        The standard deviation on the number of missing resonances, estimated using Porter-Thomas
        distribution.
    """

    widths = np.sort(abs(np.array(widths)))
    num_found_widths = len(widths)
    widths_trunc = widths[widths >= thres]
    num_trunc_widths = len(widths_trunc)
    bounds = (0.0, 2*np.max(widths))
    X = np.linspace(*bounds, 100_000)
    Y = np.searchsorted(widths_trunc, X) / num_trunc_widths
    func = lambda g2, g2m: porter_thomas_dist(mean=g2m, df=dof, trunc=thres).cdf(g2)
    mean_width, mean_width_cov = curve_fit(func, X, Y, bounds=(0.0, np.inf))
    np.set_printoptions(threshold=np.inf)
    # FIXME: the standard deviation of the mean of the widths is under-estimated!
    mean_width_std = np.sqrt(mean_width_cov[0,0])
    # FIXME: the fraction missing is incorrect!
    frac_below_thres = fraction_missing_gn2(thres, mean_width, dof)
    num_pred_widths = num_trunc_widths / (1-frac_below_thres)
    frac_missing = num_found_widths / num_pred_widths
    frac_missing_std = None # FIXME: find the standard deviation on the fraction of missing resonances!
    return mean_width[0], mean_width_std, \
           frac_missing, frac_missing_std