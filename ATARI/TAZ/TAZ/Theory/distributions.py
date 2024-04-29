from math import pi, sqrt
import numpy as np
from scipy.stats import rv_continuous, chi2
from scipy.special import erf, erfc, expm1

# =================================================================================================
#   Wigner Distribution
# =================================================================================================

class __WignerDistribution(rv_continuous):
    'Wigner Distribution'
    def _pdf(self, x, beta:int):
        beta = beta[0]
        if   beta == 1:
            coef1 = pi/4
            coef2 = pi/2
        elif beta == 2:
            coef1 = 4/pi
            coef2 = 32/pi**2
        elif beta == 4:
            coef1 = 64/(9*pi)
            coef2 = 262144/(729*pi**3)
        else:
            raise ValueError(f'beta = {beta} does not exist. Choose beta = 1, 2, or 4.')
        return coef2 * x**beta * np.exp(-coef1*x**2)
    def _cdf(self, x, beta:int):
        beta = beta[0]
        if   beta == 1:
            coef = pi/4
            return -expm1(-coef*x**2)
        elif beta == 2:
            coef = 4/pi
            return erf(sqrt(coef)*x) - coef*x*np.exp(-coef*x**2)
        elif beta == 4:
            coef = 64/(9*pi)
            return erf(sqrt(coef)*x) - coef/4*x*(2*coef*x**2+3)*np.exp(-coef*x**2)
        else:
            raise ValueError(f'beta = {beta} does not exist. Choose beta = 1, 2, or 4.')
    def _sf(self, x, beta:int):
        beta = beta[0]
        if   beta == 1:
            coef = pi/4
            return np.exp(-coef*x**2)
        elif beta == 2:
            coef = 4/pi
            return erfc(sqrt(coef)*x) + coef*x*np.exp(-coef*x**2)
        elif beta == 4:
            coef = 64/(9*pi)
            return erfc(sqrt(coef)*x) + coef/4*x*(2*coef*x**2+3)*np.exp(-coef*x**2)
        else:
            raise ValueError(f'beta = {beta} does not exist. Choose beta = 1, 2, or 4.')
wigner_dist = __WignerDistribution(name='Wigner distribution', a=0.0, b=np.inf, shapes='beta')

# =================================================================================================
#   Level-Spacing Ratio Distribution
# =================================================================================================

class __LevelSpacingRatioDistribution(rv_continuous):
    'Level-spacing Ratio Distribution'
    def _pdf(self, x, beta:int):
        beta = beta[0]
        if   beta == 1:   C_beta = 27/8
        elif beta == 2:   C_beta =  81*sqrt(3)/(4*pi)
        elif beta == 4:   C_beta = 729*sqrt(3)/(4*pi)
        else:           raise ValueError(f'beta = {beta} does not exist. Choose beta = 1, 2, or 4.')
        gamma = 1+(3/2)*beta
        return C_beta * (x + x**2)**beta / (1 + x + x**2)**gamma
    def _cdf(self, x, beta:int):
        beta = beta[0]
        if   beta == 1:
            numerator = np.polyval([2,3,-3,-2], x)
            return (1/4) * (numerator/((1+x+x**2)**(3/2))) + 0.5
        elif beta == 2:
            numerator = x*np.polyval([2,5,0,-5,-2], x)
            return (3/pi) * ((sqrt(3)/4) * (numerator / (1+x+x**2)**3) \
                             + np.arctan((2*x+1)/sqrt(3))) - 0.5
        elif beta == 4:
            numerator = x * np.polyval([4,22,72,159,168,0,-168,-159,-72,-22,-4], x)
            return (3/pi) * ((sqrt(3)/8) * (numerator / (1+x+x**2)**6) \
                             + np.arctan((2*x+1)/sqrt(3))) - 0.5
        else:
            raise ValueError(f'beta = {beta} does not exist. Choose beta = 1, 2, or 4.')
lvl_spacing_ratio_dist = __LevelSpacingRatioDistribution(name='Level-spacing ratio distribution', a=0.0, b=np.inf, shapes='beta')

# =================================================================================================
#   SemiCircle Distribution
# =================================================================================================

class __SemicircleDistribution(rv_continuous):
    """
    Level-spacing Ratio Distribution.

    The scale increases as sqrt(2*beta*num_res)
    """
    def _pdf(self, x):
        return (2/pi) * np.sqrt(1 - x**2)
    def _cdf(self, x):
        return (x/np.pi) * np.sqrt(1.0 - x**2) + np.arcsin(x)/np.pi + 0.5
semicircle_dist = __SemicircleDistribution(name='Wigner Semicircle Distribution', a=-1.0, b=1.0)

# =================================================================================================
#   Porter-Thomas Distribution
# =================================================================================================

class __PorterThomasDistribution(rv_continuous):
    def _argcheck(self, mean, df, trunc):
        return np.isfinite(trunc) & (trunc >= 0.0) \
                & np.isfinite(mean) & (mean >= 0) \
                & (df >= 1) & (df % 1 == 0)
    def _pdf(self, x, mean:float, df:int, trunc:float):
        x = abs(x)
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        y = np.zeros_like(x)
        y[x >  trunc] = chi2.pdf(x[x >  trunc], df=df, scale=mean/df) / norm
        y[(x < 0) & (df == 1)] = 0
        return y
    def _cdf(self, x, mean:float, df:int, trunc:float):
        x = abs(x)
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        y = np.zeros_like(x)
        y[x >  trunc] = 1 + (chi2.cdf(x, df=df, scale=mean/df)-1) / norm
        return y
    def _sf(self, x, mean:float, df:int, trunc:float):
        x = abs(x)
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        y = np.ones_like(x)
        y[x >  trunc] = chi2.sf(x, df=df, scale=mean/df) / norm
        return y
    def _ppf(self, q, mean:float, df:int, trunc:float):
        # Getting the sign:
        sign = np.ones_like(q)
        sign[q < 0.5] = -1
        q = 2.0 * abs(q - 0.5)
        # Calculating the value:
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        qp = (q - 1) * norm + 1
        x = chi2.ppf(qp, df=df, scale=mean/df)
        return sign * x
    def _isf(self, q, mean:float, df:int, trunc:float):
        # Getting the sign:
        sign = np.ones_like(q)
        sign[q < 0.5] = -1
        q = 2.0 * abs(q - 0.5)
        # Calculating the value:
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        qp = q * norm
        x = chi2.isf(qp, df=df, scale=mean/df)
        return sign * x
porter_thomas_dist = __PorterThomasDistribution(name='Porter-Thomas distribution', shapes='mean, df, trunc')

# =================================================================================================
#    Dyson-Mehta ∆3 Metric:
# =================================================================================================

def deltaMehta3(E, EB:tuple):
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

def deltaMehtaPredict(L:int, ensemble:str='GOE'):
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
        delta_3 = pi**(-2) * (np.log(L) - 0.0687)
    elif ensemble.lower() == 'poisson':
        delta_3 = L/15
    elif ensemble.lower() == 'picket':
        delta_3 = 1/12   # "picket" refers to "picket fence", where the levels are uniformly distributed
    else:
        raise ValueError(f'Unknown ensemble, {ensemble}. Please choose from "GOE", "Poisson" or "picket".')
    return delta_3

# =================================================================================================
#    More Width Distribution Functions:
# =================================================================================================

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
    # fraction_missing = gammainc(dof/2, dof*gn2_trunc/(2*gn2m))
    fraction_missing = porter_thomas_dist(mean=gn2m, df=dof, trunc=0.0).cdf(gn2_trunc)
    return fraction_missing