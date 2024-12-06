from math import pi, sqrt
import numpy as np
from scipy.stats import rv_continuous, chi2
from scipy.special import erf, erfc, expm1, gamma

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
    # def _ppf(self, q, beta:int):
    #     beta = beta[0]
    #     if beta == 1:
    #         coef = pi/4
    #         return np.sqrt(-np.log(1-q)/coef)
    #     else:
    #         raise NotImplementedError('The percent point function is only implemented for beta = 1 at this time.')
wigner_dist = __WignerDistribution(name='Wigner distribution', a=0.0, b=np.inf, shapes='beta')

# =================================================================================================
#   Level-Spacing Ratio Distribution
# =================================================================================================

class __LevelSpacingRatioDistribution(rv_continuous):
    'Level-spacing Ratio Distribution'
    def _pdf(self, x, beta:int):
        beta = beta[0]
        if   beta == 1:     C_beta = 27/8
        elif beta == 2:     C_beta =  81*sqrt(3)/(4*pi)
        elif beta == 4:     C_beta = 729*sqrt(3)/(4*pi)
        else:               raise ValueError(f'beta = {beta} does not exist. Choose beta = 1, 2, or 4.')
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
#   Gaussian Ensemble Distribution
# =================================================================================================

class __GEDistribution(rv_continuous):
    """
    Distribution of eigenvalues in a Gaussian Ensemble matrix.
    """
    def _pdf(self, x, beta:int):
        beta = beta[0]
        x = np.sort(x)
        n = len(x)
        prob = np.exp(-(1/4) * np.sum(x**2))
        for i, xi in enumerate(x):
            prob *= np.prod(x[i+1:]-xi)
        prob = abs(prob)**beta
        j = np.arange(1, n+1, 1)
        Z = np.prod((sqrt(2*pi)*gamma(1+j*beta/2)) / (beta**((n+1)/2)*gamma(1+beta/2)))
        prob /= Z
        return prob
gaussian_ensemble_dist = __GEDistribution(name='Gaussian Ensemble Distribution')

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
        x_trunc     =    x[x > trunc]
        df_trunc    =   df[x > trunc]
        scale_trunc = mean[x > trunc] / df_trunc
        norm_trunc  = norm[x > trunc]
        y[x > trunc] = chi2.pdf(x_trunc, df=df_trunc, scale=scale_trunc) / norm_trunc
        return y
    def _cdf(self, x, mean:float, df:int, trunc:float):
        x = abs(x)
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        y = np.zeros_like(x)
        x_trunc     =    x[x > trunc]
        df_trunc    =   df[x > trunc]
        scale_trunc = mean[x > trunc] / df_trunc
        norm_trunc  = norm[x > trunc]
        y[x > trunc] = 1 + (chi2.cdf(x_trunc, df=df_trunc, scale=scale_trunc) - 1) / norm_trunc
        return y
    def _sf(self, x, mean:float, df:int, trunc:float):
        x = abs(x)
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        y = np.ones_like(x)
        x_trunc     =    x[x > trunc]
        df_trunc    =   df[x > trunc]
        scale_trunc = mean[x > trunc] / df_trunc
        norm_trunc  = norm[x > trunc]
        y[x > trunc] = chi2.sf(x_trunc, df=df_trunc, scale=scale_trunc) / norm_trunc
        return y
    def _ppf(self, q, mean:float, df:int, trunc:float):
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        qp = (q - 1) * norm + 1
        x = chi2.ppf(qp, df=df, scale=mean/df)
        return x
    def _isf(self, q, mean:float, df:int, trunc:float):
        norm = chi2.sf(trunc, df=df, scale=mean/df)
        qp = q * norm
        x = chi2.isf(qp, df=df, scale=mean/df)
        return x
porter_thomas_dist = __PorterThomasDistribution(name='Porter-Thomas distribution', shapes='mean, df, trunc')