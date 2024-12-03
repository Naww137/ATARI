from typing import Tuple, List
from math import pi, sqrt, log, floor
from sys import float_info
import numpy as np
from numpy.random import Generator
from numpy import newaxis as NA
from scipy.special import gamma, gammaincc, gammainccinv, erfc, erfcx, erfcinv
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

# =================================================================================================
#    Spacing Distribution Class:
# =================================================================================================

class SpacingDistributionBase:
    """
    A base class for level-spacing probability distributions.
    """

    def __init__(self, lvl_dens:float=1.0, **kwargs):
        'Sets SpacingDistributionBase attributes.'
        self.lvl_dens = float(lvl_dens)
        for key, value in kwargs.items():
            setattr(self, key, value)

    # Probability Distributions:
    def _f0(self, x):
        'Level-spacing PDF.'
        raise ValueError('A "_f0" method must be defined.')
    def _f1(self, x):
        'Edge level-spacing PDF.'
        func = lambda t: self._f0(t)
        intfunc = lambda x: quad(func, a=x, b=np.inf)[0]
        return np.vectorize(intfunc)(x) * self.lvl_dens
    def _f2(self, x):
        'Double edge level-spacing PDF.'
        func = lambda t,x: (t-x)*self._f0(t)
        intfunc = lambda x: quad(func, a=x, b=np.inf, args=(x,))[0]
        return np.vectorize(intfunc)(x) * self.lvl_dens**2
    
    # Distribution Ratios:
    def _r1(self, x):
        'Ratio of _f0/_f1.'
        return self._f0(x) / self._f1(x)
    def _r2(self, x):
        'Ratio of _f1/_f2.'
        return self._f1(x) / self._f2(x)
    
    # Survival Inverse Functions:
    def _iF0(self, q):
        'Inverse of the survival function of the "f0" probability density function.'
        def func(u, y):
            with np.errstate(divide='ignore'):
                x = -np.log(u)
            return self._f1(x) - y
        def invfunc(y):
            if func(1.0, y) <  1e-13: return 0.0
            u = brentq(func, a=0.0, b=1.0, args=(y,), xtol=float_info.epsilon, rtol=1e-15)
            with np.errstate(divide='ignore'):
                x = -np.log(u)
            return x
        return np.vectorize(invfunc)(q)
    def _iF1(self, q):
        'Inverse of the survival function of the "f1" probability density function.'
        def func(u, y):
            with np.errstate(divide='ignore'):
                x = -np.log(u)
            return self._f2(x) - y
        def invfunc(y):
            if func(1.0, y) <  1e-13: return 0.0
            u = brentq(func, a=0.0, b=1.0, args=(y,), xtol=float_info.epsilon, rtol=1e-15)
            with np.errstate(divide='ignore'):
                x = -np.log(u)
            return x
        return np.vectorize(invfunc)(q)
    
    # Samplers:
    def _sample_f0(self, size:tuple=None, rng:Generator=None, seed:int=None):
        'Sampling of "f0" distribution.'
        if rng is None:
            rng = np.random.RandomState(seed)
        return self._iF0(rng.random(size))
    def _sample_f1(self, size:tuple=None, rng:Generator=None, seed:int=None):
        'Sampling of "f1" distribution.'
        if rng is None:
            rng = np.random.RandomState(seed)
        return self._iF1(rng.random(size))
    
    # Function Handlers:
    def f0(self, x):
        'Level-spacing PDF.'
        x = np.array(x)
        y = np.zeros_like(x)
        y[~np.isinf(x)] = self.lvl_dens * self._f0(self.lvl_dens * x[~np.isinf(x)])
        return y
    def f1(self, x):
        'Edge level-spacing PDF.'
        x = np.array(x)
        y = np.zeros_like(x)
        y[~np.isinf(x)] = self.lvl_dens * self._f1(self.lvl_dens * x[~np.isinf(x)])
        return y
    def f2(self, x):
        'Double edge level-spacing PDF.'
        x = np.array(x)
        y = np.zeros_like(x)
        y[~np.isinf(x)] = self.lvl_dens * self._f2(self.lvl_dens * x[~np.isinf(x)])
        return y
    def r1(self, x):
        'Ratio of f0/f1.'
        x = np.array(x)
        return self._r1(self.lvl_dens * x)
    def r2(self, x):
        'Ratio of f1/f2.'
        x = np.array(x)
        r = np.zeros_like(x)
        return self._r2(self.lvl_dens * x)
    def iF0(self, q):
        'Inverse of the survival function of the "f0" probability density function.'
        q = np.array(q)
        return self._iF0(q) / self.lvl_dens
    def iF1(self, q):
        'Inverse of the survival function of the "f1" probability density function.'
        q = np.array(q)
        return self._iF1(q) / self.lvl_dens
    def sample_f0(self, size:tuple=None, rng:Generator=None, seed:int=None):
        'Sampling of "f0" distribution.'
        return self._sample_f0(size, rng, seed) / self.lvl_dens
    def sample_f1(self, size:tuple=None, rng:Generator=None, seed:int=None):
        'Sampling of "f1" distribution.'
        return self._sample_f1(size, rng, seed) / self.lvl_dens
    
    # Nicely Named Functions:
    def pdf(self, x):
        'Probability density function (same as f0).'
        return self.lvl_dens*self._f0(self.lvl_dens * x)
    def cdf(self, x):
        'Cumulative probability density function.'
        return 1 - self.sf(x)
    def sf(self, x):
        'Survival function.'
        return self._f1(self.lvl_dens * x)
    def ppf(self, x):
        'Percent point function'
        return self.isf(1-x)
    def isf(self, x):
        'Inverse survival function'
        return self.iF0(x)
    
    # xMax Functions:
    def xMax_f0(self, err):
        return self.iF0(err)
    def xMax_f1(self, err):
        return self.iF1(err)
    
    def __call__(self, x):
        return self.pdf(x)
    
# =================================================================================================
#    Poisson Distribution:
# =================================================================================================
    
class PoissonGen(SpacingDistributionBase):
    """
    Generates a Poisson level-spacing distribution.

    Great for debugging TAZ.
    """
    def _f0(self, x):
        return np.exp(-x)
    def _f1(self, x):
        return np.exp(-x)
    def _f2(self, x):
        return np.exp(-x)
    def _r1(self, x):
        return np.ones_like(x)
    def _r2(self, x):
        return np.ones_like(x)
    def _iF0(self, q):
        return -np.log(q)
    def _iF1(self, q):
        return -np.log(q)
    
# =================================================================================================
#    Wigner Distribution:
# =================================================================================================
    
class WignerGen(SpacingDistributionBase):
    """
    Generates a Wigner level-spacing distribution.
    """
    def __init__(self, lvl_dens:float):
        self.lvl_dens = float(lvl_dens)
    def _f0(self, x):
        coef = pi/4
        return 2 * coef * x * np.exp(-coef * x*x)
    def _f1(self, x):
        return np.exp((-pi/4) * x*x)
    def _f2(self, x):
        root_coef = sqrt(pi/4)
        return erfc(root_coef * x)
    def _r1(self, x):
        return (pi/2) * x
    def _r2(self, x):
        root_coef = sqrt(pi/4)
        return 1.0 / erfcx(root_coef * x)
    def _iF0(self, q):
        return np.sqrt((-4/pi) * np.log(q))
    def _iF1(self, q):
        root_coef = sqrt(pi/4)
        return erfcinv(q) / root_coef
    
# =================================================================================================
#    Brody Distribution:
# =================================================================================================
    
class BrodyGen(SpacingDistributionBase):
    """
    Generates a Brody level-spacing distribution.
    """
    def __init__(self, lvl_dens:float, w:float):
        self.lvl_dens = float(lvl_dens)
        self.w        = float(w)
    def _f0(self, x):
        w1 = self.w + 1.0
        a = gamma(1/w1+1)**w1
        axw = a*x**self.w
        return w1 * axw * np.exp(-axw*x)
    def _f1(self, x):
        w1 = self.w + 1.0
        a = gamma(1/w1+1)**w1
        return np.exp(-a*x**w1)
    def _f2(self, x):
        w1 = self.w + 1.0
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        return gammaincc(w1i, a*x**w1)
    def _r1(self, x):
        w1 = self.w + 1.0
        a = gamma(1/w1+1)**w1
        return w1*a*x**self.w
    def _r2(self, x):
        w1 = self.w + 1.0
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        axw1 = a*x**w1
        F2 = gammaincc(w1i, axw1)
        return np.exp(-axw1) / F2
    def _iF0(self, q):
        w1 = self.w + 1.0
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        return (-np.log(q) / a) ** w1i
    def _iF1(self, q):
        w1 = self.w + 1
        w1i = 1.0 / w1
        a = gamma(w1i+1)**w1
        return (gammainccinv(w1i, q) / a) ** w1i
    
# =================================================================================================
#    Higher-Order Spacing Distributions:
# =================================================================================================
    
@np.vectorize(otypes=[float])
def _gamma_ratio(x:float):
    """
    A function to calculate the ratio, `Gamma(x/2) / Gamma((x-1)/2)`. This function is used instead
    of calculating each gamma separately for numerical stability.

    Parameters
    ----------
    x : float
        The function parameter of `Gamma(x/2) / Gamma((x-1)/2)`.

    Returns
    -------
    ratio : float
        The calculated ratio `Gamma(x/2) / Gamma((x-1)/2)`.

    See Also
    --------
    `HighOrderSpacingGen`
    """
    if x < 2.0:
        raise TypeError('"x" must be greater than or equal to 2.')
    r = (0.5*x) % 1.0
    coef = gamma(r+1.0)/gamma(r+0.5)
    K = np.arange(floor(0.5*x)-1.0)
    ratio = coef * np.prod(1.0 + 0.5/(r+K+0.5))
    return ratio

@np.vectorize(otypes=[float])
def _high_order_variance(n:int):
    """
    A function for calculating the variance of the `n+1`-th nearest level-spacing distribution.
    This is used for the Gaussian Approximation when the analytical solution becomes too costly
    to compute.

    Parameters
    ----------
    n : int or float
        The number of levels between the two selected levels.

    Returns
    -------
    variance : float
        The variance of the high-order level-spacing distribution.

    See Also
    --------
    `HighOrderSpacingGen`
    """
    a = (n**2 + 5*n + 2)/2
    if a > 1e3:
        return 1.0 # variance asymptotically approaches 1.0
    B = (_gamma_ratio(a+2) / (n+1))**2
    variance = (a+1)/(2*B) - (n+1)**2
    return variance

class HighOrderSpacingGen(SpacingDistributionBase):
    """
    Generates the `n+1`-th nearest neighbor level-spacing distribution as determined by the
    Gaussian Orthogonal Ensemble (GOE). The distribution is calculated at each value in the numpy
    array, `x`.

    Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
    """

    NTHRES = 15

    def __init__(self, lvl_dens:float=1, n:int=1):
        if (n % 1 > 0) or (n < 0):
            raise ValueError(f'The skipping index, "n", must be a positive integer number.')
        self.n = int(n)
        self.lvl_dens = float(lvl_dens)
    @property
    def lvl_dens_dist(self):
        return self.lvl_dens / (self.n+1)
    def _f0(self, x):
        n = self.n
        if n <= self.NTHRES: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            coef = 2 * rB / gamma((a+1)/2)
            rBx  = rB * x
            return coef * rBx**a * np.exp(-rBx**2) # (Eq. 11)
        else: # Higher n --> Gaussian Approximation
            sig = sqrt(_high_order_variance(n))
            return norm.pdf(x, n+1, sig)
    def _f1(self, x):
        n = self.n
        if n <= self.NTHRES: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            return gammaincc((a+1)/2, (rB * x)**2) / (n+1)
        else: # Higher n --> Gaussian Approximation
            sig = sqrt(_high_order_variance(n))
            return norm.sf(x, n+1, sig) / (n+1)
    def _f2(self, x):
        n = self.n
        if n <= self.NTHRES: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            return ((n+1)*gammaincc((a+2)/2, (rB * x)**2) - x*gammaincc((a+1)/2, (rB * x)**2)) / (n+1)**2
        else: # Higher n --> Gaussian Approximation
            var = _high_order_variance(n)
            sig = sqrt(var)
            f0 = norm.pdf(x, n+1, sig)
            f1 = norm.sf(x, n+1, sig)
            return (var*f0 - (x-n-1)*f1) / (n+1)**2
    def _iF0(self, q):
        n = self.n
        if n <= self.NTHRES: # Lower n --> Exact Calculation
            a = n + (n+1)*(n+2)/2 # (Eq. 10)
            rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
            return np.sqrt(gammainccinv((a+1)/2, q)) / rB
        else: # Higher n --> Gaussian Approximation
            sig = sqrt(_high_order_variance(n))
            y = norm.isf(q, n+1, sig)
            return np.maximum(y, 0)
    def _iF1(self, q):
        raise NotImplementedError('Current iF1 function is unstable for HighOrderSpacing.')
    
# =================================================================================================
#    Missing Distribution:
# =================================================================================================
    
class MissingGen(SpacingDistributionBase):
    """
    Generates a missing resonances level-spacing distribution.
    """

    NTHRES = (13, 20)

    def __init__(self, lvl_dens:float, pM:float, err:float):
        if not (0 < pM <= 1):
            raise ValueError('The missing resonance fraction, "pM", must be in the range, 0 < pM <= 1.')
        self.pM = float(pM)
        self.lvl_dens = float(lvl_dens)
    @property
    def lvl_dens_true(self):
        return self.lvl_dens / (1-self.pM)
    def _f0(self, x):
        XTHRES = self.NTHRES[0] + 1
        y = np.zeros_like(x)

        # Below threshold:
        mult_fact = (1-self.pM)
        for n in range(self.NTHRES[1]+1):
            f0n = HighOrderSpacingGen(lvl_dens=1/(1-self.pM), n=n).f0(x[x < XTHRES])
            y[x < XTHRES] += mult_fact * f0n
            mult_fact *= self.pM
        
        # Above threshold:
        u = x[x >= XTHRES]/(1-self.pM) - 1
        y[x >= XTHRES] = (1.0/(1-self.pM)) * self.pM**u * np.exp(0.5*_high_order_variance(u)*(log(self.pM))**2)
        return y
    def _f1(self, x):
        XTHRES = self.NTHRES[0] + 1
        y = np.zeros_like(x)

        # Below threshold:
        mult_fact = (1-self.pM)**2
        for n in range(self.NTHRES[1]+1):
            f1n = HighOrderSpacingGen(lvl_dens=1/(1-self.pM), n=n).f1(x[x < XTHRES]) * (n+1)
            y[x < XTHRES] += mult_fact * f1n
            mult_fact *= self.pM

        # Above threshold:
        u = x[x >= XTHRES]/(1-self.pM) - 1
        y[x >= XTHRES] = (-1.0/log(self.pM)) * self.pM**u * np.exp(0.5*_high_order_variance(u)*(log(self.pM))**2)
        return y
    def _f2(self, x):
        XTHRES = self.NTHRES[0] + 1
        y = np.zeros_like(x)

        # Below threshold:
        mult_fact = (1-self.pM)**3
        for n in range(self.NTHRES[1]+1):
            f2n = HighOrderSpacingGen(lvl_dens=1/(1-self.pM), n=n).f2(x[x < XTHRES]) * (n+1)**2
            y[x < XTHRES] += mult_fact * f2n
            mult_fact *= self.pM

        # Above threshold:
        u = x[x >= XTHRES]/(1-self.pM) - 1
        y[x >= XTHRES] = (1-self.pM)/(log(self.pM))**2 * self.pM**u * np.exp(0.5*_high_order_variance(u)*(log(self.pM))**2)
        return y
    def _r1(self, x):
        XTHRES = self.NTHRES[0] + 1
        y = np.zeros_like(x)

        # Below threshold:
        y[x < XTHRES] = self._f0(x[x < XTHRES]) / self._f1(x[x < XTHRES])

        # Above threshold:
        y[x >= XTHRES] = -log(self.pM) / (1-self.pM)
        return y
    def _r2(self, x):
        XTHRES = self.NTHRES[0] + 1
        y = np.zeros_like(x)

        # Below threshold:
        y[x < XTHRES] = self._f1(x[x < XTHRES]) / self._f2(x[x < XTHRES])

        # Above threshold:
        y[x >= XTHRES] = -log(self.pM) / (1-self.pM)
        return y

# =================================================================================================
#    Distribution Merger:
# =================================================================================================

class MergedDistributionBase:
    """
    A base class for merged level-spacing probability distributions.
    """

    def __init__(self, lvl_densities:List[float]):
        self.lvl_densities = np.array(lvl_densities, dtype=float)
        self.lvl_dens = np.sum(self.lvl_densities)
    def _f0(self, x, priorL, priorR):
        raise ValueError('"f0" has not been defined.')
    def _f1(self, x, prior):
        raise ValueError('"f1" has not been defined.')
    def _f2(self, x):
        raise ValueError('"f2" has not been defined.')
    def _xMax_f0(self, err):
        raise ValueError('"xMax_f0" has not been defined.')
    def _xMax_f1(self, err):
        raise ValueError('"xMax_f1" has not been defined.')
    
    def f0(self, x, priorL=None, priorR=None):
        if priorL is None:
            priorL = np.tile(self.lvl_densities, (x.size, 1))
        if priorR is None:
            priorR = np.tile(self.lvl_densities, (x.size, 1))
        return self._f0(x, priorL, priorR)
    def f1(self, x, prior=None):
        if prior is None:
            prior = np.tile(self.lvl_densities, (x.size, 1))
        return self._f1(x, prior)
    def f2(self, x):
        return self._f2(x)
    def xMax_f0(self, err):
        return self._xMax_f0(err)
    def xMax_f1(self, err):
        return self._xMax_f1(err)
    
    # Distribution ratios:
    def r1(self, x):
        return self.f0(x) / self.f1(x)
    def r2(self, x):
        return self.f1(x) / self.f2(x)
    
    # Survival Inverse Functions:
    def iF0(self, q):
        raise NotImplementedError('"iF0" has not been implemented for merged distributions.')
    def iF1(self, q):
        raise NotImplementedError('"iF1" has not been implemented for merged distributions.')
    
    # Samplers:
    def sample_f0(self, size:tuple=None, rng:Generator=None, seed:int=None):
        'Sampling of "f0" distribution.'
        if rng is None:
            rng = np.random.RandomState(seed)
        return self.iF0(rng.random(size))
    def sample_f1(self, size:tuple=None, rng:Generator=None, seed:int=None):
        'Sampling of "f1" distribution.'
        if rng is None:
            rng = np.random.RandomState(seed)
        return self.iF1(rng.random(size))
    
    def pdf(self, x):
        return self.f0(x)
    def cdf(self, x):
        return 1.0 - self.sf(x)
    def sf(self, x):
        return self.f1(x) / self.lvl_dens
    def ppf(self, q):
        return self.iF0(1-q)
    def isf(self, q):
        return self.iF0(q)

def merge(*distributions:Tuple[SpacingDistributionBase]):
    """
    Merges level-spacing distributions.
    """
    
    G = len(distributions)
    if G == 1:
        distribution = distributions[0]
        lvl_densities = np.array([distribution.lvl_dens])
        class MergedSpacingDistributionGen(MergedDistributionBase):
            def _f0(self, x, priorL, priorR):
                return distribution.f0(x)
            def _f1(self, x, prior):
                return distribution.f1(x)
            def _f2(self, x):
                return distribution.f2(x)
            def _xMax_f0(self, err):
                return distribution.iF0(err)
            def _xMax_f1(self, err):
                return distribution.iF1(err)
        Z0 = Z1 = Z2 = None
    else:
        lvl_densities = np.array([distribution.lvl_dens for distribution in distributions])
        def c_func(x):
            x = np.array(x)
            c = np.ones(x.shape)
            for distribution in distributions:
                c *= distribution.f2(x)
            return c

        # Normalization Factors:
        Z0 = np.zeros((G,G))
        for i in range(1,G):
            for j in range(i):
                func = lambda x: c_func(x) * distributions[i].r2(x) * distributions[j].r2(x)
                Z0[i,j] = quad(func, a=0.0, b=np.inf)[0]
                Z0[j,i] = Z0[i,j]
        for i in range(G):
            func = lambda x: c_func(x) * distributions[i].r2(x) * distributions[i].r1(x)
            Z0[i,i] = quad(func, a=0.0, b=np.inf)[0]
        Z1 = np.zeros((G,))
        for i in range(G):
            func = lambda x: c_func(x) * distributions[i].r2(x)
            Z1[i] = quad(func, a=0.0, b=np.inf)[0]
        Z2 = quad(c_func, a=0.0, b=np.inf)[0]

        # Merged Distribution:
        class MergedSpacingDistributionGen(MergedDistributionBase):
            def _f0(self, x, priorL, priorR):
                x = np.array(x)
                L = len(x)
                priorL = np.array(priorL)
                priorR = np.array(priorR)
                v = np.zeros((L,G))
                d = np.zeros((L,G))
                norm = (priorL[:,NA,:] @ Z0[NA,:,:] @ priorR[:,:,NA])[:,0,0]
                for i, distribution in enumerate(distributions):
                    v[:,i] = distribution.r2(x)
                    u = distribution.r1(x)
                    d[:,i] = v[:,i] * (u - v[:,i])
                F = c_func(x) / norm * ( \
                    np.sum(priorL * v, axis=1) \
                    * np.sum(priorR * v, axis=1) \
                    + np.sum(priorL * priorR * d, axis=1))
                return F
            def _f1(self, x, prior):
                x = np.array(x)
                L = len(x)
                norm = np.sum(prior * Z1)
                prior = np.array(prior)
                v = np.zeros((L,G))
                for i, distribution in enumerate(distributions):
                    v[:,i] = distribution.r2(x)
                F = c_func(x) / norm * np.sum(prior*v, axis=1)
                return F
            def _f2(self, x):
                F = c_func(x) / Z2
                return F
            
            def _xMax_f0(self, err):
                def func(u):
                    if u == 0.0:
                        return -err
                    x = -np.log(u)
                    fx_max = 0.0
                    for g,distribution in enumerate(distributions):
                        fx = distribution.r2(x) / np.min(lvl_densities*Z0[:,g])
                        if fx > fx_max:
                            fx_max = fx
                    fx *= c_func(x)
                    return fx - err
                u = brentq(func, a=0.0, b=1.0, xtol=float_info.epsilon, rtol=1e-15)
                x = -np.log(u)
                return x
            def _xMax_f1(self, err):
                def func(u):
                    if u == 0.0:
                        return -err
                    x = -np.log(u)
                    fx = c_func(x) / np.min(lvl_densities*Z1)
                    return fx - err
                u = brentq(func, a=0.0, b=1.0, xtol=float_info.epsilon, rtol=1e-15)
                x = -np.log(u)
                return x

    merged_spacing_distribution = MergedSpacingDistributionGen(lvl_densities)
    return merged_spacing_distribution