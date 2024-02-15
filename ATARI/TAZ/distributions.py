from math import pi, sqrt, ceil, log
import numpy as np
from scipy.special import gamma, gammaincc, gammainccinv, erfc, erfcx, erfcinv
from scipy.integrate import quad
from scipy.stats import norm

__doc__ = """
This module is the collection of level-spacing distribution functions and classes.
"""

# =================================================================================================
#    Level-Spacing Probability Distributions
# =================================================================================================

def _gamma_ratio(x):
    """
    A function to calculate the ratio, `Gamma(x/2) / Gamma((x-1)/2)`. This function is used instead
    of calculating each gamma separately for numerical stability.
    """
    rpii = 1.0 / sqrt(pi)
    if hasattr(x, '__iter__'):
        ratio = np.zeros(len(x))
        for idx, w in enumerate(x):
            q = rpii
            for i in range(3,int(w)):
                q = (i-2) / (2*q)
            ratio[idx] = q
    else:
        ratio = rpii
        for i in range(3,int(x)+1):
            ratio = (i-2) / (2*ratio)
    return ratio

def _high_order_variance(n:int):
    """
    A function for calculating the variance of the `n+1`-th nearest level-spacing distribution.
    This is used for the Gaussian Approximation when the analytical solution becomes too costly
    to compute.
    """
    a = (n**2 + 5*n + 2)/2
    B =(_gamma_ratio(a+2) / (n+1))**2
    return (a+1)/(2*B) - (n+1)**2

def _high_order_level_spacing_parts(X, n:int, orders:tuple=(0,1,2)):
    """
    Generates the `n+1`-th nearest neighbor level-spacing distribution as determined by the
    Gaussian Orthogonal Ensemble (GOE). The distribution is calculated at each value in the numpy
    array, `X`. Each order in `orders` request the order-th integrated level-spacing distribution.

    Source: https://journals.aps.org/pre/pdf/10.1103/PhysRevE.60.5371
    """
    out = []
    if n <= 15: # Lower n --> Exact Calculation
        a = n + (n+1)*(n+2)/2 # (Eq. 10)
        rB = _gamma_ratio(a+2) / (n+1) # square root of B (Eq. 12)
        coef = 2 * rB / gamma((a+1)/2)
        rBX  = rB * X
        for order in orders:
            if   order == 0: # Level-Spacing PDF
                F0 = coef * rBX**a * np.exp(-rBX**2) # (Eq. 11)
                out.append(F0)
            elif order == 1: # First Integral
                F1 = gammaincc((a+1)/2, rBX**2)
                out.append(F1)
            elif order == 2: # Second Integral
                F2 = gammaincc((a+2)/2, rBX**2)
                out.append(F2)
    else: # Higher n --> Gaussian Approximation
        sig = np.sqrt(_high_order_variance(n))
        for order in orders:
            if   order == 0: # Level-Spacing PDF
                F0 = norm.pdf(X, n+1, sig)
                out.append(F0)
            elif order == 1: # First Integral
                F1 = (1/2) * erfc((X+n+1)/(sig*sqrt(2)))
                out.append(F1)
            elif order == 2: # Second Integral
                if 0 not in orders:      F0 = norm.pdf(X, n+1, sig)
                if 1 not in orders:      F1 = (1/2) * erfc((X+n+1)/(sig*sqrt(2)))
                F2 = sig**2*F0 + (X+1)*F1 - 1
                out.append(F2)
    return tuple(out)

def high_order_level_spacing(X, n:int):
    return _high_order_level_spacing_parts(X, n, orders=(0,))[0]

class Distribution:
    """
    A class for level-spacing distributions, their integrals and inverses. Such distributions
    have been defined for Wigner distribution, Brody distribution, and the missing distribution.
    """
    def __init__(self, f0, f1=None, f2=None, parts=None, if1=None, if2=None, lvl_dens:float=None):
        """
        Initializes a Distribution object.

        Parameters:
        ----------
        f0    :: function
            Probability density function for the distribution.
        f1    :: function
            The reversed CDF of the level-spacing distribution.
        f2    :: function
            The doubly integrated level-spacing distribution.
        parts :: function
            Function that finds f2, f0/f1, and f1/f2 which are used when merging
            distributions. Default = None (calculated from f0, f1, and f2).
        if1   :: function
            The inverse function of f1. Default = None.
        if2   :: function
            The inverse function of f2. Default = None.
        lvl_dens  :: float
            The expected level-density for the distribution. Default = None (calculated from f0).
        """
        self.__f0 = f0
        if f1 is None:  raise NotImplementedError('Integration for f1 has not been implemented yet.')
        self.__f1 = f1
        if f2 is None:  raise NotImplementedError('Integration for f2 has not been implemented yet.')
        self.__f2 = f2
        if if1 is None: raise NotImplementedError('The inverse of f1 has not been implemented yet.')
        self.__if1 = if1
        if if2 is None: raise NotImplementedError('The inverse of f2 has not been implemented yet.')
        self.__if2 = if2

        if parts is None:
            def parts(x):
                F0 = self.__f0(x)
                F1 = self.__f1(x)
                F2 = self.__f2(x)
                return F2, F0/F1, F1/F2
        else:
            self.__parts = parts

        if lvl_dens is None:
            mean_lvl_spacing = quad(lambda x: x*self.__f0(x), 0, np.inf)[0]
            self.__lvl_dens = 1 / mean_lvl_spacing
        else:
            self.__lvl_dens = float(lvl_dens)

    # Functions and properties:
    def __call__(self, X):
        'Evaluates the probability density function for each distribution.'
        return self.__f0(X)
    def f0(self, X):
        'Probability density function for the distribution.'
        return self.__f0(X)
    def f1(self, X):
        'The reversed CDF of the level-spacing distribution.'
        return self.__f1(X)
    def f2(self, X):
        'The doubly integrated level-spacing distribution.'
        return self.__f2(X)
    def if1(self, X):
        'Inverse function of the "f1" function.'
        if np.any(X > 1.0) or np.any(X < 0.0):
            raise ValueError('Inverse functions can only be evaluated for values between 0 and 1.')
        return self.__if1(X)
    def if2(self, X):
        'Inverse function of the "f2" function.'
        if np.any(X > 1.0) or np.any(X < 0.0):
            raise ValueError('Inverse functions can only be evaluated for values between 0 and 1.')
        return self.__if2(X)
    def parts(self, X):
        'Provides f2, f0/f1, and f1/f2 which are used when merging distributions.'
        return self.__parts(X)
    def pdf(self, X):
        'Probability density function for the distribution.'
        return self.__f0(X)
    def cdf(self, X):
        'Cumulative probability function for the distribution.'
        return 1.0 - self.__f1(X)
    @property
    def lvl_dens(self):
        'The expected level-density for the distribution.'
        return self.__lvl_dens
    @property
    def MLS(self):
        'The mean level-spacing for the distribution.'
        return 1.0 / self.__lvl_dens

    # Sampling distributions using inverse CDF:
    def sample_f0(self, size:tuple=None, rng=None, seed:int=None):
        'Inverse CDF sampling on f0.'
        if rng is None:
            rng = np.random.default_rng(seed)
        return self.__if1(rng.random(size))
    def sample_f1(self, size:tuple=None, rng=None, seed:int=None):
        'Inverse CDF sampling on f1.'
        if rng is None:
            rng = np.random.default_rng(seed)
        return self.__if2(rng.random(size))

    # Distribution constructors:
    @classmethod
    def wigner(cls, lvl_dens:float=1.0):
        """
        Sample Wigner distribution without missing resonances considered.

        Parameters:
        ----------
        lvl_dens :: float
            Mean level-density. Default = 1.0.

        Returns:
        -------
        distribution :: Distribution
            The distribution object for the Wigner level-spacing distribution.
        """
        pid4  = pi/4
        coef = pid4*lvl_dens**2
        root_coef = sqrt(coef)
        def get_f0(X):
            return (2*coef) * X * np.exp(-coef * X*X)
        def get_f1(X):
            return np.exp(-coef * X*X)
        def get_f2(X):
            return erfc(root_coef * X) / lvl_dens
        def get_parts(X):
            fX = root_coef * X
            R1 = 2 * root_coef * fX
            R2 = lvl_dens / erfcx(fX)       # "erfcx(x)" is "exp(x^2)*erfc(x)"
            F2 = np.exp(-fX*fX) / R2    # Getting the "erfc(x)" from "erfcx(x)"
            return F2, R1, R2
        def get_if1(R):
            return np.sqrt(-np.log(R)) / root_coef
        def get_if2(R):
            return erfcinv(R) / root_coef
        return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, lvl_dens=lvl_dens)
    @classmethod
    def brody(cls, lvl_dens:float=1.0, w:float=0.0):
        """
        Sample Brody distribution without missing levels considered. The Brody distribution
        is an interpolation between Wigner distribution and Poisson distribution. Brody
        distribution is said to be more applicable for even-mass isotopes with an atomic mass
        number above 150. The Brody parameter can range from 0 to 1, where w=0 gives a Poisson
        distribution and w=1 gives a Wigner distribution.

        Parameters:
        ----------
        lvl_dens :: float
            Mean level-density. Default = 1.0.
        w    :: float
            Brody parameter. Default = 0.0.

        Returns:
        -------
        distribution :: Distribution
            The distribution object for the Brody level-spacing distribution.
        """
        w1i = 1.0 / (w+1)
        a = (lvl_dens*w1i*gamma(w1i))**(w+1)
        def get_f0(X):
            aXw = a*X**w
            return (w+1) * aXw * np.exp(-aXw*X)
        def get_f1(X):
            return np.exp(-a*X**(w+1))
        def get_f2(X):
            return (w1i*a**(-w1i)) * gammaincc(w1i, a*X**(w+1))
        def get_parts(X):
            aXw = a*X**w
            aXw1 = aXw*X
            R1 = (w+1)*aXw
            F2 = (w1i * a**(-w1i)) * gammaincc(w1i, aXw1)
            R2 = np.exp(-aXw1) / F2
            return F2, R1, R2
        def get_if1(R):
            return (-np.log(R) / a) ** w1i
        def get_if2(R):
            return (gammainccinv(w1i, R) / a) ** w1i
        return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, lvl_dens=lvl_dens)
    @classmethod
    def missing(cls, lvl_dens:float=1.0, pM:float=0.0, err:float=0.005):
        """
        Sample Wigner distribution with missing resonances considered.

        Source: http://www.lib.ncsu.edu/resolver/1840.16/4284 (Eq. 4.1)

        Parameters:
        ----------
        lvl_dens :: float
            Mean level-density. Default = 1.0.
        pM   :: float
            Fraction of missing resonances. Default = 0.0.
        err  :: float
            Maximum error in PDF. Default = 0.005.

        Returns:
        -------
        distribution :: Distribution
            The distribution object for the missing-resonances level-spacing distribution.
        """
        
        # If we assume there are no missing resonances, the PDF converges to Wigner:
        if pM == 0.0:
            print(RuntimeWarning('Warning: the "missing" distribution has a zero missing resonance fraction.'))
            return cls.wigner(lvl_dens)
        
        N_max = ceil(log(err, pM))
        coef = (pM**np.arange(N_max+1))[:,np.newaxis]
        mult_fact = lvl_dens * (1-pM) / (1 - pM**(N_max+1))
        def get_f0(X):
            func = lambda _n: _high_order_level_spacing_parts(lvl_dens*X, _n, orders=(0,))
            values = [func(n) for n in range(N_max+1)]
            return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
        def get_f1(X):
            func = lambda _n: _high_order_level_spacing_parts(lvl_dens*X, _n, orders=(1,))
            values = [func(n) for n in range(N_max+1)]
            return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
        def get_f2(X):
            func = lambda _n: _high_order_level_spacing_parts(lvl_dens*X, _n, orders=(2,))
            values = [func(n) for n in range(N_max+1)]
            return mult_fact * np.sum(coef * np.array([v[0] for v in values]), axis=0)
        def get_parts(X):
            func = lambda _n: _high_order_level_spacing_parts(lvl_dens*X, _n, orders=(0,1,2))
            values = [func(n) for n in range(N_max+1)]
            V0, V1, V2 = zip(*values)
            F0 = mult_fact * np.sum(coef*V0, axis=0)
            F1 = mult_fact * np.sum(coef*V1, axis=0)
            F2 = mult_fact * np.sum(coef*V2, axis=0)
            R1 = F0 / F1
            R2 = F1 / F2
            return F2, R1, R2
        def get_if1(X):
            raise NotImplementedError('Inverse Function for f1 has not been implemented yet.')
        def get_if2(X):
            raise NotImplementedError('Inverse Function for f2 has not been implemented yet.')
        return cls(f0=get_f0, f1=get_f1, f2=get_f2, parts=get_parts, if1=get_if1, if2=get_if2, lvl_dens=(1-pM)*lvl_dens)

class Distributions:
    """
    A class that collects multiple `Distribution` objects together.
    """
    def __init__(self, *distributions:Distribution):
        'Initializing distributions'
        self.distributions = list(distributions)
        self.__lvl_dens = np.array([distr.lvl_dens for distr in self.distributions])
    
    # Functions and properties:
    def __call__(self, X):
        'Evaluates the probability density function for each distribution.'
        return self.f0(X)
    def f0(self, X):
        'The PDF of the level-spacing distribution.'
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.f0(X)    for distr in self.distributions]).T
    def f1(self, X):
        'The reversed CDF of the level-spacing distribution.'
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.f1(X)    for distr in self.distributions]).T
    def f2(self, X):
        'The doubly integrated level-spacing distribution.'
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.f2(X)    for distr in self.distributions]).T
    def if1(self, X):
        'Inverse function of the "f1" function.'
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.if1(X)   for distr in self.distributions]).T
    def if2(self, X):
        'Inverse function of the "f2" function.'
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        return np.array([distr.if2(X)   for distr in self.distributions]).T
    def parts(self, X):
        'Provides f2, f0/f1, and f1/f2 which are used when merging distributions.'
        if not hasattr(X, '__iter__'):
            X = np.array([X])
        parts = np.array([distr.parts(X) for distr in self.distributions]).transpose(2,0,1)
        return parts[:,:,0], parts[:,:,1], parts[:,:,2]
    def pdf(self, X):
        'Probability density functions for each distribution.'
        return self.f0(X)
    def cdf(self, X):
        'Cumulative probability density functions for each distribution.'
        return 1.0 - self.f1(X)
    @property
    def lvl_dens(self):
        'The expected level-densities for each distribution.'
        return self.__lvl_dens
    @property
    def MLS(self):
        'The mean level-spacing for each distribution.'
        return 1.0 / self.__lvl_dens
    @property
    def lvl_dens_tot(self):
        'The total level-density between all distributions.'
        return np.sum(self.__lvl_dens)
    @property
    def num_dists(self):
        'The number of distributions.'
        return len(self.distributions)
    def __len__(self):
        return len(self.distributions)

    # Getting items:
    def __getitem__(self, indices):
        if hasattr(indices, '__iter__'):
            distributions = [self.distributions[idx] for idx in indices]
            return self.__class__(*distributions)
        else:
            return self.distributions[indices]

    # Distribution constructors:
    @classmethod
    def wigner(cls, lvl_dens):
        'Sample Wigner distribution for each spingroup.'
        lvl_dens = lvl_dens.reshape(-1,)
        distributions = [Distribution.wigner(lvl_dens_g) for lvl_dens_g in lvl_dens]
        return cls(*distributions)
    @classmethod
    def brody(cls, lvl_dens, w=None):
        'Sample Brody distribution for each spingroup.'
        G = len(lvl_dens)
        if w is None:
            w = np.zeros((G,))
        lvl_dens = lvl_dens.reshape(-1,)
        w    = w.reshape(-1,)
        distributions = [Distribution.brody(lvl_dens_g, w_g) for lvl_dens_g,w_g in zip(lvl_dens,w)]
        return cls(*distributions)
    @classmethod
    def missing(cls, lvl_dens, pM=None, err:float=5e-3):
        'Sample Missing distribution for each spingroup.'
        G = len(lvl_dens)
        if pM is None:
            pM = np.zeros((G,))
        lvl_dens = lvl_dens.reshape(-1,)
        pM   = pM.reshape(-1,)
        distributions = [Distribution.missing(lvl_dens_g, pM_g, err) for lvl_dens_g,pM_g in zip(lvl_dens,pM)]
        return cls(*distributions)