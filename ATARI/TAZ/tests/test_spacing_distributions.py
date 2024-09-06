import sys
sys.path.append('../TAZ')
from TAZ.Theory import PoissonGen, WignerGen, BrodyGen, MissingGen, HighOrderSpacingGen, merge

import numpy as np
from scipy.integrate import quad

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

__doc__ = """
This file verifies the level-spacing distribution classes and the merging functionality.
"""

class TestSpacingDistributions(unittest.TestCase):
    """
    Here, we intend to verify that all of the level-spacing distribution quantities are correct
    and return expected quantities.
    """

    places = 7

    def test_poisson(self):
        'Tests the PoissonGen distribution generator.'

        MLS = 42.0

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'PoissonGen(lvl_dens={1/MLS})'
        dist = PoissonGen(lvl_dens=1/MLS)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, self.places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, self.places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x = } to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, self.places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x = } to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at {x=}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at {x=}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at {x=}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at {x=}.')

    def test_wigner(self):
        'Tests the WignerGen distribution generator.'

        MLS = 42.0

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'WignerGen(lvl_dens={1/MLS})'
        dist = WignerGen(lvl_dens=1/MLS)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, self.places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, self.places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x = } to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, self.places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x = } to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at {x=}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at {x=}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at {x=}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at {x=}.')
    
    def test_brody(self):
        'Tests the BrodyGen distribution generator.'

        MLS = 42.0
        w = 0.8

        X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])

        DISTNAME = f'BrodyGen(lvl_dens={1/MLS}, w={w})'
        dist = BrodyGen(lvl_dens=1/MLS, w=w)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, self.places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, self.places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x = } to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, self.places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x = } to infinity.')

        X_ = dist.iF0(MLS*dist.f1(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at {x=}.')

        X_ = dist.iF1(MLS*dist.f2(X))
        for x, x_ in zip(X, X_):
            self.assertAlmostEqual(x, x_, self.places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at {x=}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            self.assertAlmostEqual(R1[i], F0[i]/F1[i], self.places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at {x=}.')
            self.assertAlmostEqual(R2[i], F1[i]/F2[i], self.places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at {x=}.')
    
    def test_missing(self):
        'Tests the MissingGen distribution generator.'

        MLS = 42.0
        pM = 0.2
        err = 1e-6
        places = 4

        X = np.array([0, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])

        DISTNAME = f'MissingGen(lvl_dens={1/MLS:.6f}, pM={pM}, err={err})'
        dist = MissingGen(lvl_dens=1/MLS, pM=pM, err=err)

        I = quad(dist.f0, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I}.')

        I = quad(dist.f1, 0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I}.')

        xf = lambda x: x * dist.f0(x)
        E = quad(xf, 0, np.inf)[0]
        self.assertAlmostEqual(E, MLS, places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E}.')

        for x in X:
            F1 = dist.f1(x) * MLS
            F1_  = quad(dist.f0, x, np.inf)[0]
            self.assertAlmostEqual(F1, F1_, places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x = } to infinity.')

            F2 = dist.f2(x) * MLS
            F2_  = quad(dist.f1, x, np.inf)[0]
            self.assertAlmostEqual(F2, F2_, places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x = } to infinity.')

        Y = dist.f1(X)
        for x, y in zip(X, Y):
            if y <= 1e-30:
                continue
            x_ = dist.iF0(MLS*y)
            self.assertAlmostEqual(x, x_, places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at {x=}.')

        Y = dist.f2(X)
        for x, y in zip(X, Y):
            if y <= 1e-30:
                continue
            x_ = dist.iF1(MLS*y)
            self.assertAlmostEqual(x, x_, places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at {x=}.')

        R1 = dist.r1(X)
        R2 = dist.r2(X)
        F0 = dist.f0(X)
        F1 = dist.f1(X)
        F2 = dist.f2(X)
        for i, x, in enumerate(X):
            if F1[i] != 0.0:
                self.assertAlmostEqual(R1[i], F0[i]/F1[i], places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at {x=}.')
            if F2[i] != 0.0:
                self.assertAlmostEqual(R2[i], F1[i]/F2[i], places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at {x=}.')
    
    def test_high_order(self):
        'Tests the High-Order level-spacing distribution generator.'
        
        MLS = 42.0
        places = 4

        # X = np.array([0, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
        X_norm = np.array([0, 0.01, 0.05, 0.1, 1.0, 10.0, 20.0, 100.0])
        Y = np.linspace(0.0, 1.0, 10)

        ns = [4, 16]
        for n in ns:
            X = X_norm * MLS * (n+1)
            upper_limit = MLS*(n+6) # this should be high enough

            DISTNAME = f'HighOrderSpacingGen(lvl_dens={1/MLS:.5f}, {n=})'
            dist = HighOrderSpacingGen(lvl_dens=1/MLS, n=n)

            I = quad(dist.f0, 0, upper_limit)[0]
            self.assertAlmostEqual(I, 1.0, places, f'"{DISTNAME}.f0" does not integrate to 1, but instead {I} for {n=}.')

            I = quad(dist.f1, 0, upper_limit)[0]
            self.assertAlmostEqual(I, 1.0, places, f'"{DISTNAME}.f1" does not integrate to 1, but instead {I} for {n=}.')

            xf = lambda x: x * dist.f0(x)
            E = quad(xf, 0, upper_limit)[0]
            self.assertAlmostEqual(E, (n+1)*MLS, places, f'"{DISTNAME}.f0" should have a mean level-spacing of {MLS}, but it is instead {E} for {n=}.')

            for x in X:
                
                F1 = dist.f1(x) * MLS * (n+1)
                F1_  = quad(dist.f0, x, upper_limit)[0]
                self.assertAlmostEqual(F1, F1_, places, f'"{DISTNAME}.f1({x})" should be the integral of f0 from {x} to infinity for {n=}.')

                F2 = dist.f2(x) * MLS * (n+1)
                F2_  = quad(dist.f1, x, upper_limit)[0]
                self.assertAlmostEqual(F2, F2_, places, f'"{DISTNAME}.f2({x})" should be the integral of f1 from {x} to infinity for {n=}.')

            X_ = dist.iF0(Y)
            Y_ = MLS*(n+1)*dist.f1(X_)
            for y, y_ in zip(Y, Y_):
                self.assertAlmostEqual(y, y_, places, f'{DISTNAME}.iF0 is not the inverse CDF of f0 when evaluated at {y=} for {n=}.')
            
            # # FIXME: iF1 is unstable
            # X_ = dist.iF1(Y)
            # Y_ = MLS*(n+1)*dist.f2(X_)
            # for y, y_ in zip(Y, Y_):
            #     self.assertAlmostEqual(y, y_, places, f'{DISTNAME}.iF1 is not the inverse CDF of f1 when evaluated at {y = } for {n = }.')

            # # FIXME: Ratios are unstable
            # R1 = dist.r1(X)
            # R2 = dist.r2(X)
            # F0 = dist.f0(X)
            # F1 = dist.f1(X)
            # F2 = dist.f2(X)
            # for i, x, in enumerate(X):
            #     self.assertAlmostEqual(R1[i], F0[i]/F1[i], places, f'{DISTNAME}.r1 is not f0/f1 when evaluated at {x = }.')
            #     self.assertAlmostEqual(R2[i], F1[i]/F2[i], places, f'{DISTNAME}.r2 is not f1/f2 when evaluated at {x = }.')
                
class TestMerger(unittest.TestCase):
    """
    Here, we intend to verify that the level-spacing merger works properly.
    """

    lvl_denses = [1, 5]
    places = 7

    def test_poisson(self):
        """
        Tests the merge function with Poisson distribution. This should return another Poisson
        distribution with a known level-spacing.
        """

        X = np.logspace(-4, 4, 100)
        WL = np.random.dirichlet(self.lvl_denses, size=(X.size,))
        WR = np.random.dirichlet(self.lvl_denses, size=(X.size,))
        distributions = []
        for lvl_dens in self.lvl_denses:
            distribution = PoissonGen(lvl_dens=lvl_dens)
            distributions.append(distribution)
        merged_dist = merge(*distributions)

        lvl_dens_tot = np.sum(self.lvl_denses)
        
        Y_merge = merged_dist.f0(X, WL, WR)
        Y_pred  = lvl_dens_tot * np.exp(-lvl_dens_tot * X)
        for x, y_merge, y_pred in zip(X, Y_merge, Y_pred):
            self.assertAlmostEqual(y_merge, y_pred, self.places, f'Poisson distributions did not merge to the expected distribution at {x=}.')
        
if __name__ == '__main__':
    unittest.main()