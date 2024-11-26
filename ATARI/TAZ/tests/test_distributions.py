import sys
sys.path.append('../TAZ')
from TAZ.Theory import wigner_dist, lvl_spacing_ratio_dist, porter_thomas_dist, semicircle_dist

import numpy as np
from scipy.integrate import quad

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

class TestDistributions(unittest.TestCase):
    """
    This class tests if all distributions satisfy expected qualities, such as being normalized,
    having the expected mean or median, etc.
    """

    places = 7

    def test_wigner(self):
        'Tests Wigner distribution.'
        MLS = 42.0
        X = [10.0, 30.0, 50.0, 100.0]
        for beta in (1,2,4):
            wigner = wigner_dist(scale=MLS, beta=beta)
            I = quad(wigner.pdf, 0.0, np.inf)[0]
            self.assertAlmostEqual(I, 1.0, self.places, f'"wigner_dist.pdf" does not integrate to 1, but instead {I} for beta = {beta}.')

            funcx = lambda x: x*wigner.pdf(x)
            M = quad(funcx, 0.0, np.inf)[0]
            self.assertAlmostEqual(M, MLS, self.places, f'"wigner_dist.pdf" does not have a mean of {MLS}, but instead {M} for beta = {beta}.')

            for x in X:
                f1 = wigner.cdf(x)
                f2 = 1-wigner.sf(x)
                self.assertAlmostEqual(f1, f2, self.places, f'"wigner_dist.sf" does not align with "wigner_dist.cdf" for beta = {beta}.')

                I = quad(wigner.pdf, 0.0, x)[0]
                self.assertAlmostEqual(I, f1, self.places, f'"wigner_dist.cdf" does not match integration of "wigner_dist.pdf" for beta = {beta}.')

    def test_level_spacing_ratio(self):
        'Tests the level-spacing ratio distribution.'
        X = [0.1, 0.4, 0.7, 1.7]
        for beta in (1,2,4):
            lvl_spacing_ratio = lvl_spacing_ratio_dist(beta=beta)
            I = quad(lvl_spacing_ratio.pdf, 0.0, np.inf)[0]
            self.assertAlmostEqual(I, 1.0, self.places, f'"lvl_spacing_ratio_dist.pdf" does not integrate to 1, but instead {I} for beta = {beta}.')

            median = lvl_spacing_ratio_dist.median(beta=beta)
            self.assertAlmostEqual(median, 1.0, self.places, f'"lvl_spacing_ratio_dist" does have a median of 1, but instead {median} for beta = {beta}.')
            
            for x in X:
                f1 = lvl_spacing_ratio.cdf(x)
                I = quad(lvl_spacing_ratio.pdf, 0.0, x)[0]
                self.assertAlmostEqual(I, f1, self.places, f'"lvl_spacing_ratio_dist.cdf" does not match integration of "lvl_spacing_ratio_dist.pdf" for beta = {beta}.')

                f2 = 1 - lvl_spacing_ratio.cdf(1/x)
                self.assertAlmostEqual(f1, f2, self.places, f'"lvl_spacing_ratio_dist.cdf" does not follow the reciprocity rule of ratio distributions for beta = {beta}.')

    def test_porter_thomas(self):
        'Tests Porter-Thomas distribution.'
        GMEAN = 42.0
        X = [10.0, 30.0, 50.0, 100.0]
        for DOF in (1,2,500):
            for TRUNC in (0.0, 10.0):
                porter_thomas = porter_thomas_dist(mean=GMEAN, df=DOF, trunc=TRUNC)
                I = quad(porter_thomas.pdf, 0.0, np.inf)[0]
                self.assertAlmostEqual(I, 1.0, self.places, f'"porter_thomas_dist.pdf" does not integrate to 1, but instead {I} for DoF = {DOF} and trunc = {TRUNC}.')

                if TRUNC == 0.0:
                    funcx = lambda x: x*porter_thomas.pdf(x)
                    M = quad(funcx, 0.0, np.inf)[0]
                    self.assertAlmostEqual(M, GMEAN, self.places, f'"porter_thomas_dist.pdf" does not have a mean of {GMEAN}, but instead {M} for DoF = {DOF} and trunc = {TRUNC}.')

                for x in X:
                    cdf = lambda x: porter_thomas.cdf(x)
                    sf  = lambda x: porter_thomas.sf(x)
                    f1 = cdf(x)
                    f2 = 1-sf(x)
                    self.assertAlmostEqual(f1, f2, self.places, f'"porter_thomas_dist.sf" does not align with "porter_thomas_dist.cdf" for DoF = {DOF} and trunc = {TRUNC}.')

                    I = quad(porter_thomas.pdf, 0.0, x)[0]
                    self.assertAlmostEqual(I, f1, self.places, f'"porter_thomas_dist.cdf" does not match integration of "porter_thomas_dist.pdf" for DoF = {DOF} and trunc = {TRUNC}.')

    def test_semicircle(self):
        "Tests Wigner's semicircle distribution."
        semicircle = semicircle_dist()
        I = quad(semicircle.pdf, -1.0, 1.0)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"semicircle.pdf" does not integrate to 1, but instead {I}.')

        for x in (-0.7, -0.2, 0.1, 0.9):
            f1 = semicircle.cdf(x)
            I = quad(semicircle.pdf, -1.0, x)[0]
            self.assertAlmostEqual(I, f1, self.places, f'"semicircle.cdf" does not match integration of "semicircle.pdf".')

if __name__ == '__main__':
    unittest.main()