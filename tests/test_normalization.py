import sys
sys.path.append('../ATARI')
from ATARI.theory.resonance_statistics import wigner_PDF, chisquare_PDF, general_wigner_pdf, level_spacing_ratio_pdf

import numpy as np
from scipy.integrate import quad

import unittest

class TestNormalizations(unittest.TestCase):
    """
    This class tests if all distributions normalize to the correct value and have the correct means.
    """

    places = 7

    def test_wigner(self):
        MLS = 42.0
        func = lambda x: wigner_PDF(x, MLS)
        I = quad(func, 0.0, np.inf)[0]
        self.assertAlmostEqual(I, 1.0, self.places, f'"wigner_PDF" does not integrate to 1, but instead {I}.')

        funcx = lambda x: x*func(x)
        M = quad(funcx, 0.0, np.inf)[0]
        self.assertAlmostEqual(M, MLS, self.places, f'"wigner_PDF" does not have a mean of {MLS}, but instead {M}.')

        for beta in (1,2,4):
            func = lambda x: general_wigner_pdf(x, MLS, beta=beta)
            I = quad(func, 0.0, np.inf)[0]
            self.assertAlmostEqual(I, 1.0, self.places, f'"general_wigner_pdf" does not integrate to 1, but instead {I} for beta = {beta}.')

            funcx = lambda x: x*func(x)
            M = quad(funcx, 0.0, np.inf)[0]
            self.assertAlmostEqual(M, MLS, self.places, f'"general_wigner_pdf" does not have a mean of {MLS}, but instead {M} for beta = {beta}.')

    def test_level_spacing_ratio(self):
        for beta in (1,2,4):
            func = lambda x: level_spacing_ratio_pdf(x, beta=beta)
            I = quad(func, 0.0, np.inf)[0]
            self.assertAlmostEqual(I, 1.0, self.places, f'"level_spacing_ratio_pdf" does not integrate to 1, but instead {I} for beta = {beta}.')

    def test_chisquare(self):
        GMEAN = 42.0
        for DOF in (1,2,500):
            func = lambda x: chisquare_PDF(x, DOF, GMEAN)
            I = quad(func, 0.0, np.inf)[0]
            self.assertAlmostEqual(I, 1.0, self.places, f'"chisquare_PDF" does not integrate to 1, but instead {I} for DoF = {DOF}.')

            funcx = lambda x: x*func(x)
            M = quad(funcx, 0.0, np.inf)[0]
            self.assertAlmostEqual(M, GMEAN, self.places, f'"chisquare_PDF" does not have a mean of {GMEAN}, but instead {M} for DoF = {DOF}.')

if __name__ == '__main__':
    unittest.main()