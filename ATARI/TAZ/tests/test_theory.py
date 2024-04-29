import sys
sys.path.append('../TAZ')
from TAZ import Theory

import numpy as np

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

__doc__ == """
This file tests the general theory equations.
"""
        
class TestPenetration(unittest.TestCase):

    def test_penetration_iterative(self):
        """
        Tests the penetration factor equations by relating the iterative representation
        with the explicit representation of the penetration factor.
        """

        self.skipTest('Not implemented yet.')

    def test_penetration_against_ATARI(self):
        """
        Tests the penetration factor equations against the ATARI implementation to ensure that the
        Values line up.
        """

        E = np.array([1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
        Ls = np.array([0,1,2,3,4])
        mass_targ = 238.0 # amu
        mass_proj = 1.008665 # amu
        ac = 8.0 # sqrt(b)
        P_ATARI = np.array([[1.750029371889e-02, 5.534078787364e-02, 1.750029371889e-01, 5.534078787364e-01, 1.750029371889e+00, 5.534078787364e+00, 1.750029371889e+01],
[5.358003914981e-06, 1.689693659792e-04, 5.200377938284e-03, 1.297496789850e-01, 1.319263811736e+00, 5.359093841956e+00, 1.744333779311e+01],
[1.823643073927e-10, 5.761566732200e-08, 1.805212301306e-05, 5.184189339991e-03, 5.954314338858e-01, 4.996679632395e+00, 1.732888431246e+01],
[2.234128993946e-15, 7.061042166508e-12, 2.220608336326e-08, 6.641127525777e-05, 1.122570818261e-01, 4.421175035874e+00, 1.715582567544e+01],
[1.396401931705e-20, 4.414072108069e-16, 1.390363976487e-11, 4.226136876804e-07, 8.802711790969e-03, 3.598673712290e+00, 1.692243731651e+01]])
        for l in Ls:
            k = Theory.k_wavenumber(mass_targ, E, mass_proj)
            rho = Theory.rho(k, ac)
            P_TAZ = Theory.penetration_factor(rho, l)
            P_ATARI_l = P_ATARI[l,:]
            self.assertTrue(np.allclose(P_TAZ, P_ATARI_l), f"""
TAZ and ATARI Penetration do not match for {l = }:

Energies: {E}
TAZ:      {P_TAZ}
ATARI:    {P_ATARI_l}
""")

if __name__ == '__main__':
    unittest.main()