import sys
sys.path.append('../TAZ')
import TAZ
from TAZ import analysis

import numpy as np

import unittest

class TestPTBayes(unittest.TestCase):
    """
    Tests that PTBayes provides accurate probabilities given the provided width information.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    num_groups = 4

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        # cls.EB = (1e-5,1e6)
        # cls.false_dens = 1/15.0
        # cls.lvl_dens  = [1/4.0, 1/6.0]
        # cls.gn2m  = [45.0, 30.0]
        # cls.gg2m   = [55.0, 52.0]
        # cls.dfn   = [1, 1]
        # cls.dfg   = [100, 80]
        # cls.l     = [0, 0]
        # cls.j     = [3.0, 4.0]
        # cls.EB = (1e-5,1e5)
        cls.EB = (1e-5,1e4)
        cls.false_dens = 1/15.0
        cls.lvl_dens  = [1/4.0, 1/5.0, 1/6.0, 1/7.0]
        cls.gn2m  = [44.11355, 33.38697, 60.0, 25.0]
        cls.gg2m   = [55.0, 52.0, 50.0, 60.0]
        cls.dfn   = [1, 1, 1, 2]
        cls.dfg   = [100, 100, 40, 50]
        cls.l     = [0, 0, 1, 1]
        cls.j     = [3.0, 4.0, 3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder, cls.true_assignments, _, _ = cls.reaction.sample(cls.ensemble)

    def test_probability_frequency(self):
        """
        Here, we intend to verify that PTBayes returns probabilities that match the fraction
        of resonances with said probability within statistical error.
        """

        # self.skipTest('Not implemented yet')
        probabilities, log_likelihood = TAZ.PTBayes(self.res_ladder, self.reaction, gamma_width_on=False)

        Qs = analysis.correlate_probabilities(probabilities, self.true_assignments)
        for g, Q in enumerate(Qs):
            errlim = 0.01
            self.assertTrue(np.all(Q > errlim), f"""
PTBayes probabilities do not match the frequency of correct sampling to within {errlim} standard deviations for gamma width information off for group {g} of {self.num_groups}.
Lowest probability density = {np.min(Q):.5f}.
""")
            
        probabilities, log_likelihood = TAZ.PTBayes(self.res_ladder, self.reaction, gamma_width_on=True)

        Qs = analysis.correlate_probabilities(probabilities, self.true_assignments)
        for g, Q in enumerate(Qs):
            errlim = 0.01
            self.assertTrue(np.all(Q > errlim), f"""
PTBayes probabilities do not match the frequency of correct sampling to within {errlim} standard deviations for gamma width information off for group {g} of {self.num_groups}.
Lowest probability density = {np.min(Q):.5f}.
""")

if __name__ == '__main__':
    unittest.main()