import sys
sys.path.append('../TAZ')
import TAZ

from copy import copy
import numpy as np

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

class TestBayesMaxLogLikelihoods(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigMaxLikelihoods algorithm is working correctly.
    """

    ensemble = 'Poisson' # Poisson Ensemble
    err = 1e-8
    num_best = 5

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB         = (1e-5, 150)
        cls.false_dens = 1/20.0
        cls.lvl_dens   = [1/5.0, 1/6.0]
        cls.gn2m       = [ 40,  70]
        cls.gg2m       = [55.00000, 55.00000]
        cls.dfn        = [  1,   1]
        cls.dfg        = [250, 250]
        cls.l          = [  0,   0]
        cls.j          = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()

    def test_poisson(self):
        """
        Test that WigMaxLikelihoods returns the spingroups with the maximum prior probabilities
        when provided Poisson level-spacing distributions. This is because Poisson distributions
        have the unique property of providing no addition information beyond frequency.
        """
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        best_spingroup_ladders_prior, best_log_likelihoods_prior = TAZ.PTMaxLogLikelihoods(prior, self.num_best)

        distributions = self.reaction.distributions(dist_type='Poisson')
        best_spingroup_ladders_posterior, best_log_likelihoods_posterior = TAZ.RunMaster.WigMaxLikelihoods(self.E, self.EB, distributions, self.num_best, self.err, prior)
        best_spingroup_ladders_posterior = np.array(best_spingroup_ladders_posterior, dtype=np.int8)

        self.assertTrue(np.all(best_spingroup_ladders_posterior == best_spingroup_ladders_prior), """
The prior and posterior samples do not match with Poisson spacing distributions.
""")
        
class TestBayesMaxLogLikelihoodsSymmetric(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigMaxLikelihoods algorithm is working correctly.
    """

    ensemble = 'GOE' # Gaussian Orthogonal Ensemble
    err = 1e-8
    num_best = 10

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB         = (1e-5, 150)
        cls.false_dens = 0.0 #1/20.0
        cls.lvl_dens   = [1/5.0, 1/5.0]
        cls.gn2m       = [ 40,  40]
        cls.gg2m       = [55.00000, 55.00000]
        cls.dfn        = [  1,   1]
        cls.dfg        = [250, 250]
        cls.l          = [  0,   0]
        cls.j          = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()

    def test_symmetry_no_false(self):
        """
        Test that WigMaxLikelihoods returns the spingroups follow a symmetry principle when the
        mean parameters are the same between spingroups.
        """
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        best_spingroup_ladders, best_log_likelihoods = TAZ.RunMaster.WigMaxLikelihoods(self.E, self.EB, distributions, self.num_best, self.err, prior)

        for i in range(self.num_best//2):
            self.assertTrue(np.all(best_spingroup_ladders[2*i] + best_spingroup_ladders[2*i+1] == 1), """
The WigMaxLikelihoods method does not follow the symmetric principle.
""")
    
if __name__ == '__main__':
    unittest.main()