import sys
sys.path.append('../TAZ')
import TAZ
from TAZ.analysis import correlate_probabilities

import numpy as np

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

class TestBayesSampler2SG(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigBayes algorithm is working correctly. This
    will be verified with cross-case verification and special cases with known results.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    err = 1e-8
    num_groups = 2

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB = (1e-5,1000)
        cls.false_dens = 1/15.0
        cls.lvl_dens  = [1/4.3166, 1/4.3166]
        cls.gn2m  = [44.11355, 33.38697]
        cls.gg2m   = [55.00000, 55.00000]
        cls.dfn   = [1, 1]
        cls.dfg   = [250, 250]
        cls.l     = [0, 0]
        cls.j     = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder, cls.true_assignments, _, _ = cls.reaction.sample(cls.ensemble)
        cls.E = cls.res_ladder.E.to_numpy()
    
    def test_poisson(self):
        """
        Here, we intend to verify that WigBayes returns the prior when provided Poisson
        distributions.
        """
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Poisson')
        runmaster = TAZ.RunMaster(self.E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)
        posterior = runmaster.WigBayes()
        
        perrors = abs(posterior - prior) / prior
        perror_max = np.max(perrors)
        perror_mean = np.mean(perrors)
        self.assertTrue(np.allclose(prior, posterior, rtol=1e-6, atol=1e-15), f"""
The prior and posterior were not the same when Poisson distribution was used.
Maximum error = {perror_max:.6%}
Mean error    = {perror_mean:.6%}
""")
    
    def test_probability_frequency(self):
        """
        Here, we intend to verify that WigBayes returns probabilities that match the fraction
        of resonances with said probability within statistical error.
        """
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        runmaster = TAZ.RunMaster(self.E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)
        posterior = runmaster.WigBayes()

        Qs = correlate_probabilities(posterior, self.true_assignments)
        for g, Q in enumerate(Qs):
            errlim = 0.01
            self.assertTrue(np.all(Q > errlim), f"""
WigBayes probabilities do not match the frequency of correct sampling to within {errlim} standard deviations for group {g} of {self.num_groups}.
Lowest probability density = {np.min(Q):.5f}.
""")

class TestBayesSampler1or2SG(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    eps = 0.05
    err = 1e-8

    def test_1_2_sg_match(self):
        """
        Here, we intend to verify that the 1-spingroup Encore algorithms converges to the
        2-spingroup Encore algorithms when the one spingroup is very infrequent (i.e. low
        level-density).
        """
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron
        EB = (1e-5,1000)
        false_dens = 1 / 6.0
        lvl_dens_tot = 1 / 4.0
        lvl_dens = [lvl_dens_tot*(1-self.eps), lvl_dens_tot*self.eps]
        gn2m  = [44.11355, 33.38697]
        gg2m  = [55.00000, 55.00000]
        dfn   = [1, 1]
        dfg   = [250, 250]
        l     = [0, 0]
        j     = [3.0, 4.0]
        SGs = TAZ.Spingroup.zip(l, j)
        reaction2 = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=lvl_dens, gn2m=gn2m, nDOF=dfn, gg2m=gg2m, gDOF=dfg, spingroups=SGs, EB=EB, false_dens=false_dens)
        reaction1 = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=[lvl_dens_tot], gn2m=gn2m[:1], nDOF=dfn[:1], gg2m=gg2m[:1], gDOF=dfg[:1], spingroups=SGs[:1], EB=EB, false_dens=false_dens)
        res_ladder, true_assignments, _, _ = reaction1.sample(self.ensemble)
        E = res_ladder.E.to_numpy()

        prior1, log_likelihood_prior1 = TAZ.PTBayes(res_ladder, reaction1)
        distributions1 = reaction1.distributions(dist_type='Wigner')
        runmaster1 = TAZ.RunMaster(E, EB, distributions1, false_dens, prior1, log_likelihood_prior1, err=self.err)
        posterior1 = runmaster1.WigBayes()
        
        prior2, log_likelihood_prior2 = TAZ.PTBayes(res_ladder, reaction2)
        distributions2 = reaction2.distributions(dist_type='Wigner')
        runmaster2 = TAZ.RunMaster(E, EB, distributions2, false_dens, prior2, log_likelihood_prior2, err=self.err)
        posterior2 = runmaster2.WigBayes()

        # taking out new spingroup:
        posterior2_alt = posterior2[:,::2]
        posterior2_alt /= np.sum(posterior2_alt, axis=1, keepdims=True)

        err_true  = np.mean(posterior1[:,0] - posterior2_alt[:,0])
        err_false = np.mean(posterior1[:,1] - posterior2_alt[:,1])
        self.assertLess(err_true , self.eps, f'\nThe 1-2 WigBayes test failed for the true group.\nerr = {err_true}\n')
        self.assertLess(err_false, self.eps, f'\nThe 1-2 WigBayes test failed for the false group.\nerr = {err_false}\n')

class TestBayesSampler2or3SG(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    eps = 0.02
    err = 1e-8

    def test_2_3_sg_match(self):
        """
        Here, we intend to verify that the 2-spingroup Encore algorithms converges to the
        3-spingroup Encore algorithms when the one spingroup is very infrequent (i.e. low
        level-density).
        """
        self.skipTest('Not implemented')
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        runmaster = TAZ.RunMaster(self.E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)

class TestBayesSamplerNoFalse(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigBayes algorithm is working correctly when
    no false resonances are present.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    err = 1e-8
    num_groups = 2

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB = (1e-5,1000)
        cls.false_dens = 0.0
        cls.lvl_dens  = [1/5.0, 1/4.0]
        cls.gn2m  = [44.11355, 33.38697]
        cls.gg2m   = [55.00000, 55.00000]
        cls.dfn   = [1, 1]
        cls.dfg   = [250, 250]
        cls.l     = [0, 0]
        cls.j     = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder, cls.true_assignments, _, _ = cls.reaction.sample(cls.ensemble)
        cls.E = cls.res_ladder.E.to_numpy()
    
    def test_false(self):
        """
        Here, we intend to verify that WigBayes returns near-zero false probabilities when the
        false level-density is zero.
        """
        prior, log_likelihood_prior = TAZ.PTBayes(self.res_ladder, self.reaction)
        distributions = self.reaction.distributions(dist_type='Wigner')
        runmaster = TAZ.RunMaster(self.E, self.EB, distributions, self.false_dens, prior, log_likelihood_prior, err=self.err)
        posterior = runmaster.WigBayes()
        
        perrors = posterior[:,-1]
        perror_max = np.max(perrors)
        perror_mean = np.mean(perrors)
        self.assertTrue(perror_max < 1e-4, f"""
WigBayes returns non-zero false probabilities when the false level-density is zero.
Maximum error = {perror_max:.6%}
Mean error    = {perror_mean:.6%}
""")
    
if __name__ == '__main__':
    unittest.main()