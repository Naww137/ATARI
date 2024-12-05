from ATARI.ModelData.particle import Particle, Neutron
from ATARI.ModelData.spingroups import Spingroup
from ATARI.TAZ.DataClasses.Reaction import Reaction
from ATARI.TAZ.PTBayes import PTBayes
from ATARI.TAZ.RunMaster import RunMaster
from utils import chi2_test

import numpy as np

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

class TestBayesSample1(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigSample algorithm is working correctly for 1 spingroup. This will be verified with distribution analysis, including chi-squared goodness of fit on the level-spacing distribution.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    num_trials = 100 # number of sample trials
    num_groups = 1   # number of spingroups

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = Neutron

        # Mean Parameters
        cls.EB = (1e-5,1000)
        cls.false_dens = 1/8.0
        cls.lvl_dens  = [1/4.0]
        cls.gn2m  = [44.11355]
        cls.gg2m   = [55.00000]
        cls.dfn   = [1]
        cls.dfg   = [250]
        cls.l     = [0]
        cls.j     = [3.0]

        SGs = Spingroup.zip(cls.l, cls.j)
        cls.reaction = Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]

        cls.prior, log_likelihood_prior = PTBayes(cls.res_ladder, cls.reaction)
        cls.distributions = cls.reaction.distributions(dist_type='Wigner')
        cls.E = cls.res_ladder.E.to_numpy()
        runMaster = RunMaster(cls.E, cls.EB,
                                  cls.distributions, cls.false_dens,
                                  cls.prior, log_likelihood_prior)
        cls.samples = runMaster.WigSample(cls.num_trials)

    def test_level_densities(self):
        """
        Tests that spingroups are sampled with unbiased frequency.
        """
        # self.skipTest('This test may not actually be statistically correct.')
        lvl_dens_all = np.concatenate((self.lvl_dens, [self.false_dens]))
        counts_exp = (lvl_dens_all / np.sum(lvl_dens_all))
        for g in range(self.num_groups+1):
            counts_obs_trials = np.count_nonzero(self.samples == g, axis=0) / self.samples.shape[0]
            counts_obs_mean = np.mean(counts_obs_trials)
            counts_obs_std = np.std(counts_obs_trials) #/ np.sqrt(self.num_trials)
            err = abs(counts_exp[g] - counts_obs_mean) / counts_obs_std
            self.assertLess(err, 5, f"""
The {g}-spingroup assignment samples do not have the expected frequency according to statistics.
Discrepancy = {err:.5f} standard deviations.
""")

    def test_distributions(self):
        """
        Tests that the sampled assignments produce level-spacings that match the expected level-spacing distributions.
        """
        num_bins = 40
        spacings = [np.empty((0,)), np.empty((0,))]
        for trial in range(self.num_trials):
            sample = self.samples[:,trial]
            for g in range(self.num_groups):
                Eg = self.E[sample == g]
                spacings[g] = np.concatenate((spacings[g], np.diff(Eg)))
        for g in range(self.num_groups):
            chi2_test(self.distributions[g], spacings[g], num_bins, self, threshold=100, quantity_name='level spacing', p_or_chi2='chi2')

class TestBayesSample2(unittest.TestCase):
    """
    The purpose of this test is to verify that the WigSample algorithm is working correctly for 2 spingroups. This will be verified with distribution analysis, including chi-squared goodness of fit on the level-spacing distribution.
    """

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    num_trials = 100 # number of sample trials
    num_groups = 2   # number of spingroups

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = Neutron

        # Mean Parameters
        cls.EB = (1e-5,1000)
        cls.false_dens = 1/20.0
        cls.lvl_dens  = [1/6.0, 1/4.0]
        cls.gn2m  = [44.11355, 33.38697]
        cls.gg2m   = [55.00000, 55.00000]
        cls.dfn   = [1, 1]
        cls.dfg   = [250, 250]
        cls.l     = [0, 0]
        cls.j     = [3.0, 4.0]

        SGs = Spingroup.zip(cls.l, cls.j)
        cls.reaction = Reaction(targ=Target, proj=Projectile, lvl_dens=cls.lvl_dens, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB, false_dens=cls.false_dens)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]

        cls.prior, log_likelihood_prior = PTBayes(cls.res_ladder, cls.reaction)
        cls.distributions = cls.reaction.distributions(dist_type='Wigner')
        cls.E = cls.res_ladder.E.to_numpy()
        runMaster = RunMaster(cls.E, cls.EB,
                                  cls.distributions, cls.false_dens,
                                  cls.prior, log_likelihood_prior)
        cls.samples = runMaster.WigSample(cls.num_trials)

    def test_level_densities(self):
        """
        Tests that spingroups are sampled with unbiased frequency.
        """
        # self.skipTest('This test may not actually be statistically correct.')
        lvl_dens_all = np.concatenate((self.lvl_dens, [self.false_dens]))
        counts_exp = (lvl_dens_all / np.sum(lvl_dens_all))
        for g in range(self.num_groups+1):
            counts_obs_trials = np.count_nonzero(self.samples == g, axis=0) / self.samples.shape[0]
            counts_obs_mean = np.mean(counts_obs_trials)
            counts_obs_std = np.std(counts_obs_trials) #/ np.sqrt(self.num_trials)
            err = abs(counts_exp[g] - counts_obs_mean) / counts_obs_std
            self.assertLess(err, 5, f"""
The {g}-spingroup assignment samples do not have the expected frequency according to statistics.
Discrepancy = {err:.5f} standard deviations.
""")

    def test_distributions(self):
        """
        Tests that the sampled assignments produce level-spacings that match the expected level-spacing distributions.
        """
        num_bins = 40
        spacings = [np.empty((0,)), np.empty((0,))]
        for trial in range(self.num_trials):
            sample = self.samples[:,trial]
            for g in range(self.num_groups):
                Eg = self.E[sample == g]
                spacings[g] = np.concatenate((spacings[g], np.diff(Eg)))
        for g in range(self.num_groups):
            chi2_test(self.distributions[g], spacings[g], num_bins, self, threshold=100, quantity_name='level spacing', p_or_chi2='chi2')
    
if __name__ == '__main__':
    unittest.main()