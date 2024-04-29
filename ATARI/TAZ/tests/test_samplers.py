import sys
sys.path.append('../TAZ')
import TAZ
from TAZ.Theory import wigner_dist, lvl_spacing_ratio_dist, porter_thomas_dist, deltaMehta3, deltaMehtaPredict
from TAZ.Theory import WignerGen, BrodyGen, MissingGen, HighOrderSpacingGen
from utils import chi2_test, chi2_uniform_test

import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import chisquare

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

__doc__ == """
This file tests resonance sampling using level-spacing distributions (Wigner distribution),
level-spacing ratio distributions, Dyson-Mehta Delta-3 statistic, and more. Reduced width samples
are also compared to Porter-Thomas distribution.
"""
        
class TestResonanceGeneration(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample('NNE')[0]

    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E  = self.res_ladder.E.to_numpy()
        lvl_spacing = np.diff(E)

        dist = wigner_dist(scale=self.mls[0], beta=1)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, 'level spacing', 'Wigner distribution')
        
        dist = self.reaction.distributions('Wigner')[0]
        # dist = WignerGen(1/self.mls[0])
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, 'level spacing', 'Wigner distribution')
        
    def test_gamma_widths(self):
        """
        Tests if gamma widths follow the expected Chi-squared distribution.
        """

        NUM_BINS = 40
        Gg = self.res_ladder.Gg.to_numpy()
        gg2 = self.reaction.Gg_to_gg2(Gg)
        dist = porter_thomas_dist(mean=self.gg2m[0], df=self.dfg[0], trunc=0.0)
        chi2_test(dist, gg2, NUM_BINS, self, 0.001, 'gamma widths', 'Porter-Thomas distribution')
        
    def test_neutron_widths(self):
        """
        Tests if neutron widths follow the expected Chi-squared distribution.
        """

        NUM_BINS = 40
        E  = self.res_ladder.E.to_numpy()
        Gn = self.res_ladder.Gn1.to_numpy()
        gn2 = self.reaction.Gn_to_gn2(Gn, E, self.l[0])
        dist = porter_thomas_dist(mean=self.gn2m[0], df=self.dfn[0], trunc=0.0)
        chi2_test(dist, gn2, NUM_BINS, self, 0.001, 'neutron widths', 'Porter-Thomas distribution')
        
class TestGOESampler(unittest.TestCase):

    ensemble = 'GOE' # Gaussian Orthogonal Ensemble
    beta = 1

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()

    def test_dyson_mehta_3(self):
        """
        Tests if the resonance ladder's Dyson-Mehta ∆3 statistic aligns with the prediction.
        """

        D3_calc = deltaMehta3(self.E, self.EB)
        D3_pred = deltaMehtaPredict(len(self.E), 'GOE')

        perc_err = (D3_calc-D3_pred)/D3_pred
        self.assertLess(perc_err, 0.4, f"""
The {self.ensemble} calculated and predicted Dyson-Mehta ∆3 statistic differ by {perc_err:.2%}.
Calculated ∆3 = {D3_calc:.5f}
Predicted ∆3  = {D3_pred:.5f}
""")
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        chi2_bar, p = chi2_uniform_test(self.E, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        dist = wigner_dist(scale=self.mls[0], beta=self.beta)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', 'Wigner distribution')

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_test(dist, ratio, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', None)
        
    def test_high_order_level_spacing(self):
        """
        Tests if the high-order level-spacing samples match the distribution.
        """
        NUM_BINS = 40
        for n in (5, 18):
            lvl_spacing = self.E[n+1:] - self.E[:-(n+1)]
            dist = HighOrderSpacingGen(1/self.mls[0], n)
            chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', f'{n+1}th level-spacing distribution')
        
class TestGUESampler(unittest.TestCase):

    ensemble = 'GUE' # Gaussian Unitary Ensemble
    beta = 2

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        chi2_bar, p = chi2_uniform_test(self.E, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        dist = wigner_dist(scale=self.mls[0], beta=self.beta)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', 'Wigner distribution')

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_test(dist, ratio, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', None)
        
class TestGSESampler(unittest.TestCase):

    ensemble = 'GSE' # Gaussian Symplectic Ensemble
    beta = 4

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile, MLS=cls.mls, gn2m=cls.gn2m, nDOF=cls.dfn, gg2m=cls.gg2m, gDOF=cls.dfg, spingroups=SGs, EB=cls.EB)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        chi2_bar, p = chi2_uniform_test(self.E, NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        dist = wigner_dist(scale=self.mls, beta=self.beta)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', 'Wigner distribution')

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_test(dist, ratio, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', None)
        
class TestBrodySampler(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.w    = [0.8]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile,
                                    MLS=cls.mls,
                                    gn2m=cls.gn2m, nDOF=cls.dfn,
                                    gg2m=cls.gg2m, gDOF=cls.dfg,
                                    spingroups=SGs,
                                    EB=cls.EB,
                                    brody_param=cls.w)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()

    def test_brody(self):
        """
        Tests if the Brody distribution sampler is working correctly.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        dist = self.reaction.distributions('Brody')[0]
        # dist = BrodyGen(1/self.mls[0], w=self.w[0])
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'level spacings', 'Brody distribution')
        
class TestMissingSampler(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    err = 1e-4

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        # Particle Types:
        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        # Mean Parameters
        cls.EB   = (1e-5,5000)
        cls.mls  = [4.3166]
        cls.pM   = [0.2]
        cls.gn2m = [441.1355]
        cls.gg2m = [55.00000]
        cls.dfn  = [1]
        cls.dfg  = [250]
        cls.l    = [0]
        cls.j    = [3.0]

        # 2 Spingroup Case:
        SGs = TAZ.Spingroup.zip(cls.l, cls.j)
        cls.reaction = TAZ.Reaction(targ=Target, proj=Projectile,
                                    MLS=cls.mls,
                                    gn2m=cls.gn2m, nDOF=cls.dfn,
                                    gg2m=cls.gg2m, gDOF=cls.dfg,
                                    spingroups=SGs,
                                    EB=cls.EB,
                                    MissFrac=cls.pM)
        cls.res_ladder = cls.reaction.sample(cls.ensemble)[0]
        cls.E = cls.res_ladder.E.to_numpy()

    def test_missing(self):
        """
        Tests if the Brody distribution sampler is working correctly.
        """

        NUM_BINS = 40
        lvl_spacing = np.diff(self.E)
        dist = self.reaction.distributions('Missing', err=self.err)[0]
        # dist = MissingGen((1-self.pM[0])/self.mls[0], pM=self.pM[0], err=self.err)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'level spacings', 'Missing distribution')
        
class TestMerger(unittest.TestCase):

    ensemble = 'NNE' # Nearest Neighbor Ensemble
    err = 1e-4

    def test_merger(self):
        """
        Tests that merged distributions follow the expected distribution given by the samplers.
        """

        num_bins = 40
        xMax = 3.0

        bins = np.linspace(0, xMax, num_bins+1)

        Target = TAZ.Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = TAZ.Neutron

        EB = (1e-5,2e5)
        lvl_dens  = [1.3, 0.5]
        gn2m  = [44.11355, 33.38697]
        gg2m   = [55.00000, 55.00000]
        dfn   = [1, 1]
        dfg   = [250, 250]
        l     = [0, 0]
        j     = [3.0, 4.0]

        SGs = TAZ.Spingroup.zip(l, j)
        reaction = TAZ.Reaction(targ=Target, proj=Projectile, lvl_dens=lvl_dens, gn2m=gn2m, nDOF=dfn, gg2m=gg2m, gDOF=dfg, spingroups=SGs, EB=EB)
        res_ladder = reaction.sample(self.ensemble)[0]
        E = res_ladder.E.to_numpy()
        level_spacings = np.diff(E)
        freq_obs, _ = np.histogram(level_spacings, bins)

        X = np.linspace(0.0, xMax, 10_000)
        level_spacing_dists = reaction.distributions('Wigner')
        merged_dist = TAZ.Theory.merge(*level_spacing_dists)
        Y = merged_dist.pdf(X)
        I = cumtrapz(Y, X, initial=0.0)
        Ibins = np.interp(bins, X, I)
        freq_exp = np.diff(Ibins)
        freq_exp *=  np.sum(freq_obs) / np.sum(freq_exp)

        chi2, p = chisquare(freq_obs, freq_exp)
        chi2_bar = chi2 / num_bins
        self.assertGreater(p, 0.0001, f"""
The merged level-spacing samples do not follow the merged level-spacing distribution according to the null hypothesis.
Calculated χ² / dof = {chi2_bar:.5f}; p = {p:.5f}
""")

if __name__ == '__main__':
    unittest.main()