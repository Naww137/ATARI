import sys
sys.path.append('../ATARI')
from ATARI.theory.resonance_statistics import dyson_mehta_delta_3, dyson_mehta_delta_3_predict
from ATARI.ModelData.particle import Neutron, Ta181
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.theory.distributions import wigner_dist, lvl_spacing_ratio_dist, porter_thomas_dist

from utils import chi2_test, chi2_uniform_test

import numpy as np
import pandas as pd
from scipy.stats import chisquare

import unittest

__doc__ == """
This file tests resonance sampling using level-spacing distributions (Wigner distribution),
level-spacing ratio distributions, Dyson-Mehta Delta-3 statistic, and reduced width distributions
(Porter-Thomas distribution).
"""
        
class TestResonanceGeneration(unittest.TestCase):
    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'NNE' # Nearest Neighbor Ensemble

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        cls.energy_range = [cls.E0, cls.E0+9.0030*cls.num_res]
        cls.part_pair = Particle_Pair(isotope = "Ta181",
                                    resonance_ladder = pd.DataFrame(),
                                    formalism = "XCT",
                                    energy_range = cls.energy_range,
                                    ac = 0.8127,
                                    target=Ta181,
                                    projectile=Neutron,
                                    l_max = 1
        )

        cls.part_pair.map_quantum_numbers(print_out=False)
        cls.mean_level_spacing = 9.0030 # eV
        cls.mean_neutron_width = 452.56615 ; cls.gn2_dof = 1    # meV
        cls.mean_gamma_width   = 32.0      ; cls.gg2_dof = 200 # meV

        cls.L = 0
        cls.part_pair.add_spin_group(Jpi='3.0',
                                   J_ID=1,
                                   D=cls.mean_level_spacing,
                                   gn2_avg=cls.mean_neutron_width,
                                   gn2_dof=cls.gn2_dof,
                                   gg2_avg=cls.mean_gamma_width,
                                   gg2_dof=cls.gg2_dof)

        cls.part_pair.sample_resonance_ladder(ensemble=cls.ensemble)
        cls.res_ladder = cls.part_pair.resonance_ladder

    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mean_level_spacing, beta=1)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacing', 'Wigner distribution', 'p')
        
    def test_gamma_widths(self):
        """
        Tests if gamma widths follow the expected Chi-squared distribution.
        """

        NUM_BINS = 40
        Gg = self.res_ladder['Gg']
        gg2 = self.part_pair.Gg_to_gg2(Gg)
        dist = porter_thomas_dist(mean=self.mean_gamma_width, df=self.gg2_dof, trunc=0.0)
        chi2_test(dist, abs(gg2), NUM_BINS, self, 0.001, 'gamma widths', 'Porter-Thomas distribution', 'p')
        
        obs_counts = np.bincount(np.array(gg2>=0, dtype=int), minlength=2)
        exp_counts = np.array([0.5, 0.5]) * len(gg2)
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / 2
        self.assertGreater(p, 0.001, f"""
The gamma (capture) widths do not have 50% positive-negative split.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_neutron_widths(self):
        """
        Tests if neutron widths follow the expected Chi-squared distribution.
        """

        NUM_BINS = 40
        E  = np.array(self.res_ladder['E'])
        Gn = np.array(self.res_ladder['Gn1'])
        gn2 = self.part_pair.Gn_to_gn2(Gn, E, self.L)
        dist = porter_thomas_dist(mean=self.mean_neutron_width, df=self.gn2_dof, trunc=0.0)
        chi2_test(dist, abs(gn2), NUM_BINS, self, 0.001, 'neutron widths', 'Porter-Thomas distribution', 'p')
        
        obs_counts = np.bincount(np.array(gn2>=0, dtype=int), minlength=2)
        exp_counts = np.array([0.5, 0.5]) * len(gn2)
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / 2
        self.assertGreater(p, 0.001, f"""
The neutron widths do not have 50% positive-negative split.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
class TestGOESampler(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GOE' # Gaussian Orthogonal Ensemble
    beta = 1

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        cls.energy_range = [cls.E0, cls.E0+9.0030*cls.num_res]
        cls.part_pair = Particle_Pair(isotope = "Ta181",
                                    resonance_ladder = pd.DataFrame(),
                                    formalism = "XCT",
                                    energy_range = cls.energy_range,
                                    ac = 0.8127,
                                    target=Ta181,
                                    projectile=Neutron,
                                    l_max = 1
        )

        cls.part_pair.map_quantum_numbers(print_out=False)
        cls.mean_level_spacing = 9.0030 # eV
        cls.mean_neutron_width = 452.56615 ; cls.gn2_dof = 1    # eV
        cls.mean_gamma_width   = 32.0      ; cls.gg2_dof = 1000 # eV

        cls.L = 0
        cls.part_pair.add_spin_group(Jpi='3.0',
                                   J_ID=1,
                                   D=cls.mean_level_spacing,
                                   gn2_avg=cls.mean_neutron_width,
                                   gn2_dof=cls.gn2_dof,
                                   gg2_avg=cls.mean_gamma_width,
                                   gg2_dof=cls.gg2_dof)

        cls.part_pair.sample_resonance_ladder(ensemble=cls.ensemble)
        cls.res_ladder = cls.part_pair.resonance_ladder

    def test_dyson_mehta_3(self):
        """
        Tests if the resonance ladder's Dyson-Mehta ∆3 statistic aligns with the prediction.
        """

        E = np.array(self.res_ladder['E'])
        D3_calc = dyson_mehta_delta_3(E, self.energy_range)
        D3_pred = dyson_mehta_delta_3_predict(len(E), 'GOE')

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
        chi2_bar, p = chi2_uniform_test(np.array(self.res_ladder['E']), NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mean_level_spacing, beta=self.beta)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', 'Wigner distribution', 'p')

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_test(dist, ratio, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', None, 'p')
        
class TestGUESampler(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GUE' # Gaussian Unitary Ensemble
    beta = 2

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        cls.energy_range = [cls.E0, cls.E0+9.0030*cls.num_res]
        cls.part_pair = Particle_Pair(isotope = "Ta181",
                                    resonance_ladder = pd.DataFrame(),
                                    formalism = "XCT",
                                    energy_range = cls.energy_range,
                                    ac = 0.8127,
                                    target=Ta181,
                                    projectile=Neutron,
                                    l_max = 1
        )

        cls.part_pair.map_quantum_numbers(print_out=False)
        cls.mean_level_spacing = 9.0030 # eV
        cls.mean_neutron_width = 452.56615 ; cls.gn2_dof = 1    # eV
        cls.mean_gamma_width   = 32.0      ; cls.gg2_dof = 1000 # eV

        cls.L = 0
        cls.part_pair.add_spin_group(Jpi='3.0',
                                   J_ID=1,
                                   D=cls.mean_level_spacing,
                                   gn2_avg=cls.mean_neutron_width,
                                   gn2_dof=cls.gn2_dof,
                                   gg2_avg=cls.mean_gamma_width,
                                   gg2_dof=cls.gg2_dof)

        cls.part_pair.sample_resonance_ladder(ensemble=cls.ensemble)
        cls.res_ladder = cls.part_pair.resonance_ladder
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        chi2_bar, p = chi2_uniform_test(np.array(self.res_ladder['E']), NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mean_level_spacing, beta=self.beta)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', 'Wigner distribution', 'p')

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_test(dist, ratio, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', None, 'p')
        
class TestGSESampler(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GSE' # Gaussian Symplectic Ensemble
    beta = 4

    @classmethod
    def setUpClass(cls):
        """
        Generates the resonances.
        """

        cls.energy_range = [cls.E0, cls.E0+9.0030*cls.num_res]
        cls.part_pair = Particle_Pair(isotope = "Ta181",
                                    resonance_ladder = pd.DataFrame(),
                                    formalism = "XCT",
                                    energy_range = cls.energy_range,
                                    ac = 0.8127,
                                    target=Ta181,
                                    projectile=Neutron,
                                    l_max = 1
        )

        cls.part_pair.map_quantum_numbers(print_out=False)
        cls.mean_level_spacing = 9.0030 # eV
        cls.mean_neutron_width = 452.56615 ; cls.gn2_dof = 1    # eV
        cls.mean_gamma_width   = 32.0      ; cls.gg2_dof = 1000 # eV

        cls.L = 0
        cls.part_pair.add_spin_group(Jpi='3.0',
                                   J_ID=1,
                                   D=cls.mean_level_spacing,
                                   gn2_avg=cls.mean_neutron_width,
                                   gn2_dof=cls.gn2_dof,
                                   gg2_avg=cls.mean_gamma_width,
                                   gg2_dof=cls.gg2_dof)

        cls.part_pair.sample_resonance_ladder(ensemble=cls.ensemble)
        cls.res_ladder = cls.part_pair.resonance_ladder
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        chi2_bar, p = chi2_uniform_test(np.array(self.res_ladder['E']), NUM_BINS)
        self.assertGreater(p, 0.001, f"""
The {self.ensemble} energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        dist = wigner_dist(scale=self.mean_level_spacing, beta=self.beta)
        chi2_test(dist, lvl_spacing, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', 'Wigner distribution', 'p')

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        dist = lvl_spacing_ratio_dist(beta=self.beta)
        chi2_test(dist, ratio, NUM_BINS, self, 0.001, f'{self.ensemble} level spacings', None, 'p')

if __name__ == '__main__':
    unittest.main()