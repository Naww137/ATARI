import sys
sys.path.append('../ATARI')
from ATARI.theory.resonance_statistics import general_wigner_pdf, level_spacing_ratio_pdf, dyson_mehta_delta_3, dyson_mehta_delta_3_predict
from ATARI.ModelData.particle import Neutron, Ta181
from ATARI.ModelData.particle_pair import Particle_Pair

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.stats import chisquare

import unittest

__doc__ == """
This file tests resonance sampling using level-spacing distributions (Wigner distribution),
level-spacing ratio distributions, Dyson-Mehta Delta-3 statistic, and reduced width distributions
(Porter-Thomas distribution).
"""

class TestLevelSpacingMethods(unittest.TestCase):

    E0 = 200 # eV
    num_res = 10_000 # resonances
    ensemble = 'GOE' # Gaussian Orthogonal Ensemble

    @classmethod
    def setUpClass(cls):
        """
        Generates the ensemble.
        """

        cls.energy_range = [cls.E0, cls.E0+9.0030*cls.num_res]
        Ta_pair = Particle_Pair(isotope = "Ta181",
                                resonance_ladder = pd.DataFrame(),
                                formalism = "XCT",
                                energy_range = cls.energy_range,
                                ac = 0.8127,
                                target=Ta181,
                                projectile=Neutron,
                                l_max = 1
        )

        Ta_pair.map_quantum_numbers(print_out=False)
        cls.mean_level_spacing = 9.0030 # eV
        cls.mean_neutron_width = 452.56615 ; cls.gn2_dof = 1    # eV
        cls.mean_gamma_width   = 32.0      ; cls.gg2_dof = 1000 # eV
        Ta_pair.add_spin_group(Jpi='3.0',
                            J_ID=1,
                            D=cls.mean_level_spacing,
                            gn2_avg=cls.mean_neutron_width,
                            gn2_dof=cls.gn2_dof,
                            gg2_avg=cls.mean_gamma_width,
                            gg2_dof=cls.gg2_dof)

        Ta_pair.sample_resonance_ladder(ensemble=cls.ensemble)
        cls.resonance_ladder = Ta_pair.resonance_ladder

    def test_dyson_mehta_3(self):
        """
        Tests if the resonance ladder's Dyson-Mehta ∆3 statistic aligns with the prediction.
        """

        E = np.array(self.res_ladder['E'])
        D3_calc = dyson_mehta_delta_3(E, self.energy_range)
        D3_pred = dyson_mehta_delta_3_predict(len(E), 'GOE')

        perc_err = (D3_calc-D3_pred)/D3_pred
        self.assertLess(perc_err, 0.2, f"""
The calculated and predicted Dyson-Mehta ∆3 statistic differ by {perc_err:.2%}.
Calculated ∆3 = {D3_calc:.5f}
Predicted ∆3  = {D3_pred:.5f}
""")
    
    def test_uniform_density(self):
        """
        Tests if the resonance ladder is sampling resonances with uniform level densities.
        """

        NUM_BINS = 40
        E = np.array(self.res_ladder['E'])
        num_ergs = len(E)
        obs_counts, bin_edges = np.histogram(E, NUM_BINS)
        exp_counts = (num_ergs/NUM_BINS) * np.ones((NUM_BINS,))
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / NUM_BINS
        self.assertGreater(p, 0.001, f"""
The energies do not follow a uniform density curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
    def test_wigner(self):
        """
        Tests if the resonance ladder follows Wigner distribution.
        """

        NUM_BINS = 40
        # Getting equal probable bins:
        quantiles = np.linspace(0.0, 1.0, NUM_BINS+1)
        with np.errstate(divide='ignore'):
            edges = np.sqrt(-4*self.mean_level_spacing**2/np.pi*np.log(1.0-quantiles))
        # Chi-squared test:
        E = np.array(self.res_ladder['E'])
        num_spacings = len(E) - 1
        lvl_spacing = np.diff(E)
        obs_counts, edges = np.histogram(lvl_spacing, edges)
        exp_counts = (num_spacings / NUM_BINS) * np.ones((NUM_BINS,))
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / NUM_BINS
        self.assertGreater(p, 0.001, f"""
The level-spacings do not follow Wigner distribution according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")

    def test_level_spacing_ratio(self):
        """
        Tests if the resonance ladder follows the level-spacing ratio distribution.
        """

        NUM_BINS = 40
        # Getting equal probable bins:
        quantiles = np.linspace(0.0, 1.0, NUM_BINS+1)[:-1]
        X = np.linspace(0.0, 100, 10_000)
        Y = (X-1)*(X+0.5)*(X+2)/(2*(X**2+X+1)**(1.5)) + 0.5
        edges = np.interp(quantiles,Y,X)
        edges = np.concatenate((edges, [np.inf]))
        # Chi-squared test:
        E = np.array(self.res_ladder['E'])
        num_ratios = len(E) - 2
        lvl_spacing = np.diff(E)
        ratio = lvl_spacing[1:] / lvl_spacing[:-1]
        obs_counts, bin_edges = np.histogram(ratio, edges)
        func = lambda x: level_spacing_ratio_pdf(x, beta=1)
        probs = np.zeros((NUM_BINS,))
        for i in range(NUM_BINS):
            probs[i] = quad(func, bin_edges[i], bin_edges[i+1])[0]
        exp_counts = probs * num_ratios
        chi2, p = chisquare(f_obs=obs_counts, f_exp=exp_counts)
        chi2_bar = chi2 / NUM_BINS
        self.assertGreater(p, 0.001, f"""
The level-spacing ratios do not follow the expected curve according to the null hypothesis.
Calculated chi-squared bar = {chi2_bar:.5f}; p = {p:.5f}
""")
        
# ... need width methods

if __name__ == '__main__':
    unittest.main()