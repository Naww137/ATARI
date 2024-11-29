import numpy as np

from ATARI.ModelData.particle import Particle, Neutron

from ATARI.TAZ.TAZ.DataClasses.Spingroups import Spingroup
from ATARI.TAZ.TAZ.DataClasses.Reaction import Reaction
from ATARI.TAZ.TAZ.Theory.Samplers import SampleEnergies

from ATARI.ASTERIODS.mean_parameter_estimation import mean_spacing_averaging, mean_width_averaging, mean_width_CDF_regression

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

class TestMeanParameters(unittest.TestCase):
    """
    Tests that the mean parameter estimation methods are statistically valid.
    """

    ensemble = 'GOE' # Gaussian Orthogonal Ensemble

    def test_spacing_averaging(self):
        """
        Ensures that the the "mean_spacing_averaging" function is providing mean level-spacings within reasonable confidence.
        """

        mls_true = 2.3
        EB = (1e-5, 1000)
        E = SampleEnergies(EB, lvl_dens=1/mls_true, ensemble=self.ensemble)
        mls_mean, mls_std = mean_spacing_averaging(E)
        err = abs(mls_true - mls_mean) / mls_std
        self.assertLess(err, 3.0, f'The "mean_spacing_averaging" function predicts a mean level-spacing beyond reasonable statistics.\n{err = :.3f} > 3.')

    def test_width_averaging(self):
        """
        Ensures that the the "mean_width_averaging" function is providing mean neutron widths within reasonable confidence.
        """

        Target = Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = Neutron

        EB = (1e-5,1e4)
        lvl_dens  = 1.3
        gn2m_true = 44.11355
        gg2m      = 55.00000
        dfn       = 1
        dfg       = 250
        l         = 0
        j         = 3.0

        spingroup = Spingroup(l, j)
        reaction = Reaction(targ=Target, proj=Projectile, lvl_dens=[lvl_dens], gn2m=[gn2m_true], nDOF=[dfn], gg2m=[gg2m], gDOF=[dfg], spingroups=[spingroup], EB=EB)
        res_ladder = reaction.sample(self.ensemble)[0]
        gn2 = reaction.Gn_to_gn2(res_ladder['Gn1'], res_ladder['E'], l)
        gn2m_mean, gn2m_std = mean_width_averaging(gn2)
        err = abs(gn2m_true - gn2m_mean) / gn2m_std
        self.assertLess(err, 3.0, f'The "mean_width_averaging" function predicts a mean neutron width beyond reasonable statistics.\n{err = :.3f} > 3.')

    def test_width_averaging(self):
        """
        Ensures that the the "mean_width_averaging" function is providing mean neutron widths within reasonable confidence.
        """

        Target = Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = Neutron

        EB = (1e-5,2.5e3)
        lvl_dens  = 1.3
        gn2m_true = 44.11355
        gg2m      = 55.00000
        dfn       = 1
        dfg       = 250
        l         = 0
        j         = 3.0

        spingroup = Spingroup(l, j)
        reaction = Reaction(targ=Target, proj=Projectile, lvl_dens=[lvl_dens], gn2m=[gn2m_true], nDOF=[dfn], gg2m=[gg2m], gDOF=[dfg], spingroups=[spingroup], EB=EB)
        res_ladder = reaction.sample(self.ensemble)[0]
        gn2 = reaction.Gn_to_gn2(res_ladder['Gn1'], res_ladder['E'], l)

        gn2m_mean, gn2m_std = mean_width_averaging(gn2)
        err = abs(gn2m_true - gn2m_mean) / gn2m_std
        self.assertLess(err, 3.0, f'The "mean_width_averaging" function predicts a mean level-spacing beyond reasonable statistics.\n{err = :.3f} > 3.')

    def test_width_CDF_regression(self):
        """
        Ensures that the the "mean_width_CDF_regression" function is providing mean neutron widths within reasonable confidence.
        """

        self.skipTest('The standard deviations here do not work.')

        Target = Particle(Z=73, A=181, I=7/2, mass=180.9479958, name='Ta-181')
        Projectile = Neutron

        EB = (1e-5,2.5e3)
        lvl_dens  = 1.3
        gn2m_true = 44.11355
        gg2m      = 55.00000
        dfn       = 1
        dfg       = 250
        l         = 0
        j         = 3.0

        miss_start = 0.05
        thres = 0.0

        spingroup = Spingroup(l, j)
        reaction = Reaction(targ=Target, proj=Projectile, lvl_dens=[lvl_dens], gn2m=[gn2m_true], nDOF=[dfn], gg2m=[gg2m], gDOF=[dfg], spingroups=[spingroup], EB=EB)
        res_ladder = reaction.sample(self.ensemble)[0]

        # Artificially missing resonances:
        randnums = np.random.uniform(0, 1, size=(len(res_ladder),))
        num_res_true = len(res_ladder)
        res_ladder = res_ladder.loc[randnums > res_ladder['Gn1']/miss_start]
        num_res_obs = len(res_ladder)
        miss_frac_true = 1 - num_res_obs / num_res_true

        gn2 = reaction.Gn_to_gn2(res_ladder['Gn1'], res_ladder['E'], l)

        gn2m_mean, gn2m_std, miss_frac_mean, miss_frac_std = mean_width_CDF_regression(gn2, dof=dfn, thres=thres)

        # FIXME: The standard deviation here is not accurate.
        err = abs(gn2m_true - gn2m_mean) / gn2m_std
        self.assertLess(err, 3.0, f'The "mean_width_CDF_regression" function predicts a mean neutron width reasonable statistics.\n{err = :.3f} > 3.')


        # FIXME: missing fraction standard deviation not implemented yet.
        err = abs(miss_frac_true - miss_frac_mean) / miss_frac_std
        self.assertLess(err, 3.0, f'The "ean_width_CDF_regression" function predicts a missing resonance fraction beyond reasonable statistics.\n{err = :.3f} > 3.')
        

if __name__ == '__main__':
    unittest.main()