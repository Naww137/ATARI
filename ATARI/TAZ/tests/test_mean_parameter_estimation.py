import sys
sys.path.append('../TAZ')
from TAZ.Theory.MeanParameterEstimation import MeanSpacingAveraging
from TAZ.Theory.Samplers import SampleEnergies

import warnings
warnings.filterwarnings('error', category=RuntimeWarning)

import unittest

class TestMeanSpacings(unittest.TestCase):
    """
    Tests that the mean parameter estimation methods are statistically valid.
    """

    ensemble = 'GOE' # Gaussian Orthogonal Ensemble

    def test_spacing_averaging(self):
        """
        Ensures that the the "MeanSpacingAveraging" function is providing mean level-spacings
        within reasonable confidence.
        """

        mls_true = 2.3
        EB = (1e-5, 1000)
        E = SampleEnergies(EB, lvl_dens=1/mls_true, ensemble=self.ensemble)
        mls_mean, mls_std = MeanSpacingAveraging(E)
        err = abs(mls_true - mls_mean) / mls_std
        self.assertLess(err, 3.0, f'The "MeanSpacingAveraging" function predicts a mean level-spacing beyond reasonable statistics.\n{err = :.3f} > 3.')

if __name__ == '__main__':
    unittest.main()