from ATARI.utils.stats import corr2cov, cov2corr
import unittest
import numpy as np


__doc__ == """
"""

# os.chdir(os.path.dirname(__file__))
        

class TestCovCorr(unittest.TestCase):

    def test_cov2corr(self):
        Cov = np.array([[4, 2],
                        [2, 16]])
        Corr = np.array([[1, 2/8],
                         [2/8, 1]])
        self.assertTrue(np.all(Corr == cov2corr(Cov)))

    def test_corr2cov(self):
        Corr = np.array([[1, 2/8],
                         [2/8, 1]])
        std = np.array([2,4])
        Cov = np.array([[4, 2],
                        [2, 16]])
        self.assertTrue(np.all(Cov == corr2cov(Corr, std**2)))






if __name__ == '__main__':
    unittest.main()