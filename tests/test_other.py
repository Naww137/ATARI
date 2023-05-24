
from ATARI.utils.stats import chi2_val
from ATARI.theory.experimental import trans_2_xs
import numpy as np

import unittest

# def test_trans_2_xs(unittest.TestCase)

Tdat = np.array([0.1,0.8])
Tcov = np.array([[0.1,0.0],[0.0,0.1]])
XSdat, XScov = trans_2_xs(Tdat, n=.01, n_unc=0.1, CovT=Tcov)

Tfit = np.array([0.101, 0.799])
XSfit, _ = trans_2_xs(Tfit, n=.01, n_unc=0.1, CovT=None)

print(np.isclose( chi2_val(Tfit, Tdat, Tcov), chi2_val(XSfit, XSdat, XScov), rtol=1e-1))

# print((2e-5 - 1.96e-5)/2e-5)