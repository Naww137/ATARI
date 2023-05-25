import numpy as np
from pandas import DataFrame
from ATARI.utils.io.pointwise import PointwiseContainer
import unittest

class TestPointwiseContainer(unittest.TestCase):


    def test_add_experimental(self):

        PwConObj = PointwiseContainer(
                DataFrame({'E':[10,0]}),
                DataFrame({'E':[10,5,0]})
                )
        PwConObj.add_experimental(
            exp_df = DataFrame({'E':[10,0], 'exp_trans':[0.01, 0.8]}),
            CovT = DataFrame({'10':[.01,0],'0':[0,.01]}, index=['10','0']),
            exp_parm = type('TestClass', (), {'n': 1, 'dn': 0.0, 'blackthreshold': 0.1})()
        )

        # self.assertTrue(    np.isnan( PwConObj.exp.exp_xs.loc[PwConObj.exp.E==10] ).item() )
        # self.assertTrue(    np.isnan( PwConObj.exp.exp_xs_unc.loc[PwConObj.exp.E==10] ).item() )
        # self.assertTrue(     np.all(np.isnan(PwConObj.CovXS.loc[0:10, 10])) )
        # self.assertTrue(     np.all(np.isnan(PwConObj.CovXS.loc[10, 0:10])) )

        # self.assertFalse(   np.isnan( PwConObj.exp.exp_xs.loc[PwConObj.exp.E==0] ).item() )
        # self.assertFalse(   np.isnan( PwConObj.CovXS.loc[0,0] ).item() )

        self.assertTrue(    np.isnan(PwConObj.exp.exp_xs.loc[PwConObj.exp.E==10]).item() )
        self.assertTrue(    np.isnan(PwConObj.exp.exp_xs_unc.loc[PwConObj.exp.E==10]).item() )
        self.assertTrue(     np.all(np.isnan(PwConObj.CovXS.loc[0:10, 10])) )
        self.assertTrue(     np.all(np.isnan(PwConObj.CovXS.loc[10, 0:10])) )

        self.assertFalse(   np.isnan(PwConObj.exp.exp_xs.loc[PwConObj.exp.E==0]).item() )
        self.assertFalse(   np.isnan(PwConObj.CovXS.loc[0,0]).item() )




unittest.main()