
import unittest
import numpy as np
import pandas as pd
from ATARI.ModelData.measurement_models.transmission_rpi import get_covT, reduce_raw_count_data, inverse_reduction, transmission
from ATARI.syndat.general_functions import neutron_background_function  #!! I should move this function to ATARI.theory.experimental.py



__doc__ == """
This file tests the linear, first order covariance propagation from raw observables to experimental object (transmisison or yield).
It does so by comparison to monte carlo random sampling approach
"""


class TestTransmissionRPICovariance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ndat = 100
        cls.samples = 50000
        # choose a nice tof range in microseconds
        cls.tof = np.linspace(20e-6,1000e-6,cls.ndat)

        # choose true and open counts to be large s.t. we are in a mostly linear regime
        cls.cts_true = 1.9e10
        cls.dcts = np.sqrt(cls.cts_true)
        cls.cto_true = 2e10
        cls.dcto = np.sqrt(cls.cto_true)

        # if bw == trig, then counts = count rate
        cls.bw = np.array([1e-7]*cls.ndat)
        cls.trig = 1e7

        # set measurement parameters and uncertainty
        cls.m1     = (1,            0.0016)
        cls.m2     = (1,            0.0008)
        cls.m3     = (1,            0.0018)
        cls.m4     = (1,            0.0005)
        cls.ks     = (0.563,        0.002402339737495515)
        cls.ko     = (1.471,        0.005576763648617445)
        cls.b0s    = (9.9,          0.01)
        cls.b0o    = (13.4,         0.07)
        cls.a_b    = ([582.7768594580712, 0.05149689096209191],
                    [[1.14395753e+03,  1.42659922e-1],
                        [1.42659922e-1,   2.19135003e-05]])

    def monte_carlo(self, bkg_func):

        save_T = []
        for i in range(self.samples):

            # counts sampled as poisson
            cts = np.random.default_rng().normal(self.cts_true, scale=np.sqrt(self.cts_true), size = self.ndat)
            cto = np.random.default_rng().normal(self.cto_true, scale=np.sqrt(self.cto_true), size = self.ndat)
            df_open = pd.DataFrame({'tof': self.tof, 'c':cto})

            # measurement parameters sampled as normal
            m1_i = np.random.default_rng().normal(loc=self.m1[0], scale=self.m1[1])
            m2_i = np.random.default_rng().normal(loc=self.m2[0], scale=self.m2[1])
            m3_i = np.random.default_rng().normal(loc=self.m3[0], scale=self.m3[1])
            m4_i = np.random.default_rng().normal(loc=self.m4[0], scale=self.m4[1])
            ks_i = np.random.default_rng().normal(loc=self.ks[0], scale=self.ks[1])
            ko_i = np.random.default_rng().normal(loc=self.ko[0], scale=self.ko[1])
            b0s_i = np.random.default_rng().normal(loc=self.b0s[0], scale=self.b0s[1])
            b0o_i = np.random.default_rng().normal(loc=self.b0o[0], scale=self.b0o[1])
            a_b_i = np.random.default_rng().multivariate_normal(mean=self.a_b[0], cov=self.a_b[1])

            # build background and monitor array with sampled parameters
            monitor_array_i = [m1_i, m2_i, m3_i, m4_i]
            Bi_i = neutron_background_function(self.tof*1e-3, a_b_i[0], a_b_i[1], bkg_func=bkg_func)

            save_T.append(transmission(cts, cto, Bi_i, ks_i, ko_i, b0s_i, b0o_i, monitor_array_i))
        
        ie_sample = np.array(save_T)
        CovT = np.cov(ie_sample.T)
        avg = np.mean(ie_sample, axis=0)
        std = np.std(ie_sample, axis=0)

        return avg, std, CovT

    def linear_first_order(self, bkg_func):

        monitor_array = [self.m1[0], self.m2[0], self.m3[0], self.m4[0]]
        Bi = neutron_background_function(self.tof*1e-3, self.a_b[0][0], self.a_b[0][1], bkg_func=bkg_func)
        sys_unc = [self.ks[1], self.ko[1], self.b0s[1], self.b0o[1], self.m1[1], self.m2[1], self.m3[1], self.m4[1]]

        exp, unc_data, rates = reduce_raw_count_data(self.tof, [self.cts_true], [self.cto_true], [self.dcts], [self.dcto], self.bw, self.trig, self.trig,
                                                    self.a_b[0][0], self.a_b[0][1], self.ks[0], self.ko[0],
                                                    Bi,self.b0s[0],self.b0o[0],
                                                    monitor_array,
                                                    sys_unc,
                                                    self.a_b[1],
                                                    True,
                                                    bkg_func)

        diag_stat, diag_sys, Jac_sys, Cov_sys = unc_data
        CovT_sys = Jac_sys.T @ Cov_sys @ Jac_sys
        CovT = np.diag(diag_stat) + CovT_sys
        std = np.sqrt(np.diag(CovT))
        
        return exp, std, CovT

    def test_with_Bi_exp(self):
        print("testing background function 'exp'")
        mc_mean, mc_std, mc_cov = self.monte_carlo('exp')
        lp_mean, lp_std, lp_cov = self.linear_first_order('exp')
        self.assertAlmostEqual(np.max(abs((mc_mean - lp_mean)/mc_mean)),    0, places=3)
        self.assertAlmostEqual(np.max(abs((mc_std - lp_std)/mc_std)),       0, places=1)
        self.assertAlmostEqual(np.max(np.abs((mc_cov-lp_cov)/mc_cov)),      0, places=1)

    def test_with_Bi_power(self):
        print("testing background function 'power'")
        mc_mean, mc_std, mc_cov = self.monte_carlo('power')
        lp_mean, lp_std, lp_cov = self.linear_first_order('power')
        self.assertAlmostEqual(np.max(abs((mc_mean - lp_mean)/mc_mean)),    0, places=3)
        self.assertAlmostEqual(np.max(abs((mc_std - lp_std)/mc_std)),       0, places=1)
        self.assertAlmostEqual(np.max(np.abs((mc_cov-lp_cov)/mc_cov)),      0, places=1)




# class TestCaptureYieldRPICovariance(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.ndat = 100
#         cls.samples = 50000
#         # choose a nice tof range in microseconds
#         cls.tof = np.linspace(20e-6,1000e-6,cls.ndat)




if __name__ == '__main__':
    unittest.main()
