import numpy as np
import pandas as pd
from scipy.stats import normaltest
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI

from ATARI.syndat.control import syndatOPT, Syndat_Model

import unittest


__doc__ == """
This file tests various syndat functionality, including pre-loaded measurment models.
"""

class TestMeasurementModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pair = Particle_Pair()
        cls.energy_grid = np.linspace(10,3000,10) # energy below 10 has very low counts due to approximate open spectrum
        cls.exp_model = Experimental_Model(energy_grid = cls.energy_grid,
                                           energy_range =[0, 3000])

        cls.ipert = 5000
        cls.df_true = pd.DataFrame({'E':cls.energy_grid, 'true':np.random.default_rng().uniform(0.01,1,10)})
        cls.true = np.array(cls.df_true.sort_values('E', ascending=True)["true"])


    def noise_distribution_test(self, generative_model, reductive_model):
        """
        Tests the following for data sampling distributions from given measurement model:
            1. the mean of all data samples converges to the true value
            2. the normalized residuals (data-true)/(data_uncertainty) fall on a standard normal distribution
        """

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=False)
        synT = Syndat_Model(self.exp_model, generative_model, reductive_model, synOPT)

        exp = np.zeros([self.ipert,len(self.energy_grid)])
        exp_unc = np.zeros([self.ipert,len(self.energy_grid)])
        synT.sample(pw_true=self.df_true, num_samples=self.ipert)

        for i in range(self.ipert):
            data = synT.samples[i].pw_reduced
            exp[i,:] = np.array(data.exp)
            exp_unc[i,:] = np.array(data.exp_unc)

        self.assertTrue( np.all(np.isclose(np.mean(exp, axis=0), self.true, rtol=1e-2)))
        self.assertTrue( np.all(normaltest((exp-self.true)/exp_unc).pvalue>1e-5))



    def no_sampling_returns_true_test(self, generative_model, reductive_model):
        """
        Tests that when nothing is sampled, the measurement model returns the true value.
        """

        synOPT = syndatOPT(sampleRES=False, 
                        sample_counting_noise= False, 
                        calculate_covariance=False, 
                        sampleTMP=False, 
                        smoothTNCS =True)
        
        synT = Syndat_Model(self.exp_model, generative_model, reductive_model, synOPT)

        exp = np.zeros([10,len(self.energy_grid)])
        exp_unc = np.zeros([10,len(self.energy_grid)])
        synT.sample(pw_true=self.df_true, num_samples=10)

        for i in range(10):
            data = synT.samples[i].pw_reduced
            exp[i,:] = np.array(data.exp)
            exp_unc[i,:] = np.array(data.exp_unc)

        self.assertAlmostEqual(np.sum(abs(exp-self.true)), 0, places=10)




    def test_transmissionRPI(self):
        # print("testing RPI transmission measurement model")
        generative = Transmission_RPI()
        reductive = Transmission_RPI()
        self.no_sampling_returns_true_test(generative, reductive)

        # increase counting statistics and remove covariance for distribution tests
        for mod in [generative, reductive]:
            mod.trigs=(1e10,0)
            mod.trigo=(1e10,0)
            mod.a_b = ([582.7768594580712, 0.05149689096209191],
                        [[1.14395753e+03,  0],
                        [0,   2.19135003e-05]]
                        ) 
        self.noise_distribution_test(generative, reductive)


    def test_yieldRPI(self):
        # print("testing RPI yield measurement model")
        generative = Capture_Yield_RPI()
        reductive = Capture_Yield_RPI()
        self.no_sampling_returns_true_test(generative, reductive)

        # increase counting statistics and remove covariance for distribution tests
        for mod in [generative, reductive]:
            mod.trigs=(1e10,0)
            mod.trigo=(1e10,0)
        self.noise_distribution_test(generative, reductive)
        



# class TestSyndatWithSammy(unittest.TestCase):





if __name__ == '__main__':
    unittest.main()