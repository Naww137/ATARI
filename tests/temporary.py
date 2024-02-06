# #%%
# import numpy as np
# import pandas as pd
# from scipy.stats import normaltest
# from ATARI.ModelData.experimental_model import Experimental_Model
# from ATARI.ModelData.particle_pair import Particle_Pair
# from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
# from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI

# from ATARI.syndat.control import syndatOPT, Syndat_Model
# from matplotlib.pyplot import *
# #%%
# pair = Particle_Pair()
# energy_grid = np.linspace(10,3000,10) 
# exp_model = Experimental_Model(energy_grid = energy_grid,
#                                     energy_range =[0, 3000])#,
#                                 #    channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})

# ipert = 5000
# df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.01,1,10)})
# true = np.array(df_true.sort_values('E', ascending=True)["true"])


# # def test_sampling_distribution_transmissionRPI(self):
# # """
# # Tests the data sampling distribution for RPI transmission measurement model. 
# # Over 5000 samples, test:
# #     1. the mean of all data samples converges to the true value
# #     2. the normalized residuals (data-true)/(data_uncertainty) fall on a standard normal distribution
# # """
# open_neutron_spectrum = pd.DataFrame({"E":energy_grid, 
#                                       "c":np.ones(len(energy_grid))*1000, 
#                                       "dc":np.ones(len(energy_grid))*np.sqrt(1000)})
# # open_neutron_spectrum = None

# generative_model = Transmission_RPI()
#     # trigs=(1e10,0),
#     #                             trigo=(1e10,0),

#     #                             a_b    = ([582.7768594580712, 0.05149689096209191],
#     #                                         [[1.14395753e+03,  0],
#     #                                         [0,   2.19135003e-05]]) )
# reductive_model = Transmission_RPI()
# # trigs=(1e10,0),
# #                                 trigo=(1e10,0),

# #                                 a_b    = ([582.7768594580712, 0.05149689096209191],
# #                                             [[1.14395753e+03,  0],
# #                                             [0,   2.19135003e-05]] ))

# # synOPT = syndatOPT(sampleRES=False, calculate_covariance=False)
# # synT = Syndat_Model(exp_model, generative, reductive, synOPT)

# # exp_trans = np.zeros([ipert,10])
# # exp_trans_unc = np.zeros([ipert,10])
# # synT.sample(pw_true=df_true, num_samples=ipert)

# # for i in range(ipert):
# #     data = synT.samples[i].pw_reduced
# #     exp_trans[i,:] = np.array(data.exp)
# #     exp_trans_unc[i,:] = np.array(data.exp_unc)

# # print( np.all(np.isclose(np.mean(exp_trans, axis=0), true, rtol=1e-2)))
# # print( np.all(normaltest((exp_trans-true)/exp_trans_unc).pvalue>0.001))

# # figure()
# # hist((exp_trans-true)/exp_trans_unc)
# # show()


# synOPT = syndatOPT(sampleRES=False, 
#                    sample_counting_noise= False, 
#                    calculate_covariance=False, 
#                    sampleTMP=False, 
#                    smoothTNCS =True)
# synT = Syndat_Model(exp_model, generative_model, reductive_model, synOPT)

# exp_trans = np.zeros([10,len(energy_grid)])
# exp_trans_unc = np.zeros([10,len(energy_grid)])
# synT.sample(pw_true=df_true, num_samples=10)

# for i in range(10):
#     data = synT.samples[i].pw_reduced
#     exp_trans[i,:] = np.array(data.exp)
#     exp_trans_unc[i,:] = np.array(data.exp_unc)

# np.isclose(np.sum(abs(exp_trans-true)), 0, atol=1e-10)



import unittest

class TestCalculator(unittest.TestCase):
    def helper_method(self, a, b):
        # Helper method to perform some calculation
        return a * b

    def test_add(self):
        # Test case for the add method
        result = self.helper_method(2, 3)
        self.assertEqual(result, 6, "Multiplication result is incorrect")

    def test_subtract(self):
        # Test case for the subtract method
        result = self.helper_method(5, 2)
        self.assertEqual(result, 10, "Multiplication result is incorrect")

if __name__ == '__main__':
    unittest.main()
