import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest, chi2, ks_1samp, norm
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI
from ATARI.ModelData.structuring import Generative_Measurement_Model

from ATARI.syndat.control import syndatOPT, Syndat_Model
from ATARI.syndat.tests import noise_distribution_test2, noise_distribution_test, no_sampling_returns_true_test
from ATARI.syndat.general_functions import approximate_neutron_spectrum_Li6det
import unittest


__doc__ == """
This file tests various syndat functionality, including pre-loaded measurment models.
The noise distribution test can be used outside of the unit testing framework to verify that normality assumptions hold for any given syndat model.
"""


        

class TestTransmissionRPIModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pair = Particle_Pair()

        ## parameters to reduce default uncertainties to be in linear regime
        # but don't eliminate covariance because we want to test it 
        cls.model_par = {
        'trigs' :   (1e12,  0),
        'trigo' :   (1e12,  0),
        'm1'    :   (1,     0.00016) ,#0.016
        'm2'    :   (1,     0.00008) ,#0.008
        'm3'    :   (1,     0.00018) ,#0.018
        'm4'    :   (1,     0.00005) ,#0.005
        'ks'    :   (0.563, 0.000240), #0.02402339737495515
        'ko'    :   (1.471, 0.000557), #0.05576763648617445
        'b0s'   :   (9.9,   0.001) ,#0.1
        'b0o'   :   (13.4,  0.007) ,#0.7
        }


    def test_no_sampling(self):
        """
        Tests that the measurement model returns the same input as true if all sampling options are off.
        """
        generative_model = Transmission_RPI()
        reductive_model = Transmission_RPI()
        residual = no_sampling_returns_true_test(generative_model, reductive_model)
        self.assertAlmostEqual(residual, 0, places=10)


    # def test_random_energy(self):

    #     energy_grid = np.sort(np.random.default_rng().uniform(10,3000,10)) #np.linspace(min(energy_range),max(energy_range),10) # energy below 10 has very low counts due to approximate open spectrum
    #     df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.7,1.0,10)})
    #     df_true.sort_values('E', ascending=True, inplace=True)

    #     generative_model = Transmission_RPI(**self.model_par)
    #     reductive_model = Transmission_RPI(**self.model_par)

    #     synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
    #     # exp_model = Experimental_Model(energy_grid=energy_grid, energy_range = [min(energy_grid), max(energy_grid)])
    #     SynMod = Syndat_Model(generative_experimental_model=None, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
        
    #     # reset energy grid, and open neutron spectrum
    #     SynMod.generative_experimental_model.energy_grid = energy_grid
    #     SynMod.generative_experimental_model.energy_range = [10,3000]
    #     SynMod.generative_measurement_model.model_parameters.open_neutron_spectrum = None
    #     SynMod.reductive_measurement_model.model_parameters.open_neutron_spectrum = None

    #     mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynMod, df_true = df_true, ipert=500, print_out=True) #noise_distribution_test(SynMod, print_out=True)
    #     self.assertTrue( np.isclose(mean_of_residual, 0, atol=1e-1), 
    #                     "Mean of residuals is not 0")
    #     self.assertTrue( norm_test_on_residual.pvalue>1e-5, 
    #                     "Normalized residuals are not standard normal")
    #     self.assertTrue( kstest_on_chi2.pvalue>1e-5, 
    #                     "Chi2 of data does not have appropriate DOF")
        

    def test_realistic_energy(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'true': np.random.default_rng().uniform(0.7,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        generative_model = Transmission_RPI(**self.model_par)
        reductive_model = Transmission_RPI(**self.model_par)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
        
        mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynMod, df_true = df_true, ipert=250, print_out=True) #noise_distribution_test(SynMod, print_out=True)
        self.assertTrue( np.isclose(mean_of_residual, 0, atol=1e-2), 
                        "Mean of residuals is not 0")
        self.assertTrue( norm_test_on_residual.pvalue>1e-5, 
                        "Normalized residuals are not standard normal")
        self.assertTrue( kstest_on_chi2.pvalue>1e-5, 
                        "Chi2 of data does not have appropriate DOF")
        

    def test_with_given_TNCS(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.7,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        open_df = pd.DataFrame({"E":exp_model.energy_grid, 
                                'tof': exp_model.tof_grid,
                                'bw': abs(np.append(np.diff(exp_model.tof_grid), np.diff(exp_model.tof_grid)[-1])*1e-9),
                               'c':np.ones(len(exp_model.energy_grid))*1e9,
                               'dc': np.ones(len(exp_model.energy_grid))*np.sqrt(1e9)})
        # open_df = approximate_neutron_spectrum_Li6det(exp_model.energy_grid, True, exp_model.FP[0], exp_model.t0[0], self.model_par['trigo'][0])

        generative_model = Transmission_RPI(**self.model_par, open_neutron_spectrum=open_df)
        reductive_model = Transmission_RPI(**self.model_par, open_neutron_spectrum=open_df)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
        
        mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynMod, df_true = df_true, ipert=250, print_out=True) #noise_distribution_test(SynMod, print_out=True)
        self.assertTrue( np.isclose(mean_of_residual, 0, atol=1e-2), 
                        "Mean of residuals is not 0")
        self.assertTrue( norm_test_on_residual.pvalue>1e-5, 
                        "Normalized residuals are not standard normal")
        self.assertTrue( kstest_on_chi2.pvalue>1e-5, 
                        "Chi2 of data does not have appropriate DOF")

        


# class TestYieldRPIModel(unittest.TestCase):

#     @classmethod
#     def setUpClass(cls):
#         cls.pair = Particle_Pair()

#         ## parameters to reduce default uncertainties to be in linear regime
#         # but don't eliminate covariance because we want to test it 
#         cls.model_par = {
#         'trigs' :   (1e12,  0),
#         'trigo' :   (1e12,  0),
#         }

#     def test_yieldRPI(self):
#         print("Testing RPI Yield measurement model")
#         generative_model = Capture_Yield_RPI()
#         reductive_model = Capture_Yield_RPI()

#         self.no_sampling_returns_true_test(generative_model, reductive_model)

#         synOPT = syndatOPT(smoothTNCS=True) 
#         SynMod = Syndat_Model(generative_experimental_model=None, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
#         mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test(SynMod)

#         self.assertTrue( np.isclose(mean_of_residual, 0, atol=1e-1), 
#                         "Mean of residuals is not 0")
#         self.assertTrue( norm_test_on_residual.pvalue>1e-5 ,
#                         "Normalized residuals are not standard normal")
#         self.assertTrue( kstest_on_chi2.pvalue>1e-5 , 
#                         "Chi2 of data does not have appropriate DOF") 




# class TestSyndatWithSammy(unittest.TestCase):





if __name__ == '__main__':
    unittest.main()