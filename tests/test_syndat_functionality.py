import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.stats import normaltest, chi2, ks_1samp, norm
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI

from ATARI.syndat.data_classes import syndatOPT
from ATARI.syndat.control import Syndat_Control
from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.syndat.data_classes import syndatOUT, syndatOPT
from ATARI.syndat.tests import noise_distribution_test2, noise_distribution_test, no_sampling_returns_true_test
# from ATARI.syndat.general_functions import approximate_neutron_spectrum_Li6det
import unittest
import os
import ATARI.utils.hdf5 as h5io
from copy import deepcopy


__doc__ == """
This file tests various syndat functionality, including pre-loaded measurment models.
The noise distribution test can be used outside of the unit testing framework to verify that normality assumptions hold for any given syndat model.
"""

os.chdir(os.path.dirname(__file__))
        

class TestTransmissionRPIModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pair = Particle_Pair()
        
        cls.print_out = False

        ## parameters to reduce default uncertainties to be in linear regime
        # but don't eliminate covariance because we want to test it 
        cls.model_par = {
        'trigs' :   (1e10,  0),
        'trigo' :   (1e10,  0),
        }


    def test_no_sampling(self):
        """
        Tests that the measurement model returns the same input as true if all sampling options are off.
        """
        generative_model = Transmission_RPI()
        reductive_model = Transmission_RPI()
        residual = no_sampling_returns_true_test(generative_model, reductive_model)
        self.assertAlmostEqual(residual, 0, places=10)


    def test_setting_bkg_func(self):
        _ = Transmission_RPI(bkg_func='exp')
        _ = Transmission_RPI(bkg_func='power')
        self.assertRaises(ValueError, Transmission_RPI, bkg_func='pwower')


    def test_random_energy(self):

        energy_grid = np.sort(np.random.default_rng().uniform(10,3000,10)) #np.linspace(min(energy_range),max(energy_range),10) # energy below 10 has very low counts due to approximate open spectrum
        df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.01,1.0,10)})
        df_true.sort_values('E', ascending=True, inplace=True)

        exp_model = Experimental_Model(energy_grid=energy_grid, energy_range = [min(energy_grid), max(energy_grid)])
        
        generative_model = Transmission_RPI(**self.model_par)
        reductive_model = Transmission_RPI(**self.model_par)
        for measurement_model in [generative_model, reductive_model]:
            measurement_model.approximate_unknown_data(exp_model=exp_model, smooth=False, check_trig=True)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

        SynMod.sample(pw_true = df_true)
        

    def test_default_energy_window(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        generative_model = Transmission_RPI(**self.model_par)
        reductive_model = Transmission_RPI(**self.model_par)
        for measurement_model in [generative_model, reductive_model]:
            measurement_model.approximate_unknown_data(exp_model=exp_model, smooth=False, check_trig=True)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
        
        SynMod.sample(pw_true = df_true)


    def test_with_given_TNCS(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        open_df = pd.DataFrame({"E":exp_model.energy_grid, 
                                'tof': exp_model.tof_grid,
                                'bw': abs(np.append(np.diff(exp_model.tof_grid), np.diff(exp_model.tof_grid)[-1])*1e-9),
                               'ct':np.ones(len(exp_model.energy_grid))*1e10,
                               'dct': np.ones(len(exp_model.energy_grid))*np.sqrt(1e10)})

        generative_model = Transmission_RPI(**self.model_par, open_neutron_spectrum=open_df)
        reductive_model = Transmission_RPI(**self.model_par, open_neutron_spectrum=open_df)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

        SynMod.sample(pw_true = df_true)


    def test_covariance(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        generative_model = Transmission_RPI(**self.model_par)
        reductive_model = Transmission_RPI(**self.model_par)
        for measurement_model in [generative_model, reductive_model]:
            measurement_model.approximate_unknown_data(exp_model=exp_model, smooth=False, check_trig=True)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

        SynMod.sample(pw_true=df_true, num_samples=5)
        s1 = SynMod.samples[0]
        s5 = SynMod.samples[4]

        self.assertTrue(all(np.isclose(np.diag(s1.covariance_data['CovT']), s1.pw_reduced.exp_unc**2, atol=1e-5)))
        self.assertTrue(all(np.isclose(np.diag(s5.covariance_data['CovT']), s5.pw_reduced.exp_unc**2, atol=1e-5)))

        self.assertTrue(np.all(np.isclose(np.diag(s1.covariance_data['diag_stat']) + s1.covariance_data['Jac_sys'].values.T @ s1.covariance_data['Cov_sys'] @ s1.covariance_data['Jac_sys'].values, s1.covariance_data['CovT'], atol=1e-5)))
        self.assertTrue(np.all(np.isclose(np.diag(s5.covariance_data['diag_stat']) + s5.covariance_data['Jac_sys'].values.T @ s5.covariance_data['Cov_sys'] @ s5.covariance_data['Jac_sys'].values, s5.covariance_data['CovT'], atol=1e-5)))
        
        self.assertFalse(any([np.all(s1.covariance_data[key] == s5.covariance_data[key]) for key in s1.covariance_data.keys() if key not in ['Cov_sys']]))

    
    # def test_sample_saving(self, sammy_path):
    #     self.pair.add_spin_group(Jpi='3.0', J_ID=1, D=9.0030, gn2_avg=452.56615, gn2_dof=1, gg2_avg=32.0, gg2_dof=1000)

    #     exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
    #     df_true = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

    #     generative_model = Transmission_RPI(**self.model_par)
    #     reductive_model = Transmission_RPI(**self.model_par)

    #     synOPT = syndatOPT(sampleRES=True, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
    #     SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

    #     SynMod.sample(pw_true=df_true, num_samples=5)
    #     s1 = SynMod.samples[0]
    #     s5 = SynMod.samples[4]

                
    #     self.assertTrue(all(s1.pw_reduced.tof == s5.pw_reduced.tof))
    #     self.assertFalse(all(s1.pw_reduced.exp_unc == s5.pw_reduced.exp_unc))
    #     self.assertFalse(all(s1.pw_reduced.exp == s5.pw_reduced.exp))

    #     self.assertTrue([np.all(np.isclose(s1.covariance_data[key], s5.covariance_data[key], atol=1e-5)) for key in s1.covariance_data.keys()])





class TestYieldRPIModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pair = Particle_Pair()

        cls.print_out = False

        ## parameters to reduce default uncertainties to be in linear regime
        # but don't eliminate covariance because we want to test it 
        cls.model_par = {
        'trigs' :   (1e7,  0),
        'trigo' :   (1e7,  0),
        }


    def test_no_sampling(self):
        """
        Tests that the measurement model returns the same input as true if all sampling options are off.
        """
        generative_model = Capture_Yield_RPI()
        reductive_model = Capture_Yield_RPI()
        residual = no_sampling_returns_true_test(generative_model, reductive_model)
        self.assertAlmostEqual(residual, 0, places=10)


    def test_default_energy_window(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        generative_model = Capture_Yield_RPI(**self.model_par)
        reductive_model = Capture_Yield_RPI(**self.model_par)
        for measurement_model in [generative_model, reductive_model]:
            measurement_model.approximate_unknown_data(exp_model=exp_model, smooth=False, check_trig=True)

        synOPT = syndatOPT(smoothTNCS=True, sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
        
        SynMod.sample(pw_true = df_true)


    def test_with_given_TNCS(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        df = pd.DataFrame({"E":exp_model.energy_grid, 
                                'tof': exp_model.tof_grid,
                                'bw': abs(np.append(np.diff(exp_model.tof_grid), np.diff(exp_model.tof_grid)[-1])*1e-9),
                               'ct':np.ones(len(exp_model.energy_grid))*1e10,
                               'dct': np.ones(len(exp_model.energy_grid))*np.sqrt(1e10)})
        df_b = pd.DataFrame({"E":exp_model.energy_grid, 
                            'tof': exp_model.tof_grid,
                            'bw': abs(np.append(np.diff(exp_model.tof_grid), np.diff(exp_model.tof_grid)[-1])*1e-9),
                            'ct':np.ones(len(exp_model.energy_grid))*1e1,
                            'dct': np.ones(len(exp_model.energy_grid))*np.sqrt(1e1)})

        generative_model = Capture_Yield_RPI(**self.model_par, background_spectrum_bg= df_b, incident_neutron_spectrum_f= df, background_spectrum_bf= df_b)
        reductive_model = Capture_Yield_RPI(**self.model_par, background_spectrum_bg= df_b, incident_neutron_spectrum_f= df, background_spectrum_bf= df_b)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

        SynMod.sample(pw_true = df_true)


    def test_covariance(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'tof':exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        generative_model = Capture_Yield_RPI(**self.model_par)
        reductive_model = Capture_Yield_RPI(**self.model_par)
        for measurement_model in [generative_model, reductive_model]:
            measurement_model.approximate_unknown_data(exp_model=exp_model, smooth=False, check_trig=True)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

        SynMod.sample(pw_true=df_true, num_samples=5)
        s1 = SynMod.samples[0]
        s5 = SynMod.samples[4]

        self.assertTrue(all(np.isclose(np.diag(s1.covariance_data['CovY']), s1.pw_reduced.exp_unc**2, atol=1e-5)))
        self.assertTrue(all(np.isclose(np.diag(s5.covariance_data['CovY']), s5.pw_reduced.exp_unc**2, atol=1e-5)))

        ### need to update with Aarons covariance work before we can do the following assertions
        # self.assertTrue(np.all(np.isclose(np.diag(s1.covariance_data['diag_stat']) + s1.covariance_data['Jac_sys'].values.T @ s1.covariance_data['Cov_sys'] @ s1.covariance_data['Jac_sys'].values, s1.covariance_data['CovY'], atol=1e-5)))
        # self.assertTrue(np.all(np.isclose(np.diag(s5.covariance_data['diag_stat']) + s5.covariance_data['Jac_sys'].values.T @ s5.covariance_data['Cov_sys'] @ s5.covariance_data['Jac_sys'].values, s5.covariance_data['CovY'], atol=1e-5)))
        # self.assertFalse(any([np.all(s1.covariance_data[key] == s5.covariance_data[key]) for key in s1.covariance_data.keys() if key not in ['Cov_sys']]))



class TestSyndatControl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pair = Particle_Pair()
        cls.print_out = False
        cls.exp_model = Experimental_Model()
        cls.syndat_models = [Syndat_Model(generative_experimental_model=cls.exp_model), 
                             Syndat_Model(generative_experimental_model=cls.exp_model),
                             Syndat_Model(generative_experimental_model=cls.exp_model)]

    def test_sampling(self):
        
        syndat = Syndat_Control(self.pair,
                        syndat_models = self.syndat_models,
                        model_correlations={},
                        sampleRES=False)
        
        df_true_1 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })
        df_true_2 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })
        df_true_3 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })

        syndat.sample(pw_true_list=[df_true_1, df_true_2, df_true_3])


    def test_model_correlations(self):
        model_correlations = [
                                {'models':[1,1,0],  
                                 'a_b': ([582.7, 0.0515], [[1.144e+03,  1.427e-1], 
                                                            [1.427e-1,   2.191e-05]])
                                },

                                {'models':[0,1,1],  
                                 'b0o': (13.4,         0.7)
                                }

                                ]
          
        syndat = Syndat_Control(self.pair,
                        syndat_models = self.syndat_models,
                        model_correlations=model_correlations,
                        sampleRES=False)
        
        df_true_1 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })
        df_true_2 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })
        df_true_3 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })

        syndat.sample(pw_true_list=[df_true_1, df_true_2, df_true_3])
        
        ab0 = syndat.syndat_models[0].generative_measurement_model.true_model_parameters.a_b
        ab1 = syndat.syndat_models[1].generative_measurement_model.true_model_parameters.a_b
        ab2 = syndat.syndat_models[2].generative_measurement_model.true_model_parameters.a_b
        self.assertTrue(np.all(ab0[0] == ab1[0]))
        self.assertFalse(np.all(ab0[0] == ab2[0]))

        b0o0 = syndat.syndat_models[0].generative_measurement_model.true_model_parameters.b0o
        b0o1 = syndat.syndat_models[1].generative_measurement_model.true_model_parameters.b0o
        b0o2 = syndat.syndat_models[2].generative_measurement_model.true_model_parameters.b0o
        self.assertFalse(np.all(b0o0[0] == b0o1[0]))
        self.assertTrue(np.all(b0o1[0] == b0o2[0]))

        return
    

    def test_samples_to_from_hdf5(self):
                
        df_true_1 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })
        df_true_2 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })
        df_true_3 = pd.DataFrame({'E': self.exp_model.energy_grid, 'tof':self.exp_model.tof_grid,'true': np.random.default_rng().uniform(0.1,1.0,len(self.exp_model.energy_grid)) })

        save_file = os.path.realpath("./test.hdf5")
        if os.path.isfile(save_file):
            os.remove(save_file)
        
        # test raises error if all syndat models have the same name
        syndat = Syndat_Control(self.pair, syndat_models = self.syndat_models, model_correlations=[], sampleRES=False)
        self.assertRaises(ValueError, syndat.sample, num_samples=3, pw_true_list=[df_true_1, df_true_2, df_true_3],  save_samples_to_hdf5=True, hdf5_file=save_file)
        
        # test no error when writing fresh samples without explicit covariance
        updated_syndat_models =[]; title = 1
        for mod in  self.syndat_models:
            new_mod = deepcopy(mod)
            new_mod.title = str(title)
            title+=1
            updated_syndat_models.append(new_mod)

        syndat = Syndat_Control(self.pair, syndat_models = updated_syndat_models, model_correlations=[], sampleRES=False)
        syndat.sample(num_samples=3, pw_true_list=[df_true_1, df_true_2, df_true_3],  save_samples_to_hdf5=True, hdf5_file=save_file)
        for isample in range(3):
            read_par = h5io.read_par(save_file, isample, 'true')
            pw_reduced_df, cov_data = h5io.read_pw_reduced(save_file, isample, "1")
            self.assertIsInstance(read_par, pd.DataFrame)
            self.assertIsInstance(pw_reduced_df, pd.DataFrame)
            self.assertIsInstance(cov_data['Cov_sys'], np.ndarray)
            self.assertIsInstance(cov_data['Jac_sys'], pd.DataFrame)
            self.assertIsInstance(cov_data['diag_stat'], pd.DataFrame)

        # test no error when writing samples with explicit covariance
        if os.path.isfile(save_file):
            os.remove(save_file)
        for mod in updated_syndat_models:
            mod.options.explicit_covariance = True
        syndat = Syndat_Control(self.pair, syndat_models = updated_syndat_models, model_correlations=[], sampleRES=False)
        syndat.sample(num_samples=3, pw_true_list=[df_true_1, df_true_2, df_true_3],  save_samples_to_hdf5=True, hdf5_file=save_file)
        for isample in range(3):
            read_par = h5io.read_par(save_file, isample, 'true')
            pw_reduced_df, cov_data = h5io.read_pw_reduced(save_file, isample, "1")
            self.assertIsInstance(read_par, pd.DataFrame)
            self.assertIsInstance(pw_reduced_df, pd.DataFrame)
            self.assertIsInstance(cov_data['Cov_sys'], np.ndarray)
            self.assertIsInstance(cov_data['Jac_sys'], pd.DataFrame)
            self.assertIsInstance(cov_data['diag_stat'], pd.DataFrame)
            self.assertIsInstance(cov_data['CovT'], pd.DataFrame)


        # test building syndat out object from hdf5
        for isample in range(3):
            out_list = [syndatOUT.from_hdf5(save_file, isample, title) for title in ["1", "2", "3"]]
            self.assertTrue(len(out_list) == len(self.syndat_models))
            for each in out_list:
                self.assertIsInstance(each, syndatOUT)
                self.assertIsInstance(each.pw_reduced, pd.DataFrame)
                self.assertIsInstance(each.par_true, pd.DataFrame)
                self.assertIsInstance(each.title, str)
                self.assertIsInstance(each.covariance_data, dict)
                self.assertIsInstance(each.pw_raw, type(None))

        # if os.path.isfile(save_file):
        #     os.remove(save_file)


# # class TestSyndatWithSammy(unittest.TestCase):





if __name__ == '__main__':
    unittest.main()
