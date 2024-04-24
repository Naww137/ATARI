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
        
        cls.print_out = False

        ## parameters to reduce default uncertainties to be in linear regime
        # but don't eliminate covariance because we want to test it 
        cls.model_par = {
        'trigs' :   (1e10,  0),
        'trigo' :   (1e10,  0),
        # 'm1'    :   (1,     0.0016) ,#0.016
        # 'm2'    :   (1,     0.0008) ,#0.008
        # 'm3'    :   (1,     0.0018) ,#0.018
        # 'm4'    :   (1,     0.0005) ,#0.005
        # 'ks'    :   (0.563, 0.00240), #0.02402339737495515
        # 'ko'    :   (1.471, 0.00557), #0.05576763648617445
        # 'b0s'   :   (9.9,   0.01) ,#0.1
        # 'b0o'   :   (13.4,  0.07) ,#0.7
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

        generative_model = Transmission_RPI(**self.model_par)
        reductive_model = Transmission_RPI(**self.model_par)

        synOPT = syndatOPT(sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True, smoothTNCS=True) 
        exp_model = Experimental_Model(energy_grid=energy_grid, energy_range = [min(energy_grid), max(energy_grid)])
        SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

        SynMod.sample(pw_true = df_true)
        # mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynMod, df_true = df_true, ipert=250, print_out=self.print_out) #noise_distribution_test(SynMod, print_out=True)
        # self.assertTrue( np.isclose(mean_of_residual, 0, atol=1e-1), 
        #                 "Mean of residuals is not 0")
        # self.assertTrue( norm_test_on_residual.pvalue>1e-5, 
        #                 "Normalized residuals are not standard normal")
        # self.assertTrue( kstest_on_chi2.pvalue>1e-5, 
        #                 "Chi2 of data does not have appropriate DOF")
        

    def test_default_energy_window(self):

        exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
        df_true = pd.DataFrame({'E': exp_model.energy_grid, 'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

        generative_model = Transmission_RPI(**self.model_par)
        reductive_model = Transmission_RPI(**self.model_par)

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
        #General Purpose Model
        cls.exp_model=Experimental_Model()
        cls.yield_true = pd.DataFrame({'tof':cls.exp_model.tof_grid,
                                       'E':cls.exp_model.energy_grid,
                                       'true':np.ones(len(cls.exp_model.energy_grid))*1000})
        
        #Monte Carlo Test Settings
        cls.run_count=500
        cls.convergence_tolerance=0.01
        cls.uncertainty_tolerance=0.1

        #Diagnostics
        cls.print_out = False
        cls.failure_logging=False

    def test_basic_functionality(self):
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        exp = Capture_Yield_RPI()
        exp.covariance_data={"cr_flux_cov":0,"br_flux_cov":0}
        exp.approximate_unknown_data(self.exp_model,False)
        generated_counts=exp.generate_raw_data(self.yield_true,exp.model_parameters,False)
        generated_counts=exp.generate_raw_data(self.yield_true,exp.model_parameters,True)
        model_parameters=exp.sample_true_model_parameters({})
        generated_yield,covariance_data=exp.reduce_raw_data(self.yield_true,generated_counts,model_parameters,True)
        generated_yield,covariance_data=exp.reduce_raw_data(self.yield_true,generated_counts,model_parameters,False)
        generated_yield,covariance_data=exp.reduce_raw_data(self.yield_true,generated_counts,exp.model_parameters,True)
        generated_yield,covariance_data=exp.reduce_raw_data(self.yield_true,generated_counts,exp.model_parameters,False)

    def test_data_type(self):
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts=exp.generate_raw_data(self.yield_true,exp.model_parameters,False)
        generated_yield,covariance_data=exp.reduce_raw_data(self.yield_true,generated_counts,exp.model_parameters,False)
        
        assert type(generated_counts)==pd.DataFrame,                                    "Returned count data frame is not actually a data frame."
        assert not(generated_counts.empty),                                             "Returned count data frame has no data."
        assert "ctg_true" in generated_counts,                                          "Returned count data frame has no true count data."
        assert "ctg" in generated_counts,                                               "Returned count data frame has no count data."
        assert "dctg" in generated_counts,                                              "Returned count data frame has count uncertainty data."
        assert len(generated_counts.ctg_true)==len(self.exp_model.energy_grid),         "True count data is on a different energy grid."
        assert len(generated_counts.ctg)==len(self.exp_model.energy_grid),              "Count data is on a different energy grid."
        assert len(generated_counts.dctg)==len(self.exp_model.energy_grid),             "Count uncertainty data is on a different energy grid."
        
        assert type(generated_yield)==pd.DataFrame,                                     "Returned yield data frame is not aactually a data frame."
        assert not(generated_yield.empty),                                              "Returned yield data frame has no data."
        assert "E" in generated_yield,                                                  "Returned count data frame has no energy grid data."
        assert "tof" in generated_yield,                                                "Returned count data frame has no time of flight data."
        assert "true" in generated_yield,                                               "Returned count data frame has true yield data."
        assert "exp" in generated_yield,                                                "Returned count data frame has no yield data."
        assert "exp_unc" in generated_yield,                                               "Returned count data frame has yield uncertainty data."
        assert np.all(np.equal(generated_yield.E.values,self.exp_model.energy_grid)),   "Returned energy grid is different than input."
        assert np.all(np.equal(generated_yield.tof.values,self.exp_model.tof_grid)),    "Returned time of flight info is different than input."
        assert np.all(np.equal(generated_yield.true.values,self.yield_true.true)),      "Returned true yield is different than input."
        assert len(generated_yield.exp)==len(self.exp_model.energy_grid),               "Yield data is on a different energy grid."
        assert len(generated_yield.exp_unc)==len(self.exp_model.energy_grid),              "Yield uncertainty data is on a different energy grid."
    
    def test_parameter_sampling(self):
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        true_model_parameters_dictionary={key:value for key, value in exp.model_parameters.__dict__.items() if not key.startswith('__') and not callable(key)}
        model_parameters=exp.sample_true_model_parameters({})
        sampled_model_parameters_dictionary={key:value for key, value in model_parameters.__dict__.items() if not key.startswith('__') and not callable(key)}
        
        same_value=0
        for key,value in true_model_parameters_dictionary.items():
            if(type(value)==pd.DataFrame):
                #I am not dealing with this right now, good luck
                pass
            else:
                if(value==sampled_model_parameters_dictionary[key] and not(value[1]==0)):
                    same_value+=1
        
        assert same_value==0, str(same_value)+" paramters are not sampled."
    
    def test_no_sampling(self):
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts=exp.generate_raw_data(self.yield_true,exp.model_parameters,False)
        generated_yield,covariance_data=exp.reduce_raw_data(self.yield_true,generated_counts,exp.model_parameters,False)
        
        assert np.allclose(generated_yield.exp.values,self.yield_true.true.values), "Returned yield is different than input."
    
    def test_sampling(self):
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts_no_sample=exp.generate_raw_data(self.yield_true,exp.model_parameters,False)
        generated_yield_no_sample,___=exp.reduce_raw_data(self.yield_true,generated_counts_no_sample,exp.model_parameters,False)
        
        #Case 1: count data is sampled
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts_samp=exp.generate_raw_data(self.yield_true,exp.model_parameters,True)
        generated_yield_samp,___=exp.reduce_raw_data(self.yield_true,generated_counts_samp,exp.model_parameters,False)
        assert not(np.allclose(generated_counts_samp.ctg.values,generated_counts_no_sample.ctg.values)), "Count data is the same with or without count sampling."
        assert not(np.allclose(generated_yield_samp.exp.values,generated_yield_no_sample.exp.values)), "Yield data is the same with or without count sampling"
        
        #Case 2: model parameters are sampled
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts_samp=exp.generate_raw_data(self.yield_true,exp.model_parameters,False)
        model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0], 
                                                        "trig_bg":exp.model_parameters.trig_bg[0],
                                                        "trig_f":exp.model_parameters.trig_f[0], 
                                                        "trig_bf":exp.model_parameters.trig_bf[0],
                                                        "fn":exp.model_parameters.fn[0],
                                                        "yield_flux":exp.model_parameters.yield_flux.ct})
        generated_yield_samp,___=exp.reduce_raw_data(self.yield_true,generated_counts_samp,model_parameters,False)
        assert np.allclose(generated_counts_samp.ctg.values,generated_counts_no_sample.ctg.values), "Count data is different with only parameter sampling."
        assert not(np.allclose(generated_yield_samp.exp.values,generated_yield_no_sample.exp.values)), "Yield data is the same with or without parameter sampling"
        
        #Case 3: both are sampled
        exp = Capture_Yield_RPI()
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts_samp=exp.generate_raw_data(self.yield_true,exp.model_parameters,True)
        model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0], 
                                                        "trig_bg":exp.model_parameters.trig_bg[0],
                                                        "trig_f":exp.model_parameters.trig_f[0], 
                                                        "trig_bf":exp.model_parameters.trig_bf[0],
                                                        "fn":exp.model_parameters.fn[0],
                                                        "yield_flux":exp.model_parameters.yield_flux.ct})
        generated_yield_samp,___=exp.reduce_raw_data(self.yield_true,generated_counts_samp,model_parameters,False)
        assert not(np.allclose(generated_counts_samp.ctg.values,generated_counts_no_sample.ctg.values)), "Count data is the same with or without count sampling."
        assert not(np.allclose(generated_yield_samp.exp.values,generated_yield_no_sample.exp.values)), "Yield data is the same with or without count sampling"
    
    def test_monte_carlo(self):
        exp = Capture_Yield_RPI()
        yields=np.empty((len(self.exp_model.energy_grid),self.run_count,2))
        counts=np.empty((len(self.exp_model.energy_grid),self.run_count,3))
        for run in range(self.run_count):
            exp.approximate_unknown_data(self.exp_model,True)
            generated_counts=exp.generate_raw_data(self.yield_true,exp.model_parameters,True)
            model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0],
                                                            "trig_bg":exp.model_parameters.trig_bg[0],
                                                            "trig_f":exp.model_parameters.trig_f[0],
                                                            "trig_bf":exp.model_parameters.trig_bf[0],
                                                            "fn":exp.model_parameters.fn[0],
                                                            "yield_flux":exp.model_parameters.yield_flux.ct})
            yield_run,___=exp.reduce_raw_data(self.yield_true,generated_counts,model_parameters,False)
            yields[:,run,0]=yield_run.exp.values
            yields[:,run,1]=yield_run.exp_unc.values
            counts[:,run,0]=generated_counts.ctg.values
            counts[:,run,1]=generated_counts.dctg.values
        
        assert np.allclose(np.mean(yields[:,:,0],1),self.yield_true.true.values,rtol=self.convergence_tolerance), "Mean of data does not converge to true."
        assert np.allclose(np.std(yields[:,:,0],1),np.mean(yields[:,:,1],1),rtol=self.uncertainty_tolerance), "Reported uncertainty is different than distribution width. (try rerunning, sometimes this just fails)"
    
    def test_covariance(self):
        exp = Capture_Yield_RPI()
        exp.covariance_data={"cr_flux_cov":0,"br_flux_cov":0}
        exp.approximate_unknown_data(self.exp_model,True)
        generated_counts=exp.generate_raw_data(self.yield_true,exp.model_parameters,True)
        model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0], 
                                                            "trig_bg":exp.model_parameters.trig_bg[0],
                                                            "trig_f":exp.model_parameters.trig_f[0], 
                                                            "trig_bf":exp.model_parameters.trig_bf[0],
                                                            "fn":exp.model_parameters.fn[0],
                                                            "yield_flux":exp.model_parameters.yield_flux.ct})
        yield_run_T,covariance_data_T=exp.reduce_raw_data(self.yield_true,generated_counts,model_parameters,True,calculate_covariance_matrix=False)
        yield_run_F,covariance_data_F=exp.reduce_raw_data(self.yield_true,generated_counts,model_parameters,False)
        full_matrix=covariance_data_T["Cov_Y_jac"].T@covariance_data_T["Cov_Y_cov"]@covariance_data_T["Cov_Y_jac"]
        
        assert np.allclose(yield_run_T.exp_unc.values,yield_run_F.exp_unc.values), "Reported uncertainty is differnt between using and not using covariance calcs."
        assert np.allclose(np.sqrt(np.diag(full_matrix)),yield_run_T.exp_unc.values), "Digonal of covariance matrix does not match reported uncertainty."
        assert not(("Cov_Y" in covariance_data_F)or("Cov_Y_jac" in covariance_data_F)or("Cov_Y_cov" in covariance_data_F)), "Covariance data is included in output when not using calcs."

    # def test_default_energy_window(self):

    #     exp_model = Experimental_Model(channel_widths={"maxE": [3000],"chw": [1200.0],"dchw": [0.8]})
    #     df_true = pd.DataFrame({'E': exp_model.energy_grid, 'true': np.random.default_rng().uniform(0.1,1.0,len(exp_model.energy_grid)) })#np.ones(len(exp_model.energy_grid))*0.9 })

    #     generative_model = Capture_Yield_RPI(**self.model_par)
    #     reductive_model = Capture_Yield_RPI(**self.model_par)

    #     synOPT = syndatOPT(smoothTNCS=True, sampleRES=False, calculate_covariance=True, explicit_covariance=True, sampleTMP=True) 
    #     SynMod = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)
    #     mean_of_residual, norm_test_on_residual, kstest_on_chi2 = noise_distribution_test2(SynMod, df_true = df_true, ipert=250, print_out=self.print_out) 

    #     self.assertTrue( np.isclose(mean_of_residual, 0, atol=1e-1), 
    #                     "Mean of residuals is not 0")
    #     self.assertTrue( norm_test_on_residual.pvalue>1e-5 ,
    #                     "Normalized residuals are not standard normal")
    #     self.assertTrue( kstest_on_chi2.pvalue>1e-5 , 
    #                     "Chi2 of data does not have appropriate DOF") 




# # class TestSyndatWithSammy(unittest.TestCase):





if __name__ == '__main__':
    unittest.main()