#%%
import pandas as pd
import numpy as np
import os

from ATARI.sammy_interface import sammy_functions, sammy_classes

from ATARI.ModelData.particle import Particle, Neutron
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model

from ATARI.utils.atario import expand_sammy_ladder_2_atari
from copy import copy

import unittest




class TestRunSammy(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        sammyexe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'

        cls.exp_model = Experimental_Model(channel_widths={"maxE": [250],"chw": [100.0],"dchw": [0.8]})
        # template_creator.make_input_template('samtemplate.inp', Ta_pair, cls.exp_model, rto)
        cls.exp_model.template = os.path.realpath('samtemplate.inp')

        cls.rto = sammy_classes.SammyRunTimeOptions(sammyexe)

        Ta_pair = Particle_Pair()
        Ta_pair.add_spin_group(Jpi='3.0', J_ID=1, D=8.79, gn2_avg=46.5, gn2_dof=1, gg2_avg=64.0, gg2_dof=1000)
        cls.pair = Ta_pair
        cls.resonance_ladder = Ta_pair.sample_resonance_ladder()
        
        sammyINP = sammy_classes.SammyInputData(
            cls.pair,
            cls.resonance_ladder,
            template = cls.exp_model.template,
            experiment=cls.exp_model,
            energy_grid=cls.exp_model.energy_grid,
        )

        cls.samout = sammy_functions.run_sammy(sammyINP, cls.rto)

        data_unc = np.sqrt(cls.samout.pw['theo_trans'])/10
        data = np.random.default_rng().normal(cls.samout.pw['theo_trans'], data_unc)

        cls.data_df = pd.DataFrame({'E':cls.samout.pw['E'],
                                'exp': data,
                                'exp_unc': data_unc})



    def test_novaried_parameters_error_catch(self):
        # print("test error catching for bayes w/o varied parameter")
        self.rto.bayes=True
        sammyINP = sammy_classes.SammyInputData(
            self.pair,
            self.resonance_ladder,
            template =self.exp_model.template,
            experiment=self.exp_model,
            energy_grid=self.exp_model.energy_grid,
        )
        self.assertRaises(ValueError, sammy_functions.run_sammy, sammyINP, self.rto)


    def test_varied_parameters(self):
        # print("test bayes solve")
        resonance_ladder_fit = copy(self.resonance_ladder)
        resonance_ladder_fit["varyE"] = np.ones(len(resonance_ladder_fit))
        resonance_ladder_fit["varyGg"] = np.ones(len(resonance_ladder_fit))
        resonance_ladder_fit["varyGn1"] = np.ones(len(resonance_ladder_fit))
        
        self.rto.bayes=True
        self.rto.get_ECSCM = False
        sammyINP = sammy_classes.SammyInputData(
            self.pair,
            resonance_ladder_fit,
            template = self.exp_model.template,
            experiment=self.exp_model,
            energy_grid=self.exp_model.energy_grid,
        )
        self.assertRaises(ValueError, sammy_functions.run_sammy, sammyINP, self.rto)

        sammyINP.experimental_data = self.data_df
        sammyOUT_fit = sammy_functions.run_sammy(sammyINP, self.rto)
        self.assertIsNotNone(sammyOUT_fit.par_post)
        self.assertIsNotNone(sammyOUT_fit.chi2_post)
        self.assertIsNotNone(sammyOUT_fit.pw_post)
        self.assertIsNone(sammyOUT_fit.est_df)
    
    def test_get_ECSCM(self):
        # print("test bayes solve")
        resonance_ladder_fit = copy(self.resonance_ladder)
        resonance_ladder_fit["varyE"] = np.ones(len(resonance_ladder_fit))
        resonance_ladder_fit["varyGg"] = np.ones(len(resonance_ladder_fit))
        resonance_ladder_fit["varyGn1"] = np.ones(len(resonance_ladder_fit))
        
        self.rto.bayes=True
        self.rto.get_ECSCM = True
        sammyINP = sammy_classes.SammyInputData(
            self.pair,
            resonance_ladder_fit,
            template = self.exp_model.template,
            experiment=self.exp_model,
            experimental_data = self.data_df
        )
        
        sammyOUT_fit = sammy_functions.run_sammy(sammyINP, self.rto)
        self.assertIsNotNone(sammyOUT_fit.est_df)

        # test with separate exp definition
        ECSCM_experiment = Experimental_Model(title = "theo",reaction='total',temp = (300,0))
        ECSCM_experiment.template = self.exp_model.template
        sammyINP.ECSCM_experiment = ECSCM_experiment
        sammyOUT_fit2 = sammy_functions.run_sammy(sammyINP, self.rto)
        self.assertIsNotNone(sammyOUT_fit2.est_df)
        self.assertFalse(np.all(sammyOUT_fit.est_df== sammyOUT_fit2.est_df))
    

    def test_expand_sammy_ladder_2_atari(self):
        # print("test ladder to atari function")
        self.rto.bayes=False
        sammyINP = sammy_classes.SammyInputData(
            self.pair,
            self.resonance_ladder,
            template=self.exp_model.template,
            experiment=self.exp_model,
            energy_grid= self.exp_model.energy_grid
        )
        sammyOUT = sammy_functions.run_sammy(sammyINP, self.rto)
        atari_par_post = expand_sammy_ladder_2_atari(self.pair, sammyOUT.par)

        self.assertTrue(np.all([each in atari_par_post.keys() for each in ["gg2", "gn2", "Jpi", "L"]]))




class TestMisc(unittest.TestCase):
    
    def test_batch_vector(self):
        steps_per_batch = 2
        save = []
        for i in range(20):
            if i%steps_per_batch == 0:
                save.append(sammy_functions.get_batch_vector(10, 1, int(i/steps_per_batch)))
        np.array(save)

        self.assertTrue(np.all(np.diag(save)==1))
        self.assertTrue(np.all(np.diag(save, k=1)==0))
        self.assertTrue(np.all(np.diag(save, k=2)==0))







class TestRunSammyYW(unittest.TestCase):


    @classmethod
    def setUpClass(cls):
        sammyexe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'

        cls.exp_model = Experimental_Model(channel_widths={"maxE": [250],"chw": [100.0],"dchw": [0.8]})
        # template_creator.make_input_template('samtemplate.inp', Ta_pair, cls.exp_model, rto)
        cls.exp_model.template = os.path.realpath('samtemplate.inp')

        cls.rto = sammy_classes.SammyRunTimeOptions(sammyexe, Print=False, keep_runDIR=True)

        Ta_pair = Particle_Pair()
        Ta_pair.add_spin_group(Jpi='3.0', J_ID=1, D=8.79, gn2_avg=46.5, gn2_dof=1, gg2_avg=64.0, gg2_dof=1000)
        cls.pair = Ta_pair
        cls.resonance_ladder = Ta_pair.sample_resonance_ladder()
        
        sammyINP = sammy_classes.SammyInputData(
            cls.pair,
            cls.resonance_ladder,
            template = cls.exp_model.template,
            experiment=cls.exp_model,
            energy_grid=cls.exp_model.energy_grid,
        )

        cls.samout = sammy_functions.run_sammy(sammyINP, cls.rto)

        data_unc = np.sqrt(cls.samout.pw['theo_trans'])/10
        data = np.random.default_rng().normal(cls.samout.pw['theo_trans'], data_unc)

        cls.data_df = pd.DataFrame({'E':cls.samout.pw['E'],
                                'exp': data,
                                'exp_unc': data_unc})



    # def test_novaried_parameters_error_catch(self):
    #     # print("test error catching for bayes w/o varied parameter")
    #     self.rto.bayes=True
    #     sammyINP = sammy_classes.SammyInputData(
    #         self.pair,
    #         self.resonance_ladder,
    #         self.exp_model.template,
    #         self.exp_model,
    #         energy_grid=self.exp_model.energy_grid,
    #     )
    #     self.assertRaises(ValueError, sammy_functions.run_sammy, sammyINP, self.rto)


    def test_varied_parameters(self):
        # print("test bayes solve")
        resonance_ladder_fit = copy(self.resonance_ladder)
        resonance_ladder_fit["varyE"] = np.ones(len(resonance_ladder_fit))
        resonance_ladder_fit["varyGg"] = np.ones(len(resonance_ladder_fit))
        resonance_ladder_fit["varyGn1"] = np.ones(len(resonance_ladder_fit))
        
        self.rto.bayes=True
        self.rto.get_ECSCM = False
        sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = self.pair,
            resonance_ladder = self.resonance_ladder,
            
            datasets= [self.data_df],
            experiments = [self.exp_model],
            experimental_covariance= [{}], 
            
            max_steps = 1,
            iterations = 2,

            LevMar = False,
            initial_parameter_uncertainty = 0.01
        )
        # self.assertRaises(ValueError, sammy_functions.run_sammy_YW, sammyINPyw, self.rto)

        sammyINPyw.resonance_ladder = resonance_ladder_fit
        sammyOUT_fit = sammy_functions.run_sammy_YW(sammyINPyw, self.rto)
        self.assertIsNotNone(sammyOUT_fit.par_post)
        self.assertIsNotNone(sammyOUT_fit.chi2_post)
        self.assertIsNotNone(sammyOUT_fit.pw_post)
        self.assertIsNone(sammyOUT_fit.est_df)
    

# Make solvebayes bare, the following works:
# Title
# Ta181     180.94803 000.0     000000.0       0     
# TWENTY
# EV
# GENERATE PLOT FILE AUTOMATICALLY
# XCT
# SOLVE BAYES EQUATIONS
# wy
# CHI SQUARED IS WANTED
# Remember original parameter values

#   0000000   00000     0000            
#   000000    000000000                       0.00000          
# reaction
#   1      1    0  3.0       1.0  3.5
#     1    1    0    0       3.0
#   2      1    0  4.0       1.0  3.5
#     1    1    0    0       4.0



    # def test_get_ECSCM(self):
    #     # print("test bayes solve")
    #     resonance_ladder_fit = copy(self.resonance_ladder)
    #     resonance_ladder_fit["varyE"] = np.ones(len(resonance_ladder_fit))
    #     resonance_ladder_fit["varyGg"] = np.ones(len(resonance_ladder_fit))
    #     resonance_ladder_fit["varyGn1"] = np.ones(len(resonance_ladder_fit))
        
    #     self.rto.bayes=True
    #     self.rto.get_ECSCM = True
    #     sammyINP = sammy_classes.SammyInputData(
    #         self.pair,
    #         resonance_ladder_fit,
    #         self.exp_model.template,
    #         self.exp_model,
    #         experimental_data = self.data_df
    #     )
        
    #     sammyOUT_fit = sammy_functions.run_sammy(sammyINP, self.rto)
    #     self.assertIsNotNone(sammyOUT_fit.est_df)
    

    # def test_expand_sammy_ladder_2_atari(self):
    #     # print("test ladder to atari function")
    #     self.rto.bayes=False
    #     sammyINP = sammy_classes.SammyInputData(
    #         self.pair,
    #         self.resonance_ladder,
    #         self.exp_model.template,
    #         self.exp_model,
    #         energy_grid= self.exp_model.energy_grid
    #     )
    #     sammyOUT = sammy_functions.run_sammy(sammyINP, self.rto)
    #     atari_par_post = expand_sammy_ladder_2_atari(self.pair, sammyOUT.par)

    #     self.assertTrue(np.all([each in atari_par_post.keys() for each in ["gg2", "gn2", "Jpi", "L"]]))






if __name__ == '__main__':
    unittest.main()
# %%
