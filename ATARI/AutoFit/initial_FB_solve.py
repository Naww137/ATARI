from typing import Protocol
from ATARI.AutoFit.functions import get_parameter_grid, get_resonance_ladder, eliminate_small_Gn, update_vary_resonance_ladder
from ATARI.sammy_interface import sammy_classes, sammy_functions
import numpy as np
import pandas as pd
from copy import copy




class Evaluation_Data:
    def __init__(self):
        self.pw_data = []
        self.covariance_data = []
        self.experimental_models = []

    def add_dataset(self,
                    pointwise_data,
                    covariance_data,
                    experimental_model
                    ):
        self.pw_data.append(pointwise_data)
        self.covariance_data.append(covariance_data)
        self.experimental_models.append(experimental_model)



class InitialFBOPT:
    def __init__(self, **kwargs):
        self._Fit = True

        self._width_elimination = True
        self._Gn_threshold = 1e-2
        self._decrease_chi2_threshold_for_width_elimination = True

        self._max_steps = 30
        self._iterations = 2
        self._step_threshold = 0.1
        self._LevMar = True
        self._LevMarV = 1.5
        self._LevMarVd = 5
        self._LevMarV0 = 0.5

        self._fit_Gg = True
        self._fit_all_spin_groups = True
        self._spin_group_keys = []



        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    

    @property
    def Fit(self):
        return self._Fit
    @Fit.setter
    def Fit(self, Fit):
        self._Fit = Fit

    @property
    def width_elimination(self):
        return self._width_elimination
    @width_elimination.setter
    def width_elimination(self, width_elimination):
        self._width_elimination = width_elimination

    @property
    def Gn_threshold(self):
        return self._Gn_threshold
    @Gn_threshold.setter
    def Gn_threshold(self, Gn_threshold):
        self._Gn_threshold = Gn_threshold

    @property
    def decrease_chi2_threshold_for_width_elimination(self):
        return self._decrease_chi2_threshold_for_width_elimination
    @decrease_chi2_threshold_for_width_elimination.setter
    def decrease_chi2_threshold_for_width_elimination(self, decrease_chi2_threshold_for_width_elimination):
        self._decrease_chi2_threshold_for_width_elimination = decrease_chi2_threshold_for_width_elimination

    @property
    def max_steps(self):
        return self._max_steps
    @max_steps.setter
    def max_steps(self, max_steps):
        self._max_steps = max_steps

    @property
    def iterations(self):
        return self._iterations
    @iterations.setter
    def iterations(self, iterations):
        # if iterations < 1:
        #     raise ValueError("iterations must be at least 1")
        self._iterations = iterations

    @property
    def step_threshold(self):
        return self._step_threshold
    @step_threshold.setter
    def step_threshold(self, step_threshold):
        self._step_threshold = step_threshold
    
    @property
    def LevMar(self):
        return self._LevMar
    @LevMar.setter
    def LevMar(self, LevMar):
        self._LevMar = LevMar
    
    @property
    def LevMarV(self):
        return self._LevMarV
    @LevMarV.setter
    def LevMarV(self, LevMarV):
        self._LevMarV = LevMarV

    @property
    def LevMarVd(self):
        return self._LevMarVd
    @LevMarVd.setter
    def LevMarVd(self, LevMarVd):
        self._LevMarVd = LevMarVd

    @property
    def LevMarV0(self):
        return self._LevMarV0
    @LevMarV0.setter
    def LevMarV0(self, LevMarV0):
        self._LevMarV0 = LevMarV0


    @property
    def fit_Gg(self):
        return self._fit_Gg
    @fit_Gg.setter
    def fit_Gg(self, fit_Gg):
        self._fit_Gg = fit_Gg

    @property
    def fit_all_spin_groups(self):
        return self._fit_all_spin_groups
    @fit_all_spin_groups.setter
    def fit_all_spin_groups(self, fit_all_spin_groups):
        self._fit_all_spin_groups = fit_all_spin_groups

    @property
    def spin_group_keys(self):
        return self._spin_group_keys
    @spin_group_keys.setter
    def spin_group_keys(self, spin_group_keys):
        self._spin_group_keys = spin_group_keys












class InitialFB:

    def __init__(self,
                 options: InitialFBOPT):
        
        self.options = options
        # if options.Fit:
            # self.fit()



    def fit(self,
            particle_pair,
            energy_range,
            datasets,
            experiments,
            covariance_data,
            sammyRTO
            ):
        
        rto = copy(sammyRTO)
        assert rto.bayes == True

        
        if self.options.fit_all_spin_groups:
            spin_groups = [each[1] for each in particle_pair.spin_groups.items()] 
        else:
            assert len(self.options.spin_group_keys)>0
            spin_groups = [each[1] for each in particle_pair.spin_groups.items() if each[0] in self.options.spin_group_keys]
        initial_resonance_ladder = self.get_starting_feature_bank(energy_range, spin_groups)

        sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = particle_pair,
            resonance_ladder = initial_resonance_ladder,  

            datasets= datasets,
            experiments = experiments,
            experimental_covariance= covariance_data, 
            
            max_steps = self.options.max_steps,
            iterations = self.options.iterations,
            step_threshold = self.options.step_threshold,
            LevMar = self.options.LevMar,
            LevMarV = self.options.LevMarV,
            LevMarVd = self.options.LevMarVd,
            initial_parameter_uncertainty = self.options.LevMarV0,
            
            autoelim_threshold = None,
            LS = False,
            )

        ### Fit 1 on Gn only
        print("========================================\n\tFIT 1\n========================================")
        outs_fit_1 = self.fit_and_eliminate(rto,
                                             sammyINPyw)
        # if save:
        #     self.outs_fit_Gn = outs_fit_Gn
        reslad_1 = outs_fit_1[-1].par_post

        ### Fit 2 on E and optionally Gg
        print("========================================\n\tFIT 2\n========================================")
        if self.options.fit_Gg:
            reslad_1 = update_vary_resonance_ladder(reslad_1, varyE=1, varyGg=1, varyGn1=1)
        else:
            reslad_1 = update_vary_resonance_ladder(reslad_1, varyE=1, varyGg=0, varyGn1=1)

        sammyINPyw.resonance_ladder = reslad_1
        outs_fit_2 = self.fit_and_eliminate(rto,
                                             sammyINPyw)
        
        return outs_fit_2




    def get_starting_feature_bank(self,
                                  energy_range,
                                  spin_groups,
                                  num_Elam= None,
                                  varyE = 0,
                                  varyGg = 0,
                                  varyGn1 = 1
                                  ):
        if num_Elam is None:
            num_Elam = int((np.max(energy_range)-np.min(energy_range)) * 1.25)

        Er, Gg, Gn, J_ID = [], [], [], []
        for each in spin_groups:
            Er_1, Gg_1, Gn_1, J_ID_1 = get_parameter_grid(energy_range, each, num_Elam, option=1)
            Er.append(Er_1); Gg.append(Gg_1); Gn.append(Gn_1); J_ID.append(J_ID_1); 
        Er = np.concatenate(Er)
        Gg = np.concatenate(Gg)
        Gn = np.concatenate(Gn)
        J_ID = np.concatenate(J_ID) 

        return get_resonance_ladder(Er, Gg, Gn, J_ID, varyE=varyE, varyGg=varyGg, varyGn1=varyGn1)




    def fit_and_eliminate(self, 
               rto,
               sammyINPyw,
               ):
        
        print(f"Initial solve from {len(sammyINPyw.resonance_ladder)} resonance features\n")
        sammyOUT_fit = sammy_functions.run_sammy_YW(sammyINPyw, rto)
        outs = [sammyOUT_fit]

        if self.options.width_elimination:
            eliminating = True
            while eliminating:
                return_resonance_ladder, fraction_eliminated = eliminate_small_Gn(sammyOUT_fit.par_post, self.options.Gn_threshold)
                if fraction_eliminated == 0.0:
                    eliminating = False
                else:
                    if self.options.decrease_chi2_threshold_for_width_elimination:
                        sammyINPyw.step_threshold *= 0.1
                    print(f"\n----------------------------------------\nEliminated {round(fraction_eliminated*100, 2)}% of resonance features based on neuton width")
                    print(f"Resolving with {len(return_resonance_ladder)} resonance features\n----------------------------------------\n")
                    sammyINPyw.resonance_ladder = return_resonance_ladder
                    sammyOUT_fit = sammy_functions.run_sammy_YW(sammyINPyw, rto)
                    outs.append(sammyOUT_fit)
            print(f"\nComplete after no neutron width features below threshold\n")
            
        # self.fit_Gn_steps = outs

        return outs
    

    def fit_E_Gn(self,
                 rto,
                 sammyINPyw):
        

        return



    # def report(self, string):
    #     if self.report_to_file:
    #         self.