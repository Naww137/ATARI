from typing import Protocol
from ATARI.AutoFit.functions import * #get_parameter_grid, get_resonance_ladder, eliminate_small_Gn, update_vary_resonance_ladder
from ATARI.sammy_interface import sammy_classes, sammy_functions
import numpy as np
import pandas as pd
from copy import copy
from itertools import permutations, product



def get_all_resonance_ladder_combinations(possible_J_ID, df):
    all_J_ID = list(product(possible_J_ID, repeat=len(df)))
    dfs = []
    for possibility in all_J_ID:
        temp_df = copy(df)
        temp_df["J_ID"] = np.array(possibility)
        dfs.append(temp_df)
    return dfs


def filter_resonance_ladders_by_LL_of_parameters(all_possible_reslad, 
                                                 spin_groups,
                                                 max_ladders,
                                                 include = [1,1,1]):
    LLs = []
    for ladder in all_possible_reslad:
        LL_par = get_LL_by_parameter(ladder, spin_groups)
        LL = np.sum(LL_par*include)
        LLs.append(LL)
    LLs = np.array(LLs)
    sort_index = np.argsort(LLs)
    max_num = min( max_ladders, len(sort_index))
    indices_to_evaluate = sort_index[-max_num:]
    prioritized_ladders = [all_possible_reslad[i] for i in indices_to_evaluate]

    return prioritized_ladders




class SpinSelectOPT:
    def __init__(self, **kwargs):
        self._max_steps = 5
        self._iterations = 2
        self._step_threshold = 0.1
        self._LevMar = True
        self._LevMarV = 2
        self._LevMarVd = 5
        self._LevMarV0 = 0.1

        self._filter_ladders_by_LL = False
        self._max_ladders = 50
        self._LL_parameters = ["E", "Gg", "Gn"]

        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    

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
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
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
    def filter_ladders_by_LL(self):
        """Filter possible spin group combinations by Log-Likelihood of parameters, by default False"""
        return self._filter_ladders_by_LL
    @filter_ladders_by_LL.setter
    def filter_ladders_by_LL(self, filter_ladders_by_LL):
        self._filter_ladders_by_LL = filter_ladders_by_LL

    @property
    def max_ladders(self):
        """if filtering possible spin group combinations, this number of top performing assignments will be used, by default 50"""
        return self._max_ladders
    @max_ladders.setter
    def max_ladders(self, max_ladders):
        self._max_ladders = max_ladders

    @property
    def LL_parameters(self, t = False):
        """if filtering possible spin group combinations based on LL, which parameters will be used, by default ["E", "Gg", "Gn"]"""
        return self._LL_parameters
    @LL_parameters.setter
    def LL_parameters(self, LL_parameters):
        self._LL_parameters = LL_parameters








class SpinSelectOUT:

    def __init__(self):
        
        self.history = {}

    def add_model(self, N, spin_models, leading_model):
        self.history[N] = {"all_spin_models": spin_models,
                           "leading_model"  : leading_model,
                           "all_chi2n"      : [np.sum(each.chi2_post)/np.sum([len(i) for i in each.pw]) for each in spin_models]}



class SpinSelect:

    def __init__(self,
                 options: SpinSelectOPT):
        
        self.options = options
        # if options.Fit:
            # self.fit()
        
    def fit_multiple_models(self,
                            models,
                            possible_J_ID,
                            particle_pair,
                            datasets,
                            experiments,
                            covariance_data,
                            sammyRTO,
                            external_resonance_indices):
        
        # model_Ns = [len(model.par_post) for model in models]
        out = SpinSelectOUT()

        for model in models:
            N = len(model.par_post)
            model.par_post['varyGg'] = np.ones(len(model.par_post))
            all_outs, leading_model = self.try_all_spin_groups(possible_J_ID, #[1.0,2.0],
                                                                model.par_post, #sammyOUT_elim.par_post,
                                                                particle_pair, #Ta_pair,
                                                                datasets,
                                                                experiments,
                                                                covariance_data,
                                                                sammyRTO, #sammy_rto_fit,
                                                                external_resonance_indices = external_resonance_indices)
            out.add_model(N, all_outs, leading_model)

        return out
            

    def try_all_spin_groups(self, 
                      possible_J_ID,
                      starting_ladder,
                      particle_pair,
                      datasets,
                      experiments,
                      covariance_data,
                      sammyRTO,
                      external_resonance_indices=[]):


        sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = particle_pair,
            resonance_ladder = starting_ladder,  

            datasets= datasets,
            experiments = experiments,
            experimental_covariance=covariance_data,  #[{}, {}, {}, {}, {}], # 
            
            max_steps = self.options.max_steps,
            iterations = self.options.iterations,
            step_threshold = self.options.step_threshold,
            autoelim_threshold = None,

            LS = False,
            LevMar = self.options.LevMar,
            LevMarV = self.options.LevMarV,
            LevMarVd= self.options.LevMarVd,
            initial_parameter_uncertainty = self.options.LevMarV0
            )

        # in_window_df = copy(starting_ladder)
        # fixed_resonances_df = in_window_df.iloc[fixed_resonances_indices, :]
        # in_window_df.drop(index = fixed_resonances_indices, inplace=True)
        internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(starting_ladder, external_resonance_indices)
        possible_dfs = get_all_resonance_ladder_combinations(possible_J_ID, internal_resonance_ladder)

        if self.options.filter_ladders_by_LL:
            include = [1 if par in self.options.LL_parameters else 0 for par in ["E","Gg","Gn"]]
            possible_dfs = filter_resonance_ladders_by_LL_of_parameters(possible_dfs, particle_pair.spin_groups, self.options.max_ladders, include = include)

        leading_model = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)
        leading_chi2 = leading_model.chi2_post

        all_outs = []
        for each_df in possible_dfs:
            # sammyINPyw.resonance_ladder = pd.concat([each_df, fixed_resonances_df])
            resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(each_df, external_resonance_ladder)
            sammyINPyw.resonance_ladder = resonance_ladder
            sammyOUT_temp = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)
            all_outs.append(sammyOUT_temp)
            if np.sum(sammyOUT_temp.chi2_post) < np.sum(leading_chi2):
                leading_chi2 = sammyOUT_temp.chi2_post
                leading_model = sammyOUT_temp
            else:
                pass
        
        return all_outs, leading_model