### 
### Elimination of resonances N -> N-1 -> N-2 ...
### using chi2 threshold given
### utilizing YW scheme for intermediate fitting
###


# imports
import pandas as pd
import numpy as np

import os
import uuid
from datetime import datetime
import time

from ATARI.sammy_interface.sammy_classes import SammyInputDataYW, SammyRunTimeOptions, SammyOutputData
from ATARI.sammy_interface import sammy_functions

from ATARI.ModelData.particle_pair import Particle_Pair

from dataclasses import dataclass
from typing import Optional, Union

from ATARI.AutoFit import elim_addit_funcs

# end imports


### classes definitions
class elim_OPTs:
    # TODO: change set all params smoothly!
    """
    Options and settings for a single elimination routine.
    
    Parameters & Attributes
    ----------
    chi2_allowed : float
        value of chi2 allowed difference when comparing 2 models

    stop_at_chi2_thr: Bool
        Boolean value which tells to stop if during search of a models there was 
        no models that passed the chi2 test, if false - continue to delete resonances until we
        will not have at least one resonance.

    interm_fit_max_iter: int
        allowed number of iterations for intermediate stage to perform YW fitting

    deep_fit_max_iter: int
        allowed number of iterations for deep-fitting stage 

    interm_fit_step_thr: float,
        chi2 step threshold used for YW fitting procedure in intermediate fitting stage

    deep_fit_step_thr: float,
        chi2 step threshold used for YW fitting procedure

    start_fudge_for_deep_stage: float
        Starting value of a fudge factor used for YW scheme

    LevMarV0_priorpassed: float
        Starting value of a fidge factor used for YW scheme if model passed the test without refitting
        priors tests

    greedy_mode: bool
        If set to True - eliminator will not test all models of current branch
        and select best of them - it will use first model of current branch that passed the test 

    start_deep_fit_from: int
        this determines the number of resonances in a current level on a path from N->N-1 where deep fit start from (see default value)
   
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    """
    def __init__(self, **kwargs):
        
        # default values for all 

        self._chi2_allowed = kwargs.get('chi2_allowed', 0)
                
        self._deep_fit_max_iter = kwargs.get('deep_fit_max_iter', 20)
        self._deep_fit_step_thr = kwargs.get('deep_fit_step_thr', 0.001)

        self._interm_fit_max_iter = kwargs.get('interm_fit_max_iter', 10)
        self._interm_fit_step_thr = kwargs.get('interm_fit_step_thr', 0.01)
         
        self._start_fudge_for_deep_stage = kwargs.get('start_fudge_for_deep_stage', 0.1)

        self._LevMarV0_priorpassed = kwargs.get('LevMarV0_priorpassed', 0.01)

        self._stop_at_chi2_thr = kwargs.get('stop_at_chi2_thr', True) # by default stops when didn't find the model that passed the test
        
        self._greedy_mode = kwargs.get('greedy_mode', False) # by default False  - search all solutions

        self._start_deep_fit_from = kwargs.get('start_deep_fit_from', 10)

        # all that passed through
        for key, value in kwargs.items():
            setattr(self, key, value)
    

    # allowed difference in chi2 to pass the test
    @property
    def chi2_allowed(self):
        return self._chi2_allowed
    @chi2_allowed.setter
    def chi2_allowed(self, chi2_allowed):
        self._chi2_allowed = chi2_allowed
    
    # stop or not if we didn't found the model that passess the test (just dive deep to 1 res. model)
    @property
    def stop_at_chi2_thr(self):
        return self._stop_at_chi2_thr
    @stop_at_chi2_thr.setter
    def stop_at_chi2_thr(self, stop_at_chi2_thr):
        self._stop_at_chi2_thr = stop_at_chi2_thr

     # stop or not if we found a solution inside a subset that passed the test (greedy in time)
    @property
    def greedy_mode(self):
        return self._greedy_mode
    @greedy_mode.setter
    def greedy_mode(self, greedy_mode):
        self._greedy_mode = greedy_mode

    # start_deep_fit_from
    @property 
    def start_deep_fit_from(self):
        return self._start_deep_fit_from
    @start_deep_fit_from.setter
    def start_deep_fit_from(self, start_deep_fit_from):
        self._start_deep_fit_from = start_deep_fit_from
    
    #deep fitting stage, start fudge value
    @property 
    def start_fudge_for_deep_stage(self):
        return self._start_fudge_for_deep_stage
    @start_fudge_for_deep_stage.setter
    def start_fudge_for_deep_stage(self, start_fudge_for_deep_stage):
        self._start_fudge_for_deep_stage = start_fudge_for_deep_stage

    # max num of iterations for deep stage & interm stages
    @property
    def deep_fit_max_iter(self):
        return self._deep_fit_max_iter
    @deep_fit_max_iter.setter
    def deep_fit_max_iter(self, deep_fit_max_iter):
        self._deep_fit_max_iter = deep_fit_max_iter

    # step threshold by chi2 decrease for deep fit stage during elimination
    @property
    def deep_fit_step_thr(self):
        return self._deep_fit_step_thr
    @deep_fit_step_thr.setter
    def deep_fit_step_thr(self,deep_fit_step_thr):
        self._deep_fit_step_thr = deep_fit_step_thr

    @property
    def interm_fit_max_iter(self):
        return self._interm_fit_max_iter
    @interm_fit_max_iter.setter
    def interm_fit_max_iter(self, interm_fit_max_iter):
        self._interm_fit_max_iter = interm_fit_max_iter

    # step threshold by chi2 decrease for interm fit stage during elimination
    @property
    def interm_fit_step_thr(self):
        return self._interm_fit_step_thr
    @interm_fit_step_thr.setter
    def interm_fit_step_thr(self,interm_fit_step_thr):
        self._interm_fit_step_thr = interm_fit_step_thr


    # if solution passed the prior it's recommended to start from small value of V-coeff. for LM alg. to save time
    @property
    def LevMarV0_priorpassed(self):
        return self._LevMarV0_priorpassed
    @LevMarV0_priorpassed.setter
    def LevMarV0_priorpassed(self,LevMarV0_priorpassed):
        self._LevMarV0_priorpassed = LevMarV0_priorpassed
    

    
    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string



@dataclass 
class eliminator_OUTput:
    """
    Output data for eliminator.

    This object holds at minimum the resulting resonance ladder and the history going from N_start -> N_final .
    """
    
    ladder_OUT: pd.DataFrame
    ladder_IN: pd.DataFrame

    elimination_history: Optional[dict] = None
    success: bool = False
    elim_tot_time: float = np.inf

    
    

class eliminator_by_chi2:
    def __init__(self, 
                 rto: SammyRunTimeOptions, 
                 #sammyINPyw: SammyInputDataYW, 
                 options: elim_OPTs,

                Ta_pair: Particle_Pair,
                datasets: list,
                covariance_data: list,
                experiments: list,

                 ):
        """
        Initialize the class with all parameters specified
        """

        # creating a sammyINPyw object for internal use
        elim_sammyINPyw = SammyInputDataYW(
            particle_pair = Ta_pair,
            resonance_ladder = pd.DataFrame(),

            datasets = datasets,
            experimental_covariance=covariance_data,
            experiments = experiments,

            max_steps = 0,
            iterations = 2,
            step_threshold = 0.01,
            autoelim_threshold = None,

            LS = False,
            LevMar =  True,
            LevMarV = 1.5,

            minF = 1e-5,
            maxF = 10,
            initial_parameter_uncertainty = 0.05
        )


        self.rto = rto
        self.sammyINPyw = elim_sammyINPyw
        self.options = options


    def eliminate(self, 
                  ladder_df : pd.DataFrame,
                  fixed_resonances_df: pd.DataFrame = pd.DataFrame()
                  ) -> eliminator_OUTput: 

        """Main func to eliminate resonances from the input ladder that is in SammyINPyw.resonance_ladder """
        
        start_time = time.time()

        delta_chi2_allowed = self.options.chi2_allowed

        deep_fit_step_thr = self.options.deep_fit_step_thr # threshold for deep fitting stage
        deep_fit_max_iter = self.options.deep_fit_max_iter
        start_deep_fit_from = self.options.start_deep_fit_from

        
        # Initializing model history dictionary
        model_history = {}

        final_model_passed_test = True # set it from the start.

        # ladder for processing - from direct input
        ladder_df = elim_addit_funcs.set_varying_fixed_params(ladder_df = ladder_df,
                                                                    vary_list=[1,0,1]
                                                                    )
        
        if (self.rto.Print):
            print('*'*40)
            print('Input ladder:')
            print(ladder_df)
        

        # set all req. vary fields
        fixed_res_df = elim_addit_funcs.set_varying_fixed_params(ladder_df = fixed_resonances_df,
                                                                    vary_list=[0,1,1] # do not allow to change energy
                                                                    )
        
        if (self.rto.Print):
            print('Side resonances used:')
            print(fixed_res_df)
            print()


        # compiling to one ladder
        ladder = pd.concat([fixed_res_df, ladder_df], ignore_index=True)

        self.sammyINPyw.resonance_ladder = ladder

        if (self.rto.Print):
            print('Combined ladder to start from:')
            print(self.sammyINPyw.resonance_ladder)
            print()


        # saving initial input
        ladder_IN = ladder.copy()
        ladder_OUT = ladder.copy()

        # printout
        if (self.rto.Print):           
            print()
            print('Elimination options used:')
            print('*'*40)
            print(self.options)
            print('*'*40)
            print()            
            

        ### Start elimination
        while True: 

            level_start_time = time.time()

            ### set current level variables to def values
            any_model_passed_test = False 
            best_model_chi2 = float('inf')
            best_removed_resonance = None
            best_model_chars = None

            current_level = len(ladder) # note - from which we are going to N-1!

            ### Identify fixed resonances
            if (fixed_res_df.shape[0] > 0):

                # Extract energies from fixed_side_resonances
                fixed_res_energies = fixed_res_df["E"].tolist()
                # Find those energies in the ladder by energy values (!)
                fixed_resonances = ladder[ladder["E"].isin(fixed_res_energies)]
                # indexes of side resonances to track them
                fixed_resonances_indices = ladder[ladder['E'].isin(fixed_res_energies)].index.tolist()

                if (self.rto.Print):
                    print('Indexes of fixed resonances for current level:')
                    print(fixed_resonances_indices)

            else:
                fixed_resonances = pd.DataFrame()
                fixed_resonances_indices=[]

            
            ### Setup all model information of this level and starting chi2

            if (self.rto.Print):
                print('*'*40)
                print(f'Current level: {current_level}')
                print(f'\t Searching for models with {current_level - 1} resonances...')
                print('*'*40)
                print()
            
            fit_code_init_level = f'init_sol_level_{current_level}'
            initial_ladder_chars = self.evaluate_prior(ladder)

            if (self.rto.Print):        

                print()
                print(f'\t{fit_code_init_level}')
                print(f'\tBase chi2 for level: {initial_ladder_chars.chi2}')
                print()
            
            best_model_chars = initial_ladder_chars
            base_chi2 = np.sum(initial_ladder_chars.chi2)

            # if we are on the level of one resonance - just stop
            if (current_level==1):
                break

            ### test all N-1 priors 
            time_test_priors_start = time.time()
            
            prior_test_out = self.test_priors(current_level, 
                                              fixed_resonances_indices, 
                                              fixed_resonances,
                                              ladder,
                                              delta_chi2_allowed,
                                              base_chi2)
            
            priors_test_time = time.time() - time_test_priors_start

            any_prior_passed_test, any_model_passed_test, best_prior_model_chars, best_prior_chi2, priors_passed_cnt, best_removed_resonance_prior = prior_test_out
        
            ### if any priors passed remove, 
            if (any_prior_passed_test):
                if (self.rto.Print):
                    print()
                    print(f'Priors passed the test...{priors_passed_cnt}')
                    print(f'Best model found {best_removed_resonance_prior}:')
                    print(f'Σχ²:\t{best_prior_chi2}')
                    #print(f'Time for priors test: {np.round(priors_test_time,2)} sec')
                    print(f'Time for priors test: {elim_addit_funcs.format_time_2_str(priors_test_time)[1]}')
                    print()

                best_model_chi2 = best_prior_chi2
                best_removed_resonance = best_removed_resonance_prior
                best_model_chars = best_prior_model_chars
                any_model_passed_test = True

                LevMarV0 = self.options.LevMarV0_priorpassed # if prior passed, starting step should be small

            ### else test all N-1 fitted models 
            else:
                fitted_test_out = self.test_fitted_models(current_level,
                                        fixed_resonances_indices,
                                        fixed_resonances,
                                        ladder,
                                        base_chi2,
                                        delta_chi2_allowed,
                                        best_model_chi2,
                                        any_model_passed_test)
                
                best_removed_resonance, best_model_chars, any_model_passed_test = fitted_test_out

                LevMarV0 = self.options.start_fudge_for_deep_stage


            ### Do deep fitting after selecting model from prior or deep fit       
            
            # if we are doing deep fit on this stage
            # TODO: change this code
            # deep fitting stage - using best_model_chars - 
            # the problem is that after 2 different stages we have different params..
            if (any_prior_passed_test):
                deep_stage_ladder_start = best_model_chars.par
            else:
                deep_stage_ladder_start = best_model_chars.par_post

            if ((deep_stage_ladder_start.shape[0]- fixed_res_df.shape[0]) <= start_deep_fit_from):
                if (self.rto.Print):
                    print()
                    print('Starting "deep" fitting of best initial guess by chi2...')
                    print(f'DA = {deep_fit_max_iter}/{deep_fit_step_thr}')
                    print()

                posterior_deep_SO, sol_fit_time_deep = self.fit_YW_by_ig(ladder_df = deep_stage_ladder_start, 
                                                                        max_steps = deep_fit_max_iter,
                                                                        step_threshold = deep_fit_step_thr,
                                                                        LevMarV0 = LevMarV0)           

            else:
                if (self.rto.Print):
                    print()
                    print('Skipping "deep" fitting stage, utilizing best model for the next step w/o chi2...')
                    print()
                
                posterior_deep_SO = best_model_chars
                sol_fit_time_deep = 0
                
                # based on the priors we should determine if we need to use par or par_post...
                if (any_prior_passed_test):
                    posterior_deep_SO.par_post = posterior_deep_SO.par
                    posterior_deep_SO.chi2_post = posterior_deep_SO.chi2
                    posterior_deep_SO.chi2n_post = posterior_deep_SO.chi2n
                    posterior_deep_SO.pw_post = posterior_deep_SO.pw

                else:
                    pass

            cur_sol_chars_deep = posterior_deep_SO
            deep_chi2 = np.sum(cur_sol_chars_deep.chi2_post)
            benefit_deep_chi2 = deep_chi2 - base_chi2

            selected_ladder_chars = cur_sol_chars_deep
            
            ### printout
            if (self.rto.Print):

                print(f'\t proc_time: {elim_addit_funcs.format_time_2_str(sol_fit_time_deep)[1]}')
                print()
                print(f'\t Benefit in chi2: {benefit_deep_chi2}, while initial benefit for {self.options.interm_fit_max_iter} iter. was {sum(posterior_deep_SO.chi2) - base_chi2}')
    
                print('Deep fitting decision about model selection:')
                print()
                print(f'\t Before: {np.sum(posterior_deep_SO.chi2)}')
                
                print(f'\t After: {np.sum(posterior_deep_SO.chi2_post)}')
                print()


            # checking if final model passed the test
            if ((benefit_deep_chi2 <= delta_chi2_allowed) & final_model_passed_test):
                final_model_passed_test = True
                ladder_OUT = selected_ladder_chars.par_post
            else: 
                final_model_passed_test = False

            level_time = time.time() - level_start_time

            ### final printout, save model data, and continue or break while loop
            if (self.rto.Print):
                print(f'Current Stage Results {current_level}, N_res = {selected_ladder_chars.par_post.shape[0]}')
                cols_to_show = ['E', 'Gn1', 'Gg', 'varyE', 'varyGg', 'varyGn1', 'J_ID']
                print(selected_ladder_chars.par_post[cols_to_show])
                print()
                print(f'Current N_res: {selected_ladder_chars.par_post.shape[0]}')
                print(f'Level {current_level} passed the test: {final_model_passed_test}')
                print('End of deep fitting stage...')
                print()
                print()
                print(f'Level time: \t {elim_addit_funcs.format_time_2_str(level_time)[1]}')
                print(f'Priors test time: \t {elim_addit_funcs.format_time_2_str(priors_test_time)[1]}')

                tot_el_time = time.time() - start_time
                resdif = max(ladder_IN.shape[0] - selected_ladder_chars.par_post.shape[0], 1)

                time_per_res  = np.round(tot_el_time/resdif,1)

                print('*'*40)
                print(f'Current elapsed time: \t {elim_addit_funcs.format_time_2_str(tot_el_time)[1]}')
                print(f'Time per res.: \t {time_per_res} sec')
                print()
                print(f'Estimated TEE: \t {np.round(time_per_res * (selected_ladder_chars.par_post.shape[0] - fixed_res_df.shape[0]) /3600 , 1)} h')
                print(f'\t ~ {elim_addit_funcs.format_time_2_str(time_per_res * (selected_ladder_chars.par_post.shape[0] - fixed_res_df.shape[0]))[1]}')
                print()
                print()
                print('*'*40)
                print()

            model_history[current_level-fixed_res_df.shape[0]] = {
                
                'input_ladder' : ladder,
                'selected_ladder_chars': selected_ladder_chars, # solution with min chi2 on this level
                'any_model_passed_test': any_model_passed_test, # any model on this level of N-1 res..
                'final_model_passed_test': final_model_passed_test,
                'level_time': level_time,
                'total_time': time.time() - start_time
            }

            # stopping criteria
            if (self.options.stop_at_chi2_thr):

                # if stop after the first shot when no models pass the test (speedup)
                if best_removed_resonance is not None and any_model_passed_test:
                    ladder = ladder_OUT 
                else:
                    print(f'Note, stopping on criteria by chi2 threshold {delta_chi2_allowed}')
                    print(f'Any model passed the test: {any_model_passed_test}')
                    break
            else:
                
                # not stopping continuing up to 1 res..
                print('Skipping stopping by chi2 test, going to 1 res model')
                if(ladder.shape[0]==fixed_res_df.shape[0]+1):
                    print('Reached one resonance model.. stopping')
                    break
                else:
                    ladder = selected_ladder_chars.par_post

        
        

    
        ### Now outside of while loop, form output object and return

        total_time = time.time() - start_time

        if (ladder.shape[0]<ladder_IN.shape[0]):
            elim_success = True
        else:
            elim_success = False

        el_out = eliminator_OUTput(
            ladder_OUT = ladder_OUT,
            ladder_IN = ladder_IN,
            elimination_history= model_history,
            success = elim_success,
            elim_tot_time = total_time
            )

        return el_out


    def test_priors(self,
                    current_level, 
                    fixed_resonances_indices, 
                    fixed_resonances,
                    ladder,
                    delta_chi2_allowed,
                    base_chi2
                    ):
        
        # check if prior model pass the test even without refitting after resonance deletion (insignificant resonances)
        any_prior_passed_test = False
        any_model_passed_test = False
        best_removed_resonance_prior = None
        best_prior_chi2 = np.inf
        best_prior_model_chars = None
        priors_passed_cnt = 0

        for j in range(current_level):  # For every resonance in the current ladder
            
            # Skip if side resonances
            if j in fixed_resonances_indices:
                if (self.rto.Print):
                    print('Warning!')
                    print(f'Res. index {j} in fixed:')
                    print()
                    print(fixed_resonances)
                    print()
                continue
            
            ### Create and evaluate a ladder with the j-th resonance removed
            # note - always keep the side-resonances
            prior_ladder, row_removed = self.remove_resonance(ladder, j)
            prior_chars = self.evaluate_prior(prior_ladder) 

            prior_sum_chi2 = np.sum(prior_chars.chi2)
            prior_benefit_chi2 = prior_sum_chi2 - base_chi2

            ### Check if un-fitted N-1 model still is acceptable
            test_result = "✗"  # Cross mark if the test is not passed
            if ((prior_benefit_chi2<=delta_chi2_allowed)):

                test_result = "✓"  # Check mark if the test is passed
                sign = "<="

                any_prior_passed_test = True

                priors_passed_cnt += 1

                # reset best prior chi2 if better
                # best_removed_resonance_prior = None
                if (prior_sum_chi2 < best_prior_chi2):
                    best_prior_chi2 = prior_sum_chi2
                    best_prior_model_chars = prior_chars
                    best_removed_resonance_prior = j
            else:
                sign = ">"

            if (self.rto.Print):
                print()
                print(f'Prior ladder check, deleted {j}, E_λ  = {row_removed["E"].item()}')
                print(f'\tΣχ²: \t {np.round(prior_sum_chi2,4)} \t| base: {np.round(base_chi2,4)} | current best: {np.round(best_prior_chi2,4)}')
                print(f'\t\t\t {np.round(prior_benefit_chi2,4)}\t\t{sign}\t\t{delta_chi2_allowed}\t\t=>\t\t{test_result}')
                print(f'\t {priors_passed_cnt} / {current_level} passed.    ({fixed_resonances.shape[0]} side res.)')
                print()
            
            # check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)
            if (self.options.greedy_mode and any_prior_passed_test):
                if (self.rto.Print):
                    print()
                    print('Using "greedy" mode')
                    print('Skipping search of solutions - taking first solution that satisfies conditions.')
                    print(f'Deleted res. # {best_removed_resonance_prior}')
                    print(f'Chi2: {best_prior_chi2} | {base_chi2}')
                break
            # end check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)


        return (any_prior_passed_test, any_model_passed_test, best_prior_model_chars, best_prior_chi2, priors_passed_cnt, best_removed_resonance_prior)


    def test_fitted_models(self,
                           current_level,
                           fixed_resonances_indices,
                           fixed_resonances,
                           ladder,
                           base_chi2,
                           delta_chi2_allowed,
                           best_model_chi2,
                           any_model_passed_test
                           ):
       
        # if no priors passed the test - do the fitting for each model
        if (self.rto.Print):
            print('*'*40)
            print('No priors passed the test...')
            print(f'Doing limited iterations to find the best model inside current level - {current_level} - > {current_level-1}...')
        
        posteriors_passed_cnt = 0 # number of models that pass the test after fitting
        
        # TODO: change this..
        best_removed_resonance = None
        best_model_chars = None


        # selecting the most perspective model from the chi2 point of view with limited iterations allowed
        for j in range(current_level):  # For every resonance in the current ladder
        
            # Refit the data with this temporary ladder
            # Skip if side resonances
            if j in fixed_resonances_indices:
                if (self.rto.Print):
                    print('Warning!')
                    print(f'Res. index {j} in fixed:')
                    print()
                    print(fixed_resonances)
                    print()
                continue
        
            # Create a ladder with the j-th resonance removed
            # note - always keep the side-resonances
            prior_ladder, row_removed = self.remove_resonance(ladder, j)

            posterior_interm_SO, sol_fit_time_interm = self.fit_YW_by_ig(
                ladder_df = prior_ladder,
                max_steps = self.options.interm_fit_max_iter,
                step_threshold = self.options.interm_fit_step_thr
                )
            
            ### NEW: NOAH - Again, don't need to re-characterize just use old chi2
            # # TODO: ask Noah about situations where we stuck from the init. solution
            # if (posterior_interm_SO.par_post.shape[0]>0):
            #     interm_posterior_ladder = posterior_interm_SO.par_post
            # else:
            #     interm_posterior_ladder =  posterior_interm_SO.par
            # print()
            # characterizing this sol
            # cur_sol_chars, _ = self.fit_YW_by_ig(ladder_df=interm_posterior_ladder, 
            #                             max_steps=0)

            cur_sol_chars = posterior_interm_SO
            interm_step_chi2 = np.sum(cur_sol_chars.chi2_post)

            benefit_chi2 = interm_step_chi2 - base_chi2
            
            test_result = "✗"

            # Check if this model is best so far
            if ((benefit_chi2 <= delta_chi2_allowed)):
                
                # mark that at least one sol passed the chi2 test
                any_model_passed_test = True
                current_model_passed_test = True

                test_result = "✓"  # Check mark if the test is passed
                sign = "<="

                posteriors_passed_cnt +=1

                if (self.rto.Print):
                    print()
                    print('Model passed chi2 test!')
            else:

                current_model_passed_test = False
                sign = ">"

            # TODO: check - so we will use always
            if (interm_step_chi2 < best_model_chi2):
                
                best_model_chi2 = interm_step_chi2
                best_removed_resonance = j
                best_model_chars = cur_sol_chars

            if (self.rto.Print):
                print()
                print(f'Intermediate fitting stage (IA = {self.options.interm_fit_max_iter}/{self.options.interm_fit_step_thr}), deleted {j}, E_λ  = {row_removed["E"].item()}')
                print(f'\tΣχ²:\t{np.round(interm_step_chi2,4)}\tbase: {np.round(base_chi2,4)} | current best: {best_model_chi2}  ')
                print(f'\t\t\t{np.round(benefit_chi2,4)}\t{sign}\t{delta_chi2_allowed}\t => \t {test_result}')
                #print(f'\t\t\t{sol_fit_time_interm} sec for processing')
                print(f'\t\t\tproc_time: {elim_addit_funcs.format_time_2_str(sol_fit_time_interm)[1]}')
                print()
                print(f'\t{posteriors_passed_cnt} / {current_level} passed.\t({fixed_resonances.shape[0]} side res.)')
                print()
            
            # check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)
            if (self.options.greedy_mode and any_model_passed_test):
                if (self.rto.Print):
                    print()
                    print('Using "greedy" mode')
                    print('Skipping search of solutions - taking first solution that satisfies conditions.')
                    print(f'Deleted res. # {best_removed_resonance}')
                    print(f'Chi2: {best_model_chi2} | {base_chi2}')
                break
            # end check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)

                
        if (self.rto.Print):
            print('*'*40)
            # if some of the models passed the test using intermediate stage
            if (any_model_passed_test):
                print(f'{posteriors_passed_cnt} models passed the test, using best for a deep fit')
                print(f'Best model is\t{best_removed_resonance}.')
                print(f'Σχ²:\t{best_model_chi2}')
            else:
                print('No models passed the test!')
                print(f'Best model is {best_removed_resonance}.')

            print('End Doing limited iterations to find the best model inside current level...')

        return (best_removed_resonance, best_model_chars, any_model_passed_test)





    def evaluate_prior(self,
                       ladder_df: pd.DataFrame):

        # redefining rto and inputs
        cur_rto = SammyRunTimeOptions(
            sammyexe = self.rto.path_to_SAMMY_exe,
            options = {"Print":         self.rto.Print,
                       "bayes":         False,
                       "keep_runDIR":   self.rto.keep_runDIR,
                       "sammy_runDIR":  self.rto.sammy_runDIR
                        }
        )

        cur_SI_YW = SammyInputDataYW(
            particle_pair = self.sammyINPyw.particle_pair,
            resonance_ladder = ladder_df,

            datasets = self.sammyINPyw.datasets,
            experimental_covariance=self.sammyINPyw.experimental_covariance,
            experiments = self.sammyINPyw.experiments,
        )

        sammy_OUT = sammy_functions.run_sammy_YW(sammyINPyw = cur_SI_YW, 
                                                 sammyRTO = cur_rto)

        return sammy_OUT
    


    def fit_YW_by_ig(self, 
                     ladder_df:pd.DataFrame,
                     max_steps: int = 0,
                     step_threshold: float = 0.01,
                     LevMarV0 = 0.1):
        """Wrapper to fit the data with given params using YW scheme"""

        time_start = time.time()

        # redefining rto and inputs
        cur_rto = SammyRunTimeOptions(
            sammyexe = self.rto.path_to_SAMMY_exe,
            options = {"Print":  self.rto.Print,
                "bayes":  True,
                "keep_runDIR": self.rto.keep_runDIR,
                "sammy_runDIR": self.rto.sammy_runDIR
                }
        )

        cur_SI_YW = SammyInputDataYW(
            particle_pair = self.sammyINPyw.particle_pair,
            resonance_ladder = ladder_df,

            datasets = self.sammyINPyw.datasets,
            experimental_covariance=self.sammyINPyw.experimental_covariance,
            experiments = self.sammyINPyw.experiments,

            max_steps = max_steps,
            iterations = self.sammyINPyw.iterations,
            step_threshold = step_threshold,
            autoelim_threshold = None,

            LS = False,
            LevMar = True,
            LevMarV = self.sammyINPyw.LevMarV,

            minF = self.sammyINPyw.minF,
            maxF = self.sammyINPyw.maxF,

            initial_parameter_uncertainty = LevMarV0
        )

        sammy_OUT = sammy_functions.run_sammy_YW(sammyINPyw = cur_SI_YW, 
                                                 sammyRTO = cur_rto)

        time_proc = time.time() - time_start

        return sammy_OUT, time_proc


    def remove_resonance(self,
                         ladder: pd.DataFrame,
                         index_to_remove: int):
        """Removes a resonance from the ladder and returns the removed row."""

        if index_to_remove in ladder.index:
            removed_row = ladder.loc[[index_to_remove]].copy()  # Get the row before removal
            new_ladder = ladder.drop(index_to_remove)  # Drop the row to get the new ladder
            new_ladder.reset_index(drop=True, inplace=True)  # Reindex the new ladder
        else:
            print(ladder)
            raise ValueError(f'Invalid index {index_to_remove}')
        return new_ladder, removed_row