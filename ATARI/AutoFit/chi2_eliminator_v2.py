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

# end imports


### classes definitions
class elim_OPTs:
    """
    Options and settings for a single elimination routine.
    
    Parameters
    ----------
    chi2_allowed : float
        value of chi2 allowed difference when comparing 2 models,
    fixed_resonances_df: pd.Dataframe
        dataframe with side resonances, fixed to track during elimination,
    stop_at_chi2_thr: Bool
        Boolean value which tells to stop if during search of a models there was 
        no models that passed the chi2 test, if false - continue to delete resonances until we will not have at least one resonance.
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    Attributes
    ----------
    chi2_allowed : float
        value of chi2 difference allowed when comparing 2 models.
    fixed_resonances_df: pd.Dataframe
        dataframe with side resonances, fixed to track during elimination.
    deep_fit_max_iter: int
        allowed number of iterations to perform YW fitting
    deep_fit_step_thr: flota,
        chi2 step threshold used for YW fitting procedure
    start_fudge_for_deep_stage: float
        Starting value of a fudge factor used for YW scheme
    
    """
    def __init__(self, **kwargs):
        
        # default values for all 
        #self._input_ladder = kwargs.get('sampleRES', pd.DataFrame())
        self._chi2_allowed = kwargs.get('chi2_allowed', 8)
        self._fixed_resonances_df = kwargs.get('fixed_resonances_df', pd.DataFrame())
        
        self._deep_fit_max_iter = kwargs.get('deep_fit_max_iter', 20)
        self._deep_fit_step_thr = kwargs.get('deep_fit_step_thr', 0.01)
        self._start_fudge_for_deep_stage = kwargs.get('start_fudge_for_deep_stage', 0.1)

        self._LevMarV0_priorpassed = kwargs.get('LevMarV0_priorpassed', 0.01)

        self._stop_at_chi2_thr = kwargs.get('stop_at_chi2_thr', True) # by default stops when didn't find the model that passed the test

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

    #side resonances (fixed in energy only)
    @property 
    def fixed_resonances_df(self):
        return self._fixed_resonances_df
    @fixed_resonances_df.setter
    def fixed_resonances_df(self, fixed_resonances_df):
        self._fixed_resonances_df = fixed_resonances_df

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
                 sammyINPyw: SammyInputDataYW, 
                 options: elim_OPTs):
        """
        Initialize the class with all parameters specified, all obligatory
        """

        self.rto = rto
        self.sammyINPyw = sammyINPyw
        self.options = options


    def eliminate(self, 
                  ladder_df : pd.DataFrame = pd.DataFrame() 
                  ) -> eliminator_OUTput: 

        """Main func to eliminate resonances from the input ladder that is in SammyINPyw.resonance_ladder """
        
        start_time = time.time()

        delta_chi2_allowed = self.options.chi2_allowed
        interm_iter_allowed = self.options.deep_fit_max_iter # allowed number of iterations for YW if no priors passes the test
        deep_fit_step_thr = self.options.deep_fit_step_thr # threshold for deep fitting stage
        fixed_res_df = self.options.fixed_resonances_df

        # Initializing model history dictionary
        model_history = {}
        final_model_passed_test = True # set it from the start.

        # ladder for processing - from direct input or from sammyINPyw
        if (ladder_df.shape[0]==0):
            ladder = self.sammyINPyw.resonance_ladder.copy()
        else:
            ladder = ladder_df

        # saving initial input
        ladder_IN = ladder.copy()
        ladder_OUT = ladder.copy()

        # printout
        if (self.rto.Print):
            print('*'*40)
            print('Elimination cycle printout enabled')
            print('*'*40)
            print()
            print(f'Chi2 thresold applied for model selection: {delta_chi2_allowed}')
            print(f'')
            print()
            print('Input ladder:')
            print(ladder)
            print('Side resonances used:')
            print(fixed_res_df)
            print()
            print('Stopping option:')
            print(f'\t stop_at_chi2_thr: {self.options.stop_at_chi2_thr}')
            

        ### Start elimination
        while True: 

            level_start_time = time.time()

            ### set current level variables to def values
            any_model_passed_test = False 
            best_model_chi2 = float('inf')
            best_removed_resonance = None
            best_model_chars = None
            current_level = len(ladder)

            ### Identify fixed resonances
            if (fixed_res_df.shape[0] == 2):

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
            all_models = {}

            if (self.rto.Print):
                print('*'*40)
                print(f'Current level: {current_level}')
                print(f'\t Searching for models with {current_level - 1} resonances...')
                print('*'*40)
                print()
            
            fit_code_init_level = f'init_sol_level_{current_level}'
            initial_ladder_chars = self.evaluate_prior(ladder)

            if (self.rto.Print):        
                # self.printout_prior_params(sammy_OUT = initial_ladder_chars, 
                #                         addit_str = fit_code_init_level)
                print()
                print(f'\t {fit_code_init_level}')
                print(f'\t {initial_ladder_chars.chi2}') # TODO: check this also..
                print()
            
            best_model_chars = initial_ladder_chars
            base_chi2 = np.sum(initial_ladder_chars.chi2)

            # if we are on the level of one resonance - just stop
            if (current_level==1):
                break

            ### test all N-1 priors 
            prior_test_out = self.test_priors(current_level, 
                                              fixed_resonances_indices, 
                                              fixed_resonances,
                                              ladder,
                                              delta_chi2_allowed,
                                              base_chi2,
                                              all_models)
            
            any_prior_passed_test, any_model_passed_test, best_prior_model_chars, best_prior_chi2, priors_passed_cnt, best_removed_resonance_prior = prior_test_out
        
            ### if any priors passed remove, 
            if (any_prior_passed_test):
                if (self.rto.Print):
                    print()
                    print(f'Priors passed the test...{priors_passed_cnt}')
                    print(f'Best model found {best_removed_resonance_prior}:')
                    print(f'Σχ²: \t {best_prior_chi2}')
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
                                        interm_iter_allowed,
                                        base_chi2,
                                        delta_chi2_allowed,
                                        best_model_chi2,
                                        all_models,
                                        any_model_passed_test)
                
                best_removed_resonance, best_model_chars, any_model_passed_test = fitted_test_out

                LevMarV0 = self.options.start_fudge_for_deep_stage


            ### Do deep fitting after selecting model from prior or deep fit       
            if (self.rto.Print):
                print()
                print('Starting "deep" fitting of best initial guess by chi2...')
                print()

            # TODO: change this code
            # deep fitting stage - using best_model_chars - 
            # the problem is that after 2 different stages we have different params..
            if (any_prior_passed_test):
                deep_stage_ladder_start = best_model_chars.par
            else:
                deep_stage_ladder_start = best_model_chars.par_post

            posterior_deep_SO, sol_fit_time_deep = self.fit_YW_by_ig(ladder_df = deep_stage_ladder_start, 
                                                                     max_steps = interm_iter_allowed,
                                                                     step_threshold = deep_fit_step_thr,
                                                                     LevMarV0 = LevMarV0) 

            cur_sol_chars_deep = posterior_deep_SO

            deep_chi2 = np.sum(cur_sol_chars_deep.chi2_post)

            benefit_deep_chi2 = deep_chi2 - base_chi2
            
            ### printout
            if (self.rto.Print):

                # self.printout_prior_params(sammy_OUT = cur_sol_chars_deep, 
                #                         addit_str = fit_code)
                
                print(f'\t proc_time {sol_fit_time_deep}  s')
                print()
                print(f'\t Benefit in chi2: {benefit_deep_chi2}, while initial benefit for {interm_iter_allowed} iter. was {sum(posterior_deep_SO.chi2) - base_chi2}')
       
                print('Deep fitting decision about model selection:')
                print()
                print(f'\t Before: {np.sum(posterior_deep_SO.chi2)}')
                
                print(f'\t After: {np.sum(posterior_deep_SO.chi2_post)}')
                print()

            ### Logic for re-assigning models in loop
            # TODO: it must be always >=

            # if (benefit_deep_chi2 > (sum(best_model_chars.chi2) - base_chi2) ):
                
            #     # taking a model before
            #     selected_ladder_chars = best_model_chars

            #     if (self.rto.Print):
            #         print('Note, after deep fitting - the benefit of chi2 is worse than before.')

            # else:
            #     selected_ladder_chars = cur_sol_chars_deep
            #     if (self.rto.Print):
            #         print('After deep fitting - the benefit of chi2 is bettter than before.')
            #         print('Applying deep fit results as final results for level.')

            selected_ladder_chars = cur_sol_chars_deep

            # checking if final model passed the test
            if ((benefit_deep_chi2 <= delta_chi2_allowed) & final_model_passed_test):
                final_model_passed_test = True
                ladder_OUT = selected_ladder_chars.par_post
            else: 
                final_model_passed_test = False

            level_time = time.time() - level_start_time

            ### final printout, save model data, and continue or break while loop
            if (self.rto.Print):
                print('Deep fitting stage. Results')
                cols_to_show = ['E', 'Gn1', 'Gg', 'varyE', 'varyGg', 'varyGn1', 'J_ID']
                print(selected_ladder_chars.par_post[cols_to_show])
                print()
                print(f'Current N_res: {selected_ladder_chars.par_post.shape[0]}')
                print(f'Level {current_level} passed the test: {final_model_passed_test}')
                print('End of deep fitting stage...')
                print()
                print()
                print(f'Level time: {np.round(level_time, 1)} sec')
                tot_el_time = time.time() - start_time
                resdif = max(ladder_IN.shape[0] - selected_ladder_chars.par_post.shape[0], 1)

                print(f'Total elapsed time: {np.round(tot_el_time, 1)} sec')
                print(f'time per res.: {np.round(tot_el_time/resdif, 1)} sec')
                print()
                print()
                print('*'*40)
                print()

            model_history[current_level] = {
                
                'input_ladder' : ladder,
                #'all_models': all_models, # store all models??
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
                    print(f'Note, using the stopping criteria by chi2 threshold {delta_chi2_allowed}')
                    print('Threshold reached')
                    print(f'Any model passed the test: {any_model_passed_test}')
                    break
            else:
                
                # not stopping continuing up to 1 res..
                print('Skipping stopping by chi2 test, going to 1 res model')
                if(ladder.shape[0]==1):
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

        
        
    # """TODO: remove this function somewhere, maybe in utils?"""
    # def generate_sammy_rundir_uniq_name(self, 
    #                                     path_to_sammy_temps: str = './sammy_temps/',
    #                                     addit_str: str = ''):

    #     if not os.path.exists(path_to_sammy_temps):
    #         os.mkdir(path_to_sammy_temps)

    #     timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    #     # Combine timestamp and random characters
    #     unique_string = timestamp + str(uuid.uuid4())
        
    #     # Truncate the string to 100 characters
    #     unique_string = unique_string[:100]

    #     sammy_rundirname = path_to_sammy_temps+'SAMMY_RD_'+addit_str+'_'+unique_string+'/'

    #     return sammy_rundirname




    def test_priors(self,
                    current_level, 
                    fixed_resonances_indices, 
                    fixed_resonances,
                    ladder,
                    delta_chi2_allowed,
                    base_chi2,
                    all_models
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
            current_prior_passed_test = False
            test_result = "✗"  # Cross mark if the test is not passed
            if ((prior_benefit_chi2<=delta_chi2_allowed)):

                test_result = "✓"  # Check mark if the test is passed
                sign = "<="

                any_prior_passed_test = True
                current_prior_passed_test = True
                priors_passed_cnt += 1

                # reset best prior chi2 if better
                if (prior_sum_chi2 < best_prior_chi2):
                    best_prior_chi2 = prior_sum_chi2
                    best_prior_model_chars = prior_chars
                    best_removed_resonance_prior = j
            else:
                sign = ">"

            if (self.rto.Print):
                print(f'Prior ladder check, deleted {j}, E_λ  = {row_removed["E"].item()}')
                print(f'\t\tΣχ²: \t {np.round(prior_sum_chi2,4)} | \t base: {np.round(base_chi2,4)}   ')
                print(f'\t\t{np.round(prior_benefit_chi2,4)}\t\t{sign}\t\t{delta_chi2_allowed}\t\t=>\t\t{test_result}')
                print()

            # saving models data
            all_models[j] = {
                'prior_chars': prior_chars,
                'posterior_chars': prior_chars,
                'fit_time': 0,
                'test_passed': current_prior_passed_test
            }

        return (any_prior_passed_test, any_model_passed_test, best_prior_model_chars, best_prior_chi2, priors_passed_cnt, best_removed_resonance_prior)


    def test_fitted_models(self,
                           current_level,
                           fixed_resonances_indices,
                           fixed_resonances,
                           ladder,
                           interm_iter_allowed,
                           base_chi2,
                           delta_chi2_allowed,
                           best_model_chi2,
                           all_models,
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
                max_steps = interm_iter_allowed,
                # step_threshold = self.options.deep_fit_step_thr
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

                # # TODO: check - so we will use always
                # if (interm_step_chi2 < best_model_chi2):
                    
                #     best_model_chi2 = interm_step_chi2
                #     best_removed_resonance = j
                #     best_model_chars = cur_sol_chars


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
                print(f'Intermediate fitting, deleted {j}, E_λ  = {row_removed["E"].item()}')
                print(f'\t Σχ²: \t {np.round(interm_step_chi2,4)} | \t base: {np.round(base_chi2,4)}   ')
                print(f'\t\t  {np.round(benefit_chi2,4)}  \t {sign} \t {delta_chi2_allowed} \t => \t {test_result}')
                print()
                
            # # Store this model's data
            # all_models[j] = {
            #     'posterior_chars': cur_sol_chars,
            #     'fit_time': sol_fit_time_interm,
            #     'test_passed': current_model_passed_test
            # }

        if (self.rto.Print):
            print('*'*40)
            # if some of the models passed the test using intermediate stage
            if (any_model_passed_test):
                print(f'{posteriors_passed_cnt} models passed the test, using best for a deep fit')
                print(f'Best model is {best_removed_resonance}.')
                print(f'Σχ²: {best_model_chi2}')
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
            options = {"Print":   False, #self.rto.Print,
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
    
    
    def printout_prior_params(self, 
                                 sammy_OUT: SammyOutputData,
                                 addit_str: str = ''):
    
        print()
        print(f'Solution characterization: {addit_str} ')
        print(f'    N_res: \t {sammy_OUT.par.shape[0]}')
        #print(f'    NLLW: \t {sol_chars_dict["fit"]["NLLW"]}')
        print(f'    Σχ²: \t {np.round(np.sum(sammy_OUT.chi2_post),3)}' ) # / {np.round(sum(sol_chars_dict["fit"]["chi2_n"].values()),3)}')
        #print(f'    Σ AICc: \t {np.round(sum(sol_chars_dict["fit"]["AICc"].values()),3)}')
        #print(f'    SSE: \t {sol_chars_dict["SSE"]}')
        print()

        return True