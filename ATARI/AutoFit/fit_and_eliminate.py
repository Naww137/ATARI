from typing import Protocol
from ATARI.AutoFit.functions import * #eliminate_small_Gn, update_vary_resonance_ladder, get_external_resonance_ladder, get_starting_feature_bank
import numpy as np
import pandas as pd
from copy import copy
from ATARI.AutoFit import sammy_interface_bindings
from ATARI.ModelData.particle_pair import Particle_Pair
import time
from ATARI.AutoFit import elim_addit_funcs
from datetime import datetime
from typing import Optional
from dataclasses import dataclass


class FitAndEliminateOPT:
    """
    Options for the fit and eliminate class.
    Currently only implemented for elimination with stepwise regression.

    Parameters
    ----------
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    Attributes
    ----------
    fitpar1: list = [0,0,1]
        Boolean list for fit 1 that determines which parameters will be optimized (E, Gg, Gn1).
    fitpar2: list = [1,1,1]
        Boolean list for fit 2 that determines which parameters will be optimized (E, Gg, Gn1).
    width_elimination: bool = False
        Option to eliminate resonances during initial fitting stages based on neutron width.
    Gn_threshold: Float = 1e-2
        Neutron width threshold for width-based elimination if width_elimination=True.
    decrease_chi2_threshold_for_width_elimination: bool = True
        If width_elimination=True, decrease the chi2 threshold convergence criteria

    fitpar_elim: list = [1,0,1]
        Boolean list for elimination that determines which parameters will be optimized (E, Gg, Gn1).
    chi2_allowed : float = 0
        value of chi2 allowed difference when comparing 2 models
    
    greedy_mode: bool = False
        If set to True - eliminator will not test all models of current branch and select best of them - it will use first model of current branch that passed the test 
    stop_at_chi2_thr: Bool
        Boolean value which tells to stop if during search of a models there was 
        no models that passed the chi2 test, if false - continue to delete resonances until we
        will not have at least one resonance.

    interm_fit_max_iter: int
        allowed number of iterations for intermediate stage to perform YW fitting
    interm_fit_step_thr: float,
        chi2 step threshold used for YW fitting procedure in intermediate fitting stage

    LevMarV0_priorpassed: float
        Starting value of a fidge factor used for YW scheme if model passed the test without refitting priors tests
    deep_fit_max_iter: int
        allowed number of iterations for deep-fitting stage 
    deep_fit_step_thr: float,
        chi2 step threshold used for YW fitting procedure
    start_fudge_for_deep_stage: float
        Starting value of a fudge factor used for YW scheme
    start_deep_fit_from: int
        determines the number of resonances on a current level on a path from N->N-1 where deep fit start from (see default value)

    stop_at_chi2_thr: bool = False
        Boolean value which tells to stop if during search of a models there was no models that passed the chi2 test, if false - continue to delete resonances until we will not have at least one resonance.

    """
    def __init__(self, **kwargs):

        ### Initial fit settings
        self._fitpar1 = [0,0,1]
        self._fitpar2 = [1,0,1]
        self._width_elimination = False
        self._Gn_threshold = 1e-2
        self._decrease_chi2_threshold_for_width_elimination = False

        ### elimination options
        self._fitpar_elim = [1,0,1]
        self._chi2_allowed = kwargs.get('chi2_allowed', 0)
        self._greedy_mode = kwargs.get('greedy_mode', False)

        self._interm_fit_max_iter = kwargs.get('interm_fit_max_iter', 20)
        self._interm_fit_step_thr = kwargs.get('interm_fit_step_thr', 0.01)

        self._LevMarV0_priorpassed = kwargs.get('LevMarV0_priorpassed', 0.01)
        self._start_deep_fit_from = kwargs.get('start_deep_fit_from', np.inf) 
        self._deep_fit_max_iter = kwargs.get('deep_fit_max_iter', 20)
        self._deep_fit_step_thr = kwargs.get('deep_fit_step_thr', 0.01)
        self._start_fudge_for_deep_stage = kwargs.get('start_fudge_for_deep_stage', 0.1)

        self._stop_at_chi2_thr = kwargs.get('stop_at_chi2_thr', False)             
        self._final_stage_vary_pars = kwargs.get('final_stage_vary_pars', [1,0,1]) 

        ### Other
        self._print_bool = True

        ### Resonance Statistics
        self._Wigner_informed_variable_selection = False
        self._PorterThomas_informed_variable_selection = False

        ### set kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    ### define repr
    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    

    ### Other
    @property
    def print_bool(self):
        return self._print_bool
    @print_bool.setter
    def print_bool(self, print_bool):
        self._print_bool = print_bool

    ### Resonance Statistics
    @property
    def Wigner_informed_variable_selection(self):
        return self._Wigner_informed_variable_selection
    @Wigner_informed_variable_selection.setter
    def Wigner_informed_variable_selection(self, Wigner_informed_variable_selection):
        self._Wigner_informed_variable_selection = Wigner_informed_variable_selection
    
    @property
    def PorterThomas_informed_variable_selection(self):
        return self._PorterThomas_informed_variable_selection
    @PorterThomas_informed_variable_selection.setter
    def PorterThomas_informed_variable_selection(self, PorterThomas_informed_variable_selection):
        self._PorterThomas_informed_variable_selection = PorterThomas_informed_variable_selection

    ### Initial Fit Opts
    @property
    def fitpar1(self):
        return self._fitpar1
    @fitpar1.setter
    def fitpar1(self, fitpar1):
        self._fitpar1 = fitpar1

    @property
    def fitpar2(self):
        return self._fitpar2
    @fitpar2.setter
    def fitpar2(self, fitpar2):
        self._fitpar2 = fitpar2

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


    ### Elim Opts
    @property
    def chi2_allowed(self):
        return self._chi2_allowed
    @chi2_allowed.setter
    def chi2_allowed(self, chi2_allowed):
        self._chi2_allowed = chi2_allowed
    
    @property
    def stop_at_chi2_thr(self):
        return self._stop_at_chi2_thr
    @stop_at_chi2_thr.setter
    def stop_at_chi2_thr(self, stop_at_chi2_thr):
        self._stop_at_chi2_thr = stop_at_chi2_thr

    @property
    def final_stage_vary_pars(self):
        return self._final_stage_vary_pars 
    @final_stage_vary_pars.setter
    def final_stage_vary_pars(self, final_stage_vary_pars ):
        self._final_stage_vary_pars = final_stage_vary_pars 

    @property
    def greedy_mode(self):
        return self._greedy_mode
    @greedy_mode.setter
    def greedy_mode(self, greedy_mode):
        self._greedy_mode = greedy_mode

    @property 
    def start_deep_fit_from(self):
        return self._start_deep_fit_from
    @start_deep_fit_from.setter
    def start_deep_fit_from(self, start_deep_fit_from):
        self._start_deep_fit_from = start_deep_fit_from
    
    @property 
    def start_fudge_for_deep_stage(self):
        return self._start_fudge_for_deep_stage
    @start_fudge_for_deep_stage.setter
    def start_fudge_for_deep_stage(self, start_fudge_for_deep_stage):
        self._start_fudge_for_deep_stage = start_fudge_for_deep_stage

    @property
    def deep_fit_max_iter(self):
        return self._deep_fit_max_iter
    @deep_fit_max_iter.setter
    def deep_fit_max_iter(self, deep_fit_max_iter):
        self._deep_fit_max_iter = deep_fit_max_iter

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

    @property
    def interm_fit_step_thr(self):
        return self._interm_fit_step_thr
    @interm_fit_step_thr.setter
    def interm_fit_step_thr(self,interm_fit_step_thr):
        self._interm_fit_step_thr = interm_fit_step_thr

    @property
    def LevMarV0_priorpassed(self):
        return self._LevMarV0_priorpassed
    @LevMarV0_priorpassed.setter
    def LevMarV0_priorpassed(self,LevMarV0_priorpassed):
        self._LevMarV0_priorpassed = LevMarV0_priorpassed

    @property
    def fitpar_elim(self):
        return self._fitpar_elim
    @fitpar_elim.setter
    def fitpar_elim(self,fitpar_elim):
        self._fitpar_elim = fitpar_elim


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

@dataclass 
class FitAndEliminateOUT:

    initial_fits        : Optional[list] = None
    elimination_history : Optional[dict] = None

    derivative_evaluations : Optional[list] = None
    external_resonance_indices : Optional[list] = None



class FitAndEliminate:

    def __init__(self,
                 solver_initial: sammy_interface_bindings.Solver,
                 solver_eliminate: sammy_interface_bindings.Solver,
                 options: FitAndEliminateOPT,
                 particle_pair:Particle_Pair = None,
                ):

        """
        Initialize the class with all parameters specified
        """

        self.rto = copy(solver_eliminate.sammyRTO)
        self.rto.bayes = False
        self.rto_bayes = copy(solver_eliminate.sammyRTO)
        self.rto_bayes.bayes = True

        self.solver_initial = solver_initial
        self.solver_eliminate = solver_eliminate
        self.options = options
        self.total_derivative_evaluations = 0
        # self.derivative_evaluations = []
        # self.options = options

        self.particle_pair = particle_pair

        self.output = FitAndEliminateOUT()
        self.output.derivative_evaluations = []
        self.output.initial_fits = []


    def initial_fit(self,
                    initial_feature_bank,
                    fixed_resonance_ladder = pd.DataFrame(),
                    ):

        ### Fit 1 on Gn only
        if self.options.print_bool: 
            print("========================================\nInitial Fit 1\n========================================")
            print(f"Options to vary: {self.options.fitpar1}")

        initial_feature_bank = update_vary_resonance_ladder(initial_feature_bank, varyE = self.options.fitpar1[0], varyGg = self.options.fitpar1[1], varyGn1 = self.options.fitpar1[2])
        # external_resonance_ladder = update_vary_resonance_ladder(external_resonance_ladder, varyE = self.options.fitpar_external[0], varyGg = self.options.fitpar_external[1], varyGn1 = self.options.fitpar_external[2])
        total_resonance_ladder, fixed_resonance_indices = concat_external_resonance_ladder(initial_feature_bank, fixed_resonance_ladder)
        self.output.external_resonance_indices = fixed_resonance_indices

        outs_fit_1 = self.fit_and_eliminate_by_Gn(total_resonance_ladder, fixed_resonance_indices)
        reslad_1 = copy(outs_fit_1[-1].par_post)
        assert(isinstance(reslad_1, pd.DataFrame))

        if np.all(self.options.fitpar1 == self.options.fitpar2):
            if self.options.print_bool: print(f"Options to vary for initial fit 2 are the same, skipping")
            samout_final = outs_fit_1[-1]
        else:
            if self.options.print_bool: 
                print("========================================\nInitial Fit 2\n========================================")
                print(f"Options to vary: {self.options.fitpar2}")

            initial_feature_bank, fixed_resonance_ladder = separate_external_resonance_ladder(reslad_1, fixed_resonance_indices)
            initial_feature_bank = update_vary_resonance_ladder(initial_feature_bank, varyE = self.options.fitpar2[0], varyGg = self.options.fitpar2[1], varyGn1 = self.options.fitpar2[2])
            initial_feature_bank, fixed_resonance_indices = concat_external_resonance_ladder(initial_feature_bank, fixed_resonance_ladder)
            self.output.external_resonance_indices = fixed_resonance_indices

            outs_fit_2 = self.fit_and_eliminate_by_Gn(initial_feature_bank, fixed_resonance_indices)
            samout_final = outs_fit_2[-1]

            for out in outs_fit_1 + outs_fit_2:
                self.output.initial_fits.append(out)
                self.total_derivative_evaluations += out.total_derivative_evaluations
                self.output.derivative_evaluations.append(out.total_derivative_evaluations)
        
        return samout_final 
    


    def fit_and_eliminate_by_Gn(self, resonance_ladder, external_resonance_indices):
        
        if self.options.print_bool: print(f"Initial solve from {len(resonance_ladder)-len(external_resonance_indices)} resonance features\n")
        sammyOUT_fit = self.solver_initial.fit(resonance_ladder, external_resonance_indices)
        outs = [sammyOUT_fit]

        if self.options.width_elimination:
            eliminating = True
            while eliminating:
                internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(sammyOUT_fit.par_post, external_resonance_indices)
                internal_resonance_ladder_reduced, fraction_eliminated = eliminate_small_Gn(internal_resonance_ladder, self.options.Gn_threshold)
                resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(internal_resonance_ladder_reduced, external_resonance_ladder)
                if fraction_eliminated == 0.0:
                    eliminating = False
                elif fraction_eliminated == 100.0:
                    raise ValueError("Eliminated all resonances due to width, please change settings")
                else:
                    # if self.options.decrease_chi2_threshold_for_width_elimination:
                    #     self.solver.sammyINP.step_threshold *= 0.1
                    if self.options.print_bool: 
                        print(f"\n----------------------------------------\nEliminated {round(fraction_eliminated*100, 2)}% of resonance features based on neutron width")
                        print(f"Resolving with {len(internal_resonance_ladder_reduced)} resonance features\n----------------------------------------\n")
                    sammyOUT_fit = self.solver_initial.fit(resonance_ladder, external_resonance_indices)
                    outs.append(sammyOUT_fit)
            if self.options.print_bool: print(f"\nComplete after no neutron width features below threshold\n")

        return outs
    

    def eliminate(self, 
                  initial_feature_bank : pd.DataFrame,
                  target_ires = 0,
                  fixed_resonance_ladder: pd.DataFrame = pd.DataFrame()
                  ): 

        """Main func to eliminate resonances from the input ladder that is in SammyINPyw.resonance_ladder """
        
        start_time = time.time()

        delta_chi2_allowed = self.options.chi2_allowed
        delta_obj_allowed = delta_chi2_allowed # FIXME: ADD OBJECTIVE VALUE FUNCTIONALITY

        deep_fit_step_thr = self.options.deep_fit_step_thr # threshold for deep fitting stage
        deep_fit_max_iter = self.options.deep_fit_max_iter
        start_deep_fit_from = self.options.start_deep_fit_from


        # Initializing model history dictionary
        model_history = {}
        final_model_passed_test = True # set it from the start.

        # set_vary options for initial feature bank
        initial_feature_bank = elim_addit_funcs.set_varying_fixed_params(ladder_df=initial_feature_bank, vary_list=self.options.fitpar_elim)

        if (self.options.print_bool):
            print('*'*40); print('Input ladder:'); print(initial_feature_bank)
        

        # set all req. vary fields
        # fixed_res_df = elim_addit_funcs.set_varying_fixed_params(ladder_df = fixed_resonances_df,
        #                                                             vary_list=self.options.fitpar_external #[0,1,1] # do not allow to change energy
        #                                                             )
        
        if (self.options.print_bool):
            print('Side resonances used:')
            print(fixed_resonance_ladder)
            print()


        # compiling to one ladder
        ladder, fixed_resonances_indices = concat_external_resonance_ladder(initial_feature_bank, fixed_resonance_ladder)
        # saving initial input
        ladder_IN = ladder.copy()
        ladder_OUT = ladder.copy()

        # printout
        if (self.options.print_bool):
            print('Combined ladder to start from:')
            print(ladder)
            print()
        if (self.options.print_bool):           
            print()
            print('Elimination options used:')
            print('*'*40)
            print(self.options)
            print('*'*40)
            print()            
        
        # Do we need to keep initial level in history of elimination?
        # TODO: add initial level 
        input_ladder_chars = self.evaluate_prior(ladder)

        # set the post = to prior
        input_ladder_chars.par_post = input_ladder_chars.par
        input_ladder_chars.chi2_post = input_ladder_chars.chi2 
        input_ladder_chars.chi2n_post = input_ladder_chars.chi2n 

        input_ladder_chars.pw_post = input_ladder_chars.pw
        
        input_obj_stopping_criteria = {}
        input_ladder_chi2 = sum(input_ladder_chars.chi2)

        for Wigner_informed_stopping_criteria in (False, True):
            for PorterThomas_informed_stopping_criteria in (False, True):
                obj_stopping_criteria = objective_func(input_ladder_chi2, input_ladder_chars.par, self.particle_pair, fixed_resonances_indices,
                                                        Wigner_informed_stopping_criteria, PorterThomas_informed_stopping_criteria)
                obj_type = 'chi2'
                if Wigner_informed_stopping_criteria:
                    obj_type += '+Wig'
                if PorterThomas_informed_stopping_criteria:
                    obj_type += '+PT'
                input_obj_stopping_criteria[obj_type] = obj_stopping_criteria

        input_num_res = ladder.shape[0]-fixed_resonance_ladder.shape[0]
        model_history[input_num_res] = {
                'input_ladder' : ladder,
                'selected_ladder_chars': input_ladder_chars, # solution with min chi2 on this level
                'any_model_passed_test': True, # any model on this level of N-1 res..
                'final_model_passed_test': True,
                'level_time': 0,
                'total_time': time.time() - start_time,
                'objective_value_stopping_criteria': input_obj_stopping_criteria,
            }
        
        if (self.options.print_bool): 
            print('Initial model embedded to history')

        # End to add initial level


        ### Start elimination
        while True: 

            level_start_time = time.time()

            ### set current level variables to def values
            any_model_passed_test = False 
            best_model_obj = np.inf
            best_removed_resonance = None
            best_model_chars = None

            current_level = len(initial_feature_bank) # note - from which we are going to N-1!

            if (self.options.greedy_mode):
                initial_feature_bank = initial_feature_bank.sort_values(by='Gn1') # sort ladder by Gn - to del smallest res. first
            
            initial_feature_bank.reset_index(drop=True, inplace=True)

            ### Identify fixed resonances
            if (fixed_resonance_ladder.shape[0] > 0):
                # fixed_res_energies = fixed_resonance_ladder["E"].tolist() # Extract energies from fixed_side_resonances
                # fixed_resonances = ladder[ladder["E"].isin(fixed_res_energies)]# Find those energies in the ladder by energy values (!)
                # fixed_resonances_indices = ladder[ladder['E'].isin(fixed_res_energies)].index.tolist()# indexes of side resonances to track them
                if (self.options.print_bool):
                    print('Indexes of fixed resonances for current level:')
                    print(fixed_resonances_indices)
            # else:
            #     fixed_res_energies = []
            #     fixed_resonances = pd.DataFrame()
            #     fixed_resonances_indices=[]

            ### Setup all model information of this level and starting chi2
            if (self.options.print_bool):
                print('*'*40)
                print(f'Current level: {current_level}, "fixed resonances": {len(fixed_resonances_indices)}')
                print(f'\t Searching for models with {current_level - 1} resonances...')
                print('*'*40)
                print()
            
            fit_code_init_level = f'init_sol_level_{current_level}'
            ladder, fixed_resonances_indices = concat_external_resonance_ladder(initial_feature_bank, fixed_resonance_ladder)
            initial_ladder_chars = self.evaluate_prior(ladder)
            
            best_model_chars = initial_ladder_chars
            base_chi2 = np.sum(initial_ladder_chars.chi2)
            base_obj = objective_func(base_chi2, initial_ladder_chars.par, self.particle_pair, fixed_resonances_indices,
                                      self.options.Wigner_informed_variable_selection, self.options.PorterThomas_informed_variable_selection)

            if (self.options.print_bool):
                print()
                print(f'\t{fit_code_init_level}')
                print(f'\tBase objective value for level: {base_obj}')
                print()

            # if we are on the level of target resonances - just stop - by default don't stop until 0
            if (current_level==target_ires):
                break

            ### test all N-1 priors 
            time_test_priors_start = time.time()
            prior_test_out = self.test_priors(current_level,
                                              fixed_resonance_ladder,
                                              initial_feature_bank,
                                              delta_obj_allowed,
                                              base_obj)
            priors_test_time = time.time() - time_test_priors_start
            any_prior_passed_test, any_model_passed_test, best_prior_model_chars, best_prior_obj, priors_passed_cnt, best_removed_resonance_prior = prior_test_out
        
            ### if any priors passed remove, 
            if (any_prior_passed_test):
                if (self.options.print_bool):
                    print()
                    print(f'Priors passed the test...{priors_passed_cnt}')
                    print(f'Best model found {best_removed_resonance_prior}:')
                    print(f'Objective value:\t{best_prior_obj}')
                    #print(f'Time for priors test: {np.round(priors_test_time,2)} sec')
                    print(f'Time for priors test: {elim_addit_funcs.format_time_2_str(priors_test_time)[1]}')
                    print()

                best_model_obj = best_prior_obj
                best_removed_resonance = best_removed_resonance_prior
                best_model_chars = best_prior_model_chars
                any_model_passed_test = True

                LevMarV0 = self.options.LevMarV0_priorpassed # if prior passed, starting step should be small

                level_derivative_evaluations = 0
            ### else test all N-1 fitted models 
            else:
                fitted_test_out = self.test_fitted_models(current_level,
                                        fixed_resonance_ladder,
                                        initial_feature_bank,
                                        base_obj,
                                        delta_obj_allowed,
                                        best_model_obj,
                                        any_model_passed_test)
                
                best_removed_resonance, best_model_chars, any_model_passed_test, level_derivative_evaluations = fitted_test_out

                LevMarV0 = self.options.start_fudge_for_deep_stage

            


            ### Do deep fitting after selecting model from prior or deep fit       
            # if we are doing deep fit on this stage
            if (any_prior_passed_test):
                deep_stage_ladder_start = best_model_chars.par
            else:
                deep_stage_ladder_start = best_model_chars.par_post

            if ((deep_stage_ladder_start.shape[0]- fixed_resonance_ladder.shape[0]) <= start_deep_fit_from):
                if (self.options.print_bool):
                    print()
                    print('Starting "deep" fitting of best initial guess by chi2...')
                    print(f'DA = {deep_fit_max_iter}/{deep_fit_step_thr}')
                    print()

                posterior_deep_SO, sol_fit_time_deep, derivative_evaluations = self.fit_YW_by_ig(ladder_df = deep_stage_ladder_start, 
                                                                                                 fixed_resonance_df= fixed_resonance_ladder,
                                                                                                 max_steps = deep_fit_max_iter,
                                                                                                 step_threshold = deep_fit_step_thr,
                                                                                                 LevMarV0 = LevMarV0)           
                level_derivative_evaluations += derivative_evaluations
            else:
                if (self.options.print_bool):
                    print()
                    print('Skipping "deep" fitting stage, utilizing best model for the next step w/o chi2...')
                    print()
                
                posterior_deep_SO = best_model_chars
                sol_fit_time_deep = 0
                
                # based on the priors we should determine if we need to use par or par_post...
                if (any_prior_passed_test):
                    posterior_deep_SO.par_post   = posterior_deep_SO.par
                    posterior_deep_SO.chi2_post  = posterior_deep_SO.chi2
                    posterior_deep_SO.chi2n_post = posterior_deep_SO.chi2n
                    posterior_deep_SO.pw_post    = posterior_deep_SO.pw

                else:
                    pass

            cur_sol_chars_deep = posterior_deep_SO
            deep_chi2_prior = np.sum(cur_sol_chars_deep.chi2)
            deep_obj_prior = objective_func(deep_chi2_prior, cur_sol_chars_deep.par, self.particle_pair, fixed_resonances_indices,
                                            self.options.Wigner_informed_variable_selection, self.options.PorterThomas_informed_variable_selection)
            deep_chi2 = np.sum(cur_sol_chars_deep.chi2_post)
            deep_obj = objective_func(deep_chi2, cur_sol_chars_deep.par_post, self.particle_pair, fixed_resonances_indices,
                                      self.options.Wigner_informed_variable_selection, self.options.PorterThomas_informed_variable_selection)
            benefit_deep_obj = deep_obj - base_obj

            selected_ladder_chars = cur_sol_chars_deep
            
            ### printout
            if (self.options.print_bool):
                print(f'\t proc_time: {elim_addit_funcs.format_time_2_str(sol_fit_time_deep)[1]}')
                print()
                print(f'\t Benefit in objective: {benefit_deep_obj}, while initial benefit for {self.options.interm_fit_max_iter} iter. was {deep_obj_prior - base_obj}')
                print('Deep fitting decision about model selection:')
                print()
                print(f'\t objective value before: {deep_obj_prior}')
                print(f'\t objective value after: {deep_obj}')
                print()

            # checking if final model passed the test
            if ((benefit_deep_obj <= delta_obj_allowed) & final_model_passed_test):
                final_model_passed_test = True
                ladder_OUT = selected_ladder_chars.par_post
            else: 
                final_model_passed_test = False

            level_time = time.time() - level_start_time
            self.output.derivative_evaluations.append(level_derivative_evaluations)

            ### printout
            if (self.options.print_bool):
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
                cur_date = datetime.fromtimestamp(time.time()).strftime("%d.%m.%Y %H:%M:%S")
                print(cur_date)
                
                print(f'Current elapsed time: \t {elim_addit_funcs.format_time_2_str(tot_el_time)[1]}')
                print(f'Time per res.: \t {time_per_res} sec')
                print()
                print(f'Estimated TEE: \t {np.round(time_per_res * (selected_ladder_chars.par_post.shape[0] - fixed_resonance_ladder.shape[0]) /3600 , 1)} h')
                print(f'\t ~ {elim_addit_funcs.format_time_2_str(time_per_res * (selected_ladder_chars.par_post.shape[0] - fixed_resonance_ladder.shape[0]))[1]}')
                print()
                print()
                print('*'*40)
                print()

            ### save model data, and continue or break while loop
            deep_obj_stopping_criteria = {}
            for Wigner_informed_stopping_criteria in (False, True):
                for PorterThomas_informed_stopping_criteria in (False, True):
                    obj_stopping_criteria = objective_func(deep_chi2_prior, cur_sol_chars_deep.par, self.particle_pair, fixed_resonances_indices,
                                                           Wigner_informed_stopping_criteria, PorterThomas_informed_stopping_criteria)
                    obj_type = 'chi2'
                    if Wigner_informed_stopping_criteria:
                        obj_type += '+Wig'
                    if PorterThomas_informed_stopping_criteria:
                        obj_type += '+PT'
                    deep_obj_stopping_criteria[obj_type] = obj_stopping_criteria
                    
            current_num_of_res_wo_sides = selected_ladder_chars.par_post.shape[0] - fixed_resonance_ladder.shape[0]
            model_history[current_num_of_res_wo_sides] = {
                'input_ladder' : ladder,
                'selected_ladder_chars': selected_ladder_chars, # solution with min chi2 on this level
                'any_model_passed_test': any_model_passed_test, # any model on this level of N-1 res..
                'final_model_passed_test': final_model_passed_test,
                'level_time': level_time,
                'total_time': time.time() - start_time,
                'objective_value_stopping_criteria': deep_obj_stopping_criteria,
            }

            ### Printout the history of chi2 - for all the elimination
            all_curr_levels = []
            all_curr_chi2 = []
            all_curr_chi2_stat = []
            for hist_level in model_history.keys():
                curr_sol_chars = model_history[hist_level]['selected_ladder_chars']
                cur_N_dat = np.sum([df.shape[0] for df in curr_sol_chars.pw])
                cur_N_par = 3 * curr_sol_chars.par_post.shape[0]
                cur_sum_chi2 = np.sum(curr_sol_chars.chi2_post)
                cur_sum_chi2_stat = cur_sum_chi2 / (cur_N_dat - cur_N_par)
                all_curr_levels.append(hist_level)
                all_curr_chi2.append(cur_sum_chi2)
                all_curr_chi2_stat.append(cur_sum_chi2_stat)
            
            if (self.options.print_bool):
                print('Chi2_s tracking...')
                for index, hist_level in enumerate(all_curr_levels):
                    print(f"\t{hist_level}\t->\t{np.round(all_curr_chi2_stat[index],4)} ({np.round(all_curr_chi2[index],2)})\t->\t{np.round(model_history[hist_level]['level_time'],1)} s")
                print()
            
            ### stopping criteria
            if (self.options.stop_at_chi2_thr): # FIXME: stop_at_obj_thr
                # if stop after the first shot when no models pass the test (speedup)
                if best_removed_resonance is not None and any_model_passed_test:
                    initial_feature_bank, fixed_resonance_ladder = separate_external_resonance_ladder(ladder_OUT, fixed_resonances_indices)
                else:
                    if self.options.print_bool:
                        print(f'Note, stopping on criteria by objective value threshold {delta_obj_allowed}')
                        print(f'Any model passed the test: {any_model_passed_test}')
                    break
            else:
                # not stopping continuing up to 1 res..
                if self.options.print_bool: print('Skipping stopping by objective threshold test, going to 1 res model')
                if(selected_ladder_chars.par_post.shape[0]==fixed_resonance_ladder.shape[0]):
                #if(ladder.shape[0]==fixed_res_df.shape[0]+1):
                    if self.options.print_bool: print('Reached 0 resonance model.. stopping')
                    break
                else:
                    initial_feature_bank, fixed_resonance_ladder = separate_external_resonance_ladder(selected_ladder_chars.par_post, fixed_resonances_indices)

    
        ### Now outside of while loop, form output object and return

        total_time = time.time() - start_time

        elim_success = (ladder.shape[0] < ladder_IN.shape[0])

        # el_out = eliminator_OUTput(
        #     ladder_OUT = ladder_OUT,
        #     ladder_IN = ladder_IN,
        #     elimination_history= model_history,
        #     success = elim_success,
        #     elim_tot_time = total_time
        #     )
        self.output.elimination_history = model_history

        return model_history


    def test_priors(self,
                    current_level, 
                    fixed_resonance_ladder,
                    initial_feature_bank,
                    delta_obj_allowed,
                    base_obj
                    ):
        
        # check if prior model pass the test even without refitting after resonance deletion (insignificant resonances)
        any_prior_passed_test = False
        any_model_passed_test = False
        best_removed_resonance_prior = None
        best_prior_obj = np.inf
        best_prior_model_chars = None
        priors_passed_cnt = 0

        for j in range(current_level):  # For every resonance in the current ladder
            
            # Skip if side resonances
            # if j in fixed_resonances_indices:
            #     if (self.options.print_bool):
            #         print('Warning!')
            #         print(f'Res. index {j} in fixed:')
            #         print()
            #         print(fixed_resonances)
            #         print()
            #     continue
            
            ### Create and evaluate a ladder with the j-th resonance removed
            # note - always keep the side-resonances
            N_minus_1_ifb, row_removed = self.remove_resonance(initial_feature_bank, j)
            ladder, fixed_resonances_indices = concat_external_resonance_ladder(N_minus_1_ifb, fixed_resonance_ladder)
            prior_chars = self.evaluate_prior(ladder) 

            prior_sum_chi2 = np.sum(prior_chars.chi2)

            prior_sum_obj = objective_func(prior_sum_chi2, prior_chars.par, self.particle_pair, fixed_resonances_indices,
                                           self.options.Wigner_informed_variable_selection, self.options.PorterThomas_informed_variable_selection)
            prior_benefit_obj = prior_sum_obj - base_obj
            prior_benefit_obj_per_Ndata = prior_benefit_obj / self.solver_eliminate.Ndata

            # reset best prior chi2 if better
            # best_removed_resonance_prior = None
            if (prior_sum_obj < best_prior_obj):
                best_prior_obj = prior_sum_obj
                best_prior_model_chars = prior_chars
                best_removed_resonance_prior = j

            ### Check if un-fitted N-1 model still is acceptable
            if ((prior_benefit_obj_per_Ndata<=delta_obj_allowed)):
                test_result = "✓"  # Check mark if the test is passed
                sign = "<="
                any_prior_passed_test = True
                priors_passed_cnt += 1
            else:
                test_result = "✗"  # Cross mark if the test is not passed
                sign = ">"

            if (self.options.print_bool):
                print()
                print(f'Prior ladder check, deleted {j}, E_λ  = {row_removed["E"].item()}')
                print(f'\tChi2: \t {np.round(prior_sum_chi2, 4)}')
                print(f'\tObjective value: \t {np.round(prior_sum_obj, 4)} \t| base: {np.round(base_obj, 4)} | current best: {np.round(best_prior_obj, 4)}')
                print(f'\t\t\t {np.round(prior_benefit_obj_per_Ndata, 4)}\t\t{sign}\t\t{delta_obj_allowed}\t\t=>\t\t{test_result}')
                print(f'\t {priors_passed_cnt} / {current_level} passed.    ({len(fixed_resonances_indices)} fixed res.)')
                print()
            
            # check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)
            if (self.options.greedy_mode and any_prior_passed_test):
                if (self.options.print_bool):
                    print()
                    print('Using "greedy" mode')
                    print('Skipping search of solutions - taking first solution that satisfies conditions.')
                    print(f'Deleted res. # {best_removed_resonance_prior}')
                    print(f'Objective value: {best_prior_obj} | {base_obj}')
                break
            # end check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)


        return (any_prior_passed_test, any_model_passed_test, best_prior_model_chars, best_prior_obj, priors_passed_cnt, best_removed_resonance_prior)




    def test_fitted_models(self,
                           current_level,
                           fixed_resonance_ladder,
                           initial_feature_bank,
                           base_obj,
                           delta_obj_allowed,
                           best_model_obj,
                           any_model_passed_test
                           ):
       
        # if no priors passed the test - do the fitting for each model
        if (self.options.print_bool):
            print('*'*40)
            print('No priors passed the test...')
            print(f'Doing limited iterations to find the best model inside current level - {current_level} - > {current_level-1}...')
        
        posteriors_passed_cnt = 0 # number of models that pass the test after fitting
        
        # TODO: change this..
        best_removed_resonance = None
        best_model_chars = None
        level_derivative_evaluations = 0

        # selecting the most perspective model from the chi2 point of view with limited iterations allowed
        for j in range(current_level):  # For every resonance in the current ladder
        
            # # Refit the data with this temporary ladder
            # # Skip if side resonances
            # if j in fixed_resonances_indices:
            #     if (self.options.print_bool):
            #         print('Warning!')
            #         print(f'Res. index {j} in fixed:')
            #         print()
            #         print(fixed_resonances)
            #         print()
            #     continue
        
            # Create a ladder with the j-th resonance removed
            # note - always keep the side-resonances
            # prior_ladder, row_removed = self.remove_resonance(initial_feature_bank, j)
            N_minus_1_ifb, row_removed = self.remove_resonance(initial_feature_bank, j)
            prior_ladder, fixed_resonances_indices = concat_external_resonance_ladder(N_minus_1_ifb, fixed_resonance_ladder)
            

            if self.options.print_bool:
                print('prior ladder passed to fit')
                print(prior_ladder)
                print()

            posterior_interm_SO, sol_fit_time_interm, derivative_evaluations = self.fit_YW_by_ig(
                ladder_df = prior_ladder,
                fixed_resonance_df= fixed_resonance_ladder,
                max_steps = self.options.interm_fit_max_iter,
                step_threshold = self.options.interm_fit_step_thr
                )
            level_derivative_evaluations += derivative_evaluations
            cur_sol_chars = posterior_interm_SO
            interm_step_chi2 = np.sum(cur_sol_chars.chi2_post)
            interm_step_obj = objective_func(interm_step_chi2, cur_sol_chars.par_post, self.particle_pair, fixed_resonances_indices,
                                             self.options.Wigner_informed_variable_selection, self.options.PorterThomas_informed_variable_selection)

            benefit_obj = interm_step_obj - base_obj

            # Check if this model is best so far
            if ((benefit_obj <= delta_obj_allowed)):
                
                # mark that at least one sol passed the chi2 test
                any_model_passed_test = True
                current_model_passed_test = True

                test_result = "✓"  # Check mark if the test is passed
                sign = "<="

                posteriors_passed_cnt +=1

                if (self.options.print_bool):
                    print()
                    print('Model passed chi2 test!')
            else:

                current_model_passed_test = False

                test_result = "✗"
                sign = ">"

            # TODO: check - so we will use always
            if (interm_step_obj < best_model_obj):
                
                best_model_obj = interm_step_obj
                best_removed_resonance = j
                best_model_chars = cur_sol_chars

            if (self.options.print_bool):
                print()
                print(f'Intermediate fitting stage (IA = {self.options.interm_fit_max_iter}/{self.options.interm_fit_step_thr}), deleted {j}, E_λ  = {row_removed["E"].item()}')
                print(f'\tChi2:\t{np.round(interm_step_chi2,4)}')
                print(f'\tObjective Value:\t{np.round(interm_step_obj,4)}\tbase: {np.round(base_obj,4)} | current best: {best_model_obj}  ')
                print(f'\t\t\t{np.round(benefit_obj,4)}\t{sign}\t{delta_obj_allowed}\t => \t {test_result}')
                #print(f'\t\t\t{sol_fit_time_interm} sec for processing')
                print(f'\t\t\tproc_time: {elim_addit_funcs.format_time_2_str(sol_fit_time_interm)[1]}')
                print()
                print(f'\t{posteriors_passed_cnt} / {current_level} passed.\t({len(fixed_resonances_indices)} fixed res.)')
                print()
            
            # check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)
            if (self.options.greedy_mode and any_model_passed_test):
                if (self.options.print_bool):
                    print()
                    print('Using "greedy" mode')
                    print('Skipping search of solutions - taking first solution that satisfies conditions.')
                    print(f'Deleted res. # {best_removed_resonance}')
                    print(f'Objective Value: {best_model_obj} | {base_obj}')
                break
            # end check if we have at least one solution that passed the test and if we have - just give it as output (without seaarch of the best - taking first)

                
        if (self.options.print_bool):
            print('*'*40)
            # if some of the models passed the test using intermediate stage
            if (any_model_passed_test):
                print(f'{posteriors_passed_cnt} models passed the test, using best for a deep fit')
                print(f'Best model is\t{best_removed_resonance}.')
                print(f'Objective Value:\t{best_model_obj}')
            else:
                print('No models passed the test!')
                print(f'Best model is {best_removed_resonance}.')

            print('End Doing limited iterations to find the best model inside current level...')

        return (best_removed_resonance, best_model_chars, any_model_passed_test, level_derivative_evaluations)





    def evaluate_prior(self, resonance_ladder):

        self.solver_eliminate.sammyRTO = self.rto
        sammy_OUT = self.solver_eliminate.fit(resonance_ladder, [])

        return sammy_OUT
    


    def fit_YW_by_ig(self, 
                     ladder_df:pd.DataFrame,
                     fixed_resonance_df, 
                     max_steps: int = 0,
                     step_threshold: float = 0.01,
                     LevMarV0 = 0.1,
                     ):
        """Wrapper to fit the data with given params using YW scheme"""

        time_start = time.time()

        # redefining rto and inputs
        self.solver_eliminate.sammyRTO = self.rto_bayes

        # self.solver.sammyINP.resonance_ladder = ladder_df
        self.solver_eliminate.sammyINP.max_steps = max_steps
        self.solver_eliminate.sammyINP.step_threshold = step_threshold
        self.solver_eliminate.sammyINP.autoelim_threshold = None
        self.solver_eliminate.sammyINP.LS = False
        self.solver_eliminate.sammyINP.LevMar = True
        self.solver_eliminate.sammyINP.initial_parameter_uncertainty = LevMarV0
        
        if ladder_df.shape[0] == fixed_resonance_df.shape[0]:
            sammy_OUT = self.evaluate_prior(ladder_df)
            sammy_OUT.pw_post = sammy_OUT.pw
            sammy_OUT.par_post = sammy_OUT.par
            sammy_OUT.chi2_post = sammy_OUT.chi2
            sammy_OUT.chi2n_post = sammy_OUT.chi2n
        else:
            sammy_OUT = self.solver_eliminate.fit(ladder_df, [])
            self.total_derivative_evaluations += sammy_OUT.total_derivative_evaluations
        # self.output.derivative_evaluations.append(sammy_OUT.total_derivative_evaluations)

        time_proc = time.time() - time_start

        return sammy_OUT, time_proc, sammy_OUT.total_derivative_evaluations


    def remove_resonance(self,
                         ladder: pd.DataFrame,
                         index_to_remove: int):
        """Removes a resonance from the ladder and returns the removed row."""

        if index_to_remove in ladder.index:
            removed_row = ladder.loc[[index_to_remove]].copy()  # Get the row before removal
            new_ladder = ladder.drop(index_to_remove)  # Drop the row to get the new ladder
            new_ladder.reset_index(drop=True, inplace=True)  # Reindex the new ladder
        else:
            if self.options.print_bool: print(ladder)
            raise ValueError(f'Invalid index {index_to_remove}')
        return new_ladder, removed_row
    
