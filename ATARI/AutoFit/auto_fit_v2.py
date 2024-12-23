from copy import deepcopy
from math import ceil
import numpy as np
from scipy.stats import norm
from typing import Optional, List, Union
from dataclasses import dataclass
import multiprocessing
import os
from scipy.linalg import block_diag

from ATARI.AutoFit.sammy_interface_bindings import Solver_factory, get_parent_solver_options
from ATARI.AutoFit.fit_and_eliminate import FitAndEliminate, FitAndEliminateOPT, FitAndEliminateOUT
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SolverOPTs, Particle_Pair, SammyOutputData, SammyInputDataYW
from ATARI.sammy_interface.sammy_misc import get_idc_at_theory
from ATARI.utils.stats import add_normalization_uncertainty_to_covariance
from ATARI.utils.file_handling import clean_and_make_directory, return_random_subdirectory
from ATARI.utils.datacontainers import Evaluation, Evaluation_Data
from ATARI.AutoFit.functions import * 
from ATARI.AutoFit.cross_validation import find_CV_scores, find_model_complexity, split_correlated_datasets_into_folds
# from ATARI.AutoFit.auto_fit import AutoFitOPT, AutoFitOUT, CrossValidationOUT
    


def find_max_num_resonances(particle_pair:Particle_Pair, window_size:float, tol:float=1e-2):
    """
    ...
    """

    # Collecting mean level spacings:
    mean_lvl_spacings = []
    for Jpi, spingroup in particle_pair.spin_groups.items():
        mean_lvl_spacing = spingroup['<D>']
        mean_lvl_spacings.append(mean_lvl_spacing)
    num_spingroups = len(mean_lvl_spacings)

    num_res_exp = np.sum(float(window_size)/np.array(mean_lvl_spacing))
    std_of_num_res = 2.0 # according to GOE, the STD of the number of resonances asymptotically approaches 2
    num_std_desired = -norm.ppf(tol)
    num_res_max = ceil(num_res_exp + std_of_num_res * num_std_desired * np.sqrt(num_spingroups))
    return num_res_max


@dataclass
class CrossValidationOUT:
    chi2_test   : Union[float,np.ndarray] = None
    obj_test    : Union[float,np.ndarray] = None
    ndata_test  : Union[int  ,np.ndarray] = None
    chi2_train  : Union[float,np.ndarray] = None
    obj_train   : Union[float,np.ndarray] = None
    ndata_train : Union[int  ,np.ndarray] = None


@dataclass
class AutoFitOUT:
    final_evaluation            : Optional[Evaluation]               = None
    final_samout                : Optional[SammyOutputData]          = None
    Nres_target                 : Optional[int]                      = None
    whitening_model             : Optional[SammyOutputData]          = None
    fit_and_eliminate_output    : Optional[FitAndEliminateOUT]       = None
    cross_validation_output     : Optional[List[CrossValidationOUT]] = None
    total_time                  : Optional[float]                    = None



@dataclass
class AutoFitOPT:
    # save options
    save_elimination_history        : bool  = False
    save_CV_elimination_history     : bool  = False

    # parallel options
    parallel_CV                     : bool  = False
    parallel_processes              : int   = 5

    # other
    K_folds                         : int   = 5
    print_bool                      : bool  = True
    use_1std_rule                   : bool  = True
    use_MAD                         : bool  = False
    final_fit_to_0_res              : bool  = False

    # resonance statistics
    Wigner_informed_cross_validation        : bool = False
    PorterThomas_informed_cross_validation  : bool = False



class AutoFit2:

    def __init__(self,
                 sammyRTO                   : SammyRunTimeOptions,
                 particle_pair              : Particle_Pair,

                 solver_options_initial     : SolverOPTs,
                 solver_options_eliminate   : SolverOPTs,
                
                 AutoFit_options            : Optional[AutoFitOPT]          = None,
                 fit_and_elim_options       : Optional[FitAndEliminateOPT]  = None,

                 cap_norm_unc               : float = 0.0384200
                ):

        """
        Initialize the class with all parameters specified
        """

        # if cardinality_options is not None:
        #     self.cardinality_options = cardinality_options
        # else:
        #     pass
        
        if solver_options_initial is None: raise ValueError("Default not implemented")
        else:                              self.solver_options_initial = solver_options_initial

        if solver_options_eliminate is None: raise ValueError("Default not implemented")
        else:                                self.solver_options_eliminate = solver_options_eliminate

        if AutoFit_options is None: self.options = AutoFitOPT() 
        else:                       self.options = deepcopy(AutoFit_options)

        if fit_and_elim_options is None: self.fit_and_elim_options = FitAndEliminateOPT()
        else:                            self.fit_and_elim_options = deepcopy(fit_and_elim_options)
        
        self.particle_pair = particle_pair

        self.sammyRTO = sammyRTO

        self.cap_norm_unc = cap_norm_unc

        if not self.options.parallel_CV:
            self.options.parallel_processes = 1

        self.output = AutoFitOUT()


    def fit(self, evaluation_data:Evaluation_Data, total_resonance_ladder:pd.DataFrame, fixed_resonance_indices:list=[]):
        resonance_ladder, fixed_resonance_ladder = separate_external_resonance_ladder(total_resonance_ladder, fixed_resonance_indices)
        ### if resonance ladder is all fixed
        if len(total_resonance_ladder) == len(fixed_resonance_indices):
            assert np.all(total_resonance_ladder.index == fixed_resonance_indices)
            if self.options.print_bool:
                print(f"=============\nResonance ladder is all fixed\n=============")
            if self.options.save_elimination_history:
                self.output.elimination_history = None
            rto_prior = deepcopy(self.sammyRTO)
            rto_prior.bayes = False
            solve_prior = Solver_factory(rto_prior, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data)
            sammyOUT = solve_prior.fit(total_resonance_ladder)
            sammyOUT.pw_post = sammyOUT.pw; sammyOUT.par_post = sammyOUT.par; sammyOUT.chi2_post = sammyOUT.chi2; sammyOUT.chi2n_post = sammyOUT.chi2n
            self.output.final_samout = sammyOUT
            return self.output
        
        if evaluation_data.experimental_models_no_pup is None:
            raise ValueError('"evaluation_data" must have experimental models without propagated uncertainty parameters ("evaluation_data.experimental_models_no_pup").')
        
        ### Find the maximum expected number of resonances
        window_size = max([np.max(data['E'].to_numpy()) for data in evaluation_data.datasets]) \
                    - min([np.min(data['E'].to_numpy()) for data in evaluation_data.datasets])
        Nres_internal_upper_limit = find_max_num_resonances(self.particle_pair, window_size, tol=1e-2)
        if (self.fit_and_elim_options.width_elimination) and (self.fit_and_elim_options.width_elimination_Nres_threshold is None):
            self.fit_and_elim_options.width_elimination_Nres_threshold = Nres_internal_upper_limit
        
        ### Final solve to target Nres with all data 
        rto_fit = deepcopy(self.sammyRTO)
        rto_fit.bayes = True
        solver_initial = Solver_factory(rto_fit, self.solver_options_initial._solver,   self.solver_options_initial,   self.particle_pair, evaluation_data) 
        solver_elim    = Solver_factory(rto_fit, self.solver_options_eliminate._solver, self.solver_options_eliminate, self.particle_pair, evaluation_data)

        if self.options.print_bool:
            print(f"=============\nRunning Full Elimination\n=============")
            
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=self.fit_and_elim_options, particle_pair=self.particle_pair)
        initial_samout = fe.initial_fit(resonance_ladder, fixed_resonance_ladder=fixed_resonance_ladder)
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(initial_samout.par_post, fe.output.external_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, target_ires=Nres_internal_upper_limit, fixed_resonance_ladder=fixed_resonances) # FIXME: REPLACE "Nres_internal_upper_limit" WITH 0 AFTER DEBUGGING!

        if self.options.save_elimination_history:
            self.output.fit_and_eliminate_output = fe.output

        ### Run CV
        if self.options.print_bool:
            print("=============\nRunning Cross Validation\n=============")
        overfit_samout = elimination_history[Nres_internal_upper_limit]['selected_ladder_chars']
        self.output.whitening_model = overfit_samout
        folds_data = self.cross_validation(evaluation_data, sammy_out_theo=overfit_samout, Nres_internal_upper_limit=Nres_internal_upper_limit, total_resonance_ladder=total_resonance_ladder, fixed_resonance_indices=fixed_resonance_indices)
        CV_test_scores, CV_train_scores = find_CV_scores(folds_data, use_MAD=self.options.use_MAD)
        
        ### Get cardinality from CV results
        Nres_selected = find_model_complexity(CV_test_scores, use_1std_rule=self.options.use_1std_rule)

        # ### Gathering final results
        # if self.options.final_fit_to_0_res: Nres_target = 0
        # else: Nres_target = Nres_selected

        self.output.final_samout = elimination_history[Nres_selected]['selected_ladder_chars']
        self.output.final_evaluation = Evaluation.from_samout('fit', self.output.final_samout, post=True, external_resonance_indices=fe.output.external_resonance_indices)
        self.output.total_time = fe.total_derivative_evaluations

        return self.output


    def cross_validation(self, evaluation_data:Evaluation_Data, sammy_out_theo:SammyOutputData, Nres_internal_upper_limit:int, total_resonance_ladder:pd.DataFrame, fixed_resonance_indices:list=[]):

        ### Updating covariance data
        # FIXME: ALSO CONSIDER CASE FOR CAPTURE USING add_normalization_uncertainty_to_covariance
        solver_options_get_idc_at_theory = deepcopy(self.solver_options_eliminate)
        solver_options_get_idc_at_theory.idc_at_theory = True
        solver_get_idc_at_theory = Solver_factory(rto=self.sammyRTO,
                                                  solver=solver_options_get_idc_at_theory._solver, solver_options=solver_options_get_idc_at_theory,
                                                  particle_pair=self.particle_pair, evaluation_data=evaluation_data)
        covariance_data_at_theory = solver_get_idc_at_theory.get_idc_at_theory(sammy_out_theo.par_post)
        evaluation_data_CV = deepcopy(evaluation_data)
        evaluation_data_CV.covariance_data = covariance_data_at_theory

        ### Splitting data for cross validation
        Ds, Vis_train_folds, Vis_test_folds, folds_weights = split_correlated_datasets_into_folds(evaluation_data=evaluation_data_CV, cap_norm_unc=self.cap_norm_unc, K_folds=self.options.K_folds)
        if self.options.print_bool:
            print("Fold weights:")
            for ifold, fold_weights in enumerate(folds_weights):
                print(f'\tFold #{ifold}:\t{fold_weights}')

        ### Creating evaluation data for cross-validation solve:
        evaluation_data_CV_solve = deepcopy(evaluation_data)
        # evaluation_data_CV_solve = Evaluation_Data(evaluation_data.experimental_titles, experimental_models=evaluation_data.experimental_models,
        #                                            datasets=None, covariance_data=None, measurement_models=None, # the data should be pulled from the solver
        #                                            experimental_models_no_pup=evaluation_data.experimental_models_no_pup)

        ### Could have option to save folds here

        ### get CVE score in parallel
        if self.options.parallel_CV:
            ## Setup
            clean_and_make_directory(self.sammyRTO.sammy_runDIR) # multiple runDIR's will be created inside of this runDIR
            if self.options.parallel_processes > self.options.K_folds:
                print(f"User specified more CPUs than folds ({self.options.K_folds}), setting CPUs = {self.options.K_folds}")
                self.options.parallel_processes = self.options.K_folds
            ## Run
            multi_input = [(Ds, Vis_train_fold, Vis_test_fold, evaluation_data_CV_solve, total_resonance_ladder, fixed_resonance_indices) for Vis_train_fold, Vis_test_fold in zip(Vis_train_folds, Vis_test_folds)]
            with multiprocessing.Pool(processes=self.options.parallel_processes) as pool:
                folds_results = pool.map(self.get_cross_validation_score, multi_input)
            assert len(folds_results) == self.options.K_folds

        ### get CVE score in serial
        else:
            folds_results = []
            for Vis_train_fold, Vis_test_fold in zip(Vis_train_folds, Vis_test_folds):
                fold_results = self.get_cross_validation_score((Ds, Vis_train_fold, Vis_test_fold, evaluation_data_CV_solve, total_resonance_ladder, fixed_resonance_indices))
                folds_results.append(fold_results)

        # if save:
        self.output.cross_validation_output = folds_results
        # save_test_scores, save_train_scores, save_ires, save_Ntest, save_Ntrain = np.array(save_test_scores), np.array(save_train_scores), np.array(save_ires), np.array(save_Ntest), np.array(save_Ntrain)
        # self.output.cross_validation_output = CrossValidationOUT(ires=save_ires, test_scores=save_test_scores, train_scores=save_train_scores, Ntest=save_Ntest, Ntrain=save_Ntrain)

        # Finding the number of resonances cases run that are common to all models:
        Nres_start = len(total_resonance_ladder)
        Nres_all = []
        for Nres in range(Nres_start):
            for fold_results in folds_results:
                if Nres not in fold_results.keys():
                    break
            else:
                Nres_all.append(Nres)

        # Reformatting the data:
        folds_data = {}
        for Nres in range(Nres_internal_upper_limit):
            chi2_test_values   = []
            obj_test_values    = []
            ndata_test_values  = []
            chi2_train_values  = []
            obj_train_values   = []
            ndata_train_values = []
            for fold in range(self.options.K_folds):
                chi2_test_values.append(folds_results[fold][Nres].chi2_test)
                obj_test_values.append(folds_results[fold][Nres].obj_test)
                ndata_test_values.append(folds_results[fold][Nres].ndata_test)
                chi2_train_values.append(folds_results[fold][Nres].chi2_train)
                obj_train_values.append(folds_results[fold][Nres].obj_train)
                ndata_train_values.append(folds_results[fold][Nres].ndata_train)
                folds_data[Nres] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)

        return folds_data
    

    def get_cross_validation_score(self, input_arguments):
        
        Ds, Vis_train, Vis_test, evaluation_data, total_resonance_ladder, fixed_resonance_indices = input_arguments
        resonance_ladder, fixed_resonance_ladder = separate_external_resonance_ladder(total_resonance_ladder, fixed_resonance_indices)

        # D = np.concatenate(Ds)
        # for Vi_train, Vi_test in zip(Vis_train, Vis_test):
        #     print('\t', Vi_train.shape, Vi_test.shape)
        # Vi_train = block_diag(*Vis_train)
        # Vi_test  = block_diag(*Vis_test)
        

        # Set RTO options:
        rto_train = deepcopy(self.sammyRTO);    rto_test = deepcopy(self.sammyRTO)
        fit_and_elim_options = deepcopy(self.fit_and_elim_options)
        if self.options.parallel_CV: # set RTO options if in parallel
            rto_train.sammy_runDIR = return_random_subdirectory(self.sammyRTO.sammy_runDIR)
            rto_test.sammy_runDIR  = return_random_subdirectory(self.sammyRTO.sammy_runDIR)
            rto_train.keep_runDIR = False;  rto_test.keep_runDIR = False
            rto_train.Print       = False;  rto_test.Print       = False
            rto_train.bayes       = True;   rto_test.bayes       = False
            fit_and_elim_options.print_bool = False

        # create solvers
        solver_options_CV_initial = get_parent_solver_options(self.solver_options_initial)
        solver_options_CV_initial.idc_at_theory = False
        solver_options_CV_eliminate = get_parent_solver_options(self.solver_options_eliminate)
        solver_options_CV_eliminate.idc_at_theory = False
        
        # print('V INVERSE SHAPES:')
        # print(Vi_train.shape, Vi_test.shape)
        solver_initial = Solver_factory(rto_train, 'EXT', solver_options_CV_initial,   self.particle_pair, evaluation_data,
                                        cap_norm_unc=self.cap_norm_unc, remove_V=False, V_is_inv=True, Vinv=Vis_train, D=Ds) 
        solver_elim    = Solver_factory(rto_train, 'EXT', solver_options_CV_eliminate, self.particle_pair, evaluation_data,
                                        cap_norm_unc=self.cap_norm_unc, remove_V=False, V_is_inv=True, Vinv=Vis_train, D=Ds)
        solver_test    = Solver_factory(rto_test,  'EXT', solver_options_CV_eliminate, self.particle_pair, evaluation_data, 
                                        cap_norm_unc=self.cap_norm_unc, remove_V=False, V_is_inv=True, Vinv=Vis_test,  D=Ds)

        # fit and eliminate
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=fit_and_elim_options, particle_pair=self.particle_pair)
        initial_samout = fe.initial_fit(resonance_ladder,fixed_resonance_ladder=fixed_resonance_ladder)
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(initial_samout.par_post, fe.output.external_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, fixed_resonance_ladder=fixed_resonances)#, target_ires=len(fixed_resonances))

        # get test and train scores
        fold_results = {}
        for ires, elim_hist_case in elimination_history.items():
            res_ladder = elim_hist_case['selected_ladder_chars'].par_post
            test_out = solver_test.fit(res_ladder)
            Ndata_test = len(test_out.pw[0])
            Ndata_train = np.sum([len(each) for each in elim_hist_case['selected_ladder_chars'].pw_post])
            # Train
            chi2_train = np.sum(elim_hist_case['selected_ladder_chars'].chi2_post)
            obj_train = objective_func(chi2_train, res_ladder, self.particle_pair, None, Wigner_informed=self.options.Wigner_informed_cross_validation, PorterThomas_informed=self.options.PorterThomas_informed_cross_validation)
            # Test
            chi2_test = np.sum(test_out.chi2)
            obj_test = objective_func(chi2_test, res_ladder, self.particle_pair, None, Wigner_informed=self.options.Wigner_informed_cross_validation, PorterThomas_informed=self.options.PorterThomas_informed_cross_validation)

            # fold_results[key] = {'obj_test'    : obj_test,
            #                      'ndata_test'  : Ndata_test,
            #                      'obj_train'   : obj_train,
            #                      'ndata_train' : Ndata_train}

            fold_results[ires] = CrossValidationOUT(chi2_test=chi2_test,   obj_test=obj_test,   ndata_test=Ndata_test,
                                                    chi2_train=chi2_train, obj_train=obj_train, ndata_train=Ndata_train)

        return fold_results