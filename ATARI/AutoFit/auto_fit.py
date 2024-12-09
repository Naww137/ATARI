from copy import copy
from ATARI.AutoFit.sammy_interface_bindings import Solver_factory
from ATARI.AutoFit.fit_and_eliminate import FitAndEliminate, FitAndEliminateOPT, FitAndEliminateOUT
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SolverOPTs, Particle_Pair, SammyOutputData
import numpy as np
from typing import Optional, List, Union
from dataclasses import dataclass
import multiprocessing
import os
from ATARI.utils.file_handling import clean_and_make_directory, return_random_subdirectory
from ATARI.utils.datacontainers import Evaluation
from ATARI.AutoFit.functions import * 
from ATARI.AutoFit.cross_validation import find_CV_scores, find_model_complexity

# @dataclass
# class CrossValidationOUT:
#     ires                : Optional[np.ndarray]  = None
#     train_scores        : Optional[np.ndarray]  = None
#     test_scores         : Optional[np.ndarray]  = None
#     Ntest               : Optional[np.ndarray]  = None
#     Ntrain              : Optional[np.ndarray]  = None

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
    print_bool                      : bool  = True
    use_1std_rule                   : bool  = True
    use_MAD                         : bool  = False
    final_fit_to_0_res              : bool  = False

    ### Resonance Statistics
    Wigner_informed_cross_validation        : bool = False
    PorterThomas_informed_cross_validation  : bool = False
    




class AutoFit:

    def __init__(self,
                 sammyRTO                   : SammyRunTimeOptions,
                 particle_pair              : Particle_Pair,

                 solver_options_initial     : SolverOPTs,
                 solver_options_eliminate   : SolverOPTs,
                
                 AutoFit_options            : Optional[AutoFitOPT]          = None,
                 fit_and_elim_options       : Optional[FitAndEliminateOPT]  = None,
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
        else:                       self.options = copy(AutoFit_options)

        if fit_and_elim_options is None: self.fit_and_elim_options = FitAndEliminateOPT()
        else:                            self.fit_and_elim_options = copy(fit_and_elim_options)
        
        self.particle_pair = particle_pair

        self.rto_test = copy(sammyRTO)
        self.rto_test.bayes = False
        self.rto_train = copy(sammyRTO)
        self.rto_train.bayes = True

        if not self.options.parallel_CV:
            self.options.parallel_processes = 1

        self.output = AutoFitOUT()


    def fit(self, evaluation_data, total_resonance_ladder, fixed_resonance_indices=[]):
        resonance_ladder, fixed_resonance_ladder = separate_external_resonance_ladder(total_resonance_ladder, fixed_resonance_indices)
        ### if resonance ladder is all fixed
        if len(total_resonance_ladder) == len(fixed_resonance_indices):
            assert np.all(total_resonance_ladder.index == fixed_resonance_indices)
            if self.options.print_bool:
                print(f"=============\nResonance ladder is all fixed\n=============")
            if self.options.save_elimination_history:
                self.output.elimination_history = None
            solve_prior = Solver_factory(self.rto_test, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data)
            sammyOUT = solve_prior.fit(total_resonance_ladder)
            sammyOUT.pw_post = sammyOUT.pw; sammyOUT.par_post = sammyOUT.par; sammyOUT.chi2_post = sammyOUT.chi2; sammyOUT.chi2n_post = sammyOUT.chi2n
            self.output.final_samout = sammyOUT
            return self.output

        ### Run CV
        if self.options.print_bool:
            print("=============\nRunning Cross Validation\n=============")
        folds_data, kfolds = self.cross_validation(evaluation_data, total_resonance_ladder, fixed_resonance_indices=fixed_resonance_indices)
        CV_test_scores, CV_train_scores = find_CV_scores(folds_data, use_MAD=self.options.use_MAD)
        
        ### Get cardinality from CV results
        Nres_selected = find_model_complexity(CV_test_scores, use_1std_rule=self.options.use_1std_rule)

        if self.options.final_fit_to_0_res: Nres_target = 0
        else: Nres_target = Nres_selected

        ### Final solve to target Nres with all data 
        solver_initial = Solver_factory(self.rto_train, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data) 
        solver_elim = Solver_factory(self.rto_train, self.solver_options_eliminate._solver, self.solver_options_eliminate, self.particle_pair, evaluation_data)

        if self.options.print_bool:
            print(f"=============\nFitting to {Nres_target} Resonances\n=============")
            
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=self.fit_and_elim_options, particle_pair=self.particle_pair)
        initial_samout = fe.initial_fit(resonance_ladder, fixed_resonance_ladder=fixed_resonance_ladder)
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(initial_samout.par_post, fe.output.external_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, target_ires=Nres_target, fixed_resonance_ladder=fixed_resonances)

        if self.options.save_elimination_history:
            self.output.fit_and_eliminate_output = fe.output
        self.output.final_samout = elimination_history[Nres_selected]['selected_ladder_chars']
        self.output.final_evaluation = Evaluation.from_samout('fit', self.output.final_samout, post=True, external_resonance_indices=fe.output.external_resonance_indices)
        self.output.total_time = fe.total_derivative_evaluations

        return self.output


    def cross_validation(self, evaluation_data, total_resonance_ladder, fixed_resonance_indices=[]):

        ### Split CV data
        if True: #measurement_wise
            kfolds = len(evaluation_data.experimental_models)
            list_evaluation_data_train, list_evaluation_data_test = [], []
            for k in range(kfolds):
                evaluation_data_train, evaluation_data_test = evaluation_data.get_train_test_over_datasets(k)
                list_evaluation_data_train.append(evaluation_data_train); list_evaluation_data_test.append(evaluation_data_test)
        else:
            pass # perform SVD and stuff


        ### Could have option to save folds here

        ### get CVE score in parallel
        if self.options.parallel_CV:
            ## Setup
            clean_and_make_directory(self.rto_train.sammy_runDIR) # multiple runDIR's will be created inside of this runDIR
            if self.options.parallel_processes > kfolds:
                print(f"User specified more CPUs than folds ({kfolds}), setting CPUs = {kfolds}")
                self.options.parallel_processes = kfolds
            ## Run
            multi_input = [(train, test, total_resonance_ladder, fixed_resonance_indices) for train, test in zip(list_evaluation_data_train, list_evaluation_data_test)]
            with multiprocessing.Pool(processes=self.options.parallel_processes) as pool:
                folds_results = pool.map(self.get_cross_validation_score, multi_input)
            assert len(folds_results) == kfolds

        ### get CVE score in serial
        else:
            folds_results = []
            for train, test in zip(list_evaluation_data_train, list_evaluation_data_test):
                fold_results = self.get_cross_validation_score((train, test, total_resonance_ladder, fixed_resonance_indices))
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
        for Nres in Nres_all:
            chi2_test_values   = []
            obj_test_values    = []
            ndata_test_values  = []
            chi2_train_values  = []
            obj_train_values   = []
            ndata_train_values = []
            for fold in range(kfolds):
                chi2_test_values.append(folds_results[fold][Nres].chi2_test)
                obj_test_values.append(folds_results[fold][Nres].obj_test)
                ndata_test_values.append(folds_results[fold][Nres].ndata_test)
                chi2_train_values.append(folds_results[fold][Nres].chi2_train)
                obj_train_values.append(folds_results[fold][Nres].obj_train)
                ndata_train_values.append(folds_results[fold][Nres].ndata_train)
                folds_data[Nres] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)

        return folds_data, kfolds
    

    def get_cross_validation_score(self, input_arguments):
        evaluation_data_train, evaluation_data_test, total_resonance_ladder, fixed_resonance_indices = input_arguments
        resonance_ladder, fixed_resonance_ladder = separate_external_resonance_ladder(total_resonance_ladder, fixed_resonance_indices)
        
        # set RTO options if in parallel
        if self.options.parallel_CV:
            rto_train=copy(self.rto_train); rto_test=copy(self.rto_test)
            rto_train.Print=False; rto_test.Print=False; fit_and_elim_options = copy(self.fit_and_elim_options); fit_and_elim_options.print_bool=False
            rto_train.keep_runDIR=False; rto_test.keep_runDIR=False
            rto_train.sammy_runDIR = return_random_subdirectory(self.rto_train.sammy_runDIR); rto_test.sammy_runDIR = return_random_subdirectory(self.rto_train.sammy_runDIR)
        else:
            rto_train=self.rto_train; rto_test=self.rto_test; fit_and_elim_options = self.fit_and_elim_options

        # create solvers
        solver_initial = Solver_factory(rto_train, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data_train) 
        solver_elim = Solver_factory(rto_train, self.solver_options_eliminate._solver, self.solver_options_eliminate, self.particle_pair, evaluation_data_train)
        solver_test = Solver_factory(rto_test, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data_test)

        # fit and eliminate
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=fit_and_elim_options, particle_pair=self.particle_pair)
        initial_samout = fe.initial_fit(resonance_ladder,fixed_resonance_ladder=fixed_resonance_ladder)
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(initial_samout.par_post, fe.output.external_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, fixed_resonance_ladder=fixed_resonances)#, target_ires=len(fixed_resonances))

        # get test and train scores
        fold_results = {}
        for key, val in elimination_history.items():
            res_ladder = val['selected_ladder_chars'].par_post
            test_out = solver_test.fit(res_ladder)
            Ndata_test = len(test_out.pw[0])
            Ndata_train = np.sum([len(each) for each in val['selected_ladder_chars'].pw_post])
            # Train
            chi2_train = np.sum(val['selected_ladder_chars'].chi2_post)#/Ntrain*(Ntrain+Ntest)
            obj_train = objective_func(chi2_train, res_ladder, self.particle_pair, None, Wigner_informed=self.options.Wigner_informed_cross_validation, PorterThomas_informed=self.options.PorterThomas_informed_cross_validation)
            # Test
            chi2_test = np.sum(test_out.chi2)#/Ntest*(Ntrain+Ntest)
            obj_test = objective_func(chi2_test, res_ladder, self.particle_pair, None, Wigner_informed=self.options.Wigner_informed_cross_validation, PorterThomas_informed=self.options.PorterThomas_informed_cross_validation)

            # fold_results[key] = {'obj_test'    : obj_test,
            #                      'ndata_test'  : Ndata_test,
            #                      'obj_train'   : obj_train,
            #                      'ndata_train' : Ndata_train}

            fold_results[key] = CrossValidationOUT(chi2_test=chi2_test,   obj_test=obj_test,   ndata_test=Ndata_test,
                                                   chi2_train=chi2_train, obj_train=obj_train, ndata_train=Ndata_train)

        return fold_results
