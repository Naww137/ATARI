from copy import copy
from ATARI.AutoFit.sammy_interface_bindings import Solver_factory
from ATARI.AutoFit.fit_and_eliminate import FitAndEliminate, FitAndEliminateOPT, FitAndEliminateOUT
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SolverOPTs, Particle_Pair, SammyOutputData
import numpy as np
from typing import Optional, List, Union
from dataclasses import dataclass
import multiprocessing
from scipy.stats import norm
from math import ceil

from ATARI.utils.file_handling import clean_and_make_directory, return_random_subdirectory
from ATARI.utils.datacontainers import Evaluation
from ATARI.AutoFit.functions import * 
from ATARI.AutoFit.cross_validation_v3 import split_train_data, evaluate_chi2_test, find_CV_scores, find_model_complexity

def find_max_num_resonances(particle_pair:Particle_Pair, window_size:tuple, tol:float=1e-2):
    """
    ...
    """

    # Collecting mean level spacings:
    mean_lvl_spacings = []
    for Jpi, spingroup in particle_pair.spin_groups.items():
        mean_lvl_spacing = spingroup['<D>']
        mean_lvl_spacings.append(mean_lvl_spacing)
    num_spingroups = len(mean_lvl_spacings)

    num_res_exp = np.sum((window_size[1] - window_size[0])/np.array(mean_lvl_spacing))
    std_of_num_res = 2.0 # according to GOE, the STD of the number of resonances asymptotically approaches 2
    num_std_desired = -norm.ppf(tol)
    num_res_max = ceil(num_res_exp + std_of_num_res * num_std_desired * np.sqrt(num_spingroups))
    return num_res_max


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

    # Resonance Statistics
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
        
        Nres_max_num_res = find_max_num_resonances(self.particle_pair, window_size=self.fit_and_elim_options.E_window_spin)

        ### Run CV
        if self.options.print_bool:
            print("=============\nRunning Cross Validation\n=============")
        folds_data, kfolds = self.cross_validation(evaluation_data, total_resonance_ladder, fixed_resonance_indices=fixed_resonance_indices, Nres_max_num_res=Nres_max_num_res)
        CV_test_scores, CV_train_scores = find_CV_scores(folds_data, use_MAD=self.options.use_MAD)
        
        ### Get cardinality from CV results
        Nres_selected = find_model_complexity(CV_test_scores, use_1std_rule=self.options.use_1std_rule)

        if self.options.final_fit_to_0_res: Nres_target = 0
        else:                               Nres_target = Nres_selected

        ### Final solve to target Nres with all data 
        solver_options_pre_shuffle = copy(self.solver_options_eliminate)
        if solver_options_pre_shuffle._solver == 'EXT':
            solver_options_pre_shuffle.Porter_Thomas_fitting = False
            solver_options_pre_shuffle.Wigner_fitting        = False
        solver_options_post_shuffle = copy(self.solver_options_eliminate)
        solver_initial      = Solver_factory(self.rto_train, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data) 
        solver_pre_shuffle  = Solver_factory(self.rto_train, solver_options_pre_shuffle._solver , solver_options_pre_shuffle , self.particle_pair, evaluation_data)
        solver_post_shuffle = Solver_factory(self.rto_train, solver_options_post_shuffle._solver, solver_options_post_shuffle, self.particle_pair, evaluation_data)

        if self.options.print_bool:
            print(f"=============\nFitting to {Nres_max_num_res} Resonances Without Spin Shuffling\n=============")
            
        fit_eliminate_options_pre_shuffle = copy(self.fit_and_elim_options)
        fit_eliminate_options_pre_shuffle.spin_shuffle = False
        fit_eliminate_options_pre_shuffle.width_elimination_Nres_threshold = Nres_max_num_res
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_pre_shuffle, options=fit_eliminate_options_pre_shuffle, particle_pair=self.particle_pair)
        initial_samout = fe.initial_fit(resonance_ladder, fixed_resonance_ladder=fixed_resonance_ladder)
        fixed_resonance_indices = fe.output.external_resonance_indices
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(initial_samout.par_post, fixed_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, target_ires=Nres_max_num_res, fixed_resonance_ladder=fixed_resonances)

        if self.options.print_bool:
            print(f"=============\nFitting to {Nres_target} Resonances With Spin Shuffling\n=============")
            
        fit_eliminate_options_with_shuffle = copy(self.fit_and_elim_options)
        fit_eliminate_options_with_shuffle.spin_shuffle = True
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_post_shuffle, options=fit_eliminate_options_with_shuffle, particle_pair=self.particle_pair)
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(elimination_history[Nres_max_num_res]['selected_ladder_chars'].par_post, fixed_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, target_ires=Nres_target, fixed_resonance_ladder=fixed_resonances)

        self.output.Nres_target = Nres_selected # The number of resonances in the model
        if self.options.save_elimination_history:
            self.output.fit_and_eliminate_output = fe.output
        self.output.final_samout     = elimination_history[Nres_selected]['selected_ladder_chars']
        self.output.final_evaluation = Evaluation.from_samout('fit', self.output.final_samout, post=True, external_resonance_indices=fixed_resonance_indices)
        self.output.total_time       = fe.total_derivative_evaluations

        return self.output


    def cross_validation(self, evaluation_data, total_resonance_ladder, fixed_resonance_indices=[], Nres_max_num_res:int=0, kfolds:int=5):

        ### Split CV data
        list_evaluation_data_train, test_indices, train_indices = split_train_data(evaluation_data, K_folds=kfolds, rng=None, seed=None)
        list_evaluation_data_test = [evaluation_data]*kfolds
        # if True: #measurement_wise
        #     kfolds = len(evaluation_data.experimental_models)
        #     list_evaluation_data_train, list_evaluation_data_test = [], []
        #     for k in range(kfolds):
        #         evaluation_data_train, evaluation_data_test = evaluation_data.get_train_test_over_datasets(k)
        #         list_evaluation_data_train.append(evaluation_data_train); list_evaluation_data_test.append(evaluation_data_test)
        # else:
        #     pass # perform SVD and stuff


        ### Could have option to save folds here

        #################
        ### get CVE score in parallel
        if self.options.parallel_CV:
            ## Setup
            clean_and_make_directory(self.rto_train.sammy_runDIR) # multiple runDIR's will be created inside of this runDIR
            if self.options.parallel_processes > kfolds:
                print(f"User specified more CPUs than folds ({kfolds}), setting CPUs = {kfolds}")
                self.options.parallel_processes = kfolds
            ## Run
            multi_input = [(train, test, total_resonance_ladder, fixed_resonance_indices, Nres_max_num_res, ifold, test_indices[ifold], train_indices[ifold]) for ifold, (train, test) in enumerate(zip(list_evaluation_data_train, list_evaluation_data_test))]
            with multiprocessing.Pool(processes=self.options.parallel_processes) as pool:
                folds_results = pool.map(self.get_cross_validation_score, multi_input)
            assert len(folds_results) == kfolds

        ### get CVE score in serial
        else:
            folds_results = []
            for ifold, train, test in enumerate(zip(list_evaluation_data_train, list_evaluation_data_test)):
                fold_results = self.get_cross_validation_score((train, test, total_resonance_ladder, fixed_resonance_indices, Nres_max_num_res, ifold, test_indices[ifold], train_indices[ifold]))
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
        evaluation_data_train, evaluation_data_test, total_resonance_ladder, fixed_resonance_indices, Nres_max_num_res, ifold, test_indices, train_indices = input_arguments
        resonance_ladder, fixed_resonance_ladder = separate_external_resonance_ladder(total_resonance_ladder, fixed_resonance_indices)
        
        # set RTO options if in parallel
        rto_train = copy(self.rto_train)
        rto_test  = copy(self.rto_test)
        if self.options.parallel_CV:
            rto_train.Print=False; rto_test.Print=False
            rto_train.keep_runDIR=False; rto_test.keep_runDIR=False
            fit_and_elim_options = copy(self.fit_and_elim_options)
            fit_and_elim_options.print_bool = False
        else:
            fit_and_elim_options = copy(self.fit_and_elim_options)
        rto_train.sammy_runDIR = f'{rto_train.sammy_runDIR}_CV_fold_{ifold}'
        rto_test .sammy_runDIR = f'{rto_test .sammy_runDIR}_CV_fold_{ifold}'

        # create solvers
        solver_options_pre_shuffle = copy(self.solver_options_eliminate)
        if solver_options_pre_shuffle._solver == 'EXT':
            solver_options_pre_shuffle.Porter_Thomas_fitting = False
            solver_options_pre_shuffle.Wigner_fitting        = False
        solver_options_post_shuffle = copy(self.solver_options_eliminate)
        solver_initial      = Solver_factory(rto_train, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data_train) 
        solver_pre_shuffle  = Solver_factory(rto_train, solver_options_pre_shuffle ._solver, solver_options_pre_shuffle , self.particle_pair, evaluation_data_train)
        solver_post_shuffle = Solver_factory(rto_train, solver_options_post_shuffle._solver, solver_options_post_shuffle, self.particle_pair, evaluation_data_train)
        solver_test         = Solver_factory(rto_test , self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data_test )

        # Optimization of the resonance ladder prior to spin-shuffling:
        fit_eliminate_options_pre_shuffle = copy(self.fit_and_elim_options)
        fit_eliminate_options_pre_shuffle.spin_shuffle = False
        fit_eliminate_options_pre_shuffle.width_elimination_Nres_threshold = Nres_max_num_res
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_pre_shuffle, options=fit_eliminate_options_pre_shuffle, particle_pair=self.particle_pair)
        initial_samout = fe.initial_fit(resonance_ladder, fixed_resonance_ladder=fixed_resonance_ladder)
        fixed_resonance_indices = fe.output.external_resonance_indices
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(initial_samout.par_post, fixed_resonance_indices)
        elimination_history_pre_shuffle = fe.eliminate(internal_resonance_ladder, target_ires=Nres_max_num_res, fixed_resonance_ladder=fixed_resonances)

        # Optimization of the resonance ladder after spin-shuffling:
        fit_eliminate_options_with_shuffle = copy(self.fit_and_elim_options)
        fit_eliminate_options_with_shuffle.spin_shuffle = True
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_post_shuffle, options=fit_eliminate_options_with_shuffle, particle_pair=self.particle_pair)
        internal_resonance_ladder, fixed_resonances = separate_external_resonance_ladder(elimination_history_pre_shuffle[Nres_max_num_res]['selected_ladder_chars'].par_post, fixed_resonance_indices)
        elimination_history = fe.eliminate(internal_resonance_ladder, target_ires=0, fixed_resonance_ladder=fixed_resonances)

        # get test and train scores
        fold_results = {}
        for key, val in elimination_history.items():
            res_ladder = val['selected_ladder_chars'].par_post
            Ndata_test  = np.sum([len(test_idx ) for test_idx  in test_indices ])
            Ndata_train = np.sum([len(train_idx) for train_idx in train_indices])
            # Train
            chi2_train = np.sum(val['selected_ladder_chars'].chi2_post)
            obj_train = objective_func(chi2_train, res_ladder, self.particle_pair, None, Wigner_informed=self.options.Wigner_informed_cross_validation, PorterThomas_informed=self.options.PorterThomas_informed_cross_validation)
            # Test
            chi2_eff = evaluate_chi2_test(res_ladder, solver_test, test_indices, train_indices)
            chi2_test = np.sum(chi2_eff)
            # test_out = solver_test.fit(res_ladder)
            # chi2_eff = test_out.chi2
            # chi2_test = np.sum(chi2_eff)
            obj_test = objective_func(chi2_test, res_ladder, self.particle_pair, None, Wigner_informed=self.options.Wigner_informed_cross_validation, PorterThomas_informed=self.options.PorterThomas_informed_cross_validation)

            fold_results[key] = CrossValidationOUT(chi2_test  = chi2_test , obj_test  = obj_test , ndata_test  = Ndata_test ,
                                                   chi2_train = chi2_train, obj_train = obj_train, ndata_train = Ndata_train)

        return fold_results
