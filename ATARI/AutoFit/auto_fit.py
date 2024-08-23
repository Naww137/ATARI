from copy import copy
from ATARI.AutoFit.sammy_interface_bindings import Solver_factory
from ATARI.AutoFit.fit_and_eliminate import FitAndEliminate, FitAndEliminateOPT, eliminator_OUTput
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SolverOPTs, Particle_Pair, SammyOutputData
import numpy as np
from typing import Optional
from dataclasses import dataclass
import multiprocessing
import os
from ATARI.utils.file_handling import clean_and_make_directory, return_random_subdirectory

@dataclass
class CrossValidationOUT:
    ires                : Optional[np.ndarray]  = None
    train_scores        : Optional[np.ndarray]  = None
    test_scores         : Optional[np.ndarray]  = None


@dataclass
class AutoFitOUT:
    final_samout                : Optional[SammyOutputData]     = None
    Nres_target                 : Optional[int]                 = None
    elimination_history         : Optional[eliminator_OUTput]   = None
    cross_validation_output     : Optional[CrossValidationOUT]  = None
    total_time                  : Optional[float]               = None


@dataclass
class AutoFitOPT:
    # save options
    save_elimination_history        : bool  = True
    save_CV_elimination_history     : bool  = False

    # parallel options
    parallel_CV                     : bool  = False
    parallel_processes              : int   = 5

    # other
    print_bool                      : bool  = True



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
        else:                       self.options = AutoFit_options

        if fit_and_elim_options is None: self.fit_and_elim_options = FitAndEliminateOPT()
        else:                            self.fit_and_elim_options = fit_and_elim_options                  
        
        self.particle_pair = particle_pair

        self.rto_test = copy(sammyRTO)
        self.rto_test.bayes = False
        self.rto_train = copy(sammyRTO)
        self.rto_train.bayes = True

        self.output = AutoFitOUT()


    def fit(self, evaluation_data, resonance_ladder):
        ### Run CV
        save_test_scores, save_train_scores, save_ires, kfolds = self.cross_validation(evaluation_data, resonance_ladder)
        try:
            save_ires = np.array(save_ires)
            test = np.mean(np.array(save_test_scores), axis=0);     test_std = np.std(np.array(save_test_scores), axis=0, ddof=1)/np.sqrt(kfolds)
        except:
            raise ValueError("Not all folds resulted in the same number of resonances - change greediness or update code")
        
        ### Get cardinality from CV results
        ilowest = np.argmin(test)
        lowest = test[ilowest]
        lowest_std = test_std[ilowest]
        iselect = ilowest
        for i in range(ilowest, len(test)):
            if test[i] < lowest + lowest_std:
                iselect = i
        Nres_target = np.unique(save_ires, axis=0)[0][iselect]

        ### Final solve to target Nres with all data 
        solver_initial = Solver_factory(self.rto_train, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data) 
        solver_elim = Solver_factory(self.rto_train, self.solver_options_eliminate._solver, self.solver_options_eliminate, self.particle_pair, evaluation_data)

        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=self.fit_and_elim_options)
        initial_samout = fe.initial_fit(resonance_ladder)
        elimination_history = fe.eliminate(initial_samout.par_post, target_ires=Nres_target)

        # if save_elimination_history:
        self.output.elimination_history = elimination_history
        self.output.final_samout = elimination_history.elimination_history[Nres_target]['selected_ladder_chars']

        return self.output


    def cross_validation(self, evaluation_data, resonance_ladder):

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
            multi_input = [(train, test, resonance_ladder) for train, test in zip(list_evaluation_data_train, list_evaluation_data_test)]
            with multiprocessing.Pool(processes=self.options.parallel_processes) as pool:
                results = pool.map(self.get_cross_validation_score, multi_input)
            ## Pull results
            assert len(results) == kfolds
            save_test_scores, save_train_scores, save_ires = [], [], []
            for k in range(kfolds):
                test_scores, train_scores, ires = results[k]
                save_test_scores.append(test_scores); save_train_scores.append(train_scores); save_ires.append(ires)

        ### get CVE score in serial
        else:
            save_test_scores, save_train_scores, save_ires = [], [], []
            for train, test in zip(list_evaluation_data_train, list_evaluation_data_test):
                test_scores, train_scores, ires = self.get_cross_validation_score((train, test, resonance_ladder))
                save_test_scores.append(test_scores); save_train_scores.append(train_scores); save_ires.append(ires)

        # if save:
        self.output.cross_validation_output = CrossValidationOUT(ires=np.array(save_ires), test_scores=np.array(save_test_scores), train_scores=np.array(save_train_scores))

        return save_test_scores, save_train_scores, save_ires, kfolds
    

    def get_cross_validation_score(self, input_arguement):
        evaluation_data_train, evaluation_data_test, resonance_ladder = input_arguement
        
        # set RTO options if in parallel
        if self.options.parallel_CV:
            rto_train=copy(self.rto_train); rto_test=copy(self.rto_test)
            rto_train.Print=False; rto_test.Print=False; fit_and_elim_options = copy(self.fit_and_elim_options); fit_and_elim_options.print_bool=False
            rto_train.keep_runDIR=False; rto_test.keep_runDIR=False
            rto_train.sammy_runDIR = return_random_subdirectory(self.rto_train.sammy_runDIR); rto_test.sammy_runDIR = return_random_subdirectory(self.rto_train.sammy_runDIR)
        else:
            rto_train=self.rto_train; rto_test=self.rto_test

        # create solvers
        solver_initial = Solver_factory(rto_train, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data_train) 
        solver_elim = Solver_factory(rto_train, self.solver_options_eliminate._solver, self.solver_options_eliminate, self.particle_pair, evaluation_data_train)
        solver_test = Solver_factory(rto_test, self.solver_options_initial._solver, self.solver_options_initial, self.particle_pair, evaluation_data_test)

        # fit and eliminate
        fe = FitAndEliminate(solver_initial=solver_initial, solver_eliminate=solver_elim, options=fit_and_elim_options)
        initial_samout = fe.initial_fit(resonance_ladder)
        elimination_history = fe.eliminate(initial_samout.par_post)

        # get test and train scores
        test_scores, train_scores, ires = [], [], []
        for key, val in elimination_history.elimination_history.items():
            N_train = np.sum([len(each) for each in val['selected_ladder_chars'].pw_post])
            train_scores.append(np.sum(val['selected_ladder_chars'].chi2_post)/N_train)

            test_out = solver_test.fit(val['selected_ladder_chars'].par_post, [])
            N_test = len(test_out.pw[0])

            test_scores.append(np.sum(test_out.chi2)/N_test)
            ires.append(key)

        return (test_scores, train_scores, ires)