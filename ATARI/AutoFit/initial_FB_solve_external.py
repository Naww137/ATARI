from typing import Protocol
from ATARI.AutoFit.functions import * #eliminate_small_Gn, update_vary_resonance_ladder, get_external_resonance_ladder, get_starting_feature_bank
from ATARI.sammy_interface import sammy_classes, sammy_functions
import numpy as np
import pandas as pd
from copy import copy
from ATARI.AutoFit import external_fit
import scipy


class InitialFBOPT:
    """
    Options for the initial feature bank solver module.

    Parameters
    ----------
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    Attributes
    ----------
    external_resonances: bool = True
        If True, one resonance of variable widths for each spin group will be fixed at one average level spacing outside of the window.
    width_elimination: bool = True
        Option to eliminate resonances during fitting stages based on neutron width.
    Gn_threshold: Float = 1e-2
        Neutron width threshold for width-based elimination
    decrease_chi2_threshold_for_width_elimination: bool = True
        If running width elimination, decrease the chi2 threshold convergence criteria

    max_steps: int = 50
        Maximum number of steps in non-linear least squares solution scheme.
    iterations: int = 2
        Number of internal SAMMY iterations of G for nonlinearity.
    step_threshold: bool = True
        Chi2 improvement threshold convergence criteria.
    step_threshold_lag: int = 1
        Requires a number of steps that the improvement threshold is not met before terminating.
        Intended to be used with batch settings.

    LevMar: bool = True
        Option to use the Levenberg-Marquardt algorithm.
    LevMarV: float = 1.5
        Levenberg-Marquardt dampening parameter up-scaling factor.
    LevMarVd: float = 5
        Levenberg-Marquardt dampening parameter down-scaling factor.

    batch_fitpar: bool = False
        Option to batch fitting parameters, batches will be done by resonances with fitted parameters determined by fitpar1 or fitpar2 setting.
    batch_fitpar_ifit = 10
        Number of resonances to fit per batch.
    steps_per_batch = 2
        Number of update steps to take for a given batch.
    batch_fitpar_random = False
        Option to randomly assign resonances to a batch, when false, 
        the number of fitted resonances (determined by batch_fitpar_ifit) is uniformly spaced across the resonance dataframe and shifted by 1.

    initial_parameter_uncertainty: float = 0.05
        Initial fractional parameter uncertainty, will be updated with each step if LevMar=True.

    fitpar1: list = [0,0,1]
        Boolean list for fit 1 that determines which parameters will be optimized (E, Gg, Gn1).
    fitpar2: list = [1,1,1]
        Boolean list for fit 2 that determines which parameters will be optimized (E, Gg, Gn1).
    fit_all_spin_groups: bool = True
        Option to initialize feature bank with all spin groups present in particle_pair.
        If False, spin_group_keys will control which spin groups to use and must be provided.
    spin_group_keys: list = []
        List of spin group keys (corresponding to spin groups in particle pair) to be used in initialization of feature bank.
        Only used if fit_all_spin_groups == False.

    num_Elam: Optional[int] = None
        Number of resonance features in starting feature bank for each spin group.
        If None, the number of resonance features will be set to approximately 1.6 per eV.
    starting_Gg_multiplier: float = 1.0
        Factor of average capture width used in initial feature bank.
    starting_Gn1_multiplier: float = 50.0
        Factor of Q01 neutron width used in initial feature bank.
    """
    def __init__(self, **kwargs):
        self.external_resonances = True
        self.width_elimination = True
        self.Gn_threshold = 1e-2
        self.fitpar1 = [0,0,1]
        self.fitpar2 = [1,1,1]
        self.fit_all_spin_groups = True
        self.spin_group_keys = []
        self.num_Elam = None
        self.starting_Gg_multiplier = 1
        self.starting_Gn1_multiplier = 50


        self.max_steps = 50
        # self._iterations = 2
        self.step_threshold = 0.001
        # self._step_threshold_lag = 1
        self.steps = 100, 
        self.thresh = 0.01, 
        self.alpha = 1e-6, 
        self.print_bool = True, 
        self.gaus_newton = False, 
        self.LevMar = True, 
        self.LevMarV = 2, 
        self.LevMarVd = 5, 
        self.maxV = 1e-4, 
        self.minV=1e-8, 
        
        self.lasso = False,
        self.lasso_parameters = {"lambda":1, 
                            "gamma":0,
                            "weights":None},
        self.ridge = False,
        self.ridge_parameters = {"lambda":1, 
                            "gamma":0,
                            "weights":None},
        self.elastic_net = True,
        self.elastic_net_parameters = {"lambda":1, 
                                "gamma":0,
                                "alpha":0.7},

        self.batch_fitpar = False
        self.batch_fitpar_ifit = 10
        self.steps_per_batch = 2
        self.batch_fitpar_random = False



        for key, value in kwargs.items():
            setattr(self, key, value)






class InitialFBOUT:
    def __init__(self,
                 outs_fit_1: list[sammy_classes.SammyOutputData],
                 outs_fit_2: list[sammy_classes.SammyOutputData],
                 external_resonance_indices):
        
        self.sammy_outs_fit_1 = outs_fit_1
        self.sammy_outs_fit_2 = outs_fit_2
        # outs_fit_1.extend(outs_fit_2)
        # self.sammy_outs = outs_fit_1

        samout_final = outs_fit_2[-1]
        final_par = copy(samout_final.par_post)
        internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(final_par, external_resonance_indices)
        self.final_internal_resonances = internal_resonance_ladder
        self.final_external_resonances = external_resonance_ladder
        self.final_external_resonance_indices = external_resonance_indices
        self.final_resonace_ladder = final_par

    
    @property
    def chi2_all(self):
        chi2_list = []
        for samout in self.sammy_outs_fit_1:
            chi2_list.append(np.sum(samout.chi2_post))
        for samout in self.sammy_outs_fit_2:
            chi2_list.append(np.sum(samout.chi2_post))
        return chi2_list

    @property
    def Nres_all(self):
        N_res = []
        for samout in self.sammy_outs_fit_1:
            N_res.append(len(samout.par_post))
        for samout in self.sammy_outs_fit_2:
            N_res.append(len(samout.par_post))
        return N_res




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
            sammyRTO,
            external_resonance_ladder = pd.DataFrame()#,
            # internal_resonance_ladder = None,
            ):
        
        rto = copy(sammyRTO)
        assert rto.bayes == True

        initial_resonance_ladder = get_initial_resonance_ladder(self.options, particle_pair, energy_range)

        ### Fit 1 on Gn only
        print("========================================\n\tFIT 1\n========================================")
        outs_fit_1 = self.fit_and_eliminate(rto,
               initial_resonance_ladder,
               external_resonance_indices,
               particle_pair,
               experiments,
               datasets,
               covariance_data)
        # if save:
        #     self.outs_fit_Gn = outs_fit_Gn
        reslad_1 = copy(outs_fit_1[-1].par_post)
        assert(isinstance(reslad_1, pd.DataFrame))

        ### Fit 2 on E and optionally Gg
        print("========================================\n\tFIT 2\n========================================")
        internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(reslad_1, external_resonance_indices)
        internal_resonance_ladder = update_vary_resonance_ladder(internal_resonance_ladder, 
                                                                 varyE = self.options.fitpar2[0],
                                                                 varyGg = self.options.fitpar2[1],
                                                                 varyGn1 = self.options.fitpar2[2])
        reslad_1, external_resonance_indices = concat_external_resonance_ladder(internal_resonance_ladder, external_resonance_ladder)
        outs_fit_2 = self.fit_and_eliminate(rto,
               reslad_1,
               external_resonance_indices,
               particle_pair,
               experiments,
               datasets,
               covariance_data)
        
        return InitialFBOUT(outs_fit_1, outs_fit_2, external_resonance_indices)
    




    def fit_and_eliminate(self, 
               rto,
               resonance_ladder,
               external_resonance_indices,
               particle_pair,
               experiments,
               datasets,
               covariance_data
               ):
        
        print(f"Solving from {len(resonance_ladder.resonance_ladder)-len(external_resonance_indices)} resonance features\n")
        Ds, covs = external_fit.get_Ds_Vs(experiments, datasets, covariance_data)
        D = np.concatenate(Ds)
        V = scipy.linalg.block_diag(*covs)
        saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log = external_fit.fit(rto, resonance_ladder,  external_resonance_indices,  particle_pair, D, V, experiments, datasets, covariance_data,
                                                                                                        steps = self.options.steps, thresh = self.options.thresh, alpha = self.options.alpha, print_bool = self.options.print_bool, 
                                                                                                        gaus_newton = self.options.gaus_newton,  LevMar = self.options.LevMar,  LevMarV = self.options.LevMarV,  LevMarVd = self.options.LevMarVd,  maxV = self.options.maxV,  minV= self.options.minV, 
                                                                                                        lasso = self.options.lasso, lasso_parameters =self.options.lasso_parameters,
                                                                                                        ridge = self.options.ridge, ridge_parameters =self.options.ridge_parameters,
                                                                                                        elastic_net = self.options.elastic_net, elastic_net_parameters =self.options.elastic_net_parameters)
        samout = sammy_classes.SammyOutputData(saved_pw_lists[0], saved_res_lads[0], chi2_log[0], obj_log[0],saved_pw_lists[-1], saved_res_lads[-1],chi2_log[-1], obj_log[-1])
        outs = [samout]

        if self.options.width_elimination:
            eliminating = True
            while eliminating:
                internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(saved_res_lads[-1], external_resonance_indices)
                internal_resonance_ladder_reduced, fraction_eliminated = eliminate_small_Gn(internal_resonance_ladder, self.options.Gn_threshold)
                resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(internal_resonance_ladder_reduced, external_resonance_ladder)
                if fraction_eliminated == 0.0:
                    eliminating = False
                elif fraction_eliminated == 100.0:
                    raise ValueError("Eliminated all resonances due to width, please change settings")
                else:
                    print(f"\n----------------------------------------\nEliminated {round(fraction_eliminated*100, 2)}% of resonance features based on neuton width")
                    print(f"Resolving with {len(internal_resonance_ladder_reduced)} resonance features\n----------------------------------------\n")
                    saved_res_lads, save_Pu, saved_pw_lists, saved_gradients, chi2_log, obj_log = external_fit.fit(rto, resonance_ladder,  external_resonance_indices,  particle_pair, D, V, experiments, datasets, covariance_data,
                                                                                                        steps = self.options.steps, thresh = self.options.thresh, alpha = self.options.alpha, print_bool = self.options.print_bool, 
                                                                                                        gaus_newton = self.options.gaus_newton,  LevMar = self.options.LevMar,  LevMarV = self.options.LevMarV,  LevMarVd = self.options.LevMarVd,  maxV = self.options.maxV,  minV= self.options.minV, 
                                                                                                        lasso = self.options.lasso, lasso_parameters =self.options.lasso_parameters,
                                                                                                        ridge = self.options.ridge, ridge_parameters =self.options.ridge_parameters,
                                                                                                        elastic_net = self.options.elastic_net, elastic_net_parameters =self.options.elastic_net_parameters)
                    samout = sammy_classes.SammyOutputData(saved_pw_lists[0], saved_res_lads[0], chi2_log[0], obj_log[0],saved_pw_lists[-1], saved_res_lads[-1],chi2_log[-1], obj_log[-1])
                    outs.append(samout)
                    
            print(f"\nComplete after no neutron width features below threshold\n")

        return outs
    




    # def report(self, string):
    #     if self.report_to_file:
    #         self.