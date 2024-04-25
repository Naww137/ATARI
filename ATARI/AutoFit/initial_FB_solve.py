from typing import Protocol
from ATARI.AutoFit.functions import * #eliminate_small_Gn, update_vary_resonance_ladder, get_external_resonance_ladder, get_starting_feature_bank
from ATARI.sammy_interface import sammy_classes, sammy_functions
import numpy as np
import pandas as pd
from copy import copy




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
        self._external_resonances = True

        self._width_elimination = True
        self._Gn_threshold = 1e-2
        self._decrease_chi2_threshold_for_width_elimination = True

        self._max_steps = 50
        self._iterations = 2
        self._step_threshold = 0.001
        self._step_threshold_lag = 1

        self._LevMar = True
        self._LevMarV = 1.5
        self._LevMarVd = 5

        self._initial_parameter_uncertainty = 0.05

        self._batch_fitpar = False
        self._batch_fitpar_ifit = 10
        self._steps_per_batch = 2
        self._batch_fitpar_random = False

        self._fitpar1 = [0,0,1]
        self._fitpar2 = [1,1,1]
        self._fit_all_spin_groups = True
        self._spin_group_keys = []

        self._num_Elam = None
        
        self._Elam_shift = 0

        self._starting_Gg_multiplier = 1
        self._starting_Gn1_multiplier = 50



        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    
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
    def external_resonances(self):
        return self._external_resonances
    @external_resonances.setter
    def external_resonances(self, external_resonances):
        self._external_resonances = external_resonances

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
    def step_threshold_lag(self):
        return self._step_threshold_lag
    @step_threshold_lag.setter
    def step_threshold_lag(self, step_threshold_lag):
        self._step_threshold_lag = step_threshold_lag

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
    def initial_parameter_uncertainty(self):
        return self._initial_parameter_uncertainty
    @initial_parameter_uncertainty.setter
    def initial_parameter_uncertainty(self, initial_parameter_uncertainty):
        self._initial_parameter_uncertainty = initial_parameter_uncertainty

    @property
    def batch_fitpar(self):
        return self._batch_fitpar
    @batch_fitpar.setter
    def batch_fitpar(self, batch_fitpar):
        self._batch_fitpar = batch_fitpar

    @property
    def batch_fitpar_ifit(self):
        return self._batch_fitpar_ifit
    @batch_fitpar_ifit.setter
    def batch_fitpar_ifit(self, batch_fitpar_ifit):
        self._batch_fitpar_ifit = batch_fitpar_ifit

    @property
    def steps_per_batch(self):
        return self._steps_per_batch
    @steps_per_batch.setter
    def steps_per_batch(self, steps_per_batch):
        self._steps_per_batch = steps_per_batch

    @property
    def batch_fitpar_random(self):
        return self._batch_fitpar_random
    @batch_fitpar_random.setter
    def batch_fitpar_random(self, batch_fitpar_random):
        self._batch_fitpar_random = batch_fitpar_random

    @property
    def fitpar2(self):
        return self._fitpar2
    @fitpar2.setter
    def fitpar2(self, fitpar2):
        self._fitpar2 = fitpar2

    @property
    def fitpar1(self):
        return self._fitpar1
    @fitpar1.setter
    def fitpar1(self, fitpar1):
        self._fitpar1 = fitpar1

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
        self._spin_group_keys = [float(each) for each in spin_group_keys]


    @property
    def num_Elam(self):
        return self._num_Elam
    @num_Elam.setter
    def num_Elam(self, num_Elam):
        self._num_Elam = num_Elam

    
    @property
    def Elam_shift(self):
        """Shift in energy for all res. beginning from left border of the window (used for each spin group) """
        return self._Elam_shift
    @Elam_shift.setter
    def Elam_shift(self, Elam_shift):
        self._Elam_shift = Elam_shift


    @property
    def starting_Gg_multiplier(self):
        return self._starting_Gg_multiplier
    @starting_Gg_multiplier.setter
    def starting_Gg_multiplier(self, starting_Gg_multiplier):
        self._starting_Gg_multiplier = starting_Gg_multiplier

    @property
    def starting_Gn1_multiplier(self):
        return self._starting_Gn1_multiplier
    @starting_Gn1_multiplier.setter
    def starting_Gn1_multiplier(self, starting_Gn1_multiplier):
        self._starting_Gn1_multiplier = starting_Gn1_multiplier







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

        # ### setup spin groups
        # if self.options.fit_all_spin_groups:
        #     spin_groups = [each[1] for each in particle_pair.spin_groups.items()] 
        # else:
        #     assert len(self.options.spin_group_keys)>0
        #     spin_groups = [each[1] for each in particle_pair.spin_groups.items() if each[0] in self.options.spin_group_keys]
        
        # ### generate intial_feature bank
        # initial_resonance_ladder = get_starting_feature_bank(energy_range,
        #                                                     particle_pair,
        #                                                     spin_groups,
        #                                                     num_Elam= self.options.num_Elam,
        #                                                     starting_Gg_multiplier = self.options.starting_Gg_multiplier,
        #                                                     starting_Gn1_multiplier = self.options.starting_Gn1_multiplier, 
        #                                                     varyE = self.options.fitpar1[0], 
        #                                                     varyGg = self.options.fitpar1[1], 
        #                                                     varyGn1 = self.options.fitpar1[2])
    
        # ### setup external resonances
        # if self.options.external_resonances:
        #     if external_resonance_ladder.empty:
        #         external_resonance_ladder = generate_external_resonance_ladder(spin_groups, energy_range, particle_pair)
        #     else:
        #         pass
        # else:
        #     external_resonance_ladder = pd.DataFrame()
        # initial_resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(initial_resonance_ladder, external_resonance_ladder)

        ### setup sammy inp
        sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = particle_pair,
            resonance_ladder = initial_resonance_ladder,  

            datasets= datasets,
            experiments = experiments,
            experimental_covariance= covariance_data, 
            
            max_steps = self.options.max_steps,
            iterations = self.options.iterations,
            step_threshold = self.options.step_threshold,
            step_threshold_lag = self.options.step_threshold_lag,

            LevMar = self.options.LevMar,
            LevMarV = self.options.LevMarV,
            LevMarVd = self.options.LevMarVd,

            batch_fitpar = self.options.batch_fitpar,
            batch_fitpar_ifit = self.options.batch_fitpar_ifit,
            steps_per_batch = self.options.steps_per_batch,
            batch_fitpar_random = self.options.batch_fitpar_random,

            external_resonance_indices = external_resonance_indices,

            minF = 1e-5,
            maxF = 2.0,
            initial_parameter_uncertainty = self.options.initial_parameter_uncertainty,
            
            autoelim_threshold = None,
            LS = False,
            )

        ### Fit 1 on Gn only
        print("========================================\n\tFIT 1\n========================================")
        print(f"Options to vary: {self.options.fitpar1}")

        outs_fit_1 = self.fit_and_eliminate(rto, sammyINPyw, external_resonance_indices)
        # if save:
        #     self.outs_fit_Gn = outs_fit_Gn
        reslad_1 = copy(outs_fit_1[-1].par_post)
        assert(isinstance(reslad_1, pd.DataFrame))

        ### Fit 2 on E and optionally Gg
        print("========================================\n\tFIT 2\n========================================")
        print(f"Options to vary: {self.options.fitpar2}")

        internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(reslad_1, external_resonance_indices)
        internal_resonance_ladder = update_vary_resonance_ladder(internal_resonance_ladder, 
                                                                 varyE = self.options.fitpar2[0],
                                                                 varyGg = self.options.fitpar2[1],
                                                                 varyGn1 = self.options.fitpar2[2])
        reslad_1, external_resonance_indices = concat_external_resonance_ladder(internal_resonance_ladder, external_resonance_ladder)
        sammyINPyw.resonance_ladder = reslad_1

        outs_fit_2 = self.fit_and_eliminate(rto,sammyINPyw,external_resonance_indices)
        
        return InitialFBOUT(outs_fit_1, outs_fit_2, external_resonance_indices)
    




    def fit_and_eliminate(self, 
               rto,
               sammyINPyw,
               external_resonance_indices
               ):
        
        print(f"Initial solve from {len(sammyINPyw.resonance_ladder)-len(external_resonance_indices)} resonance features\n")
        sammyOUT_fit = sammy_functions.run_sammy_YW(sammyINPyw, rto)
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
                    if self.options.decrease_chi2_threshold_for_width_elimination:
                        sammyINPyw.step_threshold *= 0.1
                    print(f"\n----------------------------------------\nEliminated {round(fraction_eliminated*100, 2)}% of resonance features based on neutron width")
                    print(f"Resolving with {len(internal_resonance_ladder_reduced)} resonance features\n----------------------------------------\n")
                    sammyINPyw.resonance_ladder = resonance_ladder
                    sammyOUT_fit = sammy_functions.run_sammy_YW(sammyINPyw, rto)
                    outs.append(sammyOUT_fit)
            print(f"\nComplete after no neutron width features below threshold\n")

        return outs
    

    # def report(self, string):
    #     if self.report_to_file:
    #         self.