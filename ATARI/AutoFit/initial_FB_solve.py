from typing import Protocol
from ATARI.AutoFit.functions import * #eliminate_small_Gn, update_vary_resonance_ladder, get_external_resonance_ladder, get_starting_feature_bank
from ATARI.sammy_interface import sammy_classes, sammy_functions
import numpy as np
import pandas as pd
from copy import copy
from ATARI.AutoFit import sammy_interface_bindings


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
    width_elimination_Gn_threshold: float = 1e-2
        Neutron width threshold for width-based elimination
    width_elimination_Nres_threshold: int = None
        Number of resonances, below which width-based elimination stops.
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

        ### IFB settings
        self._external_resonances = True
        self._fit_all_spin_groups = True
        self._spin_group_keys = []
        self._num_Elam = None
        self._Elam_shift = 0
        self._starting_Gg_multiplier = 1
        self._starting_Gn1_multiplier = 50
        # self._off_diag_covariance = False

        ### Procedural settings
        self._fitpar1 = [0,0,1]
        self._fitpar2 = [1,1,1]
        self._width_elimination = True
        self._width_elimination_Gn_threshold = 1e-2
        self._width_elimination_Nres_threshold = None
        self._decrease_chi2_threshold_for_width_elimination = True

        # ### Solver 
        # self._solver_options = None
        # self._solver = "YW"

        ### set kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # if self.solver_options is None:
        #     if self.solver == "YW":
        #         self.solver_options = sammy_classes.SolverOPTs_YW(
        #                                             max_steps = 50,
        #                                             iterations = 3,
        #                                             step_threshold=0.001,
        #                                             LevMar=True, LevMarV=1.5,LevMarVd=5,
        #                                             maxF=2.0, minF=1e-5,
        #                                             initial_parameter_uncertainty=0.05)
        #     elif self.solver == "EXT":
        #         self.solver_options = sammy_classes.SolverOPTs_EXT(
        #                                             max_steps = 50,
        #                                             step_threshold=0.001,
        #                                             LevMar=True, LevMarV=1.5,LevMarVd=5,
        #                                             maxF=1e-5, minF=1e-8,
        #                                             alpha=1e-6, gaus_newton=True)


    ### define repr
    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    

    ### Getters and setters for options

    # @property
    # def solver(self):
    #     return self._solver
    # @solver.setter
    # def solver(self, solver):
    #     if solver == "YW":
    #         pass
    #     elif solver == "EXT":
    #         pass
    #     else:
    #         raise ValueError(f"Solver string {solver} not recognized")
    #     self._solver = solver
    
    # @property
    # def solver_options(self):
    #     return self._solver_options
    # @solver_options.setter
    # def solver_options(self, solver_options):
    #     self._solver_options = solver_options

    @property
    def width_elimination(self):
        return self._width_elimination
    @width_elimination.setter
    def width_elimination(self, width_elimination):
        self._width_elimination = width_elimination

    @property
    def width_elimination_Gn_threshold(self):
        return self._width_elimination_Gn_threshold
    @width_elimination_Gn_threshold.setter
    def width_elimination_Gn_threshold(self, width_elimination_Gn_threshold):
        self._width_elimination_Gn_threshold = width_elimination_Gn_threshold

    @property
    def width_elimination_Nres_threshold(self):
        return self._width_elimination_Nres_threshold
    @width_elimination_Gn_threshold.setter
    def width_elimination_Nres_threshold(self, width_elimination_Nres_threshold):
        self._width_elimination_Nres_threshold = width_elimination_Nres_threshold

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
                 solver: sammy_interface_bindings.Solver,
                 options: InitialFBOPT,
                 ):
        
        self.solver = solver
        self.options = options

    def fit(self,
            particle_pair,
            energy_range,
            # datasets,
            # experiments,
            # covariance_data,
            # sammyRTO,
            external_resonance_ladder = None,
            # internal_resonance_ladder = None,
            # experiments_no_pup = None,
            # cap_norm_unc=0.0
            ):

        initial_resonance_ladder, external_resonance_indices = get_initial_resonance_ladder(self.options, particle_pair, energy_range, external_resonance_ladder=external_resonance_ladder)

        ### Fit 1 on Gn only
        print("========================================\n\tFIT 1\n========================================")
        print(f"Options to vary: {self.options.fitpar1}")

        outs_fit_1 = self.fit_and_eliminate(initial_resonance_ladder, external_resonance_indices)
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

        outs_fit_2 = self.fit_and_eliminate(reslad_1, external_resonance_indices)
        
        return InitialFBOUT(outs_fit_1, outs_fit_2, external_resonance_indices)
    




    def fit_and_eliminate(self, resonance_ladder, external_resonance_indices):
        
        print(f"Initial solve from {len(resonance_ladder)-len(external_resonance_indices)} resonance features\n")
        sammyOUT_fit = self.solver.fit(resonance_ladder, external_resonance_indices)
        outs = [sammyOUT_fit]

        if self.options.width_elimination:
            for it in range(10_000):
                internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(sammyOUT_fit.par_post, external_resonance_indices)
                internal_resonance_ladder_reduced, fraction_eliminated = eliminate_small_Gn(internal_resonance_ladder, self.options.width_elimination_Gn_threshold)
                resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(internal_resonance_ladder_reduced, external_resonance_ladder)
                if fraction_eliminated == 0.0:
                    break # no longer eliminating after not eliminating any more resonances
                elif fraction_eliminated == 100.0:
                    raise ValueError("Eliminated all resonances due to width, please change settings")
                else:
                    if self.options.decrease_chi2_threshold_for_width_elimination:
                        self.solver.sammyINP.step_threshold *= 0.1
                    print(f"\n----------------------------------------\nEliminated {round(fraction_eliminated*100, 2)}% of resonance features based on neutron width")
                    print(f"Resolving with {len(internal_resonance_ladder_reduced)} resonance features\n----------------------------------------\n")
                    sammyOUT_fit = self.solver.fit(resonance_ladder, external_resonance_indices)
                    outs.append(sammyOUT_fit)
                if (self.options.width_elimination_Nres_threshold is not None) \
                    and (len(internal_resonance_ladder_reduced) <= self.options.width_elimination_Nres_threshold):
                    break # no longer eliminating after going below the threshold number of resonances
            else:
                raise RuntimeError('Initial IFB solve never stopped eliminating somehow.')
            print(f"\nComplete after no neutron width features below threshold\n")

        return outs
    

    # def report(self, string):
    #     if self.report_to_file:
    #         self.