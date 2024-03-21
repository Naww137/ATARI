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
    Gn_threshold: bool = True
        Neutron width threshold for width-based elimination
    decrease_chi2_threshold_for_width_elimination: bool = True
        If running width elimination, decrease the chi2 threshold convergence criteria
    max_steps: bool = True
        Maximum number of steps in non-linear least squares solution scheme.
    iterations: bool = True
        Number of internal SAMMY iterations of G for nonlinearity.
    step_threshold: bool = True
        Chi2 improvement threshold convergence criteria.
    LevMar: bool = True

    LevMarV: bool = True

    LevMarVd: bool = True

    LevMarV0: bool = True
    
    fit_Gg: bool = True
        Fit gamma width in fit 2.
    fit_all_spin_groups: bool = True

    spin_group_keys: list = []

    num_Elam: Optional[int] = None
        Number of resonance features in starting feature bank for each spin group
    starting_Gg_multiplier: float = 1.0
        Factor of average capture width used in initial feature bank
    starting_Gn1_multiplier: float = 1.0
        Factor of Q01 neutron width used in initial feature bank
    """
    def __init__(self, **kwargs):
        self._external_resonances = True

        self._width_elimination = True
        self._Gn_threshold = 1e-2
        self._decrease_chi2_threshold_for_width_elimination = True

        self._max_steps = 30
        self._iterations = 2
        self._step_threshold = 0.001
        self._LevMar = True
        self._LevMarV = 1.5
        self._LevMarVd = 5
        self._LevMarV0 = 0.05
        self._batch_fitpar = False

        self._fit_Gg = True
        self._fit_all_spin_groups = True
        self._spin_group_keys = []

        self._num_Elam = None
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
    def LevMarV0(self):
        return self._LevMarV0
    @LevMarV0.setter
    def LevMarV0(self, LevMarV0):
        self._LevMarV0 = LevMarV0

    @property
    def batch_fitpar(self):
        return self._batch_fitpar
    @batch_fitpar.setter
    def batch_fitpar(self, batch_fitpar):
        self._batch_fitpar = batch_fitpar


    @property
    def fit_Gg(self):
        return self._fit_Gg
    @fit_Gg.setter
    def fit_Gg(self, fit_Gg):
        self._fit_Gg = fit_Gg

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
            external_resonance_ladder = pd.DataFrame()
            ):
        
        rto = copy(sammyRTO)
        assert rto.bayes == True

        ### setup spin groups
        if self.options.fit_all_spin_groups:
            spin_groups = [each[1] for each in particle_pair.spin_groups.items()] 
        else:
            assert len(self.options.spin_group_keys)>0
            spin_groups = [each[1] for each in particle_pair.spin_groups.items() if each[0] in self.options.spin_group_keys]
        
        ### generate intial_feature bank
        initial_resonance_ladder = get_starting_feature_bank(energy_range,
                                                             particle_pair,
                                                             spin_groups,
                                                            num_Elam= self.options.num_Elam,
                                                            starting_Gg_multiplier = self.options.starting_Gg_multiplier,
                                                            starting_Gn1_multiplier = self.options.starting_Gn1_multiplier, 
                                                            varyE = 0, varyGg = 0, varyGn1 = 1)
    
        ### setup external resonances
        if self.options.external_resonances:
            if external_resonance_ladder.empty:
                external_resonance_ladder = generate_external_resonance_ladder(spin_groups, energy_range, particle_pair)
            else:
                pass
        else:
            external_resonance_ladder = pd.DataFrame()
        initial_resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(initial_resonance_ladder, external_resonance_ladder)

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
            LevMar = self.options.LevMar,
            LevMarV = self.options.LevMarV,
            LevMarVd = self.options.LevMarVd,
            batch_fitpar = self.options.batch_fitpar,
            minF = 1e-5,
            maxF = 2.0,
            initial_parameter_uncertainty = self.options.LevMarV0,
            
            autoelim_threshold = None,
            LS = False,
            )

        ### Fit 1 on Gn only
        print("========================================\n\tFIT 1\n========================================")
        outs_fit_1 = self.fit_and_eliminate(rto, sammyINPyw, external_resonance_indices)
        # if save:
        #     self.outs_fit_Gn = outs_fit_Gn
        reslad_1 = copy(outs_fit_1[-1].par_post)
        assert(isinstance(reslad_1, pd.DataFrame))

        ### Fit 2 on E and optionally Gg
        print("========================================\n\tFIT 2\n========================================")
        internal_resonance_ladder, external_resonance_ladder = separate_external_resonance_ladder(reslad_1, external_resonance_indices)
        if self.options.fit_Gg:
            internal_resonance_ladder = update_vary_resonance_ladder(internal_resonance_ladder, varyE=1, varyGg=1, varyGn1=1)
        else:
            internal_resonance_ladder = update_vary_resonance_ladder(internal_resonance_ladder, varyE=1, varyGg=0, varyGn1=1)
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
                    print(f"\n----------------------------------------\nEliminated {round(fraction_eliminated*100, 2)}% of resonance features based on neuton width")
                    print(f"Resolving with {len(internal_resonance_ladder_reduced)} resonance features\n----------------------------------------\n")
                    sammyINPyw.resonance_ladder = resonance_ladder
                    sammyOUT_fit = sammy_functions.run_sammy_YW(sammyINPyw, rto)
                    outs.append(sammyOUT_fit)
            print(f"\nComplete after no neutron width features below threshold\n")

        return outs
    

    def fit_E_Gn(self,
                 rto,
                 sammyINPyw):
        

        return



    # def report(self, string):
    #     if self.report_to_file:
    #         self.