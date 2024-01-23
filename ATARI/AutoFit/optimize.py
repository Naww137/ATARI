from typing import Protocol, Optional, Union
from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.AutoFit.control import Evaluation_Data
from ATARI.ModelData.particle_pair import Particle_Pair
import numpy as np
import pandas as pd
from copy import copy, deepcopy



class OptimizeOPT:
    def __init__(self, **kwargs):

        self._use_sammy_YW = True

        self._max_steps = 30
        self._iterations = 2
        self._step_threshold = 0.001

        self._LevMar = True
        self._LevMarV = 1.5
        self._LevMarVd = 5
        self._LevMarV0 = 0.05
        self._minF = 1e-5
        self._maxF = 2.0

        for key, value in kwargs.items():
            setattr(self, key, value)
    

    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    

    @property
    def use_sammy_YW(self):
        return self._use_sammy_YW
    @use_sammy_YW.setter
    def use_sammy_YW(self, use_sammy_YW):
        self._use_sammy_YW = use_sammy_YW

    @property
    def max_steps(self):
        """
        Maximum number of steps in non-linear least squares solution scheme.
        """
        return self._max_steps
    @max_steps.setter
    def max_steps(self, max_steps):
        self._max_steps = max_steps

    @property
    def iterations(self):
        """
        Number of internal SAMMY iterations of G for nonlinearity.
        """
        return self._iterations
    @iterations.setter
    def iterations(self, iterations):
        # if iterations < 1:
        #     raise ValueError("iterations must be at least 1")
        self._iterations = iterations

    @property
    def step_threshold(self):
        """
        Chi2 improvement threshold convergence criteria.
        """
        return self._step_threshold
    @step_threshold.setter
    def step_threshold(self, step_threshold):
        self._step_threshold = step_threshold
    
    @property
    def LevMar(self):
        """
        
        """
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
    def minF(self):
        return self._minF
    @minF.setter
    def minF(self, minF):
        self._minF = minF

    @property
    def maxF(self):
        return self._maxF
    @maxF.setter
    def maxF(self, maxF):
        self._maxF = maxF




class OptimizeOUT:
    def __init__(self, 
                 
                 par: pd.DataFrame,
                 par_post: pd.DataFrame,

                 pw: list[pd.DataFrame], 
                 pw_post: list[pd.DataFrame],

                 chi2: list[float],
                 chi2_post: list[float]
                 ):
            
            self.par = par
            self.par_post = par_post
    
            self.pw = pw
            self.pw_post = pw_post

            self.chi2 = chi2
            self.chi2_post = chi2_post

    
def samout_2_optout(samout):
    return OptimizeOUT(par        = samout.par,
                    par_post   = samout.par_post,
                    pw         = samout.pw,
                    pw_post    = samout.pw_post,
                    chi2       = samout.chi2,
                    chi2_post  = samout.chi2_post)


class Optimize:

    def __init__(self, 
                 sammy_rto: sammy_classes.SammyRunTimeOptions,
                 options: Optional[OptimizeOPT] = None):

        self.sammy_rto = sammy_rto

        if options is None:
            self.options = OptimizeOPT()
        else:
            self.options = options



    def optimize(self, 
                 resonance_ladder: pd.DataFrame,
                 particle_pair: Particle_Pair,
                 evaluation_data: Evaluation_Data):

        if self.options.use_sammy_YW:
            self.optimize_with_internal_YW(resonance_ladder, particle_pair, evaluation_data)
        else:
            pass


    def optimize_with_internal_YW(self,
                                  resonance_ladder: pd.DataFrame,
                                  particle_pair: Particle_Pair,
                                  evaluation_data: Evaluation_Data):
        
        ### setup sammy inp
        sammyINPyw = sammy_classes.SammyInputDataYW(
            particle_pair = particle_pair,
            resonance_ladder = resonance_ladder,  

            datasets= evaluation_data.pw_data,
            experiments = evaluation_data.experimental_models,
            experimental_covariance= evaluation_data.covariance_data, 
            
            max_steps = self.options.max_steps,
            iterations = self.options.iterations,
            step_threshold = self.options.step_threshold,
            LevMar = self.options.LevMar,
            LevMarV = self.options.LevMarV,
            LevMarVd = self.options.LevMarVd,
            minF = self.options.minF,
            maxF = self.options.maxF,
            initial_parameter_uncertainty = self.options.LevMarV0,
            
            autoelim_threshold = None,
            LS = False,
            )


        sammyOUT = sammy_functions.run_sammy_YW(sammyINPyw, self.sammy_rto)

        return samout_2_optout(sammyOUT)



        