from typing import Union, List, Callable
import pandas as pd
from copy import copy
from dataclasses import fields

from ATARI.sammy_interface import sammy_functions
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputDataYW, SammyInputDataEXT, SammyOutputData, SolverOPTs
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.utils.datacontainers import Evaluation_Data
from ATARI.AutoFit.external_fit_cole import run_sammy_EXT # FIXME: THIS IS A TEMPORARY CHANGE FOR TESTING!
from ATARI.sammy_interface.sammy_functions import get_idc_at_theory

# def filter_kwargs_for_class(cls, **kwargs):
#     cls_signature = inspect.signature(cls)
#     valid_params = cls_signature.parameters
#     # return {k: v for k, v in kwargs.items() if k in valid_params}
#     dictionary = {}
#     for k, v in kwargs.items():
#         if k in valid_params:
#             dictionary[k] = v
#         else:
#             print(f"Warning: kwarg {k} not need for solver, will not have effect")
#     return dictionary
            
def filter_public_attributes(obj):
    return {key: getattr(obj, key) for key in dir(obj) if not key.startswith('_')}

class Solver:
    def __init__(self, sammyRTO:SammyRunTimeOptions, sammyINP:Union[SammyInputDataYW,SammyInputDataEXT], fit_func:Callable):
        self.sammyRTO = sammyRTO
        self.sammyINP = sammyINP
        self.fit_func = fit_func

    def fit(self, resonance_ladder:pd.DataFrame, external_resonance_indices:List[int]=[]) -> SammyOutputData:
        self.sammyINP.resonance_ladder=resonance_ladder
        self.sammyINP.external_resonance_indices=external_resonance_indices
        return self.fit_func(self.sammyINP, self.sammyRTO)
    
    def get_idc_at_theory(self, resonance_ladder:pd.DataFrame) -> List[dict]:
        idc_cov_at_theory = get_idc_at_theory(self.sammyINP, self.sammyRTO, resonance_ladder)
        return idc_cov_at_theory
    
    def set_bayes(self, bayes_boolean:bool):
        self.sammyRTO.bayes=bayes_boolean

    @property
    def Ndata(self) -> int:
        return sum([len(each) for each in self.sammyINP.datasets])

    # def get_smaller_energy_range()
    

def Solver_factory(rto, 
                   solver:str, 
                   solver_options:SolverOPTs, 
                   particle_pair:Particle_Pair, 
                   evaluation_data:Evaluation_Data,
                   cap_norm_unc:float=0.0384200,
                   remove_V:bool = False,
                   V_is_inv:bool = False,
                   Vinv = None,
                   D = None,
                   ) -> Solver:

    if solver_options.idc_at_theory:
        if evaluation_data.measurement_models is None:
            raise ValueError("User specified idc_at_theory, but did not supply a measurement model")

    if solver == "YW":
        sammyINP = SammyInputDataYW(particle_pair, pd.DataFrame(), 
                                    datasets=evaluation_data.datasets, 
                                    experiments=evaluation_data.experimental_models, 
                                    experiments_no_pup=evaluation_data.experimental_models_no_pup, 
                                    experimental_covariance=evaluation_data.covariance_data, 
                                    external_resonance_indices = [], 
                                    measurement_models = evaluation_data.measurement_models,
                                    **filter_public_attributes(solver_options))
        fit_func = sammy_functions.run_sammy_YW

    elif solver == "EXT":

        if evaluation_data.experimental_models_no_pup is None:
            raise ValueError("experiments with no PUP must be specified")

        sammyINP = SammyInputDataEXT(particle_pair, pd.DataFrame(), 
                                     evaluation_data.datasets, 
                                     experiments=evaluation_data.experimental_models,
                                     experiments_no_pup=evaluation_data.experimental_models_no_pup, 
                                     experimental_covariance=evaluation_data.covariance_data, 
                                     external_resonance_indices=[], 
                                     cap_norm_unc=cap_norm_unc, 
                                     remove_V = remove_V,
                                     V_is_inv = V_is_inv,
                                     Vinv = Vinv,
                                     D = D, 
                                     measurement_models = evaluation_data.measurement_models,
                                     **filter_public_attributes(solver_options))
        fit_func = run_sammy_EXT
    else:
        raise ValueError("Solver not recognized")
    
    return Solver(rto, sammyINP, fit_func)

def get_parent_solver_options(solver:SolverOPTs):
    if not isinstance(solver, SolverOPTs):
        raise TypeError(f"Expected a SolverOPTs or a subclass of SolverOPTs, got {type(solver).__name__}")
    # Handle both parent and any subclass of Parent
    parent_fields = {field.name for field in fields(SolverOPTs)}
    # Extract fields belonging to parent only
    parent_data = {key: getattr(solver, key) for key in parent_fields}
    # Return a parent object using the filtered data
    return SolverOPTs(**parent_data)

