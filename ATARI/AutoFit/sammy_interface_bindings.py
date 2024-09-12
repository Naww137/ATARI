import inspect
from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.AutoFit.external_fit import run_sammy_EXT
import pandas as pd

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
    def __init__(self, sammyRTO, sammyINP, fit_func):
        self.sammyRTO = sammyRTO
        self.sammyINP = sammyINP
        self.fit_func = fit_func

    def fit(self, resonance_ladder, external_resonance_indices=[]):
        self.sammyINP.resonance_ladder=resonance_ladder
        self.sammyINP.external_resonance_indices=external_resonance_indices
        return self.fit_func(self.sammyINP, self.sammyRTO)
    
    def set_bayes(self, bayes_boolean):
        self.sammyRTO.bayes=bayes_boolean

    # def get_smaller_energy_range()
    

def Solver_factory(rto, 
                   solver, 
                   solver_options, 
                   particle_pair, 
                #    datasets, experiments, experimental_covariance, 
                   evaluation_data,
                #    experiments_no_pup=None, 
                   cap_norm_unc=0.0384200,
                   remove_V = False,
                   V_is_inv = False,
                   Vinv = None,
                   D = None,
                #    V_projection = None,
                #    measurement_models = None,
                   ):

    if solver_options.idc_at_theory:
        if evaluation_data.measurement_models is None:
            raise ValueError("User specified idc_at_theory, but did not supply a measurement model")

    if solver == "YW":
        sammyINP = sammy_classes.SammyInputDataYW(particle_pair, pd.DataFrame(), 
                                                  datasets=evaluation_data.datasets, 
                                                  experiments=evaluation_data.experimental_models, 
                                                  experimental_covariance=evaluation_data.covariance_data, 
                                                  external_resonance_indices = [], 
                                                  measurement_models = evaluation_data.measurement_models,
                                                  **filter_public_attributes(solver_options))
        fit_func = sammy_functions.run_sammy_YW

    elif solver == "EXT":

        if evaluation_data.experimental_models_no_pup is None:
            raise ValueError("experiments with no PUP must be specified")

        sammyINP = sammy_classes.SammyInputDataEXT(particle_pair, pd.DataFrame(), 
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
                                                #    V_projection = V_projection,
                                                   **filter_public_attributes(solver_options))
        fit_func = run_sammy_EXT
    else:
        raise ValueError("Solver not recognized")
    
    return Solver(rto, sammyINP,fit_func)





