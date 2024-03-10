from copy import deepcopy
import pandas as pd

from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t 
import ATARI.utils.hdf5 as h5io

from ATARI.ModelData.particle_pair import Particle_Pair
from typing import Optional

from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.syndat.data_classes import syndatOPT, syndatOUT


class Model_Correlations:

    def __init__(self):
        pass



class Syndat_Control:
    """
    Syndat Control module for sampling from multiple syndat models.

    _extended_summary_

    Parameters
    ----------
    particle_pair: Particle_Pair
        ATARI class describing the reaction model.
    syndat_models: list[Syndat_Model]
        list of individual Syndat_Model classes
    model_correlations: dict
        Dictionary of uncertain parameters that correlate individual Syndat_Models. 
        Format is the same as other model parameters: key:(val, unc).
        If supplied, this parameter defined by key will overwrite individual Syndat_Model parameters of the same key.
    options: syndatOPT
        Syndat Options object, only option that will be used is SampleRES.
        Otherwise, individual syndat_models have their own options.
    """
    

    def __init__(self, 
                 particle_pair: Particle_Pair,
                 syndat_models: list[Syndat_Model],
                 model_correlations: list = [], 
                 options: syndatOPT = syndatOPT()
                 ):
        
        ### user supplied options
        self.particle_pair = particle_pair
        self.syndat_models = syndat_models
        self.model_correlations = model_correlations
        self.options = deepcopy(options)

        # self.clear_samples()


    # @property
    # def samples(self) -> list:
    #     return self._samples
    # @samples.setter
    # def samples(self, samples):
    #     self._samples = samples
    # def clear_samples(self):
    #     self.samples = []  
    
    # def clear_samples(self):
    #     for syn_mod in self.sy
    #     return
    
    def get_sample(self,i):
        data = {}
        for syn_mod in self.syndat_models:
            data[syn_mod.title] = syn_mod.samples[i]
        return data
    
    # @samples.setter
    # def samples(self, samples):
    #     self._samples = samples
    # def clear_samples(self):
    #     self.samples = []  

    @property
    def titles(self) -> list:
        return [syn_mod.title for syn_mod in self.syndat_models]



    def sample(self, 
               sammyRTO=None,
               num_samples=1,
               pw_true_list: Optional[list[pd.DataFrame]] = None,
               save_samples_to_hdf5 = False,
               hdf5_file = None
               ):

        generate_pw_true_with_sammy = False
        par_true = None
        if pw_true_list is not None:
            if self.options.sampleRES:
                raise ValueError("User provided a pw_true but also asked to sampleRES")
            if len(pw_true_list) != len(self.syndat_models):
                raise ValueError("User provided a pw_true list not of the same length as syndat_models")
        if pw_true_list is None:
            generate_pw_true_with_sammy = True
            if sammyRTO is None:
                raise ValueError("User did not supply a sammyRTO or a pw_true, one of these is needed")
        par_true = self.particle_pair.resonance_ladder
        self.pw_true_list = deepcopy(pw_true_list)


        for isample in range(num_samples):
            
            ### sample resonance ladder
            if self.options.sampleRES:
                self.particle_pair.sample_resonance_ladder()
                par_true = self.particle_pair.resonance_ladder
            
            ### sample correlated model parameters - need to pass to generate_true_experimental_objects and generate_true_raw_obs
            sampled_parameter_correlations = self.sample_model_correlations()

            ### generate true experimental objects with sammy or just take pw_true arguement
            pw_true_list = []
            for i, syn_mod in enumerate(self.syndat_models):
                if self.pw_true_list is None: pwtruelist = None
                else: pwtruelist = self.pw_true_list[i]
                pw_true = syn_mod.generate_true_experimental_objects(self.particle_pair, 
                                                                     sammyRTO, 
                                                                     generate_pw_true_with_sammy,
                                                                     pwtruelist)
                pw_true_list.append(pw_true)


            ### generate raw datasets from samples model parameters
            gen_raw_data_list = []
            for i, syn_mod in enumerate(self.syndat_models):
                true_parameters = {}
                for each in sampled_parameter_correlations:
                        for key,val in each.items():
                            if key == "models":
                                if val[i] == 0: # skip if not flagged as correlated
                                    update = False
                                elif val[i] == 1: # skip if not flagged as correlated
                                    update = True
                            else:
                                if update:
                                    true_parameters.update({key:val})
                                else:
                                    pass
                        
                raw_data = syn_mod.generate_raw_observables(pw_true_list[i], 
                                                            true_parameters)
                gen_raw_data_list.append(raw_data)


            ### reduce raw data with reductive reduction model 
                # Perhaps I could sample correlated reductive parameters?
            reduced_data_list, covariance_data_list, raw_data_list = [], [], []
            for i, syn_mod in enumerate(self.syndat_models):
                reduced_data, covariance_data, raw_data = syn_mod.reduce_raw_observables(gen_raw_data_list[i])
                reduced_data_list.append(reduced_data)
                covariance_data_list.append(covariance_data)
                raw_data_list.append(raw_data)

            ### for each model, append syndat out
            # sample_dict = {}
            for i, syn_mod in enumerate(self.syndat_models):

                if syn_mod.options.save_raw_data:
                    out = syndatOUT(title = syn_mod.title,
                                    par_true=par_true,
                                    pw_reduced=reduced_data_list[i], 
                                    pw_raw=raw_data_list[i])
                else:
                    out = syndatOUT(title = syn_mod.title,
                                    par_true=par_true,
                                    pw_reduced=reduced_data_list[i])

                if syn_mod.options.calculate_covariance:
                    out.covariance_data = covariance_data_list[i]
                
                if save_samples_to_hdf5:
                    if hdf5_file is None:
                        raise ValueError("If save_samples_to_hdf5, please provide an hdf5 file.")
                    out.to_hdf5(hdf5_file, isample)
                else:
                    syn_mod.samples.append(out)

        return
    


    def sample_model_correlations(self):

        sampled_model_correlations = []

        for each in self.model_correlations:

            sampled = False
            sampled_dict = {'models':each['models']}
            for corr_bool, syn_mod in zip(each['models'], self.syndat_models):
        
                # do nothing if not flagged as correlated model
                if corr_bool == 0:
                    pass

                # if flagged, do stuff
                else:
                    # if a sample has already been taken, check to make sure the flagged model has the appropriate parameters
                    if sampled:
                        if isinstance(syn_mod.generative_measurement_model,instance_check):
                            pass
                        else:
                            raise ValueError("You have flagged correlations between two different measurement model types, this capability has not yet been implemented")
                        
                        for key,val in sampled_dict.items():
                            if key == 'models':
                                pass
                            else:
                                if key not in syn_mod.generative_measurement_model.model_parameters.__dict__.keys():
                                    raise ValueError(f"Assigned a correlated parameter to a syndat model {syn_mod.title} that does not have that parameter")

                    # if a sample has not yet been taken, draw a sample
                    else:
                        instance_check = type(syn_mod.generative_measurement_model)
                        sampled = True
                        # sampled_parameters = syn_mod.generative_measurement_model.model_parameters.sample_parameters({})
                        for param_name, param_values in each.items():
                            if param_name == 'models':
                                pass
                            else: 
                                if isinstance(param_values, tuple) and len(param_values) == 2:
                                    mean, uncertainty = param_values
                                    if np.all(np.array(uncertainty) == 0):
                                        sample = mean
                                    else:
                                        if param_name == 'a_b':
                                            sample = np.random.multivariate_normal(mean, uncertainty)
                                        else:
                                            sample = np.random.normal(loc=mean, scale=uncertainty)
                                    sampled_dict[param_name] = (sample, 0.0)
                                if isinstance(param_values, pd.DataFrame):
                                    new_c = np.random.normal(loc=param_values.ct, scale=param_values.dct)
                                    df = deepcopy(param_values)
                                    df.loc[:,'ct'] = new_c
                                    df.loc[:,'dct'] = np.sqrt(new_c)
                                    sampled_dict[param_name] = df

            sampled_model_correlations.append(sampled_dict)
        
        return sampled_model_correlations

