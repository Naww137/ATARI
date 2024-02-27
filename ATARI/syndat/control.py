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
                 model_correlations: dict = {}, 
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
               pw_true_list: Optional[list[pd.DataFrame]] = None
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


        for i in range(num_samples):
            
            ### sample resonance ladder
            if self.options.sampleRES:
                self.particle_pair.sample_resonance_ladder()
                par_true = self.particle_pair.resonance_ladder
            
            ### sample correlated model parameters - need to pass to generate_true_experimental_objects and generate_true_raw_obs
            true_parameters = self.sample_model_correlations()

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
            for syn_mod, pw_true in zip(self.syndat_models, pw_true_list):
                raw_data = syn_mod.generate_raw_observables(pw_true, 
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
                    out = syndatOUT(par_true=par_true,
                                    pw_reduced=reduced_data_list[i], 
                                    pw_raw=raw_data_list[i])
                else:
                    out = syndatOUT(par_true=par_true,
                                    pw_reduced=reduced_data_list[i])

                if syn_mod.options.calculate_covariance:
                    out.covariance_data = covariance_data_list[i]
                    
                syn_mod.samples.append(out)
                # sample_dict[syn_mod.title] = out
            
            # self.samples.append(sample_dict)

        return
    

    def sample_model_correlations(self):
        true_parameters = {}
        return true_parameters

