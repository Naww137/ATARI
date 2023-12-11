from copy import deepcopy
import pandas as pd

from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t 
import ATARI.utils.hdf5 as h5io

from ATARI.Models.particle_pair import Particle_Pair
from typing import Optional

from ATARI.syndat.syndat_model import Syndat_Model
from ATARI.syndat.data_classes import syndatOPT, syndatOUT


class Syndat_Control:
    """
    Syndat control module for generating synthetic data samples.
    """
    

    def __init__(self, 
                 particle_pair: Particle_Pair,
                 syndat_models: list[Syndat_Model],
                 model_correlations, 
                 options: syndatOPT
                 ):
        
        ### user supplied options
        self.particle_pair = particle_pair
        self.syndat_models = syndat_models
        self.options = options

        # self.clear_samples()


    # @property
    # def samples(self) -> list:
    #     return self._samples
    # @samples.setter
    # def samples(self, samples):
    #     self._samples = samples
    # def clear_samples(self):
    #     self.samples = []  
    
    
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
        self.pw_true_list = pw_true_list


        for i in range(num_samples):
            
            ### sample resonance ladder
            if self.options.sampleRES:
                self.particle_pair.sample_resonance_ladder()
                par_true = self.particle_pair.resonance_ladder
            
            ### sample correlated model parameters - need to pass to generate_true_exp and generate_true_raw_obs
            true_parameters = {}

            ### generate true experimental objects with sammy or just take pw_true arguement
            pw_true_list = []
            for syn_mod in self.syndat_models:
                pw_true = syn_mod.generate_true_experimental_objects(self.particle_pair, 
                                                                     sammyRTO, 
                                                                     generate_pw_true_with_sammy)
                pw_true_list.append(pw_true)


            ### generate raw datasets from samples model parameters
            for syn_mod, pw_true in zip(self.syndat_models, pw_true_list):
                syn_mod.generate_raw_observables(pw_true, true_parameters)


            ### reduce raw data with reductive reduction model 
                ### Perhaps I could sample correlated reductive parameters?
            for syn_mod in self.syndat_models:
                syn_mod.reduce_raw_observables()


            ### for each model, append syndat out
            # sample_dict = {}
            for syn_mod in self.syndat_models:

                if syn_mod.options.save_raw_data:
                    out = syndatOUT(par_true=par_true,
                                    pw_reduced=syn_mod.red_data, 
                                    pw_raw=syn_mod.raw_data)
                else:
                    out = syndatOUT(par_true=par_true,
                                    pw_reduced=syn_mod.red_data)

                if syn_mod.options.calculate_covariance:
                    out.covariance_data = syn_mod.covariance_data
                    
                syn_mod.samples.append(out)
                # sample_dict[syn_mod.title] = out
            
            # self.samples.append(sample_dict)

        return
    


    # def to_hdf5(self, filepath):

    #     for isample, sample_dict in enumerate(self.samples):

    #         assert all(sample_dict[self.titles[0]].par_true.equals(sample_dict[t].par_true for t in self.titles))
    #         h5io.write_par(filepath, isample, sample_dict[self.titles[0]].par_true, 'true')

    #         for syn_mod in self.syndat_models:
    #             # syn_mod.to_hdf5(filepath)
    #             h5io.write_pw_exp(filepath, isample, sample_dict[t].pw_reduced, title=t, CovT=None, CovXS=None)




# class syndat:
#     """
#     Syndat model for a single experimental dataset that can be used to generate similar statistical realizations.
#     This class holds all the necessary subclasses for the reaction model, experimental model, and reduction model.
    
#     Parameters
#     ----------
#     particle_pair: Particle_Pair
#         Class describing the reaction model.
#     experimental_model: Experimental_Model
#         Class describing the experimental model, used with the SAMMY code.
#     generative_measurement_model: Generative_Measurement_Model
#         Class describing the measurement model used for data generation.
#     reductive_measurement_model: Reductive_Measurement_Model
#         Class describing the measurement model used for data reduction.
#     options: syndatOPT
#         Syndat options class, providing information about how to run syndat.
#     neutron_spectrum: pd.DataFrame = pd.DataFrame()
#         Optional input of a measured, experimental neutron spectrum if this data is accessable.

#     Attributes
#     ----------
#     particle_pair: Particle_Pair
#         Class describing the reaction model.
#     experimental_model: Experimental_Model
#         Class describing the experimental model, used with the SAMMY code.
#     generative_measurement_model: Generative_Measurement_Model
#         Class describing the measurement model used for data generation.
#     reductive_measurement_model: Reductive_Measurement_Model
#         Class describing the measurement model used for data reduction.
#     options: syndatOPT
#         Syndat options class, providing information about how to run syndat.
#     neutron_spectrum: pd.DataFrame = pd.DataFrame()
#         Optional input of a measured, experimental neutron spectrum if this data is accessable.
    
#     datasets: list
#         List of sampled syndatOUT samples from the current model.
    
#     """

#     def __init__(self, 
#                  particle_pair: Particle_Pair,
#                  experimental_model: Experimental_Model,
#                  generative_measurement_model: Generative_Measurement_Model,
#                  reductive_measurement_model: Reductive_Measurement_Model,
#                  options: syndatOPT,
#                  neutron_spectrum: pd.DataFrame = pd.DataFrame()
#                  ):
        
#         ### user supplied options
#         self.particle_pair = particle_pair
#         self.experimental_model = experimental_model
#         self.generative_measurement_model = generative_measurement_model
#         self.reductive_measurement_model = reductive_measurement_model
    
#         self.options = options
#         self.neutron_spectrum = neutron_spectrum

#         ### some conveinient definitions
#         self.reaction = self.experimental_model.reaction
#         self.pw_true = pd.DataFrame()
#         self.datasets = []

    
#     @property
#     def datasets(self):
#         return self._datasets
#     @datasets.setter
#     def datasets(self, datasets):
#         self._datasets = datasets

    # def sample(self, 
    #            sammyRTO=None,
    #            num_samples=1,
    #            pw_true = pd.DataFrame()
    #            ):
    #     """
    #     Sample from the syndat model. 
    #     The samples are stored in the datasets attribute.

    #     Parameters
    #     ----------
    #     sammyRTO : _type_
    #         Sammy runtime options
    #     num_samples : int, optional
    #         Number of samples to draw from the syndat model, by default 1
    #     pw_true : _type_, optional
    #         _description_, by default pd.DataFrame()

    #     Raises
    #     ------
    #     ValueError
    #         _description_
    #     """
        
    #     if not pw_true.empty and self.options.sampleRES:
    #             raise ValueError("User provided a pw_true but also asked to sampleRES")
    #     if pw_true.empty and sammyRTO is None:
    #         raise ValueError("User did not supply a sammyRTO or a pw_true, one of these is needed")
    #     self.pw_true = pw_true

    #     datasets = []
    #     for i in range(num_samples):
            
    #         if self.options.sampleRES:
    #             self.particle_pair.sample_resonance_ladder(self.experimental_model.energy_range)
            
    #         ### generate pointwise true from experimental model
    #         if self.pw_true.empty or False or self.options.sampleRES: #options.sample_experimental_model
    #             assert sammyRTO is not None
    #             self.pw_true = self.generate_true_experimental_objects(sammyRTO) # calculate pw_truw with sammy
    #         else:
    #             pass # use existing pw_true from experimental model
    #         self.pw_true["tof"] = e_to_t(self.pw_true.E.values, self.experimental_model.FP[0], True)*1e9+self.experimental_model.t0[0]

    #         ### generate raw data from generative reduction model
    #         self.generate_raw_observables(self.neutron_spectrum)
    #         ### reduce raw data with reductive reduction model 
    #         self.reduce_raw_observables()

    #         if self.options.save_raw_data:
    #             out = syndatOUT(pw_reduced=self.red_data, 
    #                             pw_raw = self.raw_data)
    #         else:
    #             out = syndatOUT(pw_reduced=self.red_data)
    #         if self.options.calculate_covariance:
    #             out.covariance_data = self.covariance_data
                
    #         datasets.append(out)

    #     self.datasets = datasets
    #     return


    # def tohdf5(self):
    #     return



    # def generate_true_experimental_objects(self, 
    #                                        sammyRTO:sammy_classes.SammyRunTimeOptions
    #                                        ):
    #     rto = deepcopy(sammyRTO)
    #     rto.bayes = False
    #     template = self.experimental_model.template

    #     if template is None: raise ValueError("Experimental model sammy template has not been assigned")

    #     sammyINP = sammy_classes.SammyInputData(
    #         self.particle_pair,
    #         self.particle_pair.resonance_ladder,
    #         template= template,
    #         experiment= self.experimental_model,
    #         energy_grid= self.experimental_model.energy_grid
    #     )
    #     sammyOUT = sammy_functions.run_sammy(sammyINP, rto)

    #     if self.reaction == "transmission":
    #         true = "theo_trans"
    #     else:
    #         true = "theo_xs"
    #     pw_true = sammyOUT.pw.loc[:, ["E", true]]
    #     pw_true.rename(columns={true: "true"}, inplace=True)
    #     # pw_true["tof"] = e_to_t(pw_true.E.values, self.experimental_model.FP[0], True)*1e9+self.experimental_model.t0[0]
        
    #     return pw_true


    
    
    # def generate_raw_observables(self, neutron_spectrum):

    #     ### define raw data attribute as true_df
    #     self.raw_data = deepcopy(self.pw_true)

    #     ### if no open spectra supplied, approximate it         #!!! Should I use true reduction parameters here?
    #     if neutron_spectrum.empty:
    #         self.neutron_spectrum = approximate_neutron_spectrum_Li6det(self.pw_true.E, 
    #                                                                     self.options.smoothTNCS, 
    #                                                                     self.experimental_model.FP[0],
    #                                                                     self.experimental_model.t0[0],
    #                                                                     self.generative_measurement_model.neutron_spectrum_triggers)
    #     else:
    #         self.neutron_spectrum = neutron_spectrum
        
    #     ### sample a realization of the true, true-underlying open count spectra
    #     if self.options.sampleTNCS:
    #         self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
    #     else:
    #         self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)

    #     ### generate raw count data from generative reduction model
    #     self.raw_data = self.generative_measurement_model.generate_raw_data(self.pw_true, self.true_neutron_spectrum, self.options)

    #     return
    

    # def reduce_raw_observables(self):
    #     self.red_data = self.reductive_measurement_model.reduce_raw_data(self.raw_data, self.neutron_spectrum, self.options)
    #     self.covariance_data = self.reductive_measurement_model.covariance_data
    #     return





# # ================================================================
# #  testing!
# # ================================================================



# from ATARI.models.particle_pair import Particle_Pair
# from ATARI.sammy_interface import sammy_classes, sammy_functions, template_creator
# import os 
# from copy import copy

# ### define reaction model
# Ta_pair = Particle_Pair()  
# Ta_pair.add_spin_group(Jpi='3.0',
#                        J_ID=1,
#                        D_avg=8.79,
#                        Gn_avg=46.5,
#                        Gn_dof=1,
#                        Gg_avg=64.0,
#                        Gg_dof=1000)

# # energy_range = [200, 250]
# # resonance_ladder = Ta_pair.sample_resonance_ladder(energy_range)

# ### define experimental model
# exp_model_T = Experimental_Model()
# generation_T1 = transmission_rpi()
# generative_model = GenerativeModel(Ta_pair, exp_model_T, generation_T1)

# rto = sammy_classes.SammyRunTimeOptions('/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
#                                         {"Print":   True,
#                                          "bayes":   False,
#                                          "keep_runDIR": False,
#                                          "sammy_runDIR": "/Users/noahwalton/Documents/GitHub/ATARI/ATARI/syndat/sammy_runDIR"
#                                          })

# template_creator.make_input_template('/Users/noahwalton/Documents/GitHub/ATARI/ATARI/syndat/template_T.inp', Ta_pair, exp_model_T, rto)
# exp_model_T.template = '/Users/noahwalton/Documents/GitHub/ATARI/ATARI/syndat/template_T.inp'



# reduction_T1 = transmission_rpi()

# options=syndatOPT()
# synT = syndat(generative_model, reductive_model=reduction_T1, options=options)

# test = synT.sample(rto)

# print(test.pw_raw)
# print(test.pw_reduced)



