
# class SyndatOptions:
#     def __init__(self):


from copy import deepcopy
from ATARI.sammy_interface import sammy_classes, sammy_functions
import pandas as pd
from new_classes import GenerativeModel


from ATARI.models.Y_reduction_rpi import yield_rpi
from ATARI.models.T_reduction_rpi import transmission_rpi
from typing import Protocol
from pandas import DataFrame


from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e

from ATARI.models.experimental_model import Experimental_Model
from ATARI.models.particle_pair import Particle_Pair



class syndatOUT:
    def __init__(self,
                 pw_reduced,
                 pw_raw):
        
        self.pw_reduced = pw_reduced
        self.pw_raw = pw_raw
    
    @property
    def pw_raw(self):
        return self._pw_raw
    @pw_raw.setter
    def pw_raw(self, pw_raw):
        self._pw_raw = pw_raw

    @property
    def pw_reduced(self):
        return self._pw_reduced
    @pw_reduced.setter
    def pw_reduced(self, pw_reduced):
        self._pw_reduced = pw_reduced




class syndatOPT:
    def __init__(self, **kwargs):
        self._sampleRES = True
        self._sample_counting_noise = True
        self._calculate_covariance = False
        self._sampleTURP = True
        self._sampleTNCS = True
        self._smoothTNCS = False

        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @property
    def sampleTURP(self):
        return self._sampleTURP
    @sampleTURP.setter
    def sampleTURP(self, sampleTURP):
        self._sampleTURP = sampleTURP

    @property
    def sampleTNCS(self):
        return self._sampleTNCS
    @sampleTNCS.setter
    def sampleTNCS(self, sampleTNCS):
        self._sampleTNCS = sampleTNCS

    @property
    def smoothTNCS(self):
        return self._smoothTNCS
    @smoothTNCS.setter
    def smoothTNCS(self, smoothTNCS):
        self._smoothTNCS = smoothTNCS

    @property
    def sample_counting_noise(self):
        return self._sample_counting_noise
    @sample_counting_noise.setter
    def sample_counting_noise(self, sample_counting_noise):
        self._sample_counting_noise = sample_counting_noise

    @property
    def calculate_covariance(self):
        return self._calculate_covariance
    @calculate_covariance.setter
    def calculate_covariance(self, calculate_covariance):
        self._calculate_covariance = calculate_covariance





class ReductiveModel(Protocol):
    def reduce(self, raw_data, neutron_spectrum, options) -> pd.DataFrame:
        ...




class syndat:

    def __init__(self, 
                 generative_model: GenerativeModel,
                 reductive_model: ReductiveModel,
                 options: syndatOPT,
                 neutron_spectrum: pd.DataFrame = pd.DataFrame()
                 ):
        
        ### user supplied options
        self.generative_model = generative_model
        self.reductive_model = reductive_model
        self.options = options
        self.neutron_spectrum = neutron_spectrum

        ### some conveinient definitions
        self.reaction = self.generative_model.experimental_model.reaction
        self.pw_true = pd.DataFrame()


    def sample(self, 
               sammyRTO
               ):

        ### generate pointwise true from experimental model
        if self.pw_true.empty or False: #options.sample_experimental_model
            # calculate pw_truw with sammy
            self.generate_true_experimental_objects(sammyRTO)
        else:
            # use existing pw_true from experimental model
            pass 

        ### generate raw data from generative reduction model
        self.generate_raw_observables(self.neutron_spectrum)

        ### reduce raw data with reductive reduction model 
        self.reduce_raw_observables()

        return syndatOUT(self.red_data, self.raw_data)



    def generate_true_experimental_objects(self, 
                                           sammyRTO:sammy_classes.SammyRunTimeOptions
                                           ):
        rto = deepcopy(sammyRTO)
        rto.bayes = False
        template = self.generative_model.experimental_model.template

        sammyINP = sammy_classes.SammyInputData(
            self.generative_model.particle_pair,
            self.generative_model.particle_pair.resonance_ladder,
            template= template,
            experiment= self.generative_model.experimental_model,
            energy_grid= self.generative_model.experimental_model.energy_grid
        )
        sammyOUT = sammy_functions.run_sammy(sammyINP, rto)

        if self.reaction == "transmission":
            true = "theo_trans"
        else:
            true = "theo_xs"
        pw_true = sammyOUT.pw.loc[:, ["E", true]]
        pw_true.rename(columns={true: "true"}, inplace=True)
        pw_true["tof"] = e_to_t(pw_true.E.values, self.generative_model.experimental_model.FP[0], True)*1e9+self.generative_model.experimental_model.t0[0]
        
        self.pw_true = pw_true


    
    
    def generate_raw_observables(self, neutron_spectrum):

        ### define raw data attribute as true_df
        self.raw_data = deepcopy(self.pw_true)

        ### if no open spectra supplied, approximate it         #!!! Should I use true reduction parameters here?
        if neutron_spectrum.empty:
            self.neutron_spectrum = approximate_neutron_spectrum_Li6det(self.pw_true.E, 
                                                                        self.options.smoothTNCS, 
                                                                        self.generative_model.experimental_model.FP[0],
                                                                        self.generative_model.experimental_model.t0[0],
                                                                        self.generative_model.reduction_model.neutron_spectrum_triggers)
        else:
            self.neutron_spectrum = neutron_spectrum
        
        ### sample a realization of the true, true-underlying open count spectra
        if self.options.sampleTNCS:
            self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
        else:
            self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)

        ### generate raw count data from generative reduction model
        self.raw_data = self.generative_model.reduction_model.generate_raw_data(self.pw_true, self.true_neutron_spectrum, self.options)

        return
    

    def reduce_raw_observables(self):
        self.red_data = self.reductive_model.reduce(self.raw_data, self.neutron_spectrum, self.options)
        return





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



