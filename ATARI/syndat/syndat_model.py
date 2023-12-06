
import pandas as pd
from ATARI.models.experimental_model import Experimental_Model
from ATARI.models.particle_pair import Particle_Pair
from typing import Protocol, Optional
from copy import deepcopy

from ATARI.syndat.general_functions import *
from ATARI.syndat.data_classes import syndatOPT, syndatOUT

from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.models.structuring import Generative_Measurement_Model, Reductive_Measurement_Model
from ATARI.models.measurement_models.transmission_rpi import Transmission_RPI


class Syndat_Model:
    """
    Syndat model for a single experimental dataset that can be used to generate similar statistical realizations.
    This class holds all the necessary subclasses for the reaction model, experimental model, and reduction model.
    
    Parameters
    ----------
    particle_pair: Particle_Pair
        Class describing the reaction model.
    experimental_model: Experimental_Model
        Class describing the experimental model, used with the SAMMY code.
    generative_measurement_model: Generative_Measurement_Model
        Class describing the measurement model used for data generation.
    reductive_measurement_model: Reductive_Measurement_Model
        Class describing the measurement model used for data reduction.
    options: syndatOPT
        Syndat options class, providing information about how to run syndat.
    neutron_spectrum: pd.DataFrame = pd.DataFrame()
        Optional input of a measured, experimental neutron spectrum if this data is accessable.

    Attributes
    ----------
    particle_pair: Particle_Pair
        Class describing the reaction model.
    experimental_model: Experimental_Model
        Class describing the experimental model, used with the SAMMY code.
    generative_measurement_model: Generative_Measurement_Model
        Class describing the measurement model used for data generation.
    reductive_measurement_model: Reductive_Measurement_Model
        Class describing the measurement model used for data reduction.
    options: syndatOPT
        Syndat options class, providing information about how to run syndat.
    neutron_spectrum: pd.DataFrame = pd.DataFrame()
        Optional input of a measured, experimental neutron spectrum if this data is accessable.
    
    datasets: list
        List of sampled syndatOUT samples from the current model.
    
    """

    def __init__(self,
                 generative_experimental_model: Optional[Experimental_Model] = None,
                 generative_measurement_model: Optional[Generative_Measurement_Model] = None,
                 reductive_measurement_model: Optional[Reductive_Measurement_Model] = None,
                 options: Optional[syndatOPT] = None
                 ):

        self.generative_experimental_model = Experimental_Model()
        self.generative_measurement_model = Transmission_RPI()  # Generative_Reduction_Model()
        self.reductive_measurement_model = Transmission_RPI()
        self.options = syndatOPT()

        if generative_experimental_model is not None:
            self.generative_experimental_model = generative_experimental_model
        if generative_measurement_model is not None:
            self.generative_measurement_model = generative_measurement_model
        if reductive_measurement_model is not None:
            self.reductive_measurement_model = reductive_measurement_model
        if options is not None:
            self.options = options

        ### some convenient definitions
        self.reaction = self.generative_experimental_model.reaction
        self.pw_true = pd.DataFrame()
        self.datasets = []

        ### approximate neutron spectrum if none given
        # if self.generative_measurement_model.model_parameters.neutron_spectrum is None:
        #     neutron_spectrum = approximate_neutron_spectrum_Li6det(self.generative_experimental_model.energy_grid, 
        #                                                             False, #self.options.smoothTNCS, 
        #                                                             self.generative_experimental_model.FP[0],
        #                                                             self.generative_experimental_model.t0[0],
        #                                                             self.generative_measurement_model.neutron_spectrum_triggers)
            
        #     self.generative_measurement_model.model_parameters.neutron_spectrum = neutron_spectrum
        #     self.reductive_measurement_model.model_parameters.neutron_spectrum = neutron_spectrum
        self.generative_measurement_model.approximate_unknown_data(self.generative_experimental_model)
        self.reductive_measurement_model.approximate_unknown_data(self.generative_experimental_model)
        

    @property
    def datasets(self):
        return self._datasets
    @datasets.setter
    def datasets(self, datasets):
        self._datasets = datasets

    @property
    def generative_experimental_model(self):
        return self._generative_experimental_model
    @generative_experimental_model.setter
    def generative_experimental_model(self, generative_experimental_model):
        self._generative_experimental_model = generative_experimental_model

    @property
    def generative_measurement_model(self):
        return self._generative_measurement_model
    @generative_measurement_model.setter
    def generative_measurement_model(self, generative_measurement_model):
        self._generative_measurement_model = generative_measurement_model

    @property
    def reductive_measurement_model(self):
        return self._reductive_measurement_model
    @reductive_measurement_model.setter
    def reductive_measurement_model(self, reductive_measurement_model):
        self._reductive_measurement_model = reductive_measurement_model


    def sample(self,
               particle_pair: Optional[Particle_Pair] = None,
               sammyRTO=None,
               pw_true: Optional[pd.DataFrame] = None,
               num_samples=1
               ):
        
        generate_pw_true_with_sammy = False
        if pw_true is not None:
            if self.options.sampleRES:
                raise ValueError("User provided a pw_true but also asked to sampleRES")
            
        if pw_true is None:
            generate_pw_true_with_sammy = True
            if sammyRTO is None:
                raise ValueError("User did not supply a sammyRTO or a pw_true, one of these is needed")
        self.pw_true_list = pw_true

        datasets = []
        for i in range(num_samples):
            
            ### sample resonance ladder
            if self.options.sampleRES:
                assert particle_pair is not None
                particle_pair.sample_resonance_ladder()
           
            pw_true = self.generate_true_experimental_objects(particle_pair, 
                                                    sammyRTO, 
                                                    generate_pw_true_with_sammy)

            self.generate_raw_observables(pw_true, true_model_parameters={})

            self.reduce_raw_observables()

            if self.options.save_raw_data:
                out = syndatOUT(pw_reduced=self.red_data, pw_raw=self.raw_data)
            else:
                out = syndatOUT(pw_reduced=self.red_data)
            if self.options.calculate_covariance:
                out.covariance_data = self.covariance_data

            datasets.append(out)

        self.datasets = datasets

    def to_hdf5(self, filepath):
        pass

    def generate_raw_observables(self, pw_true, true_model_parameters):

        ### define raw data attribute as true_df
        # self.raw_data = deepcopy(pw_true)

        # if not in true_model_parameters, sample uncorrelated true_model_parameter
        true_model_parameters = self.generative_measurement_model.sample_true_model_parameters(true_model_parameters)
        
        # ### sample a realization of the true, true-underlying open count spectra
        # if self.options.sampleTNCS:
        #     self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
        # else:
        #     self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)


        ### generate raw count data from generative reduction model
        self.raw_data = self.generative_measurement_model.generate_raw_data(pw_true, 
                                                                            true_model_parameters, 
                                                                            self.options)

        
    

    def reduce_raw_observables(self):
        self.red_data = self.reductive_measurement_model.reduce_raw_data(self.raw_data, 
                                                                         self.reductive_measurement_model.model_parameters.neutron_spectrum, 
                                                                         self.options)
        self.covariance_data = self.reductive_measurement_model.covariance_data
        
    


    def generate_true_experimental_objects(self,
                                           particle_pair: Optional[Particle_Pair],
                                           sammyRTO: Optional[sammy_classes.SammyRunTimeOptions],
                                           generate_pw_true_with_sammy: bool,
                                           pw_true: Optional[pd.DataFrame] = None,
                                           ):
        
        if generate_pw_true_with_sammy:
            assert sammyRTO is not None
            assert particle_pair is not None

            rto = deepcopy(sammyRTO)
            rto.bayes = False
            template = self.generative_experimental_model.template

            if template is None: raise ValueError("Experimental model sammy template has not been assigned")

            sammyINP = sammy_classes.SammyInputData(
                particle_pair,
                particle_pair.resonance_ladder,
                template= template,
                experiment= self.generative_experimental_model,
                energy_grid= self.generative_experimental_model.energy_grid
            )
            sammyOUT = sammy_functions.run_sammy(sammyINP, rto)

            if self.reaction == "transmission":
                true = "theo_trans"
            else:
                true = "theo_xs"
            assert isinstance(sammyOUT.pw, pd.DataFrame)
            pw_true = sammyOUT.pw.loc[:, ["E", true]]
            pw_true.rename(columns={true: "true"}, inplace=True)

        else:
            assert pw_true is not None

            pw_true = pw_true
        
        pw_true["tof"] = e_to_t(pw_true.E.values, self.generative_experimental_model.FP[0], True)*1e9+self.generative_experimental_model.t0[0]
        
        return pw_true
    

