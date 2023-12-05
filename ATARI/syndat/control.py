from copy import deepcopy
from ATARI.sammy_interface import sammy_classes, sammy_functions
import pandas as pd

from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t 

from ATARI.models.experimental_model import Experimental_Model
from ATARI.models.particle_pair import Particle_Pair


from ATARI.models.structuring import Generative_Measurement_Model, Reductive_Measurement_Model



class syndatOUT:
    def __init__(self, **kwargs):
        
        self._pw_reduced = None
        self._pw_raw = None
        self._covariance_data = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

        
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

    @property
    def covariance_data(self):
        return self._covariance_data
    @covariance_data.setter
    def covariance_data(self, covariance_data):
        self._covariance_data = covariance_data





class syndatOPT:
    """
    Options and settings for a single syndat case.
    
    Parameters
    ----------
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    Attributes
    ----------
    sampleRES : bool
        Sample a new resonance ladder with each sample.
    sample_counting_noise : bool = False
        Option to sample counting statistic noise for data generation, if False, no statistical noise will be sampled.
    calculate_covariance : bool = True
        Indicate whether to calculate off-diagonal elements of the data covariance matrix.
    explicit_covariance : bool = False
        Indicate whether to return explicit data covariance elements or the decomposed statistical and systematic covariances with systematic derivatives.
    sampleTURP : bool
        Option to sample true underlying measurement model (data-reduction) parameters for data generation.
    sampleTNCS : bool
        Option to sample true neutron count spectrum for data generation.
    smoothTNCS : bool
        Option to use a smoothed function for the true neutron count spectrum for data generation.
    save_raw_data : bool
        Option to save raw count data, if False, only the reduced transmission data will be saved.
    """
    def __init__(self, **kwargs):
        self._sampleRES = True
        self._sample_counting_noise = True
        self._calculate_covariance = False
        self._explicit_covariance = False
        self._sampleTURP = True
        self._sampleTNCS = True
        self._smoothTNCS = False
        self._save_raw_data = False

        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    
    @property
    def sampleRES(self):
        return self._sampleRES
    @sampleRES.setter
    def sampleRES(self, sampleRES):
        self._sampleRES = sampleRES
        
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
    
    @property
    def explicit_covariance(self):
        return self._explicit_covariance
    @explicit_covariance.setter
    def explicit_covariance(self, explicit_covariance):
        self._explicit_covariance = explicit_covariance

    @property
    def save_raw_data(self):
        return self._save_raw_data
    @save_raw_data.setter
    def save_raw_data(self, save_raw_data):
        self._save_raw_data = save_raw_data






class syndat:
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
                 particle_pair: Particle_Pair,
                 experimental_model: Experimental_Model,
                 generative_measurement_model: Generative_Measurement_Model,
                 reductive_measurement_model: Reductive_Measurement_Model,
                 options: syndatOPT,
                 neutron_spectrum: pd.DataFrame = pd.DataFrame()
                 ):
        
        ### user supplied options
        self.particle_pair = particle_pair
        self.experimental_model = experimental_model
        self.generative_measurement_model = generative_measurement_model
        self.reductive_measurement_model = reductive_measurement_model
    
        self.options = options
        self.neutron_spectrum = neutron_spectrum

        ### some conveinient definitions
        self.reaction = self.experimental_model.reaction
        self.pw_true = pd.DataFrame()
        self.datasets = []

    
    @property
    def datasets(self):
        return self._datasets
    @datasets.setter
    def datasets(self, datasets):
        self._datasets = datasets


    def sample(self, 
               sammyRTO=None,
               num_samples=1,
               pw_true = pd.DataFrame()
               ):
        """
        Sample from the syndat model. 
        The samples are stored in the datasets attribute.

        Parameters
        ----------
        sammyRTO : _type_
            Sammy runtime options
        num_samples : int, optional
            Number of samples to draw from the syndat model, by default 1
        pw_true : _type_, optional
            _description_, by default pd.DataFrame()

        Raises
        ------
        ValueError
            _description_
        """
        
        if not pw_true.empty and self.options.sampleRES:
                raise ValueError("User provided a pw_true but also asked to sampleRES")
        if pw_true.empty and sammyRTO is None:
            raise ValueError("User did not supply a sammyRTO or a pw_true, one of these is needed")
        self.pw_true = pw_true

        datasets = []
        for i in range(num_samples):
            
            if self.options.sampleRES:
                self.particle_pair.sample_resonance_ladder(self.experimental_model.energy_range)
            
            ### generate pointwise true from experimental model
            if self.pw_true.empty or False or self.options.sampleRES: #options.sample_experimental_model
                assert sammyRTO is not None
                self.pw_true = self.generate_true_experimental_objects(sammyRTO) # calculate pw_truw with sammy
            else:
                pass # use existing pw_true from experimental model
            self.pw_true["tof"] = e_to_t(self.pw_true.E.values, self.experimental_model.FP[0], True)*1e9+self.experimental_model.t0[0]

            ### generate raw data from generative reduction model
            self.generate_raw_observables(self.neutron_spectrum)
            ### reduce raw data with reductive reduction model 
            self.reduce_raw_observables()

            if self.options.save_raw_data:
                out = syndatOUT(pw_reduced=self.red_data, 
                                pw_raw = self.raw_data)
            else:
                out = syndatOUT(pw_reduced=self.red_data)
            if self.options.calculate_covariance:
                out.covariance_data = self.covariance_data
                
            datasets.append(out)

        self.datasets = datasets
        return
    

    def tohdf5(self):
        return



    def generate_true_experimental_objects(self, 
                                           sammyRTO:sammy_classes.SammyRunTimeOptions
                                           ):
        rto = deepcopy(sammyRTO)
        rto.bayes = False
        template = self.experimental_model.template

        if template is None: raise ValueError("Experimental model sammy template has not been assigned")

        sammyINP = sammy_classes.SammyInputData(
            self.particle_pair,
            self.particle_pair.resonance_ladder,
            template= template,
            experiment= self.experimental_model,
            energy_grid= self.experimental_model.energy_grid
        )
        sammyOUT = sammy_functions.run_sammy(sammyINP, rto)

        if self.reaction == "transmission":
            true = "theo_trans"
        else:
            true = "theo_xs"
        pw_true = sammyOUT.pw.loc[:, ["E", true]]
        pw_true.rename(columns={true: "true"}, inplace=True)
        # pw_true["tof"] = e_to_t(pw_true.E.values, self.experimental_model.FP[0], True)*1e9+self.experimental_model.t0[0]
        
        return pw_true


    
    
    def generate_raw_observables(self, neutron_spectrum):

        ### define raw data attribute as true_df
        self.raw_data = deepcopy(self.pw_true)

        ### if no open spectra supplied, approximate it         #!!! Should I use true reduction parameters here?
        if neutron_spectrum.empty:
            self.neutron_spectrum = approximate_neutron_spectrum_Li6det(self.pw_true.E, 
                                                                        self.options.smoothTNCS, 
                                                                        self.experimental_model.FP[0],
                                                                        self.experimental_model.t0[0],
                                                                        self.generative_measurement_model.neutron_spectrum_triggers)
        else:
            self.neutron_spectrum = neutron_spectrum
        
        ### sample a realization of the true, true-underlying open count spectra
        if self.options.sampleTNCS:
            self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
        else:
            self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)

        ### generate raw count data from generative reduction model
        self.raw_data = self.generative_measurement_model.generate_raw_data(self.pw_true, self.true_neutron_spectrum, self.options)

        return
    

    def reduce_raw_observables(self):
        self.red_data = self.reductive_measurement_model.reduce_raw_data(self.raw_data, self.neutron_spectrum, self.options)
        self.covariance_data = self.reductive_measurement_model.covariance_data
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



