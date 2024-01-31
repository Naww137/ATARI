from syndat.old.control import syndat
from ATARI.models.Y_reduction_rpi import yield_rpi
from ATARI.models.T_reduction_rpi import transmission_rpi
from typing import Protocol
from pandas import DataFrame


from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e
from ATARI.models.experimental_model import Experimental_Model


class reduction_parameters(Protocol):
    ...


class generation_model(Protocol):

    def generate_raw_data(self, pw_true, neutron_spectrum, options) -> DataFrame:
        ...

    @property
    def reduction_parameters(self) -> reduction_parameters:
        ...


class reduction_model(Protocol):

    def reduce(self, raw_data, neutron_spectrum, options) -> DataFrame:
        ...

    @property
    def reduction_parameters(self) -> reduction_parameters:
        ...







class syndat:

    def __init__(self, 
                 options={}
                 ):

        default_options = { 'Sample Counting Noise' : True, 
                    'Sample TURP'           : True,
                    'Sample TNCS'           : True, 
                    'Smooth TNCS'           : False,
                    'Calculate Covariance'  : True,
                    'Compression Points'    : [],
                    'Grouping Factors'      : None, 
                    } 
        
        self.options = update_dict(default_options, options)


    def run(self, 
            pw_true: DataFrame, 
            generation_model: generation_model, 
            reduction_model: reduction_model,
            experimental_model: experimental_model,
            neutron_spectrum= DataFrame()):

        ### define raw data attribute as true_df
        pw_true["tof"] = e_to_t(pw_true.E, experimental_model.parameters["FP"][0], True)*1e6+experimental_model.parameters["t0"][0]
        self.raw_data = pw_true


        ### if no open spectra supplied, approximate it         #!!! Should I use true reduction parameters here?
        if neutron_spectrum.empty:          
            self.neutron_spectrum = approximate_neutron_spectrum_Li6det(pw_true.E, 
                                                                        self.options["Smooth TNCS"], 
                                                                        experimental_model.parameters["FP"][0],
                                                                        experimental_model.parameters["t0"][0],
                                                                        reduction_model.reduction_parameters.trigo[0])
        else:
            self.neutron_spectrum = neutron_spectrum
        
        ### sample a realization of the true, true-underlying open count spectra
        if self.options["Sample TNCS"]:
            self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
        else:
            self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)


        ### generate raw count data for sample in given theoretical transmission and assumed true reduction parameters/open count data
        self.raw_data = generation_model.generate_raw_data(pw_true, self.true_neutron_spectrum, self.options)

        ### reduce the raw count data with the estimatec reduction model
        self.data = reduction_model.reduce(self.raw_data, self.neutron_spectrum, self.options)