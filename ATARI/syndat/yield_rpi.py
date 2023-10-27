import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e




# ========================================================================================
#            Transmission reduction equations at RPI
# ========================================================================================

def gammayield():
    return  


def get_covT():
        return 

    
def reduce_raw_count_data():
    return


def inverse_reduction():
    return







# ========================================================================================
#            syndat_T class 
# ========================================================================================

class syndat_Y:

    def __init__(self, 
                 options={}, 
                 reduction_parameters={}, 
                 ):

        default_options = { 'Sample Counting Noise' : True, 
                            'Sample TURP'           : True,
                            'Sample TNCS'           : True, 
                            'Smooth TNCS'           : False,
                            'Calculate Covariance'  : True,
                            'Compression Points'    : [],
                            'Grouping Factors'      : None, 
                            } 
        
        default_reduction_parameters = {
                            'n'         :   (0.067166,            0),
                            'trigo'     :   (9758727,             0),
                            'trigs'     :   (18476117,            0),
                            'FP'        :   (35.185,              0),
                            't0'        :   (3.326,               0),
                            'bw'        :   (0.0064,              0)
                            }

        self.options = update_dict(default_options, options)
        self.reduction_parameters = update_dict(default_reduction_parameters, reduction_parameters)


    def run(self, 
            pw_true,
            neutron_spectrum=pd.DataFrame()
            ):
        
        ### define raw data attribute as true_df
        pw_true["tof"] = e_to_t(pw_true.E, self.reduction_parameters["FP"][0], True)*1e6+self.reduction_parameters["t0"][0]
        self.raw_data = pw_true

        ### sample true underlying resonance parameters from measured values - defines self.theo_redpar
        self.true_reduction_parameters = sample_true_underlying_parameters(self.reduction_parameters, True)

        ### if no open spectra supplied, approximate it         #!!! Should I use true reduction parameters here?
        if neutron_spectrum.empty:          
            self.neutron_spectrum = approximate_neutron_spectrum_Li6det(pw_true.E, 
                                                            self.options["Smooth TNCS"], 
                                                            self.reduction_parameters["FP"][0], 
                                                            self.reduction_parameters["t0"][0], 
                                                            self.reduction_parameters["trigo"][0])
        else:
            self.neutron_spectrum = neutron_spectrum
        
        ### sample a realization of the true, true-underlying open count spectra
        if self.options["Sample TNCS"]:
            self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
        else:
            self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)

        ### generate raw count data for sample in given theoretical transmission and assumed true reduction parameters/open count data
        # self.raw_data = self.generate_raw_data(self.true_neutron_spectrum, self.true_reduction_parameters)

        ### reduce the raw count data
        # self.Yg = self.reduce(self.raw_data, self.neutron_spectrum, self.reduction_parameters)
        
        ### stupid gaussian sampling around true
        Yg = pd.DataFrame()
        Yg['tof'] = self.raw_data.tof
        Yg['E'] = self.raw_data.E
        Yg['true'] = self.raw_data.true
        Yg_std = (np.sqrt(Yg['true'])+1) * 0.05
        Yg["exp"] = abs(np.random.default_rng().normal(Yg['true'],  Yg_std))
        Yg['exp_unc'] = Yg_std

        self.data = Yg





    def generate_raw_data(self, 
                          true_neutron_spectrum,
                          true_reduction_parameters
                          ):
        """
        Generates a set of noisy, count data from a theoretical yield via the novel inverse-reduction method (Walton, et al.).

        Parameters
        ----------
        add_noise : bool
            Whether or not to add noise to the generated sample in data.

        Raises
        ------
        ValueError
            _description_
        """

        if len(true_neutron_spectrum) != len(self.raw_data):
            raise ValueError("Experiment open data and sample data are not of the same length, check energy domain")

        true_Bi = gamma_background_function() 
        raw_data, true_c = inverse_reduction(self.raw_data, 
                                                    true_neutron_spectrum,
                                                    self.options["Sample Counting Noise"], 
                                                    self.options["Sample TURP"])
        
        return raw_data
        
    

    def reduce(self, raw_data, neutron_spectrum, reduction_parameters):
        """
        Reduces the raw count data (sample in/out) to Transmission data and propagates uncertainty.

        """

        # create yield dataframe
        Yg = pd.DataFrame()
        Yg['tof'] = raw_data.tof
        Yg['E'] = raw_data.E
        Yg['true'] = raw_data.true

        # estimated background function
        Bi = gamma_background_function()

        # define systematic uncertainties
        Yg['exp_trans'], unc_data, rates = reduce_raw_count_data(self.options["Calculate Covariance"])

        # sort out covariances
        if self.options["Calculate Covariance"]:
            self.CovT, self.CovY_stat, self.CovY_sys, self.Jac_sys, self.Cov_sys = unc_data
            Yg['exp_unc'] = np.sqrt(np.diag(self.CovT))
            self.CovT = pd.DataFrame(self.CovT, columns=Yg.E, index=Yg.E)
            self.CovT.index.name = None
        else:
            diag_tot, diag_stat, diag_sys = unc_data
            Yg['exp_unc'] = np.sqrt(diag_tot)
            self.CovT = None

        return Yg