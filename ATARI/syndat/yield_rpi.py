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
    #What is this function for?
    pass


def get_covT():
    #We need to come back to this later
    pass

    
def reduce_raw_count_data(raw_data,countrate, background, neutron_spectrum, TURP):
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    scaling=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*TURP["Scaling"][0],
                          'dc'   :   np.ones(len(raw_data.E))*TURP["Scaling"][1]})
    
    #Distribution Calculations:
    Yield             = np.multiply(scaling.c, np.divide(countrate.c - background.c, neutron_spectrum.c))
    
    #Uncertainty Calculations:
    partial_Y_Cc      = np.divide(scaling.c, neutron_spectrum.c)
    partial_Y_Cb      =-np.divide(scaling.c, neutron_spectrum.c)
    partial_Y_flux    =-np.multiply(scaling.c, np.divide(countrate.c - background.c, np.power(neutron_spectrum.c,2)))
    partial_Y_scaling = np.divide(countrate.c - background.c, neutron_spectrum.c)
    
    Yield_uncertainty = np.sqrt(np.power(np.multiply(partial_Y_Cc     , countrate.dc)       ,2)
                               +np.power(np.multiply(partial_Y_Cb     , background.dc)      ,2)
                               +np.power(np.multiply(partial_Y_flux   , neutron_spectrum.dc),2)
                               +np.power(np.multiply(partial_Y_scaling, scaling.dc)         ,2))
    
    return(Yield,Yield_uncertainty)


def inverse_reduction(raw_data, 
                      neutron_spectrum,
                      background,
                      TURP):
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    scaling=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*TURP["Scaling"][0],
                          'dc'   :   np.ones(len(raw_data.E))*TURP["Scaling"][1]})
    
    #Distribution Calculations:
    count_rate             = np.divide(np.multiply(raw_data.true,neutron_spectrum.c),scaling.c) + background.c
    
    #Uncertainty Calculations:
    partial_Cc_Cb         = 1
    partial_Cc_flux       = np.divide(raw_data.true, scaling.c)
    partial_Cc_scaling    = np.divide(np.multiply(neutron_spectrum.c, raw_data.true), np.power(scaling.c,2))
    
    count_rate_uncertainty = np.sqrt(np.power(np.multiply(partial_Cc_Cb     , background.dc)      ,2)
                                    +np.power(np.multiply(partial_Cc_flux   , neutron_spectrum.dc),2)
                                    +np.power(np.multiply(partial_Cc_scaling, scaling.dc)         ,2))
    
    return(count_rate,count_rate_uncertainty)






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
                            'bw'        :   (0.0064,              0),
                            'Scaling'   :   (1,                   0)
                            }

        self.options = update_dict(default_options, options)
        self.reduction_parameters = update_dict(default_reduction_parameters, reduction_parameters)


    def run(self, 
            pw_true, #Contains E (Energy) and true (?)
            neutron_spectrum=pd.DataFrame(), #Sets a default, empty dataframe.
            ):
        ### define raw data attribute as true_df
        pw_true["tof"] = e_to_t(pw_true.E, self.reduction_parameters["FP"][0], True)*1e6+self.reduction_parameters["t0"][0]
        self.raw_data = pw_true

        ### sample true underlying resonance parameters from measured values - defines self.theo_redpar
        self.true_reduction_parameters = sample_true_underlying_parameters(self.reduction_parameters, self.options["Sample TURP"])

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
        self.count_rate, self.background = self.generate_raw_data(self.true_neutron_spectrum, self.true_reduction_parameters)

        ### reduce the raw count data
        self.data       = self.reduce(self.neutron_spectrum, self.reduction_parameters, self.count_rate, self.background)
        
        ### stupid gaussian sampling around true
        # Yg = pd.DataFrame()
        # Yg['tof'] = self.raw_data.tof
        # Yg['E'] = self.raw_data.E
        # Yg['true'] = self.raw_data.true
        # Yg_std = (np.sqrt(Yg['true'])+1) * 0.05
        # Yg["exp"] = abs(np.random.default_rng().normal(Yg['true'],  Yg_std))
        # Yg['exp_unc'] = Yg_std

        # self.data = Yg





    def generate_raw_data(self,
                          neutron_spectrum,
                          reduction_parameters
                          ):
        """
        Generates a set of noisy, count data from a theoretical yield via the novel inverse-reduction method (Walton, et al.).

        Parameters
        ----------
        neutron_spectrum:
            A dataframe that describes the neutron flux spectrum for this problem.
            Has elements c and dc representing the distribution and the uncertainty respectivly
        
        reduction_parameters
            A dictionary with all reduction parameters.
            For this, it must contain 'scaling' so that the function can operate.

        Returns
        -------
        count_rate
            A dataframe that describes the generated countrate
            Has elements c and dc representing the distribution and the uncertainty respectivly
        
        Raises
        ------
        ValueError
            _description_
        """

        if len(neutron_spectrum) != len(self.raw_data):
            raise ValueError("Experiment open data and sample data are not of the same length, check energy domain")

        #true_Bi = gamma_background_function()
        #We use a linear aproximation for the background function for now.
        background=pd.DataFrame({'c'    :   np.ones(len(self.raw_data.E))*25,
                                 'dc'   :   np.ones(len(self.raw_data.E))*0})
        
        #If we are sampling TURP, we also should sample background. Might it be better to have a distinct setting for this?
        if(self.options["Sample TURP"]):
            background.c=pois_noise(background.c)
            background.dc=np.sqrt(background.c)
        
        count_rate,count_rate_uncertainty = inverse_reduction(self.raw_data, 
                                     neutron_spectrum,
                                     background,
                                     reduction_parameters
                                    )
        count_rate=pd.DataFrame({'c'    :   count_rate,
                                 'dc'   :   count_rate_uncertainty})
        
        #We sample countrate here instead of in the reduction function.
        if(self.options["Sample Counting Noise"]):
            count_rate.c=pois_noise(count_rate.c)
        
        #Uncertainty from this operation still needs to be nailed down.
        
        return count_rate,background
        
    

    def reduce(self, neutron_spectrum, reduction_parameters, countrate, background):
        """
        Reduces the raw count data (sample in/out) to Transmission data and propagates uncertainty.

        Parameters
        ----------
        neutron_spectrum:
            A dataframe that describes the neutron flux spectrum for this problem.
            Has elements c and dc representing the distribution and the uncertainty respectivly
        
        reduction_parameters
            A dictionary with all reduction parameters.
            For this, it must contain 'scaling' so that the function can operate.
            
        background
            A dataframe that describes the background for the experiment, should be same as used in the inverse reduction

        count_rate
            A dataframe that describes the generated countrate
            Has elements c and dc representing the distribution and the uncertainty respectivly

        Returns
        -------
        Yield
            A dataframe that describes the reduced sampled yield
            Has elements exp and exp_unc representing the distribution and the uncertainty respectivly
        
        Raises
        ------
        ValueError
            _description_
        """

        # create yield dataframe
        Yg = pd.DataFrame()
        Yg['tof']  = self.raw_data.tof
        Yg['E']    = self.raw_data.E
        Yg['true'] = self.raw_data.true

        # estimated background function
        #Bi = gamma_background_function()
        #We are just going to pass the background from the inverse reduction here since it's the same thing anyways

        # define systematic uncertainties
        #I will implement covariance later once we get to defining that.
        Yg['exp'],Yg['exp_unc'] = reduce_raw_count_data(self.raw_data, countrate, background, neutron_spectrum, reduction_parameters)

        return Yg