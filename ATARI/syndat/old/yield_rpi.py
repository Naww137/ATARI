# cSpell:disable

import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e


"""Summary
----------
    Description

Parameters
----------
    Name - Type: Text - Default: Text
    
        -Description

Modes
-----
    Name - Type: Text - Default: Text
    
        -Value - Function

Returns
-------
    Name - Type: Text - Default: Text
    
        -Description

Raises
------
    Error - Likely cause: Text
    
        -Description

Outputs
-------
    Text - Cause: Text
    
        -Description
"""


# ========================================================================================
#            Yield reduction equations at RPI
# ========================================================================================


def gammayield():
    #What is this function for?
    pass


def get_covT():
    #We need to come back to this later
    pass

    
def reduce_raw_count_data(raw_data,countrate, background, neutron_spectrum, TURP):
    """Summary
    ----------
        Given count rate data and the experiment description (background, neutron spectrum, and TURP), finds the associated yield while propogating the uncertainty

    Parameters
    ----------
        raw_data - Type: Dataframe
        
            -A dataframe describing the true yield around which we are sampling and it's energy space
        
            ! Must contain E, the energy spectrum
        
        countrate - Type: Dataframe
        
            -A dataframe describing the count rate data that is being reduced
        
            ! Must contain c, the count rate, and dc, the uncertainty in the count rate
        
        background - Type: Dataframe
        
            -A dataframe describing the background for the experiment
        
            ! Must contain c, the background count rate, and dc, the uncertainty in the background count rate
        
        neutron_spectrum - Type: Dataframe
        
            -A dataframe describing the neutron flux for the experiment
        
            ! Must contain c, the neutron flux, and dc, the uncertainty in the neutron flux
        
        TURP - Type: Dictionary
        
            -A dictionary containing the underlying parameters that describe the experiment, notably the scaling function
        
            ! Must contain 'Scaling', a (2) shape NumPy array storing the scaling value and it's uncertainty respectively
    
    Returns
    -------
        Yield - Type: (n) shape NumPy array
        
            -An array describing the yield associated with the input count rate
        
        Yield_uncertainty - Type: (n) shape NumPy array
        
            -An array describing the uncertainty in the yield
    """
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    scaling=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*TURP['Scaling'][0],
                          'dc'   :   np.ones(len(raw_data.E))*TURP['Scaling'][1]})
    
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
    """Summary
    ----------
        Given the true yield data and the experiment description (background, neutron spectrum, and TURP), finds the associated count rate while propagating the uncertainty

    Parameters
    ----------
        raw_data - Type: Dataframe
        
            -A dataframe describing the true yield and it's energy space
        
            ! Must contain E, the energy spectrum, and true, the true yield
        
        background - Type: Dataframe
        
            -A dataframe describing the background for the experiment
        
            ! Must contain c, the background count rate, and dc, the uncertainty in the background count rate
        
        neutron_spectrum - Type: Dataframe
        
            -A dataframe describing the neutron flux for the experiment
        
            ! Must contain c, the neutron flux, and dc, the uncertainty in the neutron flux
        
        TURP - Type: Dictionary
        
            -A dictionary containing the underlying parameters that describe the experiment, notably the scaling function
        
            ! Must contain 'Scaling', a (2) shape NumPy array storing the scaling value and it's uncertainty respectively
    
    Returns
    -------
        Count_rate - Type: (n) shape NumPy array
        
            -An array describing the count rate associated with the input yield
        
        Yield_uncertainty - Type: (n) shape NumPy array
        
            -An array describing the uncertainty in the count rate
    """
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    scaling=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*TURP['Scaling'][0],
                          'dc'   :   np.ones(len(raw_data.E))*TURP['Scaling'][1]})
    
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
#            syndat_Y class 
# ========================================================================================

class syndat_Y:

    def __init__(self, 
                 options={}, 
                 reduction_parameters={}, 
                 ):
        """Summary
        ----------
            Generates fake yield distributions around a given true distribution. The fake data is produced via sampling uncertainties found in real experiments.

        Parameters
        ----------
            options - Type: Dictionary - Default: {}
            
                -A dictionary listing different options for the fitting algorithm to use
            
            reduction_parameters - Type: Dictionary - Default: {}
            
                -A dictionary containing different default reduction parameters for the fitting algorithm to use
        """
        
        default_options = { 'Sample Counting Noise' : True,
                            'Sample TURP'           : True,
                            'Sample TNCS'           : True,
                            'Smooth TNCS'           : False,
                            'Calculate Covariance'  : True,
                            'Compression Points'    : [],
                            'Grouping Factors'      : None,
                            'TURP Sample Type'      : 'poisson'
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
            pw_true,
            neutron_spectrum=pd.DataFrame(),
            ):
        """Summary
        ----------
            Takes the true yield as input and generates a fake data distribution around it
            
            Final Data is stored in a class variable data, a dataframe containing exp and exp_unc which represented the generated distribution and it's uncertainty respectively

        Parameters
        ----------
            pw_true - Type: Dataframe
            
                -A dataframe containing information about the true yield that data should be generated around
                
                ! Must contain E, which represents the energy range used, and true, which represents the true yield
            
            neutron_spectrum - Type: Dataframe
            
                -A dataframe that describes the neutron spectrum for the problem. If this is omitted, a standard Li6 neutron spectrum will be generated and used instead
                
                ! If used, must contain c and dc which represent the neutron spectrum and it's uncertainty respectively
        """
        
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
        if self.options['Sample TNCS']:
            self.true_neutron_spectrum = sample_true_neutron_spectrum(self.neutron_spectrum)
        else:
            self.true_neutron_spectrum = deepcopy(self.neutron_spectrum)

        ### generate raw count data for sample in given theoretical transmission and assumed true reduction parameters/open count data
        self.count_rate, self.background = self.generate_raw_data(self.raw_data, self.true_neutron_spectrum, self.true_reduction_parameters, self.options)

        ### reduce the raw count data
        self.data = self.reduce(self.raw_data, self.neutron_spectrum, self.count_rate, self.background, self.reduction_parameters)
        
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
                          raw_data,
                          neutron_spectrum,
                          reduction_parameters,
                          options
                          ):
        """Summary
        ----------
            Generates a set of noisy, count data from a theoretical yield via the novel inverse-reduction method (Walton, et al.).

        Parameters
        ----------
            raw_data - Type: Dataframe

                -A dataframe containing information about the true yield that data should be generated around
                
                ! Must contain E, which represents the energy range used, and true, which represents the true yield
            
            neutron_spectrum - Type: Dataframe
            
                -A dataframe that describes the neutron flux spectrum for this problem.
                
                ! Must contain c and dc which represent the neutron spectrum and it's uncertainty respectively
            
            reduction_parameters - Type: Dictionary
            
                -A dictionary containing all reduction parameters to be used
                
                ! must contain 'scaling' for this function to operate
            
            options - Type: Dictionary

                -A dictionary containing the various user options for operation
                
                ! Must contain 'Sample TURP', 'TURP Sample Type', and 'Sample Counting Noise' for the function to operate
        
        Returns
        -------
            count_rate - Type: Dataframe
            
                -A dataframe that represents the countrate generated (and possibly sampled) from the yield

        Raises
        ------
            "Experiment open data and sample data are not of the same length, check energy domain" - Likely cause: Given neutron spectrum is not the same length as true yield
        """

        if len(neutron_spectrum) != len(raw_data):
            raise ValueError("Experiment open data and sample data are not of the same length, check energy domain")

        #true_Bi = gamma_background_function()
        #We use a linear approximation for the background function for now.
        background=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*25,
                                 'dc'   :   np.ones(len(raw_data.E))*0})
        
        #If we are sampling TURP, we also should sample background. Might it be better to have a distinct setting for this?
        if(options['Sample TURP']):
            if(options['TURP Sample Type']=='poisson'):
                background.c=pois_noise(background.c)
                background.dc=np.sqrt(background.c)
            if(options['TURP Sample Type']=='gaussian'):
                background.c=gaus_noise(background.c,background.dc)
        
        #Performs the inverse reduction to get countrate and puts it into a dataframe
        count_rate,count_rate_uncertainty = inverse_reduction(raw_data, 
                                     neutron_spectrum,
                                     background,
                                     reduction_parameters
                                    )
        count_rate=pd.DataFrame({'c'    :   count_rate,
                                 'dc'   :   count_rate_uncertainty})
        
        #Sampling countrate
        if(options['Sample Counting Noise']):
            count_rate.c=pois_noise(count_rate.c)
        
        #Uncertainty from this operation still needs to be nailed down.
        
        return count_rate,background
        
    

    def reduce(self,
               data,
               neutron_spectrum,
               countrate,
               background,
               reduction_parameters):
        """Summary
        ----------
            Reduces the raw count data (sample in/out) to yield data and propagates uncertainty.

        Parameters
        ----------
            raw_data - Type: Dataframe

                -A dataframe containing information about the true yield that data should be generated around
                
                ! Must contain E, which represents the energy range used

            neutron_spectrum - Type: Dataframe
            
                -A dataframe describing the neutron flux for the experiment
            
                ! Must contain c, the neutron flux, and dc, the uncertainty in the neutron flux
            
            reduction_parameters - Type: Dictionary
            
                -A dictionary containing all reduction parameters to be used
                
                ! must contain 'scaling' for this function to operate
        
            countrate - Type: Dataframe
            
                -A dataframe describing the count rate data that is being reduced
            
                ! Must contain c, the count rate, and dc, the uncertainty in the count rate
            
            background - Type: Dataframe
            
                -A dataframe describing the background for the experiment
            
                ! Must contain c, the background count rate, and dc, the uncertainty in the background count rate

        Returns
        -------
            Yg - Type: Dataframe
            
                -A dataframe describing the reduced yield corresponding to the sampled data
        """

        #Estimated background function
        #Bi = gamma_background_function()
        #We are just going to pass the background from the inverse reduction here since it's the same thing anyways

        #Reduces the sampled data to get sampled yield and adds it to the input data
        #I will implement covariance later once we get to defining that.
        data['exp'], data['exp_unc'] = reduce_raw_count_data(data, countrate, background, neutron_spectrum, reduction_parameters)

        return data