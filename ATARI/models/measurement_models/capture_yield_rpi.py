import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e




# ========================================================================================
#            Capture reduction equations at RPI
# ========================================================================================


def gammayield():
    #What is this function for?
    pass


def get_covT():
    #We need to come back to this later
    pass

    
def reduce_raw_count_data(raw_data, neutron_spectrum, model_parameters):
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    scaling=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*model_parameters.Scaling[0],
                          'dc'   :   np.ones(len(raw_data.E))*model_parameters.Scaling[1]})
    
    #Distribution Calculations:
    Yield             = np.multiply(scaling.c, np.divide(raw_data.cg - raw_data.cb, neutron_spectrum.c))
    
    #Uncertainty Calculations:
    partial_Y_Cc      = np.divide(scaling.c, neutron_spectrum.c)
    partial_Y_Cb      =-np.divide(scaling.c, neutron_spectrum.c)
    partial_Y_flux    =-np.multiply(scaling.c, np.divide(raw_data.cg - raw_data.cb, np.power(neutron_spectrum.c,2)))
    partial_Y_scaling = np.divide(raw_data.cg - raw_data.cb, neutron_spectrum.c)
    
    Yield_uncertainty = np.sqrt(np.power(np.multiply(partial_Y_Cc     , raw_data.dcg)       ,2)
                               +np.power(np.multiply(partial_Y_Cb     , raw_data.dcb)      ,2)
                               +np.power(np.multiply(partial_Y_flux   , neutron_spectrum.dc),2)
                               +np.power(np.multiply(partial_Y_scaling, scaling.dc)         ,2))
    
    return(Yield,Yield_uncertainty)


def inverse_reduction(raw_data, 
                      neutron_spectrum,
                      gamma_background_spectrum,
                      TURP):
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    scaling=pd.DataFrame({'c'    :   np.ones(len(raw_data.E))*TURP.Scaling[0],
                          'dc'   :   np.ones(len(raw_data.E))*TURP.Scaling[1]})
    
    #Distribution Calculations:
    count_rate             = np.divide(np.multiply(raw_data.true,neutron_spectrum.c),scaling.c) + gamma_background_spectrum.c
    
    #Uncertainty Calculations:
    partial_Cc_Cb         = 1
    partial_Cc_flux       = np.divide(raw_data.true, scaling.c)
    partial_Cc_scaling    = np.divide(np.multiply(neutron_spectrum.c, raw_data.true), np.power(scaling.c,2))
    
    count_rate_uncertainty = np.sqrt(np.power(np.multiply(partial_Cc_Cb     , gamma_background_spectrum.dc)      ,2)
                                    +np.power(np.multiply(partial_Cc_flux   , neutron_spectrum.dc),2)
                                    +np.power(np.multiply(partial_Cc_scaling, scaling.dc)         ,2))
    
    return(count_rate,count_rate_uncertainty)



from ATARI.models.structuring import parameter, vector_parameter
from ATARI.syndat.data_classes import syndatOPT



class capture_yield_rpi_parameters:

    trigo = parameter()
    trigs = parameter()
    Scaling  = parameter()
    neutron_spectrum = vector_parameter()
    gamma_background_spectrum = vector_parameter()

    def __init__(self, **kwargs):

        self.trigo =  (9758727,         0)
        self.trigs =  (18476117,        0)
        self.Scaling =  (1,             0)
        
        self.neutron_spectrum = None
        self.gamma_background_spectrum = None

        for key, value in kwargs.items():
            setattr(self, key, value)


    def sample_parameters(self, true_model_parameters: dict):
        sampled_params = {}

        for param_name, param_values in self.__dict__.items():
            if param_name in true_model_parameters:
                sampled_params[param_name] = (true_model_parameters[param_name], 0.0)
            else:
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    mean, uncertainty = param_values
                    if uncertainty == 0:
                        pass
                    else:
                        sample = np.random.normal(loc=mean, scale=uncertainty)
                        sampled_params[param_name] = (sample, 0.0)
                if isinstance(param_values, pd.DataFrame):
                    new_c = np.random.normal(loc=param_values.c, scale=param_values.dc)
                    df = deepcopy(param_values)
                    df.loc[:,'c'] = new_c
                    df.loc[:,'dc'] = np.sqrt(new_c)
                    sampled_params[param_name] = df

        return capture_yield_rpi_parameters(**sampled_params)
    


# ========================================================================================
#            syndat_T class 
# ========================================================================================

class Capture_Yield_RPI:

    def __init__(self,**kwargs):
        self._model_parameters = capture_yield_rpi_parameters(**kwargs)
        self._covariance_data = {}

    @property
    def model_parameters(self) -> capture_yield_rpi_parameters:
        return self._model_parameters
    @model_parameters.setter
    def model_parameters(self, model_parameters):
        self._model_parameters = model_parameters

    @property
    def neutron_spectrum_triggers(self) -> int:
        return self.model_parameters.trigo[0]

    @property
    def covariance_data(self) -> dict:
        return self._covariance_data
    @covariance_data.setter
    def covariance_data(self, covariance_data):
        self._covariance_data = covariance_data


    def __repr__(self):
        string = 'Measurement model (data reduction) parameters:\n'
        string += str(vars(self.model_parameters))
        return string


    def sample_true_model_parameters(self, true_model_parameters: dict):
        return self.model_parameters.sample_parameters(true_model_parameters)


    def approximate_unknown_data(self, exp_model):
        if self.model_parameters.neutron_spectrum is None:
            neutron_spectrum = approximate_neutron_spectrum_Li6det(exp_model.energy_grid, 
                                                                    False, #self.options.smoothTNCS, 
                                                                    exp_model.FP[0],
                                                                    exp_model.t0[0],
                                                                    self.neutron_spectrum_triggers)
            
            self.model_parameters.neutron_spectrum = neutron_spectrum
            self.model_parameters.neutron_spectrum = neutron_spectrum

        if self.model_parameters.gamma_background_spectrum is None:
            gamma_background_spectrum= pd.DataFrame({'c'    :   np.ones(len(exp_model.energy_grid))*25,
                                                    'dc'   :   np.ones(len(exp_model.energy_grid))*0})
            self.model_parameters.gamma_background_spectrum = gamma_background_spectrum



    def generate_raw_data(self,
                          pw_true,
                          true_model_parameters, # need to build better protocol for this 
                          options: syndatOPT
                          ) -> pd.DataFrame:
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

        assert true_model_parameters.neutron_spectrum is not None
        if len(true_model_parameters.neutron_spectrum) != len(pw_true):
            raise ValueError(
                "neutron spectrum and sample data are not of the same length, check energy domain")
        assert true_model_parameters.gamma_background_spectrum is not None
        if len(true_model_parameters.gamma_background_spectrum) != len(pw_true):
            raise ValueError(
                "neutron spectrum and sample data are not of the same length, check energy domain")
        
        # ====================
        #### Noah's Changes: True reduction parameters (including neutron spectrum and background) are now sampled using the sample_true_model_parameters
        # ====================

        # # true_reduction_parameters = sample_true_underlying_parameters(self.reduction_parameters, options["Sample TURP"])

        #true_Bi = gamma_background_function()
        #We use a linear aproximation for the background function for now.
        # background=pd.DataFrame({'c'    :   np.ones(len(pw_true.E))*25,
        #                          'dc'   :   np.ones(len(pw_true.E))*0})
        
        #If we are sampling TURP, we also should sample background. Might it be better to have a distinct setting for this?
        # if(options.sampleTURP):
        #     background.c=pois_noise(background.c)
        #     background.dc=np.sqrt(background.c)
        # ====================
        #### 
        # ====================

        count_rate,count_rate_uncertainty = inverse_reduction(pw_true, 
                                     true_model_parameters.neutron_spectrum,
                                     true_model_parameters.gamma_background_spectrum,
                                     true_model_parameters
                                    )
        count_rate=pd.DataFrame({'c'    :   count_rate,
                                 'dc'   :   count_rate_uncertainty})
        
        #We sample countrate here instead of in the reduction function.
        if(options.sample_counting_noise):
            count_rate.c=pois_noise(count_rate.c)
        
        #Uncertainty from this operation still needs to be nailed down.

        raw_data = pd.DataFrame({'E':pw_true.E, 
                                 'tof':pw_true.tof, 
                                 'true':pw_true.true, 
                                 'cg':count_rate.c,
                                 'dcg':count_rate.dc, 
                                 'cb':true_model_parameters.gamma_background_spectrum.c, 
                                 'dcb':true_model_parameters.gamma_background_spectrum.dc})

        return raw_data
        
    

    def reduce_raw_data(self, raw_data, neutron_spectrum, options): # neutron_spectrum, reduction_parameters, countrate, background):
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
        Yg['tof']  = raw_data.tof
        Yg['E']    = raw_data.E
        Yg['true'] = raw_data.true

        # estimated background function
        #Bi = gamma_background_function()
        #We are just going to pass the background from the inverse reduction here since it's the same thing anyways

        # define systematic uncertainties
        #I will implement covariance later once we get to defining that.
        Yg.loc[:,'exp'],Yg.loc[:,'exp_unc'] = reduce_raw_count_data(raw_data, neutron_spectrum, self.model_parameters)

        return Yg