import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e


from ATARI.Models.structuring import parameter, vector_parameter
from ATARI.syndat.data_classes import syndatOPT




# ========================================================================================
#            Capture reduction equations at RPI
# ========================================================================================


def gammayield():
    #What is this function for?
    pass


def get_covT():
    #We need to come back to this later
    pass

    
def reduce_raw_count_data(raw_data, model_parameters):

    ### counts to count rates for target gamma and background measurements
    crg, dcrg = cts_to_ctr(raw_data.cg, raw_data.dcg, model_parameters.incident_neutron_spectrum_f.bw, model_parameters.trig_g[0])
    brg, dbrg = cts_to_ctr(model_parameters.background_spectrum_bg.c, 
                           model_parameters.background_spectrum_bg.dc,
                           model_parameters.incident_neutron_spectrum_f.bw, 
                           model_parameters.trig_bg[0])
    ### counts to count rates for incident flux measurement
    cr_flux, dcr_flux = cts_to_ctr(model_parameters.incident_neutron_spectrum_f.c,
                         model_parameters.incident_neutron_spectrum_f.dc,
                         model_parameters.incident_neutron_spectrum_f.bw,
                         model_parameters.trig_f[0])
    br_flux, dbr_flux = cts_to_ctr(model_parameters.background_spectrum_bf.c,
                         model_parameters.background_spectrum_bf.dc,
                         model_parameters.incident_neutron_spectrum_f.bw,
                         model_parameters.trig_bf[0])

    #Distribution Calculations:
    relative_flux_rate = cr_flux - br_flux # neet to mornalize by yield here!
    relative_flux_rate_uncertainty = np.sqrt(np.power(dcr_flux,2) - np.power(dbr_flux,2))
    Yield             = model_parameters.fn[0] * np.divide(crg - brg, relative_flux_rate)
    
    #Uncertainty Calculations:
    partial_Y_Cc      = relative_flux_rate/model_parameters.fn[0]
    partial_Y_Cb      =-relative_flux_rate/model_parameters.fn[0]
    partial_Y_flux    =-model_parameters.fn[0]*np.divide(crg - brg, np.power(relative_flux_rate,2))
    partial_Y_scaling = np.divide(crg - brg, relative_flux_rate)
    
    Yield_uncertainty = np.sqrt(np.power(np.multiply(partial_Y_Cc     , dcrg)       ,2)
                               +np.power(np.multiply(partial_Y_Cb     , dbrg)      ,2)
                               +np.power(np.multiply(partial_Y_flux   , relative_flux_rate_uncertainty),2)
                               +np.power(partial_Y_scaling*model_parameters.fn[1]        ,2))
    
    return Yield, Yield_uncertainty


def inverse_reduction(pw_true, true_model_parameters):
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    # scaling=pd.DataFrame({'c'    :   np.ones(len(pw_true.E))*TURP.Scaling[0],
    #                       'dc'   :   np.ones(len(pw_true.E))*TURP.Scaling[1]})
    
    ### relative flux rate measurement
    cr_flux, dcr_flux = cts_to_ctr(true_model_parameters.incident_neutron_spectrum_f.c,
                                true_model_parameters.incident_neutron_spectrum_f.dc,
                                true_model_parameters.incident_neutron_spectrum_f.bw,
                                true_model_parameters.trig_f[0])
    br_flux, dbr_flux = cts_to_ctr(true_model_parameters.background_spectrum_bf.c,
                                true_model_parameters.background_spectrum_bf.dc,
                                true_model_parameters.incident_neutron_spectrum_f.bw,
                                true_model_parameters.trig_bf[0])

    relative_flux_rate = cr_flux - br_flux # need to normalize by yield here!
    
    ### target gamma background and count rate
    br_gamma, dbr_gamma = cts_to_ctr(true_model_parameters.background_spectrum_bg.c,
                                    true_model_parameters.background_spectrum_bg.dc,
                                    true_model_parameters.incident_neutron_spectrum_f.bw,
                                    true_model_parameters.trig_bg[0])
    cr_gamma_true = np.multiply(pw_true.true, relative_flux_rate)/true_model_parameters.fn[0] + br_gamma
    
    ### target gamma count rate to counts and add uncertainty
    c_true = cr_gamma_true*pw_true.tof*true_model_parameters.trig_g[0]
    
    ### =========================
    ### Why uncertainty calculations here? On the generation side, everything is determined
    ### =========================

    # #Uncertainty Calculations:
    # partial_Cc_Cb         = 1
    # partial_Cc_flux       = np.divide(raw_data.true, scaling.c)
    # partial_Cc_scaling    = np.divide(np.multiply(neutron_spectrum.c, raw_data.true), np.power(scaling.c,2))
    
    # count_rate_uncertainty = np.sqrt(np.power(np.multiply(partial_Cc_Cb     , gamma_background_spectrum.dc)      ,2)
    #                                 +np.power(np.multiply(partial_Cc_flux   , neutron_spectrum.dc),2)
    #                                 +np.power(np.multiply(partial_Cc_scaling, scaling.dc)         ,2))
    

    return c_true




class capture_yield_rpi_parameters:

    trig_g  = parameter()
    trig_bg = parameter()
    trig_f  = parameter()
    trig_bf = parameter()
    fn = parameter()

    # incident_neutron_spectrum_g = vector_parameter()
    background_spectrum_bg = vector_parameter()
    incident_neutron_spectrum_f = vector_parameter()
    background_spectrum_bf = vector_parameter()

    def __init__(self, **kwargs):

        self.trig_g     =  (10000000,   0)
        self.trig_bg    =  (100000000,  0)
        self.trig_f     =  (10000000,   0)
        self.trig_bf    =  (100000000,  0)
        self.fn         =  (1,          0)
    
        self.background_spectrum_bg = None
        self.incident_neutron_spectrum_f = None
        self.background_spectrum_bf = None

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
#            Handler class 
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

    # @property
    # def neutron_spectrum_triggers(self) -> int:
    #     return self.model_parameters.trigo[0]

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

        if self.model_parameters.background_spectrum_bg is None:
            background_spectrum_bg = approximate_gamma_background_spectrum(exp_model.energy_grid, 
                                                                           False, 
                                                                           exp_model.FP[0], 
                                                                           exp_model.t0[0], 
                                                                           self.model_parameters.trig_bg[0])
            self.model_parameters.background_spectrum_bg = background_spectrum_bg

        if self.model_parameters.incident_neutron_spectrum_f is None:
            incident_neutron_spectrum_f = approximate_neutron_spectrum_Li6det(exp_model.energy_grid, 
                                                                            False, #self.options.smoothTNCS, 
                                                                            exp_model.FP[0],
                                                                            exp_model.t0[0],
                                                                            self.model_parameters.trig_bf[0])
            
            self.model_parameters.incident_neutron_spectrum_f = incident_neutron_spectrum_f
            self.model_parameters.incident_neutron_spectrum_f = incident_neutron_spectrum_f

        if self.model_parameters.background_spectrum_bf is None:
            background_spectrum_bf = approximate_gamma_background_spectrum(exp_model.energy_grid, 
                                                                           False, 
                                                                           exp_model.FP[0], 
                                                                           exp_model.t0[0], 
                                                                           self.model_parameters.trig_bf[0])
            self.model_parameters.background_spectrum_bf = background_spectrum_bf



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

        assert true_model_parameters.incident_neutron_spectrum_f is not None
        if len(true_model_parameters.incident_neutron_spectrum_f) != len(pw_true):
            raise ValueError("neutron flux spectrum and sample data are not of the same length, check energy domain")
        assert true_model_parameters.background_spectrum_bg is not None
        if len(true_model_parameters.background_spectrum_bg) != len(pw_true):
            raise ValueError("gamma background spectrum for target capture measurement and sample data are not of the same length, check energy domain")
        assert true_model_parameters.background_spectrum_bf is not None
        if len(true_model_parameters.background_spectrum_bf) != len(pw_true):
            raise ValueError("gamma background spectrum for flux yield measurement and sample data are not of the same length, check energy domain")
        
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

        raw_data = deepcopy(pw_true)
        true_gamma_counts = inverse_reduction(pw_true, true_model_parameters)

        if options.sample_counting_noise:
            c = pois_noise(true_gamma_counts)
        else:
            c = true_gamma_counts
        
        assert(c.all() >= 0)
        dc = np.sqrt(c)

        raw_data.loc[:, 'cg'] = c
        raw_data.loc[:, 'dcg'] = dc
        
        # count_rate=pd.DataFrame({'c'    :   count_rate,
        #                          'dc'   :   count_rate_uncertainty})
        
        #We sample countrate here instead of in the reduction function.
        # if(options.sample_counting_noise):
        #     count_rate.c=pois_noise(count_rate.c)
        
        #Uncertainty from this operation still needs to be nailed down.

        # raw_data = pd.DataFrame({'E':pw_true.E, 
        #                          'tof':pw_true.tof, 
        #                          'true':pw_true.true, 
        #                          'cg':count_rate.c,
        #                          'dcg':count_rate.dc, 
        #                          'cb':true_model_parameters.gamma_background_spectrum.c, 
        #                          'dcb':true_model_parameters.gamma_background_spectrum.dc})

        return raw_data
        
    

    def reduce_raw_data(self, raw_data, options): # neutron_spectrum, reduction_parameters, countrate, background):
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
        Yg.loc[:,'exp'], unc_data = reduce_raw_count_data(raw_data, self.model_parameters)

        if options.calculate_covariance:
            raise ValueError("not implemented")

            if options.explicit_covariance:
                pass

            else:
                pass

        else:
            diag_tot = unc_data
            Yg.loc[:,'exp_unc'] = diag_tot

        return Yg