import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e


from ATARI.ModelData.structuring import parameter, vector_parameter
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
    crg, dcrg = cts_to_ctr(raw_data.ctg, raw_data.dctg, model_parameters.incident_neutron_spectrum_f.bw, model_parameters.trig_g[0])
    brg, dbrg = cts_to_ctr(model_parameters.background_spectrum_bg.ct, 
                           model_parameters.background_spectrum_bg.dct,
                           model_parameters.incident_neutron_spectrum_f.bw, 
                           model_parameters.trig_bg[0])
    ### counts to count rates for incident flux measurement
    cr_flux, dcr_flux = cts_to_ctr(model_parameters.incident_neutron_spectrum_f.ct,
                         model_parameters.incident_neutron_spectrum_f.dct,
                         model_parameters.incident_neutron_spectrum_f.bw,
                         model_parameters.trig_f[0])
    br_flux, dbr_flux = cts_to_ctr(model_parameters.background_spectrum_bf.ct,
                         model_parameters.background_spectrum_bf.dct,
                         model_parameters.incident_neutron_spectrum_f.bw,
                         model_parameters.trig_bf[0])

    #Distribution Calculations:
    relative_flux_rate = cr_flux - br_flux # neet to mornalize by yield here!
    relative_flux_rate_uncertainty = np.sqrt(np.power(dcr_flux,2) + np.power(dbr_flux,2))
    Yield             = model_parameters.fn[0] * np.divide(crg - brg, relative_flux_rate)
    
    #Uncertainty Calculations:
    partial_Y_Cc      = model_parameters.fn[0]/relative_flux_rate
    partial_Y_Cb      =-model_parameters.fn[0]/relative_flux_rate
    partial_Y_flux    =-model_parameters.fn[0]*np.divide(crg - brg, np.power(relative_flux_rate,2))
    partial_Y_fn      = np.divide(crg - brg, relative_flux_rate)
    
    Yield_uncertainty = np.sqrt(np.power(np.multiply(partial_Y_Cc     , dcrg)       ,2)
                               +np.power(np.multiply(partial_Y_Cb     , dbrg)      ,2)
                               +np.power(np.multiply(partial_Y_flux   , relative_flux_rate_uncertainty),2)
                               +np.power(partial_Y_fn*model_parameters.fn[1]        ,2))
    
    # if calc_cov:
        # do covariance stuff
    
    # diag_stat = None
    # diag_sys = None
    # data =[diag_stat, diag_sys, Jac_sys, Cov_sys]
    

    return Yield, Yield_uncertainty


def inverse_reduction(pw_true, true_model_parameters):
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    # scaling=pd.DataFrame({'c'    :   np.ones(len(pw_true.E))*TURP.Scaling[0],
    #                       'dc'   :   np.ones(len(pw_true.E))*TURP.Scaling[1]})
    
    ### relative flux rate measurement
    cr_flux, dcr_flux = cts_to_ctr(true_model_parameters.incident_neutron_spectrum_f.ct,
                                true_model_parameters.incident_neutron_spectrum_f.dct,
                                true_model_parameters.incident_neutron_spectrum_f.bw,
                                true_model_parameters.trig_f[0])
    br_flux, dbr_flux = cts_to_ctr(true_model_parameters.background_spectrum_bf.ct,
                                true_model_parameters.background_spectrum_bf.dct,
                                true_model_parameters.incident_neutron_spectrum_f.bw,
                                true_model_parameters.trig_bf[0])

    relative_flux_rate = cr_flux - br_flux # need to normalize by yield here!
    
    ### target gamma background and count rate
    br_gamma, dbr_gamma = cts_to_ctr(true_model_parameters.background_spectrum_bg.ct,
                                    true_model_parameters.background_spectrum_bg.dct,
                                    true_model_parameters.incident_neutron_spectrum_f.bw,
                                    true_model_parameters.trig_bg[0])
    cr_gamma_true = np.multiply(pw_true.true, relative_flux_rate)/true_model_parameters.fn[0] + br_gamma
    
    ### target gamma count rate to counts and add uncertainty
    c_true = cr_gamma_true*true_model_parameters.incident_neutron_spectrum_f.bw*true_model_parameters.trig_g[0]
    
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

        self.trig_g     =  (1.2e7,   0)
        self.trig_bg    =  (0.8e7,  0)
        self.trig_f     =  (1e7,   0)
        self.trig_bf    =  (0.8e7,  0)
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
                        sample = mean
                    else:
                        sample = np.random.normal(loc=mean, scale=uncertainty)
                    sampled_params[param_name] = (sample, 0.0)
                if isinstance(param_values, pd.DataFrame):
                    new_c = np.random.poisson(param_values.ct)#, scale=param_values.dc)
                    new_c[new_c==0] = 1
                    df = deepcopy(param_values)
                    df.loc[:,'ct'] = new_c
                    df.loc[:,'dct'] = np.sqrt(new_c)
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
    

    def truncate_energy_range(self, new_energy_range):
        if min(new_energy_range) < min(self.model_parameters.background_spectrum_bg.E):
            raise ValueError("new energy range is less than existing transmission_rpi.background_spectrum_bg energy")
        if max(new_energy_range) > max(self.model_parameters.background_spectrum_bg.E):
            raise ValueError("new energy range is more than existing transmission_rpi.background_spectrum_bg energy")
        if min(new_energy_range) < min(self.model_parameters.incident_neutron_spectrum_f.E):
            raise ValueError("new energy range is less than existing transmission_rpi.incident_neutron_spectrum_f energy")
        if max(new_energy_range) > max(self.model_parameters.incident_neutron_spectrum_f.E):
            raise ValueError("new energy range is more than existing transmission_rpi.incident_neutron_spectrum_f energy")
        if min(new_energy_range) < min(self.model_parameters.background_spectrum_bf.E):
            raise ValueError("new energy range is less than existing transmission_rpi.background_spectrum_bf energy")
        if max(new_energy_range) > max(self.model_parameters.background_spectrum_bf.E):
            raise ValueError("new energy range is more than existing transmission_rpi.background_spectrum_bf energy")
        self.model_parameters.background_spectrum_bg =      self.model_parameters.background_spectrum_bg.loc[(self.model_parameters.background_spectrum_bg.E.values < max(new_energy_range)) & (self.model_parameters.background_spectrum_bg.E.values > min(new_energy_range))].copy()
        self.model_parameters.incident_neutron_spectrum_f = self.model_parameters.incident_neutron_spectrum_f.loc[(self.model_parameters.incident_neutron_spectrum_f.E.values < max(new_energy_range)) & (self.model_parameters.incident_neutron_spectrum_f.E.values > min(new_energy_range))].copy()
        self.model_parameters.background_spectrum_bf =      self.model_parameters.background_spectrum_bf.loc[(self.model_parameters.background_spectrum_bf.E.values < max(new_energy_range)) & (self.model_parameters.background_spectrum_bf.E.values > min(new_energy_range))].copy()
        return


    def approximate_unknown_data(self, exp_model, smooth, check_trig=False, overwrite=False, nominal=25):
        
        if check_trig:
            for each in [self.model_parameters.trig_g, self.model_parameters.trig_bg, self.model_parameters.trig_f, self.model_parameters.trig_bf]:
                if exp_model.channel_widths['chw'][0]*1e-9 * each[0] < 1:
                    print("WARNING: the linac trigers in you generative measurement model are not large enough for the channel bin width. This will result in many bins with 0 counts")

        if self.model_parameters.background_spectrum_bg is None or overwrite:
            background_spectrum_bg = approximate_gamma_background_spectrum(exp_model.energy_grid, 
                                                                           smooth, 
                                                                           exp_model.FP[0], 
                                                                           exp_model.t0[0], 
                                                                           self.model_parameters.trig_bg[0],
                                                                           nominal)
            self.model_parameters.background_spectrum_bg = background_spectrum_bg

        if self.model_parameters.incident_neutron_spectrum_f is None or overwrite:
            incident_neutron_spectrum_f = approximate_neutron_spectrum_Li6det(exp_model.energy_grid, 
                                                                            smooth, #self.options.smoothTNCS, 
                                                                            exp_model.FP[0],
                                                                            exp_model.t0[0],
                                                                            self.model_parameters.trig_f[0])
            
            self.model_parameters.incident_neutron_spectrum_f = incident_neutron_spectrum_f
            # self.model_parameters.incident_neutron_spectrum_f = incident_neutron_spectrum_f

        if self.model_parameters.background_spectrum_bf is None or overwrite:
            background_spectrum_bf = approximate_gamma_background_spectrum(exp_model.energy_grid, 
                                                                           smooth, 
                                                                           exp_model.FP[0], 
                                                                           exp_model.t0[0], 
                                                                           self.model_parameters.trig_bf[0],
                                                                           nominal)
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

        if true_model_parameters.incident_neutron_spectrum_f is None:
            raise ValueError("incident neutron flux spectrum is None, please provide this data or approximate it using the method in this class")
        if len(true_model_parameters.incident_neutron_spectrum_f) != len(pw_true):
            raise ValueError("neutron flux spectrum and sample data are not of the same length, check energy domain")
        if true_model_parameters.background_spectrum_bg is None:
            raise ValueError("background gamma spectrum is None, please provide this data or approximate it using the method in this class")
        if len(true_model_parameters.background_spectrum_bg) != len(pw_true):
            raise ValueError("gamma background spectrum for target capture measurement and sample data are not of the same length, check energy domain")
        if true_model_parameters.background_spectrum_bf is None:
            raise ValueError("background neutron flux spectrum is None, please provide this data or approximate it using the method in this class")
        if len(true_model_parameters.background_spectrum_bf) != len(pw_true):
            raise ValueError("gamma background spectrum for flux yield measurement and sample data are not of the same length, check energy domain")


        raw_data = deepcopy(pw_true)
        true_gamma_counts = inverse_reduction(pw_true, true_model_parameters)

        if options.sample_counting_noise:
            c = pois_noise(true_gamma_counts)
        else:
            c = true_gamma_counts

        if options.force_zero_to_1:
            c[c==0] = 1
        else:
            if np.any(c==0):
                raise ValueError("Syndat Option force_zero_to_1 is set to false and you have bins with 0 counts")

        dc = np.sqrt(c)

        raw_data.loc[:, 'ctg_true'] = true_gamma_counts
        raw_data.loc[:, 'ctg'] = c
        raw_data.loc[:, 'dctg'] = dc
        


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

        diag_tot = unc_data**2
        if options.calculate_covariance:
            CovY = np.diag(diag_tot)
            CovY = pd.DataFrame(CovY, columns=Yg.E, index=Yg.E)
            CovY.index.name = None
            Yg['exp_unc'] = np.sqrt(np.diag(CovY))
            covariance_data = {}
            if options.explicit_covariance:
                covariance_data['CovY'] = CovY
            else:
                raise ValueError("not implemented")

        else:
            diag_tot = unc_data
            Yg.loc[:,'exp_unc'] = diag_tot
            covariance_data = {}

        ## fix for zero gamma counts
        # Yg = Yg.loc[Yg.exp!=0]
        # assert(np.all(Yg.exp!=0))

        return Yg, covariance_data, raw_data