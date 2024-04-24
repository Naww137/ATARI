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



def get_covT(crg,dcrg,brg,dbrg,cr_flux,dcr_flux,br_flux,dbr_flux,Y_flux,dY_flux,fn,dfn,Yield,covariance_data,calc_cov,calc_matrix):
    """Calculates the uncertainty on Yield with or without covariance.

    Args:
        crg (Array-Like): _description_
        dcrg (Array-Like): _description_
        brg (Array-Like): _description_
        dbrg (Array-Like): _description_
        cr_flux (Array-Like): _description_
        dcr_flux (Array-Like): _description_
        br_flux (Array-Like): _description_
        dbr_flux (Array-Like): _description_
        Y_flux (Array-Like): _description_
        dY_flux (Array-Like): _description_
        fn (Float): _description_
        dfn (Float): _description_
        Yield (Array-Like): The calculated gamma yield
        cr_flux_cov (Float): General covariance between each flux gross count rate
        br_flux_cov (Float): General covariance between each flux background count rate
        calc_cov (Boolean): If true, the function will calculate full covariance
    """
    #Defining derivatives that are used in either method
    cr_flux_der=-Yield/(cr_flux-br_flux)
    br_flux_der=-cr_flux_der
        
    
    #Defining non-covariance derivatives
    crg_der=fn*(Y_flux/(cr_flux-br_flux))
    brg_der=-crg_der
    Y_flux_der=fn*((crg-brg)/(cr_flux-br_flux))
    fn_der=Yield/fn
    Yield_Uncertainty=np.sqrt(np.power(crg_der*dcrg,2)+
                                np.power(brg_der*dbrg,2)+
                                np.power(Y_flux_der*dY_flux,2)+
                                np.power(cr_flux_der*dcr_flux,2)+
                                np.power(br_flux_der*dbr_flux,2)+
                                np.power(fn_der*dfn,2))
    
    if(calc_cov):
        covariance_data=dict(covariance_data)
        #Calculates uncertainty for elements without covariance treating it as one combined element, Q
        crg_Q_der=fn*Y_flux
        brg_Q_der=-crg_Q_der
        Y_flux_Q_der=fn*(crg-brg)
        fn_Q_der=(crg-brg)*Y_flux
        
        dQ=np.sqrt(np.power(crg_Q_der*dcrg,2)+
                   np.power(brg_Q_der*dbrg,2)+
                   np.power(Y_flux_Q_der*dY_flux,2)+
                   np.power(fn_Q_der*dfn,2))
        
        Q_der=1/(cr_flux-br_flux)
        
        Jacobian=np.concatenate((np.diag(cr_flux_der),np.diag(br_flux_der),np.diag(Q_der)),1)
        
        #Fills out the Covariance blocks for the flux count rates
        dcr_flux_cov=np.ones((len(dcr_flux),len(dcr_flux)))*covariance_data["cr_flux_cov"]
        np.fill_diagonal(dcr_flux_cov,np.array(dcr_flux)**2)
        
        dbr_flux_cov=np.ones((len(dbr_flux),len(dbr_flux)))*covariance_data["br_flux_cov"]
        np.fill_diagonal(dbr_flux_cov,np.array(dbr_flux)**2)
        
        
        Covariance_matrix=np.block([[dcr_flux_cov,                            np.zeros((len(dcr_flux),len(dbr_flux))),np.zeros((len(dcr_flux),len(dQ)))],
                                  [np.zeros((len(dbr_flux),len(dcr_flux))), dbr_flux_cov,                           np.zeros((len(dbr_flux),len(dQ)))],
                                  [np.zeros((len(dQ),len(dcr_flux))),       np.zeros((len(dQ),len(dbr_flux))),      np.diag(dQ**2)]])
        
        if(calc_matrix):
            Yield_Covariance=Jacobian@Covariance_matrix@Jacobian.T
            covariance_data["Cov_Y"]=Yield_Covariance
        else:
            covariance_data["Cov_Y_jac"]=Jacobian.T
            covariance_data["Cov_Y_cov"]=Covariance_matrix
    return(Yield_Uncertainty,covariance_data)


    
def reduce_raw_count_data(raw_data, model_parameters, covariance_data, calc_cov, calc_matrix):

    ### counts to count rates for target gamma and background measurements
    crg, dcrg = cts_to_ctr(raw_data.ctg,
                           raw_data.dctg,
                           model_parameters.incident_neutron_spectrum_f.bw,
                           model_parameters.trig_g[0])
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
    relative_flux_rate = (cr_flux - br_flux)/model_parameters.yield_flux.ct
    Yield             = model_parameters.fn[0] * np.divide(crg - brg, relative_flux_rate)
    
    Yield_uncertainty, CovY=get_covT(crg,dcrg,brg,dbrg,cr_flux,dcr_flux,br_flux,dbr_flux,model_parameters.yield_flux.ct,model_parameters.yield_flux.dct,model_parameters.fn[0],model_parameters.fn[1],Yield,covariance_data,calc_cov,calc_matrix)
    # diag_stat = None
    # diag_sys = None
    # data =[diag_stat, diag_sys, Jac_sys, Cov_sys]
    

    return Yield, Yield_uncertainty, CovY



def inverse_reduction(pw_true, model_parameters):
    
    #turns 0D scaling factoring into a 1D scaling factor so that it can align with the other 1D values while calculating
    # scaling=pd.DataFrame({'c'    :   np.ones(len(pw_true.E))*TURP.Scaling[0],
    #                       'dc'   :   np.ones(len(pw_true.E))*TURP.Scaling[1]})
    
    ### relative flux rate measurement
    cr_flux, ___ = cts_to_ctr(model_parameters.incident_neutron_spectrum_f.ct,
                                model_parameters.incident_neutron_spectrum_f.dct,
                                model_parameters.incident_neutron_spectrum_f.bw,
                                model_parameters.trig_f[0])
    br_flux, ___ = cts_to_ctr(model_parameters.background_spectrum_bf.ct,
                                model_parameters.background_spectrum_bf.dct,
                                model_parameters.incident_neutron_spectrum_f.bw,
                                model_parameters.trig_bf[0])

    relative_flux_rate = (cr_flux - br_flux)/model_parameters.yield_flux.ct
    
    ### target gamma background and count rate
    br_gamma, ___ = cts_to_ctr(model_parameters.background_spectrum_bg.ct,
                                    model_parameters.background_spectrum_bg.dct,
                                    model_parameters.incident_neutron_spectrum_f.bw,
                                    model_parameters.trig_bg[0])
    cr_gamma_true = (pw_true.true * relative_flux_rate)/model_parameters.fn[0] + br_gamma
    
    ### target gamma count rate to counts and add uncertainty
    c_true = cr_gamma_true*model_parameters.incident_neutron_spectrum_f.bw*model_parameters.trig_g[0]
    
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
    yield_flux = vector_parameter()

    def __init__(self, **kwargs):

        self.trig_g     =  (1.2e7,   0)
        self.trig_bg    =  (0.8e7,  0)
        self.trig_f     =  (1e7,   0)
        self.trig_bf    =  (0.8e7,  0)
        self.fn         =  (1,          0)
    
        self.background_spectrum_bg = None
        self.incident_neutron_spectrum_f = None
        self.background_spectrum_bf = None
        self.yield_flux = None

        for key, value in kwargs.items():
            setattr(self, key, value)


    def sample_parameters(self, true_model_parameters: dict):
        sampled_params = {}

        for param_name, param_values in self.__dict__.items():
            if param_name in true_model_parameters:
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    sampled_params[param_name] = (true_model_parameters[param_name], 0.0)
                if isinstance(param_values, pd.DataFrame):
                    df = deepcopy(param_values)
                    df.loc[:,'ct'] = true_model_parameters[param_name]
                    df.loc[:,'dct'] = np.zeros(len(true_model_parameters[param_name]))
                    sampled_params[param_name] = df
            else:
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    mean, uncertainty = param_values
                    if uncertainty == 0:
                        sample = mean
                    else:
                        sample = np.random.normal(loc=mean, scale=uncertainty)
                    sampled_params[param_name] = (sample, uncertainty)
                if isinstance(param_values, pd.DataFrame):
                    new_c = np.random.poisson(param_values.ct)#, scale=param_values.dc)
                    df = deepcopy(param_values)
                    df['ct'] = df['ct'].astype(float) # we are sampling counts as floats
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
                                                                           self.model_parameters.trig_bg[0])
            #As these are taken as true, we need to set the uncertainty to zero for the uncertainty calculations to work properly.
            background_spectrum_bg.dct=np.zeros(len(background_spectrum_bg.dct))
            self.model_parameters.background_spectrum_bg = background_spectrum_bg

        if self.model_parameters.incident_neutron_spectrum_f is None or overwrite:
            incident_neutron_spectrum_f = approximate_neutron_spectrum_Li6det(exp_model.energy_grid, 
                                                                            smooth, #self.options.smoothTNCS, 
                                                                            exp_model.FP[0],
                                                                            exp_model.t0[0],
                                                                            self.model_parameters.trig_f[0])
            incident_neutron_spectrum_f.dct=np.zeros(len(incident_neutron_spectrum_f.dct))
            self.model_parameters.incident_neutron_spectrum_f = incident_neutron_spectrum_f
            # self.model_parameters.incident_neutron_spectrum_f = incident_neutron_spectrum_f

        if self.model_parameters.background_spectrum_bf is None or overwrite:
            background_spectrum_bf = approximate_gamma_background_spectrum(exp_model.energy_grid, 
                                                                           smooth, 
                                                                           exp_model.FP[0], 
                                                                           exp_model.t0[0], 
                                                                           self.model_parameters.trig_bf[0])
            background_spectrum_bf.dct=np.zeros(len(background_spectrum_bf.dct))
            self.model_parameters.background_spectrum_bf = background_spectrum_bf
        
        if self.model_parameters.yield_flux is None:
            yield_flux = pd.DataFrame({'E':exp_model.energy_grid,
                                       'tof':self.model_parameters.background_spectrum_bg.tof,
                                       'bw':self.model_parameters.background_spectrum_bg.bw,
                                       'ct':np.ones(len(exp_model.energy_grid))*1000,
                                       'dct':np.zeros(len(exp_model.energy_grid))})
            self.model_parameters.yield_flux = yield_flux



    def generate_raw_data(self,
                          pw_true,
                          model_parameters, # need to build better protocol for this 
                          sample_counting_noise,
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

        assert model_parameters.incident_neutron_spectrum_f is not None
        if len(model_parameters.incident_neutron_spectrum_f) != len(pw_true):
            raise ValueError("neutron flux spectrum and sample data are not of the same length, check energy domain")
        assert model_parameters.background_spectrum_bg is not None
        if len(model_parameters.background_spectrum_bg) != len(pw_true):
            raise ValueError("gamma background spectrum for target capture measurement and sample data are not of the same length, check energy domain")
        assert model_parameters.background_spectrum_bf is not None
        if len(model_parameters.background_spectrum_bf) != len(pw_true):
            raise ValueError("gamma background spectrum for flux yield measurement and sample data are not of the same length, check energy domain")


        raw_data = deepcopy(pw_true)
        true_gamma_counts = inverse_reduction(pw_true, model_parameters)

        if sample_counting_noise:
            c = pois_noise(true_gamma_counts)
            dc = np.sqrt(c)
            
        else:
            c = true_gamma_counts
            dc = np.zeros(len(c))
            
        c[c==0] = 1
        # assert(all(c> 0))

        raw_data.loc[:, 'ctg_true'] = true_gamma_counts
        raw_data.loc[:, 'ctg'] = c
        raw_data.loc[:, 'dctg'] = dc
        


        return raw_data
        
    

    def reduce_raw_data(self, pw_true, raw_data, model_parameters, calculate_covariance, calculate_covariance_matrix=False): # neutron_spectrum, reduction_parameters, countrate, background):
        """
        Reduces the raw count data (sample in/out) to Transmission data and propagates uncertainty.

        Parameters
        ----------
        neutron_spectrum:
            A dataframe that describes the neutron count spectrum for this problem.
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
        Yg = deepcopy(pw_true)

        # estimated background function
        #Bi = gamma_background_function()
        #We are just going to pass the background from the inverse reduction here since it's the same thing anyways

        # define systematic uncertainties
        #I will implement covariance later once we get to defining that.
        Yg.loc[:,'exp'], Yg.loc[:,'exp_unc'], CovY = reduce_raw_count_data(raw_data, model_parameters, self.covariance_data, calculate_covariance, calculate_covariance_matrix)
        ## fix for zero gamma counts
        # Yg = Yg.loc[Yg.exp!=0]
        # assert(np.all(Yg.exp!=0))

        return Yg, CovY