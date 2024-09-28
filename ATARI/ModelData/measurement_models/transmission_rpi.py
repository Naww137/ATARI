import numpy as np
import pandas as pd
from ATARI.syndat.general_functions import *
from ATARI.ModelData.structuring import parameter, vector_parameter
from ATARI.syndat.data_classes import syndatOPT
from copy import copy

# ========================================================================================
#         Transmission reduction/generation functions for RPI methodology
# ========================================================================================

def transmission(cr,Cr, Bi, k,K, b0,B0, alpha):
    [m1,m2,m3,m4] = alpha
    return (m1*cr - m2*k*Bi - b0) / (m3*Cr - m4*K*Bi - B0) 


def get_covT(tof, c,C, dc,dC, a,b, k,K, Bi, b0,B0, alpha, sys_unc_inp, ab_cov, calc_cov, bkg_func):
        """
        Calculates the output covariance matrix of transmission from input uncertainties.

        This function uses the covariance sandwhich rule:
        .. math:: C_y = J^T*C_x*J
        to propagate input variance-covariance to transmission data

        Parameters
        ----------
        tof : array-like
            Array of time of flight values for each data point in microseconds.
        c : float
            Count rate for sample in.
        C : float
            Count rate for sample out.
        dc : float
            Uncertainty in the count rate for sample-in.
        dC : array-like
            Uncertainty in the count rate for sample-out.
        a : float
            Shaping parameter for exponential background function.
        b : float
            Shaping parameter for exponential background function.
        k : float
            Background normalization for sample in.
        K : float
            Background normalization for sample out.
        Bi : array-like
            Background shape function stored as a vector.
        b0 : float
            Constant background for sample in.
        B0 : float
            Constant background for sample out.
        alpha : array-like
            Vector of monitor stability factors [m1,m2,m3,m4]
        sys_unc : array-like
            Vector of systematic uncertainties: [da,db,dk_i,dk_o,dB0_i,dB0_o,m1,m2,m3,m4].
        ab_cov  : float
            Covariance between a & b background function parameters.
        calc_cov : bool
            Option to calculate covariances, if false, only the diagonal of the covariance matrix will be calculated.
        bkg_func : str
            Background function to use, options are 'exp' or 'power'.
        
        Notes
        -----
        Background function must be of form Bi = a*exp(-b). Explicitly coded derivatives for Jacobian.
        
        Returns
        -------
        Output variance/covariance data for transmission.
        """
        
        # cast all into numpy arrays
        c = np.array(c); dc = np.array(dc)
        C = np.array(C); dC = np.array(dC)
        tof = np.array(tof)
        Bi = np.array(Bi)
        sys_unc_inp = np.array(sys_unc_inp)

        # numerator and denominator
        D = alpha[2]*C - alpha[3]*K*Bi - B0
        N = alpha[0]*c - alpha[1]*k*Bi - b0
        
        # construct statistical covariance and jacobian
        dTi_dci = alpha[0]/D
        dTi_dCi = -N*alpha[2]/D**2
        diag_stat = (dc**2)*(dTi_dci**2) + (dC**2)*(dTi_dCi**2)
        
        # systematic derivatives dependent on background function
        if bkg_func == 'exp':
            dTi_da = (-k*alpha[1]*D+K*alpha[3]*N)*np.exp(-b*tof) / (D**2)
            dTi_db =  (k*alpha[1]*D-N*alpha[3]*K)*Bi*tof / (D**2)
        elif bkg_func == 'power':
            dTi_da = (-k*alpha[1]*D + K*alpha[3]*N)*tof**(-b) / (D**2)
            dTi_db = (k*alpha[1]*D - K*alpha[3]*N)*Bi*np.log(tof) / (D**2) 
        else:
            raise ValueError("Background function not regocnized")
        # more systematic derivatives
        dTi_dk = -alpha[1]*Bi/D       
        dTi_dK = N*alpha[3]*Bi/D**2
        dTi_db0 = -1/D
        dTi_dB0 = N/D**2
        dTi_dalpha = [ c/D, -k*Bi/D, -C*N/D**2, K*Bi*N/D**2 ]


        sys_unc = [ab_cov[0][0]]+[ab_cov[1][1]] + list(sys_unc_inp)
        diag_sys = (sys_unc[0]**2)*(dTi_da**2) + (sys_unc[1]**2)*(dTi_db**2) + (sys_unc[2]**2)*(dTi_dk**2) + (sys_unc[3]**2)*(dTi_dK**2) + (sys_unc[4]**2)*(dTi_db0**2) \
                + (sys_unc[5]**2)*(dTi_dB0**2) + (sys_unc[6]**2)*(dTi_dalpha[0]**2) + (sys_unc[7]**2)*(dTi_dalpha[1]**2) + (sys_unc[8]**2)*(dTi_dalpha[2]**2) + (sys_unc[9]**2)*(dTi_dalpha[3]**2)

        Jac_sys = None
        Cov_sys = None
    
        if calc_cov:

            Cov_sys = np.diag(sys_unc_inp**2)
            ab_cov = np.array(ab_cov)
            Cov_sys = np.block([
                                [ab_cov,                                        np.zeros((ab_cov.shape[0], Cov_sys.shape[1]))   ],
                                [np.zeros((Cov_sys.shape[0],ab_cov.shape[1])),  Cov_sys                                         ]
                            ])   
            Jac_sys = np.array([dTi_da, dTi_db, dTi_dk, dTi_dK, dTi_db0, dTi_dB0, dTi_dalpha[0], dTi_dalpha[1],dTi_dalpha[2],dTi_dalpha[3]])

            # CovT_sys = Jac_sys.T @ Cov_sys @ Jac_sys
            ### T covariance is sum of systematic and statistical covariance 
            # CovT = np.diag(diag_stat) + CovT_sys

        return [diag_stat, diag_sys, Jac_sys, Cov_sys]

    
def reduce_raw_count_data(tof, c,C, dc,dC, bw, trigo,trigs, a,b, k,K, Bi, b0,B0, alpha, sys_unc, ab_cov, calc_cov, bkg_func):
    """
    Reduces raw count data to transmission data with propagated uncertainty.

    This function uses the covariance sandwhich rule:
    .. math:: C_y = J^T*C_x*J
    to propagate input variance-covariance from both statistical uncertainties
    and systematic uncertainties to transmission data.

    Parameters
    ----------
    tof : array-like
        Array of time of flight values for each data point in microseconds.
    c : float
        Count rate for sample in.
    C : float
        Count rate for sample out.
    dc : float
        Uncertainty in the count rate for sample-in.
    dC : array-like
        Uncertainty in the count rate for sample-out.
    bw : float
        Width in time a given channel is open, bin width.
    trig : int
        Number of times the LINAC is fired, corresponding to the number of times each channel is openned for counts.
    a : float
        Shaping parameter for exponential background function.
    b : float
        Shaping parameter for exponential background function.
    k : float
        Background normalization for sample in.
    K : float
        Background normalization for sample out.
    Bi : array-like
        Background shape function stored as a vector.
    b0 : float
        Constant background for sample in.
    B0 : float
        Constant background for sample out.
    alpha : array-like
        Vector of monitor stability factors [m1,m2,m3,m4]
    sys_unc : array-like
        Vector of systematic uncertainties: [da,db,dk_i,dk_o,dB0_i,dB0_o,m1,m2,m3,m4].
    ab_cov  : float
        Covariance between a & b background function parameters.
    calc_cov : bool
        Option to calculate covariances, if false, only the diagonal of the covariance matrix will be calculated.
    
    Notes
    -----
    Background function must be of form Bi = a*exp(-b)
    
    Returns
    -------
    Tn : array-like
        Transmission data.
    dT

    """
    # calculate count rate and propagate uncertainty
    Cr, dCr = cts_to_ctr(C, dC, bw, trigo) 
    cr, dcr = cts_to_ctr(c, dc, bw, trigs)
    rates = Cr,dCr, cr,dcr

    # calculate transmission
    Tn = transmission(cr,Cr, Bi, k,K, b0,B0, alpha)

    # propagate uncertainty to transmission
    unc_data = get_covT(tof, cr,Cr, dcr,dCr, a,b, k,K, Bi, b0,B0, alpha, sys_unc, ab_cov, calc_cov, bkg_func)
    
    return Tn, unc_data, rates


def inverse_reduction(sample_df, open_df, add_noise, sample_turp, trigo,trigs, k,K, Bi, b0,B0, alpha, dm1):
    """
    Generates raw count data for sample-in given a theoretical tranmission. 
    
    This function performs the inverse of the reduction process, calculating raw count data
    from a theoretical transmission. This process requires the assumption of known,
    true underlying reduction parameters and open count data.

    Parameters
    ----------
    sample_df : pandas.DataFrame
        Sample in dataframe with a column for theoretical tranmission ['true'] and energy ['E'].
    open_df : pandas.DataFrame
        Open dataframe, columns ['E'], ['bw']
    add_noise : bool
        Option to sample noise on expected counts.
    sample_turp : bool
        Option to sample true underlying resonance parameters - here this is monitor corrections.
    trigo : int
        Number of times the LINAC is fired for the open counts, corresponding to the number of times each channel is openned for counts.
    trigs : int
        Number of times the LINAC is fired for the sample-in counts, corresponding to the number of times each channel is openned for counts.
    k : float
        Background normalization for sample in.
    K : float
        Background normalization for sample out.
    Bi : array-like
        Background shape function stored as a vector.
    b0 : float
        Constant background for sample in.
    B0 : float
        Constant background for sample out.
    alpha : array-like
        Vector of monitor stability factors [m1,m2,m3,m4]
    
    Returns
    -------
    sample_df : pandas.DataFrame
        Dataframe containing data for sample in.
    open_df : pandas.DataFrame
        Dataframe containing data for sample out.
    """
    # calculate open count rates
    Cr, dCr = cts_to_ctr(open_df.ct, open_df.dct, open_df.bw, trigo) # cts_o/(bw*trig)
    open_df['c'] = Cr; 
    open_df['dc'] = dCr
    
    # calculate sample in count rate from theoretical transmission, bkg, m,k, and open count rate
    [m1,m2,m3,m4] = alpha
    cr = (sample_df["true"]*(m3*Cr - m4*K*Bi - B0) + m2*k*Bi + b0)/m1
    sample_df['c_true'] = cr

    # calculate expected sample in counts from count rate
    c = cr*open_df.bw*trigs
    theo_c = c
    # if random monitor norms caused negative counts force to 0
    theo_c = np.where(theo_c > 0, theo_c, 0)

    # split into cycles and apply monitor normalizations
            # TODO: also include deadtime corrections at each cycle
            # TODO: update function to take in # of cycles and std of monitor normalizations
    cycles = 35
    c_cycle = theo_c/cycles
    
    if sample_turp:
        monitor_factors = np.random.default_rng().normal(1,dm1, size=cycles)
    else:
        monitor_factors = np.ones((cycles))

    if add_noise:
        c = pois_noise(c_cycle)*monitor_factors[0]
        for i in range(cycles-1):
            c += pois_noise(c_cycle)*monitor_factors[i]
    else:
        c = c_cycle*monitor_factors[0]
        for i in range(cycles-1):
            c += c_cycle*monitor_factors[i]

    # if options.force_zero_to_1:
    #             c[c==0] = 1
    #         else:
    #             if np.any(c==0):
    #                 raise ValueError("Syndat Option force_zero_to_1 is set to false and you have bins with 0 counts")
    assert(c.all() >= 0)
    dc = np.sqrt(c)
    
    sample_df['cts'] = c
    sample_df['dcts'] = dc
    sample_df['cts_true'] = theo_c
    sample_df['bw'] = open_df.bw

    return sample_df, open_df




# ========================================================================================
#            Default parameters class
# ========================================================================================


class transmission_rpi_parameters:

    trigo = parameter()
    trigs = parameter()
    m1    = parameter()
    m2    = parameter()
    m3    = parameter()
    m4    = parameter()
    ks    = parameter()
    ko    = parameter()
    b0s   = parameter()
    b0o   = parameter()
    a_b   = parameter()
    open_neutron_spectrum = vector_parameter()

    def __init__(self, **kwargs):

        self.trigo  = (9758727,      0)
        self.trigs  = (18476117,     0)
        self.m1     = (1,            0.016)
        self.m2     = (1,            0.008)
        self.m3     = (1,            0.018)
        self.m4     = (1,            0.005)
        self.ks     = (0.563,        0.02402339737495515)
        self.ko     = (1.471,        0.05576763648617445)
        self.b0s    = (9.9,          0.1)
        self.b0o    = (13.4,         0.7)
        self.a_b    = ([582.7768594580712, 0.05149689096209191],
                        [[1.14395753e+03,  1.42659922e-1],
                         [1.42659922e-1,   2.19135003e-05]])
        
        self.open_neutron_spectrum = None

        self._bkg_func = 'exp'
        
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def bkg_func(self) -> str:
        return self._bkg_func
    @bkg_func.setter
    def bkg_func(self, bkg_func):
        if bkg_func not in ['exp', 'power']:
            raise ValueError("background function set on bkg_func not recognized")
        self._bkg_func = bkg_func

    def sample_parameters(self, true_model_parameters: dict):
        sampled_params = {}

        for param_name, param_values in self.__dict__.items():
            if param_name in true_model_parameters:
                assert true_model_parameters[param_name][1] == 0.0, "provided true model parameter with non-zero uncertainty"
                sampled_params[param_name] = true_model_parameters[param_name]
            else:
                if isinstance(param_values, tuple) and len(param_values) == 2:
                    mean, uncertainty = param_values
                    if np.all(np.array(uncertainty) == 0):
                        sample = mean
                    else:
                        if param_name == 'a_b':
                            sample = np.random.multivariate_normal(mean, uncertainty)
                        else:
                            sample = np.random.normal(loc=mean, scale=uncertainty)
                    sampled_params[param_name] = (sample, 0.0)
                if isinstance(param_values, pd.DataFrame):
                    new_c = np.random.normal(loc=param_values.ct, scale=param_values.dct)
                    df = deepcopy(param_values)
                    df['ct'] = new_c
                    df['dct'] = np.sqrt(new_c)
                    sampled_params[param_name] = df

        return transmission_rpi_parameters(**sampled_params)
    

# true_par = {'m1':1.0}
# test = transmission_rpi_parameters(neutron_spectrum=pd.DataFrame({'c':np.ones(5),
#                                                                  'dc':np.ones(5)*0.01}))
# out = test.sample_parameters(true_par)
# print(out.m1)
# test.m1= (2.0, 0.1)
# print(test.m1)
# test1 = transmission_rpi_parameters()
# print(test1.m1)




# ========================================================================================
#            Handler class
# ========================================================================================

class Transmission_RPI:
    """
    Handler class for the rpi tranmission measurement model. 
    This holds parameters and methods used to both generate raw observable data and reduce it to transmission.
    """

    def __init__(self, **kwargs):
        self._model_parameters = transmission_rpi_parameters(**kwargs)
        self._covariance_data = {}

    @property
    def model_parameters(self) -> transmission_rpi_parameters:
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
        if min(new_energy_range) < min(self.model_parameters.open_neutron_spectrum.E):
            # raise ValueError("new energy range is less than existing transmission_rpi.open_netron_spectrum energy")
            print(f"WARNING: new energy min {min(new_energy_range)} is less than existing transmission_rpi.open_netron_spectrum energy {min(self.model_parameters.open_neutron_spectrum.E)}")
        if max(new_energy_range) > max(self.model_parameters.open_neutron_spectrum.E):
            # raise ValueError("new energy range is more than existing transmission_rpi.open_netron_spectrum energy")
            print(f"WARNING: new energy max {max(new_energy_range)} is more than existing transmission_rpi.open_netron_spectrum energy {max(self.model_parameters.open_neutron_spectrum.E)}")
        self.model_parameters.open_neutron_spectrum = self.model_parameters.open_neutron_spectrum.loc[(self.model_parameters.open_neutron_spectrum.E.values < max(new_energy_range)) & (self.model_parameters.open_neutron_spectrum.E.values > min(new_energy_range))].copy()
        return
    

    def approximate_unknown_data(self, exp_model, smooth, check_trig = False, overwrite = False):
        if self.model_parameters.open_neutron_spectrum is None or overwrite:
            open_neutron_spectrum = approximate_neutron_spectrum_Li6det(exp_model.energy_grid, 
                                                                    smooth, #self.options.smoothTNCS, 
                                                                    exp_model.FP[0],
                                                                    exp_model.t0[0],
                                                                    self.model_parameters.trigo[0])
            
            self.model_parameters.open_neutron_spectrum = open_neutron_spectrum
  

    def generate_raw_data(self,
                          pw_true,
                          true_model_parameters, # need to build better protocol for this 
                          options: syndatOPT
                          ) -> pd.DataFrame:
        """
        !!! Document !!!!
        """
        assert true_model_parameters.open_neutron_spectrum is not None
        if len(true_model_parameters.open_neutron_spectrum) != len(pw_true):
            raise ValueError("neutron spectrum and sample data are not of the same length, check energy domain")
        if not np.all(np.isclose(pw_true.E.values, true_model_parameters.open_neutron_spectrum.E.values, rtol=1e-5)):
            raise ValueError(f"neutron spectrum and sample data are not on the same grid, check energy domain, max difference = {np.max(abs(pw_true.E.values-true_model_parameters.open_neutron_spectrum.E.values))}")
        
        # gather monitor array
        monitor_array = [true_model_parameters.m1[0], true_model_parameters.m2[0], true_model_parameters.m3[0], true_model_parameters.m4[0]]
        
        # calculate background - uses time-of-flight in microseconds
        true_Bi = neutron_background_function(true_model_parameters.open_neutron_spectrum.tof*1e-3, 
                                              true_model_parameters.a_b[0][0], 
                                              true_model_parameters.a_b[0][1],
                                              self.model_parameters.bkg_func)

        sample_data, open_data = inverse_reduction(copy(pw_true) ,
                                                    true_model_parameters.open_neutron_spectrum ,
                                                    options.sample_counting_noise ,
                                                    options.sampleTMP ,
                                                    true_model_parameters.trigo[0],
                                                    true_model_parameters.trigs[0],
                                                    true_model_parameters.ks[0],
                                                    true_model_parameters.ko[0],
                                                    true_Bi,
                                                    true_model_parameters.b0s[0],
                                                    true_model_parameters.b0o[0],
                                                    monitor_array, true_model_parameters.m1[1])
        
        raw_data = copy(sample_data)

        raw_data['bs_true'] = true_model_parameters.m2[0]*true_Bi*true_model_parameters.ks[0] + true_model_parameters.b0s[0]
        raw_data['bo_true'] = true_model_parameters.m4[0]*true_Bi*true_model_parameters.ko[0] + true_model_parameters.b0o[0]

        raw_data['cto_true'] = open_data['ct']
        raw_data['dcto_true'] = open_data['dct']
        raw_data['co_true'] = open_data['c']
        raw_data['dco_true'] = open_data['dc']

        return raw_data



    def reduce_raw_data(self, 
                        raw_data,  
                        options):
        """
        Reduces the raw count data (sample in/out) to Transmission data and propagates uncertainty.

        """
        if self.model_parameters.open_neutron_spectrum is None:
            raise ValueError("background neutron flux spectrum is None, please provide this data or approximate it using the method in this class")
        assert isinstance(self.model_parameters.open_neutron_spectrum, pd.DataFrame)
        assert(np.isclose(max(abs(self.model_parameters.open_neutron_spectrum.tof.values - raw_data.tof.values)/raw_data.tof.values), 0,atol=1e-7))

        ### complete raw dataframe with count rates using reductive model parameters
        cr, dcr = cts_to_ctr(raw_data.cts, raw_data.dcts, raw_data.bw, self.model_parameters.trigs[0])
        raw_data['cs']  = cr
        raw_data['dcs'] = dcr
        Cr, dCr = cts_to_ctr(self.model_parameters.open_neutron_spectrum.ct, self.model_parameters.open_neutron_spectrum.dct, self.model_parameters.open_neutron_spectrum.bw, self.model_parameters.trigo[0])
        raw_data['co']  = Cr
        raw_data['dco'] = dCr

        # create transmission object
        trans = pd.DataFrame()
        trans['tof'] = raw_data.tof
        trans['E'] = raw_data.E
        trans['true'] = raw_data.true

        # calculate estimated background function - tof in microseconds
        Bi = neutron_background_function(self.model_parameters.open_neutron_spectrum.tof*1e-3, 
                                         self.model_parameters.a_b[0][0], 
                                         self.model_parameters.a_b[0][1], 
                                         self.model_parameters.bkg_func)
        raw_data['bs'] = self.model_parameters.m2[0]*Bi*self.model_parameters.ks[0] + self.model_parameters.b0s[0]
        raw_data['bo'] = self.model_parameters.m4[0]*Bi*self.model_parameters.ko[0] + self.model_parameters.b0o[0]

        # define systematic uncertainties
        sys_unc = [getattr(self.model_parameters, attr)[1]for attr in ['ks', 'ko', 'b0s', 'b0o', 'm1', 'm2', 'm3', 'm4']]

        # gather monitor array and reduce data - again tof should be in microseconds
        monitor_array = [self.model_parameters.m1[0], self.model_parameters.m2[0], self.model_parameters.m3[0], self.model_parameters.m4[0]]
        trans['exp'], unc_data, rates = reduce_raw_count_data(raw_data.tof.values*1e-3,
                                                              raw_data.cts.values, 
                                                              self.model_parameters.open_neutron_spectrum.ct.values,
                                                              raw_data.dcts.values, 
                                                              self.model_parameters.open_neutron_spectrum.dct.values,
                                                              self.model_parameters.open_neutron_spectrum.bw.values,
                                                              self.model_parameters.trigo[0],
                                                              self.model_parameters.trigs[0],
                                                              self.model_parameters.a_b[0][0],
                                                              self.model_parameters.a_b[0][1],
                                                              self.model_parameters.ks[0],
                                                              self.model_parameters.ko[0],
                                                              Bi,
                                                              self.model_parameters.b0s[0],
                                                              self.model_parameters.b0o[0],
                                                              monitor_array,
                                                              sys_unc,
                                                              self.model_parameters.a_b[1],
                                                              options.calculate_covariance,
                                                              self.model_parameters.bkg_func)

        diag_stat, diag_sys, Jac_sys, Cov_sys = unc_data

        if options.calculate_covariance:
            # TODO: update options to be like sammy - if not explicit covariance then trans['exp_unc'] should be statistical error
            CovT_sys = Jac_sys.T @ Cov_sys @ Jac_sys
            CovT = np.diag(diag_stat) + CovT_sys
            CovT = pd.DataFrame(CovT, columns=trans.E, index=trans.E)
            CovT.index.name = None
            trans['exp_unc'] = np.sqrt(np.diag(CovT))
            covariance_data = {}
        
            if options.explicit_covariance:
                covariance_data['CovT'] = CovT
            else:
                pass

            covariance_data['Cov_sys'] = Cov_sys
            Jac_sys = pd.DataFrame(Jac_sys, columns=trans.E)
            covariance_data['Jac_sys'] = Jac_sys
            covariance_data["diag_stat"] = pd.DataFrame({'var_stat':diag_stat}, index=trans.E)
        
        
        else:
            trans['exp_unc'] = np.sqrt(diag_stat + diag_sys)
            covariance_data = {}

        return trans, covariance_data, raw_data



