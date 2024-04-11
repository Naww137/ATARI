import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e
from ATARI.models import experimental_model




# ========================================================================================
#            Transmission reduction equations at RPI
# ========================================================================================

def transmission(cr,Cr, Bi, k,K, b0,B0, alpha):
    [m1,m2,m3,m4] = alpha
    return (m1*cr - m2*k*Bi - b0) / (m3*Cr - m4*K*Bi - B0) 


def get_covT(tof, c,C, dc,dC, a,b, k,K, Bi, b0,B0, alpha, sys_unc, ab_cov, calc_cov):
        """
        Calculates the output covariance matrix of transmission from input uncertainties.

        This function uses the covariance sandwhich rule:
        .. math:: C_y = J^T*C_x*J
        to propagate input variance-covariance to transmission data

        Parameters
        ----------
        tof : array-like
            Array of time of flight values for each data point - corresponds to energy.
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
        sys_unc = np.array(sys_unc)

        # numerator and denominator
        D = alpha[2]*C - alpha[3]*K*Bi - B0
        N = alpha[0]*c - alpha[1]*k*Bi - b0
        
        # construct statistical covariance and jacobian
        dTi_dci = alpha[0]/D
        dTi_dCi = N*alpha[2]/D**2
        diag_stat = (dc**2)*(dTi_dci**2) + (dC**2)*(dTi_dCi**2)
        
        # systematic derivatives
        # dTi_da = -1*(k*D+K*N)/(a*D**2)  
        # dTi_db = (k*D+K*N)*Bi*tof/D**2 
        dTi_da = -(k*alpha[1]*D+K*alpha[3]*N)*np.exp(-b*tof) / (D**2)
        dTi_db =  (k*alpha[1]*D-N*alpha[3]*K)*Bi*tof / (D**2)
        dTi_dk = -alpha[1]*Bi/D       
        dTi_dK = N*alpha[3]*Bi/D**2
        dTi_db0 = -1/D
        dTi_dB0 = N/D**2
        dTi_dalpha = [ c/D, -k*Bi/D, -C*N/D**2, K*Bi*N/D**2 ]

        if calc_cov:

            ### statistical covarance
            CovT_stat = np.diag(diag_stat)

            ### systematic covariance
            Cov_sys = np.diag(sys_unc**2)
            ab_cov = np.array(ab_cov)
            Cov_sys = np.block([
                                [ab_cov,                                        np.zeros((ab_cov.shape[0], Cov_sys.shape[1]))   ],
                                [np.zeros((Cov_sys.shape[0],ab_cov.shape[1])),  Cov_sys                                         ]
                            ])   
            Jac_sys = np.array([dTi_da, dTi_db, dTi_dk, dTi_dK, dTi_db0, dTi_dB0, dTi_dalpha[0], dTi_dalpha[1],dTi_dalpha[2],dTi_dalpha[3]])
            CovT_sys = Jac_sys.T @ Cov_sys @ Jac_sys
            
            ### T covariance is sum of systematic and statistical covariance 
            CovT = CovT_stat + CovT_sys

            data = [CovT, CovT_stat, CovT_sys, Jac_sys, Cov_sys]
            
        else:
            
            diag_sys = (sys_unc[0]**2)*(dTi_da**2) + (sys_unc[1]**2)*(dTi_db**2) + (sys_unc[2]**2)*(dTi_dk**2) + (sys_unc[3]**2)*(dTi_dK**2) + (sys_unc[4]**2)*(dTi_db0**2) \
                    + (sys_unc[5]**2)*(dTi_dB0**2) + (sys_unc[6]**2)*(dTi_dalpha[0]**2) + (sys_unc[7]**2)*(dTi_dalpha[1]**2) + (sys_unc[8]**2)*(dTi_dalpha[2]**2) + (sys_unc[9]**2)*(dTi_dalpha[3]**2)

            diag_tot = diag_stat + diag_sys

            data = [diag_tot, diag_stat, diag_sys]


        return data

    
def reduce_raw_count_data(tof, c,C, dc,dC, bw, trigo,trigs, a,b, k,K, Bi, b0,B0, alpha, sys_unc, ab_cov, calc_cov):
    """
    Reduces raw count data to transmission data with propagated uncertainty.

    This function uses the covariance sandwhich rule:
    .. math:: C_y = J^T*C_x*J
    to propagate input variance-covariance from both statistical uncertainties
    and systematic uncertainties to transmission data.

    Parameters
    ----------
    tof : array-like
        Array of time of flight values for each data point - corresponds to energy.
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
    unc_data = get_covT(tof, cr,Cr, dcr,dCr, a,b, k,K, Bi, b0,B0, alpha, sys_unc, ab_cov, calc_cov)
    
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
    Cr, dCr = cts_to_ctr(open_df.c, open_df.dc, open_df.bw, trigo) # cts_o/(bw*trig)
    # open_df['cps'] = Cr; open_df['dcps'] = dCr
    
    # calculate sample in count rate from theoretical transmission, bkg, m,k, and open count rate
    [m1,m2,m3,m4] = alpha
    cr = (sample_df["true"]*(m3*Cr - m4*K*Bi - B0) + m2*k*Bi + b0)/m1
    
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

    assert(c.all() >= 0)
    dc = np.sqrt(c)
    
    sample_df.loc[:,'c'] = c
    sample_df.loc[:,'dc'] = dc

    return sample_df, theo_c







# ========================================================================================
#            syndat_T class 
# ========================================================================================

class syndat_T:

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
                            'm1'        :   (1,                   0.016),
                            'm2'        :   (1,                   0.008),
                            'm3'        :   (1,                   0.018),
                            'm4'        :   (1,                   0.005),
                            'ks'        :   (0.563,               0.02402339737495515),
                            'ko'        :   (1.471,               0.05576763648617445),
                            'b0s'       :   (9.9,                 0.1),
                            'b0o'       :   (13.4,                0.7),
                            'a_b'        :   (
                                                [582.7768594580712, 0.05149689096209191],  

                                                [[1.14395753e+03,  1.42659922e-1 ],
                                                 [1.42659922e-1,   2.19135003e-05]]       )
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
        self.raw_data = self.generate_raw_data(self.true_neutron_spectrum, self.true_reduction_parameters)

        ### reduce the raw count data
        self.data = self.reduce(self.raw_data, self.neutron_spectrum, self.reduction_parameters)





    def generate_raw_data(self, 
                          true_neutron_spectrum,
                          true_reduction_parameters
                          ):
        """
        Generates a set of noisy, sample in count data from a theoretical cross section via the novel un-reduction method (Walton, et al.).

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

        monitor_array = [true_reduction_parameters["m1"][0], true_reduction_parameters["m2"][0], true_reduction_parameters["m3"][0], true_reduction_parameters["m4"][0]]
        true_Bi = neutron_background_function(self.true_neutron_spectrum.tof, self.true_reduction_parameters["a_b"][0][0], self.true_reduction_parameters["a_b"][0][1])
        
        raw_data, true_c = inverse_reduction(self.raw_data, 
                                                    true_neutron_spectrum,
                                                    self.options["Sample Counting Noise"], 
                                                    self.options["Sample TURP"],
                                                    true_reduction_parameters["trigo"][0], 
                                                    true_reduction_parameters["trigs"][0], 
                                                    true_reduction_parameters["ks"][0],
                                                    true_reduction_parameters["ko"][0], 
                                                    true_Bi, 
                                                    true_reduction_parameters["b0s"][0], 
                                                    true_reduction_parameters["b0o"][0], 
                                                    monitor_array, true_reduction_parameters["m1"][1])
        
        return raw_data
        
    
    def reduce(self, raw_data, neutron_spectrum, reduction_parameters):
        """
        Reduces the raw count data (sample in/out) to Transmission data and propagates uncertainty.

        """
        ### Code for re-binning data
        # if self.gfactors is not None:
        #     # Re-bin the data according to new structure
        #     grouped_odat = exp_effects.regroup(self.odat.tof, self.odat.c, self.gfactors, self.cpts)
        #     grouped_sdat = exp_effects.regroup(self.sdat.tof, self.sdat.c, self.gfactors, self.cpts)
        #     odat = pd.DataFrame(grouped_odat, columns=['tof','bw','c','dc'])
        #     sdat = pd.DataFrame(grouped_sdat, columns=['tof','bw','c','dc'])

        #     # calculate energy and redefine experiment.odat/sdat with the regrouped data
        #     odat['E'] = exp_effects.t_to_e((odat.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True)
        #     sdat['E'] = exp_effects.t_to_e((odat.tof-self.redpar.val.t0)*1e-6, self.redpar.val.FP, True) 
        #     self.odat = odat
        #     self.sdat = sdat

        # create transmission object
        trans = pd.DataFrame()
        trans['tof'] = raw_data.tof
        trans['E'] = raw_data.E
        trans['true'] = raw_data.true

        # get count rates for sample in data
        # self.sdat['cps'], self.sdat['dcps'] = exp_effects.cts_to_ctr(self.sdat.c, self.sdat.dc, self.sdat.bw*1e-6, self.redpar.val.trigs)
        # self.odat['cps'], self.odat['dcps'] = exp_effects.cts_to_ctr(self.odat.c, self.odat.dc, self.odat.bw*1e-6, self.redpar.val.trigs)

        # estimated background function
        Bi = neutron_background_function(self.neutron_spectrum.tof, reduction_parameters["a_b"][0][0], reduction_parameters["a_b"][0][1])

        # define systematic uncertainties
        # sys_unc = self.redpar.unc[['a','b','ks','ko','b0s','b0o','m1','m2','m3','m4']].astype(float)
        sys_unc = [reduction_parameters[key][1] for key in ['ks','ko','b0s','b0o','m1','m2','m3','m4']]
            
        # monitor_array = [self.redpar.val.m1, self.redpar.val.m2, self.redpar.val.m3, self.redpar.val.m4]
        monitor_array = [reduction_parameters["m1"][0], reduction_parameters["m2"][0], reduction_parameters["m3"][0], reduction_parameters["m4"][0]]
        

        trans['exp'], unc_data, rates = reduce_raw_count_data(raw_data.tof,
                                                                    raw_data.c, neutron_spectrum.c, 
                                                                    raw_data.dc, neutron_spectrum.dc,
                                                                    neutron_spectrum.bw, 
                                                                    reduction_parameters["trigo"][0], 
                                                                    reduction_parameters["trigs"][0], 
                                                                    reduction_parameters["a_b"][0][0],
                                                                    reduction_parameters["a_b"][0][1], 
                                                                    reduction_parameters["ks"][0], 
                                                                    reduction_parameters["ko"][0], 
                                                                    Bi, 
                                                                    reduction_parameters["b0s"][0],
                                                                    reduction_parameters["b0o"][0], 
                                                                    monitor_array, 
                                                                    sys_unc, 
                                                                    reduction_parameters["a_b"][1], 
                                                                    self.options["Calculate Covariance"])

        if self.options["Calculate Covariance"]:
            self.CovT, self.CovT_stat, self.CovT_sys, self.Jac_sys, self.Cov_sys = unc_data
            trans['exp_unc'] = np.sqrt(np.diag(self.CovT))
            self.CovT = pd.DataFrame(self.CovT, columns=trans.E, index=trans.E)
            self.CovT.index.name = None
        else:
            diag_tot, diag_stat, diag_sys = unc_data
            trans['exp_unc'] = np.sqrt(diag_tot)
            self.CovT = None

        # define data cps
        # self.neutron_spectrum['cps'] = rates[0]
        # self.neutron_spectrum['dcps'] = rates[1]
        # self.neutron_spectrum['cps'] = rates[2]
        # self.neutron_spectrum['dcps'] = rates[3]

        return trans