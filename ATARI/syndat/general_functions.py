import numpy as np
from copy import deepcopy
import pandas as pd
from ATARI.theory.experimental import e_to_t, t_to_e
import scipy.stats as sts


def gaus_noise(vector, std_vec):
    """
    Samples gaussian noise around a vector of mean values.

    Parameters
    ----------
    vector : array-like
        Vector of mean values.
    std_vec : array-like
        Vector of standard deviations (standard errors).
    
    Returns
    -------
    Noisy vector sampled as guassian around each mean/std.
    """
    # scale (std) = sqrt(mean) resembles almost identically the poisson distribution with a number of counts>20
    noise = np.random.default_rng().normal(loc=0.0, scale=std_vec, size=len(vector)) #np.mean(vector)*level
    return vector + noise


def pois_noise(vector, rng:np.random.Generator=None, seed:int=None):
    """
    Samples poissonian noise around a vector of values.
    Parameters
    ----------
    vector : array-like
        vector of expected values.
    Returns
    -------
    noisy_vector
        Noisy vector sampled as poissonian around each expected value
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed
    noisy_vector = rng.poisson(lam=vector)
    return noisy_vector


def cts_to_ctr(cts, d_cts, bw, trig):
    """
    Converts counts to count rate and propagates uncertainty.

    Parameters
    ----------
    cts : array-like
        Array of count data corresponting to each tof bin.
    d_cts : array-like
        Array of uncertainty on each count data point corresponting to each tof bin.
    bw : array-like
        Array of tof bin widths (seconds).
    trig : float or int
        Number of linac pulses.
        
    Notes
    _____
    Uncertainty propagation with sandwich rule (JxCxJ.T) is over 1000x slower. 
    A more simple error propagtion is used because there is no covariance 
    between the statistical variables.
    
    Returns
    -------
    ctr : array-like
        Array of count rates corresponding to each tof bin.
    d_nctr : array-like
        Array of propagated uncertainty on each count rate point.
    """ 
    ctr = np.array(cts/(bw*trig))
    partial = np.array(1/(bw*trig))
    d_nctr = [np.sqrt((partial[i]**2)*dc**2) for i,dc in enumerate(d_cts)]
    return ctr, d_nctr



def sample_true_underlying_parameters(parameter_dict, bool:bool, rng:np.random.Generator=None, seed:int=None):
    """
    Given a parameter dictionary of form {'key': (val, unc)}, this function will sample true underlying parameters from N(val, unc).
    A true underlying parameter dictionary is returned with the input parameter dictionary unchanged.

    Parameters
    ----------
    parameter_dict : _type_
        _description_
    bool : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    # Random number generator:
    if rng is None:
        if seed is None:
            rng = np.random # uses np.random.seed
        else:
            rng = np.random.default_rng(seed) # generates rng from provided seed

    true_parameter_dict = deepcopy(parameter_dict)
    if bool:
        for par in true_parameter_dict:
            if isinstance(true_parameter_dict[par][1], list):
                true_parameter_dict[par] = (list(rng.multivariate_normal(true_parameter_dict[par][0], true_parameter_dict[par][1])), 0)
            else:
                mean = true_parameter_dict[par][0]
                stdev = true_parameter_dict[par][1]
                true_parameter_dict[par] = (rng.normal(mean, stdev), 0)

    return true_parameter_dict

def neutron_background_function(tof,a,b,bkg_func):
    if bkg_func == "exp":
        return bkg_func_exp(tof,a,b)
    elif bkg_func == "power":
        return bkg_func_power(tof,a,b)
    else:
        raise ValueError("bkg_func not regocnized")

def bkg_func_exp(tof,a,b):
    """
    Exponential neutron backgroun function

    .. math:: bkg = a*e^(-b*tof)

    Parameters
    ----------
    tof : array-like
        time-of-flight in micro-seconds (us)
    a : float
        parameter
    b : float
        parameter

    Returns
    -------
    _type_
        _description_
    """
    return a*np.exp(tof*-b)

def bkg_func_power(tof,a,b):
    """
    Power law neutron background function

    .. math:: bkg = a*tof^(-b)

    Parameters
    ----------
    tof : array-like
        time-of-flight in micro-seconds (us)
    a : float
        parameter
    b : float
        parameter

    Returns
    -------
    _type_
        _description_
    """
    return a*(tof)**(-b)


def approximate_gamma_background_spectrum(energy_grid, smooth, FP, t0, trigo, nominal=25):
    # background_spectrum_bg= pd.DataFrame({'c'    :   np.ones(len(exp_model.energy_grid))*25,
    #                                  'dc'   :   np.ones(len(exp_model.energy_grid))*np.sqrt(25)})

    # calculate a tof count rate spectra, convert to counts 
    tof = e_to_t(energy_grid, FP, True)*1e9 + t0 
    cps_open_approx = np.ones(len(energy_grid))*nominal
    bin_width = abs(np.append(np.diff(tof), np.diff(tof)[-1])*1e-9)
    cts_open_approx = cps_open_approx * bin_width * trigo

    # add noise
    if smooth:
        cts_open_measured = cts_open_approx
    else:
        cts_open_measured = pois_noise(cts_open_approx)
    cts_open_measured[cts_open_measured==0] = 1
    dataframe = pd.DataFrame({'tof'    :   tof,
                                'bw'    :   bin_width,
                                'ct'     :   cts_open_measured,
                                'dct'    :   np.sqrt(cts_open_measured)})

    dataframe['E'] = t_to_e((dataframe.tof-t0)*1e-9, FP, True) 

    return dataframe


def approximate_neutron_spectrum_Li6det(energy_grid, smooth, FP, t0, trigo):

    def open_count_rate(tof):
        return (2212.70180199 * np.exp(-3365.55134779*tof*1e-9) + 23.88486286) 

    # calculate a tof count rate spectra, convert to counts 
    tof = e_to_t(energy_grid, FP, True)*1e9 + t0 
    cps_open_approx = open_count_rate(tof)
    bin_width = abs(np.append(np.diff(tof), np.diff(tof)[-1])*1e-9)
    cts_open_approx = cps_open_approx * bin_width * trigo

    # add noise
    if smooth:
        cts_open_measured = cts_open_approx
    else:
        cts_open_measured = pois_noise(cts_open_approx)

    open_dataframe = pd.DataFrame({'tof'    :   tof,
                                    'bw'    :   bin_width,
                                    'ct'     :   cts_open_measured,
                                    'dct'    :   np.sqrt(cts_open_measured)})

    open_dataframe['E'] = t_to_e((open_dataframe.tof-t0)*1e-9, FP, True) 

    return open_dataframe




def approximate_neutron_spectrum_gaussian(energy, flux_mag, mean, std):
    """
    Approximate the neutron flux spectrum with a wide gaussian wrt energy.

    Parameters
    ----------
    energy : array-like
        Array of energy values for each data point - corresponds to tof.
    flux_mag : float
        Magnitude scaling factor applied to flux shape, how many counts!
    mean : float
        Average value for gaussian shape.
    std : float
        Standard deviation for gaussian shape.
    
    Returns
    -------
    Open counts (sample out).
    """
    cts_o = sts.norm.pdf(energy, loc=mean, scale=std)*flux_mag # gaussian in energy, std=range of energy
    return cts_o

    


def sample_true_neutron_spectrum(spectrum_df, cycles=35, mon_unc=0.0160*2):
    
    true_spectrum = spectrum_df.copy()

    theo_cycle_data = true_spectrum.c/cycles
    monitor_factors = np.random.default_rng().normal(1, mon_unc, size=cycles)
    noisy_cycle_data = pois_noise(theo_cycle_data)*monitor_factors[0]
    for i in range(cycles-1):
        noisy_cycle_data += pois_noise(theo_cycle_data)*monitor_factors[i]

    true_spectrum['ct'] = noisy_cycle_data
    true_spectrum['dct'] = np.sqrt(noisy_cycle_data)

    return true_spectrum





def regroup(tof, c, grouping_factors, compression_points):

    if len(compression_points) != 0:
        raise ValueError('Need to implement capability for multiple compression points for re-binning')

    c = np.array(c)
    tof = np.array(tof)
    length = len(tof)
    leftover_bins = length%grouping_factors[0]
    full_bins = length-leftover_bins
    new_length = int(full_bins/grouping_factors[0])

    if leftover_bins > 0:
        grouping_factors = grouping_factors + [leftover_bins]        # last bin structure
    else:
        leftover_c = []
        leftover_tof = []
        leftover_dt = []

    for i, g in enumerate(grouping_factors):
        if i == 1:
            leftover_c = c[full_bins:-1].sum()
            leftover_tof = tof[full_bins:-1].mean()
            leftover_dt = max(tof[full_bins:-1])-min(tof[full_bins:-1])
        elif i == 0:
            grouped_c = np.reshape(c[0:full_bins], (new_length,grouping_factors[0])).sum(axis=1)
            grouped_tof = np.reshape(tof[0:full_bins], (new_length,grouping_factors[0])).mean(axis=1)

            grouped_tof_median = np.median(np.reshape(tof[0:full_bins], (new_length,grouping_factors[0])), axis=1)
            if max(grouped_tof_median-grouped_tof) > 1e-5:
                raise ValueError('Median/mean of tof grouping is diverging')

            grouped_dt = np.diff(grouped_tof)
            grouped_dt = np.insert(grouped_dt,0,grouped_dt[0])

    gc = np.append(grouped_c,leftover_c)            #type: ignore
    gtof = np.append(grouped_tof,leftover_tof)      #type: ignore
    gdt = np.append(grouped_dt, leftover_dt)*1e-6   #type: ignore

    data = np.array([gtof, gdt, gc, np.sqrt(gc)]).T

    return data
