
import numpy as np
import pandas as pd
import os
from scipy import integrate
import h5py
import matplotlib.pyplot as plt
from ATARI.syndat import scattering_theory
from numpy.linalg import inv

# ===========================================================================================
#   METHODS FOR READING DATA
# ===========================================================================================

###
def check_case_file_or_dir(case_file):
    if os.path.isfile(case_file):
        use_hdf5 = True
    else:
        use_hdf5 = False
    return use_hdf5

###
def read_par_datasets(case_file, i, fit_name):
    use_hdf5 = check_case_file_or_dir(case_file)
    if use_hdf5:
        # TODO: allow for multiple fit_names to be given
        theo_par_df = pd.read_hdf(case_file, f'sample_{i}/theo_par')
        if fit_name is not None:
            est_par_df = pd.read_hdf(case_file, f'sample_{i}/est_par_{fit_name}')
        else:
            est_par_df = None
    else:
        if os.path.isfile(os.path.join(case_file, f'sample_{i}', f'est_par_{fit_name}.csv')):
            est_par_df = pd.read_csv(os.path.join(case_file, f'sample_{i}', f'est_par_{fit_name}.csv'))
        else:
            est_par_df = None
        theo_par_df = pd.read_csv(os.path.join(case_file, f'sample_{i}', f'theo_par.csv'))
    return theo_par_df, est_par_df

###
def read_pw_datasets(case_file, i):
    use_hdf5 = check_case_file_or_dir(case_file)
    if use_hdf5:
        exp_pw_df = pd.read_hdf(case_file, f'sample_{i}/exp_pw')
        theo_pw_df = pd.read_hdf(case_file, f'sample_{i}/theo_pw')
    else:
        exp_pw_df = pd.read_csv(os.path.join(case_file, f'sample_{i}/exp_pw.csv'))
        try:
            theo_pw_df = pd.read_csv(os.path.join(case_file, f'sample_{i}/theo_pw.csv'))
        except: 
            theo_pw_df = pd.DataFrame()

    return exp_pw_df, theo_pw_df

###
def read_sample_case_data(case_file, isample, fit_name):
    # read in data
    exp_pw_df, theo_pw_df = read_pw_datasets(case_file, isample)
    theo_par_df, est_par_df = read_par_datasets(case_file, isample, fit_name)
    with h5py.File(case_file) as f:
        exp_cov = f[f'sample_{isample}/exp_cov'][()]
        f.close()

    # sort parameter data
    theo_par_df.sort_values('E', inplace=True)
    if est_par_df is not None:
        est_par_df.sort_values('E', inplace=True)
    # exp_pw_df.sort_values('E', inplace=True)
    # theo_pw_df.sort_values('E', inplace=True)

    return exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov




# ===========================================================================================
#   METHODS FOR ANLYZING SYNTHETIC DATA
# ===========================================================================================


###
def analyze_syndat(case_file, isample):
    
    exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov = read_sample_case_data(case_file, isample, None)

    theo_exp_SE = np.sum((exp_pw_df.exp_trans-exp_pw_df.theo_trans)**2)
    NumRes = len(theo_par_df)
    NumEpts = len(exp_pw_df)
    avg_gnx2 = np.mean(theo_par_df.gnx2)
    avg_Gg = np.mean(theo_par_df.Gg)
    min_gnx2 = min(theo_par_df.gnx2)
    min_Gg = min(theo_par_df.Gg)
    max_gnx2 = max(theo_par_df.gnx2)
    max_Gg = max(theo_par_df.Gg)

    return [isample, NumRes, NumEpts, theo_exp_SE, avg_gnx2, avg_Gg, min_gnx2, min_Gg, max_gnx2, max_Gg]




# ===========================================================================================
#   METHODS FOR RECONSTRUCTING FITS
# ===========================================================================================

###
def fine_egrid(energy):
    minE = min(energy); maxE = max(energy)
    n = int((maxE - minE)*1e2)
    new_egrid = np.linspace(minE, maxE, n)
    return new_egrid

###
def take_syndat_spingroups(theo_par_df, est_par_df):
    if all(item in est_par_df.columns for item in ['J', 'chs', 'lwave', 'J_ID']):
        pass
    else:
        standard_spingroups = np.array([theo_par_df[['J', 'chs', 'lwave', 'J_ID']].iloc[0]])
        est_par_df[['J', 'chs', 'lwave', 'J_ID']] = np.repeat(standard_spingroups, len(est_par_df), axis=0)
    return est_par_df

###
def calculate_xs(energy, particle_pair, theo_par_df, est_par_df):

    xs_tot_theo, xs_scat_syndat, xs_cap_syndat = scattering_theory.SLBW(energy, particle_pair, theo_par_df)
    take_syndat_spingroups(theo_par_df, est_par_df)
    xs_tot_est, xs_scat_fit, xs_cap_fit = scattering_theory.SLBW(energy, particle_pair, est_par_df)

    return xs_tot_theo, xs_tot_est

### 
def reconstruct_fit(experiment, particle_pair, 
                    case_file, isample,
                    exp_pw_df, theo_pw_df, theo_par_df, est_par_df, fit_name,
                    overwrite_reconstructed_fits = False):

    est_par_df = take_syndat_spingroups(theo_par_df, est_par_df)

    if f'est_trans_{fit_name}' not in exp_pw_df:
        est_xs_tot, est_xs_scat, est_xs_cap = scattering_theory.SLBW(exp_pw_df.E, particle_pair, est_par_df)
        n = experiment.redpar.val.n  # atoms per barn or atoms/(1e-12*cm^2)
        est_trans = np.exp(-n*est_xs_tot)
        exp_pw_df[f'est_trans_{fit_name}'] = est_trans
        exp_pw_df.to_hdf(case_file, f'sample_{isample}/exp_pw')
    else:
        if overwrite_reconstructed_fits:
            est_xs_tot, est_xs_scat, est_xs_cap = scattering_theory.SLBW(exp_pw_df.E, particle_pair, est_par_df)
            n = experiment.redpar.val.n  # atoms per barn or atoms/(1e-12*cm^2)
            est_trans = np.exp(-n*est_xs_tot)
            exp_pw_df[f'est_trans_{fit_name}'] = est_trans
            exp_pw_df.to_hdf(case_file, f'sample_{isample}/exp_pw')
        else:
            pass 

    if f'est_xs_{fit_name}' not in theo_pw_df:
        est_xs_tot, est_xs_scat, est_xs_cap = scattering_theory.SLBW(theo_pw_df.E, particle_pair, est_par_df)
        theo_pw_df[f'est_xs_{fit_name}'] = est_xs_tot
        theo_pw_df.to_hdf(case_file, f'sample_{isample}/theo_pw')
    else:
        if overwrite_reconstructed_fits:
            est_xs_tot, est_xs_scat, est_xs_cap = scattering_theory.SLBW(theo_pw_df.E, particle_pair, est_par_df)
            theo_pw_df[f'est_xs_{fit_name}'] = est_xs_tot
            theo_pw_df.to_hdf(case_file, f'sample_{isample}/theo_pw')
        else:
            pass

    return  exp_pw_df, theo_pw_df, est_par_df




# ===========================================================================================
#   METHODS FOR ANLYZING FITS
# ===========================================================================================


###
def calculate_integral_pw_FoMs(isample, exp_pw_df, theo_pw_df, exp_cov, fit_name):
    
    # pull vectors from dfs and cast into numpy arrays
    energy = np.array(exp_pw_df.E)
    theo_trans = np.array(exp_pw_df.theo_trans)
    exp_trans = np.array(exp_pw_df.exp_trans)
    est_trans = np.array(exp_pw_df[f'est_trans_{fit_name}'])

    # calculate integral_FoMs on experimental grid
    dof = len(exp_trans)-1
    est_SE = np.sum((exp_trans-est_trans)**2)
    sol_SE = np.sum((exp_trans-theo_trans)**2)
    est_chi_square = (exp_trans-est_trans) @ inv(exp_cov) @ (exp_trans-est_trans).T
    sol_chi_square = (exp_trans-theo_trans) @ inv(exp_cov) @ (exp_trans-theo_trans).T

    # calculate integral_FoMs on theoretical grid 
    est_sol_MSE = integrate.trapezoid((theo_pw_df.theo_xs-theo_pw_df[f'est_xs_{fit_name}'])**2, theo_pw_df.E)
    # return as list
    integral_pw_FoMs = [isample, est_sol_MSE, est_SE, est_chi_square, est_chi_square/dof, sol_SE, sol_chi_square, sol_chi_square/dof]

    return integral_pw_FoMs


###
def calc_bias_variance_pw(isample, theo_pw_df, exp_pw_df, fit_name):
    # cast to numpy arrays
    theo_xs = np.array(theo_pw_df.theo_xs)
    theo_trans = np.array(exp_pw_df.theo_trans)
    est_xs = np.array(theo_pw_df[f'est_xs_{fit_name}'])
    est_trans = np.array(exp_pw_df[f'est_trans_{fit_name}'])
    energy_xs = np.array(theo_pw_df.E)
    energy_trans = np.array(exp_pw_df.E)

    # calculate bias and second moment so we can calculate variance
    bias_xs = theo_xs-est_xs
    first_moment_xs_est = est_xs
    second_moment_xs_est = est_xs**2

    bias_trans = theo_trans-est_trans
    first_moment_trans_est = est_trans
    second_moment_trans_est = est_trans**2

    return [bias_xs, first_moment_xs_est, second_moment_xs_est, energy_xs] , [bias_trans, first_moment_trans_est, second_moment_trans_est, energy_trans]

def calc_bias_variance_window(isample, theo_pw_df, exp_pw_df, fit_name):
    # cast to numpy arrays
    energy_xs = np.array(theo_pw_df.E); energy_trans=np.array(exp_pw_df.E)
    theo_xs = np.array(theo_pw_df.theo_xs)
    theo_trans = np.array(exp_pw_df.theo_trans)
    est_xs = np.array(theo_pw_df[f'est_xs_{fit_name}'])
    est_trans = np.array(exp_pw_df[f'est_trans_{fit_name}'])

    # calculate bias/variance in window
    window_energy_midpoint  = (min(energy_xs)+max(energy_xs))/2
    assert window_energy_midpoint == (min(energy_trans)+max(energy_trans))/2

    window_bias_xs = np.mean(theo_xs-est_xs)
    window_bias_trans = np.mean(theo_trans-est_trans)

    window_variance_xs = np.mean( (np.mean(est_xs)-est_xs)**2 )
    window_variance_trans = np.mean( (np.mean(est_trans)-est_trans)**2 )

    return [isample, window_energy_midpoint, window_bias_xs, window_bias_trans, window_variance_xs, window_variance_trans]

def calculate_integral_par_FoMs(isample, est_par_df, theo_par_df):

    # drop est resonance with 0 neutron width
    est_par_df_nonzero = est_par_df.loc[est_par_df['gnx2']>0.0, 'E':'gnx2']

    # sort by energy
    theo_par_df.sort_values('E', ignore_index=True, inplace=True)
    est_par_df_nonzero.sort_values('E', ignore_index=True, inplace=True)

    # calculate averages
    est_avg_Gg = np.mean(est_par_df.Gg) 
    est_avg_gnx2 = np.mean(est_par_df.gnx2)
    est_min_Gg = min(est_par_df.Gg) 
    est_min_gnx2 = min(est_par_df.gnx2)
    est_max_Gg = max(est_par_df.Gg) 
    est_max_gnx2 = max(est_par_df.gnx2)

    # calculate cardinality
    est_cardinality = len(est_par_df_nonzero)

    return [isample, est_cardinality, est_avg_Gg, est_avg_gnx2, est_min_Gg, est_min_gnx2, est_max_Gg, est_max_gnx2]

###
def analyze_fit(case_file, isample, experiment, particle_pair, fit_name, vary_Erange):

    # read sample case data and check for fit pw reconstruction
    exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov = read_sample_case_data(case_file, isample, fit_name)
    exp_pw_df, theo_pw_df, est_par_df = reconstruct_fit(experiment, particle_pair, 
                                                        case_file, isample,
                                                        exp_pw_df, theo_pw_df, theo_par_df, est_par_df, fit_name)

    # Get integral FoMs from data on experimental grid (transmission) and fine grid (xs)
    integral_pw_FoMs = calculate_integral_pw_FoMs(isample, exp_pw_df, theo_pw_df, exp_cov, fit_name)
    integral_pw_FoMs.append(est_par_df.tfit[0])

    # calulate bias variance averages in the window and if the window is static calculate it pw
    bias_variance_window = calc_bias_variance_window(isample, theo_pw_df, exp_pw_df, fit_name)
    if vary_Erange is None:
        bias_variance_pw = calc_bias_variance_pw(isample, theo_pw_df, exp_pw_df, fit_name)
        bv = (bias_variance_window, bias_variance_pw)
    else:
        bv = bias_variance_window

    # get integreal FoMs from parameter data 
    integral_par_FoMs = calculate_integral_par_FoMs(isample, est_par_df, theo_par_df)

    return integral_pw_FoMs, integral_par_FoMs, bv



# ===========================================================================================
#   METHODS FOR PLOTTING A SAMPLE CASE
# ===========================================================================================


###
def plot(case_file, isample, fit_name):

        # read sample_case data 
        exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov = read_sample_case_data(case_file, isample, fit_name)

        # create plot
        fig, ax = plt.subplots(1,2, figsize=(10,4), sharex=True)
        # plot trans
        ax[0].errorbar(exp_pw_df.E, exp_pw_df.exp_trans, yerr=np.sqrt(np.diag(exp_cov)), zorder=0, 
                                fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
        ax[0].plot(exp_pw_df.E, exp_pw_df.theo_trans, lw=2, color='g', label='sol', zorder=1)
        ax[0].plot(exp_pw_df.E, exp_pw_df[f'est_trans_{fit_name}'], lw=1, color='r', label='est', zorder=2)
        ax[0].set_ylim([-.1, 1]); #ax[0].set_xscale('log')
        ax[0].set_xlabel('Energy'); ax[0].set_ylabel('Transmission')
        ax[0].legend()
        # plot xs
        ax[1].plot(theo_pw_df.E, theo_pw_df.theo_xs, lw=2, color='g', label='sol', zorder=1)
        ax[1].plot(theo_pw_df.E, theo_pw_df[f'est_xs_{fit_name}'], lw=1, color='r', label='est', zorder=2)
        ax[1].set_yscale('log'); #ax[1].set_xscale('log')
        ax[1].set_xlabel('Energy'); ax[1].set_ylabel('Total Cross Section')
        ax[1].legend()
        fig.tight_layout()

        return fig





# ===========================================================================================
#  OTHER UTILITY METHODS
# ===========================================================================================


###
def csv_2_hdf5(directory, case_file, isample, fit_name):
        
    est_par_df = pd.read_csv(os.path.join(directory, f'par_est_{isample}.csv'))
    tfit = est_par_df.tfit[0]
    est_par_df.drop('tfit', axis=1)

    est_par_df.to_hdf(os.path.join(directory,case_file), f"sample_{isample}/est_par_{fit_name}")

    h5file = h5py.File(os.path.join(directory,case_file), 'a')
    sample_est_dataset = h5file[f"sample_{isample}/est_par_{fit_name}"]
    sample_est_dataset.attrs['tfit'] = tfit
    h5file.close()

    return