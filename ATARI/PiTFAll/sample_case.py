
import numpy as np
import pandas as pd
import os
from scipy import integrate
import h5py
import matplotlib.pyplot as plt
from syndat import scattering_theory


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

###
def fine_egrid(energy):
    minE = min(energy); maxE = max(energy)
    n = int((maxE - minE)*1e2)
    new_egrid = np.linspace(minE, maxE, n)
    return new_egrid

###
def take_syndat_spingroups(theo_par_df, est_par_df):
    if all(item in est_par_df.columns for item in ['J', 'chs', 'lwave', 'J_ID']):
        print('yes')
    else:
        est_par_df[['J', 'chs', 'lwave', 'J_ID']] = theo_par_df[['J', 'chs', 'lwave', 'J_ID']]
    return est_par_df

###
def calculate_xs(energy, particle_pair, theo_par_df, est_par_df):

    xs_tot_theo, xs_scat_syndat, xs_cap_syndat = scattering_theory.SLBW(egrid, particle_pair, theo_par_df)
    take_syndat_spingroups(theo_par_df, est_par_df)
    xs_tot_est, xs_scat_fit, xs_cap_fit = scattering_theory.SLBW(egrid, particle_pair, est_par_df)

    return xs_tot_theo, xs_tot_est

### 
def reconstruct_fit(experiment, particle_pair, 
                    case_file, isample,
                    exp_pw_df, theo_pw_df, theo_par_df, est_par_df, fit_name):

    est_par_df = take_syndat_spingroups(theo_par_df, est_par_df)

    if f'est_trans_{fit_name}' not in exp_pw_df:
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
        pass

    return  exp_pw_df, theo_pw_df, est_par_df

###
def analyze_exp_pw(exp_pw_df, exp_cov, fit_name):
    
    # pull vectors from dfs and cast into numpy arrays
    energy = np.array(exp_pw_df.E)
    theo_trans = np.array(exp_pw_df.theo_trans)
    exp_trans = np.array(exp_pw_df.exp_trans)
    est_trans = np.array(exp_pw_df[f'est_trans_{fit_name}'])

    # def integral_FoMs(exp_trans, exp_trans_unc, theo_trans, est_trans):)
    dof = len(exp_trans)-1
    est_SE = np.sum((exp_trans-est_trans)**2)
    sol_SE = np.sum((exp_trans-theo_trans)**2)
    est_chi_square = (exp_trans-est_trans) @ exp_cov @ (exp_trans-est_trans).T
    sol_chi_square = (exp_trans-theo_trans) @ exp_cov @ (exp_trans-theo_trans).T

    return est_SE, sol_SE, est_chi_square, sol_chi_square, dof

###
def analyze_fit(case_file, isample, experiment, particle_pair, fit_name):

    # read sample case data and check for fit pw reconstruction
    exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov = read_sample_case_data(case_file, isample, fit_name)
    exp_pw_df, theo_pw_df, est_par_df = reconstruct_fit(experiment, particle_pair, 
                                                        case_file, isample,
                                                        exp_pw_df, theo_pw_df, theo_par_df, est_par_df, fit_name)

    # analyze pw data on experimental grid (transmission) and fine grid (xs)
    est_SE, sol_SE, est_chi_square,  sol_chi_square, dof = analyze_exp_pw(exp_pw_df, exp_cov, fit_name)
    est_sol_SE = integrate.trapezoid((theo_pw_df.theo_xs-theo_pw_df[f'est_xs_{fit_name}'])**2, theo_pw_df.E)

    FoMs = pd.DataFrame({'fit_exp'     :   [est_SE,     est_chi_square, est_chi_square/dof], 
                        'theo_exp'     :   [sol_SE,     sol_chi_square, sol_chi_square/dof],
                        'fit_theo'     :   [est_sol_SE, None,           None],
                        'FoM'          :   ['SE',      'Chi2',         'Chi2/dof']})

    FoMs.set_index('FoM', inplace=True)

    return FoMs

###
def csv_2_hdf5(directory, case_file, isample, fit_name):
        
    est_par_df = pd.read_csv(os.path.join(directory, f'./par_est_{isample}.csv'))
    tfit = est_par_df.tfit[0]
    est_par_df.drop('tfit', axis=1)

    est_par_df.to_hdf(os.path.join(directory,case_file), f"sample_{isample}/est_par_{fit_name}")

    h5file = h5py.File(os.path.join(directory,case_file), 'a')
    sample_est_dataset = h5file[f"sample_{isample}/est_par_{fit_name}"]
    sample_est_dataset.attrs['tfit'] = tfit
    h5file.close()

    return


###
def analyze_syndat(case_file, isample):
    
    exp_pw_df, theo_pw_df, theo_par_df, est_par_df, exp_cov = read_sample_case_data(case_file, isample, None)

    theo_exp_SE = np.sum((exp_pw_df.exp_trans-exp_pw_df.theo_trans)**2)
    NumRes = len(theo_par_df)
    NumEpts = len(exp_pw_df)

    return NumRes, NumEpts, theo_exp_SE



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

