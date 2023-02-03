
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
def read_fit_datasets(case_file, i):
    use_hdf5 = check_case_file_or_dir(case_file)
    if use_hdf5:
        with h5py.File(case_file, 'r') as f:
            fit_pw_df = pd.DataFrame(f[f'sample_{i}/fit_pw'][()].T, columns=['E','est_trans'])
            fit_par_df = pd.DataFrame(f[f'sample_{i}/fit_par'][()].T, columns=['E','Gg','gnx2'])
            f.close()
    else:
        fit_pw_df = pd.read_csv(os.path.join(case_file, f'sample_{i}', 'fit_pw.csv'))
        fit_par_df = pd.read_csv(os.path.join(case_file, f'sample_{i}', 'fit_par.csv'))
    return fit_pw_df, fit_par_df

###
def read_syndat_datasets(case_file, i):
    if i == 44:
        _=0
    use_hdf5 = check_case_file_or_dir(case_file)
    if use_hdf5:
        syndat_pw_df = pd.read_hdf(case_file, f'sample_{i}/syndat_pw')
        syndat_par_df = pd.read_hdf(case_file, f'sample_{i}/syndat_par')
    else:
        syndat_pw_df = pd.read_csv(os.path.join(case_file, f'sample_{i}/syndat_pw.csv'))
        try:
            syndat_par_df = pd.read_csv(os.path.join(case_file, f'sample_{i}/syndat_par.csv'))
        except: 
            syndat_par_df = pd.DataFrame()

    return syndat_pw_df, syndat_par_df

###
def read_sample_case_data(case_file, isample):
    # read in data
    syndat_pw_df, syndat_par_df = read_syndat_datasets(case_file, isample)
    fit_pw_df, fit_par_df = read_fit_datasets(case_file, isample)

    # sort parameter data
    syndat_par_df.sort_values('E', inplace=True)
    fit_par_df.sort_values('E', inplace=True)

    # compile pw_data and sort
    pw_data = syndat_pw_df.loc[:, ['E','theo_trans','exp_trans','exp_trans_unc']]
    pw_data['est_trans'] = fit_pw_df['est_trans']
    pw_data.sort_values('E', inplace=True)

    return pw_data, syndat_par_df, fit_par_df

###
def fine_egrid(energy):
    minE = min(energy); maxE = max(energy)
    n = int((maxE - minE)*1e2)
    new_egrid = np.linspace(minE, maxE, n)
    return new_egrid

###
def take_syndat_spingroups(syndat_par_df, fit_par_df):
    if all(item in fit_par_df.columns for item in ['J', 'chs', 'lwave', 'J_ID']):
        print('yes')
    else:
        fit_par_df[['J', 'chs', 'lwave', 'J_ID']] = syndat_par_df[['J', 'chs', 'lwave', 'J_ID']]
    return

###
def calculate_xs(energy, Ta_pair, syndat_par_df, fit_par_df, finegrid_bool):

    if finegrid_bool:
        egrid = fine_egrid(energy)
    else:
        egrid = energy

    xs_tot_syndat, xs_scat_syndat, xs_cap_syndat = scattering_theory.SLBW(egrid, Ta_pair, syndat_par_df)
    take_syndat_spingroups(syndat_par_df, fit_par_df)
    xs_tot_fit, xs_scat_fit, xs_cap_fit = scattering_theory.SLBW(egrid, Ta_pair, fit_par_df)

    return xs_tot_syndat, xs_tot_fit, egrid

###
def analyze_fit(case_file, isample, particle_pair):

    pw_data, syndat_par_df, fit_par_df = read_sample_case_data(case_file, isample)

    # pull vectors from dfs and cast into numpy arrays
    energy = np.array(pw_data.E)
    theo_trans = np.array(pw_data.theo_trans)
    exp_trans = np.array(pw_data.exp_trans)
    exp_trans_unc = np.array(pw_data.exp_trans_unc)
    est_trans = np.array(pw_data.est_trans)

    # def integral_FoMs(exp_trans, exp_trans_unc, theo_trans, est_trans):)
    dof = len(exp_trans)-1
    est_SE = np.sum((exp_trans-est_trans)**2)
    sol_SE = np.sum((exp_trans-theo_trans)**2)
    est_chi_square = np.sum((exp_trans-est_trans)**2/exp_trans_unc**2)
    sol_chi_square = np.sum((exp_trans-theo_trans)**2/exp_trans_unc**2)

    xs_tot_syndat, xs_tot_fit, new_egrid = calculate_xs(energy, particle_pair, syndat_par_df, fit_par_df, True)
    est_sol_SE = integrate.trapezoid((xs_tot_syndat-xs_tot_fit)**2, new_egrid)

    FoMs = pd.DataFrame({'fit_exp'     :   [est_SE,     est_chi_square, est_chi_square/dof], 
                        'theo_exp'     :   [sol_SE,     sol_chi_square, sol_chi_square/dof],
                        'fit_theo'     :   [est_sol_SE, None,           None],
                        'FoM'          :   ['SE',      'Chi2',         'Chi2/dof']})

    FoMs.set_index('FoM', inplace=True)

    return FoMs

###
def analyze_syndat(case_file, isample):
    
    syndat_pw_df, syndat_par_df = read_syndat_datasets(case_file, isample) 
    theo_exp_SE = np.sum((syndat_pw_df.exp_trans-syndat_pw_df.theo_trans)**2)
    NumRes = len(syndat_par_df)
    NumEpts = len(syndat_pw_df)

    return NumRes, NumEpts, theo_exp_SE



###
def plot(case_file, isample, particle_pair, fit_bool, finegrid_bool):

        if fit_bool:
            # read
            pw_data, syndat_par_df, fit_par_df = read_sample_case_data(case_file, isample)
            #convert to xs
            xs_tot_syndat, xs_tot_fit, new_egrid = calculate_xs(pw_data.E, particle_pair, syndat_par_df, fit_par_df, finegrid_bool)
            # create plot
            fig, ax = plt.subplots(1,2, figsize=(10,4), sharex=True)
            # plot trans
            ax[0].errorbar(pw_data.E, pw_data.exp_trans, yerr=pw_data.exp_trans_unc, zorder=0, 
                                    fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
            ax[0].plot(pw_data.E, pw_data.theo_trans, lw=2, color='g', label='sol', zorder=1)
            ax[0].plot(pw_data.E, pw_data.est_trans, lw=1, color='r', label='est', zorder=2)
            ax[0].set_ylim([-.1, 1]); #ax[0].set_xscale('log')
            ax[0].set_xlabel('Energy'); ax[0].set_ylabel('Transmission')
            ax[0].legend()
            # plot xs
            ax[1].plot(new_egrid, xs_tot_syndat, lw=2, color='g', label='sol', zorder=1)
            ax[1].plot(new_egrid, xs_tot_fit, lw=1, color='r', label='est', zorder=2)
            ax[1].set_yscale('log'); #ax[1].set_xscale('log')
            ax[1].set_xlabel('Energy'); ax[1].set_ylabel('Total Cross Section')
            ax[1].legend()
            fig.tight_layout()

        else:
            pass
            # #read
            # syndat_pw_df, syndat_par_df = read_syndat_datasets(case_file, isample)
            # # plot
            # plt.figure()
            # plt.errorbar(syndat_pw_df.E, syndat_pw_df.exp_trans, yerr=syndat_pw_df.exp_trans_unc, zorder=0, 
            #                         fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')

            # plt.plot(syndat_pw_df.E, syndat_pw_df.theo_trans, lw=2, color='g', label='sol', zorder=1)
            # # make it pretty
            # plt.ylim([-.1, 1])
            # plt.xscale('log')
            # plt.xlabel('Energy'); plt.ylabel('Transmission')
            # plt.legend()
            # plt.show()
            # # plt.close()
            
        return fig

