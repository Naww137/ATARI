# %%
# additional functions to handle the data
from ATARI.PolePosition.polepos_funcs import load_transmission_data_csv, plot_trcs, plot_search_results, delete_nan_cs_rows, search_peaks
# from ATARI.PolePosition import *
import pandas as pd
import json

import numpy as np
from ATARI import PiTFAll as pf


# %%
import os


case_file = os.path.realpath('/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/perf_test_staticwindow_poleposition.hdf5')


# %%

for i in range(500):

    try:
        # read pw and order 
        test = pd.read_hdf(case_file, f'sample_{i}/exp_pw')
        n = 0.067166 # atoms per barn or atoms/(1e-12*cm^2)
        test['exp_cs'] = np.log(test['exp_trans']) / (-n)
        test['theo_cs'] = np.log(test['theo_trans']) / (-n)
        test.sort_values('E', inplace=True)
        test.reset_index(drop=False,inplace=True)

        test_par = pd.read_hdf(case_file, f'sample_{i}/theo_par')

        transm_df = test
        # deleting all points with the negative transmission
        transm_nonzero = delete_nan_cs_rows(transm_df)
        ladder_df = test_par

        # parameters of polepos
        polepos_hps = {'search_params' : {'part_of_variance_filter': 0.2,
                                        'base_of_peak': 0.5, 
                                        'min_width': 0.01,
                                        'min_height': 0.01,
                                        'prominence': 0.01
                                        }, 

                        'reduce_params' : {'param_cutoff_by_name': 'peak_sq_divE',
                                            'cutoff_threshold': 0.03
                                            }
        }

        pulses_sel_df, pulses_all_df  = search_peaks(transm_nonzero,                    
                        search_params = polepos_hps['search_params'],           
                        reduce_params = polepos_hps['reduce_params']
        )

        # sorting result by weighted square
        suggested_peaks = pulses_sel_df.nlargest(10, 'peak_sq_divE')#.peak_E
        suggested_peaks = pd.DataFrame(suggested_peaks.loc[:, ['peak_E','peak_sq_divE'] ])
        suggested_peaks.rename(columns={'peak_E':'E'}, inplace=True)

        # plot_search_results(transm_df, ladder_df, pulses_sel_df)
        # suggested_peaks.peak_E.to_csv(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/initial_guess/suggested_peaks_{i}.csv', header=False, index=False)
        suggested_peaks.to_hdf(case_file, f'sample_{i}/poles')

    except:
        # pd.DataFrame().to_csv(f'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/initial_guess/suggested_peaks_{i}.csv', header=False, index=False)
        pd.DataFrame(columns=['E']).to_hdf(case_file, f'sample_{i}/poles')


# %%
import h5py 

with h5py.File(case_file, 'r') as f:
    print(f['sample_0/poles'].keys())
    print(f['sample_0/poles/block0_values'][()])
    f.close()

# %%