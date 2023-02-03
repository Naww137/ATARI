#%%
import numpy as np
import pandas as pd
import h5py

dataset_range = (0,11)
case_file = './perf_test_baron.hdf5'
# est_name = est_par

for i in range(min(dataset_range), max(dataset_range)):

    est_par_df = pd.read_csv(f'./par_est_{i}.csv')
    tfit = est_par_df.tfit[0]
    tfit

    est_par_df.to_hdf(case_file, f"sample_{i}/est_par")

    f = h5py.File(case_file, 'a')
    sample_est_dataset = f[f"sample_{i}/est_par"]
    sample_est_dataset.attrs['tfit'] = tfit
    # print(sample_est_dataset.attrs['tfit'])
    f.close()

