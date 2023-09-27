#%%
import numpy as np
import pandas as pd
import h5py
from ATARI.PiTFAll.sample_case import csv_2_hdf5
import os
#%%
dataset_range = (0,49)
directory = '/home/nwalton1/reg_perf_tests/perf_tests/example'
case_file = 'perf_test_example_staticwindow.hdf5'


for fit in os.listdir():

    if os.path.isdir(fit):

        fit_directory = os.path.join(directory, fit)

        splitfit = fit.split('_')
        matfile = '_'.join([splitfit[0], 'fit', splitfit[-1]])

        for i in range(min(dataset_range), max(dataset_range)):

            try:
                csv_2_hdf5(fit_directory, os.path.join(directory,case_file), i, f'{fit}_pp')
            except:

                with open(os.path.join(directory, f'resub_job_template.sh'), 'r') as f:
                    template = f.readlines()
                    f.close()
                with open(os.path.join(fit_directory, f'resub_job_{i}.sh'), 'w') as f:
                    for line in template:
                        if line.startswith('matlab'):
                            f.write(f"""matlab -nodisplay -batch "{matfile}('{os.path.join(directory,case_file)}', {i})" \n""")
                        else:
                            f.write(line)

                print(f'Re-run fit {fit} case {i}')
                
                   


# %%
