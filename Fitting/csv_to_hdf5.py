#%%
import numpy as np
import pandas as pd
import h5py
from ATARI.PiTFAll.sample_case import csv_2_hdf5

dataset_range = (0,11)
directory = '/Users/noahwalton/Documents/GitHub/ATARI/Fitting'
case_file = 'perf_test_baron.hdf5'

for i in range(min(dataset_range), max(dataset_range)):
    csv_2_hdf5(directory, case_file, i, 'baron')


