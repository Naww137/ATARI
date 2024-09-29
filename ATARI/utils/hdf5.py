import pandas as pd
import numpy as np
import h5py


### Read data

def read_pw_reduced(case_file, isample, dataset_title):
    pw_reduced_df = pd.read_hdf(case_file, f"sample_{isample}/exp_dat_{dataset_title}/pw_reduced")
    existing_cov_fields = []
    with h5py.File(case_file, 'r') as f:
        if "cov_data" in f[f"sample_{isample}/exp_dat_{dataset_title}"].keys():
            [existing_cov_fields.append(cov_field) for cov_field in f[f"sample_{isample}/exp_dat_{dataset_title}/cov_data"].keys()]
    cov_data = {}
    for cov_field in existing_cov_fields:
        if cov_field == 'Cov_sys':
            with h5py.File(case_file, 'r') as f:
                cov_data[cov_field] = f[f"sample_{isample}/exp_dat_{dataset_title}/cov_data/{cov_field}"][:]
        else:
            cov_data[cov_field] = pd.read_hdf(case_file, f"sample_{isample}/exp_dat_{dataset_title}/cov_data/{cov_field}")
    return pw_reduced_df, cov_data

    
        

def read_pw_exp(case_file, isample, title="exp"):
    pw_exp = pd.read_hdf(case_file, f'/sample_{isample}/pw_{title}')
    try:
        CovT = pd.read_hdf(case_file, f'/sample_{isample}/CovT')
    except:
        CovT = None
    return pw_exp, CovT


def read_pw_fine(case_file, isample):
    pw_fine = pd.read_hdf(case_file, f'/sample_{isample}/pw_fine')
    return pw_fine


def read_par(case_file:str, isample:int, title:str) -> pd.DataFrame: 
    par = pd.read_hdf(case_file, f"sample_{isample}/par_{title}")
    assert(isinstance(par, pd.DataFrame))
    return par




### Write data
import os
def write_pw_reduced(case_file, isample, dataset_title, pw_reduced_df, cov_data=None, overwrite=True):
    # ### check existing samples
    # h5f = h5py.File(case_file, "a")
    # if f"sample_{isample}" in h5f:
    #     if f"exp_dat_{dataset_title}" in h5f[f"sample_{isample}"]:
    #         if overwrite:
    #             # print(f"Dataset titled exp_dat_{dataset_title} already exists in sample_{isample}, overwriting")
    #             pass
    #         else:
    #             raise ValueError("Overwrite=False is not implemented yet")
    #     else:
    #         pass
    # h5f.close()

    ### write data
    pw_reduced_df.to_hdf(case_file, key=f"sample_{isample}/exp_dat_{dataset_title}/pw_reduced")
    if cov_data is not None:
        for key, val in cov_data.items():
            if isinstance(val, np.ndarray):
                h5f = h5py.File(case_file, 'a')
                if f"sample_{isample}/exp_dat_{dataset_title}/cov_data/{key}" in h5f:
                    h5f[f"sample_{isample}/exp_dat_{dataset_title}/cov_data/{key}"][:]=val
                else:
                    h5f[f"sample_{isample}/exp_dat_{dataset_title}/cov_data/{key}"]=val
                h5f.close()
            elif isinstance(val, pd.DataFrame):
                val.to_hdf(case_file, key=f"sample_{isample}/exp_dat_{dataset_title}/cov_data/{key}")
            else:
                raise ValueError(f"Unrecognized type {type(val)} in cov_data")
        

def write_pw_exp(case_file, isample, pw_exp_df, title="exp", CovT=None, CovXS=None):
    pw_exp_df.to_hdf(case_file, key=f"sample_{isample}/pw_{title}")
    if CovT is not None:
        CovT.to_hdf(case_file, key=f"sample_{isample}/CovT")
    if CovXS is not None:
        CovXS.to_hdf(case_file, key=f"sample_{isample}/CovXS")


def write_pw_fine(case_file, isample, pw_fine_df):
    pw_fine_df.to_hdf(case_file, key=f"sample_{isample}/pw_fine")


def write_par(case_file, isample, par, title):
    par.to_hdf(case_file, key=f"sample_{isample}/par_{title}")





### Other 
def csvpar_2_hdf5(csv, case_file, isample, title):
        
    par_df = pd.read_csv(csv)
    # tfit = est_par_df.tfit[0]
    # est_par_df.drop('tfit', axis=1)

    par_df.to_hdf(case_file, key=f"sample_{isample}/par_{title}")

    return