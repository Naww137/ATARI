import pandas as pd
import numpy as np


### Read data
def read_experimental(case_file, isample):
    exp_pw = pd.read_hdf(case_file, f'/sample_{isample}/exp_pw')
    exp_cov = pd.read_hdf(case_file, f'/sample_{isample}/exp_cov')
    return exp_pw, exp_cov

def read_theoretical(case_file, isample):
    theo_pw = pd.read_hdf(case_file, f'/sample_{isample}/theo_pw')
    theo_par = pd.read_hdf(case_file, f'/sample_{isample}/theo_par')
    return theo_pw, theo_par


### Write data
def write_experimental(case_file, isample, exp_pw_df, exp_cov):
    exp_pw_df.to_hdf(case_file, f"sample_{isample}/exp_pw")
    pd.DataFrame(exp_cov, index=np.array(exp_pw_df.E), columns=exp_pw_df.E).to_hdf(case_file, f"sample_{isample}/exp_cov")
    return

def write_theoretical(case_file, isample, theo_pw_df, theo_par):
    theo_pw_df.to_hdf(case_file, f"sample_{isample}/theo_pw")
    theo_par.to_hdf(case_file, f"sample_{isample}/theo_par") 
    return
        
