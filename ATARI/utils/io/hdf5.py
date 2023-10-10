import pandas as pd
import numpy as np


### Read data

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

def write_pw_exp(case_file, isample, pw_exp_df, title="exp", CovT=None, CovXS=None):
    pw_exp_df.to_hdf(case_file, f"sample_{isample}/pw_{title}")
    if CovT is not None:
        CovT.to_hdf(case_file, f"sample_{isample}/CovT")
    if CovXS is not None:
        CovXS.to_hdf(case_file, f"sample_{isample}/CovXS")


def write_pw_fine(case_file, isample, pw_fine_df):
    pw_fine_df.to_hdf(case_file, f"sample_{isample}/pw_fine")


def write_par(case_file, isample, par, title):
    par.to_hdf(case_file, f"sample_{isample}/par_{title}")





### Other 
def csvpar_2_hdf5(csv, case_file, isample, title):
        
    par_df = pd.read_csv(csv)
    # tfit = est_par_df.tfit[0]
    # est_par_df.drop('tfit', axis=1)

    par_df.to_hdf(case_file, f"sample_{isample}/par_{title}")

    return