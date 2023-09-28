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


def read_par(case_file:str, isample:int, label:str) -> pd.DataFrame: 
    par = pd.read_hdf(case_file, f"sample_{isample}/par_{label}")
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


def write_par(case_file, isample, par, label):
    par.to_hdf(case_file, f"sample_{isample}/par_{label}")


# def write_experimental(case_file, isample, exp_pw_df, exp_cov):
#     exp_pw_df.to_hdf(case_file, f"sample_{isample}/exp_pw")
#     # pd.DataFrame(exp_cov, index=np.array(exp_pw_df.E), columns=exp_pw_df.E).to_hdf(case_file, f"sample_{isample}/exp_cov")
#     exp_cov.to_hdf(case_file, f"sample_{isample}/exp_cov")
#     return