from typing import List
import numpy as np
from numpy import newaxis as NA
from scipy.linalg import cho_factor, cho_solve

def psd_solve(V, A):
    """
    Calculates V\A, where V is symmetric, positive-semidefinite, and A is an arbitrary
    matrix. `psd_solve` uses Cholesky decomposition to perform the matrix solve faster.

    Parameters
    ----------
    V : array
        The positive-semidefinite matrix.
    A : array
        An arbitrary matrix V backdivides with.
    
    Returns
    -------
    B : array
        The solution matrix.
    """

    c, L = cho_factor(V)
    B = cho_solve((c, L), A)
    return B



#%%================================================================================================
# Implicit Data Covariance
#==================================================================================================

def implicit_data_cov_build(exp_cov:dict):
    """
    ...

    See section IV.D.3 of the SAMMY manual.
    """

    v = exp_cov['diag_stat'].values
    has_sys_unc = ('Cov_sys' in exp_cov)
    if has_sys_unc:
        m = exp_cov['Cov_sys']
        g = exp_cov['Jac_sys'].values.T
        V = np.diag(v) + g @ m @ g.T
    else:
        V = np.diag(v)
    return V

def implicit_data_cov_solve(exp_cov:dict, x:np.ndarray):
    """
    ...

    See section IV.D.3 of the SAMMY manual.
    """

    v = exp_cov['diag_stat'].values
    has_sys_unc = ('Cov_sys' in exp_cov)
    if has_sys_unc:
        m = exp_cov['Cov_sys']
        g = exp_cov['Jac_sys'].values.T
        h = g / v
        Z = np.linalg.inv(m) + g.T @ h
        y = x / v - h @ np.linalg.solve(Z, (h.T @ x))
    else:
        y = x / v
    return y