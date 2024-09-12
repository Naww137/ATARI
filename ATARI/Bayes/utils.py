import numpy as np
from numpy import newaxis as NA
from scipy.linalg import cho_factor, cho_solve

#%%================================================================================================
# Special Linear Algebra
#==================================================================================================

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

    U, opt = cho_factor(V)
    B = cho_solve((U, opt), A)
    return B

def psd_inv(V):
    """
    Inverts V, where V is symmetric, positive-semidefinite. `psd_inv` uses Cholesky decomposition
    to perform the matrix solve faster.

    Parameters
    ----------
    V : array
        The positive-semidefinite matrix.
    
    Returns
    -------
    Vi : array
        The solution matrix.
    """

    U, opt = cho_factor(V)
    Vi = cho_solve((U, opt), np.eye(V.shape[0]))
    return Vi

#%%================================================================================================
# Implicit Data Covariance
#==================================================================================================

def implicit_data_cov_build(exp_cov:dict):
    """
    ...

    See section IV.D.3 of the SAMMY manual.
    """

    v = exp_cov['diag_stat'].values
    if 'Cov_sys' in exp_cov:
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
    if 'Cov_sys' in exp_cov:
        m = exp_cov['Cov_sys']
        g = exp_cov['Jac_sys'].values.T
        h = g / v
        Z = psd_inv(m) + g.T @ h
        y = x / v - h @ psd_solve(Z, (h.T @ x)) # Woodbury Matrix Identity
    else:
        y = x / v
    return y

def YW_implicit_data_cov_solve(exp_cov:dict, G:np.ndarray, DT:np.ndarray):
    """
    ...

    See section IV.D.3 of the SAMMY manual.
    """

    DT = DT[:,NA]
    v = exp_cov['diag_stat'].values
    if 'Cov_sys' in exp_cov:
        m = exp_cov['Cov_sys']
        g = exp_cov['Jac_sys'].values.T
        h = g / v
        Z = psd_inv(m) + g.T @ h
        ViDT = (DT / v - h @ psd_solve(Z, (h.T @ DT)))[:,0]
        Y = G.T @ ViDT
        W = G.T @ (G  / v - h @ psd_solve(Z, (h.T @ G )))
    else:
        ViDT = (DT / v)[:,0]
        Y = G.T @ ViDT
        W = G.T @ (G / v)
    chi2 = float( DT.T @ ViDT )
    chi2n = chi2 / len(DT)
    return Y, W, chi2, chi2n

def implicit_data_cov_chi2(exp_cov:dict, DT:np.ndarray):
    """
    ...
    """

    DT = DT[:,NA]
    v = exp_cov['diag_stat'].values
    if 'Cov_sys' in exp_cov:
        m = exp_cov['Cov_sys']
        g = exp_cov['Jac_sys'].values.T
        h = g / v
        Z = psd_inv(m) + g.T @ h
        ViDT = (DT / v - h @ psd_solve(Z, (h.T @ DT)))[:,0]
    else:
        ViDT = (DT / v)[:,0]
    chi2 = float( DT.T @ ViDT )
    return chi2

def implicit_data_cov_inv(exp_cov:dict):
    """
    ...
    """

    v = exp_cov['diag_stat'].values
    if 'Cov_sys' in exp_cov:
        m = exp_cov['Cov_sys']
        g = exp_cov['Jac_sys'].values.T
        h = g / v
        Z = psd_inv(m) + g.T @ h
        Vinv = (1/v - h @ psd_solve(Z, h.T))[:,0]
    else:
        Vinv = (1 / v)[:,0]
    return Vinv