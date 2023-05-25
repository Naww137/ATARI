import numpy as np
from scipy.linalg import block_diag


def trans_2_xs(T, n, n_unc=0.0, CovT=None):
    """
    Converts pointwise transmission data to total cross section data. 
    If covariance data is supplied, first order linear propagation is used to approximate the cross section covariance matrix.
    If the transmission value is below 0, and nan-type is returned for the corresponding cross section.

    Parameters
    ----------
    T : ndarray
        Vector of transmission values.
    n : float
        Target thickness in atoms/bn-cm.
    n_unc : float
        Uncertainty in the target thickness, default 0.
    CovT : ndarray or None
        Transmission covariance matrix, default None.

    Returns
    -------
    ndarray
        Pointwise cross section values.
    ndarray
        Cross section covariance matrix
    """
    
    xs_exp = (-1/n)*np.log(T)

    if CovT is not None:
        dXi_dn = (1/n**2) * np.log(T)
        dXi_dT = (-1/n) * (1/T)
        Jac = np.vstack((np.diag(dXi_dT),dXi_dn))
        Cov = block_diag(CovT,n_unc**2)
        CovXS = Jac.T @ Cov @ Jac
    else:
        CovXS = None

    return xs_exp, CovXS


def xs_2_trans(xs_tot, n):
    """
    Converts pointwise total cross section data to transmission data.

    Parameters
    ----------
    xs_tot : array-like
        Total cross section.
    n : float
        Target thickness in atoms/bn-cm.

    Returns
    -------
    array-like
        Pointwise transmission.
    """
    return np.exp(-n*xs_tot)