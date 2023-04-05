import numpy as np
from numpy.linalg import inv


def calc_chi2(a,b,cov):
    """
    Calculates the $chi^2$ statistic between fit and data.

    Parameters
    ----------
    a : ndarray
        Fit vector.
    b : ndarray
        Data vector.
    cov : ndarray
        Covariance matrix for data in vector b.
    """
    # re-orient if needed
    if np.shape(a)[0] != 1:
        a = a.T
    if np.shape(b)[0] != 1:
        b = b.T
    # check that a and b are the same size
    if np.shape(b) != np.shape(a):
        raise ValueError("Input vectors are not the same size.")
    if np.shape(a)[1] != len(cov):
        raise ValueError("Covariance matrix is not the same length as the vectors.")
    
    return (b-a) @ inv(cov) @ (b-a).T