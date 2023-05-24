import numpy as np
from numpy.linalg import inv


def chi2_val(a,b,cov):
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
    # cast into numpy arrays
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    cov = np.atleast_2d(cov)

    # re-orient if needed
    if np.shape(a)[0] != 1:
        a = a.T
    if np.shape(a)[0] != 1:
        raise ValueError("The vector passed for a does not have the correct dimensions")
    if np.shape(b)[0] != 1:
        b = b.T
    if np.shape(b)[0] != 1:
        raise ValueError("The vector passed for b does not have the correct dimensions")

    # check that a and b are the same size
    if np.shape(b) != np.shape(a):
        raise ValueError("Input vectors are not the same size.")
    if np.shape(a)[1] != len(cov):
        raise ValueError("Covariance matrix is not the same length as the vectors.")
    
    return ((b-a) @ inv(cov) @ (b-a).T).item()