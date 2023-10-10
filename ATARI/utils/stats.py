import numpy as np
from numpy.linalg import inv
import scipy.stats as sts

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




def likelihood_ratio_test(X2_null, X2_alt, df):
    """
    Perform a likelihood ratio test for nested models.

    Args:
        LLmin: Log-likelihood of the null (restricted) model.
        LLmax: Log-likelihood of the alternative (unrestricted) model.
        df: Degrees of freedom difference between the two models.

    Returns:
        lrt_stat: Likelihood ratio test statistic.
        p_value: p-value associated with the test statistic.
    """
    # lrt_stat = 2 * (LLalt - LLnull)
    lrt_stat = X2_null - X2_alt
    p_value = 1 - sts.chi2.cdf(lrt_stat, df)
    return lrt_stat, p_value


def likelihood_val(fit, exp, cov):
    return sts.multivariate_normal.pdf( exp, fit, cov )



def cov2corr(cov):
    """
    Converts a covariance matrix to a correlation matrix

    Parameters
    ----------
    cov : array-like
        Covariance matrix as np array or pd dataframe

    Returns
    -------
    ndarray 
        Correlation matrix
    """
    std_deviations = np.atleast_2d(np.sqrt(np.diag(cov)))
    corr_matrix = cov/(std_deviations.T @ std_deviations)
    return corr_matrix