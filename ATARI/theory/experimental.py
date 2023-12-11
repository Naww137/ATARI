import numpy as np
from scipy.linalg import block_diag



def t_to_e(t:float, d:float, rel:bool=True):
    """
    Converts from time of flight to neutron energy using relativistic or Newtonian kinematics.

    Parameters
    ----------
    t : float or ndarray
        Time of flight measurements.
    d : float
        Distance for the time of flight measurement in meters.
    rel : bool
        Relative kinematics (True) or Newtonian kinematics (False), default True.

    Returns
    -------
    float or ndarray
        Neutron energies
    """
    if rel:
        mn = 939.56542052e6 # eV/c2
        c = 299792458 # m/s
        E = mn*(1/np.sqrt(1-(d/(c*t))**2)-1)
    else:
        mn = 1.674927498e-27 # kg
        jev = 1.6022e-19 # J/eV
        E = 0.5*mn*(d/t)**2 /jev # eV
    return E


def e_to_t(E:float, d:float, rel:bool=True):
    """
    Converts from newton energy to time of flight using relativistic or Newtonian kinematics.

    Parameters
    ----------
    E : float or ndarray
        Neutron energies
    d : float
        Distance for the time of flight measurement in meters.
    rel : bool
        Relative kinematics (True) or Newtonian kinematics (False), default True.

    Returns
    -------
    float or ndarray
        Expected time of flight measurements
    """
    if rel:
        mn = 939.56542052e6 # eV/c2
        c = 299792458 # m/s
        t = d/c * 1/np.sqrt(1-1/(E/mn+1)**2)
    else:
        jev = 1.6022e-19 # J/eV
        mn = 1.674927498e-27 #kg
        t = d/np.sqrt(E*jev*2/mn)
    return t



def trans_2_xs(T, n, n_unc=0.0, CovT=None):
    """
    Converts pointwise transmission data to total cross section data. 
    If covariance data is supplied, first order linear propagation is used to approximate the cross section covariance matrix.
    If the transmission value is below 0, nan-type is returned for the corresponding cross section.

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