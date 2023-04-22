import numpy as np


def fine_egrid(energy, ppeV):
    """
    Calculates an energy grid of the same domain with a specified number of points per eV.

    Parameters
    ----------
    energy : array-like
        Array containing energy domain, can be min/max or old grid.
    ppeV : float
        Desired data points per eV from min to max of energy domain.

    Returns
    -------
    ndarray
        Array of energy points.
    """
    minE = min(energy); maxE = max(energy)
    n = int((maxE - minE)*ppeV)
    new_egrid = np.linspace(minE, maxE, n)
    return new_egrid