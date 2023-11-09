import numpy as np
from ATARI.theory.xs import SLBW
from ATARI.utils.atario import fill_resonance_ladder



def fine_egrid(energy, ppeV=100):
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
    energy = np.array(energy).flatten()
    minE = min(energy); maxE = max(energy)
    n = int((maxE - minE)*ppeV)
    new_egrid = np.linspace(minE, maxE, n)
    return new_egrid



def calc_xs_on_fine_egrid(E, ppeV, particle_pair, resonance_ladder):
    """
    Calculates total cross section on a fine energy grid.

    Parameters
    ----------
    E : array-like
        Coarse energy grid, used to define energy domain.
    ppeV : int
        Requested points per eV on fine grid.
    particle_pair : ATARI.syndat.Particle_Pair
        ATARI.syndat particle pair object
    resonance_ladder : DataFrame
        Pandas DataFrame containing resonance ladder information.

    Returns
    -------
    array-like
        New, fine energy grid.
    array-like
        Total cross section on new, fine energy grid.
    """

    fineE = fine_egrid(E, ppeV)
    resonance_ladder = fill_resonance_ladder(resonance_ladder,particle_pair)
    xs_tot_fine, _, _ = SLBW(fineE, particle_pair, resonance_ladder)
        
    return fineE, xs_tot_fine