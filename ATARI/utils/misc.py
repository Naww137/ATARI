import numpy as np
from ATARI.theory.xs import SLBW
from ATARI.utils.atario import fill_resonance_ladder
import os
from datetime import datetime
import pandas as pd
from copy import copy
from uuid import uuid4

def fine_egrid(energy, ppeV=10):
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



def generate_sammy_rundir_uniq_name(path_to_sammy_temps: str, case_id: int = 0, addit_str: str = ''):
    """
    _summary_

    _extended_summary_

    Parameters
    ----------
    path_to_sammy_temps : str
        _description_
    case_id : int, optional
        _description_, by default 0
    addit_str : str, optional
        _description_, by default ''

    Returns
    -------
    _type_
        _description_
    """

    if not os.path.exists(path_to_sammy_temps):
        os.mkdir(path_to_sammy_temps)

    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    # # Combine timestamp and random characters
    # unique_string = timestamp

    unique_string = uuid4()

    sammy_rundirname = path_to_sammy_temps+'SAMMY_runDIR_'+addit_str+'_'+str(case_id)+'_'+str(unique_string)+'/'

    return sammy_rundirname


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

def set_vary_columns_on_resonance_ladder(resonance_ladder: pd.DataFrame, vary_list: list, external_indices=[], external_vary_list=[], inplace=False):
    """
    Updates a resonance ladder dataframe with 'vary' columns for E, Gg, and Gn1.
    External resonance indices can be provided along with a external_vary_list.
    If external_vary_list is not provided, external resonances will get the same vary settings as provided in vary_list. 

    Parameters
    ----------
    resonance_ladder : pd.DataFrame
        Resonance ladder to modify.
    vary_list : list
        List indicating vary (1) or don't vary (0) for each parameter [E, Gg, Gn1].
    external_indices : list, optional
        List indicating indices of external resonances in dataframe, by default [].
    external_vary_list  : list, optional
        Vary list for external resonances if different from vary_list, by default [].
    inplace : bool, optional
        Modify resonance ladder in-place or return a copy, by default False.

    Returns
    -------
    pd.DataFrame
        Modified resonance ladder.

    Raises
    ------
    ValueError
        _description_
    """
    if len(vary_list) != 3:
        raise ValueError("vary_list must contain exactly 3 elements.")
    if isinstance(external_indices, np.ndarray):
        external_vary_list = list(external_vary_list)
    
    if inplace:
        resonance_ladder = copy(resonance_ladder)
    else: pass

    if resonance_ladder.empty: return resonance_ladder

    columns_to_set = ['varyE', 'varyGg', 'varyGn1']
    for col, value in zip(columns_to_set, vary_list):
        # ladder_df[col] = value
        resonance_ladder.loc[:, col] = value

    if external_indices:
        if external_vary_list:
            for i in external_indices:
                resonance_ladder.loc[i, columns_to_set] = external_vary_list
        else: pass
    else: pass

    if inplace: return
    else: return resonance_ladder