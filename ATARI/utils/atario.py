import pandas as pd
import numpy as np
from ATARI.theory.scattering_params import FofE_recursive
# from ATARI.theory.scattering_params import gstat


# ----------------------------------------------------------------------------------
# HDF5 utilities
# ----------------------------------------------------------------------------------
def h5read_experimental(case_file, isample):
    exp_pw = pd.read_hdf(case_file, f'/sample_{isample}/pw_exp')
    exp_cov = pd.read_hdf(case_file, f'/sample_{isample}/CovT')
    return exp_pw, exp_cov

def h5read_theoretical(case_file, isample):
    theo_pw = pd.read_hdf(case_file, f'/sample_{isample}/theo_pw')
    theo_par = pd.read_hdf(case_file, f'/sample_{isample}/par_true')
    return theo_pw, theo_par


### Write data
def h5write_experimental(case_file, isample, exp_pw_df, exp_cov):
    exp_pw_df.to_hdf(case_file, f"sample_{isample}/pw_exp")
    pd.DataFrame(exp_cov, index=np.array(exp_pw_df.E), columns=exp_pw_df.E).to_hdf(case_file, f"sample_{isample}/CovT")
    return

def h5write_theoretical(case_file, isample, theo_pw_df, theo_par):
    theo_pw_df.to_hdf(case_file, f"sample_{isample}/pw_fine")
    theo_par.to_hdf(case_file, f"sample_{isample}/par_true") 
    return
        


# ----------------------------------------------------------------------------------
# ATARI internal object utilities
# ----------------------------------------------------------------------------------

# def take_syndat_spingroups(theo_par_df, est_par_df):
#     if all(item in est_par_df.columns for item in ['J', 'chs', 'lwave', 'J_ID']):
#         pass
#     else:
#         standard_spingroups = np.array([theo_par_df[['J', 'chs', 'lwave', 'J_ID']].iloc[0]])
#         est_par_df[['J', 'chs', 'lwave', 'J_ID']] = np.repeat(standard_spingroups, len(est_par_df), axis=0)
#     return est_par_df


def check_and_place_resonance_latter_columns(resonance_ladder, item, key):
    if key in resonance_ladder.columns:
        pass
    else:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            if len(resonance_ladder.index) != len(item):
                raise ValueError(f"A list was passed for {key} but it does not match the length of the resonance ladder")
            resonance_ladder.loc[:,key] = item
        else:
            if item == None:
                raise ValueError("Spin group information is missing and not provided")
            resonance_ladder.loc[:,key] = [item]*len(resonance_ladder.index)
    return 


def fill_resonance_ladder(resonance_ladder, particle_pair, 
                                                    J=None,
                                                    chs=None,
                                                    lwave=None,
                                                    J_ID= None  ):
    """
    Fills out the resonance ladder DataFrame. 
    Calculates partial width from reduced width and vice-versa. 
    Optional **kwargs allow for sping group information to be added.

    Parameters
    ----------
    resonance_ladder : DataFrame
        Resonance ladder you want to fill.
    particle_pair : ATARI.syndat Particle_Pair class
        Particle pair class instance.
    J : float or array-like, optional
        J will only be assigned if the column does not already exist. 
        If scalar it is applied to all resonances in the ladder. 
        If array-like it is concatenated to the DataFrame, by default None
    chs : _type_, optional
        chs will only be assigned if the column does not already exist. 
        If scalar it is applied to all resonances in the ladder. 
        If array-like it is concatenated to the DataFrame, by default None
    lwave : _type_, optional
        lwave will only be assigned if the column does not already exist. 
        If scalar it is applied to all resonances in the ladder. 
        If array-like it is concatenated to the DataFrame, by default None
    J_ID : _type_, optional
        J_ID will only be assigned if the column does not already exist. 
        If scalar it is applied to all resonances in the ladder. 
        If array-like it is concatenated to the DataFrame, by default None

    Returns
    -------
    resonance_ladder : DataFrame
    """

    def gn2G(row):
        _, P, _, _ = FofE_recursive([row.E], particle_pair.ac, particle_pair.M, particle_pair.m, row.lwave)
        Gn = 2*np.sum(P)*row.gn2
        return Gn.item()

    def G2gn(row):
        _, P, _, _ = FofE_recursive([row.E], particle_pair.ac, particle_pair.M, particle_pair.m, row.lwave)
        gn2 = row.Gn/2/np.sum(P)
        return gn2.item()

    # check spin group information
    for item, key in zip([J,chs,lwave,J_ID], ['J', 'chs', 'lwave', 'J_ID']):
        check_and_place_resonance_latter_columns(resonance_ladder, item, key)

    if len(resonance_ladder.index) == 0:
        return resonance_ladder
    else:
        # if "Gnx" in resonance_ladder:
        #     resonance_ladder['Gn'] = resonance_ladder["Gnx"]
        # if "gnx2" in resonance_ladder:
        #     resonance_ladder['gn2'] = resonance_ladder["gnx2"]
        # calculate other missing widths
        if 'Gn' not in resonance_ladder:
            resonance_ladder['Gn'] = resonance_ladder.apply(lambda row: gn2G(row), axis=1)
        if 'gn2' not in resonance_ladder:
            resonance_ladder['gn2'] = resonance_ladder.apply(lambda row: G2gn(row), axis=1)
        if 'Gt' not in resonance_ladder:
            resonance_ladder['Gt'] = resonance_ladder['Gn'] + resonance_ladder['Gg']

    return resonance_ladder





# ----------------------------------------------------------------------------------
# Try factory method
# ----------------------------------------------------------------------------------

# class ObjectStorer


