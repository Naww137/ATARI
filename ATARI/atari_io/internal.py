import pandas as pd
import numpy as np
from ATARI.theory.scattering_params import FofE_recursive
# from ATARI.theory.scattering_params import gstat



# def take_syndat_spingroups(theo_par_df, est_par_df):
#     if all(item in est_par_df.columns for item in ['J', 'chs', 'lwave', 'J_ID']):
#         pass
#     else:
#         standard_spingroups = np.array([theo_par_df[['J', 'chs', 'lwave', 'J_ID']].iloc[0]])
#         est_par_df[['J', 'chs', 'lwave', 'J_ID']] = np.repeat(standard_spingroups, len(est_par_df), axis=0)
#     return est_par_df


def check_and_place(resonance_ladder, item, key):
    if key in resonance_ladder.columns:
        pass
    else:
        if item == None:
            raise ValueError("Spin group information is missing and not provided")
        else:
            if isinstance(item, list):
                if len(resonance_ladder.index) != len(item):
                    raise ValueError(f"A list was passed for {key} but it does not match the length of the resonance ladder")
                resonance_ladder.loc[:,key] = item
            else:
                resonance_ladder.loc[:,key] = [item]*len(resonance_ladder.index)
    return 


def fill_resonance_ladder(resonance_ladder, particle_pair, 
                                                    J=None,
                                                    chs=None,
                                                    lwave=None,
                                                    J_ID= None  ):

    def gn2G(row):
        _, P, _, _ = FofE_recursive([row.E], particle_pair.ac, particle_pair.M, particle_pair.m, row.lwave)
        Gnx = 2*np.sum(P)*row.gnx2
        return Gnx.item()

    def G2gn(row):
        _, P, _, _ = FofE_recursive([row.E], particle_pair.ac, particle_pair.M, particle_pair.m, row.lwave)
        gnx2 = row.Gnx/2/np.sum(P)
        return gnx2.item()

    # check spin group information
    for item, key in zip([J,chs,lwave,J_ID], ['J', 'chs', 'lwave', 'J_ID']):
        check_and_place(resonance_ladder, item, key)

    # calculate other missing widths
    if 'Gnx' not in resonance_ladder:
        resonance_ladder['Gnx'] = resonance_ladder.apply(lambda row: gn2G(row), axis=1)
    if 'gnx' not in resonance_ladder:
        resonance_ladder['gnx2'] = resonance_ladder.apply(lambda row: G2gn(row), axis=1)
    if 'Gt' not in resonance_ladder:
        resonance_ladder['Gt'] = resonance_ladder['Gnx'] + resonance_ladder['Gg']

    return resonance_ladder