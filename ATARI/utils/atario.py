import pandas as pd
import numpy as np
from ATARI.theory.scattering_params import FofE_recursive
from copy import deepcopy, copy
import pickle
import os
from typing import Union
from ATARI.syndat.control import Syndat_Control
from ATARI.syndat.syndat_model import Syndat_Model



# ----------------------------------------------------------------------------------
# Syndat saving and loading functions
# ----------------------------------------------------------------------------------

def save_syndat_model(syndat_model: Syndat_Model, 
                      path: str,
                      clear_samples: bool = True):
    assert isinstance(syndat_model, Syndat_Model)

    if clear_samples:
        syndat_model.clear_samples()

    try:
        file = open(path, "wb")
        pickle.dump(syndat_model, file)
        file.close()
    except:
        if not clear_samples:
            raise ValueError("Pickling syndat model failed and clear_samples is False, make sure samples for this syndat_model were not generated with syndat_control.")
        else:
            raise ValueError("Pickling syndat model failed, cause is unknown")
        

def save_syndat_control(syndat_control: Syndat_Control, 
                        path: str,
                        clear_samples: bool = False):
    if clear_samples:
        raise ValueError("Not implemented")

    file = open(path, "wb")
    pickle.dump(syndat_control, file)
    file.close()

def load_syndat(filepath: str) -> Union[Syndat_Model, Syndat_Control]:
    file = open(filepath, "rb")
    syndat = pickle.load(file)
    file.close()
    return syndat


def save_general_object(object, 
                        path: str,
                        ):
    try:
        file = open(path, "wb")
        pickle.dump(object, file)
        file.close()
    except:
        raise ValueError("Pickling syndat model failed, cause is unknown")
    
def load_general_object(filepath: str):
    file = open(filepath, "rb")
    obj = pickle.load(file)
    file.close()
    return obj

# ----------------------------------------------------------------------------------
# User interface functions
# ----------------------------------------------------------------------------------

def update_dict(old, additional):
    new = deepcopy(old)
    for key in old:
        if key in additional:
            new.update({key:additional[key]})
    for key in additional:
        if key not in old:
            raise ValueError("User provided an unrecognized input option")
    return new







# ----------------------------------------------------------------------------------
# ATARI internal object utilities
# ----------------------------------------------------------------------------------



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


def add_Gw_from_gw(resonance_ladder, particle_pair):
    ladder = copy(resonance_ladder)
    Ls_present = np.unique(ladder.L)
    if len(Ls_present) == 1 and Ls_present[0] == 0:
        pass
    else:
        raise NotImplementedError("Need to update Gw_from_gw function for multiple L-waves")
    _, P_array, _, _ = FofE_recursive(ladder.E.values, particle_pair.ac, particle_pair.M, particle_pair.m, Ls_present)
    ladder['Gg'] = Gg = 2*ladder.gg2.values
    ladder['Gn1'] = 2*P_array[0]*ladder.gn2.values
    ladder = ladder[["E", "Gg", "Gn1"]+ [each for each in ladder.keys() if each not in ["E", "Gg", "Gn1"]]]
    return ladder


def expand_sammy_ladder_2_atari(particle_pair, ladder) -> pd.DataFrame:

    def get_J_L_Wred(row):
        J, Ls = [[key, val["Ls"]] for key, val in particle_pair.spin_groups.items() if val["J_ID"] == row.J_ID][0]
        if len(Ls) > 1:
            raise NotImplementedError("Multiple Ls to one spin group has not been implemented")
        else:
            L = Ls[0]

        _, P_array, _, _ = FofE_recursive(np.abs([row.E]), particle_pair.ac, particle_pair.M, particle_pair.m, L)
        gg2 = row.Gg/2
        # ladder['Gn1'] = 2*P_array[0]*ladder.gn2.values
        gn2 = row.Gn1/2/P_array[0].item()

        return gg2, gn2, J, L

    ladder[["gg2","gn2","Jpi","L"]] = ladder.apply(lambda row: get_J_L_Wred(row), axis=1,result_type='expand')

    return ladder


# def take_syndat_spingroups(theo_par_df, est_par_df):
#     if all(item in est_par_df.columns for item in ['J', 'chs', 'lwave', 'J_ID']):
#         pass
#     else:
#         standard_spingroups = np.array([theo_par_df[['J', 'chs', 'lwave', 'J_ID']].iloc[0]])
#         est_par_df[['J', 'chs', 'lwave', 'J_ID']] = np.repeat(standard_spingroups, len(est_par_df), axis=0)
#     return est_par_df



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
    else: # calculate other missing widths
        if 'Gn' not in resonance_ladder:
            resonance_ladder['Gn'] = resonance_ladder.apply(lambda row: gn2G(row), axis=1)
        if 'gn2' not in resonance_ladder:
            resonance_ladder['gn2'] = resonance_ladder.apply(lambda row: G2gn(row), axis=1)
        if 'Gt' not in resonance_ladder:
            resonance_ladder['Gt'] = resonance_ladder['Gn'] + resonance_ladder['Gg']
        if 'Gg' not in resonance_ladder:
            resonance_ladder['Gg'] = resonance_ladder['Gt'] - resonance_ladder['Gn']

    return resonance_ladder



