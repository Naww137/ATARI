import numpy as np
import pandas as pd
from copy import copy
from ATARI.theory.resonance_statistics import wigner_LL, width_LL




def get_parameter_grid(energy_range, res_par_avg, num_Er, starting_Gg_multiplier, starting_Gn1_multiplier):

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_range) + res_par_avg['Gt99']/10e3
    min_Elam = min(energy_range) - res_par_avg['Gt99']/10e3

    Gg = np.repeat(res_par_avg["<Gg>"], num_Er)*starting_Gg_multiplier
    Gn = np.repeat(res_par_avg["Gn01"], num_Er)*starting_Gn1_multiplier
        
    Er = np.linspace(              min_Elam,              max_Elam,                num_Er)
    J_ID = np.repeat(res_par_avg["J_ID"], num_Er)

    return Er, Gg, Gn, J_ID


def get_resonance_ladder(Er, Gg, Gn1, J_ID, varyE=0, varyGg=0, varyGn1=0):
    return pd.DataFrame({"E":Er, "Gg":Gg, "Gn1":Gn1, "varyE":np.ones(len(Er))*varyE, "varyGg":np.ones(len(Er))*varyGg, "varyGn1":np.ones(len(Er))*varyGn1 ,"J_ID":J_ID})

def update_vary_resonance_ladder(resonance_ladder, varyE=0, varyGg=0, varyGn1=0):
    return_resonance_ladder = copy(resonance_ladder)
    return_resonance_ladder["varyE"] = np.ones(len(return_resonance_ladder))*varyE
    return_resonance_ladder["varyGg"] = np.ones(len(return_resonance_ladder))*varyGg
    return_resonance_ladder["varyGn1"] = np.ones(len(return_resonance_ladder))*varyGn1
    return return_resonance_ladder

def eliminate_small_Gn(resonance_ladder, threshold):
    fraction_eliminated = np.count_nonzero(resonance_ladder.Gn1<threshold)/len(resonance_ladder)
    return_resonance_ladder =copy(resonance_ladder)
    return_resonance_ladder = return_resonance_ladder[return_resonance_ladder.Gn1>threshold]
    return return_resonance_ladder, fraction_eliminated


def get_starting_feature_bank(energy_range,
                            spin_groups,
                            num_Elam= None,
                            starting_Gg_multiplier = 1,
                            starting_Gn1_multiplier = 1, 
                            varyE = 0,
                            varyGg = 0,
                            varyGn1 = 1
                            ):
    if num_Elam is None:
        num_Elam = int((np.max(energy_range)-np.min(energy_range)) * 1.25)

    Er, Gg, Gn, J_ID = [], [], [], []
    for sg in spin_groups:
        Er_1, Gg_1, Gn_1, J_ID_1 = get_parameter_grid(energy_range, sg, num_Elam, starting_Gg_multiplier, starting_Gn1_multiplier)
        Er.append(Er_1); Gg.append(Gg_1); Gn.append(Gn_1); J_ID.append(J_ID_1); 
    Er = np.concatenate(Er)
    Gg = np.concatenate(Gg)
    Gn = np.concatenate(Gn)
    J_ID = np.concatenate(J_ID) 

    return get_resonance_ladder(Er, Gg, Gn, J_ID, varyE=varyE, varyGg=varyGg, varyGn1=varyGn1)


def generate_external_resonance_ladder(spin_groups: list[dict], 
                                  energy_range):
    E = []; Gg = []; Gn1 = []; J_ID = []
    for sg in spin_groups:
        E.extend([np.max(energy_range) + sg["<D>"], np.min(energy_range) - sg["<D>"] ])
        Gg.extend([sg["<Gg>"]]*2)
        Gn1.extend([sg["<Gn>"]]*2)
        J_ID.extend([sg["J_ID"]]*2)

    return get_resonance_ladder(E, Gg, Gn1, J_ID, varyE=0, varyGg=1, varyGn1=1)


def separate_external_resonance_ladder(resonance_ladder, external_resonance_indices):
    external_resonance_ladder = resonance_ladder.iloc[external_resonance_indices, :]
    internal_resonance_ladder = copy(resonance_ladder)
    internal_resonance_ladder.drop(index=external_resonance_indices, inplace=True)
    return internal_resonance_ladder, external_resonance_ladder


def concat_external_resonance_ladder(internal_resonance_ladder, external_resonance_ladder):
    resonance_ladder = pd.concat([external_resonance_ladder, internal_resonance_ladder], ignore_index=True)
    external_resonance_indices = list(range(len(external_resonance_ladder)))
    return resonance_ladder, external_resonance_indices
    

def get_LL_by_parameter(ladder, 
                        spin_groups 
                        ):

    LL_bypar_bysg = []
    for sg in ladder.groupby("J_ID"):

        for key, val in spin_groups.items():
            if float(val['J_ID']) == sg[0]:
                sg_key = key

        LLw = wigner_LL(sg[1].E, spin_groups[sg_key]['<D>'])
        LL_Gg = width_LL(sg[1].Gg, spin_groups[sg_key]['<Gg>'], spin_groups[sg_key]['g_dof'])
        LL_Gn = width_LL(sg[1].Gn1, spin_groups[sg_key]['<Gn>'], spin_groups[sg_key]['n_dof'])
        LL_bypar_bysg.append([LLw, LL_Gg, LL_Gn])
    
    LL_bypar_bysg = np.array(LL_bypar_bysg)
    LL_bypar = np.sum(LL_bypar_bysg, axis=0)

    return LL_bypar