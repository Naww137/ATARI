import numpy as np
import pandas as pd
from copy import copy
from ATARI.theory.resonance_statistics import wigner_LL, width_LL
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.utils.atario import add_Gw_from_gw


def get_parameter_grid(energy_range, res_par_avg, particle_pair, num_Er, starting_Gg_multiplier, starting_Gn1_multiplier):

    if len(res_par_avg["Ls"]) > 1:
            raise NotImplementedError("Multiple Ls to one spin group has not been implemented")
    else:
        L = res_par_avg["Ls"][0]

    # allow Elambda to be just outside of the window
    _, P_array, _, _ = FofE_recursive(np.sort(energy_range), particle_pair.ac, particle_pair.M, particle_pair.m, L)
    Gt99_min_max = res_par_avg['quantiles']['gt99']*P_array[0]
    max_Elam = max(energy_range) + Gt99_min_max[0]/10e3
    min_Elam = min(energy_range) - Gt99_min_max[1]/10e3

    # get widths
    gg2 = np.repeat(res_par_avg["<gg2>"], num_Er)*starting_Gg_multiplier
    gn2 = np.repeat(res_par_avg['quantiles']["gn01"], num_Er)*starting_Gn1_multiplier
    
    # get energies and spin info
    Er = np.linspace(              min_Elam,              max_Elam,                num_Er)
    J_ID = np.repeat(res_par_avg["J_ID"], num_Er)
    Jpi = np.repeat(res_par_avg['Jpi'], num_Er)
    Ls = np.repeat(res_par_avg['Ls'], num_Er)

    return Er, gg2, gn2, J_ID, Jpi, Ls


def get_resonance_ladder(particle_pair, Er, gg2, gn2, J_ID, Jpi, Ls, varyE=0, varyGg=0, varyGn1=0):
    atari_ladder = pd.DataFrame({"E":Er, "gg2":gg2, "gn2":gn2, "Jpi":Jpi, "L":Ls, "varyE":np.ones(len(Er))*varyE, "varyGg":np.ones(len(Er))*varyGg, "varyGn1":np.ones(len(Er))*varyGn1 ,"J_ID":J_ID})
    return add_Gw_from_gw(atari_ladder, particle_pair)


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
                            particle_pair,
                            spin_groups,
                            num_Elam = None,
                            Elam_shift = 0,
                            starting_Gg_multiplier = 1,
                            starting_Gn1_multiplier = 1, 
                            varyE = 0,
                            varyGg = 0,
                            varyGn1 = 1
                            ):
    # setup energy grid
    if num_Elam is None:
        num_Elam = int((np.max(energy_range)-np.min(energy_range)) * 1.25)
    elif num_Elam/(np.max(energy_range)-np.min(energy_range)) < 1.25:
        print("WARNING: User supplied a feature bank energy grid of <1 per eV, problem may not be convex")

    # # setup spin groups
    # if options.fit_all_spin_groups:
    #     spin_groups = [each[1] for each in particle_pair.spin_groups.items()] 
    # else:
    #     assert len(options.spin_group_keys)>0
    #     spin_groups = [each[1] for each in particle_pair.spin_groups.items() if each[0] in options.spin_group_keys]
        
    Er, gg2, gn2, J_ID, Jpi, L = [], [], [], [], [], []
    for sg in spin_groups:
        Er_1, gg2_1, gn2_1, J_ID_1, Jpi_1, L_1 = get_parameter_grid(energy_range, sg, particle_pair, num_Elam, starting_Gg_multiplier, starting_Gn1_multiplier)
        Er.append(Er_1); gg2.append(gg2_1); gn2.append(gn2_1); J_ID.append(J_ID_1); Jpi.append(Jpi_1); L.append(L_1)
    Er = np.concatenate(Er)
    gg2 = np.concatenate(gg2)
    gn2 = np.concatenate(gn2)
    J_ID = np.concatenate(J_ID) 
    Jpi = np.concatenate(Jpi)    
    L = np.concatenate(L)       

    return get_resonance_ladder(particle_pair, Er, gg2, gn2, J_ID, Jpi, L, varyE=varyE, varyGg=varyGg, varyGn1=varyGn1)



def generate_external_resonance_ladder(spin_groups: list[dict], 
                                        energy_range,
                                        particle_pair):
    
    Er = []; gg2 = []; gn2 = []; J_ID = []; Jpi = []; Ls = []
    for sg in spin_groups:
        Er.extend([np.max(energy_range) + sg["<D>"], np.min(energy_range) - sg["<D>"] ])
        gg2.extend([sg["<gg2>"]]*2)
        gn2.extend([sg["<gn2>"]]*2)
        J_ID.extend([sg["J_ID"]]*2)
        Jpi.extend([sg["Jpi"]]*2)
        
        # check for L
        if len(sg["Ls"]) > 1:
                raise NotImplementedError("Multiple Ls to one spin group has not been implemented")
        else:
            L = sg["Ls"][0]
        Ls.extend([L]*2)

    return get_resonance_ladder(particle_pair, Er, gg2, gn2, J_ID, Jpi, Ls, varyE=0, varyGg=1, varyGn1=1)


def separate_external_resonance_ladder(resonance_ladder, external_resonance_indices):
    external_resonance_ladder = resonance_ladder.iloc[external_resonance_indices, :]
    internal_resonance_ladder = copy(resonance_ladder)
    internal_resonance_ladder.drop(index=external_resonance_indices, inplace=True)
    return internal_resonance_ladder, external_resonance_ladder


def concat_external_resonance_ladder(internal_resonance_ladder, external_resonance_ladder):
    resonance_ladder = pd.concat([external_resonance_ladder, internal_resonance_ladder], join='inner', ignore_index=True)
    external_resonance_indices = list(range(len(external_resonance_ladder)))
    return resonance_ladder, external_resonance_indices
    

def get_LL_by_parameter(ladder, 
                        spin_groups 
                        ):
    if 'gg2' not in ladder:
        raise ValueError("Reduced widths not in ladder, please convert from sammy to atari ladder first")
    LL_bypar_bysg = []
    for sg in ladder.groupby("J_ID"):

        for key, val in spin_groups.items():
            if float(val['J_ID']) == sg[0]:
                sg_key = key

        LLw = wigner_LL(sg[1].E, spin_groups[sg_key]['<D>'])
        LL_Gg = width_LL(sg[1].gg2, spin_groups[sg_key]['<gg2>'], spin_groups[sg_key]['g_dof'])
        LL_Gn = width_LL(sg[1].gn2, spin_groups[sg_key]['<gn2>'], spin_groups[sg_key]['n_dof'])
        LL_bypar_bysg.append([LLw, LL_Gg, LL_Gn])
    
    LL_bypar_bysg = np.array(LL_bypar_bysg)
    LL_bypar = np.sum(LL_bypar_bysg, axis=0)

    return LL_bypar, LL_bypar_bysg