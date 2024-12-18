import numpy as np
import pandas as pd
from copy import copy
from ATARI.theory.resonance_statistics import wigner_LL, width_LL
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.utils.atario import add_Gw_from_gw


def get_parameter_grid(energy_range, res_par_avg, particle_pair, num_Er, starting_Gg_multiplier, starting_Gn1_multiplier, do_dE_function):

    if len(res_par_avg["Ls"]) > 1:
            raise NotImplementedError("Multiple Ls to one spin group has not been implemented")
    else:
        L = res_par_avg["Ls"][0]

    # allow Elambda to be just outside of the window
    _, P_array, _, _ = FofE_recursive(np.sort(energy_range), particle_pair.ac, particle_pair.M, particle_pair.m, L)
    Gt99_min_max = res_par_avg['quantiles']['gt99']*P_array[0]
    max_Elam = max(energy_range) + Gt99_min_max[0]/10e3
    min_Elam = min(energy_range) - Gt99_min_max[1]/10e3

    # get energies and spin info
    if do_dE_function:
        x_start = min_Elam; x_end = max_Elam
        d_start = 0.8; d_end = 2.0
        Er = [x_start]
        x = x_start
        while x < x_end:
            t = (x - x_start) / (x_end - x_start)
            d = d_start + t * (d_end - d_start)
            x += d
            Er.append(x)
        Er = np.array(Er)
        num_Er = len(Er)
    else:
        Er = np.linspace(              min_Elam,              max_Elam,                num_Er)

    # get widths
    gg2 = np.repeat(res_par_avg["<gg2>"], num_Er)*starting_Gg_multiplier
    gn2 = np.repeat(res_par_avg['quantiles']["gn01"], num_Er)*starting_Gn1_multiplier
    
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
    return_resonance_ladder.reset_index(inplace=True, drop=True)
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
                            varyGn1 = 1,
                            do_dE_function = False
                            ):
    # setup energy grid
    if num_Elam is None:
        num_Elam = int((np.max(energy_range)-np.min(energy_range)) * 1.25)
    elif num_Elam/(np.max(energy_range)-np.min(energy_range)) < 1.25:
        print("WARNING: User supplied a feature bank energy grid of <1 per eV, problem may not be convex")

    Er, gg2, gn2, J_ID, Jpi, L = [], [], [], [], [], []
    for sg in spin_groups:
        Er_1, gg2_1, gn2_1, J_ID_1, Jpi_1, L_1 = get_parameter_grid(energy_range, sg, particle_pair, num_Elam, starting_Gg_multiplier, starting_Gn1_multiplier, do_dE_function)
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
    external_resonance_ladder = resonance_ladder.loc[external_resonance_indices, :]
    internal_resonance_ladder = copy(resonance_ladder)
    internal_resonance_ladder.drop(index=external_resonance_indices, inplace=True)
    return internal_resonance_ladder, external_resonance_ladder


def concat_external_resonance_ladder(internal_resonance_ladder, external_resonance_ladder):
    if external_resonance_ladder.empty:
        resonance_ladder = internal_resonance_ladder
        external_resonance_indices = []
    else:
        assert(np.all([each in external_resonance_ladder.keys() for each in internal_resonance_ladder.keys()]))
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





def get_initial_resonance_ladder(initialFBopt, particle_pair, energy_window, external_resonance_ladder=None, do_dE_function=False):

    ### setup spin groups
    if initialFBopt.fit_all_spin_groups:
        spin_groups = [each[1] for each in particle_pair.spin_groups.items()] 
    else:
        assert len(initialFBopt.spin_group_keys)>0
        spin_groups = [each[1] for each in particle_pair.spin_groups.items() if each[0] in initialFBopt.spin_group_keys]

    ### generate intial_feature bank
    initial_resonance_ladder = get_starting_feature_bank(energy_window,
                                                        particle_pair,
                                                        spin_groups,
                                                        num_Elam= initialFBopt.num_Elam,
                                                        Elam_shift = initialFBopt.Elam_shift,
                                                        starting_Gg_multiplier = initialFBopt.starting_Gg_multiplier,
                                                        starting_Gn1_multiplier = initialFBopt.starting_Gn1_multiplier,
                                                        do_dE_function = do_dE_function)

    ### setup external resonances
    if initialFBopt.external_resonances:
        # external_resonance_ladder = generate_external_resonance_ladder(spin_groups, energy_window, particle_pair)
        if external_resonance_ladder is None:
            external_resonance_ladder = generate_external_resonance_ladder(spin_groups, energy_window, particle_pair)
        else:
            assert(np.all([each in external_resonance_ladder.keys() for each in initial_resonance_ladder.keys()]))
    else:
        external_resonance_ladder = pd.DataFrame()
    # initial_resonance_ladder, external_resonance_indices = concat_external_resonance_ladder(initial_resonance_ladder, external_resonance_ladder)

    return initial_resonance_ladder, external_resonance_ladder







from ATARI.theory.resonance_statistics import wigner_LL
from ATARI.ModelData.particle_pair import Particle_Pair

def objective_func(chi2, res_ladder, particle_pair:Particle_Pair, fixed_resonances_indices, 
                   Wigner_informed=True, PorterThomas_informed=False, Elimits = (0)):
    
    # Getting internal ladder:
    # fixed_resonances_indices = res_ladder[res_ladder['E'].isin(fixed_res_energies)].index.tolist()
    # res_ladder_internal = res_ladder.drop(index=fixed_resonances_indices)
    res_ladder_internal = res_ladder
    
    if Wigner_informed or PorterThomas_informed:
        if particle_pair is None:
            raise ValueError('Particle Pair must be included if using resonance statistics.')
    
        spingroups_old = particle_pair.spin_groups
        spingroups_JID = {}
        for Jpi, spingroup in spingroups_old.items():
            spingroups_JID[spingroup['J_ID']] = spingroup
        log_likelihood = 0.0
        for J_ID, spingroup in spingroups_JID.items():
            partial_ladder = res_ladder.loc[res_ladder['J_ID'] == J_ID]
            E = partial_ladder['E'].to_numpy(dtype=float)

            partial_ladder_internal = res_ladder_internal.loc[res_ladder_internal['J_ID'] == J_ID]
            E_int = partial_ladder_internal['E'].to_numpy(dtype=float)
            Gn_int = partial_ladder_internal['Gn1'].to_numpy(dtype=float) # Porter-Thomas distribution should only be used on the internal resonances

            if Wigner_informed:
                mean_level_spacing = spingroup['<D>']
                log_likelihood += wigner_LL(E, mean_level_spacing)
                wigner_correction_factor = (len(E) - 1) * np.sqrt(np.pi/(2*np.e)) / mean_level_spacing # to account for virtual resonances
                log_likelihood -= wigner_correction_factor

            if PorterThomas_informed:
                mean_neutron_width = spingroup['<gn2>']
                if len(spingroup['Ls']) > 1:
                    raise NotImplementedError('Cannot do Porter-Thomas when more than one l quantum state shares the same Jpi.')
                l = int(spingroup['Ls'][0])
                gn2 = particle_pair.Gn_to_gn2(Gn_int, E_int, l)
                log_likelihood += -np.sum(gn2)/(2*mean_neutron_width) - len(gn2)*0.5*np.log(2*np.pi*mean_neutron_width)
                log_likelihood -= 0 - len(gn2)*0.5*np.log(2*np.pi*mean_neutron_width) # to account for virtual resonances
    else:
        log_likelihood = 0.0

    obj = chi2 - 2*log_likelihood
    return obj
