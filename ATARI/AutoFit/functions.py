from ATARI.theory.resonance_statistics import make_res_par_avg
import numpy as np
import pandas as pd
from copy import copy



def get_parameter_grid(energy_range, res_par_avg, num_Er, option=0):

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_range) + res_par_avg['Gt99']/10e3
    min_Elam = min(energy_range) - res_par_avg['Gt99']/10e3

    if option == 1:
        Gn = np.repeat(res_par_avg["Gn01"], num_Er)*10
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
    else:
        Gn = np.repeat(res_par_avg["Gn01"], num_Er)*1000
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
        
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