# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import importlib
import subprocess

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions

import ATARI.utils.io.hdf5 as h5io

from copy import copy
from ATARI.theory.resonance_statistics import make_res_par_avg

# %%
# Functions for setting up feature bank

def get_parameter_grid(energy_grid, res_par_avg, num_Er, option=0):

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_grid) + res_par_avg['Gt99']/10e3
    min_Elam = min(energy_grid) - res_par_avg['Gt99']/10e3

    if option == 1:
        Gn = np.repeat(res_par_avg["Gn01"], num_Er)
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
    else:
        Gn = np.repeat(res_par_avg["<Gn>"], num_Er)
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
        
    Er = np.linspace(              min_Elam,              max_Elam,                num_Er)
    
    return Er, Gg, Gn

def get_resonance_ladder(Er, Gg, Gn1, varyE=1, varyGg=1, varyGn1=1):
    return pd.DataFrame({"E":Er, "Gg":Gg, "Gn1":Gn1, "varyE":np.ones(len(Er))*varyE, "varyGg":np.ones(len(Er))*varyGg, "varyGn1":np.ones(len(Er))*varyGn1 ,"J_ID":np.ones(len(Er))})

def reduce_ladder(par, threshold, varyE=1, varyGg=1, varyGn1=1):
    par = copy(par)
    par = par[par.Gn1 > threshold]
    par["varyE"] = np.ones(len(par))
    par["varyGg"] = np.ones(len(par))
    par["varyGn"] = np.ones(len(par))

    return par

### setup physics
Gg_DOF = 10
spin_groups = [ (3.0,1,0) ]
res_par_avg = make_res_par_avg(D_avg = 8.79, 
                            Gn_avg= 0.658, #0.658, 
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = Gg_DOF, 
                            print = False)

ac = 0.81271; M = 180.948030; m = 1; I = 3.5; i = 0.5; l_max = 1 
average_parameters = {'3.0':res_par_avg}
Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={},
                                spin_groups=spin_groups,
                                average_parameters=average_parameters )   

### setup sammy
sammyRTO = sammy_classes.SammyRunTimeOptions(
    # path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    path_to_SAMMY_exe = '~/my_sammy/SAMMY/sammy/build/install/bin/sammy',
    model = 'XCT',
    reaction = 'transmission',
    solve_bayes = False,
    inptemplate = "allexptot_1sg.inp",
    inpname = "sammy.inp",
    energy_window = None,
    sammy_runDIR = f"SAMMY_runDIR",
    keep_runDIR = False,
    shell = 'zsh'
    )


sammyINPyw = sammy_classes.SammyInputDataYW(
    particle_pair = Ta_pair,
    resonance_ladder = pd.DataFrame(),

    datasets= [],
    dataset_titles= [],
    reactions= [],
    templates= [],
    
    steps = 200,
    iterations = 2,
    step_threshold = 0.01,
    autoelim_threshold = None,

    LS = False,
    initial_parameter_uncertainty = 0.01,

    temp = 304.5,
    FP=35.185,
    frac_res_FP=0.049600,
    target_thickness=0.067166)


def fit(isample, num_Er, case_file, elim_threshold=1e-5, Gn1_opt=0):
        
    ### Import data    
    dataset_titles = ["trans1", "cap1"]
    datasets = []
    for dt in dataset_titles:
        exp_pw, _ = h5io.read_pw_exp(case_file, isample, title=dt)
        datasets.append(exp_pw)
    # theo_par = h5io.read_par(case_file, isample, 'true') 
    trans = datasets[0]
    cap = datasets[1]

    ### Setup simultaneous least squares
    # sammyINPyw.autoelim_threshold = elim_threshold  # cant do eliminations every step because the awk commands wont format correctly if threshold >1e-4
    sammyRTO.sammy_runDIR = f"SAMMY_runDIR_{isample}"
    sammyINPyw.datasets = datasets
    sammyINPyw.dataset_titles = dataset_titles
    sammyINPyw.reactions = ["transmission", "capture"]
    sammyINPyw.templates = ["allexptot_1sg.inp", "allexpcap_1sg.inp"]

    ### step 1
    Er, Gg, Gn = get_parameter_grid(trans.E, res_par_avg, num_Er, option=Gn1_opt)
    initial_reslad = get_resonance_ladder(Er, Gg, Gn, varyE=0, varyGg=0, varyGn1=1)
    
    sammyINPyw.resonance_ladder = initial_reslad
    P1, _ = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)

    # step 2
    P2 = reduce_ladder(P1, elim_threshold, varyE=1, varyGg=1, varyGn1=1)
    sammyINPyw.step_threshold = 0.001
    sammyINPyw.resonance_ladder = P2
    par, _ = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)

    final_par = reduce_ladder(par, elim_threshold, varyE=1, varyGg=1, varyGn1=1)
    return final_par


#%%



def main(i):

    # case_file = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/data.hdf5"
    case_file = "~/reg_perf_tests/sammy/data.hdf5"


    ## fit with gavg
    for thresh, title in zip([1e-4, 1e-5], ["1en4", "1en5"]):
        # root_folder = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnavg"
        basefolder = f"/home/nwalton1/reg_perf_tests/sammy/Gnavg_fits_{title}"
    
        for iE in [25, 50, 75, 100]:
            par = fit(i, iE, case_file, elim_threshold=thresh, Gn1_opt=0)
            par.to_csv(os.path.join(basefolder, f"par_i{i}_iE{iE}.csv"))


    ## fit with gmin
    for thresh, title in zip([1e-4, 1e-5], ["1en4", "1en5"]):
        # root_folder = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnavg"
        basefolder = f"/home/nwalton1/reg_perf_tests/sammy/Gnmin_fits_{title}"
    
        for iE in [25, 50, 75, 100]:
            if os.path.isfile(os.path.join(basefolder, f"par_i{i}_iE{iE}.csv")):
                pass
            else:
                par = fit(i, iE, case_file, elim_threshold=thresh, Gn1_opt=1)
                par.to_csv(os.path.join(basefolder, f"par_i{i}_iE{iE}.csv"))



import sys
i = sys.argv[1]
main(i)
