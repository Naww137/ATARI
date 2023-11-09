# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import importlib
import subprocess
import time

from models.particle_pair import Particle_Pair
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
        Gn = np.repeat(res_par_avg["Gn01"], num_Er)*100
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
    else:
        # Gn = np.repeat(res_par_avg["<Gn>"], num_Er)
        Gn = np.repeat(res_par_avg["Gn01"], num_Er)
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
        
    Er = np.linspace(              min_Elam,              max_Elam,                num_Er)
    J_ID = np.repeat(res_par_avg["J_ID"], num_Er)

    return Er, Gg, Gn, J_ID

def get_resonance_ladder(Er, Gg, Gn1, J_ID, varyE=1, varyGg=1, varyGn1=1):
    return pd.DataFrame({"E":Er, "Gg":Gg, "Gn1":Gn1, "varyE":np.ones(len(Er))*varyE, "varyGg":np.ones(len(Er))*varyGg, "varyGn1":np.ones(len(Er))*varyGn1 ,"J_ID":J_ID})

def reduce_ladder(par, threshold, varyE=1, varyGg=1, varyGn1=1):
    par = copy(par)
    par = par[par.Gn1 > threshold]
    par["varyE"] = np.ones(len(par))
    par["varyGg"] = np.ones(len(par))
    par["varyGn1"] = np.ones(len(par))

    return par

### setup physics
spin_groups = [ (3.0,1,0) ,  (4.0,1,0)]
res_par_avg_1 = make_res_par_avg(J_ID=1,
                            D_avg = 8.79, 
                            Gn_avg= 46.4,
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = 1000, 
                            print = False)
res_par_avg_2 = make_res_par_avg(J_ID=2,
                            D_avg = 4.99, 
                            Gn_avg= 35.5,
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = 1000, 
                            print = False)

ac = 0.81271; M = 180.948030; m = 1; I = 3.5; i = 0.5; l_max = 1 
average_parameters = {'3.0':res_par_avg_1, '4.0':res_par_avg_2}
Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={},
                                spin_groups=spin_groups,
                                average_parameters=average_parameters )    

### setup sammy
sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    # path_to_SAMMY_exe = '~/my_sammy/SAMMY/sammy/build/install/bin/sammy',
    model = 'XCT',
    reaction = 'transmission',
    solve_bayes = False,
    inptemplate = "allexptot_2sg.inp",
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
    initial_parameter_uncertainty = 0.1,

    temp = 304.5,
    FP=75.0,
    frac_res_FP=0.025,
    target_thickness=0.005)


def fit(isample, num_Er, case_file, basefolder, thresh_list, title_list, Gn1_opt=0):
        
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
    sammyINPyw.templates = ["allexptot_2sg.inp", "allexpcap_2sg.inp"]

    ### step 1
    Er_1, Gg_1, Gn_1, J_ID_1 = get_parameter_grid(trans.E, res_par_avg_1, num_Er, option=Gn1_opt)
    Er_2, Gg_2, Gn_2, J_ID_2 = get_parameter_grid(trans.E, res_par_avg_2, num_Er, option=Gn1_opt)
    Er = np.concatenate([Er_1, Er_2])
    Gg = np.concatenate([Gg_1, Gg_2])
    Gn = np.concatenate([Gn_1, Gn_2])
    J_ID = np.concatenate([J_ID_1, J_ID_2])
    initial_reslad = get_resonance_ladder(Er, Gg, Gn, J_ID, varyE=0, varyGg=0, varyGn1=1)
    
    sammyINPyw.resonance_ladder = initial_reslad
    t0 = time.time()
    P1, _ = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)

    t1 = time.time()
    time_step1 = t1-t0
    P1["time"] = np.ones(len(P1))*time_step1
    P1.to_csv(os.path.join(basefolder, f"par_i{i}_iE{num_Er}.csv"))

    # step 2
    for thresh, title in zip(thresh_list, title_list):
        P2 = reduce_ladder(P1, thresh, varyE=1, varyGg=1, varyGn1=1)
        sammyINPyw.resonance_ladder = P2

        t0 = time.time()
        par, _ = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)
        final_par = reduce_ladder(par, thresh, varyE=1, varyGg=1, varyGn1=1)
        t1 = time.time()
        time_step2 = t1-t0
        final_par["time"] = np.ones(len(final_par))*time_step2
        final_par.to_csv(os.path.join(basefolder, f'step2_{title}', f"par_i{i}_iE{num_Er}.csv"))

    return 


#%%

def setup_directories(basefolder, title_list):
    if os.path.isdir(basefolder):
        for title in title_list:
            if os.path.isdir(os.path.join(basefolder, f'step2_{title}')):
                pass
            else:
                os.mkdir(os.path.join(basefolder, f'step2_{title}'))
    else:
        os.mkdir(basefolder)
        for title in title_list:
            os.mkdir(os.path.join(basefolder, f'step2_{title}'))



def main(i):

    case_file = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/data_2.hdf5"
    # case_file = "~/reg_perf_tests/sammy/data_2.hdf5"
    thresh_list, title_list = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5], ["1en1", "1en2", "1en3", "1en4", "1en5"]

    ### fit with gavg
    basefolder = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnmin"
    # basefolder = f"/home/nwalton1/reg_perf_tests/sammy/Gnavg"
    setup_directories(basefolder, title_list)
    for iE in [50, 75, 100]:
        fit(i, iE, case_file, basefolder, thresh_list, title_list, Gn1_opt=0)

    ### fit with gmin
    basefolder = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gn100min"
    # basefolder = f"/home/nwalton1/reg_perf_tests/sammy/Gnmin"
    setup_directories(basefolder, title_list)
    for iE in [50, 75, 100]:
        # if os.path.isfile(os.path.join(basefolder, f"par_i{i}_iE{iE}.csv")):
        #     pass
        # else:
        fit(i, iE, case_file, basefolder, thresh_list, title_list, Gn1_opt=1)



import sys
i = sys.argv[1]
main(i)
