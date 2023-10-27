# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import importlib
import subprocess

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


def run_YW_scheme(sammyINPyw, sammyRTO, resonance_ladder):
    
    sammyINPyw.resonance_ladder = resonance_ladder
    sammy_functions.setup_YW_scheme(sammyRTO, sammyINPyw)

    os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'iterate.{sammyRTO.shell}')}")
    os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'run.{sammyRTO.shell}')}")

    result = subprocess.check_output(os.path.join(sammyRTO.sammy_runDIR, f'run.{sammyRTO.shell}'), shell=True, text=True)
    ifinal = int(result.splitlines()[-1]) -1

    par = sammy_functions.readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{ifinal}.par"))
    lsts = []
    for dt in sammyINPyw.dataset_titles:
        lsts.append(sammy_functions.readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/{dt}_step{ifinal}.lst")) )

    return par, lsts


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

    LS = False,
    initial_parameter_uncertainty = 0.01,

    temp = 304.5,
    FP=35.185,
    frac_res_FP=0.049600,
    target_thickness=0.067166)


def fit(isample, num_Er, case_file, Gn1_opt=0):
        
    ### Import data    
    dataset_titles = ["trans1", "cap1"]
    datasets = []
    for dt in dataset_titles:
        exp_pw, _ = h5io.read_pw_exp(case_file, isample, title=dt)
        datasets.append(exp_pw)
    theo_par = h5io.read_par(case_file, isample, 'true') 
    trans = datasets[0]
    cap = datasets[1]

    ### Setup simultaneous least squares
    sammyINPyw.datasets = datasets
    sammyINPyw.dataset_titles = dataset_titles
    sammyINPyw.reactions = ["transmission", "capture"]
    sammyINPyw.templates = ["allexptot_1sg.inp", "allexpcap_1sg.inp"]

    ### step 1
    Er, Gg, Gn = get_parameter_grid(trans.E, res_par_avg, num_Er, option=Gn1_opt)
    initial_reslad = get_resonance_ladder(Er, Gg, Gn, varyE=0, varyGg=0, varyGn1=1)
    P1, _ = run_YW_scheme(sammyINPyw, sammyRTO, initial_reslad)

    # step 2
    P2 = reduce_ladder(P1, 1e-5, varyE=1, varyGg=1, varyGn1=1)
    # sammyINPyw.initial_parameter_uncertainty = 0.0
    sammyINPyw.step_threshold = 0.001
    par, _ = run_YW_scheme(sammyINPyw, sammyRTO, P2)

    return par


#%%



def main(i):

    # case_file = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/data.hdf5"
    case_file = "~/reg_perf_tests/sammy/data.hdf5"

    # root_folder = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnavg"
    basefolder = "/home/nwalton1/reg_perf_tests/sammy/SAMMY_runDIR_Gnavg"
    
    for iE in [25, 50, 75, 100]:
        par = fit(i, iE, case_file, Gn1_opt=0)
        par.to_csv(os.path.join(basefolder, f"par_i{i}_iE{iE}.csv"))








#%% plotting functions

# def plot_trans(exp_pw, T1):
    
#     figure()
#     # plot(exp_pw.E, exp_pw.theo_trans, ms=1, color='g')
#     # plot(sammyOUT.pw.E, sammyOUT.pw.theo_trans, 'r', alpha=0.2, lw=3)
#     # plot(sammyOUT_bayes.pw.E, sammyOUT_bayes.pw.theo_trans_bayes, 'b-')
#     plot(T1.E, T1.theo_trans, 'b')
#     errorbar(exp_pw.E, exp_pw.exp, yerr=exp_pw.exp_unc, zorder=0, 
#                                             fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
#     ylim([-.1, 1])


# def plot_trans_cap(exp_pw, cap_pw, T1=None,C1=None):

#     fig, axes = subplots(2,1, figsize=(8,6), sharex=True)
#     axes[0].errorbar(exp_pw.E, exp_pw.exp, yerr=exp_pw.exp_unc, zorder=0, 
#                                             fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
#     axes[1].errorbar(cap_pw.E, cap_pw.exp, yerr=cap_pw.exp_unc, zorder=0, 
#                                             fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
#     if C1 is not None and T1 is not None:
#         axes[0].plot(T1.E, T1.theo_trans, 'b')
#         axes[1].plot(C1.E, C1.theo_xs, 'b')
#     else:
#         axes[0].plot(exp_pw.E, exp_pw.theo_trans, 'g')
#         axes[1].plot(cap_pw.E, cap_pw.theo_xs, 'g')
        

#     axes[0].set_ylabel("T")
#     axes[1].set_yscale('log')
#     axes[1].set_ylabel(r'$\sigma_{\gamma}$ (barns)')
#     axes[1].set_ylim(bottom=5e-4)

#     # legend()
#     fig.supxlabel('Energy (eV)')
#     fig.tight_layout()
#     return fig