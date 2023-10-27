# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import importlib
import subprocess
import imageio

from theory.particle_pair import Particle_Pair
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions

import ATARI.utils.io.hdf5 as h5io

from copy import copy
from ATARI.theory.resonance_statistics import make_res_par_avg

from ATARI.theory.scattering_params import FofE_explicit, FofE_recursive

# %%
# Functions for setting up feature bank

def get_parameter_grid(energy_grid, res_par_avg, num_Er, option=0):

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_grid) + res_par_avg['Gt99']/10e3
    min_Elam = min(energy_grid) - res_par_avg['Gt99']/10e3

    if option == 1:
        Gn = np.repeat(res_par_avg["Gn01"], num_Er)*10
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
    else:
        Gn = np.repeat(res_par_avg["<Gn>"], num_Er)
        Gg = np.repeat(res_par_avg["<Gg>"], num_Er)
        
    Er = np.linspace(              min_Elam,              max_Elam,                num_Er)
    J_ID = np.repeat(res_par_avg["J_ID"], num_Er)

    return Er, Gg, Gn, J_ID

def get_resonance_ladder(Er, Gg, Gn1, J_ID, varyE=1, varyGg=1, varyGn1=1):
    return pd.DataFrame({"E":Er, "Gg":Gg, "Gn1":Gn1, "varyE":np.ones(len(Er))*varyE, "varyGg":np.ones(len(Er))*varyGg, "varyGn1":np.ones(len(Er))*varyGn1 ,"J_ID":J_ID})

# def get_resonance_ladder_df(resonance_ladder_array):

def plot_trans(exp_pw, T1):
    
    fig = figure()
    # plot(exp_pw.E, exp_pw.theo_trans, ms=1, color='g')
    # plot(sammyOUT.pw.E, sammyOUT.pw.theo_trans, 'r', alpha=0.2, lw=3)
    # plot(sammyOUT_bayes.pw.E, sammyOUT_bayes.pw.theo_trans_bayes, 'b-')
    plot(T1.E, T1.theo_trans, 'b')
    errorbar(exp_pw.E, exp_pw.exp, yerr=exp_pw.exp_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    ylim([-.1, 1])
    return fig


def plot_trans_cap(trans, cap, T1=None,C1=None, plot_true=False):

    fig, axes = subplots(2,1, figsize=(8,6), sharex=True)
    axes[0].errorbar(trans.E, trans.exp, yerr=trans.exp_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
    axes[1].errorbar(cap.E, cap.exp, yerr=cap.exp_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
    if C1 is not None and T1 is not None:
        axes[0].plot(T1.E, T1.theo_trans, 'b')
        axes[1].plot(C1.E, C1.theo_xs, 'b')
        if plot_true:
            axes[0].plot(trans.E, trans.true, 'g')
            axes[1].plot(cap.E, cap.true, 'g')
    else:
        axes[0].plot(trans.E, trans.true, 'g')
        axes[1].plot(cap.E, cap.true, 'g')
        

    axes[0].set_ylabel("T")
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$\sigma_{\gamma}$ (barns)')
    # axes[1].set_ylim([-0.01,1.01])

    # legend()
    fig.supxlabel('Energy (eV)')
    fig.tight_layout()
    return fig


def make_gif(sammy_runDIR, pngpath, gifpath, ifinal,trans, cap):

    for i in range(ifinal):
        
        C1 = sammy_functions.readlst(os.path.join(sammy_runDIR,f"results/cap1_step{i}.lst"))
        T1 = sammy_functions.readlst(os.path.join(sammy_runDIR,f"results/trans1_step{i}.lst"))

        fig = plot_trans_cap(trans, cap, T1=T1, C1=C1)
        title(f"step {i}")
        fig.savefig(os.path.join(pngpath, f"{i}.png"))
        close()


    images = []
    for i in range(ifinal): #range(start_job,end_job):
        images.append(imageio.imread(os.path.join(pngpath, f"{i}.png")))
    imageio.mimsave(gifpath, images)


# %%
case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/data_test.hdf5'  # if using hdf5

isample = 2

dataset_titles = ["trans1", "cap1"]
datasets = []
for dt in dataset_titles:
    exp_pw, exp_cov = h5io.read_pw_exp(case_file, isample, title=dt)
    datasets.append(exp_pw)

theo_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical
theo_par["varyGn1"] = np.ones(len(theo_par))
theo_par["varyGg"] = np.ones(len(theo_par))
theo_par["varyE"] = np.ones(len(theo_par))

trans = datasets[0]
cap = datasets[1]

# fig = plot_trans_cap(trans, cap, T1=None, C1=None)

# for ir, row in theo_par.iterrows():
#     if row.J_ID == 1:
#         c = 'red'
#     elif row.J_ID ==2:
#         c='b'
#     axvline(row.E, ymin=0, ymax=1, alpha=0.5, color=c)

# %%
spin_groups = [ (3.0,1,0) ,  (4.0,1,0)]
res_par_avg_1 = make_res_par_avg(J_ID=1,
                            D_avg = 8.79, 
                            Gn_avg= 46.4, #0.658, 
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = 1000, 
                            print = False)
res_par_avg_2 = make_res_par_avg(J_ID=2,
                            D_avg = 4.99, 
                            Gn_avg= 35.5, #0.658, 
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

num_Er = 50
Er_1, Gg_1, Gn_1, J_ID_1 = get_parameter_grid(exp_pw.E, res_par_avg_1, num_Er, option=1)
Er_2, Gg_2, Gn_2, J_ID_2 = get_parameter_grid(exp_pw.E, res_par_avg_2, num_Er, option=1)

Er = np.concatenate([Er_1, Er_2])
Gg = np.concatenate([Gg_1, Gg_2])
Gn = np.concatenate([Gn_1, Gn_2])
J_ID = np.concatenate([J_ID_1, J_ID_2])
initial_reslad = get_resonance_ladder(Er, Gg, Gn, J_ID, varyE=0, varyGg=0, varyGn1=1)
# # initial_reslad

# %%
# columns=['E', 'Gg', 'Gn1', 'Gn2', 'Gn3', 'varyE', 'varyGg', 'varyGn1', 'varyGn2', 'varyGn3', 'J_ID']
# importlib.reload(sammy_functions) 
# sammy_functions.write_sampar(initial_reslad, Ta_pair, 0.1, "./test")

sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    model = 'XCT',
    reaction = 'transmission',
    solve_bayes = False,
    inptemplate = "allexptot_2sg.inp",
    inpname = "sammy.inp",
    energy_window = None,
    sammy_runDIR = 'SAMMY_runDIR',
    keep_runDIR = True,
    shell = 'zsh'
    )


reactions = ["transmission", "capture"]
templates = ["allexptot_2sg.inp", "broadcap_2sg.inp"]
# templates = ["allexptot_1sg.inp", "allexpcap_1sg.inp"]

sammyINPyw = sammy_classes.SammyInputDataYW(
    particle_pair = Ta_pair,
    # resonance_ladder = initial_reslad,  np.ones(len(theo_par))*1
    resonance_ladder = pd.concat([initial_reslad, get_resonance_ladder(theo_par["E"].values, np.ones(len(theo_par))*64, theo_par["Gn1"].values*0.5, theo_par["J_ID"].values)]), #theo_par,

    datasets= datasets,
    dataset_titles= dataset_titles,
    reactions= reactions,
    templates= templates,
    
    steps = 200,
    iterations = 2,
    step_threshold = 0.1,
    autoelim_threshold = None,

    LS = False,
    initial_parameter_uncertainty = 0.01,

    temp = 304.5,
    FP=75.0,
    frac_res_FP=0.025,
    target_thickness=0.005)


# par, lsts = sammy_functions.run_sammy_YW(sammyINPyw, sammyRTO)


# sammy_functions.setup_YW_scheme(sammyRTO, sammyINPyw)
# os.system(f"chmod +x {'/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR/iterate.sh'}")
# # os.system(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR/iterate.sh 0")

for i in range(10):
    # chi2 = subprocess.check_output(os.path.join('.',sammyRTO.sammy_runDIR, f'iterate.sh {i}'))
    # print(chi2)
    os.system(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR/iterate.sh {i}")
    # par = sammy_functions.readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{i}.par"))
    # lsts = []
    # for dt in sammyINPyw.dataset_titles:
    #     lsts.append(sammy_functions.readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/{dt}_step{i}.lst")) )

    if i > 0:
        C1 = sammy_functions.readlst(os.path.realpath( os.path.join(sammyRTO.sammy_runDIR,f"results/cap1_step{i}.lst")))
        T1 = sammy_functions.readlst(os.path.realpath( os.path.join(sammyRTO.sammy_runDIR,f"results/trans1_step{i}.lst")))

        fig = plot_trans_cap(trans, cap, T1=T1, C1=C1)
        title(f"step {i}")
        fig.savefig(os.path.join("/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/figures", f"{i}.png"))
        close()

