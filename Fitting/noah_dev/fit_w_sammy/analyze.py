# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import importlib
import subprocess
import imageio

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions

import ATARI.utils.io.hdf5 as h5io

from copy import copy
from ATARI.theory.resonance_statistics import make_res_par_avg


# %%
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


def plot_trans_cap(T1, C1):

    fig, axes = subplots(2,1, figsize=(8,6), sharex=True)
    axes[0].errorbar(T1.E, T1.exp_trans, yerr=T1.exp_trans_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
    axes[1].errorbar(C1.E, C1.exp_xs, yerr=C1.exp_xs_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
    axes[0].plot(T1.E, T1.theo_trans, 'b')
    axes[1].plot(C1.E, C1.theo_xs, 'b')


    axes[0].set_ylabel("T")
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$\sigma_{\gamma}$ (barns)')
    axes[1].set_ylim(bottom=5e-4)

    # legend()
    fig.supxlabel('Energy (eV)')
    fig.tight_layout()
    return fig

def plot_Gnhist(P):
        
    fig = figure()
    bins = hist(np.log10(P.Gn1), bins=75, density=True, label="Gn")
    # bins = hist(np.log10(P1.Gg), bins=75, density=True, alpha=0.75, label="Gg")
    xlabel(r'Log10($\Gamma$)'); title("Parameter Frequency on Final Step")
    legend()
    xlim([-8, 3])

    return fig

# %%
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

# # def get exp(isample)
# sammyRTO = sammy_classes.SammyRunTimeOptions(
#     path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
#     model = 'XCT',
#     reaction = 'transmission',
#     solve_bayes = False,
#     inptemplate = "allexptot_1sg.inp",
#     inpname = "sammy.inp",
#     energy_window = None,
#     sammy_runDIR = 'SAMMY_runDIR',
#     keep_runDIR = True,
#     shell = 'zsh'
#     )

# sammyINP = sammy_classes.SammyInputData(
#     particle_pair = Ta_pair,
#     resonance_ladder = theo_par,
#     experimental_data=exp_pw,
#     temp = 304.5,
#     FP=35.185,
#     frac_res_FP=0.049600,
#     target_thickness=0.067166,
#     initial_parameter_uncertainty=1.0)

# sammyOUT = sammy_functions.run_sammy(sammyINP, sammyRTO)

# # %%
# ## capture data if using
# sammyRTO_cap = copy(sammyRTO)
# sammyRTO_cap.inptemplate = 'allexpcap_1sg.inp'
# sammyRTO_cap.reaction = 'capture'

# sammyINP.resonance_ladder = theo_par
# out_cap = sammy_functions.run_sammy(sammyINP, sammyRTO_cap)

# unc_scale = 0.05
# cap_pw = out_cap.pw
# cap_pw["exp"] = abs(np.random.default_rng().normal(cap_pw["theo_xs"], np.sqrt(cap_pw["theo_xs"])*unc_scale ))
# cap_pw["exp_unc"] = np.sqrt(cap_pw["exp"])*unc_scale

# fig = plot_trans_cap(exp_pw, cap_pw, T1=None, C1=None)

# # %%

# # ifinal = int(result.splitlines()[-1]) -1
# ifinal=82

# P1 = sammy_functions.readpar(os.path.join(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_step1",f"results/step{ifinal}.par"))
# T1 = sammy_functions.readlst(os.path.join(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_step1",f"results/trans1_step{ifinal}.lst"))
# C1 = sammy_functions.readlst(os.path.join(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_step1",f"results/cap1_step{ifinal}.lst"))


# # %%
# figure()
# bins = hist(np.log10(P1.Gn1), bins=75, density=True, label="Gn")
# # bins = hist(np.log10(P1.Gg), bins=75, density=True, alpha=0.75, label="Gg")
# xlabel(r'Log10($\Gamma$)'); title("Parameter Frequency on Final Step")
# legend()
# xlim([-8, 3])
# # savefig(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/case_2a.png")

# fig = plot_trans_cap(exp_pw, cap_pw, T1=T1, C1=C1)
# # fig = plot_trans(exp_pw, T1)

# # %%

# # ifinal = int(result.splitlines()[-1]) -1
# ifinal=2

# P2 = sammy_functions.readpar(os.path.join(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_step2",f"results/step{ifinal}.par"))
# T2 = sammy_functions.readlst(os.path.join(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_step2",f"results/trans1_step{ifinal}.lst"))
# C2 = sammy_functions.readlst(os.path.join(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_step2",f"results/cap1_step{ifinal}.lst"))


# fig = plot_trans_cap(exp_pw, cap_pw, T1=T2, C1=C2)

#%% 

def make_fitgif(isample, iE):

    with open(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/ifinal_nE{iE}.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(f"{isample} "):
            ifinals = line.split()

    for istep in [1,2,3,4]:
        folder = f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_iE{iE}_step{istep}"
        for i in range(1,int(ifinals[istep])):
            
            C1 = sammy_functions.readlst(os.path.join(folder,f"results/cap1_step{i}.lst"))
            T1 = sammy_functions.readlst(os.path.join(folder,f"results/trans1_step{i}.lst"))
            fig = plot_trans_cap(T1, C1)
            fig.suptitle(f"Step: {istep}")
            fig.tight_layout()
            fig.savefig(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/figures/sample{isample}_iE{iE}_step{istep}_{i}.png")
            close()
        
    images = []
    for istep in [1,2,3,4]:
        for i in range(1,int(ifinals[istep])): #range(start_job,end_job):
            images.append(imageio.imread(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/figures/sample{isample}_iE{iE}_step{istep}_{i}.png"))
    imageio.mimsave(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/sample{isample}_iE{iE}.gif", images)



def make_histgif(isample, iE):

    with open(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/ifinal_nE{iE}.txt", 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith(f"{isample} "):
            ifinals = line.split()
    
    for istep in [1,2,3,4]:
        folder = f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_iE{iE}_step{istep}"
        for i in range(1,int(ifinals[istep])):
            
            # C1 = sammy_functions.readlst(os.path.join(folder,f"results/cap1_step{i}.lst"))
            # T1 = sammy_functions.readlst(os.path.join(folder,f"results/trans1_step{i}.lst"))
            # fig = plot_trans_cap(exp_pw, cap_pw, T1=T1, C1=C1)
            P1 = sammy_functions.readpar(os.path.join(folder,f"results/step{i}.par"))
            fig = plot_Gnhist(P1)
            fig.suptitle(f"Step: {istep}")
            fig.tight_layout()
            fig.savefig(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/figures/sample{isample}_iE{iE}_step{istep}_{i}_hist.png")
            close()
        
    images = []
    for istep in [1,2,3,4]:
        for i in range(1,int(ifinals[istep])): #range(start_job,end_job):
            images.append(imageio.imread(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/figures/sample{isample}_iE{iE}_step{istep}_{i}_hist.png"))
    imageio.mimsave(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/sample{isample}_iE{iE}_hist.gif", images)

# %%

iE = 60
for isample in range(1,10):

    make_fitgif(isample, iE)
    make_histgif(isample, iE)


# %%


# %%
# os.system(os.path.join(sammyRTO.sammy_runDIR, 'run.zsh'))


# %%


# %%


# %%



