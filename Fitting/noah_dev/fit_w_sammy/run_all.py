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

    par = par[par.Gn1 > threshold]
    par["varyE"] = np.ones(len(par))
    par["varyGg"] = np.ones(len(par))
    par["varyGn"] = np.ones(len(par))

    return par

def plot_trans(exp_pw, T1):
    
    figure()
    # plot(exp_pw.E, exp_pw.theo_trans, ms=1, color='g')
    # plot(sammyOUT.pw.E, sammyOUT.pw.theo_trans, 'r', alpha=0.2, lw=3)
    # plot(sammyOUT_bayes.pw.E, sammyOUT_bayes.pw.theo_trans_bayes, 'b-')
    plot(T1.E, T1.theo_trans, 'b')
    errorbar(exp_pw.E, exp_pw.exp, yerr=exp_pw.exp_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    ylim([-.1, 1])


def plot_trans_cap(exp_pw, cap_pw, T1=None,C1=None):

    fig, axes = subplots(2,1, figsize=(8,6), sharex=True)
    axes[0].errorbar(exp_pw.E, exp_pw.exp, yerr=exp_pw.exp_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
    axes[1].errorbar(cap_pw.E, cap_pw.exp, yerr=cap_pw.exp_unc, zorder=0, 
                                            fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')
    
    if C1 is not None and T1 is not None:
        axes[0].plot(T1.E, T1.theo_trans, 'b')
        axes[1].plot(C1.E, C1.theo_xs, 'b')
    else:
        axes[0].plot(exp_pw.E, exp_pw.theo_trans, 'g')
        axes[1].plot(cap_pw.E, cap_pw.theo_xs, 'g')
        

    axes[0].set_ylabel("T")
    axes[1].set_yscale('log')
    axes[1].set_ylabel(r'$\sigma_{\gamma}$ (barns)')
    axes[1].set_ylim(bottom=5e-4)

    # legend()
    fig.supxlabel('Energy (eV)')
    fig.tight_layout()
    return fig


def main(isample, num_Er, Gn1_opt = 0):
        
    # %% Import data
    case_file = '/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/data.hdf5'  # if using hdf5

    # isample = 2
    exp_pw, exp_cov = h5io.read_pw_exp(case_file, isample)
    theo_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical

    # %% setup physics
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

    # %% setup sammy
    sammyRTO = sammy_classes.SammyRunTimeOptions(
        path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
        model = 'SLBW',
        reaction = 'transmission',
        solve_bayes = False,
        inptemplate = "allexptot_1sg.inp",
        inpname = "sammy.inp",
        energy_window = None,
        sammy_runDIR = f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnavg/sample_{isample}_iE{iE}_step1",
        keep_runDIR = True,
        shell = 'zsh'
        )

    sammyINP = sammy_classes.SammyInputData(
        particle_pair = Ta_pair,
        resonance_ladder = pd.DataFrame(),
        experimental_data= exp_pw,
        temp = 304.5,
        FP=35.185,
        frac_res_FP=0.049600,
        target_thickness=0.067166,
        initial_parameter_uncertainty=1.0)

    # %% create capture data

    sammyRTO_cap = copy(sammyRTO)
    sammyRTO_cap.inptemplate = 'allexpcap_1sg.inp'
    sammyRTO_cap.reaction = 'capture'

    sammyINP.resonance_ladder = theo_par
    out_cap = sammy_functions.run_sammy(sammyINP, sammyRTO_cap)

    unc_scale = 0.05
    cap_pw = out_cap.pw
    cap_pw["exp"] = abs(np.random.default_rng().normal(cap_pw["theo_xs"], np.sqrt(cap_pw["theo_xs"])*unc_scale ))
    cap_pw["exp_unc"] = np.sqrt(cap_pw["exp"])*unc_scale

    # %% Setup simultaneous least squares

    datasets = [exp_pw, cap_pw]
    dataset_titles = ["trans1", "cap1"]
    reactions = ["transmission", "capture"]
    templates = ["/Users/noahwalton/Documents/GitHub/ATARI/ATARI/sammy_interface/sammy_templates/allexptot_1sg.inp", "/Users/noahwalton/Documents/GitHub/ATARI/ATARI/sammy_interface/sammy_templates/allexpcap_1sg.inp"]
    
    iterations = 2
    steps = 200
    threshold = 0.001
    sammyRTO.keep_runDIR = True

    def run_YW_scheme(sammyINP, sammyRTO, resonance_ladder):
        sammyINP.resonance_ladder = resonance_ladder

        sammy_functions.setup_YW_scheme(sammyRTO, sammyINP, datasets, dataset_titles, reactions, templates, 
                                                                                        steps=steps,
                                                                                        iterations=iterations,
                                                                                        threshold=threshold)

        os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'iterate.{sammyRTO.shell}')}")
        os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'run.{sammyRTO.shell}')}")

        result = subprocess.check_output(os.path.join(sammyRTO.sammy_runDIR, f'run.{sammyRTO.shell}'), shell=True, text=True)
        ifinal = int(result.splitlines()[-1]) -1
        return ifinal

    #%%

    # step 1
    Er, Gg, Gn = get_parameter_grid(exp_pw.E, res_par_avg, num_Er, option=Gn1_opt)
    initial_reslad = get_resonance_ladder(Er, Gg, Gn, varyE=0, varyGg=0, varyGn1=1)

    ifinal1 = run_YW_scheme(sammyINP, sammyRTO, initial_reslad)

    # step 2
    P1 = sammy_functions.readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{ifinal1}.par"))
    sammyRTO.sammy_runDIR = f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnavg/sample_{isample}_iE{iE}_step2"
    P2 = reduce_ladder(P1, 1e-3, varyE=1, varyGg=1, varyGn1=1)
    ifinal2 = run_YW_scheme(sammyINP, sammyRTO, P2)

    # # step 3
    # P2 = sammy_functions.readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{ifinal2}.par"))
    # sammyRTO.sammy_runDIR = f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_1en3/sample_{isample}_iE{iE}_step3"
    # P3 = reduce_ladder(P2, 1e-3, varyE=1, varyGg=1, varyGn1=1)
    # ifinal3 = run_YW_scheme(sammyINP, sammyRTO, P3)

    return ifinal1, ifinal2 #, ifinal3, ifinal4


#%%

root_folder = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIR_Gnavg"

for iE in [50, 75, 100]:

    for i in range(0, 100):

        # ifinal, ifinal2, ifinal3, ifinal4 = main(i, iE, Gn1_opt=0)
        ifinal, ifinal2 = main(i, iE, Gn1_opt=0)


        with open(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/ifinal_nE{iE}.txt", "a+") as f:
            # f.write(f"{i} {ifinal} {ifinal2} {ifinal3} {ifinal4}\n")
            f.write(f"{i} {ifinal} {ifinal2}\n")



    # sammy_functions.write_sampar(initial_reslad, sammyINP.particle_pair, 0.1, os.path.join(sammyRTO_ls.sammy_runDIR, "results/step0.par"))

    # newest = P1[P1.Gn1 > 5e-2]
    # newest["varyE"] = np.ones(len(newest))
    # sammy_functions.write_sampar(newest, sammyINP.particle_pair, 0.1, os.path.join(sammyRTO_ls.sammy_runDIR, "results/step0.par"))


    # # %%
    # ifinal2 = 200
    # P2 = sammy_functions.readpar(os.path.join(sammyRTO_ls.sammy_runDIR,f"results/step{ifinal2}.par"))
    # T2 = sammy_functions.readlst(os.path.join(sammyRTO_ls.sammy_runDIR,f"results/trans1_step{ifinal2}.lst"))

    # C2 = sammy_functions.readlst(os.path.join(sammyRTO_ls.sammy_runDIR,f"results/cap1_step{ifinal2}.lst"))

    # fig = plot_trans_cap(exp_pw, cap_pw, T1=T2, C1=C2)


    # %%
    # import imageio

    # for i in range(1,ifinal):
        
    #     C1 = sammy_functions.readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/cap1_step{i}.lst"))
    #     T1 = sammy_functions.readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/trans1_step{i}.lst"))

    #     fig = plot_trans_cap(exp_pw, cap_pw, T1=T1, C1=C1)
    #     fig.savefig(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/figures/case4b_{i}.png")
        # close()

    # %%

    # images = []
    # for i in range(1,ifinal): #range(start_job,end_job):
    #     images.append(imageio.imread(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/figures/case4b_{i}.png"))
    # imageio.mimsave(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/case_4b.gif", images)


