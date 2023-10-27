# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import importlib
import subprocess

from theory.particle_pair import Particle_Pair
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions

import ATARI.utils.io.hdf5 as h5io

from copy import copy
from ATARI.theory.resonance_statistics import make_res_par_avg
from ATARI.PiTFAll import fnorm


# %%
### Setup physics
spin_groups = [ (3.0,1,0) ]
res_par_avg = make_res_par_avg(J_ID=1,
                               D_avg = 8.79, 
                            Gn_avg= 0.658, #0.658, 
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = 1000, 
                            print = False)

ac = 0.81271; M = 180.948030; m = 1; I = 3.5; i = 0.5; l_max = 1 
average_parameters = {'3.0':res_par_avg}
Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={},
                                spin_groups=spin_groups,
                                average_parameters=average_parameters )   


sammy_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'
shell = 'zsh'
template = "dop_2sg.inp"
reactions = ["elastic", "capture"]

case_file = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/data_2.hdf5"

energy_range = [200,250]
temperature = 300
target_thickness = 0.005

result_dict = {}
res_dict = {}
time_dict = {}
# for gn in ["Gnavg", "Gnmin"]:
for gn in ["Gn10min", "Gn1000min"]:
    basepath = f"/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups/{gn}"

    for thresh in ["1en1", "1en2", "1en3", "1en4", "1en5"]:
        for iE in [50,75,100]:
            case = f"{gn}_{thresh}_iE{iE}"
        
            true_par_list = []
            est_par_list = []
            time_list = []
            for isample in range(500):
                true_par = h5io.read_par(case_file, isample, 'true') 

                try:
                    step1df = pd.read_csv(os.path.join(basepath, f"par_i{isample}_iE{iE}.csv"))
                    time_step1 = step1df["time"]

                    par_df = pd.read_csv(os.path.join(basepath, f"step2_{thresh}", f"par_i{isample}_iE{iE}.csv"))
                    time_step2 = par_df["time"]
                    est_par = par_df[["E", "Gg", "Gn1", "J_ID"]]
                    # est_par = sammy_functions.fill_sammy_ladder(est_par, Ta_pair)
                    true_par_list.append(true_par)
                    est_par_list.append(est_par)
                    time_list.append(time_step1+time_step2)

                except:
                    print(f"job_failed: {gn}, {thresh}, {iE}, {isample}")


            Rdict = fnorm.build_residual_matrix_dict(est_par_list, true_par_list,
                                    sammy_exe, shell,
                                    Ta_pair, 
                                    energy_range,
                                    temperature, 
                                    target_thickness,
                                    template, reactions, 
                                    print_bool=False)
            Fnorms = fnorm.calculate_fnorms(Rdict, reactions)

            result_dict[case] = Fnorms
            res_dict[case] = Rdict
            time_dict[case] = time_list

            print(f"Done with case {case}")


# %%
#### Compile reference Fnorms

# reactions = ["elastic", "capture"]


# basepath = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups/ref_fits"
# true_par_list = []
# est_par_list = []
# for isample in range(500):
#     true_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical

#     try:
#         csvfile = os.path.join(basepath, f"par_i{isample}.csv")
#         par_df = pd.read_csv(csvfile)
#         est_par = par_df[["E", "Gg", "Gn1", "J_ID"]]
#         # est_par = sammy_functions.fill_sammy_ladder(est_par, Ta_pair, J_ID=np.ones(len(est_par)))
#         # est_par = est_par[abs(est_par.Gn1>1e-5)]

#         true_par_list.append(true_par)
#         est_par_list.append(est_par)
#     except:
#         print(f"Failed {isample}")


# Rdict_reference = fnorm.build_residual_matrix_dict(est_par_list, true_par_list,
#                         sammy_exe, shell,
#                         Ta_pair, 
#                         energy_range,
#                         temperature, 
#                         target_thickness,
#                         template, reactions, 
#                         print_bool=True)
# Fnorms_reference = fnorm.calculate_fnorms(Rdict_reference, reactions)


# %%

# Fnorms_reference["total"]/Rdict_reference["total"].size
print(Fnorms_reference)
for rxn in reactions:
    R = Rdict_reference[rxn]
    print(np.linalg.norm(R, ord='fro'))

# %%
468.8740689676174**2 + 954.2147611538875**2

# %%

# count_dict = {}
# for gn in ["Gnavg", "Gnmin"]:
#     for ithresh in ["1en2", "1en3", "1en4", "1en5"]:

#         basepath = f"/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/{gn}_fits_{ithresh}"
        
#         counts = [0,0,0,0]
#         for i, iE in enumerate([25,50,75,100]):

#             true_par_list = []
#             est_par_list = []
#             for isample in range(500):
#                 true_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical

#                 try:
#                     csvfile = os.path.join(basepath, f"par_i{isample}_iE{iE}.csv")
#                     par_df = pd.read_csv(csvfile)
#                     est_par = par_df[["E", "Gg", "Gn1"]]
#                     est_par = sammy_functions.fill_sammy_ladder(est_par, Ta_pair, J_ID=np.ones(len(est_par)))
#                     # est_par = est_par[abs(est_par.Gn1>1e-5)]

#                     true_par_list.append(true_par)
#                     est_par_list.append(est_par)
                    
#                     counts[i] += 1
#                 except:
#                     pass

#         case = os.path.basename(basepath)
#         count_dict[case] = counts


# %%
# fnorms_iE = result_dict["Gnavg_fits_1en2"]
# x = [25,50,75, 100]
# print(x)
# print([f["capture"] for f in fnorms_iE])
# print([f["elastic"] for f in fnorms_iE])

# print([f["elastic"] + f["capture"]**2 for f in fnorms_iE])

# %%
### Compile and write dataframe



# metrics = pd.DataFrame()
# for ig, gn in enumerate(["Gn10min", "Gn1000min"]):
#     for i, thresh in enumerate(["1en1","1en2","1en3", "1en4", "1en5"]):
#         metric_v_iE = []
#         for iE in [50,75,100]:
#             label = f"{gn}_{thresh}_iE{iE}"
#             dat = result_dict[label]
#             res = res_dict[label]
#             np.save(f"/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups/res_cap_{label}.npy", res["capture"])
#             np.save(f"/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups/res_scat_{label}.npy", res["elastic"])
#             # print(dat)
#             metric = dat["capture"]**2 + dat["elastic"]**2
#             metric_v_iE.append(metric)
#         metrics[f"{gn}_{thresh}"] = metric_v_iE
# iE = [50,75,100]
# metrics.index = iE
# metrics.index.name = "iE"
# # metrics["refbest"] = Fnorms_reference["capture"]**2 + Fnorms_reference["elastic"]**2
# metrics
# # # metrics.to_csv("/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups/CaseMetrics_fp1.csv")

# %%

metrics = pd.read_csv("/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups/CaseMetrics_fp1.csv")
metrics

# %%

figure()

colors=["C1","C2","C3","C4", "C5"]
markers=['o-', 'o--']

for ig, gn in enumerate(["Gn10min", "Gn1000min"]):
# for gn in ["Gnmin"]:
    for i, ithresh in enumerate(["1en1", "1en2", "1en3", "1en4", "1en5"]):
        label = f"{gn}_{ithresh}"
        metric_v_iE = metrics[label]
        plot(iE, metric_v_iE, markers[ig], color = colors[i])

# reference_metric = Fnorms_reference["elastic"]**2 + Fnorms_reference["capture"]**2 
reference_metric = np.unique(metrics["refbest"].values)
axhline(reference_metric,  label="Reference Best", color='b', lw=3)

xlabel("Energy Grid")
title("Procedure Comparison for initial feature selection\nUsing GD sammy with fudge=0.1")
ylabel(r"$||\sigma_s||_F^2 + ||\sigma_{\gamma}||_F^2$")
yscale('log')


lines = gca().get_lines()
legend1 = legend([lines[i] for i in [0,5,10]], [r"$\Gamma_n=Q1*10$", r"$\Gamma_n=Q1*1000$", "Reference Best"], loc=1)
legend(["1e-1", "1e-2", "1e-3", "1e-4", "1e-5"], loc="lower right")
gca().add_artist(legend1)


# for ig, gn in enumerate(["Gn10min", "Gn1000min"]):
#     for i, ithresh in enumerate(["1en2", "1en5"]):
#             label = f"{gn}_{ithresh}"
#             metric_v_iE = metrics[label]
#             plot(iE, metric_v_iE, label=label)
# legend()

# %%

# figure()

# for rxn in reactions:
#     plot(x, [f[rxn] for f in fnorms_iE], '.-', label=rxn)

# plot(x, [f["all"] for f in fnorms_iE], '.-', label="all rxn")

# xlabel('iE')
# ylabel(r'$||R||_{F}$')
# title("Fnorm 500 samples; sammy GD feature selection\n <Gn>, <Gg>, elim Gn<1e-5 meV")
# legend()

# %%

# iE = 100
# true_par_list = []
# est_par_list = []
# Gn_list = []
# for isample in range(500):
#     true_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical

#     try:
#         csvfile = os.path.join(basepath, f"par_i{isample}_iE{iE}.csv")
#         par_df = pd.read_csv(csvfile)
#         est_par = par_df[["E", "Gg", "Gn1"]]
#         est_par = sammy_functions.fill_sammy_ladder(est_par, Ta_pair, J_ID=np.ones(len(est_par)))

#         true_par_list.append(true_par)
#         est_par_list.append(est_par)
#         Gn_list.extend(list(est_par.Gn1))
#     except:
#         pass

# %% [markdown]
# ### individual case analysis

# %%
# abs(np.sum(Rdict['capture'], axis=1))
gn = "gn1000min"
thresh = "1en2"
iE = 50

path = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/2spingroups"
Rdict = {}
Rdict["capture"] = np.load(os.path.join(path, f"res_cap_{gn}_{thresh}_iE{iE}.npy"))
Rdict["elastic"] = np.load(os.path.join(path, f"res_scat_{gn}_{thresh}_iE{iE}.npy"))


print(np.argmax(abs(np.sum(Rdict['capture'], axis=1))))#.shape
print(np.argmax(abs(np.sum(Rdict['elastic'], axis=1))))#.shape

# %%
### get estimate and true

isample = 280

dataset_titles = ["trans1", "cap1"]
datasets = []
for dt in dataset_titles:
    exp_pw, exp_cov = h5io.read_pw_exp(case_file, isample, title=dt)
    datasets.append(exp_pw)

trans = datasets[0]
cap = datasets[1]

true_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical

csvfile = os.path.join(path, f"{gn}", f"par_i{isample}_iE{iE}.csv")
par_df = pd.read_csv(csvfile)
est_par = par_df[["E", "Gg", "Gn1"]]
est_par = sammy_functions.fill_sammy_ladder(est_par, Ta_pair, J_ID=np.ones(len(est_par)))

par_df

# %%
### plot experimental

sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    model = 'XCT',
    reaction = 'transmission',
    solve_bayes = False,
    inptemplate = "allexptot_2sg.inp",
    )

sammyINP = sammy_classes.SammyInputData(
    particle_pair = Ta_pair,
    resonance_ladder = par_df,
    energy_grid=exp_pw.E.values,
    temp = 304.5,
    FP=75.0,
    frac_res_FP=0.025,
    target_thickness=0.005)

sammyOUT_trans = sammy_functions.run_sammy(sammyINP, sammyRTO)

sammyRTO.reaction = 'capture'
sammyRTO.inptemplate = 'allexpcap_2sg.inp'
sammyOUT_cap = sammy_functions.run_sammy(sammyINP, sammyRTO)

fig = plot_trans_cap(trans, cap, T1=sammyOUT_trans.pw,C1=sammyOUT_cap.pw, plot_true=False)

# %%
### Plot theoretical

est_df, true_df = fnorm.get_rxns(true_par, est_par,
                            sammy_exe, shell,
                            Ta_pair, 
                            exp_pw.E.values,
                            temperature, 
                            target_thickness,
                            template, reactions)
fig, axes = subplots(2,1, figsize=(10,5), sharex=True)

for i in range(2):
    axes[i].plot(est_df.E, est_df[reactions[i]], 'r')
    axes[i].plot(true_df.E, true_df[reactions[i]], 'g')

    axes[i].set_yscale('log')
    axes[i].set_ylabel(reactions[i])

# axes[1].plot(cap.E, cap.exp,'k.')

fig.tight_layout()

# %%


# %%


# %%


# %%


# %%
# import imageio


# def make_fitgif(isample, iE):

#     with open(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/ifinal_nE{iE}.txt", 'r') as f:
#         lines = f.readlines()
#     for line in lines:
#         if line.startswith(f"{isample} "):
#             ifinals = line.split()
    
#     for istep in [1,2,3,4]:
#         folder = f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/sample_{isample}_iE{iE}_step{istep}"
#         for i in range(1,int(ifinals[istep])):
            
#             C1 = sammy_functions.readlst(os.path.join(folder,f"results/cap1_step{i}.lst"))
#             T1 = sammy_functions.readlst(os.path.join(folder,f"results/trans1_step{i}.lst"))
#             fig = plot_trans_cap(exp_pw, cap_pw, T1=T1, C1=C1)
#             fig.suptitle(f"Step: {istep}")

#             fig.savefig(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/figures/sample{isample}_iE{iE}_step{istep}_{i}.png")
#             close()
        
#     images = []
#     for istep in [1,2,3,4]:
#         for i in range(1,int(ifinals[istep])): #range(start_job,end_job):
#             images.append(imageio.imread(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/SAMMY_runDIRs/figures/sample{isample}_iE{iE}_step{istep}_{i}.png"))
#     imageio.mimsave(f"/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/sample{isample}_iE{iE}.gif", images)

# %%
# for istep in [1,2,3,4]:
#     fig = makefig()
#     fig.savefig(f"somewhere/sample{istep}.png")
#     close()
    
# images = []
# for istep in [1,2,3,4]:
#     images.append(imageio.imread(f"somewhere/sample{istep}.png"))
# imageio.mimsave(f"somewhere/my.gif", images)


# %%


# %%
# os.system(os.path.join(sammyRTO.sammy_runDIR, 'run.zsh'))


# %%


# %%


# %%



