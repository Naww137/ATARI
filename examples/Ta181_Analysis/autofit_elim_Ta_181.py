# HOW TO RUN
# python3 -u autofit_elim_Ta_181.py | tee test_output.txt

# %%
from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import os
import importlib

from datetime import datetime

from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions, template_creator
from ATARI.sammy_interface.sammy_functions import fill_sammy_ladder

from ATARI.ModelData.particle_pair import Particle_Pair

from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.theory.experimental import e_to_t, t_to_e

from ATARI.utils.misc import fine_egrid

from copy import copy

from ATARI.utils.atario import fill_resonance_ladder

from ATARI.AutoFit import chi2_eliminator_v2
from ATARI.AutoFit import elim_addit_funcs

global_st_time = time.time()

start_date = datetime.fromtimestamp(global_st_time).strftime("%d.%m.%Y %H:%M:%S")
print("Start date and time =", start_date)

# 
current_dir = os.path.dirname(os.path.realpath(__file__))
print(f'Current directory: {current_dir}')

# %%
# all options to run! 
settings = {
    'path_to_SAMMY_exe': '/home/fire/SAMMY/sammy/build/install/bin/sammy',
    'path_to_SAMMY_temps': os.path.join(current_dir, './sammy_temps/'), # relative to current directory!
    'keep_runDIR_SAMMY': True,
    'shell_SAMMY': 'bash',
    'running_path': current_dir
}

# folder where to save all
savefolder = os.path.join(current_dir, 'data_new/')

if not os.path.exists(savefolder):
    os.makedirs(savefolder)
    print(f'Created folder: {savefolder}')

fig_size = (10,6)
showfigures = True # show figures and block script execution

starting_Gn_coeff = 10 # to Gn01

Gn_thr = 0.01
N_red = 50 # number of resonances to keep after the initial autofit

# number of resonances for autofit (without size)
# int( 1.5 *(energy_range_all[1]-energy_range_all[0])) #) # / (Ta_pair.spin_groups['3.0']['Gt01']/1000)

N_res_autofit = 20 # for one spin group
fit_all_spin_groups = True

energy_range_all = [202, 227]

# energy_range_all = [227, 252]
# energy_range_all = [252, 277]
# energy_range_all = [277, 302]
# energy_range_all = [302, 327]

# energy_range_all = [201, 228]
# energy_range_all = [201, 210]

# energy_range_all = [270, 340]

chi2_allowed = 0

start_deep_fit_from = 15 # excluding the side resonances provided

greedy_mode = True # if true - then first resonance that passed the test is deleted,
# otherwise - resonance that adter deletion has best chi2 (fair for both for prior and intermediate tests)

# TODO: define the start_deep_fit_from based on the prob. of having such number of resonances




# %% [markdown]
# ## Measurement Data

# %%
### Determine channel widths

# def get_chw_and_upperE(E, FP):
#     E = np.array(E)
#     tof = e_to_t(E, FP, True)
#     dt = np.diff(tof*1e6)
#     widths1, index1 = np.unique(np.round(dt, 4), return_index=True)
#     chw, Emax = np.flipud(widths1), np.flipud(E[index1])
#     strc = ''
#     stre = ''
#     for c,e in zip(chw, Emax):
#         strc += f"{c*1e3:.2f}, "
#         stre += f"{e:.2f}, "
#     return stre, strc

# # Emax, chw = get_chw_and_upperE(transdat6.E, 100.14)
# # Emax, chw = get_chw_and_upperE(capdat1.E, 45.27)
# # print(Emax)
# # print(chw)



# %%
### 1mm capture data
capdat1 = sammy_functions.readlst(os.path.join(current_dir, "yield_ta1b_unsmooth.dat"))
expcap1 = Experimental_Model(title = "cap1mm",
                                reaction ="capture", 
                                energy_range = energy_range_all,
                                n = (0.005631, 0),
                                FP = (45.27, 0.05),
                                burst= (8.0,1.0), 
                                temp= (294.2610, 0.0),
                                channel_widths={
                                    "maxE": [68.20,  122.68, 330.48, 547.57, 199359.52], 
                                    "chw":  [821.40, 410.70, 102.70, 51.30,  25.70],
                                    "dchw": [0.8,    0.8,    0.8,    0.8,    0.8]
                                }
                               )

capdat1 = capdat1.loc[(capdat1.E<max(expcap1.energy_range)) & (capdat1.E>min(expcap1.energy_range)), :]

### 2mm capture data
capdat2 = sammy_functions.readlst(os.path.join(current_dir, "yield_ta2_unsmooth.dat"))
expcap2 = Experimental_Model(   title = "cap2mm",
                                reaction = "capture", 
                                energy_range = energy_range_all,
                                n = (0.011179, 0.0),
                                FP = (45.27, 0.05),
                                burst = (8.0,1.0),
                                temp = (294.2610, 0.0),
                                channel_widths={
                                    "maxE": [68.20,  122.68, 330.48, 547.57, 199359.52], 
                                    "chw":  [821.40, 410.70, 102.70, 51.30,  25.70],
                                    "dchw": [0.8,    0.8,    0.8,    0.8,    0.8]
                                }
                               )
capdat2 = capdat2.loc[(capdat2.E<max(expcap2.energy_range)) & (capdat2.E>min(expcap2.energy_range)), :]

### 1mm Transmission data
transdat1 = sammy_functions.readlst(os.path.join(current_dir, "trans-Ta-1mm.twenty"))
transdat1_covfile = os.path.join(current_dir, 'trans-Ta-1mm.idc')

# # TODO: ask Noah for what?
# chw, Emax = get_chw_and_upperE(transdat1.E, 100.14)

exptrans1 = Experimental_Model(title = "trans1mm",
                               reaction = "transmission", 
                               energy_range = energy_range_all,

                                n = (0.00566,0.0),  
                                FP = (100.14,0.0), 
                                burst = (8, 0.0), 
                                temp = (294.2610, 0.0),

                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
                                
                               )
transdat1 = transdat1.loc[(transdat1.E<max(exptrans1.energy_range)) & (transdat1.E>min(exptrans1.energy_range)), :]

### 3mm transmission data
transdat3 = sammy_functions.readlst(os.path.join(current_dir, "trans-Ta-3mm.twenty"))
transdat3_covfile = os.path.join(current_dir, "trans-Ta-3mm.idc")

exptrans3 = Experimental_Model(title = "trans3mm",
                               reaction = "transmission", 
                               energy_range = energy_range_all,

                                n = (0.017131,0.0),  
                                FP = (100.14,0.0), 
                                burst = (8, 0.0), 
                                temp = (294.2610, 0.0),

                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
                                
                               )
transdat3 = transdat3.loc[(transdat3.E<max(exptrans3.energy_range)) & (transdat3.E>min(exptrans3.energy_range)), :]


### 6mm transmission data
transdat6 = sammy_functions.readlst(os.path.join(current_dir, "trans-Ta-6mm.twenty"))
transdat6_covfile = os.path.join(current_dir, "trans-Ta-6mm.idc")

exptrans6 = Experimental_Model(title = "trans6mm",
                               reaction = "transmission", 
                               energy_range = energy_range_all,

                                n = (0.03356,0.0),  
                                FP = (100.14,0.0), 
                                burst = (8, 0.0), 
                                temp = (294.2610, 0.0),

                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
                                
                               )
transdat6 = transdat6.loc[(transdat6.E<max(exptrans6.energy_range)) & (transdat6.E>min(exptrans6.energy_range)), :]


### Not using 12mm measurement for evaluation - this is a validation measurement

# transdat12 = sammy_functions.readlst("/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/measurement_data/trans-Ta-12mm.dat")
# # transdat12_covfile = Need to generate from sys and stat covariances
# exptrans12 = Experimental_Model(title = "trans12",
#                                 reaction = "transmission",
#                                 energy_range = erange_all,

#                                 sammy_inputs = {
#                                     'alphanumeric'       :   ["BROADENING IS WANTED"],
#                                     'ResFunc'            :   "ORRES"
#                                         },

#                                 n = (0.067166, 0.0),  
#                                 FP = (35.185,0.0), 
#                                 burst = (8,0.0), 
#                                 temp = (294.2610, 0.0),

#                                 channel_widths={
#                                         "maxE": [270], 
#                                         "chw": [102.7],
#                                         "dchw": [0.8]
#                                         },

#                                 additional_resfunc_lines=["WATER 0004 5.6822000 -0.54425 0.07733000", "WATER      0.5000000  0.05000 0.00700000", "LITHI 000  -1.000000  -1.0000 6.00000000", "LITHI      0.1000000  0.10000 0.60000000", "LITHI      166.87839 -28.7093 1.260690", "LITHI      0.2574580 -0.06871 0.004915"]
#                                )

# transdat12 = transdat12[(transdat12.E<max(exptrans12.energy_range)) & (transdat12.E>min(exptrans12.energy_range))]


# %%
### plotting function
def plot(datasets, experiments, fits=[], priors=[], true=[]):
    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    fig, axes = subplots(2,1, figsize=(8,6), sharex=True)

    for i, exp in enumerate(experiments):
        if exp.reaction == "transmission":
            model_key = "theo_trans"
            iax = 0
        elif exp.reaction == "capture":
            model_key = "theo_xs"
            iax = 1
        else:
            raise ValueError()

        axes[iax].errorbar(datasets[i].E, datasets[i].exp, yerr=datasets[i].exp_unc, zorder=0,
                                                fmt='.', color=colors[i], linewidth=0.5, markersize=1.5, capsize=1, label=exp.title)
        
        if len(fits) != 0:
            axes[iax].plot(fits[i].E, fits[i][model_key], color=colors[i], lw=1.5, label=f'fit {exp.title}')

        if len(priors) != 0:
            axes[iax].plot(priors[i].E, priors[i][model_key], '--', color=colors[i], lw=1.5) #, label=f'prior {exp.title}')
        if len(true) != 0:
            axes[iax].plot(true[i].E, true[i][model_key], '-', color=colors[i], alpha=0.5, lw=1.5) #, label=f'prior {exp.title}')

        
    axes[0].set_ylabel("T")
    axes[1].set_ylabel(r"$Y_{\gamma}$")

    ### make it pretty
    for ax in axes:

        ax.set_xscale('log')
        ax.set_ylim([-0.1,1.1])
        ax.legend()

    fig.supxlabel('Energy (eV)')
    fig.tight_layout()

    return fig



# %%
### setup in zipped lists 
datasets = [capdat1, capdat2, transdat1, transdat3, transdat6]
experiments = [expcap1, expcap2, exptrans1, exptrans3, exptrans6]
covariance_data = [{}, {}, transdat1_covfile, transdat3_covfile, transdat6_covfile]


# important - templates!!!
templates = []
for data, exp in zip(datasets, experiments):
    #filepath = f'template_{exp.title}_edited'
    filepath = os.path.join(current_dir, f'template_{exp.title}_edited')
    exp.template = os.path.realpath(filepath)


### NOISE amount estimation
        # calculation of values specific for a dataset
for index, el in enumerate(experiments):

    print(index)
    print(el.title)

    # exp & unc to estimate for current case
    case_ds_exp = datasets[index].exp
    case_ds_unc = datasets[index].exp_unc

    # calculating all the noise related parameters
    # CoV = elim_addit_funcs.coefficient_of_variation(case_ds_exp)
    mSNR = elim_addit_funcs.mean_signal_to_noise_ratio(case_ds_exp, case_ds_unc)
    mUP = elim_addit_funcs.mean_uncertainty_percentage(case_ds_exp, case_ds_unc)
    mums_R = elim_addit_funcs.uncertainty_to_signal_ratio(case_ds_exp, case_ds_unc)

    SNR_s, mean_SNR = elim_addit_funcs.calc_SNR(case_ds_exp, case_ds_unc)

    print(f'mSNR: {mSNR} dB \t {mean_SNR}')
    print(f'mUP: {mUP} ')
    print(f'mums_R: {mums_R}')


###


fig = plot(datasets, experiments)
fig.tight_layout()
# fig.show(block = True)
if (showfigures):
    show()

# input("Press Enter to continue...")

# %%
## Could also plot covariance here

# %% [markdown]
# ## Fit from ENDF or JEFF

# %%
sammyRTO = sammy_classes.SammyRunTimeOptions(
                            sammyexe=settings['path_to_SAMMY_exe'],
                            options= {"Print"   :   True,
                              "bayes"   :   False,
                              "keep_runDIR"     : True,
                              "sammy_runDIR": elim_addit_funcs.generate_sammy_rundir_uniq_name(path_to_sammy_temps = settings['path_to_SAMMY_temps'])
                              })

matnum = 7328

# endf_file = "/Users/noahwalton/research_local/data/neutrons_ENDF-B-VII.1/n-073_Ta_181.endf"
# endf_parameters = sammy_functions.get_endf_parameters(endf_file, matnum, sammyRTO)

# endf_parameters = endf_parameters[(endf_parameters.E<260) & (endf_parameters.E>190)]
# endf_parameters["varyGn1"] = np.ones(len(endf_parameters))
# endf_parameters["varyGg"] = np.ones(len(endf_parameters))*0
# endf_parameters["varyE"] = np.ones(len(endf_parameters))

jeff_file = "/Users/noahwalton/research_local/data/JEFF33_endf6/73-Ta-181g.jeff33"
jeff_file = os.path.join(current_dir, "73-Ta-181g.jeff33")

jeff_parameters = sammy_functions.get_endf_parameters(jeff_file, matnum, sammyRTO)



# cutting jeff parameters to required energy region, but adding side resonances

# jeff_parameters = jeff_parameters[(jeff_parameters.E<max(energy_range_all)+delta_E) & (jeff_parameters.E>min(energy_range_all)-delta_E)]
sel_jeff_res_in_window, sel_jeff_side_res_df = elim_addit_funcs.get_resonances_from_window(
    input_res_df = jeff_parameters,
    energy_range = energy_range_all,
    N_side_res=1
)

# don't touch Gg?
sel_jeff_res_in_window = elim_addit_funcs.set_varying_fixed_params(ladder_df = sel_jeff_res_in_window,
                                                                   vary_list=[1,0,1])

sel_jeff_side_res_df = elim_addit_funcs.set_varying_fixed_params(ladder_df = sel_jeff_side_res_df,
                                                                 vary_list=[0,1,1])

sel_jeff_parameters = pd.concat([sel_jeff_res_in_window,
                                 sel_jeff_side_res_df])

# sort by E & reindex
sel_jeff_parameters = sel_jeff_parameters.sort_values(by='E').reset_index(drop=True)

print("Selected JEFF parameters")
print(sel_jeff_parameters)

print('Min & max E for ladder:')
print(f'{sel_jeff_parameters.E.min()}..{sel_jeff_parameters.E.max()} eV')
print(f'Selected energy range: {energy_range_all}')

# testing reading from combined ladder
start_ladder_main_vary_params, start_ladder_side_vary_params  = elim_addit_funcs.extract_res_var_params(ladder_df = sel_jeff_parameters,
                                        fixed_side_resonances = sel_jeff_side_res_df)

print('Main resonances, vary params:')
print(start_ladder_main_vary_params)
print('Side resonances, vary params:')
print(start_ladder_side_vary_params)

# %%

Ta_pair = Particle_Pair(isotope="Ta181",
                        formalism="XCT",
                        ac=8.1271,     # scattering radius
                        M=180.948030,  # amu of target nucleus
                        m=1,           # amu of incident neutron
                        I=3.5,         # intrinsic spin, positive parity
                        i=0.5,         # intrinsic spin, positive parity
                        l_max=2)       # highest order l-wave to consider

Ta_pair.add_spin_group(Jpi='3.0',
                       J_ID=1,
                       D_avg=8.79,
                       Gn_avg=46.5,
                       Gn_dof=1,
                       Gg_avg=64.0,
                       Gg_dof=1000)

Ta_pair.add_spin_group(Jpi='4.0',
                       J_ID=2,
                       D_avg=4.99,
                       Gn_avg=35.5,
                       Gn_dof=1,
                       Gg_avg=64.0,
                       Gg_dof=1000)


rto = sammy_classes.SammyRunTimeOptions(
    sammyexe=settings['path_to_SAMMY_exe'],
    options = {"Print"   :   True,
                "bayes"   :   True,
                "keep_runDIR"     : True,
                "sammy_runDIR": elim_addit_funcs.generate_sammy_rundir_uniq_name(path_to_sammy_temps= settings['path_to_SAMMY_temps'])
                })


sammyINPyw = sammy_classes.SammyInputDataYW(
    particle_pair = Ta_pair,
    resonance_ladder = sel_jeff_parameters, #jeff_parameters,  

    datasets= datasets,
    experiments = experiments,
    experimental_covariance = covariance_data,  #[{}, {}, {}, {}, {}], # 
    
    max_steps = 10,
    iterations = 2,
    step_threshold = 0.1,
    autoelim_threshold = None,

    LS = False,
    LevMar = True,
    LevMarV = 2,
    LevMarVd= 5,
    initial_parameter_uncertainty = 0.1
    )

print('Spin group keys:')
print(list(Ta_pair.spin_groups.keys()))

# %%
# size of the dataset - to compare

# dataset size
sum_points = 0 
print(f'Dataset sizes: {len(datasets)}')

for index,el in enumerate(datasets):
    sum_points += el.shape[0]
    print(f'{experiments[index].title} - {el.shape[0]}')

print(f'Num of points: {sum_points}')

# %%
sammyOUT_SFJ = sammy_functions.run_sammy_YW(sammyINPyw, rto)

# %%
# assuming that it's "true" solution



# %%

elim_addit_funcs.printout_chi2(sammyOUT_SFJ, 'JEFF')

# %%

# plotting using modified func

fig2 = elim_addit_funcs.plot_datafits(datasets, experiments, 
    fits = sammyOUT_SFJ.pw_post, 
    fits_chi2 = sammyOUT_SFJ.chi2_post, 
    f_model_name = 'SFJ post',
    
    priors = sammyOUT_SFJ.pw, priors_chi2 = sammyOUT_SFJ.chi2, pr_model_name='SFJ prior',

    true = sammyOUT_SFJ.pw_post, 
    true_chi2 = sammyOUT_SFJ.chi2_post, 
    t_model_name ='SFJ post',
    
    true_pars = Ta_pair.resonance_ladder,
    
    fit_pars = sammyOUT_SFJ.par_post,
    prior_pars = sammyOUT_SFJ.par,
      
    title = 'Models Comparison',
    show_spingroups = False,
    #fig_size = fig_size
    )

f_name_to_save = savefolder+f'SFJ_Fit_Result_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}].png'
fig2.savefig(fname=f_name_to_save)

if (showfigures):
    show()


# %%
# utilizing the autofit from initial FB

from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT

sammy_rto_fit = sammy_classes.SammyRunTimeOptions(
    sammyexe = settings['path_to_SAMMY_exe'],
    options = {"Print"   :   True,
                "bayes"   :   True,
                "keep_runDIR"     : True,
                "sammy_runDIR": elim_addit_funcs.generate_sammy_rundir_uniq_name(path_to_sammy_temps=settings['path_to_SAMMY_temps'])
                }
    )

options = InitialFBOPT(Gn_threshold = Gn_thr,
                       iterations=2,
                       max_steps = 30,
                       step_threshold=0.01,
                       LevMarV0= 0.05,
                       fit_all_spin_groups = fit_all_spin_groups,
                       fit_Gg = True,
                       num_Elam = N_res_autofit,
                       spin_group_keys = ['3.0'],
                       starting_Gn1_multiplier = starting_Gn_coeff,
                       starting_Gg_multiplier = 1.0,
                       external_resonances = True
                       )

autofit_initial = InitialFB(options)

# %%

IFB_start_time = time.time()

outs = autofit_initial.fit(Ta_pair,
                               energy_range_all,
                               datasets,
                               experiments,
                               covariance_data,
                               sammy_rto_fit,
                               external_resonance_ladder = sel_jeff_side_res_df
                               )

IFB_end_time = time.time()

print(f'Fitting from IFB took: {elim_addit_funcs.format_time_2_str(IFB_end_time - IFB_start_time)[1]}')
N_initial_FB = outs.final_internal_resonances.shape[0]

print(f'Initial FB size: {N_initial_FB}')


# %%

# print(outs.sammy_outs_fit_2)


elim_addit_funcs.printout_chi2(outs.sammy_outs_fit_1[0], 'autofit prior')
elim_addit_funcs.printout_chi2(outs.sammy_outs_fit_2[-1], 'autofit posterior')

print('Full Final ladder after autofit')
print(outs.final_resonace_ladder)

# print('Posterior parameters:')
# print(outs.sammy_outs_fit_2[-1].par_post)

print('External resonances:')
print(outs.final_external_resonances)

print('Internal resonances:')
print(outs.final_internal_resonances)

# %%
# saving initial solution & chars
f_name_to_save = f'Autofit_initres{N_initial_FB}_red_{N_red}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}.pkl'

save_obj  = elim_addit_funcs.save_obj_as_pkl(folder_name=savefolder, 
                                             file_name=f_name_to_save,
                                             obj = outs)

# %%
# reading the prefitted data
f_name_to_load = f_name_to_save

outs = elim_addit_funcs.load_obj_from_pkl(folder_name=savefolder, 
                                          pkl_fname=f_name_to_load)

final_fb_output = outs.sammy_outs_fit_2[-1]

print('Starting FB:')
print(outs.sammy_outs_fit_1[0].par)

# %%

# plotting to show pos. of resonances
fig2 = elim_addit_funcs.plot_datafits(datasets, experiments, 
    fits = final_fb_output.pw_post, 
    fits_chi2 = final_fb_output.chi2_post, 
    f_model_name = 'AF post',
    
    priors = outs.sammy_outs_fit_1[0].pw, 
    priors_chi2 = outs.sammy_outs_fit_1[0].chi2, 
    pr_model_name='AF prior', # TODO - what is prior here?

    true = sammyOUT_SFJ.pw_post, 
    true_chi2 = sammyOUT_SFJ.chi2_post, 
    t_model_name ='JEFF post', 

    true_pars = sammyOUT_SFJ.par_post,
    fit_pars = final_fb_output.par_post,
    prior_pars = final_fb_output.par,
      
    title = 'Models Comparison',
    show_spingroups = True,
    #fig_size = fig_size
    )

f_name_to_save = savefolder+f'AF_Result_{N_initial_FB}_red_{N_red}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}.png'
fig2.savefig(fname=f_name_to_save)

#fig2.show()

# # %%
# print(final_fb_output.chi2)
# print(sum(final_fb_output.chi2))
# print(final_fb_output.chi2_post)
# print(sum(final_fb_output.chi2_post))
# print(f'N_res: {final_fb_output.par_post.shape[0]}')

# %% [markdown]
# ***A measure of error***

# %%

energy_grid = fine_egrid(energy = energy_range_all)

df_est, df_theo, resid_matrix, SSE_dict, xs_figure = elim_addit_funcs.calc_all_SSE_gen_XS_plot(
        est_ladder = final_fb_output.par_post,
        theo_ladder = sammyOUT_SFJ.par_post,
        Ta_pair = Ta_pair,
        settings = settings,
        energy_grid = energy_grid,
        reactions_SSE = ['capture', 'elastic'],
        fig_size = fig_size,
        calc_fig = True
)

xs_figure.show()

# f_name_to_save = savefolder+f'xs_AF_Res_{N_initial_FB}_red_{N_red}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}.png'
# xs_figure.savefig(fname=f_name_to_save)


# %% [markdown]
# # Resonance elimination 


start_ladder = outs.final_internal_resonances # internal resonances from initial FB

# start_ladder = fill_sammy_ladder(df = start_ladder,
#                                        particle_pair=Ta_pair,
#                                        vary_parm = False,
#                                        J_ID = None)

print('Start ladder without sides:')
print(start_ladder)

# take from autofit
side_resonances_df = outs.final_external_resonances

print('Side resonances used:')
side_resonances_df



# %%
# if we do not want to wait hours...
N_red = min(N_red, start_ladder.shape[0],) # not limiting


# to reduce processing time
start_ladder = elim_addit_funcs.reduce_ladder(ladder_df = start_ladder,
                             Gn1_threshold = Gn_thr,
                             vary_list = [1,0,1], # do not vary Gg
                             N = N_red,
                             #keep_fixed = True,
                             #fixed_side_resonances = side_resonances_df
                             )

print(f'Start ladder after reduction to {N_red} res.:')
print(start_ladder)


# %%

# defining rto & optins for eliminations

# defining the elim_opts
elim_opts = chi2_eliminator_v2.elim_OPTs(
    chi2_allowed = chi2_allowed,
    stop_at_chi2_thr = False,
    deep_fit_max_iter = 30,
    deep_fit_step_thr = 0.001,
    interm_fit_max_iter = 10,
    interm_fit_step_thr = 0.01,
    start_fudge_for_deep_stage = 0.05,
    greedy_mode = greedy_mode,
    start_deep_fit_from = start_deep_fit_from
)

# %%
elimi = chi2_eliminator_v2.eliminator_by_chi2(rto=sammy_rto_fit,
                                            options = elim_opts,
                                            Ta_pair=Ta_pair,

                                            #provide data
                                            datasets = datasets,
                                            covariance_data  = covariance_data,
                                            experiments = experiments
                                            )


# %%
hist = elimi.eliminate(ladder_df = start_ladder,
                       fixed_resonances_df = side_resonances_df)

# %%
# # true - using JEFF? just for comparison
true_chars = sammyOUT_SFJ

# %%
# just to show how much we aliminated with the given threshold value.
print(f'Eliminated from {hist.ladder_IN.shape[0]} res -> {hist.ladder_OUT.shape[0]}')
print(f'Selection done using chi2 threshold: {elim_opts.chi2_allowed}')
print(f'Total models tested: {len(hist.elimination_history.keys())}')
print(f'N_res for models tested (without sides!): ')
print(hist.elimination_history.keys())
# all chi2
print('Models chi2:')

print()
print(f'Elim took {np.round(hist.elim_tot_time,2)} sec')
print()



# %%
# save history?
fitted_elim_case_data = {
    'energy_range': energy_range_all,
        'datasets' : datasets,
        'covariance_data' : covariance_data,
    'experiments': experiments,
    'true_chars': true_chars, # note, jeff are used as true here
    'Ta_pair': Ta_pair,
    # elim parameters
    'elim_opts': elim_opts,
    'side_res_df': side_resonances_df
}

if (fit_all_spin_groups):
    f_name_to_save = f'{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_allspingr_sdf_{start_deep_fit_from}'
else:
    f_name_to_save = f'{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}'

saved_hist = elim_addit_funcs.save_obj_as_pkl(folder_name=savefolder, file_name=f'hist_{f_name_to_save}.pkl', obj=hist)
saved_data = elim_addit_funcs.save_obj_as_pkl(folder_name=savefolder, file_name=f'dataset_{f_name_to_save}.pkl', obj=fitted_elim_case_data)


# %%
# plot the final selected fit?

prior_level = max(hist.elimination_history.keys())
prior_numres = prior_level # hist.elimination_history[prior_level]['input_ladder'].shape[0]

print(f'Initial ladder, num of res.: {prior_numres}')

min_level_passed_test = prior_level # level - key in the hist..
min_N_res_passed_test = prior_level - 1

levels, N_ress, chi2_s = [], [], []

for level in hist.elimination_history.keys():
    
    numres_all = hist.elimination_history[level]['selected_ladder_chars'].par_post.shape[0]
    numres_wo_sides = level

    pass_test = hist.elimination_history[level]['final_model_passed_test']

    print(f'level {level}, # of resonances: {numres_wo_sides}/{numres_all}, passed the test: {pass_test}')

    if (pass_test and level<min_level_passed_test):
        min_level_passed_test = level

        min_N_res_passed_test = numres_wo_sides
    
    levels.append(level)
    chi2_s.append(np.sum(hist.elimination_history[level]['selected_ladder_chars'].chi2_post))
    N_ress.append(numres_wo_sides)


# plotting    
# differences in chi2 values between 2 models
chi2_diffs = np.diff(chi2_s, prepend=chi2_s[0]) 
fig, (ax1, ax2) = subplots(2, 1, figsize = fig_size, sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# First subplot for the original chi2 values
ax1.plot(N_ress, chi2_s, marker='o')
ax1.axvline(x=min_N_res_passed_test, color='r', linestyle='--')

ax1.set_ylabel('$\chi^2$')
ax1.grid(True)

# changes in chi2

ax2.plot(N_ress, chi2_diffs, marker='o', color='green')
ax2.axvline(x=min_N_res_passed_test, color='r', linestyle='--')
ax2.set_xlabel(r'$N_{res}$')
ax2.set_ylabel('Change in $\chi^2$')
ax2.invert_xaxis()
ax2.grid(True)

tight_layout()
f_name_to_save = f'hist_{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}'
fig.savefig(savefolder+f'{f_name_to_save}.png')
#show()




# %%
# plotting data from history
def show_plot_from_hist(
        datasets,
        experiments,
        true_chars,
        true_pars,

        level_to_compare: int,
        min_level_passed_test: int,
        elim_hist: dict,
        addit_title_str: str = ''
        ):
    
    fits = elim_hist[level_to_compare]['selected_ladder_chars'].pw_post
    fits_chi2 = elim_hist[level_to_compare]['selected_ladder_chars'].chi2_post

    prior_fit = elim_hist[prior_level]['selected_ladder_chars'].pw
    priors_chi2 = elim_hist[prior_level]['selected_ladder_chars'].chi2

    # outfit
    fig = elim_addit_funcs.plot_datafits(datasets, experiments, 
        fits=fits, fits_chi2=fits_chi2, f_model_name=f'AF {N_initial_FB} + el. {N_red}',
        priors = prior_fit, priors_chi2=priors_chi2, pr_model_name=f'AF, red. to {N_red}',
        true=true_chars.pw, 
        t_model_name='JEFF (w/o autofit)',
        true_chi2 = true_chars.chi2,
        true_pars = true_pars,
        fit_pars = elim_hist[level_to_compare]['selected_ladder_chars'].par_post,
        prior_pars = elim_hist[prior_level]['input_ladder'],
        title = f'Fit, Prior & True comparison, model # {level_to_compare}, best selected: {min_level_passed_test} '+addit_title_str,
        show_spingroups=True
        )
    fig.tight_layout()
    return fig

# %%
N_res_to_view = min_N_res_passed_test
level_to_compare = N_res_to_view+1

fig = show_plot_from_hist(datasets = datasets,
                          experiments=experiments,
                          true_chars = true_chars,
                          true_pars = true_chars.par,
                          level_to_compare=level_to_compare,
                          min_level_passed_test= min_level_passed_test,
                          elim_hist = hist.elimination_history,
                          addit_title_str=', $\Delta\chi^2$ = '+str(chi2_allowed)
                          )

fig.savefig(savefolder+f'elim_res_selected_{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}.png')

if (showfigures):
    show()

# # in xs space
# df_est, df_theo, resid_matrix, SSE_dict, xs_figure = elim_addit_funcs.calc_all_SSE_gen_XS_plot(
#         est_ladder = hist.elimination_history[level_to_compare]['selected_ladder_chars'].par_post,
#         theo_ladder = sammyOUT_SFJ.par_post,
#         Ta_pair = Ta_pair,
#         settings = settings,
#         energy_grid=energy_grid,
#         reactions_SSE = ['capture', 'elastic'],
#         fig_size=fig_size,
#         calc_fig=True
# )
# #xs_figure.show()

# f_name_to_save = savefolder+f'xs_elim_res_{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}.png'
# xs_figure.savefig(fname=f_name_to_save)

# %%
# table for analysis of the models - produce chi2

# table_df = elim_addit_funcs.create_solutions_comparison_table_from_hist(hist = hist,
#                                                 Ta_pair = Ta_pair,
#                      datasets = datasets,
#                      experiments = experiments,
#                      covariance_data = covariance_data,
#                      true_chars = true_chars,
#                      settings = settings,
#                      energy_grid_2_compare_on = energy_grid)

# # saving comparison table
# table_df.to_csv(path_or_buf=savefolder+f'comparison_{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}.csv')

# print('Sol. Comparison table:')
# print(table_df)

global_end_time = time.time()
print(f'Entire cycle took {elim_addit_funcs.format_time_2_str(global_end_time-global_st_time)[1]}')

end_date = datetime.fromtimestamp(global_end_time ).strftime("%d.%m.%Y %H:%M:%S")
print("End date and time =", end_date)
