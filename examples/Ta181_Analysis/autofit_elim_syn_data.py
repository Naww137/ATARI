# HOW TO RUN
# python3 -u autofit_elim_syn_data.py | tee test_output.txt

# %%
from matplotlib.pyplot import *
import numpy as np
import pandas as pd
import os
import importlib

from datetime import datetime

from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions, template_creator
# from ATARI.sammy_interface.sammy_functions import fill_sammy_ladder

from ATARI.ModelData.particle_pair import Particle_Pair

from ATARI.ModelData.experimental_model import Experimental_Model

#from ATARI.theory.experimental import e_to_t, t_to_e

from ATARI.utils.misc import fine_egrid

from copy import copy
import argparse

#from ATARI.utils.atario import fill_resonance_ladder

from ATARI.AutoFit import chi2_eliminator_v2
from ATARI.AutoFit import elim_addit_funcs



global_st_time = time.time()



start_date = datetime.fromtimestamp(global_st_time).strftime("%d.%m.%Y %H:%M:%S")
print("Start date and time =", start_date)

#### main settings #### 

# taken from inputs
parser = argparse.ArgumentParser(description="Fitter parameters",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--cases_dir", 
                    type=str, 
                    default = './data_raw/data_1_correct_cap.hdf5',
                    help="file with t, cap, cov...")

parser.add_argument("--case_id", 
                    type=int, 
                    default=0, 
                    help="Case_id to process")

parser.add_argument("--save_to_folder", 
                    type=str, 
                    default='./proc_cases/', 
                    help="Case_id to process")

parser.add_argument("--start_from_true", 
                    type=int, 
                    default=0, 
                    help="start from true solution")

parser.add_argument("--keep_fixed",
                    type = int,
                      default = 1,
                      help="keep fixed side resonances")

args = parser.parse_args()
config = vars(args)

# number of resonances to reduce autofit result usign threshold in Gn (after all stages!)
N_red

greedy

fit_all_spin_groups = True



# generated
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

# folder where to save all results
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

N_res_autofit = 100 # for one spin group
fit_all_spin_groups = True

energy_range_all = [202, 227]

# for elimination

chi2_allowed = 0
start_deep_fit_from = 15 # excluding the side resonances provided
greedy_mode = True

# TODO: define the start_deep_fit_from based on the prob. of having such number of resonances




# %% [markdown]
# ## Measurement Data, loading by case number


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


fig = plot(datasets, experiments)
fig.tight_layout()


if (showfigures):
    show()


# %%
## Could also plot all covariance here

# # testing reading from combined ladder
# start_ladder_main_vary_params, start_ladder_side_vary_params  = elim_addit_funcs.extract_res_var_params(ladder_df = sel_jeff_parameters,
#                                         fixed_side_resonances = sel_jeff_side_res_df)

# print('Main resonances, vary params:')
# print(start_ladder_main_vary_params)
# print('Side resonances, vary params:')
# print(start_ladder_side_vary_params)

from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT

sammy_rto_fit = sammy_classes.SammyRunTimeOptions(
    sammyexe=settings['path_to_SAMMY_exe'],
                             options = {"Print"   :   True,
                              "bayes"   :   True,
                              "keep_runDIR"     : True,
                              "sammy_runDIR": elim_addit_funcs.generate_sammy_rundir_uniq_name(path_to_sammy_temps=settings['path_to_SAMMY_temps'])
                              })

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

#xs_figure.show()

f_name_to_save = savefolder+f'xs_AF_Res_{N_initial_FB}_red_{N_red}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}.png'
xs_figure.savefig(fname=f_name_to_save)


# %% [markdown]
# # Resonance elimination 


start_ladder = outs.final_internal_resonances

print('Start ladder without sides:')
print(start_ladder)

# print('Columns:')
# print(start_ladder.columns)

# side_resonances_df = elim_addit_funcs.find_side_res_df(
#         initial_sol_df = sel_jeff_parameters,
#         energy_region = energy_range_all,
#         N_res = 2
# )

# take from autofit
side_resonances_df = outs.final_external_resonances

# enrich with all required columns
side_resonances_df = elim_addit_funcs.set_varying_fixed_params(ladder_df=side_resonances_df,
                                                               vary_list=[0,1,1])

print('Side resonances:')
print(side_resonances_df)


# compiling to one ladder
start_ladder = pd.concat([side_resonances_df, start_ladder], ignore_index=True)


print()
print('Starting ladder to eliminate from:')
print(start_ladder)
print()
print(start_ladder.columns)



# if we do not want to wait hours...
N_red = min(N_red, start_ladder.shape[0],) # not limiting

print(f'Ladder will be reduced to {N_red} resonances...')

# just to reduce processing time
start_ladder = elim_addit_funcs.reduce_ladder(ladder_df=start_ladder,
                             Gn1_threshold = Gn_thr,
                             vary_list = [1,0,1],
                             N = N_red,
                             keep_fixed = True,
                             fixed_side_resonances = side_resonances_df)

print('Start ladder:')
print(start_ladder)


# %%

# defining rto & inputs
elim_sammyINPyw = sammy_classes.SammyInputDataYW(
    particle_pair = Ta_pair,
    resonance_ladder = start_ladder,

    datasets = datasets,
    experimental_covariance=covariance_data,
    experiments = experiments,

    max_steps = 0,
    iterations = 2,
    step_threshold = 0.01,
    autoelim_threshold = None,

    LS = False,
    LevMar = True,
    LevMarV = 1.5,

    minF = 1e-5,
    maxF = 10,
    initial_parameter_uncertainty = 0.05
    )

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
                                            sammyINPyw = elim_sammyINPyw , 
                                            options = elim_opts)
print()
print('Elimination options used:')
print('*'*40)
print(elim_opts)
print('*'*40)
print()

# %%
hist = elimi.eliminate(ladder_df = start_ladder,
                       fixed_resonances_df = side_resonances_df)

# %%
# # true - using JEFF? just for comparison
true_chars = sammyOUT_SFJ

# %%
# just to show how much we aliminated with the given threshold value.
print(f'Eliminated from {hist.ladder_IN.shape[0]} res -> {hist.ladder_OUT.shape[0]}')
print(f'Elim took {np.round(hist.elim_tot_time,2)} sec')

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
prior_numres = hist.elimination_history[prior_level]['input_ladder'].shape[0]
print(f'Initial ladder, num of res.: {prior_numres}')

min_level_passed_test = prior_level # level - key in the hist..
min_N_res_passed_test = prior_level - 1

levels = []
N_ress = []
chi2_s = []

for level in hist.elimination_history.keys():
        
    numres = hist.elimination_history[level]['selected_ladder_chars'].par_post.shape[0]
    pass_test = hist.elimination_history[level]['final_model_passed_test']

    #print(f'level {level}, # of resonances: {numres},  passed the test: {pass_test}')

    if (pass_test and level<min_level_passed_test):
        min_level_passed_test = level

        min_N_res_passed_test = numres
    
    levels.append(level)
    chi2_s.append(np.sum(hist.elimination_history[level]['selected_ladder_chars'].chi2_post))
    N_ress.append(numres)


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
        true=true_chars.pw, t_model_name='JEFF (w/o autofit)',
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

# in xs space
df_est, df_theo, resid_matrix, SSE_dict, xs_figure = elim_addit_funcs.calc_all_SSE_gen_XS_plot(
        est_ladder = hist.elimination_history[level_to_compare]['selected_ladder_chars'].par_post,
        theo_ladder = sammyOUT_SFJ.par_post,
        Ta_pair = Ta_pair,
        settings = settings,
        energy_grid=energy_grid,
        reactions_SSE = ['capture', 'elastic'],
        fig_size=fig_size,
        calc_fig=True
)
#xs_figure.show()

f_name_to_save = savefolder+f'xs_elim_res_{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}.png'
xs_figure.savefig(fname=f_name_to_save)

# %%
# table for analysis of the models - produce chi2

table_df = elim_addit_funcs.create_solutions_comparison_table_from_hist(hist = hist,
                                                Ta_pair = Ta_pair,
                     datasets = datasets,
                     experiments = experiments,
                     covariance_data = covariance_data,
                     true_chars = true_chars,
                     settings = settings,
                     energy_grid_2_compare_on = energy_grid)

# saving comparison table
table_df.to_csv(path_or_buf=savefolder+f'comparison_{N_initial_FB}_red_{N_red}_greedy_{greedy_mode}_er[{np.min(energy_range_all)}_{np.max(energy_range_all)}]_chi2allowed_{chi2_allowed}_sdf_{start_deep_fit_from}.csv')

print('Sol. Comparison table:')
print(table_df)

global_end_time = time.time()
print(f'Entire cycle took {elim_addit_funcs.format_time_2_str(global_end_time-global_st_time)[1]}')

end_date = datetime.fromtimestamp(global_end_time ).strftime("%d.%m.%Y %H:%M:%S")
print("End date and time =", end_date)
