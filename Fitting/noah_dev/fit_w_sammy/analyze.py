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
from ATARI.utils.misc import fine_egrid

#%%

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


#%%
from datetime import datetime

def generate_sammy_rundir_uniq_name(path_to_sammy_temps: str, case_id: int = 0, addit_str: str = ''):

    if not os.path.exists(path_to_sammy_temps):
        os.mkdir(path_to_sammy_temps)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    # Combine timestamp and random characters
    unique_string = timestamp

    sammy_rundirname = path_to_sammy_temps+'SAMMY_runDIR_'+addit_str+'_'+str(case_id)+'_'+unique_string+'/'

    return sammy_rundirname


def calc_theo_broad_xs_for_all_reaction(sammy_exe, shell,
                                        particle_pair, 
                                        resonance_ladder, 
                                        energy_range,
                                        temperature,
                                        target_thickness,
                                        template, 
                                        reactions):

    runDIR = generate_sammy_rundir_uniq_name('./')

    sammyRTO = sammy_classes.SammyRunTimeOptions(
        path_to_SAMMY_exe = sammy_exe,
        shell = shell,
        model = 'XCT',
        sammy_runDIR = runDIR,
        inptemplate = template,
        keep_runDIR=False
        )

    E = fine_egrid(energy_range)

    sammyINP = sammy_classes.SammyInputData(
        particle_pair = particle_pair,
        resonance_ladder = resonance_ladder,
        energy_grid = E,
        temp = temperature,
        target_thickness= target_thickness)
    
    df = pd.DataFrame({"E":E})
    for rxn in reactions:
        sammyRTO.reaction = rxn
        sammyOUT = sammy_functions.run_sammy(sammyINP, sammyRTO)
        df[rxn] = sammyOUT.pw.theo_xs
    
    return df
    

def get_rxn_residuals(true_par, est_par,
                        sammy_exe, shell,
                        Ta_pair, 
                        energy_range,
                        temperature, 
                        target_thickness,
                        template, reactions):
    
    df_est = calc_theo_broad_xs_for_all_reaction(sammy_exe, shell,
                                            Ta_pair, 
                                            est_par, 
                                            energy_range,
                                            temperature, target_thickness,
                                            template, reactions)

    df_true = calc_theo_broad_xs_for_all_reaction(sammy_exe, shell,
                                            Ta_pair, 
                                            true_par, 
                                            energy_range,
                                            temperature, target_thickness,
                                            template, reactions)
    E = df_est.E
    assert(np.all(E == df_true.E))
    residuals = df_est-df_true
    residuals["E"] = E

    return residuals

def build_residual_matrix_dict(est_par_list, true_par_list,
                        sammy_exe, shell,
                        Ta_pair, 
                        energy_range,
                        temperature, 
                        target_thickness,
                        template, reactions
                        ):
    
    # initialize residual matrix dict
    ResidualMatrixDict = {}
    for rxn in reactions:
        ResidualMatrixDict[rxn] = []

    # loop over all cases in est_par and true_par lists
    for est, true in zip(est_par_list, true_par_list):
        rxn_residuals = get_rxn_residuals(true, est,
                        sammy_exe, shell,
                        Ta_pair, 
                        energy_range,
                        temperature, 
                        target_thickness,
                        template, reactions)
        # append reaction residual
        for rxn in reactions:
            ResidualMatrixDict[rxn].append(list(rxn_residuals[rxn]))

    # convert to numpy array
    for rxn in reactions:
        ResidualMatrixDict[rxn] = np.array(ResidualMatrixDict[rxn])
    
    return ResidualMatrixDict


def calculate_fnorms(ResidualMatrixDict, reactions):
    Rf = {}
    for rxn in reactions:
        R = ResidualMatrixDict[rxn]
        Rf[rxn] = np.linalg.norm(R, ord='fro')
    ResidualMatrix_allrxns = np.concatenate([ResidualMatrixDict[rxn] for rxn in reactions])
    Rf["all"] = np.linalg.norm(ResidualMatrix_allrxns, ord='fro')

    return Rf

#%%

sammy_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'
shell = 'zsh'
template = "dop_1sg.inp"
reactions = ["total", "capture", "elastic"]

basepath = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/RM_allexp/sammy/Gnavg_fits"
case_file = "/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/data.hdf5"

energy_range = [75,125]
temperature = 300
target_thickness = 0.067166 

fnorm_iE = []

for iE in [25,50,75,100]:

    true_par_list = []
    est_par_list = []
    for isample in range(500):
        true_par = h5io.read_par(case_file, isample, 'true')  #for fine grid theoretical

        # try:
        csvfile = os.path.join(basepath, f"par_i{isample}_iE{iE}.csv")
        par_df = pd.read_csv(csvfile)
        est_par = par_df[["E", "Gg", "Gn1"]]
        est_par = sammy_functions.fill_sammy_ladder(est_par, Ta_pair, J_ID=np.ones(len(est_par)))

        true_par_list.append(true_par)
        est_par_list.append(est_par)
        # except:
        #     pass


    Rdict = build_residual_matrix_dict(est_par_list, true_par_list,
                            sammy_exe, shell,
                            Ta_pair, 
                            energy_range,
                            temperature, 
                            target_thickness,
                            template, reactions)

    Fnorms = calculate_fnorms(Rdict, reactions)

    fnorm_iE.append(Fnorms)

#%%



#%%
# figure()

# plot(df.E, df["total"], 'b')
# plot(df.E, df["capture"], 'r')
# plot(df.E, df["elastic"], 'k')
# yscale('log')
# # show()


# %%