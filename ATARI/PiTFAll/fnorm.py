import numpy as np
import pandas as pd
import os
from datetime import datetime

from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.utils.misc import fine_egrid



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
        if rxn == "capture" and resonance_ladder.empty: ### if no resonance parameters - sammy does not return a column for capture theo xs
            sammyOUT.pw["theo_xs"] = np.zeros(len(sammyOUT.pw))
        df[rxn] = sammyOUT.pw.theo_xs
    
    return df
    

def get_rxns(true_par, est_par,
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
    
    return df_est, df_true
    


def get_rxn_residuals(true_par, est_par,
                        sammy_exe, shell,
                        Ta_pair, 
                        energy_range,
                        temperature, 
                        target_thickness,
                        template, reactions):
    
    df_est, df_true = get_rxns(true_par, est_par,
                            sammy_exe, shell,
                            Ta_pair, 
                            energy_range,
                            temperature, 
                            target_thickness,
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
                        template, reactions,
                        print_bool=False):
    
    # initialize residual matrix dict
    ResidualMatrixDict = {}
    for rxn in reactions:
        ResidualMatrixDict[rxn] = []

    # loop over all cases in est_par and true_par lists
    i = 0 
    for est, true in zip(est_par_list, true_par_list):
        rxn_residuals = get_rxn_residuals(true, est,
                        sammy_exe, shell,
                        Ta_pair, 
                        energy_range,
                        temperature, 
                        target_thickness,
                        template, reactions)
        if print_bool:
            i += 1
            print(f"Completed Job: {i}")

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
        Rf[rxn] = np.linalg.norm(R, ord='fro')/R.size
    # ResidualMatrix_allrxns = np.concatenate([ResidualMatrixDict[rxn] for rxn in reactions])
    # Rf["all"] = np.linalg.norm(ResidualMatrix_allrxns, ord='fro')/ResidualMatrix_allrxns.size

    return Rf