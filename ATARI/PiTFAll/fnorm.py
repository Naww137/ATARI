import numpy as np
import pandas as pd
import os
from uuid import uuid4

from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.utils.misc import fine_egrid, generate_sammy_rundir_uniq_name
from ATARI.ModelData.experimental_model import Experimental_Model



def generate_sammy_rundir_uniq_name(path_to_sammy_temps: str, case_id: int = 0, addit_str: str = ''):

    if not os.path.exists(path_to_sammy_temps):
        os.mkdir(path_to_sammy_temps)

    # Generating a unique string from uuid:
    unique_string = str(uuid4())

    sammy_rundirname = path_to_sammy_temps+'SAMMY_runDIR_'+addit_str+'_'+str(case_id)+'_'+unique_string+'/'

    return sammy_rundirname


def calc_theo_broad_xs_for_all_reaction(sammy_exe,
                                        particle_pair, 
                                        resonance_ladder, 
                                        energy_range,
                                        temperature,
                                        template, 
                                        reactions,
                                        df_column_key_extension = ''):

    runDIR = generate_sammy_rundir_uniq_name('./')

    sammyRTO = sammy_classes.SammyRunTimeOptions(sammy_exe,
                             **{"Print"   :   True,
                              "bayes"   :   False,
                              "keep_runDIR"     : False,
                              "sammy_runDIR": runDIR
                              })

    E = fine_egrid(energy_range)

    exp_theo = Experimental_Model(title = "theo",
                                      reaction='total',
                                      temp = (temperature,0),
                                      energy_grid = E,
                                      energy_range = energy_range,
                                      template=template
                                      )
    # exp_theo.template = template

    sammyINP = sammy_classes.SammyInputData(
        particle_pair,
        resonance_ladder,
        template=template,
        experiment=exp_theo,
        energy_grid=E
    )
    
    df = pd.DataFrame({"E":E})
    for rxn in reactions:
        sammyINP.experiment.reaction = rxn
        sammyOUT = sammy_functions.run_sammy(sammyINP, sammyRTO)
        if rxn == "capture" and resonance_ladder.empty: ### if no resonance parameters - sammy does not return a column for capture theo xs
            sammyOUT.pw["theo_xs"] = np.zeros(len(sammyOUT.pw))
        df[rxn+df_column_key_extension] = sammyOUT.pw.theo_xs
    
    return df
    

def get_rxns(true_par, est_par,
                sammy_exe,
                Ta_pair, 
                energy_range,
                temperature,
                template, reactions):
    
    df_est = calc_theo_broad_xs_for_all_reaction(sammy_exe, 
                                        Ta_pair, 
                                        est_par, 
                                        energy_range,
                                        temperature, 
                                        template, reactions)

    df_true = calc_theo_broad_xs_for_all_reaction(sammy_exe,
                                            Ta_pair, 
                                            true_par, 
                                            energy_range,
                                            temperature, 
                                            template, reactions)
    
    return df_est, df_true
    


def get_rxn_residuals(true_par, est_par,
                      sammy_exe,
                        Ta_pair, 
                        energy_range,
                        temperature, 
                        template, reactions):
    
    df_est, df_true = get_rxns(true_par, est_par,
                               sammy_exe,
                            Ta_pair, 
                            energy_range,
                            temperature, 
                            template, reactions)
    
    E = df_est.E
    assert(np.all(E == df_true.E))
    residuals = df_est-df_true
    residuals["E"] = E
    relative = (df_est-df_true)/df_true
    relative["E"] = E


    return residuals, relative



def build_residual_matrix_dict(est_par_list, true_par_list,
                               sammy_exe,
                        particle_pair, 
                        energy_range,
                        temperature, 
                        template, reactions,
                        print_bool=False):
    
    # initialize residual matrix dict
    ResidualMatrixDict = {}
    ResidualMatrixDictRel = {}
    for rxn in reactions:
        ResidualMatrixDict[rxn] = []
        ResidualMatrixDictRel[rxn] = []

    # loop over all cases in est_par and true_par lists
    i = 0 
    for est, true in zip(est_par_list, true_par_list):
        rxn_residuals, rxn_relative = get_rxn_residuals(true, est,
                                          sammy_exe,
                        particle_pair, 
                        energy_range,
                        temperature, 
                        template, reactions)
        if print_bool:
            i += 1
            print(f"Completed Job: {i}")

        # append reaction residual
        for rxn in reactions:
            ResidualMatrixDict[rxn].append(list(rxn_residuals[rxn]))
            ResidualMatrixDictRel[rxn].append(list(rxn_relative[rxn]))


    # convert to numpy array
    for rxn in reactions:
        ResidualMatrixDict[rxn] = np.array(ResidualMatrixDict[rxn])
        ResidualMatrixDictRel[rxn] = np.array(ResidualMatrixDictRel[rxn])
    
    return ResidualMatrixDict, ResidualMatrixDictRel




def calculate_fnorms(ResidualMatrixDict, reactions):
    Rf = {}
    for rxn in reactions:
        R = ResidualMatrixDict[rxn]
        F = np.linalg.norm(R, ord='fro')
        Rf[rxn] = F/np.sqrt(R.size)
    ResidualMatrix_allrxns = np.hstack([ResidualMatrixDict[rxn] for rxn in reactions])
    Rf["all"] = np.linalg.norm(ResidualMatrix_allrxns, ord='fro')/np.sqrt(ResidualMatrix_allrxns.size)

    return Rf, ResidualMatrix_allrxns