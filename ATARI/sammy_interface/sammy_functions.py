

import numpy as np
import os
import shutil
import pandas as pd
import subprocess
from copy import copy
from ATARI.utils.stats import chi2_val

from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyOutputData, SammyInputDataYW

from typing import Optional, Union
from ATARI.sammy_interface.sammy_classes import SammyInputData, SammyRunTimeOptions
from ATARI.sammy_interface.sammy_io import *

# module_dirname = os.path.dirname(__file__)

# ################################################ ###############################################
# Workflow
# ################################################ ###############################################


def write_shell_script(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions, use_RPCM=False, use_IDC=False):
    with open(os.path.join(sammy_RTO.sammy_runDIR, 'pipe.sh'), 'w') as f:
        # f.write(f"{sammyexe}<<EOF\n{ds}_{inp_ext}.inp\n{par}\n{ds}.dat\n{cov}\n\nEOF\n")
        f.write(f'{sammy_RTO.path_to_SAMMY_exe}<<EOF\nsammy.inp\nSAMMY.PAR\nsammy.dat\n')

        if sammy_RTO.energy_window is None:
            if use_RPCM:
                f.write("SAMMY.COV\n")
            if use_IDC:
                f.write("sammy.idc\n")
            f.write("\n")

        # energy windowed solves
        elif sammy_INP.experimental_data is not None:
            iter = np.arange(np.floor(np.min(sammy_INP.experimental_data.E)),np.ceil(np.max(sammy_INP.experimental_data.E))+sammy_RTO.energy_window,sammy_RTO.energy_window)
            if len(iter) >= 50:
                raise ValueError("To many energy windows supplied, please solve in less sections")
            string = ''
            for ie in range(len(iter)-1):
                string += f'{int(iter[ie])}. {int(iter[ie+1])}.\n'
            f.write(f' {string}')

        else:
            raise ValueError("An energy window input was provided but no experimental data.")
        
        f.write("\nEOF")

        # if sammy_RTO.bayes:
            # grep and return chi2
        f.write("""
chi2_line=$(grep -i "CUSTOMARY CHI SQUARED = " SAMMY.LPT | tail -n 1)
chi2_string=$(echo "$chi2_line" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
chi2_linen=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" SAMMY.LPT | tail -n 1)
chi2_stringn=$(echo "$chi2_linen" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
echo "$chi2_string $chi2_stringn"
            """)

    os.system(f"chmod +x {os.path.join(sammy_RTO.sammy_runDIR, f'pipe.sh')}")




def runsammy_shellpipe(sammy_RTO: SammyRunTimeOptions, getchi2= True):
    runsammy_process = subprocess.run(
                                ["sh", "-c", f"./pipe.sh"], 
                                cwd=os.path.realpath(sammy_RTO.sammy_runDIR),
                                capture_output=True, text=True, timeout=60*30
                                )
    if 'STOP' in runsammy_process.stderr:
        raise ValueError(f"\n\n===========================\nSAMMY Failed with output:\n\n {runsammy_process.stdout}")
    elif "SAMMY.LPT: No such file or directory" in runsammy_process.stderr:
        raise ValueError(f"No SAMMY.LPT was generated, check executable path bash scripting.")
    
    if getchi2:
        chi2, chi2n = [float(e) for e in runsammy_process.stdout.split('\n')[-2].split()]
    else:
        chi2=None
        chi2n = None

    return chi2, chi2n


    # # run sammy and wait for completion with subprocess
    # runsammy_process = subprocess.run(
    #                                 [f"sh", "-c", f"{sammy_RTO.path_to_SAMMY_exe}<pipe.sh"], 
    #                                 cwd=os.path.realpath(sammy_RTO.sammy_runDIR),
    #                                 capture_output=True
    #                                 )
    # if len(runsammy_process.stderr) > 0:
    #     print(f'SAMMY gave the following warning or error: {runsammy_process.stderr}')

    # return




# ################################################ ###############################################
# Read endf files
# ################################################ ###############################################


def replace_matnum(filepath, matnum):
    with open(filepath, 'r') as f:
        s = f.read()
        s = s.replace("%%%MAT%%%", str(matnum))
    with open(filepath, 'w') as f:
        f.write(s)

def get_endf_parameters(endf_file, matnum, sammyRTO: SammyRunTimeOptions):

    make_runDIR(sammyRTO.sammy_runDIR)
    fill_runDIR_with_templates("readendf.inp", "sammy.inp", sammyRTO.sammy_runDIR)

    shutil.copy(endf_file, os.path.join(sammyRTO.sammy_runDIR, os.path.basename(endf_file)))

    write_estruct_file([10,100,1000], os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    replace_matnum(os.path.join(sammyRTO.sammy_runDIR,'sammy.inp'), matnum)

    with open(os.path.join(sammyRTO.sammy_runDIR, "pipe.sh"), 'w') as f:
        f.write(f"sammy.inp\n{os.path.basename(endf_file)}\nsammy.dat\n\n")

    # _, _ = runsammy_shellpipe(sammyRTO)
    runsammy_process = subprocess.run(
                                    [f"sh", "-c", f"{sammyRTO.path_to_SAMMY_exe}<pipe.sh"], 
                                    cwd=os.path.realpath(sammyRTO.sammy_runDIR),
                                    capture_output=True, timeout=60*10
                                    )

    try:
        resonance_ladder = readpar(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF.PAR"))
    except:
        with open(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF.PAR"), 'r') as f:
            readlines = f.readlines()
        with open(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF_paronly.PAR"), 'w') as f:
            inres = False
            for line in readlines:
                if line.lower().startswith("resonances"):
                    inres = True
                if inres:
                    f.write(line)
        resonance_ladder = readpar(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF_paronly.PAR"))
    
    # could also read endf spin groups here! 

    with open(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF.INP"), 'r') as f:
        s = f.read()
    with open("SAMNDF.INP", 'w') as f:
        f.write(s) 

    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)

    return resonance_ladder










# ################################################ ###############################################
# run sammy with NV or IQ scheme
# ################################################ ###############################################



def get_ECSCM(sammyRTO, sammyINP):
    if sammyINP.ECSCM_experiment is None:
        print("No specific ECSCM experiment class was given, using the same experiment model as used to fit")
        exp = sammyINP.experiment
    else:
        exp = sammyINP.ECSCM_experiment 

    # update_input_files(sammy_INP, sammy_RTO)
    fill_runDIR_with_templates(exp.template, "sammy.inp", sammyRTO.sammy_runDIR)
    assert len(exp.energy_grid) <= 498, "ECSCM_experiment.energy grid > 498"
    energy_grid = np.linspace(min(sammyINP.experimental_data.E), max(sammyINP.experimental_data.E), len(exp.energy_grid)) # if more than 498 datapoints then I need a new reader!
    write_estruct_file(energy_grid, os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    write_saminp(
                filepath   =    os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                bayes       =   sammyRTO.bayes,
                iterations  =   sammyRTO.iterations,
                formalism   =   sammyINP.particle_pair.formalism,
                isotope     =   sammyINP.particle_pair.isotope,
                M           =   sammyINP.particle_pair.M,
                ac          =   sammyINP.particle_pair.ac*10,
                reaction    =   exp.reaction,
                energy_range=   exp.energy_range,
                temp        =   exp.temp,
                FP          =   exp.FP,
                n           =   exp.n,
                alphanumeric=["CROSS SECTION COVARIance matrix is wanted"],
                )
    
    write_shell_script(sammyINP, sammyRTO, use_RPCM=True)
    _, _ = runsammy_shellpipe(sammyRTO, getchi2=False)

    df, cov = read_ECSCM(os.path.join(sammyRTO.sammy_runDIR, "SAMCOV.PUB"))

    return df, cov
    




def execute_sammy(sammy_RTO:SammyRunTimeOptions):
    chi2, chi2n = runsammy_shellpipe(sammy_RTO)
    lst_df = readlst(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.LST'))
    par_df = readpar(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.PAR'))
    return lst_df, par_df, chi2, chi2n



def delta_chi2(lst_df):
    chi2_prior = chi2_val(lst_df.theo_xs, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
    chi2_posterior = chi2_val(lst_df.theo_xs_bayes, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
    return chi2_prior - chi2_posterior


# def recursive_sammy(pw_prior, par_prior, sammy_INP: SammyInputData, sammy_RTO: SammyRunTimeOptions, itter=0):

#     if itter >= sammy_RTO.recursive_opt["iterations"]:
#         return pw_prior, par_prior
    
#     # sammy_INP.resonance_ladder = par_posterior
#     write_sampar(par_prior, sammy_INP.particle_pair, sammy_INP.initial_parameter_uncertainty, os.path.join(sammy_RTO.sammy_runDIR,"SAMMY.PAR"), vary_parm=sammy_RTO.solve_bayes)
#     pw_posterior, par_posterior = execute_sammy(sammy_RTO)
    
#     Dchi2 = delta_chi2(pw_posterior)
#     if sammy_RTO.recursive_opt["print"]:
#         print(Dchi2)

#     if Dchi2 <= sammy_RTO.recursive_opt["threshold"]:
#         return pw_prior, par_prior
    
#     return recursive_sammy(pw_posterior, par_posterior, sammy_INP, sammy_RTO, itter + 1)

def check_inputs(sammyINP: SammyInputData, sammyRTO:SammyRunTimeOptions):
    sammyINP.resonance_ladder = fill_sammy_ladder(sammyINP.resonance_ladder, sammyINP.particle_pair, False)
    if sammyRTO.bayes:
        if np.sum(sammyINP.resonance_ladder[["varyE", "varyGg", "varyGn1"]].values) == 0.0:
            raise ValueError("Bayes is set to True but no varied parameters.")
        if sammyINP.experimental_data is None:
            if sammyINP.energy_grid is not None: 
                raise ValueError("Run Bayes is set to True but no experimental data was supplied (only an energy grid)")
            else: 
                raise ValueError("Run Bayes is set to True but no experimental data was supplied")
    if sammyRTO.get_ECSCM:
        if not sammyRTO.bayes:
            raise ValueError("get_ECSCM is set to True but bayes is False")
    if sammyINP.experimental_data is None:
        if sammyINP.energy_grid is None: 
            raise ValueError("No energy grid was provided")


def run_sammy(sammyINP: SammyInputData, sammyRTO:SammyRunTimeOptions):

    sammyINP.resonance_ladder = copy(sammyINP.resonance_ladder)
    check_inputs(sammyINP, sammyRTO)

    #### setup 
    make_runDIR(sammyRTO.sammy_runDIR)

    if isinstance(sammyINP.experimental_data, pd.DataFrame):
        write_samdat(sammyINP.experimental_data, sammyINP.experimental_covariance, os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    else:
        write_estruct_file(sammyINP.energy_grid, os.path.join(sammyRTO.sammy_runDIR,"sammy.dat"))

    if isinstance(sammyINP.experimental_covariance, dict) and len(sammyINP.experimental_covariance)>0:
        write_idc(os.path.join(sammyRTO.sammy_runDIR, 'sammy.idc'), 
                  sammyINP.experimental_covariance['Jac_sys'],
                  sammyINP.experimental_covariance['Cov_sys'],
                  sammyINP.experimental_covariance['diag_stat'])
        filter_idc(os.path.join(sammyRTO.sammy_runDIR, 'sammy.idc'), sammyINP.experimental_data)
        idc = True
    elif isinstance(sammyINP.experimental_covariance, str):
        shutil.copy(sammyINP.experimental_covariance, os.path.join(sammyRTO.sammy_runDIR, 'sammy.idc'))
        filter_idc(os.path.join(sammyRTO.sammy_runDIR, 'sammy.idc'),sammyINP.experimental_data)
        idc = True
    else:
        if not sammyINP.experimental_covariance:
            idc = False
        else:
            raise ValueError("Unknown type passed to sammyINP.experimental_covariance")
        

    write_sampar(sammyINP.resonance_ladder, 
                 sammyINP.particle_pair, 
                 sammyINP.initial_parameter_uncertainty,
                 os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PAR'))
    fill_runDIR_with_templates(sammyINP.experiment.template, 
                               "sammy.inp", 
                               sammyRTO.sammy_runDIR)
    write_saminp(
                filepath   =    os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                bayes       =   sammyRTO.bayes,
                iterations  =   sammyRTO.iterations,
                formalism   =   sammyINP.particle_pair.formalism,
                isotope     =   sammyINP.particle_pair.isotope,
                M           =   sammyINP.particle_pair.M,
                ac          =   sammyINP.particle_pair.ac*10,
                reaction    =   sammyINP.experiment.reaction,
                energy_range=   sammyINP.experiment.energy_range,
                temp        =   sammyINP.experiment.temp,
                FP          =   sammyINP.experiment.FP,
                n           =   sammyINP.experiment.n,
                use_IDC     =   idc,
                alphanumeric =  sammyINP.alphanumeric
                )
                
    write_shell_script(sammyINP, 
                       sammyRTO, 
                       use_RPCM=False, 
                       use_IDC=idc)

    lst_df, par_df, chi2, chi2n = execute_sammy(sammyRTO)

    sammy_OUT = SammyOutputData(pw=lst_df, 
                        par=sammyINP.resonance_ladder,
                        chi2=[chi2],
                        chi2n=[chi2n])#,
                        # chi2=chi2_val(lst_df.theo_xs, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc)))

    #### need to update for recursive sammy
    # if sammyRTO.recursive == True:
    #     if sammyRTO.solve_bayes == False:
    #         raise ValueError("Cannot run recursive sammy with solve bayes set to false")
    #     lst_df, par_df = recursive_sammy(lst_df, par_df, sammyINP, sammyRTO)

    if sammyRTO.bayes:
        # sammy_OUT.chi2_post = chi2_val(lst_df.theo_xs_bayes, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
        sammy_OUT.pw_post=lst_df
        sammy_OUT.par_post = par_df
        sammy_OUT.chi2_post = chi2
        sammy_OUT.chi2n_post = chi2n
        sammy_OUT.chi2 = None
        sammy_OUT.chi2n = None

        if sammyRTO.get_ECSCM:
            est_df, ecscm = get_ECSCM(sammyRTO, sammyINP)
            sammy_OUT.ECSCM = ecscm
            sammy_OUT.est_df = est_df

    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)

    return sammy_OUT





########################################################## ###############################################
# run sammy with the YW scheme
########################################################## ###############################################


def make_inputs_for_YW(sammyINPYW: SammyInputDataYW, sammyRTO:SammyRunTimeOptions, idc_list: list):

    #### make files for each dataset YW generation
    exp, idc_bool = sammyINPYW.experiments[0], idc_list
    for exp, idc_bool in zip(sammyINPYW.experiments, idc_list):  # fix this !!
        if idc_bool:
            idc_flag = ["USER-SUPPLIED IMPLICIT DATA COVARIANCE MATRIX"]
        else:
            idc_flag = []
        ### make YWY initial
        fill_runDIR_with_templates(exp.template, f"{exp.title}_initial.inp", sammyRTO.sammy_runDIR)
        write_saminp(
                    filepath   =    os.path.join(sammyRTO.sammy_runDIR, f"{exp.title}_initial.inp"),
                    bayes       =   True,
                    iterations  =   sammyRTO.iterations,
                    formalism   =   sammyINPYW.particle_pair.formalism,
                    isotope     =   sammyINPYW.particle_pair.isotope,
                    M           =   sammyINPYW.particle_pair.M,
                    ac          =   sammyINPYW.particle_pair.ac*10,
                    reaction    =   exp.reaction,
                    energy_range=   exp.energy_range,
                    temp        =   exp.temp,
                    FP          =   exp.FP,
                    n           =   exp.n,
                    # use_IDC=idc,
                    alphanumeric=["yw"]+idc_flag
                                    )
        ### make YWY for iterations
        fill_runDIR_with_templates(exp.template, f"{exp.title}_iter.inp", sammyRTO.sammy_runDIR)
        write_saminp(
                filepath   =    os.path.join(sammyRTO.sammy_runDIR, f"{exp.title}_iter.inp"),
                bayes       =   True,
                iterations  =   sammyRTO.iterations,
                formalism   =   sammyINPYW.particle_pair.formalism,
                isotope     =   sammyINPYW.particle_pair.isotope,
                M           =   sammyINPYW.particle_pair.M,
                ac          =   sammyINPYW.particle_pair.ac*10,
                reaction    =   exp.reaction,
                energy_range=   exp.energy_range,
                temp        =   exp.temp,
                FP          =   exp.FP,
                n           =   exp.n,
                # use_IDC=idc,
                alphanumeric=["yw","Use remembered original parameter values"]+idc_flag
                )
        ### make plotting
        fill_runDIR_with_templates(exp.template, f"{exp.title}_plot.inp", sammyRTO.sammy_runDIR)
        write_saminp(
                    filepath   =    os.path.join(sammyRTO.sammy_runDIR, f"{exp.title}_plot.inp"),
                    bayes       =   False,
                    iterations  =   sammyRTO.iterations,
                    formalism   =   sammyINPYW.particle_pair.formalism,
                    isotope     =   sammyINPYW.particle_pair.isotope,
                    M           =   sammyINPYW.particle_pair.M,
                    ac          =   sammyINPYW.particle_pair.ac*10,
                    reaction    =   exp.reaction,
                    energy_range=   exp.energy_range,
                    temp        =   exp.temp,
                    FP          =   exp.FP,
                    n           =   exp.n,
                    # use_IDC=idc,
                    alphanumeric=[]+idc_flag
                    )
    
    ### options for least squares
    if sammyINPYW.LS:
        alphanumeric_LS_opts = ["USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Take baby steps with Least-Squares method"]
    else:
        alphanumeric_LS_opts = []

    ### solve bayes with YWY matrices initial - don't need experimental description
    fill_runDIR_with_templates(exp.template, "solvebayes_initial.inp", sammyRTO.sammy_runDIR) # need to remove resolutin function from template
    remove_resolution_function_from_template(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_initial.inp"))
    write_saminp(
                filepath   =    os.path.join(sammyRTO.sammy_runDIR,"solvebayes_initial.inp"),
                bayes       =   True,
                iterations  =   0,
                formalism   =   sammyINPYW.particle_pair.formalism,
                isotope     =   sammyINPYW.particle_pair.isotope,
                M           =   sammyINPYW.particle_pair.M,
                ac          =   0.0,
                reaction    =   'REACTION',
                energy_range=   (0.0, 0.0),
                temp        =   (0.0, 0.0),
                FP          =   (0.0, 0.0),
                n           =   (0.0, 0.0),
                alphanumeric=["wy", "CHI SQUARED IS WANTED", "Remember original parameter values"]+alphanumeric_LS_opts
                )
    ### solve bayes with YWY matrices iterations - don't need experimental description
    fill_runDIR_with_templates(exp.template, "solvebayes_iter.inp" , sammyRTO.sammy_runDIR)
    remove_resolution_function_from_template(os.path.join(sammyRTO.sammy_runDIR, "solvebayes_iter.inp"))
    write_saminp(
                filepath   =    os.path.join(sammyRTO.sammy_runDIR, "solvebayes_iter.inp"),
                bayes       =   True,
                iterations  =   0,
                formalism   =   sammyINPYW.particle_pair.formalism,
                isotope     =   sammyINPYW.particle_pair.isotope,
                M           =   sammyINPYW.particle_pair.M,
                ac          =   0.0,
                reaction    =   'REACTION',
                energy_range=   (0.0, 0.0),
                temp        =   (0.0, 0.0),
                FP          =   (0.0, 0.0),
                n           =   (0.0, 0.0),       
                alphanumeric=["wy", "CHI SQUARED IS WANTED", "Use remembered original parameter values"]+alphanumeric_LS_opts
                )

def filter_idc(filepath, pw_df):
    minE = np.min(pw_df.E)
    maxE = np.max(pw_df.E)

    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open (filepath, 'w') as f:
        in_partial_derivatives = False
        for line in lines:
            if line.lower().startswith("free-forma"):
                in_partial_derivatives = True
            elif line.lower().startswith("uncertaint") or not line.strip():
                in_partial_derivatives = False
                in_uncertainties = True

            elif in_partial_derivatives:
                E = float(line.split()[0]) 
                if (round(E,5)>=round(minE,5)-1e-5) & (round(E,5)<=round(maxE,5)+1e-5):
                    pass
                else:
                    continue
            
            f.write(line)

from ATARI.sammy_interface.sammy_misc import get_idc_at_theory

def update_idc_to_theory(sammyINP, sammyRTO, resonance_ladder):
    covariance_data_at_theory = get_idc_at_theory(sammyINP, sammyRTO, resonance_ladder)
    for d, exp, cov in zip(sammyINP.datasets, sammyINP.experiments, covariance_data_at_theory):
        if isinstance(cov, dict) and len(cov)>0 and "theory" not in cov.keys():
            write_idc(os.path.join(sammyRTO.sammy_runDIR, f'{exp.title}.idc'), cov['Jac_sys'], cov['Cov_sys'], cov['diag_stat'])
            # filter_idc(os.path.join(rundir, f'{exp.title}.idc'), d)

    return covariance_data_at_theory


def make_data_for_YW(datasets, experiments, rundir, exp_cov):
    if np.all([isinstance(i, pd.DataFrame) for i in datasets]):
        real = True
    else:
        if np.any([isinstance(i,pd.DataFrame) for i in datasets]):
            raise ValueError("It looks like you've mixed dummy energy-grid data and real data")
        real = False

    idc = []
    for d, exp, cov in zip(datasets, experiments, exp_cov):
        if real:
            write_samdat(d, None, os.path.join(rundir,f"{exp.title}.dat"))
            write_estruct_file(d.E, os.path.join(rundir,"dummy.dat"))
            if isinstance(cov, dict) and len(cov)>0:
                write_idc(os.path.join(rundir, f'{exp.title}.idc'), cov['Jac_sys'], cov['Cov_sys'], cov['diag_stat'])
                filter_idc(os.path.join(rundir, f'{exp.title}.idc'), d)
                idc.append(True)
            elif isinstance(cov, str):
                shutil.copy(cov, os.path.join(rundir, f'{exp.title}.idc'))
                filter_idc(os.path.join(rundir, f'{exp.title}.idc'), d)
                idc.append(True)
            else:
                idc.append(False)
        else:
            write_estruct_file(d, os.path.join(rundir,f"{exp.title}.dat"))
            idc.append(False)
            # write_estruct_file(d, os.path.join(rundir,"dummy.dat"))
    return idc

def make_YWY0_bash(dataset_titles, sammyexe, rundir, idc_list, save_lsts = False):
    par = 'results/step$1.par'
    inp_ext = 'initial'
    with open(os.path.join(rundir, "YWY0.sh") , 'w') as f:
        ### Copy final iteration result to step + 1 result
        # f.write(f"\n\n\n\n############## Copy Iteration Result ###########\nplus_one=$(( $1 + 1 ))\nhead -$(($(wc -l < iterate/bayes_iter{iterations}.par) - 1)) iterate/bayes_iter{iterations}.par > results/step$plus_one.par\n\nrm REMORI.PAR\n")
        for i, ds in enumerate(dataset_titles):
            if idc_list[i]: cov=f"{ds}.idc" 
            else: cov=""
            title = f"{ds}_iter0"
            f.write(f"##################################\n# Generate YW for {ds}\n")
            f.write(f"{sammyexe}<<EOF\n{ds}_{inp_ext}.inp\n{par}\n{ds}.dat\n{cov}\n\nEOF\n")
            if save_lsts:
                f.write(f"""cp SAMMY.LST "results/trans1mm_$1.lst"\n""")
            f.write(f"""mv -f SAMMY.LPT "iterate/{title}.lpt" \nmv -f SAMMY.ODF "iterate/{title}.odf" \nmv -f SAMMY.LST "iterate/{title}.lst" \nmv -f SAMMY.YWY "iterate/{title}.ywy" \n""")    
        f.write("################# read chi2 #######################\n#\n")
        for ds in dataset_titles:
            f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED =" iterate/{ds}_iter0.lpt)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n""")
            f.write(f"""ndat_line_{ds}=$(grep -i "Number of experimental data points = " iterate/{ds}_iter0.lpt)\nndat_string_{ds}=$(echo "$ndat_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $chi2_string_{ds}")
        f.write(""""\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $ndat_string_{ds}")
        f.write(""""\n""")
    
    cov=""
    with open(os.path.join(rundir, "BAY0.sh") , 'w') as f:
        dataset_inserts = "\n".join([f"iterate/{ds}_iter0.ywy" for ds in dataset_titles])
        title = f"bayes_iter1"
        f.write(f"#######################################################\n# Run Bayes for {title}\n#\n#######################################################\n")
        f.write(f"{sammyexe}<<eod\nsolvebayes_{inp_ext}.inp\n{par}\ndummy.dat\n{dataset_inserts}\n\n{cov}\n\neod\n")
        f.write(f"""mv -f SAMMY.LPT iterate/{title}.lpt \nmv -f SAMMY.PAR iterate/{title}.par \nmv -f SAMMY.COV iterate/{title}.cov \nrm -f SAM*\n""")


def make_YWYiter_bash(dataset_titles, sammyexe, rundir, idc_list):
    cov = f"iterate/bayes_iter$1.cov"
    par = f"iterate/bayes_iter$1.par"
    inp_ext = 'iter'
    with open(os.path.join(rundir,"YWYiter.sh"), 'w') as f:
        for i, ds in enumerate(dataset_titles):
            if idc_list[i]: dcov=f"{ds}.idc" 
            else: dcov=""
            title = f"{ds}_iter$1"
            f.write(f"##################################\n# Generate YW for {ds}\n")
            f.write(f"{sammyexe}<<EOF\n{ds}_{inp_ext}.inp\n{par}\n{ds}.dat\n{cov}\n{dcov}\n\nEOF\n")
            f.write(f"""mv -f SAMMY.LPT "iterate/{title}.lpt" \nmv -f SAMMY.ODF "iterate/{title}.odf" \nmv -f SAMMY.LST "iterate/{title}.lst" \nmv -f SAMMY.YWY "iterate/{title}.ywy" \n""")    
        f.write("################# read chi2 #######################\n#\n")
        for ds in dataset_titles:
            f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED =" iterate/{ds}_iter$1.lpt)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n""")
            f.write(f"""ndat_line_{ds}=$(grep -i "Number of experimental data points = " iterate/{ds}_iter$1.lpt)\nndat_string_{ds}=$(echo "$ndat_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $chi2_string_{ds}")
        f.write(""""\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $ndat_string_{ds}")
        f.write(""""\n""")

    with open(os.path.join(rundir,"BAYiter.sh"), 'w') as f:
        dataset_inserts = "\n".join([f"iterate/{ds}_iter$1.ywy" for ds in dataset_titles])
        title = f"bayes_iter$plus_one"
        f.write(f"#######################################################\n# Run Bayes for {title}\n#\n#######################################################\n")
        f.write(f"{sammyexe}<<eod\nsolvebayes_{inp_ext}.inp\n{par}\ndummy.dat\n{dataset_inserts}\n\n{cov}\n\neod\n")
        f.write("plus_one=$(( $1 + 1 ))\n")
        f.write(f"""mv -f SAMMY.LPT iterate/{title}.lpt \nmv -f SAMMY.PAR iterate/{title}.par \nmv -f SAMMY.COV iterate/{title}.cov \nrm -f SAM*\n""")


def make_final_plot_bash(dataset_titles, sammyexe, rundir, idc_list):
    with open(os.path.join(rundir, "plot.sh") , 'w') as f:
        for i,ds in enumerate(dataset_titles):
            if idc_list[i]: dcov=f"{ds}.idc" 
            else: dcov=""
            f.write(f"##################################\n# Plot for {ds}\n")
            f.write(f"{sammyexe}<<EOF\n{ds}_plot.inp\nresults/step$1.par\n{ds}.dat\n{dcov}\n\nEOF\n")
            f.write(f"""mv -f SAMMY.LPT "results/{ds}.lpt" \nmv -f SAMMY.ODF "results/{ds}.odf" \nmv -f SAMMY.LST "results/{ds}.lst" \n\n""")    
        f.write("################# read chi2 #######################\n#\n")
        for ds in dataset_titles:
            f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED =" results/{ds}.lpt | tail -n 1)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
            f.write(f"""chi2n_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/{ds}.lpt | tail -n 1)\nchi2n_string_{ds}=$(echo "$chi2n_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $chi2_string_{ds}")
        for ds in dataset_titles:
            f.write(f" $chi2n_string_{ds}")
        f.write(""""\n""")
    

def setup_YW_scheme(sammyRTO, sammyINPyw): 

    try:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    except:
        pass

    os.mkdir(sammyRTO.sammy_runDIR)
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "results"))
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "iterate"))

    idc_list = make_data_for_YW(sammyINPyw.datasets, sammyINPyw.experiments, sammyRTO.sammy_runDIR, sammyINPyw.experimental_covariance)
    write_sampar(sammyINPyw.resonance_ladder, sammyINPyw.particle_pair, sammyINPyw.initial_parameter_uncertainty, os.path.join(sammyRTO.sammy_runDIR, "results/step0.par"))

    make_inputs_for_YW(sammyINPyw, sammyRTO, idc_list)
    dataset_titles = [exp.title for exp in sammyINPyw.experiments]
    make_YWY0_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR, idc_list, save_lsts=sammyRTO.save_lsts_YW_steps)
    make_YWYiter_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR, idc_list)
    make_final_plot_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR, idc_list)

    return idc_list
    



def iterate_for_nonlin_and_update_step_par(iterations, step, rundir, lead="", print_bool=False):
    runsammy_bay0 = subprocess.run(
                            ["sh", "-c", f"./BAY0.sh {step}"], cwd=os.path.realpath(rundir),
                            capture_output=True, timeout=60*10
                            )
    chi2_old = np.inf
    for i in range(1, iterations+1):
        runsammy_ywyiter = subprocess.run(
                                    ["sh", "-c", f"./YWYiter.sh {i}"], cwd=os.path.realpath(rundir),
                                    capture_output=True, text=True, timeout=60*10
                                    )

        runsammy_bayiter = subprocess.run(
                                    ["sh", "-c", f"./BAYiter.sh {i}"], cwd=os.path.realpath(rundir),
                                    capture_output=True, timeout=60*10
                                    )
        
        i_chi2s = [float(s) for s in runsammy_ywyiter.stdout.split('\n')[-3].split()]
        chi2 = np.sum(i_chi2s[1:])
        if np.isclose(chi2, chi2_old):
            if print_bool: print(f"{lead}\tConverged step at {i} iterations")
            converged = True
            break   
        else:
            chi2_old = chi2
            converged = False

    if i == iterations and not converged:
        if print_bool: print(f"\t{lead}Step did not converge in {i} iterations")
        converged=False
    else:
        converged=True

    # Move par file from final iteration 
    out = subprocess.run(
        ["sh", "-c", 
        f"""head -$(($(wc -l < iterate/bayes_iter{i+1}.par) - 1)) iterate/bayes_iter{i+1}.par > results/step{step+1}.par"""],
        cwd=os.path.realpath(rundir), capture_output=True, timeout=60*1)
    
    return converged, i

def run_YWY0_and_get_chi2(sammyINP, sammyRTO, step):

    if sammyINP.idc_at_theory:
        resonance_ladder = readpar(os.path.join(sammyRTO.sammy_runDIR, f"results/step{step}.par"))
        _ = update_idc_to_theory(sammyINP, sammyRTO, resonance_ladder)
    
    runsammy_ywy0 = subprocess.run(
                                ["sh", "-c", f"./YWY0.sh {step}"], 
                                cwd=os.path.realpath(sammyRTO.sammy_runDIR),
                                capture_output=True, text=True, timeout=60*10
                                )
    i_ndats = [float(s) for s in runsammy_ywy0.stdout.split('\n')[-2].split()]
    i=i_ndats[0]; ndats=i_ndats[1:]
    i_chi2s = [float(s) for s in runsammy_ywy0.stdout.split('\n')[-3].split()]
    i=i_chi2s[0]; chi2s=i_chi2s[1:] 
    if len(chi2s) != 5:
        _ =0
    return i, [c for c in chi2s]+[np.sum(chi2s), np.sum(chi2s)/np.sum(ndats)]


def update_fudge_in_parfile(rundir, step, fudge):
    out = subprocess.run(
        ["sh", "-c", 
        f"""head -$(($(wc -l < results/step{step}.par) - 1)) results/step{step}.par > results/temp;
echo "{np.round(fudge,11)}" >> "results/temp"
mv "results/temp" "results/step{step}.par" """],
        cwd=os.path.realpath(rundir), capture_output=True, timeout=60*1)





### additional functions withing step until convergence
    
def reduce_width_randomly(rundir, istep, sammyINPyw, fudge):
    # if True:# sammyINPyw.batch_reduce_width:
    parfile = os.path.join(rundir,'results',f'step{istep}.par')
    df = readpar(parfile)
    proportion = 0.
    factor = np.random.rand(len(df))
    factor[factor <= proportion] = 0.75
    factor[factor != 1] = 1
    # df['Gg'] = df['Gg']*factor
    df['Gn1'] = df['Gn1']*factor
    write_sampar(df, sammyINPyw.particle_pair, fudge, parfile)#, vary_parm=False, template=None)


def get_batch_vector(vector_length, num_ones, ibatch):
    vector = np.zeros(vector_length)
    indices = np.arange(0, vector_length, int(vector_length/num_ones))
    vector[indices] = 1

    if ibatch%vector_length == 0:
        return vector
    else:
        return np.roll(vector, ibatch)
    

### temporary functions here
def separate_external_resonance_ladder(resonance_ladder, external_resonance_indices):
    if external_resonance_indices is None: external_resonance_indices = []
    external_resonance_ladder = resonance_ladder.iloc[external_resonance_indices, :]
    internal_resonance_ladder = copy(resonance_ladder)
    internal_resonance_ladder.drop(index=external_resonance_indices, inplace=True)
    return internal_resonance_ladder, external_resonance_ladder
def concat_external_resonance_ladder(internal_resonance_ladder, external_resonance_ladder):
    if external_resonance_ladder.empty:
        resonance_ladder = internal_resonance_ladder
        external_resonance_indices = []
    else:
        resonance_ladder = pd.concat([external_resonance_ladder, internal_resonance_ladder], join='inner', ignore_index=True)
        external_resonance_indices = list(range(len(external_resonance_ladder)))
    return resonance_ladder, external_resonance_indices


def batch_fitpar(rundir, istep, sammyINPyw, fudge):

    parfile = os.path.join(rundir,'results',f'step{istep}.par')
    df = readpar(parfile)
    df_internal, df_external = separate_external_resonance_ladder(df, sammyINPyw.external_resonance_indices)
    varyE = np.any(df_internal['varyE']==1)
    varyGg = np.any(df_internal['varyGg']==1)
    varyGn1 = np.any(df_internal['varyGn1']==1)

    numpar = len(df_internal)
    if sammyINPyw.batch_fitpar_random:
        proportion_ones = sammyINPyw.batch_fitpar_ifit/numpar
        vary = np.random.rand(numpar)
        vary[vary <= proportion_ones] = 1
        vary[vary != 1] = 0
    else:
        vary = get_batch_vector(numpar, sammyINPyw.batch_fitpar_ifit, int(istep/sammyINPyw.steps_per_batch))

    df_internal['varyE'] = vary*varyE
    df_internal['varyGg'] = vary*varyGg
    df_internal['varyGn1'] = vary*varyGn1
    df, _ = concat_external_resonance_ladder(df_internal, df_external)
    write_sampar(df, sammyINPyw.particle_pair, fudge, parfile)#, vary_parm=False, template=None)    


def step_until_convergence_YW(sammyRTO, sammyINPyw):
    ### some convenient definitions
    istep = 0
    no_improvement_tracker = 0
    chi2_log = []
    fudge = sammyINPyw.initial_parameter_uncertainty
    par_df_current = sammyINPyw.resonance_ladder
    rundir = os.path.realpath(sammyRTO.sammy_runDIR)
    criteria="max steps"
    total_derivative_evaluations = 0
    ### start loop
    if sammyRTO.Print:
        print(f"Stepping until convergence\nchi2 values\nstep fudge: {[exp.title for exp in sammyINPyw.experiments]+['sum', 'sum/ndat']}")
    while istep<sammyINPyw.max_steps:
        
        ### options to batch parameters being fit
        if sammyINPyw.batch_fitpar:
            if istep%sammyINPyw.steps_per_batch == 0:
                batch_fitpar(rundir, istep, sammyINPyw, fudge)
            else:
                pass
        
        ### get initial YW matrices for iteration and chi2 for current parameters
        i, chi2_list = run_YWY0_and_get_chi2(sammyINPyw, sammyRTO, istep)

        ### after step 1, do convergence checking
        if istep>=1:
            # par_df_current = par_df_next

            ### Levenberg-Marquardt algorithm to update fudge for upcoming step based on chi2 of current parameters
            if sammyINPyw.LevMar:
                assert(sammyINPyw.LevMarV>1)

                if chi2_list[-1] < chi2_log[istep-1][-1]:
                    fudge *= sammyINPyw.LevMarV
                    fudge = min(fudge,sammyINPyw.maxF)
                    update_fudge_in_parfile(rundir, istep, fudge)
                else:
                    if sammyRTO.Print:
                        print(f"Repeat step {int(i)}, \tfudge: {[exp.title for exp in sammyINPyw.experiments]+['sum', 'sum/ndat']}")
                        print(f"\t\t{np.round(float(fudge),3):<5}: {np.round(chi2_list,4)}")

                    while True:  
                        fudge /= sammyINPyw.LevMarVd
                        fudge = max(fudge, sammyINPyw.minF)
                        update_fudge_in_parfile(rundir, istep-1, fudge) # could do batch fitting in here too, before after update fudge and before iterate
                        converged, iterations = iterate_for_nonlin_and_update_step_par(sammyINPyw.iterations, istep-1, rundir, lead="\t", print_bool= sammyRTO.Print)
                        total_derivative_evaluations += iterations
                        # if not converged: # reduce fudge if not converged after iterations regardless if chi2 improves
                        #     fudge /= sammyINPyw.LevMarVd
                        #     fudge = max(fudge, sammyINPyw.minF)
                        i, chi2_list = run_YWY0_and_get_chi2(sammyINPyw, sammyRTO, istep)

                        if sammyRTO.Print:
                            print(f"\t\t{np.round(float(fudge),3):<5}: {np.round(chi2_list,4)}")

                        if chi2_list[-1] < chi2_log[istep-1][-1] or fudge==sammyINPyw.minF:
                            break
                        else:
                            pass
            
            ### convergence check
            Dchi2 = chi2_log[istep-1][-1] - chi2_list[-1]
            if Dchi2 < sammyINPyw.step_threshold:

                no_improvement_tracker += 1
                if no_improvement_tracker >= sammyINPyw.step_threshold_lag:
                    
                    if Dchi2 < 0:
                        criteria = f"Chi2 increased, taking solution {istep-1}"
                        if sammyINPyw.LevMar and fudge==sammyINPyw.minF:
                            criteria = f"Fudge below minimum value, taking solution {istep-1}"
                        if sammyRTO.Print:
                            print(f"{int(i)}    {np.round(float(fudge),3):<5}: {np.round(chi2_list,4)}")
                            print(criteria)
                        return max(istep-1, 0), total_derivative_evaluations
                    else:
                        criteria = "Chi2 improvement below threshold"
                    if sammyRTO.Print:
                        print(f"{int(i)}    {fudge:<5.3f}: {np.round(chi2_list,4)}")
                        print(criteria)
                    return istep, total_derivative_evaluations
                
                else:
                    pass

            else:   
                no_improvement_tracker = 0
            

        ### Log chi2
        chi2_log.append(chi2_list)
        if sammyRTO.Print:
            print(f"{int(i)}    {np.round(float(fudge),3):<5}: {np.round(chi2_list,4)}")
        
        ### Solve Bayes for this step
        # update_fudge_in_parfile(rundir, istep, fudge)  !!! might not need this - put above in > if chi2_list[-1] < chi2_log[istep-1][-1]:
        # if True: reduce_width_randomly(rundir, istep, sammyINPyw, fudge)
        converged, iterations = iterate_for_nonlin_and_update_step_par(sammyINPyw.iterations, istep, rundir, print_bool= sammyRTO.Print)
        total_derivative_evaluations += iterations
        # if not converged: # reduce fudge if not converged after iterations regardless if chi2 improves
        #     fudge /= sammyINPyw.LevMarVd
        #     fudge = max(fudge, sammyINPyw.minF)

        ### update step
        istep += 1

    if sammyRTO.Print: print("Maximum steps reached")
    return max(istep, 0), total_derivative_evaluations


# from ATARI.utils.datacontainers import Evaluation_Data
# from copy import deepcopy

# def get_batched_evaluation_data_list(batches, experimental_titles, datasets, experiments, covariance_data):
#     batch_titles_all_exp = []
#     batch_experiments_all_exp = []
#     batch_datasets_all_exp = []
#     batch_covdat_all_exp = []

#     for title, dat, exp, cov_dat in zip(experimental_titles, datasets, experiments, covariance_data):
    
#         N = len(dat)
#         remainder = N%batches
#         N_per_batch = int((N-remainder)/batches)
#         rand_int = np.random.choice(N, size=N, replace=False) 
        
#         batch_titles = []
#         batch_experiments = []
#         batch_datasets = []
#         batch_covdat = []

#         for i in range(batches):
#             if i == batches - 1:
#                 end = (i+1)*N_per_batch + remainder
#             else:
#                 end = (i+1)*N_per_batch

#             index = rand_int[i*N_per_batch:end]
#             batch_dat  = dat.iloc[index, :].sort_values("E")
#             if cov_dat:
#                 stat = cov_dat['diag_stat'].iloc[index, :].sort_index()
#                 Jac  = cov_dat['Jac_sys'].iloc[:, index].T.sort_index().T
#                 cov = {'Jac_sys':Jac, 'diag_stat': stat, 'Cov_sys': cov_dat["Cov_sys"]}
#             else:
#                 cov = {}

#             batch_titles.append(title)
#             batch_experiments.append(exp)
#             batch_datasets.append(batch_dat)
#             batch_covdat.append(cov)

#         batch_titles_all_exp.append(batch_titles)
#         batch_experiments_all_exp.append(batch_experiments)
#         batch_datasets_all_exp.append(batch_datasets)
#         batch_covdat_all_exp.append(batch_covdat)


#     evaluation_data_for_each_batch = []
#     for i in range(batches):
#         titles=[]
#         exps = []
#         dats = []
#         covdats = []
#         for j in range(len(experimental_titles)):
#             titles.append(batch_titles_all_exp[j][i])
#             exps.append(batch_experiments_all_exp[j][i])
#             dats.append(batch_datasets_all_exp[j][i])
#             covdats.append(batch_covdat_all_exp[j][i])
#         evaluation_data_for_each_batch.append(Evaluation_Data(titles, exps, dats, covdats))

#     return evaluation_data_for_each_batch


# def perform_minibatch_fits(max_steps, batches, sammyINPyw, sammyRTO, dataset_titles):
#     epoch_samouts = []
#     batch_samouts = []
#     epochs = max_steps
#     sammyINPyw_copy = deepcopy(sammyINPyw)
#     sammyRTO_copy = deepcopy(sammyRTO)
#     sammyINPyw_copy.minibatch = False

#     epoch_chi2 = np.inf
#     e = 0
#     while True:

#         evaluation_data_for_each_batch = get_batched_evaluation_data_list(batches, dataset_titles, sammyINPyw.datasets, sammyINPyw.experiments, sammyINPyw.experimental_covariance)
#         for b in range(batches):
#             print(f"-----------\nBatch {b}\n-----------")
#             sammyINPyw_copy.datasets= evaluation_data_for_each_batch[b].datasets
#             sammyINPyw_copy.experiments = evaluation_data_for_each_batch[b].experimental_models
#             sammyINPyw_copy.experimental_covariance= evaluation_data_for_each_batch[b].covariance_data
#             samout = run_sammy_YW(sammyINPyw_copy, sammyRTO_copy)
#             sammyINPyw_copy.resonance_ladder = copy(samout.par_post)
#             batch_samouts.append(samout)

#         ## epoch solve on all data
#         print(f"===========\nEpoch {e}\n=============")
#         sammyINPyw_copy.datasets= sammyINPyw.datasets
#         sammyINPyw_copy.experiments = sammyINPyw.experiments
#         sammyINPyw_copy.experimental_covariance= deepcopy(sammyINPyw.experimental_covariance)
#         samout = run_sammy_YW(sammyINPyw_copy, sammyRTO_copy)
#         sammyINPyw.resonance_ladder = copy(samout.par_post)
#         epoch_samouts.append(samout)

#         epoch_chi2_new = np.sum(samout.chi2_post)
#         if epoch_chi2-epoch_chi2_new < sammyINPyw.step_threshold or e > epochs:
#             if epoch_chi2-epoch_chi2_new < 0:
#                 print(f"Taking previous epoch result")
#                 ibest = -2
#             else:
#                 ibest = -1
#                 print(f"Taking current epoch result")
#             break
#         else:
#             epoch_chi2 = epoch_chi2_new
#             e += 1

#     return epoch_samouts[ibest]



def plot_YW(sammyINP, sammyRTO, dataset_titles, i):
    par = readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{i}.par"))

    if sammyINP.idc_at_theory:
        covariance_data_at_theory = update_idc_to_theory(sammyINP, sammyRTO, par)
    else:
        covariance_data_at_theory = None

    out = subprocess.run(["sh", "-c", f"./plot.sh {i}"], 
                cwd=os.path.realpath(sammyRTO.sammy_runDIR), capture_output=True, text=True, timeout=60*10
                        )
    lsts = []
    for dt in dataset_titles:
        lsts.append(readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/{dt}.lst")) )
    i_chi2s = [float(s) for s in out.stdout.split('\n')[-2].split()]
    i=i_chi2s[0]
    chi2s=i_chi2s[1:len(dataset_titles)+1]
    chi2ns=i_chi2s[len(dataset_titles)+1:] 
    return par, lsts, chi2s, chi2ns, covariance_data_at_theory


def check_inputs_YW(sammyINPyw, sammyRTO):
    empty = False
    dataset_titles = [exp.title for exp in sammyINPyw.experiments]
    if len(np.unique(dataset_titles)) != len(dataset_titles):
        raise ValueError("Redundant experiment model titles")
    if np.any([exp.template is None for exp in sammyINPyw.experiments]):
        raise ValueError(f"One or more experiments do not have template files.")
    if sammyINPyw.step_threshold_lag > 1 and sammyINPyw.batch_fitpar is False:
        print("WARNING: you have set a step threshold lag but are not batching fit parameters, this is an odd setting.")
    if sammyINPyw.resonance_ladder.empty:
        empty = True
        sammyINPyw.resonance_ladder = pd.DataFrame({"E": [1.0], "Gg":[0.000001], "Gn":[0.000001],"varyE":[1], "varyGg":[0], "varyGn1":[0], "J_ID":[1]})
        sammyINPyw.max_steps = 1
    if sammyRTO.bayes:
        if not np.any([each in sammyINPyw.resonance_ladder for each in ["varyE", "varyGg", "varyGn1"]]):
            raise ValueError("No vary flag columns in resonance ladder")
        if np.sum([sammyINPyw.resonance_ladder["varyE"], sammyINPyw.resonance_ladder["varyGg"], sammyINPyw.resonance_ladder["varyGn1"]]) == 0:
            raise ValueError("Bayes set to true but no varied parameters")

    return dataset_titles, empty

def run_sammy_YW(sammyINPyw, sammyRTO):

    ## need to update functions to just pull titles and reactions from sammyINPyw.experiments
    dataset_titles, empty = check_inputs_YW(sammyINPyw, sammyRTO)

    setup_YW_scheme(sammyRTO, sammyINPyw)
    for bash in ["YWY0.sh", "YWYiter.sh", "BAY0.sh", "BAYiter.sh", "plot.sh"]:
        os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'{bash}')}")

    ### get prior
    par, lsts, chi2list, chi2nlist, covdat_at_theory = plot_YW(sammyINPyw, sammyRTO, dataset_titles, 0)
    sammy_OUT = SammyOutputData(pw=lsts, par=par, chi2=chi2list, chi2n=chi2nlist, covariance_data_at_theory=covdat_at_theory)
    
    ### run bayes
    if sammyRTO.bayes:
        if sammyINPyw.minibatch:
            sammy_OUT = perform_minibatch_fits(sammyINPyw.max_steps, sammyINPyw.minibatches, sammyINPyw, sammyRTO, dataset_titles)
            sammy_OUT.pw=lsts
            sammy_OUT.par=par
            sammy_OUT.chi2=chi2list
            sammy_OUT.chi2n=chi2nlist
        else:
            ifinal, total_derivative_evaluations = step_until_convergence_YW(sammyRTO, sammyINPyw)
            par_post, lsts_post, chi2list_post, chi2nlist_post, covdat_at_theory_post = plot_YW(sammyINPyw, sammyRTO, dataset_titles, ifinal)
            sammy_OUT.pw_post = lsts_post
            sammy_OUT.par_post = par_post
            sammy_OUT.chi2_post = chi2list_post
            sammy_OUT.chi2n_post = chi2nlist_post
            sammy_OUT.covariance_data_at_theory_post = covdat_at_theory_post
            sammy_OUT.total_derivative_evaluations = total_derivative_evaluations

    if empty:
        sammy_OUT.par = pd.DataFrame(columns=['E', 'Gg', 'Gn1', 'varyE', 'varyGg', 'varyGn1', 'J_ID'])
        sammy_OUT.par_post = pd.DataFrame(columns=['E', 'Gg', 'Gn1', 'varyE', 'varyGg', 'varyGn1', 'J_ID'])
        

    if not sammyRTO.keep_runDIR and os.path.isdir(sammyRTO.sammy_runDIR):
        shutil.rmtree(sammyRTO.sammy_runDIR)

    return sammy_OUT













def step_until_convergence_YW_Lasso(sammyRTO, sammyINPyw):
    ### some convenient definitions
    istep = 0
    no_improvement_tracker = 0
    lambda_lasso = 1e-4
    chi2_log = []
    fudge = sammyINPyw.initial_parameter_uncertainty
    par_df_current = sammyINPyw.resonance_ladder
    rundir = os.path.realpath(sammyRTO.sammy_runDIR)
    criteria="max steps"
    ### start loop
    if sammyRTO.Print:
        print(f"Stepping until convergence\nchi2 values\nstep fudge: {[exp.title for exp in sammyINPyw.experiments]+['sum', 'sum/ndat']}")
    while istep<sammyINPyw.max_steps:
        
        ### options to batch parameters being fit
        if sammyINPyw.batch_fitpar:
            if istep%sammyINPyw.steps_per_batch == 0:
                batch_fitpar(rundir, istep, sammyINPyw, fudge)
            else:
                pass
        
        ### get initial YW matrices for iteration and chi2 for current parameters
        i, chi2_list = run_YWY0_and_get_chi2(sammyINPyw, sammyRTO, istep)

        ### after step 1, do convergence checking
        if istep>=1:
            par_df_current = par_df_next
            
            ### convergence check
            Dchi2 = chi2_log[istep-1][-1] - chi2_list[-1]
            if Dchi2 < sammyINPyw.step_threshold:

                no_improvement_tracker += 1
                if no_improvement_tracker >= sammyINPyw.step_threshold_lag:
                    
                    if Dchi2 < 0:
                        criteria = f"Chi2 increased, taking solution {istep-1}"
                        if sammyINPyw.LevMar and fudge==sammyINPyw.minF:
                            criteria = f"Fudge below minimum value, taking solution {istep-1}"
                        if sammyRTO.Print:
                            print(f"{int(i)}    {np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")
                            print(criteria)
                        return max(istep-no_improvement_tracker, 0)
                    else:
                        criteria = "Chi2 improvement below threshold"
                    if sammyRTO.Print:
                        print(f"{int(i)}    {np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")
                        print(criteria)
                    return istep
                
                else:
                    pass

            else:   
                no_improvement_tracker = 0
            

        ### Log chi2
        chi2_log.append(chi2_list)
        if sammyRTO.Print:
            print(f"{int(i)}    {np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")
        
        ### Solve Bayes for this step
        converged, iterations = iterate_for_nonlin_and_update_step_par(sammyINPyw.iterations, istep, rundir, print_bool= sammyRTO.Print)


        ### 2 step Lasso if on
        Lasso = True
        gamma_lasso = 2
        if Lasso:
        # def Lasso_from_previous_step(rundir, istep, sammyINPyw, fudge):
            parfile_next = os.path.join(rundir,'results',f'step{istep+1}.par') # file to actually update as P = P_prior - a*dL/dP
            par_df_next = readpar(parfile_next)
            df_internal_next, df_external_next = separate_external_resonance_ladder(par_df_next, sammyINPyw.external_resonance_indices)
            df_internal_current, df_external_current = separate_external_resonance_ladder(par_df_current, sammyINPyw.external_resonance_indices)
            fit_mask = df_internal_current.varyGn1 == 1
            # fit_mask_next = df_internal_next.varyGn1 == 1

            alpha = fudge*df_internal_current['Gn1'][fit_mask].values
            # weight = could be model evaluated at par_df_current['Gn1'].values and Elam
            # more traditional weight is OLS estimate of par_df_current['Gn1'].values but we don't have that
            weight = 1/(df_internal_current['Gn1'][fit_mask].values**gamma_lasso * 2*np.sqrt(abs(df_internal_current['Gn1'][fit_mask].values)))
            dL_dP = lambda_lasso*weight*np.sign(df_internal_current['Gn1'][fit_mask].values)

            df_internal_next['Gn1'][fit_mask] = df_internal_next['Gn1'][fit_mask].values - alpha*dL_dP
            par_df_next, _ = concat_external_resonance_ladder(df_internal_next, df_external_next)
            write_sampar(par_df_next, sammyINPyw.particle_pair, fudge, parfile_next)


        ### update step
        istep += 1

    print("Maximum steps reached")
    return max(istep-1, 0)
