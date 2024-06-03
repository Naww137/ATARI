from typing import Union
import os
import shutil
from copy import deepcopy
import numpy as np
from numpy import newaxis as NA
import pandas as pd
from scipy.interpolate import interpn

from ATARI.theory.scattering_params import FofE_recursive

from ATARI.sammy_interface import sammy_functions, sammy_io
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyInputDataYW, SammyOutputData
from ATARI.sammy_interface.convert_u_p_params import convert_deriv_du2dp

def get_derivatives(sammyINP:Union[SammyInputData,SammyInputDataYW], sammyRTO:SammyRunTimeOptions, find_theo_trans:bool=False, u_or_p='u'):
    """
    Gets derivatives from SAMMY.

    Parameters
    ----------
    sammyINP : SammyInputData or SammyInputDataYW
        The SAMMY input data object.
    sammyRTO : SammyRunTimeOptions
        The SAMMY runtime options object.
    find_theo_trans : bool
        If true, SAMMY finds the theoretical transmission and uncertainty. Default is False.
    u_or_p : 'u' or 'p'
        Decides between using SAMMY defined u-parameters or p-parameters.

    Returns
    -------
    sammyOUT : SammyOutputData
        The SAMMY output data object.
    """

    sammy_io.make_runDIR(sammyRTO.sammy_runDIR)
    # Setting up sammy input file:
    if isinstance(sammyINP.experimental_data, pd.DataFrame):
        sammy_io.write_samdat(sammyINP.experimental_data, sammyINP.experimental_covariance, os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    else:
        sammy_io.write_estruct_file(sammyINP.energy_grid, os.path.join(sammyRTO.sammy_runDIR,"sammy.dat"))
    sammy_io.write_sampar(sammyINP.resonance_ladder, 
                 sammyINP.particle_pair, 
                 sammyINP.initial_parameter_uncertainty,
                 os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PAR'))
    # making templates:
    sammy_io.fill_runDIR_with_templates(sammyINP.template, "sammy.inp", sammyRTO.sammy_runDIR)
    # making sammy input file:
    sammy_io.write_saminp(
                        filepath     =   os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                        bayes        =   False,
                        iterations   =   sammyRTO.iterations,
                        formalism    =   sammyINP.particle_pair.formalism,
                        isotope      =   sammyINP.particle_pair.isotope,
                        M            =   sammyINP.particle_pair.M,
                        ac           =   sammyINP.particle_pair.ac*10,
                        reaction     =   sammyINP.experiment.reaction,
                        energy_range =   sammyINP.experiment.energy_range,
                        temp         =   sammyINP.experiment.temp,
                        FP           =   sammyINP.experiment.FP,
                        n            =   sammyINP.experiment.n,
                        use_IDC=False,
                        derivatives = True
                                 )
    # creating shell script:
    sammy_functions.write_shell_script(sammyINP, 
                                       sammyRTO, 
                                       use_RPCM=False, 
                                       use_IDC=False)
    # Executing sammy and reading outputs:
    # if sammyRTO.recursive == True:
    #     raise NotImplementedError('Recursive SAMMY has not been implemented.')
    lst_df, par_df, chi2, chi2n = sammy_functions.execute_sammy(sammyRTO)
    derivs_dict = sammy_functions.readpds(os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PDS'))
    if   u_or_p == 'p':
        derivs_dict['PARTIAL_DERIVATIVES'] = convert_deriv_du2dp(derivs_dict['PARTIAL_DERIVATIVES'], sammyINP.resonance_ladder)[0]
    elif u_or_p == 'u':
        pass
    else:
        raise ValueError('"u_or_p" can only be "u" or "p".')
    
    if find_theo_trans: # if we also need the theoretical value, run sammy again and get theoretical value
        sammy_functions.fill_runDIR_with_templates(sammyINP.template, "sammy.inp", sammyRTO.sammy_runDIR)
        sammy_functions.write_saminp(
                                    filepath     =   os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                                    bayes        =   False,
                                    iterations   =   sammyRTO.iterations,
                                    formalism    =   sammyINP.particle_pair.formalism,
                                    isotope      =   sammyINP.particle_pair.isotope,
                                    M            =   sammyINP.particle_pair.M,
                                    ac           =   sammyINP.particle_pair.ac*10,
                                    reaction     =   sammyINP.experiment.reaction,
                                    energy_range =   sammyINP.experiment.energy_range,
                                    temp         =   sammyINP.experiment.temp,
                                    FP           =   sammyINP.experiment.FP,
                                    n            =   sammyINP.experiment.n,
                                    use_IDC=False,
                                    derivatives=False,
                                    )
        sammy_functions.write_shell_script(sammyINP, 
                                        sammyRTO, 
                                        use_RPCM=False, 
                                        use_IDC=False)
        lst_df, par_df, chi2, chi2n = sammy_functions.execute_sammy(sammyRTO)
    sammy_OUT = SammyOutputData(pw=lst_df, 
                                par=par_df,
                                chi2=[chi2],
                                chi2n=[chi2n],
                                derivatives=derivs_dict['PARTIAL_DERIVATIVES'])
    # Removing run directory if specified:
    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    return sammy_OUT

def make_inputs_for_YW_deriv(sammyINPYW:SammyInputDataYW, sammyRTO:SammyRunTimeOptions, idc_list: list):

    #### make files for each dataset YW generation
    exp, idc_bool = sammyINPYW.experiments[0], idc_list
    for exp, idc_bool in zip(sammyINPYW.experiments, idc_list):  # fix this !!
        if idc_bool:
            idc_flag = ["USER-SUPPLIED IMPLICIT DATA COVARIANCE MATRIX"]
        else:
            idc_flag = []
        ### make YWY:
        sammy_functions.fill_runDIR_with_templates(exp.template, f"{exp.title}_deriv.inp", sammyRTO.sammy_runDIR)
        sammy_functions.write_saminp(
                    filepath   =    os.path.join(sammyRTO.sammy_runDIR, f"{exp.title}_deriv.inp"),
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
                    alphanumeric=idc_flag)
        
def plot_YW_deriv(sammyRTO, dataset_titles, i):
    out = subprocess.run(["sh", "-c", f"./YWY_deriv.sh {i}"], 
                cwd=os.path.realpath(sammyRTO.sammy_runDIR), capture_output=True, text=True, timeout=60*10
                        )
    par = sammy_functions.readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{i}.par"))
    lsts = []
    for dt in dataset_titles:
        lsts.append(sammy_functions.readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/{dt}.lst")) )
    i_chi2s = [float(s) for s in out.stdout.split('\n')[-2].split()]
    i=i_chi2s[0]
    chi2s=i_chi2s[1:len(dataset_titles)+1]
    chi2ns=i_chi2s[len(dataset_titles)+1:] 
    return par, lsts, chi2s, chi2ns

def get_derivatives_YW(sammyINPyw:SammyInputDataYW, sammyRTO:SammyRunTimeOptions, find_theo_trans:bool=False, u_or_p='u'):
    """
    Gets derivatives from SAMMY.

    Parameters
    ----------
    sammyINPyw : SammyInputDataYW
        The SAMMY input data object.
    sammyRTO : SammyRunTimeOptions
        The SAMMY runtime options object.
    find_theo_trans : bool
        If true, SAMMY finds the theoretical transmission and uncertainty. Default is False.
    u_or_p : 'u' or 'p'
        Decides between using SAMMY defined u-parameters or p-parameters.

    Returns
    -------
    sammyOUT : SammyOutputData
        The SAMMY output data object.
    """

    ## need to update functions to just pull titles and reactions from sammyINPyw.experiments
    # dataset_titles = [exp.title for exp in sammyINPyw.experiments]
    dataset_titles = sammy_functions.check_inputs_YW(sammyINPyw, sammyRTO)
    # sammyINPyw.reactions = [exp.reaction for exp in sammyINPyw.experiments]

    try:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    except:
        pass

    os.mkdir(sammyRTO.sammy_runDIR)
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "results"))
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "iterate"))

    idc_list = sammy_functions.make_data_for_YW(sammyINPyw.datasets, sammyINPyw.experiments, sammyRTO.sammy_runDIR, sammyINPyw.experimental_covariance)
    sammy_functions.write_sampar(sammyINPyw.resonance_ladder, sammyINPyw.particle_pair, sammyINPyw.initial_parameter_uncertainty, os.path.join(sammyRTO.sammy_runDIR, "results/step0.par"))

    make_inputs_for_YW_deriv(sammyINPyw, sammyRTO, idc_list)
    dataset_titles = [exp.title for exp in sammyINPyw.experiments]
    make_YWY_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR, idc_list)
    os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, 'YWY_deriv.sh')}")

    ### get prior
    par, lsts, chi2list, chi2nlist = plot_YW(sammyRTO, dataset_titles, 0)
    sammy_OUT = SammyOutputData(pw=lsts, par=par, chi2=chi2list, chi2n=chi2nlist)

    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    return sammy_OUT




    sammy_io.make_runDIR(sammyRTO.sammy_runDIR)
    # Setting up sammy input file:
    if isinstance(sammyINP.experimental_data, pd.DataFrame):
        sammy_io.write_samdat(sammyINP.experimental_data, sammyINP.experimental_covariance, os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    else:
        sammy_io.write_estruct_file(sammyINP.energy_grid, os.path.join(sammyRTO.sammy_runDIR,"sammy.dat"))
    sammy_io.write_sampar(sammyINPyw.resonance_ladder, 
                 sammyINPyw.particle_pair, 
                 sammyINPyw.initial_parameter_uncertainty,
                 os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PAR'))
    # making templates:
    sammy_io.fill_runDIR_with_templates(exp.template, "sammy.inp", sammyRTO.sammy_runDIR)
    # making sammy input file:
    sammy_io.write_saminp(
                        filepath     =   os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                        bayes        =   False,
                        iterations   =   sammyRTO.iterations,
                        formalism    =   sammyINPyw.particle_pair.formalism,
                        isotope      =   sammyINPyw.particle_pair.isotope,
                        M            =   sammyINPyw.particle_pair.M,
                        ac           =   sammyINPyw.particle_pair.ac*10,
                        reaction     =   exp.reaction,
                        energy_range =   exp.energy_range,
                        temp         =   exp.temp,
                        FP           =   exp.FP,
                        n            =   exp.n,
                        use_IDC=False,
                        derivatives = True
                                 )
    # creating shell script:
    sammy_functions.write_shell_script(sammyINP, 
                                       sammyRTO, 
                                       use_RPCM=False, 
                                       use_IDC=False)
    # Executing sammy and reading outputs:
    # if sammyRTO.recursive == True:
    #     raise NotImplementedError('Recursive SAMMY has not been implemented.')
    lst_df, par_df, chi2, chi2n = sammy_functions.execute_sammy(sammyRTO)
    derivs_dict = sammy_functions.readpds(os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PDS'))
    if   u_or_p == 'p':
        derivs_dict['PARTIAL_DERIVATIVES'] = convert_deriv_du2dp(derivs_dict['PARTIAL_DERIVATIVES'], sammyINP.resonance_ladder)[0]
    elif u_or_p == 'u':
        pass
    else:
        raise ValueError('"u_or_p" can only be "u" or "p".')
    
    if find_theo_trans: # if we also need the theoretical value, run sammy again and get theoretical value
        sammy_functions.fill_runDIR_with_templates(sammyINP.template, "sammy.inp", sammyRTO.sammy_runDIR)
        sammy_functions.write_saminp(
                                    filepath     =   os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                                    bayes        =   False,
                                    iterations   =   sammyRTO.iterations,
                                    formalism    =   sammyINP.particle_pair.formalism,
                                    isotope      =   sammyINP.particle_pair.isotope,
                                    M            =   sammyINP.particle_pair.M,
                                    ac           =   sammyINP.particle_pair.ac*10,
                                    reaction     =   sammyINP.experiment.reaction,
                                    energy_range =   sammyINP.experiment.energy_range,
                                    temp         =   sammyINP.experiment.temp,
                                    FP           =   sammyINP.experiment.FP,
                                    n            =   sammyINP.experiment.n,
                                    use_IDC=False,
                                    derivatives=False,
                                    )
        sammy_functions.write_shell_script(sammyINP, 
                                        sammyRTO, 
                                        use_RPCM=False, 
                                        use_IDC=False)
        lst_df, par_df, chi2, chi2n = sammy_functions.execute_sammy(sammyRTO)
    sammy_OUT = SammyOutputData(pw=lst_df, 
                                par=par_df,
                                chi2=[chi2],
                                chi2n=[chi2n],
                                derivatives=derivs_dict['PARTIAL_DERIVATIVES'])
    # Removing run directory if specified:
    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    return sammy_OUT

def find_interpolation_array(particle_pair, exp_model_T, sammyRTO:SammyRunTimeOptions, Els, Ggs, Gns, Ls, Es, u_or_p='u'):
    """
    ...
    """

    Els = np.array(Els)
    Gns = np.array(Gns)
    Ggs = np.array(Ggs)
    Ls  = np.array(Ls )
    Es  = np.array(Es )
    particle_pair = deepcopy(particle_pair)
    exp_model_T   = deepcopy(exp_model_T  )
    Ps = FofE_recursive(Es, particle_pair.ac, particle_pair.M, particle_pair.m, Ls)[1]

    derivatives_array = np.zeros((len(Els), len(Ggs), len(Gns), len(Ls), len(Es), 3)) # doesn't everybody love a 6D array?
    for ei, El in enumerate(Els):
        for gi, Gg in enumerate(Ggs):
            for ni, Gn in enumerate(Gns):
                for li, L in enumerate(Ls):
                    # El = np.array(El)
                    # Gg = np.array(Gg)
                    # Gn = np.array(Gn)
                    gg2 = Gg / 2
                    gn2 = Gn / (2*Ps[li,ei])
                    Jpi = 0 # NOTE: does not matter (I think)
                    JID = 1 # NOTE: does not matter (I think)
                    E_grid = np.array(Es) + El
                    E_limits = (E_grid[0]-1e-3, E_grid[-1]+1e-3)
                    particle_pair.energy_range = E_limits
                    particle_pair.resonance_ladder = pd.DataFrame([[El], [Gg], [Gn], [JID], [gg2], [gn2], [Jpi], [L]],
                                                        index=['E', 'Gg', 'Gn1',  'J_ID', 'gg2', 'gn2', 'Jpi', 'L',]).T
                    exp_model_T.energy_range = E_limits
                    exp_model_T.energy_grid = E_grid
                    sammyINP = SammyInputData(particle_pair,
                                              particle_pair.resonance_ladder,
                                              os.path.realpath('template_T.inp'),
                                              exp_model_T,
                                              energy_grid=exp_model_T.energy_grid)
                    sammy_out = get_derivatives(sammyINP, sammyRTO, find_theo_trans=True, u_or_p=u_or_p)
                    T = sammy_out.pw['theo_trans'].to_numpy()
                    print(T)
                    dT_dP = sammy_out.derivatives.reshape(-1,3)
                    dLT_dP = dT_dP / T[:,NA]
                    derivatives_array[ei,gi,ni,li,:,:] = dLT_dP # El, Gg, Gn

    points = (Els, Ggs, Gns, Ls, Es)
    return points, derivatives_array

    

def interpolate_derivatives(points, derivatives_array, Els, Ggs, Gns, Ls, Es, T):
    """
    ...
    """

    A = np.column_stack((Els, Ggs, Gns, Ls))
    X = np.repeat(A, len(Es), axis=0)
    X = np.column_stack((X, np.tile(Es, A.shape)))
    dT_dEl = T * interpn(points, derivatives_array[:,:,:,:,:,0], X)
    dT_dGg = T * interpn(points, derivatives_array[:,:,:,:,:,1], X)
    dT_dGn = T * interpn(points, derivatives_array[:,:,:,:,:,2], X)
    return dT_dEl, dT_dGg, dT_dGn