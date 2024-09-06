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

def get_derivatives(sammy_inp:SammyInputData, sammy_rto:SammyRunTimeOptions, u_or_p='u'):
    """
    Gets derivatives from SAMMY.

    Parameters
    ----------
    sammy_inp : SammyInputData
        The SAMMY input data object.
    sammy_rto : SammyRunTimeOptions
        The SAMMY runtime options object.
    u_or_p : 'u' or 'p'
        Decides between using SAMMY defined u-parameters or p-parameters.

    Returns
    -------
    sammy_out : SammyOutputData
        The SAMMY output data object.
    """

    sammy_io.make_runDIR(sammy_rto.sammy_runDIR)
    # Setting up sammy input file:
    if isinstance(sammy_inp.experimental_data, pd.DataFrame):
        sammy_io.write_samdat(sammy_inp.experimental_data, sammy_inp.experimental_covariance, os.path.join(sammy_rto.sammy_runDIR,'sammy.dat'))
    else:
        sammy_io.write_estruct_file(sammy_inp.energy_grid, os.path.join(sammy_rto.sammy_runDIR,"sammy.dat"))
    sammy_io.write_sampar(sammy_inp.resonance_ladder, 
                          sammy_inp.particle_pair, 
                          sammy_inp.initial_parameter_uncertainty,
                          os.path.join(sammy_rto.sammy_runDIR, 'SAMMY.PAR'))
    # making templates:
    sammy_io.fill_runDIR_with_templates(sammy_inp.experiment.template, "sammy.inp", sammy_rto.sammy_runDIR)
    # making sammy input file:
    sammy_io.write_saminp(
                        filepath     =   os.path.join(sammy_rto.sammy_runDIR,"sammy.inp"),
                        bayes        =   False,
                        iterations   =   sammy_rto.iterations,
                        formalism    =   sammy_inp.particle_pair.formalism,
                        isotope      =   sammy_inp.particle_pair.isotope,
                        M            =   sammy_inp.particle_pair.M,
                        ac           =   sammy_inp.particle_pair.ac*10,
                        reaction     =   sammy_inp.experiment.reaction,
                        energy_range =   sammy_inp.experiment.energy_range,
                        temp         =   sammy_inp.experiment.temp,
                        FP           =   sammy_inp.experiment.FP,
                        n            =   sammy_inp.experiment.n,
                        use_IDC=False,
                        derivatives = True
                                 )
    # creating shell script:
    sammy_functions.write_shell_script(sammy_inp, 
                                       sammy_rto, 
                                       use_RPCM=False, 
                                       use_IDC=False)
    # Executing sammy and reading outputs:
    # if sammyRTO.recursive == True:
    #     raise NotImplementedError('Recursive SAMMY has not been implemented.')
    lst_df, par_df, chi2, chi2n = sammy_functions.execute_sammy(sammy_rto)

    # Getting Transmission:
    derivs_dict = sammy_io.readpds(os.path.join(sammy_rto.sammy_runDIR, 'SAMMY.PDS'))
    if sammy_inp.experiment.reaction == 'transmission':
        lst_df['theo_trans'] = derivs_dict['THEORY']
    else:
        lst_df['theo_xs'] = derivs_dict['THEORY']

    if   u_or_p == 'p':
        derivatives = convert_deriv_du2dp(derivs_dict['PARTIAL_DERIVATIVES'], sammy_inp.resonance_ladder)[0] # convert from u to p
    elif u_or_p == 'u':
        derivatives = derivs_dict['PARTIAL_DERIVATIVES'] # no change
    else:
        raise ValueError('"u_or_p" can only be "u" or "p".')
    
    sammy_out = SammyOutputData(pw=lst_df, 
                                par=par_df,
                                chi2=[chi2],
                                chi2n=[chi2n],
                                derivatives=derivatives)
    # Removing run directory if specified:
    if not sammy_rto.keep_runDIR:
        shutil.rmtree(sammy_rto.sammy_runDIR)
    return sammy_out


def get_derivatives_YW(sammy_inp_yw:Union[SammyInputData,SammyInputDataYW], sammy_rto:SammyRunTimeOptions, u_or_p='u'):
    """
    Gets derivatives from SAMMY with multiple datasets using the YW formulation.

    Parameters
    ----------
    sammy_inp_yw : SammyInputDataYW or SammyInputData
        The YW SAMMY input data object.
    sammy_rto : SammyRunTimeOptions
        The SAMMY runtime options object.
    u_or_p : 'u' or 'p'
        Decides between using SAMMY defined u-parameters or p-parameters.

    Returns
    -------
    sammy_out : SammyOutputData
        The SAMMY output data object.
    """

    # If a simple SammyInputData object, pass to get_derivatives:
    if isinstance(sammy_inp_yw, SammyInputData):
        sammy_out = get_derivatives(sammy_inp_yw, sammy_rto, u_or_p=u_or_p)
        return sammy_out

    # Checking YW inputs:
    dataset_titles = sammy_functions.check_inputs_YW(sammy_inp_yw, sammy_rto)

    try:
        shutil.rmtree(sammy_rto.sammy_runDIR)
    except:
        pass

    pw_list = []; chi2_list = []; chi2n_list = []; derivatives_list = []
    for exp, dat, exp_cov in zip(sammy_inp_yw.experiments, sammy_inp_yw.datasets, sammy_inp_yw.experimental_covariance):
        sammy_inp = SammyInputData(sammy_inp_yw.particle_pair,
                                  sammy_inp_yw.resonance_ladder,
                                  template=exp.template,
                                  experiment=exp,
                                  experimental_data=dat,
                                  experimental_covariance=exp_cov)
        sammy_out = get_derivatives(sammy_inp, sammy_rto, u_or_p=u_or_p)
        pw_list.append(sammy_out.pw)
        chi2_list.append(sammy_out.chi2)
        chi2n_list.append(sammy_out.chi2n)
        derivatives_list.append(sammy_out.derivatives)

    par = sammy_out.par
    sammy_out_yw = SammyOutputData(pw_list, par, chi2_list, chi2n_list, derivatives=derivatives_list)
    return sammy_out_yw


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
                                            #   os.path.realpath('template_T.inp'),
                                                exp_model_T.template,
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