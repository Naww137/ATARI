import os
import shutil
from copy import deepcopy
import numpy as np
from numpy import newaxis as NA
import pandas as pd
from scipy.interpolate import interpn

from ATARI.theory.scattering_params import FofE_recursive

from ATARI.sammy_interface import sammy_functions, sammy_io
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyOutputData
from ATARI.sammy_interface.convert_u_p_params import convert_deriv_du2dp, get_Pu_vec_from_reslad


def get_derivatives(sammyINP:SammyInputData, sammyRTO:SammyRunTimeOptions, get_theo:bool=False, u_or_p='u'):
    """
    Gets derivatives from SAMMY.

    Parameters
    ----------
    sammyINP : SammyInputData
        The SAMMY input data object.
    sammyRTO : SammyRunTimeOptions
        The SAMMY runtime options object.
    get_theo : bool
        If true, SAMMY finds the theory value and uncertainty with an additional sammy call. Default is False.
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
    sammy_io.fill_runDIR_with_templates(sammyINP.experiment.template, "sammy.inp", sammyRTO.sammy_runDIR)
    # making sammy input file:
    sammy_io.write_saminp(
                        filepath   =    os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"),
                        bayes       =   False,
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
                        use_IDC=False,
                        derivatives = True
                                 )
    # creating shell script:
    sammy_functions.write_shell_script(sammyINP, 
                                       sammyRTO, 
                                       use_RPCM=False, 
                                       use_IDC=False)
    # runsammy
    lst_df, par_df, chi2, chi2n = sammy_functions.execute_sammy(sammyRTO)
    derivs_dict = sammy_functions.readpds(os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PDS'))

    ### sort derivatives based on resonance_ladder order
    pu_reslad = get_Pu_vec_from_reslad(sammyINP.resonance_ladder,sammyINP.particle_pair)
    pu_derivs = np.array(derivs_dict["U"]).reshape(-1,3)
    ider = []
    for row in pu_reslad.reshape(-1,3):
        for i, drow in enumerate(pu_derivs):
            if np.all(np.isclose(row,drow, atol=1e-3)):
                ider.extend([i*3,i*3+1,i*3+2])
                break                                           # TODO: This is not the best way to do this, it assumes if parameters are the same within 1e-4 then derivative is the same
    assert(len(ider) == len(sammyINP.resonance_ladder)*3)
    if not np.all(np.isclose(np.array(derivs_dict["U"])[ider], pu_reslad, atol=1e-3)):
        maxdiff = np.max( abs(np.array(derivs_dict["U"])[ider] - pu_reslad))
        print(f"WARNING: Pu in .IDF file does not agree with Pu in python API with max absolute difference: {maxdiff}")
    
    derivs_dict["PARTIAL_DERIVATIVES"] = derivs_dict["PARTIAL_DERIVATIVES"][:, ider]

    # convert to p if necessary
    if   u_or_p == 'p':
        derivs_dict['PARTIAL_DERIVATIVES'] = convert_deriv_du2dp(derivs_dict['PARTIAL_DERIVATIVES'], sammyINP.resonance_ladder)[0]
    elif u_or_p == 'u':
        pass
    else:
        raise ValueError('"u_or_p" can only be "u" or "p".')
    
    ### assert that the data read into derivs dict is the same as in lst out
    if sammyINP.experiment.reaction == "transmission":
        assert(np.isclose(np.sum((lst_df.exp_trans - np.array(derivs_dict["DATA"]))**2), 0, atol=1e-8))
        lst_df.theo_trans = np.array(derivs_dict["THEORY"])
    elif sammyINP.experiment.reaction == "capture":
        assert(np.isclose(np.sum((lst_df.exp_xs - np.array(derivs_dict["DATA"]))**2), 0, atol=1e-8))
        lst_df.theo_xs = np.array(derivs_dict["THEORY"])

    ### Collect sammy outputs and remove run directory if specified:
    sammy_OUT = SammyOutputData(pw=lst_df, 
                                par=par_df,
                                chi2=[chi2],
                                chi2n=[chi2n],
                                derivatives=derivs_dict['PARTIAL_DERIVATIVES'])
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