import os
import shutil
from copy import copy

from ATARI.sammy_interface.sammy_functions import make_runDIR, write_estruct_file, write_sampar, fill_runDIR_with_templates, write_saminp, write_shell_script, execute_sammy, readpds
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyOutputData
from ATARI.sammy_interface.convert_u_p_params import convert_deriv_du2dp

def get_derivatives(sammyINP:SammyInputData, sammyRTO:SammyRunTimeOptions, u_or_p='u'):
    make_runDIR(sammyRTO.sammy_runDIR)
    # Setting up sammy input file:
    write_estruct_file(sammyINP.energy_grid, os.path.join(sammyRTO.sammy_runDIR,"sammy.dat"))
    sammyRTO_alt = copy(sammyRTO)
    sammyRTO_alt.derivatives_only = True
    sammyRTO_alt.bayes = False
    write_sampar(sammyINP.resonance_ladder, 
                 sammyINP.particle_pair, 
                 sammyINP.initial_parameter_uncertainty,
                 os.path.join(sammyRTO_alt.sammy_runDIR, 'SAMMY.PAR'))
    # making templates:
    fill_runDIR_with_templates(sammyINP.template, 
                               "sammy.inp", 
                               sammyRTO_alt.sammy_runDIR)
    # making sammy input file:
    write_saminp(os.path.join(sammyRTO_alt.sammy_runDIR,"sammy.inp"), 
                 particle_pair=sammyINP.particle_pair, 
                 experimental_model=sammyINP.experiment, 
                 rto=sammyRTO_alt,
                 use_IDC=False,
                 use_ecscm_reaction=False)
    # creating shell script:
    write_shell_script(sammyINP, 
                       sammyRTO_alt, 
                       use_RPCM=False, 
                       use_IDC=False)
    # Executing sammy and reading outputs:
    # if sammyRTO.recursive == True:
    #     raise NotImplementedError('Recursive SAMMY has not been implemented.')
    lst_df, par_df, chi2, chi2n = execute_sammy(sammyRTO_alt)
    derivs_dict = readpds(os.path.join(sammyRTO_alt.sammy_runDIR, 'SAMMY.PDS'))
    if   u_or_p == 'p':
        raise NotImplementedError('...')
    elif u_or_p == 'u':
        derivs_dict['PARTIAL_DERIVATIVES'] = convert_deriv_du2dp(derivs_dict['PARTIAL_DERIVATIVES'], sammyINP.resonance_ladder)[0]
    else:
        raise ValueError('"u_or_p" can only be "u" or "p".')
    sammy_OUT = SammyOutputData(pw=lst_df, 
                                par=sammyINP.resonance_ladder,
                                chi2=[chi2],
                                chi2n=[chi2n],
                                derivatives=derivs_dict['PARTIAL_DERIVATIVES'])
    # Removing run directory if specified:
    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    return sammy_OUT