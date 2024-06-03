from typing import Union
from copy import copy
from math import pi
import numpy as np
import pandas as pd
from numpy import newaxis as NA
import scipy.sparse as sp
import warnings

from ATARI.sammy_interface.convert_u_p_params import p2u_E, p2u_g, p2u_n,   u2p_E, u2p_g, u2p_n,   du_dp_E, du_dp_g, du_dp_n
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyInputDataYW, SammyOutputData
from ATARI.sammy_interface.sammy_deriv import get_derivatives
from ATARI.Bayes import utils

def BayesMW(P, M, G, V, D, T):
    """
    The MW Bayes Scheme for SAMMY. This method is most useful when the use_least_squares method is
    used (i.e. M is None). 

    Source: SAMMY Manual (section IV.B.3)
    """

    # Removing all-zero columns:
    nonzero_columns = np.any(G != 0, axis=0)
    used_cols = np.nonzero(nonzero_columns)[0]
    G = G[:,used_cols]

    # Is V diagonal?
    V_diag = (V.ndim == 1)

    if V_diag:  ViG = G / V[:,NA]
    else:       ViG = np.linalg.solve(V, G)
    W = G.T @ ViG
    del ViG

    if M is None:
        MiW = W
    else:
        M = M[:,used_cols][used_cols,:]
        MiW = np.linalg.inv(M) + W
    Mpc = np.linalg.inv(MiW)
    
    if V_diag:  VDT = (D - T) / V
    else:       VDT = np.linalg.solve(V, (D-T))
    
    # Only updating non-zero gradient parameters:
    Pp = P
    Pp[used_cols] += (Mpc @ G.T @ VDT)
    Mp = np.zeros((len(P),len(P)))
    Mp[:,used_cols][used_cols,:] = Mpc
    return Pp, Mp

def BayesIQ(P, M, G, V, D, T):
    """
    The IQ Bayes Scheme for SAMMY. This method is most useful when V is diagonal.

    Source: SAMMY Manual (section IV.B.2)
    """

    # Is V diagonal?
    V_diag = (V.ndim == 1)
    
    if V_diag:  ViG = G / V[:,NA]
    else:       ViG = np.linalg.solve(V, G)
    Q = M @ (G.T @ ViG)
    del ViG

    I = np.eye(*M.shape)
    Mp = np.linalg.solve(I+Q, M)
    del I, Q

    if V_diag:  VDT = (D - T) / V
    else:       VDT = np.linalg.solve(V, (D-T))

    Pp = P + Mp @ G.T @ VDT
    return Pp, Mp

def BayesNV(P, M, G, V, D, T):
    """
    The NV Bayes Scheme for SAMMY. This method is most useful when V has covariance elements.

    Source: SAMMY Manual (section IV.B.1)
    """

    # Force V to be diagonal if in diagonal-only form:
    if V.ndim == 1:
        V = np.diag(V)

    GM = G @ M
    N = GM @ G.T
    NVG = np.linalg.solve(N+V, G)
    del N

    MGNV = M @ NVG.T
    del NVG

    Mp = M - MGNV @ GM
    Pp = P + MGNV @ (D - T)
    return Pp, Mp

def Bayes(sammy_out:SammyOutputData,
          particle_pair:Particle_Pair,
          sammyRTO:SammyRunTimeOptions,
          initial_parameter_uncertainty:float=1.0):
    """
    General Bayes equations using derivatives provided from the function `get_derivatives`.

    Parameters
    ----------
    sammy_out : SammyOutputData
        The SAMMY output data from `get_derivatives`, including pointwise data.
    particle_pair : Particle_Pair
        The particle_pair data used to convert between "u" and "p" parameters.
    sammyRTO : SammyRunTimeOptions
        The SAMMY runtime options.
    initial_parameter_uncertainty : float
        The initial relative parameter uncertainties when a prior covariance matrix is not
        provided. Default is 1.0.

    Returns
    -------
    par_post : DataFrame
        The posterior parameter estimates.
    Mpu : ndarray[float]
        The posterior parameter uncertainties covariance matrix.
    """

    model = sammyRTO.bayes_scheme
    use_least_squares = sammyRTO.use_least_squares
    if sammyRTO.iterations != 1:
        raise NotImplementedError('Only one iteration optimization has been implemented.')
    
    par = sammy_out.par

    # Derivatives:
    Gu = sammy_out.derivatives
    # Vary conditions:
    indices_e = 3*par.index[par['varyE'] == 0].to_numpy()
    indices_g = 3*par.index[par['varyGg'] == 0].to_numpy() + 1
    indices_n = 3*par.index[par['varyGn1'] == 0].to_numpy()  + 2
    Gu[:,indices_e] = 0.0
    Gu[:,indices_g] = 0.0
    Gu[:,indices_n] = 0.0
    
    # Pointwise Data:
    pw  = sammy_out.pw
    T = pw['theo_trans'].to_numpy()
    D = pw['exp_trans'].to_numpy()
    pw_unc = pw['exp_trans_unc'].to_numpy()
    if pw_unc.ndim == 1:
        V = pw_unc**2
    else:
        raise NotImplementedError('Only ')
    
    E  = par['E'].to_numpy()
    Gg = par['Gg'].to_numpy()
    Gn = par['Gn1'].to_numpy()
    J_ID = par['J_ID'].to_numpy(dtype=int)

    L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    for jpi, mean_params in particle_pair.spin_groups.items():
        jid = mean_params['J_ID']
        L[jid-1] = mean_params['Ls'][0]

    ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    Pu = np.column_stack((ue, ug, un)).reshape(-1,)
    
    if use_least_squares:
        Mu = None
        model = 'MW'
    else:
        warnings.warn('Only least squares matches SAMMY. An approximation is used with prior covariances.')
        
        # Initial Parameter Uncertainty Matrix:
        #    based on initial parameter uncertainties
        dPp = np.zeros((len(E),3))
        dPp[:,0] = 0.5 * 0.001 * (Gg + Gn)
        dPp[:,1] = initial_parameter_uncertainty * Gg
        dPp[:,2] = initial_parameter_uncertainty * Gn
        dPu = np.zeros((len(E),3))
        dPu[:,0] = du_dp_E(E ) * dPp[:,0]
        dPu[:,1] = du_dp_g(Gg) * dPp[:,1]
        dPu[:,2] = du_dp_n(Gn, E, L[J_ID-1], particle_pair) * dPp[:,2]
        dPu = dPu.reshape(-1,)
        Mu = np.diag(dPu**2)
    
    if   model == 'IQ':     Bayes = BayesIQ
    elif model == 'NV':     Bayes = BayesNV
    elif model == 'MW':     Bayes = BayesMW
    else:                   raise NotImplementedError('...')
    Pu, Mpu = Bayes(Pu, Mu, Gu, V, D, T)
    Pu  = Pu.reshape(-1,3)
    Eu  = Pu[:,0]
    Ggu = Pu[:,1]
    Gnu = Pu[:,2]
    Pp  = np.zeros_like(Pu)
    Pp[:,0] = u2p_E(Eu)
    Pp[:,1] = u2p_g(Ggu)
    Pp[:,2] = u2p_n(Gnu, Pp[:,0], L[J_ID-1], particle_pair)
    
    par_post = pd.DataFrame(Pp, columns=['E', 'Gg', 'Gn1'])
    par_post['J_ID']    = par['J_ID']
    par_post['varyE']   = par['varyE']
    par_post['varyGn1'] = par['varyGn1']
    par_post['varyGg']  = par['varyGg']
    # par_post = particle_pair.expand_ladder(par_post)
    return par_post, Mpu

# =================================================================================================
#   Incomplete Functions
# =================================================================================================

# def BayesIter(P, M, V, D, func, num_iterations:int=1, model='NV'):
#     """
#     ...
#     """

#     if M is None:
#         model = 'MW'
#     if   model == 'IQ':     Bayes = BayesIQ
#     elif model == 'NV':     Bayes = BayesNV
#     elif model == 'MW':     Bayes = BayesMW
#     else:                   raise NotImplementedError('...')
#     Pn = P;  Mn = M
#     for iter in range(num_iterations):
#         Tn, Gn = func(Pn)
#         T = Tn - Gn @ (P - Pn)
#         Pn, Mn = Bayes(P, M, Gn, V, D, T)
#     return Pn, Mn















# =================================================================================================
#   New Bayes Implementation
# =================================================================================================

# def Bayes_extended(sammy_inp:SammyInputData,
#                    sammy_rto:SammyRunTimeOptions,
#                    RPCM:np.ndarray=None,
#                    WigLL:bool=True, PTLL:bool=True):
#     """
#     ...
#     """

#     particle_pair = sammy_inp.particle_pair
    
#     # Prior matrices:
#     Mu, V, D = Bayes_prior_matrices(sammy_inp, sammy_rto, RPCM)

#     # Iterating Bayes:
#     for iter in range(sammy_rto.iterations):
#         # Getting derivatives:
#         sammy_out = get_derivatives(sammy_inp, sammy_rto, find_theo_trans=True, u_or_p='u')

#         # Gathering the rest of the matrices:
#         Pu, Gu, Tn, P_vary = Bayes_matrices(sammy_out, particle_pair, sammy_rto)
        
#         # Updating prior fit for iteration scheme:
#         if iter == 0:
#             Pu0 = Pu
#             T = Tn
#         else:
#             T = Tn + Gu @ (Pu0 - Pu)

#         # Bayes Likelihoods:
#         Ui, W, C = Bayes_likelihoods(sammy_inp.resonance_ladder, particle_pair, WigLL, PTLL)

#         # Running Bayes:
#         Ppu, Mpu = Bayes_conditioner(sammy_rto, Pu0, Mu, Gu, V, D, T, P_vary, Ui, W, C, scheme=sammy_rto.bayes_scheme)

#         # Converting from U to P parameters:
#         par = sammy_inp.resonance_ladder
#         J_IDs = par['J_ID'].to_numpy(dtype=int)
#         Ls    = par['L'   ].to_numpy(dtype=int)
#         Ppu = Ppu.reshape(-1,3)
#         Eu  = Ppu[:,0]
#         Ggu = Ppu[:,1]
#         Gnu = Ppu[:,2]
#         Ppp = np.zeros_like(Ppu)
#         Ppp[:,0] = u2p_E(Eu)
#         Ppp[:,1] = u2p_g(Ggu)
#         Ppp[:,2] = u2p_n(Gnu, Ppp[:,0], Ls[J_IDs-1], particle_pair)
#         # filling in posterior parameter dataframe:
#         par_post = pd.DataFrame(Ppp, columns=['E', 'Gg', 'Gn1'])
#         par_post['J_ID']    = par['J_ID']
#         par_post['L']       = par['L']
#         par_post['Jpi']     = par['Jpi']
#         par_post['varyE']   = par['varyE']
#         par_post['varyGn1'] = par['varyGn1']
#         par_post['varyGg']  = par['varyGg']
#         sammy_inp.resonance_ladder = par_post
#     return par_post, Mpu

# def Bayes_prior_matrices(sammy_inp:SammyInputData,
#                          sammy_rto:SammyRunTimeOptions,
#                          RPCM:np.ndarray=None):
#     """
#     ...
#     """

#     # Parameters:
#     par = sammy_inp.resonance_ladder
#     particle_pair = sammy_inp.particle_pair
#     E    = par['E'].to_numpy()
#     Gg   = par['Gg'].to_numpy()
#     Gn   = par['Gn1'].to_numpy()
#     J_ID = par['J_ID'].to_numpy(dtype=int)
#     L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
#     for jpi, mean_params in particle_pair.spin_groups.items():
#         jid = mean_params['J_ID']
#         L[jid-1] = mean_params['Ls'][0]
#     # converting to U-parameters
#     ue = p2u_E(E)
#     ug = p2u_g(Gg)
#     un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
#     Pu = np.column_stack((ue, ug, un)).reshape(-1,)
    
#     # Pointwise Data:
#     D = sammy_inp.experimental_data['exp_trans'].to_numpy()
#     V = sammy_inp.experimental_covariance
#     if V is None:
#         raise NotImplementedError('Bayes without data covariance matrix not implemented yet.')
    
#     if sammy_rto.use_least_squares:
#         Mu = None
#     elif RPCM is not None:
#         Mp = RPCM 
#         dU_dP = np.zeros_like(Pu)
#         dU_dP[ ::3] = du_dp_E(E)
#         dU_dP[1::3] = du_dp_g(Gg)
#         dU_dP[2::3] = du_dp_n(Gn, E, L[J_ID-1], particle_pair)
#         Mu = dU_dP[NA,:] * Mp * dU_dP[:,NA]
#     else: # use fudge factor
#         fudge = sammy_inp.initial_parameter_uncertainty
#         warnings.warn('Only least squares matches SAMMY. An approximation is used with prior covariances.')
#         dPp = np.zeros((len(E),3))
#         dPp[:,0] = 0.5 * 0.001 * (Gg + Gn) # FIXME: THIS IS SUPPOSED TO BE A MULTIPLE OF THE BROADENNED WIDTH!!!
#         dPp[:,1] = fudge * Gg
#         dPp[:,2] = fudge * Gn
#         dPu = np.zeros((len(E),3))
#         dPu[:,0] = du_dp_E(E ) * dPp[:,0]
#         dPu[:,1] = du_dp_g(Gg) * dPp[:,1]
#         dPu[:,2] = du_dp_n(Gn, E, L[J_ID-1], particle_pair) * dPp[:,2]
#         dPu = dPu.reshape(-1,)
#         Mu = dPu**2
#     return Mu, V, D

# def Bayes_matrices(sammy_out:SammyOutputData,
#                    particle_pair:Particle_Pair,
#                    sammy_rto:SammyRunTimeOptions):
#     """
#     ...
#     """

#     # Parameters:
#     par = sammy_out.par
#     E    = par['E'].to_numpy()
#     Gg   = par['Gg'].to_numpy()
#     Gn   = par['Gn1'].to_numpy()
#     J_ID = par['J_ID'].to_numpy(dtype=int)
#     L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
#     for jpi, mean_params in particle_pair.spin_groups.items():
#         jid = mean_params['J_ID']
#         L[jid-1] = mean_params['Ls'][0]
#     # converting to U-parameters
#     ue = p2u_E(E)
#     ug = p2u_g(Gg)
#     un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
#     Pu = np.column_stack((ue, ug, un)).reshape(-1,)

#     # Derivatives:
#     Gu = sammy_out.derivatives

#     # Parameter vary conditions:
#     P_vary = np.zeros((3*len(par),), dtype=bool)
#     P_vary[ ::3] = par['varyE'  ].to_numpy()
#     P_vary[1::3] = par['varyGg' ].to_numpy()
#     P_vary[2::3] = par['varyGn1'].to_numpy()
    
#     # Pointwise Data:
#     pw  = sammy_out.pw
#     T = pw['theo_trans'].to_numpy()
    
#     return Pu, Gu, T, P_vary

from time import time

def Bayes_extended(sammy_inp:Union[SammyInputData,SammyInputDataYW],
                   sammy_rto:SammyRunTimeOptions,
                   RPCM:np.ndarray=None,
                   WigLL:bool=True, PTLL:bool=True):
    """
    ...
    """

    print('start')
    t0 = time()

    particle_pair = sammy_inp.particle_pair
    YW = isinstance(sammy_inp, SammyInputDataYW)
    
    # Prior matrices:
    Mu, V, D = Bayes_prior_matrices(sammy_inp, sammy_rto, RPCM)

    print(f'After prior matrices: {time()-t0} s')
    t0 = time()

    # Iterating Bayes:
    for iter in range(sammy_rto.iterations):
        # Getting derivatives:
        sammy_out = get_derivatives(sammy_inp, sammy_rto, find_theo_trans=True, u_or_p='u') # FIXME: YW DERIVATIVES?
        print(f'After derivatives: {time()-t0} s')
        t0 = time()

        # Gathering the rest of the matrices:
        Pu, Gu, Tn, P_vary = Bayes_matrices(sammy_out, particle_pair, sammy_rto)
        print(f'After Bayes Matrices: {time()-t0} s')
        t0 = time()

        # Updating prior fit for iteration scheme:
        if iter == 0:
            Pu0 = Pu
            T = Tn
        else:
            if YW:
                T = [Tnj + Guj @ (Pu0j - Puj) \
                     for Tnj, Guj, Pu0j, Puj in zip(Tn, Gu, Pu0, Pu)]
            else:
                T = Tn + Gu @ (Pu0 - Pu)

            print(f'After Iteration Stuff: {time()-t0} s')
            t0 = time()

        # Bayes Likelihoods:
        Ui, W, C = Bayes_likelihoods(sammy_inp.resonance_ladder, particle_pair, WigLL, PTLL)

        print(f'After Likelihood Stuff: {time()-t0} s')
        t0 = time()

        # Running Bayes:
        Ppu, Mpu = Bayes_conditioner(sammy_rto, Pu0, Mu, Gu, V, D, T, P_vary, Ui, W, C, scheme=sammy_rto.bayes_scheme)

        print(f'After Bayes Run: {time()-t0} s')
        t0 = time()

        # Converting from U to P parameters:
        par = sammy_inp.resonance_ladder
        J_IDs = par['J_ID'].to_numpy(dtype=int)
        Ls    = par['L'   ].to_numpy(dtype=int)
        Ppu = Ppu.reshape(-1,3)
        Eu  = Ppu[:,0]
        Ggu = Ppu[:,1]
        Gnu = Ppu[:,2]
        Ppp = np.zeros_like(Ppu)
        Ppp[:,0] = u2p_E(Eu)
        Ppp[:,1] = u2p_g(Ggu)
        Ppp[:,2] = u2p_n(Gnu, Ppp[:,0], Ls[J_IDs-1], particle_pair)
        # filling in posterior parameter dataframe:
        par_post = pd.DataFrame(Ppp, columns=['E', 'Gg', 'Gn1'])
        par_post['J_ID']    = par['J_ID']
        par_post['L']       = par['L']
        par_post['Jpi']     = par['Jpi']
        par_post['varyE']   = par['varyE']
        par_post['varyGn1'] = par['varyGn1']
        par_post['varyGg']  = par['varyGg']
        sammy_inp.resonance_ladder = par_post

        print(f'After u-to-p conversion: {time()-t0} s')
        t0 = time()
    return par_post, Mpu

def Bayes_prior_matrices(sammy_inp:Union[SammyInputData,SammyInputDataYW],
                         sammy_rto:SammyRunTimeOptions,
                         RPCM:np.ndarray=None):
    """
    ...
    """

    YW = isinstance(sammy_inp, SammyInputDataYW)

    # Parameters:
    par = sammy_inp.resonance_ladder
    particle_pair = sammy_inp.particle_pair
    E    = par['E'].to_numpy()
    Gg   = par['Gg'].to_numpy()
    Gn   = par['Gn1'].to_numpy()
    J_ID = par['J_ID'].to_numpy(dtype=int)
    L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    for jpi, mean_params in particle_pair.spin_groups.items():
        jid = mean_params['J_ID']
        L[jid-1] = mean_params['Ls'][0]
    # converting to U-parameters
    ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    Pu = np.column_stack((ue, ug, un)).reshape(-1,)
    
    # Pointwise Data:
    if YW:
        D = [dataset.pw_reduced.exp.to_numpy() for dataset in sammy_inp.datasets]
    else:
        D = sammy_inp.experimental_data.exp.to_numpy()
    # Covariance:
    if sammy_inp.experimental_covariance is None:
        if YW:
            D_errs = [dataset.pw_reduced.exp_unc.to_numpy() for dataset in sammy_inp.datasets]
            V = [{'diag_stat': D_err**2} for D_err in D_errs]
        else:
            D_err = sammy_inp.experimental_data.exp_unc.to_numpy()
            V = {'diag_stat': D_err**2}
    else:
        V = sammy_inp.experimental_covariance
    
    if sammy_rto.use_least_squares:
        Mu = None
    elif RPCM is not None:
        Mp = RPCM 
        dU_dP = np.zeros_like(Pu)
        dU_dP[ ::3] = du_dp_E(E)
        dU_dP[1::3] = du_dp_g(Gg)
        dU_dP[2::3] = du_dp_n(Gn, E, L[J_ID-1], particle_pair)
        Mu = dU_dP[NA,:] * Mp * dU_dP[:,NA]
    else: # use fudge factor
        fudge = sammy_inp.initial_parameter_uncertainty
        warnings.warn('Only least squares matches SAMMY. An approximation is used with prior covariances.')
        dPp = np.zeros((len(E),3))
        dPp[:,0] = 0.5 * 0.001 * (Gg + Gn) # FIXME: THIS IS SUPPOSED TO BE A MULTIPLE OF THE BROADENNED WIDTH!!!
        dPp[:,1] = fudge * Gg
        dPp[:,2] = fudge * Gn
        dPu = np.zeros((len(E),3))
        dPu[:,0] = du_dp_E(E ) * dPp[:,0]
        dPu[:,1] = du_dp_g(Gg) * dPp[:,1]
        dPu[:,2] = du_dp_n(Gn, E, L[J_ID-1], particle_pair) * dPp[:,2]
        dPu = dPu.reshape(-1,)
        Mu = dPu**2
    return Mu, V, D

def Bayes_matrices(sammy_out:SammyOutputData,
                   particle_pair:Particle_Pair,
                   sammy_rto:SammyRunTimeOptions):
    """
    ...
    """

    # Parameters:
    par = sammy_out.par
    E    = par['E'].to_numpy()
    Gg   = par['Gg'].to_numpy()
    Gn   = par['Gn1'].to_numpy()
    J_ID = par['J_ID'].to_numpy(dtype=int)
    L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    for jpi, mean_params in particle_pair.spin_groups.items():
        jid = mean_params['J_ID']
        L[jid-1] = mean_params['Ls'][0]
    # converting to U-parameters
    ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    Pu = np.column_stack((ue, ug, un)).reshape(-1,)

    # Derivatives:
    Gu = sammy_out.derivatives # FIXME: YW DERIVATIVES?

    # Parameter vary conditions:
    P_vary = np.zeros((3*len(par),), dtype=bool)
    P_vary[ ::3] = par['varyE'  ].to_numpy()
    P_vary[1::3] = par['varyGg' ].to_numpy()
    P_vary[2::3] = par['varyGn1'].to_numpy()
    
    # Pointwise Data:
    pw = sammy_out.pw
    if type(pw) == list: # YW scheme
        T = [pw_set['theo_trans'].to_numpy() for pw_set in pw]
    else:
        T = pw['theo_trans'].to_numpy()
    
    return Pu, Gu, T, P_vary

def Bayes_likelihoods(resonance_ladder:pd.DataFrame, particle_pair:Particle_Pair, WigLL:bool=True, PTLL:bool=True):
    """
    ...
    """

    resonance_ladder = copy(resonance_ladder).sort_values(by=['J_ID','E'], ignore_index=True)
    E     = resonance_ladder['E'   ].to_numpy()
    # Gn    = resonance_ladder['Gn1' ].to_numpy()
    # Gg    = resonance_ladder['Gg'  ].to_numpy()
    J_IDs = resonance_ladder['J_ID'].to_numpy(dtype=int)
    # Ls    = resonance_ladder['L'   ].to_numpy(dtype=int)

    # Number of resonances and spingroups:
    Nres  = len(E)
    N_sgs = len(np.unique(J_IDs))
    
    # Finding first occurrences:
    first_indices = np.zeros((N_sgs+1,), dtype=int)
    first_indices[-1] = Nres
    foo = 0
    bar = -1
    for idx, jid in enumerate(J_IDs):
        if jid > bar:
            bar = jid
            first_indices[foo] = idx
            foo += 1
        elif jid < bar:
            raise RuntimeError('J_IDs are not ordered!')
    
    # Gathering mean parameters:
    Jpis  = np.unique(resonance_ladder['Jpi'].to_numpy())
    gn2ms = np.zeros((N_sgs,))
    gg2ms = np.zeros((N_sgs,))
    Dms   = np.zeros((N_sgs,))
    for sg_idx, jpi in enumerate(Jpis):
        mean_parameters = particle_pair.spin_groups[jpi]
        gn2ms[sg_idx] = mean_parameters['<gn2>']
        gg2ms[sg_idx] = mean_parameters['<gg2>']
        Dms  [sg_idx] = mean_parameters['<D>']

    # Converting from P to U parameter:
    Eu  = p2u_E(E)
    # Gnu = p2u_n(Gn, E, Ls)
    # Ggu = p2u_g(Gg)
    
    if WigLL:
        W  = np.zeros((3*Nres,3*Nres)) # Wigner matrix
        C  = np.zeros((3*Nres,))     # Wigner offset vector
    if PTLL:
        Ui = np.zeros((3*Nres,))     # Width distribution
    for sg_idx in range(N_sgs):
        if WigLL:
            indices = slice(first_indices[sg_idx], first_indices[sg_idx+1])
            Du = np.diff(Eu[indices])
            Dm = Dms[sg_idx]
            dEu1 =   pi*Du**3 / Dm**2 - 2/Du
            dEu2 = 3*pi*Du**2 / Dm**2 + 2/Du**2
            for idx, i in enumerate(range(first_indices[sg_idx], first_indices[sg_idx+1]-1)):
                W[3*i  ,3*i  ] += dEu2[idx]
                W[3*i+1,3*i+1] += dEu2[idx]
                W[3*i  ,3*i+1] -= dEu2[idx]
                W[3*i+1,3*i  ] -= dEu2[idx]
            indices_m1 = slice(3*(first_indices[sg_idx]  ), 3*(first_indices[sg_idx+1]-1))
            indices_p1 = slice(3*(first_indices[sg_idx]+1), 3*(first_indices[sg_idx+1]  ))
            C[indices_p1] += 2*dEu1
            C[indices_m1] -= 2*dEu1
        else:
            W = None;    C = None
        if PTLL:
            indices = slice(3*first_indices[sg_idx]+2, 3*first_indices[sg_idx+1]+2)
            Ui[indices] = 1.0 / gn2ms[sg_idx]
        else:
            Ui = None
    return Ui, W, C

def Bayes_conditioner(sammy_rto:SammyRunTimeOptions, P, M, G, V, D, T, P_vary, Ui=None, W=None, C=None, scheme:str='MW'):
    """
    ...
    """

    YW = isinstance(D, list)

    # Selecting only used parameters:
    used_cols = np.nonzero(P_vary)[0]
    if YW:      Gc = [Gj[:,used_cols] for Gj in G]
    else:       Gc = G[:,used_cols]
    Pc = P[used_cols]
    if M is None:       Mc = None
    elif M.ndim == 1:   Mc = M[used_cols]
    else:               Mc = M[used_cols,:][:,used_cols]
    if Ui is None:      Uic = None
    else:               Uic = Ui[used_cols]
    if W is None:       Wc = None
    else:               Wc = W[used_cols,:][:,used_cols] # FIXME: THIS DOES NOT SEEM RIGHT TO ME!
    if C is None:       Cc = None
    else:               Cc = C[used_cols]

    # Calling Bayes:
    if sammy_rto.use_least_squares and (scheme != 'MW'):
        print('Using MW scheme for least squares bayes solve.')
        scheme = 'MW'
    if   scheme == 'MW':
        bayes_func = Bayes_ext_MW
    elif scheme == 'IQ':
        bayes_func = Bayes_ext_IQ
    elif scheme == 'NV':
        bayes_func = Bayes_ext_NV
    else:
        raise ValueError(f'Unknown Bayes scheme, {scheme}.')
    Ppc, Mpc = bayes_func(Pc, Mc, Gc, V, D, T, Uic, Wc, Cc)
    
    # Restoring unprocessed parameters:
    Pp = P
    Pp[used_cols] = Ppc
    N_pars = G.shape[1]
    Mp = np.zeros((N_pars,N_pars))
    Mp[used_cols,:][:,used_cols] = Mpc
    return Pp, Mp

def Bayes_YW(G, V, D, T):
    """
    ...

    Source: SAMMY Manual (section IV.E.1)
    """
    YW = isinstance(D, list)
    if YW:
        N_pars = G[0].shape[1]
        W = np.zeros((N_pars,N_pars))
        Y = np.zeros((N_pars,))
        for Gj, Vj, Dj, Tj in zip(G, V, D, T):
            ViG = utils.implicit_data_cov_solve(Vj, Gj)
            W += Gj.T @ ViG
            Y += ViG.T @ (Dj - Tj)
    else:
        ViG = utils.implicit_data_cov_solve(V, G)
        W = G.T @ ViG
        Y = ViG.T @ (D - T)
    return Y, W

def Bayes_ext_MW(P, M, G, V, D, T, PTVar=None, WigCov=None, WigVec=None):
    """
    The MW Bayes Scheme for SAMMY. This method is most useful when the use_least_squares method is
    used (i.e. M is None). 

    Source: SAMMY Manual (section IV.B.3)
    """

    Y, W = Bayes_YW(G, V, D, T)
    Mpi = W
    if M is not None:
        M_diag = (M.ndim == 1)
        if M_diag:  Mpi += np.diag(1.0/M) # np.fill_diagonal(MiV, MiV.diagonal() + 1.0/M)
        else:       Mpi += np.linalg.inv(M)
    if PTVar is not None:
        Mpi += np.diag(PTVar) # np.fill_diagonal(MiV, MiV.diagonal() + PTVar)
    if WigCov is not None:
        Mpi += WigCov
    Mp = np.linalg.inv(Mpi)
    del Mpi

    # Log Likelihood Components:
    if PTVar is not None:
        Y += PTVar * P
    if WigVec is not None:
        Y += WigVec

    # Finding Posterior Parameters:
    Pp = P + Mp @ Y
    return Pp, Mp

def Bayes_ext_IQ(P, M, G, V, D, T, PTVar=None, WigCov=None, WigVec=None):
    """
    The IQ Bayes Scheme for SAMMY.

    Source: SAMMY Manual (section IV.B.2)
    """
    
    Y, W = Bayes_YW(G, V, D, T)
    MiQ = W
    if PTVar is not None:
        MiQ += np.diag(PTVar) # np.fill_diagonal(MiQ, MiQ.diagonal() + PTVar)
    if WigCov is not None:
        MiQ += WigCov

    M_diag = (M.ndim == 1)
    if M_diag:  Q = MiQ * M[:,NA]
    else:       Q = M @ MiQ
    del MiQ

    I = np.eye(*M.shape)
    if M_diag:  Mp = np.linalg.solve(I+Q, np.diag(M))
    else:       Mp = np.linalg.solve(I+Q, M)
    del I, Q

    # Log Likelihood Components:
    if PTVar is not None:
        Y += PTVar * P
    if WigVec is not None:
        Y += WigVec

    # Finding Posterior Parameters:
    Pp = P + Mp @ Y
    return Pp, Mp

def Bayes_ext_NV(P, M, G, V, D, T, PTVar=None, WigCov=None, WigVec=None):
    """
    The NV Bayes Scheme for SAMMY.

    Source: SAMMY Manual (section IV.B.1)
    """

    raise NotImplementedError('NV scheme for extended Bayes has not been implemented yet.')
    return Pp, Mp