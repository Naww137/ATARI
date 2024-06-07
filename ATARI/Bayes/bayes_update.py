from typing import Union, List
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
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.sammy_interface.sammy_deriv import get_derivatives, get_derivatives_YW
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
#   New Bayes Implementation
# =================================================================================================

def Bayes_extended(sammy_inp:Union[SammyInputData,SammyInputDataYW],
                   sammy_rto:SammyRunTimeOptions,
                   RPCM:np.ndarray=None,
                   WigLL:bool=True, PTLL:bool=True):
    """
    The new Bayes implementation outside of SAMMY. This includes Wigner and Porter-Thomas
    log-likelihoods when specified.

    Parameters
    ----------
    sammy_inp : SammyInputData or SammyInputDataYW
        The SAMMY input object.
    sammy_rto : SammyRunTimeOptions
        The SAMMY runtime options.
    RPCM : array, optional
        The optional resonance parameter covariance matrix. If not specified,
        `initial_parameter_uncertainty` is used. If `use_least_squares` is on, the RPCM is ignored.
    WigLL : bool
        If true, Wigner log-likelihoods are included. Default is True.
    PTLL : bool
        If true, Porter-Thoas log-likelihoods are included. Default is True.

    Returns
    -------
    par_post : DataFrame
        The optimized resonance parameters
    Mpu : array
        The outgoing resonance parameter covariance matrix.
    chi2 : float
        The outgoing chi2 value(s).
    """

    particle_pair = sammy_inp.particle_pair
    YW = isinstance(sammy_inp, SammyInputDataYW)
    if YW:      exp = sammy_inp.experiments
    else:       exp = sammy_inp.experiment
    
    # Prior matrices:
    Mu, V, D = Bayes_prior_matrices(sammy_inp, sammy_rto, RPCM)

    # Iterating Bayes:
    for iter in range(sammy_rto.iterations):
        # Getting derivatives:
        if YW:  sammy_out = get_derivatives_YW(sammy_inp, sammy_rto, u_or_p='u')
        else:   sammy_out = get_derivatives(sammy_inp, sammy_rto, u_or_p='u')

        # Gathering the rest of the matrices:
        Pu, Gu, Tn, P_vary = Bayes_matrices(sammy_out, particle_pair, exp, sammy_rto)

        # Updating prior fit for iteration scheme:
        if iter == 0:
            # sammy_out0 = sammy_out
            Pu0 = Pu
            T = Tn
        else:
            T = update_theo_matrix(Tn, Gu, Pu0, Pu, YW)
        find_Mp = (iter == sammy_rto.iterations - 1) # only find posterior matrix for the last iteration

        # Bayes Likelihoods:
        Ui, W, C = Bayes_likelihoods(sammy_inp.resonance_ladder, particle_pair, sammy_rto, WigLL, PTLL)

        # Running Bayes:
        Ppu, Mpu, chi2 = Bayes_conditioner(sammy_rto, Pu0, Mu, Gu, V, D, T, P_vary, Ui, W, C, scheme=sammy_rto.bayes_scheme, find_Mp=find_Mp)

        # Converting from U to P parameters:
        par = sammy_inp.resonance_ladder
        J_IDs = par['J_ID'].to_numpy(dtype=int)
        Ls    = par['L'   ].to_numpy(dtype=int)
        Ppu = Ppu.reshape(-1,3)
        Eu  = Ppu[:,0]
        Ggu = Ppu[:,1]
        Gnu = Ppu[:,2]
        Ppp = np.zeros_like(Ppu)
        Ppp[:,0] = Eu # Ppp[:,0] = u2p_E(Eu)
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

    # sammy_out = SammyOutputData(pw=sammy_out0.pw, par=sammy_out0.par, chi2=sammy_out0.chi2, chi2n=sammy_out0.chi2n,
    #                             par_post=par_post)

    return par_post, Mpu, chi2

def Bayes_prior_matrices(sammy_inp:Union[SammyInputData,SammyInputDataYW],
                         sammy_rto:SammyRunTimeOptions,
                         RPCM:np.ndarray=None):
    """
    Finds the Bayes matrices that are consistent between iterations.

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
    # ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    Pu = np.column_stack((E, ug, un)).reshape(-1,)
    
    # Pointwise Data and Covariance:
    if YW:
        D = [];     V = []
        for dataset, experimental_covariance in zip(sammy_inp.datasets, sammy_inp.experimental_covariance):
            D.append(dataset.exp.to_numpy())
            if experimental_covariance:
                V.append(experimental_covariance)
            else:
                D_err = dataset.exp_unc.to_numpy()
                V.append({'diag_stat': pd.DataFrame({'var_stat':D_err**2}, index=dataset.E)})
    else:
        D = sammy_inp.experimental_data.exp.to_numpy()
        if sammy_inp.experimental_covariance:
            V = sammy_inp.experimental_covariance
        else:
            D_err = sammy_inp.experimental_data.exp_unc.to_numpy()
            V = {'diag_stat': pd.DataFrame({'var_stat':D_err**2}, index=sammy_inp.experimental_data.E)}
    
    # Resonance Parameter Covariance Matrix:
    if sammy_rto.use_least_squares:
        Mu = None
    elif RPCM is not None:
        Mp = RPCM 
        dU_dP = np.zeros_like(Pu)
        dU_dP[ ::3] = 1.0 #du_dp_E(E)
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
        # dPu[:,0] = du_dp_E(E ) * dPp[:,0]
        dPu[:,0] = dPp[:,0]
        dPu[:,1] = du_dp_g(Gg) * dPp[:,1]
        dPu[:,2] = du_dp_n(Gn, E, L[J_ID-1], particle_pair) * dPp[:,2]
        dPu = dPu.reshape(-1,)
        Mu = dPu**2
    return Mu, V, D

# Using P-parameter Energies:
def Bayes_matrices(sammy_out:SammyOutputData,
                   particle_pair:Particle_Pair,
                   experiment:Union[Experimental_Model,List[Experimental_Model]],
                   sammy_rto:SammyRunTimeOptions):
    """
    Finds the Bayes matrices that change between iterations.

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
    # ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    Pu = np.column_stack((E, ug, un)).reshape(-1,)

    # Derivatives:
    Gu = sammy_out.derivatives
    # Converting u-parameter energies into p-parameter energies
    if isinstance(Gu, list):
        for Gu_ in Gu:
            Gu_[:,::3] *= du_dp_E(E)
    else:
        Gu[:,::3] *= du_dp_E(E)

    # Parameter vary conditions:
    P_vary = np.zeros((3*len(par),), dtype=bool)
    P_vary[ ::3] = par['varyE'  ].to_numpy()
    P_vary[1::3] = par['varyGg' ].to_numpy()
    P_vary[2::3] = par['varyGn1'].to_numpy()
    
    # Pointwise Data:
    pw = sammy_out.pw
    if type(pw) == list: # YW scheme
        T = []
        for pw_set, exp in zip(pw, experiment):
            if exp.reaction == "transmission":
                T.append(pw_set['theo_trans'].to_numpy())
            else:
                T.append(pw_set['theo_xs'].to_numpy())
    else:
        if experiment.reaction == "transmission":
            T = pw['theo_trans'].to_numpy()
        else:
            T = pw['theo_xs'].to_numpy()
    
    return Pu, Gu, T, P_vary

def update_theo_matrix(Tn, Gn, P0, Pn, YW:bool):
    """
    Updates the theoretical matrix, `T`, for implicit iteration.

    Source: SAMMY Manual (section IV.A.3)
    """

    if YW:
        T = []
        for Tnj, Gnj in zip(Tn, Gn):
            T.append(Tnj + Gnj @ (P0 - Pn))
    else:
        T = Tn + Gn @ (P0 - Pn)
    return T

# P-parameter Energies:
def Bayes_likelihoods(resonance_ladder:pd.DataFrame, particle_pair:Particle_Pair, sammy_rto:SammyRunTimeOptions, WigLL:bool=True, PTLL:bool=True):
    """
    Finds the Wigner and Porter-Thomas Log-likelihood matrices, if desired.
    
    ...
    """

    # energy_window = sammy_rto.energy_window
    energy_window = particle_pair.energy_range
    resonance_ladder = copy(resonance_ladder).sort_values(by=['J_ID','E'], ignore_index=True)
    Ep    = resonance_ladder['E'   ].to_numpy()
    # Gn    = resonance_ladder['Gn1' ].to_numpy()
    # Gg    = resonance_ladder['Gg'  ].to_numpy()
    J_IDs = resonance_ladder['J_ID'].to_numpy(dtype=int)
    # Ls    = resonance_ladder['L'   ].to_numpy(dtype=int)

    # Number of resonances and spingroups:
    Nres  = len(Ep)
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
    # Eu = p2u_E(Ep)
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
            Ep_g = Ep[indices]
            Dp_g = np.diff(Ep_g)
            Dm = Dms[sg_idx]
            coef = pi/Dm**2
            dEu1  = 1.0/Dp_g - coef*Dp_g
            dEu2  = -dEu1
            dEu11 = 1.0/Dp_g**2 + coef
            dEu22 = dEu11
            dEu12 = -dEu11
            dEu21 = dEu12
            for idx, i in enumerate(range(first_indices[sg_idx], first_indices[sg_idx+1]-1)):
                i1 = 3*i; i2 = 3*(i+1)
                W[i1,i1] += 2*dEu11[idx]
                W[i2,i2] += 2*dEu22[idx]
                W[i1,i2] += 2*dEu12[idx]
                W[i2,i1] += 2*dEu21[idx]
                C[i1] += 2*dEu1[idx]
                C[i2] += 2*dEu2[idx]
            # Edge cases:
            if energy_window is not None:
                F1dEu1  = coef*(Ep_g[0]  - energy_window[0] )
                F1dEu11 = coef
                F1dEu2  = coef*(Ep_g[-1] - energy_window[-1])
                F1dEu22 = coef
                iL = 3*(first_indices[sg_idx  ]  )
                iR = 3*(first_indices[sg_idx+1]-1)
                W[iL,iL] += 2*F1dEu11
                W[iR,iR] += 2*F1dEu22
                C[iL] += 2*F1dEu1
                C[iR] += 2*F1dEu2
        else:
            W = None
            C = None
        if PTLL:
            indices = slice(3*first_indices[sg_idx]+2, 3*first_indices[sg_idx+1]+2, 3)
            Ui[indices] = 1.0 / gn2ms[sg_idx]
        else:
            Ui = None
    return Ui, W, C

def Bayes_conditioner(sammy_rto:SammyRunTimeOptions, P, M, G, V, D, T, P_vary, Ui=None, W=None, C=None, scheme:str='MW', find_Mp:bool=True):
    """
    Conditions the Bayes matrices by only solving for the parameters that change.

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
    if   scheme == 'MW':    bayes_func = Bayes_ext_MW
    elif scheme == 'IQ':    bayes_func = Bayes_ext_IQ
    elif scheme == 'NV':    bayes_func = Bayes_ext_NV
    else:                   raise ValueError(f'Unknown Bayes scheme, {scheme}.')
    Ppc, Mpc, chi2 = bayes_func(Pc, Mc, Gc, V, D, T, Uic, Wc, Cc, find_Mp=find_Mp)
    
    # Restoring unprocessed parameters:
    Pp = P
    Pp[used_cols] = Ppc
    if YW:      N_pars = G[0].shape[1]
    else:       N_pars = G.shape[1]
    Mp = np.zeros((N_pars,N_pars))
    Mp[used_cols,:][:,used_cols] = Mpc
    return Pp, Mp, chi2

def Bayes_YW(G, V, D, T):
    """
    Provides the Y and W matrices for the SAMMY Bayes equations.

    Source: SAMMY Manual (section IV.E.1)
    """
    YW = isinstance(D, list)
    if YW:
        N_pars = G[0].shape[1]
        W = np.zeros((N_pars,N_pars))
        Y = np.zeros((N_pars,))
        chi2 = []
        for Gj, Vj, Dj, Tj in zip(G, V, D, T):
            Yj, Wj, chi2j = utils.YW_implicit_data_cov_solve(Vj, Gj, Dj-Tj)
            Y    += Yj
            W    += Wj
            chi2.append(chi2j)
    else:
        Y, W, chi2 = utils.YW_implicit_data_cov_solve(V, G, D-T)
    return Y, W, chi2

def Bayes_ext_MW(P, M, G, V, D, T, PTVar=None, WigCov=None, WigVec=None, find_Mp:bool=True):
    """
    The MW Bayes Scheme for SAMMY. This method is most useful when the use_least_squares method is
    used (i.e. M is None). 

    Source: SAMMY Manual (section IV.B.3)
    """

    # Getting YW parameters:
    Y, W, chi2 = Bayes_YW(G, V, D, T)

    # Compiling Mp inverse:
    Mpi = W
    if M is not None:
        M_diag = (M.ndim == 1)
        if M_diag:  Mpi += np.diag(1.0/M) # np.fill_diagonal(Mpi, Mpi.diagonal() + 1.0/M)
        else:       Mpi += utils.psd_inv(M)
    if PTVar is not None:
        Mpi += np.diag(PTVar) # np.fill_diagonal(Mpi, Mpi.diagonal() + PTVar)
    if WigCov is not None:
        Mpi += WigCov
        # print(WigCov[:5,:5])
        # print(np.linalg.inv(WigCov)[:5,:5])
    # print((np.diag(PTVar)/Mpi)[:4,:4])
    # print(W[:4,:4])
    # print(PTVar[:4])
    # print()

    # Log Likelihood Components of Y:
    if PTVar is not None:
        Y -= 2 * PTVar * P
    if WigVec is not None:
        Y -= WigVec
    # print((2*PTVar * P/Y)[:4])

    # Finding Posterior Parameters:
    if find_Mp:
        Mp = utils.psd_inv(Mpi)
        Pp = P + Mp @ Y
    else:
        Pp = P + utils.psd_solve(Mpi, Y)
        Mp = None
    return Pp, Mp, chi2

def Bayes_ext_IQ(P, M, G, V, D, T, PTVar=None, WigCov=None, WigVec=None, find_Mp:bool=True):
    """
    The IQ Bayes Scheme for SAMMY.

    Source: SAMMY Manual (section IV.B.2)
    """
    
    # Getting YW parameters:
    Y, W, chi2 = Bayes_YW(G, V, D, T)

    # Compiling Mp inverse:
    MiQ = W
    if PTVar is not None:
        MiQ += np.diag(PTVar) # np.fill_diagonal(MiQ, MiQ.diagonal() + PTVar)
    if WigCov is not None:
        MiQ += WigCov
    # finding Q:
    M_diag = (M.ndim == 1)
    if M_diag:  Q = MiQ * M[:,NA]
    else:       Q = M @ MiQ
    del MiQ

    # Log Likelihood Components:
    if PTVar is not None:
        Y += 2*PTVar * P
    if WigVec is not None:
        Y += WigVec

    # Finding Posterior Parameters:
    I = np.eye(*Q.shape)
    if find_Mp:
        if M_diag:  Mp = utils.psd_solve(I+Q, np.diag(M))
        else:       Mp = utils.psd_solve(I+Q, M)
        Pp = P + Mp @ Y
    else:
        if M_diag:  MY = M[:,NA] * Y
        else:       MY = M @ Y
        Pp = P + utils.psd_solve(I+Q, MY)
        Mp = None
    return Pp, Mp, chi2

def Bayes_ext_NV(P, M, G, V, D, T, PTVar=None, WigCov=None, WigVec=None, find_Mp:bool=True):
    """
    The NV Bayes Scheme for SAMMY.

    Source: SAMMY Manual (section IV.B.1)
    """

    # NOTE: THIS METHOD DOES NOT SEEM TO WORK WITH LIKELIHOODS or WY SCHEME!!!
    raise NotImplementedError('NV scheme for extended Bayes has not been implemented yet.')

    return Pp, Mp, chi2

# =================================================================================================
#   Legacy Code
# =================================================================================================

# U-parameter Wigner Matrix:
#             indices = slice(first_indices[sg_idx], first_indices[sg_idx+1])
#             Eu_g = Eu[indices]
#             Ep_g = Eu_g * Eu_g
#             Dp_g = np.diff(Ep_g)
#             Dm = Dms[sg_idx]
#             coef = pi/Dm**2
#             dEu1  = -coef*Dp_g*Eu_g[:-1] + 2*Eu_g[:-1] / Dp_g
#             dEu2  =  coef*Dp_g*Eu_g[1:]  - 2/Eu_g[1:]  / Dp_g
#             dEu11 =  coef*(2*Ep_g[:-1] - Dp_g) + 2*(Ep_g[1:]+Ep_g[:-1])/Dp_g**2
#             dEu22 =  coef*(2*Ep_g[1:]  + Dp_g) + 2*(Ep_g[1:]+Ep_g[:-1])/Dp_g**2
#             dEu12 = -2*coef*Eu_g[1:]*Eu_g[:-1] + 4*Eu_g[1:]*Eu_g[:-1]/Dp_g**2
#             dEu21 = dEu12
#             for idx, i in enumerate(range(first_indices[sg_idx], first_indices[sg_idx+1]-1)):
#                 i1 = 3*i; i2 = 3*(i+1)
#                 W[i1,i1] += 2*dEu11[idx]
#                 W[i2,i2] += 2*dEu22[idx]
#                 W[i1,i2] += 2*dEu12[idx]
#                 W[i2,i1] += 2*dEu21[idx]
#                 C[i1] += 2*dEu1[idx]
#                 C[i2] += 2*dEu2[idx]
#             # Edge cases:
#             if energy_window is not None:
#                 F1dEu1  = coef*(Ep_g[0]-energy_window[0])*Eu_g[0]
#                 F1dEu11 = coef*(3*Ep_g[0]-energy_window[0])
#                 F1dEu2  = coef*(Ep_g[-1]-energy_window[-1])*Eu_g[-1]
#                 F1dEu22 = coef*(3*Ep_g[-1]-energy_window[-1])
#                 iL = 3*(first_indices[sg_idx  ]  )
#                 iR = 3*(first_indices[sg_idx+1]-1)
#                 W[iL,iL] += 2*F1dEu11
#                 W[iR,iR] += 2*F1dEu22
#                 C[iL] += 2*F1dEu1
#                 C[iR] += 2*F1dEu2