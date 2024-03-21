import numpy as np
import pandas as pd
from numpy import newaxis as NA
import scipy.sparse as sp
import warnings

from ATARI.sammy_interface.convert_u_p_params import p2u_E, p2u_g, p2u_n,   u2p_E, u2p_g, u2p_n,   du_dp_E, du_dp_g, du_dp_n
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyOutputData

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

    if M is None:   MiW = W
    else:           MiW = np.linalg.inv(M) + W
    Mp = np.linalg.inv(MiW)
    
    if V_diag:  VDT = (D - T) / V
    else:       VDT = np.linalg.solve(V, (D-T))
    
    # Only updating non-zero gradient parameters:
    Pp = P
    Pp[used_cols] += (Mp @ G.T @ VDT)
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
    indices_n = 3*par.index[par['varyGn1'] == 0].to_numpy() + 1
    indices_g = 3*par.index[par['varyGg'] == 0].to_numpy()  + 2
    Gu[indices_e,:] = 0.0
    Gu[indices_n,:] = 0.0
    Gu[indices_g,:] = 0.0
    
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
    Pu = Pu.reshape(-1,3)
    Eu  = Pu[:,0]
    Ggu = Pu[:,1]
    Gnu = Pu[:,2]
    Pp = np.zeros_like(Pu)
    Pp[:,0] = u2p_E(Eu)
    Pp[:,1] = u2p_g(Ggu)
    Pp[:,2] = u2p_n(Gnu, Pp[:,0], L[J_ID-1], particle_pair)
    
    par_post = pd.DataFrame(Pp, columns=['E', 'Gg', 'Gn1'])
    par_post['J_ID']    = par['J_ID']
    par_post['varyE']   = par['varyE']
    par_post['varyGn1'] = par['varyGn1']
    par_post['varyGg']  = par['varyGg']
    par_post = particle_pair.expand_ladder(par_post)
    return par_post, Mpu

# =================================================================================================
#   Incomplete Functions
# =================================================================================================

def BayesIter(P, M, V, D, func, num_iterations:int=1, model='NV'):
    """
    ...
    """

    if M is None:
        model = 'MW'
    if   model == 'IQ':     Bayes = BayesIQ
    elif model == 'NV':     Bayes = BayesNV
    elif model == 'MW':     Bayes = BayesMW
    else:                   raise NotImplementedError('...')
    Pn = P;  Mn = M
    for iter in range(num_iterations):
        Tn, Gn = func(Pn)
        T = Tn - Gn @ (P - Pn)
        Pn, Mn = Bayes(P, M, Gn, V, D, T)
    return Pn, Mn