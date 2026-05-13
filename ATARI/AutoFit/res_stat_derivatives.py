from math import pi
import numpy as np
import pandas as pd

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.sammy_interface.convert_u_p_params import p2u_E, du_dp_E, p2u_n


def Wigner_1sg(E, MLS:float):
    """
    ...
    """

    Nres = len(E)
    if   Nres <= 1:
        LL   = 0.0
        jac  = np.zeros((Nres,))
        hess = np.zeros((Nres,Nres))
    else:
        dens = 1.0 / MLS
        E = np.array(E)
        sort_indices = np.argsort(E)
        E = E[sort_indices]
        # u = p2u_E(E)
        dE = np.diff(E)
        if   any(dE < 0):
            print('\nError found. Energies as follows:')
            print(E)
            raise RuntimeError('Some dE are negative when calculating Wigner likelihoods.')
        elif any(dE <= 0):
            print('\nError found. Energies as follows:')
            print(E)
            raise RuntimeError('Some dE are zero when calculating Wigner likelihoods.')
        dE_du = 1.0 / du_dp_E(E)

        LL = -(pi/4) * np.sum((dens*dE)**2) + np.sum(np.log((pi/2)*dens**2 * dE))

        # Building Jacobian:
        dLL_dE = np.zeros((Nres,))
        # dLL_dE[1:-1] =  (pi/2)*(dens**2) * (E[2:] - 2*E[1:-1] + E[:-2]) + (1/dE[:-1] - 1/dE[1:])
        dLL_dE[1:-1] =  (pi/2)*(dens**2) * (dE[1:] - dE[:-1]) + (1/dE[:-1] - 1/dE[1:])
        dLL_dE[0]    =  (pi/2)*(dens**2) * dE[0]  - 1/dE[0]
        dLL_dE[-1]   = -(pi/2)*(dens**2) * dE[-1] + 1/dE[-1]
        dLL_du = dLL_dE * dE_du
        jac = dLL_du

        # This part is weird. There is no good way to convert from E to u.
        d2LL_du2 = np.zeros((Nres,))
        d2LL_du2[1:-1] = pi*(dens**2) * (E[2:] - 6*E[1:-1] + E[:-2]) - 2 * ((E[1:-1] + E[:-2])/dE[:-1]**2 + (E[2:] + E[1:-1])/dE[1:]**2)
        d2LL_du2[0]    = pi*(dens**2) * (E[1]  - 3*E[0] ) - 2 * (E[1]  + E[0] )/dE[0]**2
        d2LL_du2[-1]   = pi*(dens**2) * (E[-2] - 3*E[-1]) - 2 * (E[-1] + E[-2])/dE[-1]**2

        d2LL_dEdF = (pi/2)*(dens**2) + 1/dE**2
        d2LL_dudv = d2LL_dEdF * dE_du[:-1] * dE_du[1:]

        # Building the hessian from double derivatives:
        hess = np.zeros((Nres,Nres))
        hess += np.diag(d2LL_du2)
        hess[1:,:-1] += np.diag(d2LL_dudv)
        hess[:-1,1:] += np.diag(d2LL_dudv)

        # Unsorting:
        jac[sort_indices] = jac
        hess[np.ix_(sort_indices,sort_indices)] = hess

    return LL, jac, hess

def Wigner_multi_sg(respar:pd.DataFrame, particle_pair:Particle_Pair):
    """
    ...
    """

    Nres = len(respar)
    Npar = 3*Nres
    LL   = 0
    jac  = np.zeros((Npar,))
    hess = np.zeros((Npar,Npar))

    spingroups = particle_pair.spin_groups
    for spingroup in spingroups.values():
        mean_lvl_spacing = spingroup['<D>']
        J_ID = spingroup['J_ID']
        respar_sg = respar[respar['J_ID'] == J_ID]
        E_sg = respar_sg['E'].values
        indices_sg = np.array(respar_sg.index)
        par_indices = 3*indices_sg
        LL_sg, jac_sg, hess_sg = Wigner_1sg(E_sg, mean_lvl_spacing)
        LL += LL_sg
        jac[par_indices]                      = jac_sg
        hess[np.ix_(par_indices,par_indices)] = hess_sg

    return LL, jac, hess

def Porter_Thomas_1sg(un, gn2m:float):
    """
    ...
    """
    # NOTE: THIS ONLY WORKS WITH DOF=1!!!

    LL = -1000*np.sum(un*un)/(2*gn2m) + 0.5*len(un)*np.log(1000/(2*np.pi*gn2m))

    dLL_du = -(1000/gn2m) * un
    jac = dLL_du

    d2LL_du2 = -(1000/gn2m)
    hess = np.eye(len(un)) * d2LL_du2

    return LL, jac, hess

def Porter_Thomas_multi_sg(respar:pd.DataFrame, particle_pair:Particle_Pair):
    """
    ...
    """

    Nres = len(respar)
    Npar = 3*Nres
    LL   = 0
    jac  = np.zeros((Npar,))
    hess = np.zeros((Npar,Npar))

    spingroups = particle_pair.spin_groups
    for spingroup in spingroups.values():
        gn2m = spingroup['<gn2>']
        L = spingroup['Ls'][0] # NOTE: We will only take the dominant spingroup
        J_ID = spingroup['J_ID']
        respar_sg = respar[respar['J_ID'] == J_ID]
        E_sg = respar_sg['E'].values
        Gn_sg = respar_sg['Gn1'].values
        un_sg = p2u_n(Gn_sg, E_sg, L, particle_pair)
        indices_sg = np.array(respar_sg.index)
        par_indices = 3*indices_sg+2
        LL_sg, jac_sg, hess_sg = Porter_Thomas_1sg(un_sg, gn2m)
        LL += LL_sg
        jac[par_indices]                      = jac_sg
        hess[np.ix_(par_indices,par_indices)] = hess_sg

    return LL, jac, hess