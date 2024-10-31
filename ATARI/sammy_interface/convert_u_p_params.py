import numpy as np
import pandas as pd

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.theory.scattering_params import FofE_recursive

__doc__ = """
This file contains functions that converts u-parameters to p-parameters.
"""

def u2p_E(u_E):
    return u_E * abs(u_E)
def p2u_E(p_E):
    return np.sign(p_E)*np.sqrt(abs(p_E))
def u2p_n(u_n, E, l, particle_pair:Particle_Pair):
    if isinstance(l, int):      lmax = l
    else:                       lmax = max(l)
    _, P, _, _ = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lmax)
    return 2e3*P[l,range(len(E))]*u_n*abs(u_n)
    # gn2 = 1e3 * u_n*abs(u_n)
    # return particle_pair.gn2_to_Gn(gn2, E, l)
def p2u_n(p_n, E, l, particle_pair:Particle_Pair):
    if isinstance(l, int):      lmax = l
    else:                       lmax = max(l)
    _, P, _, _ = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lmax)
    return np.sign(p_n)*np.sqrt(abs(p_n) / (2e3 * P[l,range(len(E))]))
    # gn2 = particle_pair.Gn_to_gn2(p_n, E, l)
    # return np.sign(gn2)*np.sqrt(abs(gn2) / 1e3)
def u2p_g(u_g):
    return 2e3*u_g*abs(u_g)
    # gg2 = 1e3 * u_g*abs(u_g)
    # return particle_pair.gg2_to_Gg(gg2)
def p2u_g(p_g):
    return np.sign(p_g) * np.sqrt(abs(p_g) / 2e3)
    # gg2 = particle_pair.Gg_to_gg2(p_g)
    # return np.sign(gg2) * np.sqrt(abs(gg2) / 1e3)

# Derivatives:
def du_dp_E(p_E):
    return 0.5 / np.sqrt(abs(p_E))
def du_dp_n(p_n, E, l, particle_pair:Particle_Pair):
    if isinstance(l, int):      lmax = l
    else:                       lmax = max(l)
    _, P, _, _ = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lmax)
    return 0.5 / np.sqrt(2e3 * P[l,range(len(E))] * abs(p_n))
def du_dp_g(p_g):
    return 0.5 / np.sqrt(2e3 * abs(p_g))

# Conversions with respect to gn2 and gg2:
def u2g2(u):
    return 1e3*u*abs(u)
def g22u(g2):
    return np.sign(g2) * np.sqrt(abs(g2) / 1e3)
def du_dg2(g2):
    return 0.5 / np.sqrt(1e3 * abs(g2))


# Conversion from a resonance ladder
def get_Pu_vec_from_reslad(resonance_ladder, particle_pair):

    E  = resonance_ladder['E'].to_numpy()
    Gg = resonance_ladder['Gg'].to_numpy()
    Gn = resonance_ladder['Gn1'].to_numpy()
    J_ID = resonance_ladder['J_ID'].to_numpy(dtype=int)

    L = np.zeros((len(particle_pair.spin_groups),), dtype=int)
    for jpi, mean_params in particle_pair.spin_groups.items():
        jid = mean_params['J_ID']
        L[jid-1] = mean_params['Ls'][0]

    ue = p2u_E(E)
    ug = p2u_g(Gg)
    un = p2u_n(Gn, E, L[J_ID-1], particle_pair)
    return np.column_stack((ue, ug, un)).reshape(-1,)


def convert_deriv_du2dp(df_du: np.array, 
                        ladder_df: pd.DataFrame):
    """
    Converts derivatives df/du -> df/dp,
    
    1. assuming next order of u: 
        u_e, u_g, u_n -> p_e, p_g, p_n

    2. ladder ordered by E asc with the E, Gg, Gnx inside

    """

    # print('Shape of input derivatives df/du: ', df_du.shape)

    N_par = df_du.shape[1]
    N_res = N_par // 3
    # print('Npar = ', N_par)

    # Initialize the du_dp matrix
    du_dp = np.zeros((N_par,))

    dp_names = []

    # Iterate for every three parameters
    for res_idx in range(N_res):
        par_idx = 3*res_idx
        # Extract the corresponding rows from ladder_df
        row_num = res_idx

        row = ladder_df.iloc[row_num]

        # du_e/dE
        du_dp[par_idx] = du_dp_E(row['E'])

        dp_names.append(f'df/dE_{res_idx}')

        # du_g/dGg
        du_dp[par_idx+1] = du_dg2(row['gg2'])
        dp_names.append(f'df/dGg_{res_idx}')

        # du_n/dGnx
        du_dp[par_idx+2] = du_dg2(row['gn2'])
        dp_names.append(f'df/dGn_{res_idx}')

    # Convert df/du to df/dp
    df_dp = df_du * du_dp[np.newaxis, :]

    return df_dp, du_dp, dp_names