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
    _, P, _, _ = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)
    return 2000*P[-1,:]*u_n*abs(u_n)
def p2u_n(p_n, E, l, particle_pair:Particle_Pair):
    _, P, _, _ = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)
    return np.sign(p_n)*np.sqrt(abs(p_n)/(2000 * P[-1,:]))
def u2p_g(u_g):
    return 2000*u_g*abs(u_g)
def p2u_g(p_g):
    return np.sign(p_g)*np.sqrt(abs(p_g)/2000)

# Derivatives:
def du_dp_E(p_E):
    return 0.5 / np.sqrt(p_E)
def du_dp_n(p_n, E, l, particle_pair:Particle_Pair):
    _, P, _, _ = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)
    return np.sign(p_n) * 0.5 / np.sqrt(2000 * P[-1,:] * abs(p_n))
def du_dp_g(p_g):
    return np.sign(p_g) * 0.5 / np.sqrt(2000 * abs(p_g))

# Conversions with respect to gn2 and gg2:
def u2g2(u):
    return 1000*u*abs(u)
def g22u(g2):
    return np.sign(g2) * np.sqrt(1e-3 * abs(g2))
def du_dg2(g2):
    return np.sign(g2) * 0.5/np.sqrt(1000 * abs(g2))

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
        # du_dp[i+1, i+1] = 0.5 / np.sqrt(2 * row['Gg'] * 1e-3)
        du_dp[par_idx+1] = du_dg2(row['gg2'])
        dp_names.append(f'df/dGg_{res_idx}')

        # du_n/dGnx
        du_dp[par_idx+2] = du_dg2(row['gn2'])
        dp_names.append(f'df/dGn_{res_idx}')

    # Convert df/du to df/dp
    df_dp = df_du * du_dp[np.newaxis, :]

    return df_dp, du_dp, dp_names