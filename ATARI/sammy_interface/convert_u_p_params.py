import numpy as np
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
    P = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)
    return 2000*P*u_n*abs(u_n)
def p2u_n(p_n, E, l, particle_pair:Particle_Pair):
    P = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)
    return np.sign(p_n)*np.sqrt(abs(p_n)/(2000 * P))
def u2p_g(u_g):
    return 2000*u_g*abs(u_g)
def p2u_g(p_g):
    return np.sign(p_g)*np.sqrt(abs(p_g)/2000)

# Derivatives:
def du_dp_E(p_E):
    return 0.5 / np.sqrt(p_E)
def du_dp_n(p_n, E, l, particle_pair:Particle_Pair):
    P = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, l)
    return 0.5 / np.sqrt(2000 * P * p_n)
def du_dp_g(p_g):
    return 0.5 / np.sqrt(2000 * p_g)