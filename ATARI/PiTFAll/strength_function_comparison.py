import numpy as np
import pandas as pd

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.theory.scattering_params import gstat

def strength_function_estimation(ladder:pd.DataFrame, energy_range:tuple, particle_pair:Particle_Pair):
    """
    ...
    """

    ladder_windowed = ladder[(ladder.E <= max(energy_range)) & (ladder.E >= min(energy_range))]
    dE = max(energy_range) - min(energy_range)

    Sns = {}
    for Jpi, spingroup in particle_pair.spin_groups.items():
        J_ID = spingroup['J_ID']
        l = spingroup['Ls'][0]
        gJ = gstat(abs(Jpi), particle_pair.I, particle_pair.i)
        ladder_sg = ladder_windowed[ladder_windowed['J_ID'] == J_ID]
        gn2 = particle_pair.Gn_to_gn2(ladder_sg['Gn1'], ladder_sg['E'], l)
        Sn_Jpi = gJ/((2*l+1)*dE) * np.sum(gn2)
        Sns[Jpi] = Sn_Jpi
    return Sns

def strength_function_discrepancy(ladder1:pd.DataFrame, ladder2:pd.DataFrame, energy_range:tuple, particle_pair:Particle_Pair):
    """
    ...
    """

    Sns_1 = strength_function_estimation(ladder1, energy_range, particle_pair)
    Sns_2 = strength_function_estimation(ladder2, energy_range, particle_pair)
    
    dSns = {}
    for Jpi, spingroup in particle_pair.spin_groups.items():
        dSns[Jpi] = Sns_1[Jpi] - Sns_2[Jpi]
    return dSns