from abc import ABC, abstractmethod
from dataclasses import dataclass

from ATARI.theory.experimental import trans_2_xs
from ATARI.utils.atario import fill_resonance_ladder
from ATARI.syndat.particle_pair import Particle_Pair
from pandas import DataFrame


@dataclass
class ExperimentalParameters:
    n: float
    dn: float
    blackthreshold: float
    
    @property
    def max_xs(self): 
        return trans_2_xs(self.blackthreshold, self.n)[0]

    # @n.setter
    # def n(self, value):
    #     self._diameter = None  # Reset the cached value
    #     self._radius = value


@dataclass
class TheoreticalParameters:
    particle_pair: Particle_Pair
    resonance_ladder: DataFrame

    def __post_init__(self):
        self.resonance_ladder = fill_resonance_ladder(self.resonance_ladder, self.particle_pair)

class Estimates():
    def __init__(self):
        self.container = {}

    def add_estimate(self, theopar: TheoreticalParameters, label: str):
        self.container[label] = theopar



# @dataclass
# class Parameters:
#     experimental: ExperimentalParameters
#     true: TheoreticalParameters













