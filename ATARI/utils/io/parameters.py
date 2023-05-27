from abc import ABC, abstractmethod
from dataclasses import dataclass

from ATARI.theory.experimental import trans_2_xs
from ATARI.utils.atario import fill_resonance_ladder
from ATARI.syndat.particle_pair import Particle_Pair
from pandas import DataFrame
import ATARI.atari_io.hdf5 as io


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
    label: str

    def __post_init__(self):
        self.resonance_ladder = fill_resonance_ladder(self.resonance_ladder, self.particle_pair)

    def to_hdf5(self, file: str, isample: int) -> None:
        io.write_par(file, isample, self.resonance_ladder, self.label)



# @dataclass
# class Parameters:
#     experimental: ExperimentalParameters
#     true: TheoreticalParameters













