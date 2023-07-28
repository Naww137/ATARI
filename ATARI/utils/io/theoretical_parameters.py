from ATARI.syndat.particle_pair import Particle_Pair
import pandas as pd
import numpy as np
from dataclasses import dataclass
from ATARI.utils.misc import fine_egrid 
from ATARI.utils.atario import fill_resonance_ladder
import ATARI.utils.io.hdf5 as h5io 

from abc import ABC, abstractmethod 


class TheoreticalParameters:

    def __init__(self):
        pass
        # self.particle_pair = None
        # self.resonance_ladder = None
        # self.label = None

    ### Setter functions
    def set_label(self, label: str) -> None:
        self.label = label
    def set_particle_pair(self, particle_pair: Particle_Pair) -> None:
        self.particle_pair = particle_pair
    def set_resonance_ladder(self, resonance_ladder: pd.DataFrame) -> None:
        self.resonance_ladder = resonance_ladder
    
    ### data manipulator functions
    def fill(self) -> None:
        self.resonance_ladder = fill_resonance_ladder(self.resonance_ladder, self.particle_pair)

    def to_hdf5(self, file: str, isample: int) -> None:
        h5io.write_par(file, isample, self.resonance_ladder, self.label)


    ### properties
    @property
    def NumRes(self) -> int:
        return len(self.resonance_ladder)
    @property
    def avg_gnx2(self) :
        return np.mean(self.resonance_ladder.gnx2)
    @property
    def avg_Gg(self) :
        return np.mean(self.resonance_ladder.Gg)
    @property
    def min_gnx2(self) -> float:
        return min(self.resonance_ladder.gnx2)
    @property
    def min_Gg(self) -> float:
        return min(self.resonance_ladder.Gg)
    @property
    def max_gnx2(self) -> float:
        return max(self.resonance_ladder.gnx2)
    @property
    def max_Gg(self) -> float:
        return max(self.resonance_ladder.Gg)





### Builders

class BuildTheoreticalParameters(ABC):
    def build_label(self): pass
    def build_particle_pair(self): pass
    def build_resonance_ladder(self): pass
    def construct(self):
        pass
    
    @property
    @abstractmethod
    def product(self): 
        pass


class BuildTheoreticalParameters_fromATARI(BuildTheoreticalParameters):

    def __init__(self, label: str, resonance_ladder: pd.DataFrame, particle_pair: Particle_Pair) -> None:
        """Fresh builder should be a clean slate"""
        self.reset()
        self.label = label
        self.resonance_ladder = resonance_ladder
        self.particle_pair = particle_pair

    def reset(self) -> None:
        self._product = TheoreticalParameters()

    @property
    def product(self) -> TheoreticalParameters:
        """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
        product = self._product
        self.reset()
        return product

    def build_label(self) -> None:
        self._product.set_label(self.label)

    def build_particle_pair(self) -> None:
        # TODO: read particle pair from hdf5
        self._product.set_particle_pair(self.particle_pair)

    def build_resonance_ladder(self) -> None:
        # resonance_ladder = fill_resonance_ladder(resonance_ladder, self_produce.particle_pair) # could fill resonance ladder here or not
        self._product.set_resonance_ladder(self.resonance_ladder)

    def construct(self) -> TheoreticalParameters:
        """Construct method acts as the director"""
        self.build_label()
        self.build_particle_pair()
        self.build_resonance_ladder()
        return self.product


class BuildTheoreticalParameters_fromHDF5(BuildTheoreticalParameters):

    def __init__(self, label: str, hdf5_file: str, isample: int, particle_pair: Particle_Pair) -> None:
        """Fresh builder should be a clean slate"""
        self.reset()
        self.hdf5_file = hdf5_file 
        self.isample = isample 
        self.label = label 
        self.particle_pair = particle_pair

    def reset(self) -> None:
        self._product = TheoreticalParameters()

    @property
    def product(self) -> TheoreticalParameters:
        """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
        product = self._product
        self.reset()
        return product

    def build_label(self) -> None:
        self._product.set_label(self.label)

    def build_particle_pair(self) -> None:
        # TODO: read particle pair from hdf5
        self._product.set_particle_pair(self.particle_pair)

    def build_resonance_ladder(self) -> None:
        resonance_ladder = pd.read_hdf(self.hdf5_file, f'sample_{self.isample}/par_{self.label}')
        if isinstance(resonance_ladder, pd.Series):
            resonance_ladder = resonance_ladder.to_frame().T
        # resonance_ladder = fill_resonance_ladder(resonance_ladder, self_produce.particle_pair) # could fill resonance ladder here or not
        self._product.set_resonance_ladder(resonance_ladder)

    def construct(self) -> TheoreticalParameters:
        """Construct method acts as the director"""
        self.build_label()
        self.build_particle_pair()
        self.build_resonance_ladder()
        return self.product


### Director

class DirectTheoreticalParameters:
    """ The Director is only responsible for executing the building steps in a particular sequence. """

    def __init__(self) -> None:
        pass
        # self._builder = None

    @property
    def builder(self) -> BuildTheoreticalParameters:
        return self._builder

    @builder.setter
    def builder(self, builder: BuildTheoreticalParameters) -> None:
        """The Director works with any builder instance that the client code passes to it."""
        self._builder = builder

    """The Director can construct several product variations using the same building steps."""
    # def build_minimal_viable_product(self) -> None:
    #     self.builder.produce_part_a()

    def build_product(self) -> None:
        self.builder.build_label()
        self.builder.build_particle_pair()
        self.builder.build_resonance_ladder()