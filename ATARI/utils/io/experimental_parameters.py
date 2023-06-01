from abc import ABC, abstractmethod 

from ATARI.theory.experimental import trans_2_xs


class ExperimentalParameters:

    def __init__(self):
        pass
        # self.n = None
        # self.dn = None
        # self.blackthreshold = None
    
    def set_n(self, n: float, dn: float) -> None:
        self.n = n
        self.dn = dn
    
    def set_blackthreshold(self, blackthreshold: float) -> None:
        self.blackthreshold = blackthreshold

    @property
    def max_xs(self): 
        return trans_2_xs(self.blackthreshold, self.n)[0]


class BuildExperimentalParameters(ABC):
    def build_n(self): 
        pass
    def build_blackthreshold(self):
        pass
    def construct(self):
        pass

    @property
    @abstractmethod
    def product(self): 
        pass

class BuildExperimentalParameters_fromDIRECT(BuildExperimentalParameters):

    def __init__(self, n: float, dn: float, blackthreshold: float) -> None:
        """Fresh builder should be a clean slate"""
        self.reset()
        self.n = n 
        self.dn = dn 
        self.blackthreshold = blackthreshold 

    def reset(self) -> None:
        self._product = ExperimentalParameters()

    @property
    def product(self) -> ExperimentalParameters:
        """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
        product = self._product
        self.reset()
        return product

    def build_n(self) -> None:
        self._product.set_n(self.n, self.dn)

    def build_blackthreshold(self) -> None:
        self._product.set_blackthreshold(self.blackthreshold)


    def construct(self) -> ExperimentalParameters:
        self.build_n()
        self.build_blackthreshold()
        return self.product

class DirectExperimentalParameters:
    """ The Director is only responsible for executing the building steps in a particular sequence. """

    def __init__(self) -> None:
        pass
        # self._builder = None

    @property
    def builder(self) -> BuildExperimentalParameters:
        return self._builder

    @builder.setter
    def builder(self, builder: BuildExperimentalParameters) -> None:
        """The Director works with any builder instance that the client code passes to it."""
        self._builder = builder

    """The Director can construct several product variations using the same building steps."""
    # def build_minimal_viable_product(self) -> None:
    #     self.builder.produce_part_a()

    def build_product(self) -> None:
        self.builder.build_n()
        self.builder.build_blackthreshold()