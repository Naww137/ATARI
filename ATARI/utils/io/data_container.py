


from ATARI.utils.io.experimental_parameters import ExperimentalParameters, BuildExperimentalParameters
from ATARI.utils.io.theoretical_parameters import TheoreticalParameters, BuildTheoreticalParameters
from ATARI.utils.io.pointwise_container import PointwiseContainer, BuildPointwiseContainer

from abc import ABC, abstractmethod 


class DataContainer:

    def __init__(self) -> None:
        self.theoretical_parameters = {}

    def set_pw(self, pw) -> None: #, pw: PointwiseContainer, exp_par: ExperimentalParameters, theo_par: TheoreticalParameters, est_par: dict = {}
        self.pw = pw
    def set_experimental_parameters(self, experimental_parameters: ExperimentalParameters) -> None:
        self.experimental_parameters = experimental_parameters

    def add_theoretical_parameters(self, theoretical_parameters: TheoreticalParameters) -> None:
        self.theoretical_parameters[theoretical_parameters.label] = theoretical_parameters
    
    def fill_pw(self) -> None:
        for key, theoretical_parameter_set in self.theoretical_parameters.items():
            self.pw.add_model(theoretical_parameter_set, self.experimental_parameters)

    def to_hdf5(self, file: str, isample: int) -> None:
        self.pw.to_hdf5(file, isample)
        for key, theoretical_parameter_set in self.theoretical_parameters.items():
            theoretical_parameter_set.to_hdf5(file, isample)


class BuildDataContainer(ABC):
    def build_pw(self): pass
    def build_experimental_parameters(self): pass
    def build_theoretical_parameters(self, list_of_theoretical_parameters: list = []): pass

    @property
    @abstractmethod
    def product(self): 
        pass 

class BuildDataContainer_fromBUILDERS(BuildDataContainer):

    def __init__(self, pointwise_container_builder: BuildPointwiseContainer, experimental_parameter_builder: BuildExperimentalParameters, list_of_theoretical_parameters: list = []) -> None:
        """Fresh builder should be a clean slate"""
        self.reset()
        self.pw = pointwise_container_builder.product
        self.experimental_parameters = experimental_parameter_builder.product
        self.list_of_theoretical_parameters = list_of_theoretical_parameters

    def reset(self) -> None:
        self._product = DataContainer()

    @property
    def product(self) -> DataContainer:
        """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
        product = self._product
        self.reset()
        return product

    def build_pw(self) -> None:
        self._product.set_pw(self.pw)
    def build_experimental_parameters(self) -> None:
        self._product.set_experimental_parameters(self.experimental_parameters)

    def build_theoretical_parameters(self) -> None:
        for theoretical_parameters in self.list_of_theoretical_parameters:
            self._product.add_theoretical_parameters(theoretical_parameters.product)


# class BuildDataContainer_fromATARI(BuildDataContainer):

#     def __init__(self, 
#                  pointwise_container_builder: BuildPointwiseContainer, 
#                  experimental_parameter_builder: BuildExperimentalParameters, 
#                  list_of_theoretical_parameters: list = []) -> None:
#         """Fresh builder should be a clean slate"""
#         self.reset()
#         self.pw = pointwise_container_builder.product
#         self.experimental_parameters = experimental_parameter_builder.product
#         self.list_of_theoretical_parameters = list_of_theoretical_parameters

#     def reset(self) -> None:
#         self._product = DataContainer()

#     @property
#     def product(self) -> DataContainer:
#         """After returning the end result to the client, a builder instance is expected to be ready to start producing another product."""
#         product = self._product
#         self.reset()
#         return product

#     def build_pw(self) -> None:
#         self._product.set_pw(self.pw)
#     def build_experimental_parameters(self) -> None:
#         self._product.set_experimental_parameters(self.experimental_parameters)

#     def build_theoretical_parameters(self) -> None:
#         for theoretical_parameters in self.list_of_theoretical_parameters:
#             self._product.add_theoretical_parameters(theoretical_parameters.product)

### Director

class DirectDataContainer:
    """ The Director is only responsible for executing the building steps in a particular sequence. """

    def __init__(self) -> None:
        pass
        # self._builder = None

    @property
    def builder(self) -> BuildDataContainer:
        return self._builder

    @builder.setter
    def builder(self, builder: BuildDataContainer) -> None:
        """The Director works with any builder instance that the client code passes to it."""
        self._builder = builder

    """The Director can construct several product variations using the same building steps."""
    # def build_minimal_viable_product(self) -> None:
    #     self.builder.produce_part_a()

    def build_product(self) -> None:
        self.builder.build_pw()
        self.builder.build_experimental_parameters()
        self.builder.build_theoretical_parameters()

    def construct(self) -> DataContainer:
        self.build_product()
        return self.builder.product