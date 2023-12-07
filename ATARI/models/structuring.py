from pandas import DataFrame
from typing import Protocol, Optional


### =================
### Descriptors

class parameter:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner) -> tuple:
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Tuple for parameter must be (value, uncertainty)")
        else:
            raise ValueError("Must supply tuple for parameter value and uncertainty")
        instance.__dict__[self._name] = value


class vector_parameter:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner) -> tuple:
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        if isinstance(value, DataFrame):
            assert all(key in value.keys() for key in ['c','dc'])
        elif value is None:
            pass
        else:
            raise ValueError("Must supply DataFrame for vector parameter value and uncertainty")
        instance.__dict__[self._name] = value


### =======
### protocols

class Model_Parameters(Protocol):
    
    def sample_parameters(self, true_model_parameters: dict) -> "Model_Parameters":
        ...

    # @property
    # def neutron_spectrum(self) -> Optional[vector_parameter]:
    #     ...
    # @neutron_spectrum.setter
    # def neutron_spectrum(self, value: Optional[DataFrame]):
    #     ...

    # @property
    # def model_parameter_dict(self) -> dict:
    #     ...


class Generative_Measurement_Model(Protocol):

    def generate_raw_data(self, 
                          pw_true: DataFrame,
                          true_model_parameters: Model_Parameters, 
                          options
                          ) -> DataFrame:
        ...

    def approximate_unknown_data(self, exp_model) -> None:
        ...

    # @property
    # def neutron_spectrum_triggers(self) -> int:
    #     ...
    
    @property
    def model_parameters(self) -> Model_Parameters:
        ...



class Reductive_Measurement_Model(Protocol):

    def reduce_raw_data(self,
                        raw_data: DataFrame,
                        # neutron_spectrum: DataFrame,
                        options
                        ) -> DataFrame:
        ...

    def approximate_unknown_data(self, exp_model) -> None:
        ...
        
    # @property
    # def neutron_spectrum_triggers(self) -> int:
    #     ...

    @property
    def model_parameters(self) -> Model_Parameters:
        ...

    @property
    def covariance_data(self) -> dict:
        ...

