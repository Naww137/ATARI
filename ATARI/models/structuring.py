from pandas import DataFrame
from typing import Protocol




class Generative_Measurement_Model(Protocol):

    def generate_raw_data(self, 
                          pw_true: DataFrame,
                          true_neutron_spectrum: DataFrame, 
                          options
                          ) -> DataFrame:
        ...

    @property
    def neutron_spectrum_triggers(self) -> int:
        ...



class Reductive_Measurement_Model(Protocol):

    def reduce_raw_data(self,
                        raw_data: DataFrame,
                        neutron_spectrum: DataFrame,
                        options
                        ) -> DataFrame:
        ...

    @property
    def neutron_spectrum_triggers(self) -> int:
        ...



class parameter:
    def __set_name__(self, owner, name):
        self._name = name

    def __get__(self, instance, owner) -> tuple:
        return instance.__dict__[self._name]

    def __set__(self, instance, value):
        # instance.__dict__[self._name] = date.fromisoformat(value)
        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError("Tuple for parameter must be (value, uncertainty)")
        else:
            raise ValueError("Must supply tuple for parameter value and uncertainty")
        instance.__dict__[self._name] = value
