
import pandas as pd
from ATARI.models.experimental_model import Experimental_Model
from ATARI.models.particle_pair import Particle_Pair
from typing import Protocol, Optional

from ATARI.models.structuring import Generative_Measurement_Model, Reductive_Measurement_Model


class Generative_Model:
    def __init__(self,
                 particle_pair: Optional[Particle_Pair] = None,
                 generative_experimental_model: Optional[Experimental_Model] = None,
                 generative_measurement_model: Optional[Generative_Measurement_Model] = None
                 ):

        self.particle_pair = Particle_Pair()
        self.experimental_model = Experimental_Model()
        self.reduction_model = transmission_rpi()  # Generative_Reduction_Model()

        if particle_pair is not None:
            self.particle_pair = particle_pair
        if experimental_model is not None:
            self.experimental_model = experimental_model
        if reduction_model is not None:
            self.reduction_model = reduction_model

    @property
    def particle_pair(self):
        return self._particle_pair

    @particle_pair.setter
    def particle_pair(self, particle_pair):
        self._particle_pair = particle_pair

    @property
    def experimental_model(self):
        return self._experimental_model

    @experimental_model.setter
    def experimental_model(self, experimental_model):
        self._experimental_model = experimental_model

    @property
    def reduction_model(self):
        return self._reduction_model

    @reduction_model.setter
    def reduction_model(self, reduction_model):
        self._reduction_model = reduction_model
