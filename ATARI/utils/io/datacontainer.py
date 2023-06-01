from ATARI.utils.io.pointwise import PointwiseContainer
from ATARI.utils.io.parameters import TheoreticalParameters
from ATARI.utils.io.parameters import ExperimentalParameters

class DataContainer:

    def __init__(self, pw: PointwiseContainer, exp_par: ExperimentalParameters, theo_par: TheoreticalParameters, est_par: dict = {}) -> None:
        self.pw = pw
        self.exp_par = exp_par
        self.theo_par = theo_par
        self.est_par = est_par

    def add_estimate(self, theopar: TheoreticalParameters) -> None:
        self.pw.add_model(theopar, self.exp_par)
        self.est_par[theopar.label] = theopar

    def to_hdf5(self, file: str, isample: int) -> None:
        self.pw.to_hdf5(file, isample)
        self.theo_par.to_hdf5(file,isample)
        for key, est_par in self.est_par.items():
            est_par.to_hdf5(file, isample)


# import object_factory


# # actual class
# @dataclass
# class DataContainer:
#     pw: Pointwise_Container
#     par: parameters.Parameters

# # # builder that instantiates class above using **kwargs passed by object factory
# # def create_local_music_service(data, **_ignored):
# #     return LocalService(data)

# # builder that instantiates class above using **kwargs passed by object factory
# class ConstructDataContainerFromTheoretical:
#     def __init__(self):
#         self._instance = None

#     def __call__(self, theoretical, **_ignored):

#         # if theoretical has not yet been instantiated, perform overhead calculations
#         if not self._instance:
#             square = self.sq(theoretical)
#             self._instance = Theoretical(square)
#         # else, return existing instance
#         return self._instance

#     def sq(self,theoretical):
#         return 'filled data'


# factory = object_factory.ObjectFactory()
# factory.register_builder('theo', TheoreticalBuilder()) # __call__ returns initialized SpotifyService object
# factory.register_builder('LOCAL', create_local_music_service) # returns initialized LocalService object
