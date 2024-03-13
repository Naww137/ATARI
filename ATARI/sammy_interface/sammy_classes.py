
from typing import Optional, Union
from dataclasses import dataclass, field
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model
from pandas import DataFrame, Series
from numpy import ndarray
import os


class SammyRunTimeOptions:

    def __init__(self, 
                 sammyexe: str, 
                 **kwargs
                 ):
        """
        Sammy run time option class that holds information about how you would like to run SAMMY.
        The only require arguement is sammyexe.
        

        Parameters
        ----------
        sammyexe : str
            Full path to the local sammy executable.
        **kwargs : dict, optional
            Any keyword arguments are used to set attributes on the instance.

        Attributes
        ----------
        path_to_SAMMY_exe   :   str
            Full path to the local sammy executable.
        sammy_runDIR    :   str, 'sammy_runDIR'
            Directory in which to run sammy.
        keep_runDIR :   bool, False
            Option to keep sammy_runDIR after running sammy.
        Print   :   bool, False
            Option to print out status while running sammy.
        bayes   :   bool, False
            Option to solve bayes while running sammy.
        iterations  :   int, 2
            Number of internal iterations for non-linearities
        energy_window   :   None or float, None
            Energy window size for windowed sammy runs between Emin and Emax
        get_ECSCM   :   bool, False
            Option to run an additional sammy run to calculate ECSCM from RPCM.
            Bayes must be True.
        ECSCM_rxn   :   str, 'total'
            Reaction on which to calculate the ECSCM, default is total.
        ECSCM_template  :   str or None, None
            Optional input to change the sammy template for ECSCM calculation.
            Default behavior (None) will use the template used for the basic sammy run with Bayes.
        """

        ### set defaults
        self.path_to_SAMMY_exe = sammyexe
        self.sammy_runDIR =  "sammy_runDIR"
        self.keep_runDIR = False
        self.Print =  False
        
        self.bayes = False
        self.iterations = 2

        self.energy_window = None
        self.get_ECSCM = False
        self.ECSCM_rxn = 'total'
        self.ECSCM_template = None #os.path.realpath(os.path.join(os.path.dirname(__file__), "sammy_templates/dop_2sg.inp"))

        ### update attributes to **kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


    def __repr__(self):
        return str(self.options)




arraytype_id = Union[Series, ndarray, list]
arraytype_broadparm = Union[str, float]

@dataclass
class SammyInputData:
    """
    Input data for sammy run.

    This object holds at minimum the particle pair description and a resonance ladder.
    An appropriate energy grid must also be supplied either in a DataFrame with experimental data or standalone as a series or array.
    The other attributes hold information about the data, experiment, and the initial parameter uncertainty.
    """
    particle_pair: Particle_Pair
    resonance_ladder: DataFrame
    template: str
    experiment: Experimental_Model
    experimental_data: Optional[Union[DataFrame,ndarray]] = None
    experimental_covariance: Optional[dict] = None
    energy_grid: Optional[arraytype_id] = None

    initial_parameter_uncertainty: Optional[float] = 1.0
    


@dataclass
class SammyOutputData:
    pw: Union[DataFrame, list[DataFrame]]
    par: DataFrame
    chi2: Union[float, list[float]]
    chi2n: Union[float, list[float]]
    pw_post: Optional[Union[DataFrame, list[DataFrame]]] = None
    par_post: Optional[DataFrame] = None
    chi2_post: Optional[Union[float, list[float]]] = None
    chi2n_post : Optional[Union[float, list[float]]] = None
    derivatives: Optional[ndarray] = None

    ECSCM: Optional[DataFrame] = None 
    est_df: Optional[DataFrame] = None
    




### New scheme


def update_dict(old, additional):
    new = old
    for key in old:
        if key in additional:
            new.update({key:additional[key]})
    return new







@dataclass
class SammyInputDataYW:
    """
    Input data for sammy run using YW scheme.

    This object holds at minimum the particle pair description and a resonance ladder.
    An appropriate energy grid must also be supplied either in a DataFrame with experimental data or standalone as a series or array.
    The other attributes hold information about the data, experiment, and the initial parameter uncertainty.
    """
    particle_pair: Particle_Pair
    resonance_ladder: DataFrame

    datasets : list[DataFrame]
    experiments: list[Experimental_Model]  # sammy_interface only needs title and template outside of write_saminp
    experimental_covariance: Optional[list[Union[dict, str]]] #= None

    max_steps: int = 1
    iterations: int = 2
    step_threshold: float = 0.01
    step_threshold_lag: int = 1
    autoelim_threshold: Optional[float] = None

    LS: bool = False

    batch_fitpar :  bool = False
    batch_fitpar_ifit: int = 10
    steps_per_batch: int = 1
    batch_fitpar_random: bool = False

    external_resonance_indices: Optional[list] = None

    LevMar: bool = True
    LevMarV: float = 1.5
    LevMarVd: float = 5.0
    minF:   float = 1e-5
    maxF:   float = 10
    
    initial_parameter_uncertainty: float = 1.0





