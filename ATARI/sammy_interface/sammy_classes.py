
from typing import Optional, Union, List, Any
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
        Runtime options for sammy. 

        This object holds many options for how sammy should be used.
        Running sammy with this interface is dependent on the supply of a template input file that is used to handle the extensive list of options when running sammy (i.e., spin group definitions, experimental corrections). 
        The options here fall into two primary categories:
        1) simple options that can be toggled on/off without significant change to the input (i.e., reaction model, run bayes).
        2) automated approaches such as recursion, least squares, simultaneous or sequential fitting, etc.

        There are several input templates preloaded with the ATARI package, but the user can supply one as well. 

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
        derivatives : bool, False
            Option to use sammy to get derivatives.
        bayes   :   bool, False
            Option to solve bayes while running sammy.
        iterations  :   int, 2
            Number of internal iterations for non-linearities
        bayes_scheme : str, None
            Solution scheme to bayes equations (IV, NQ, MW)
        use_least_squares : bool, None
            Option to use least squares fitting rather than Bayes.
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
        
        self.derivatives = False
        self.bayes = False

        # What to use when calculating:
        self.iterations = 2
        self.bayes_scheme = None

        ### YWY specific
        self.use_least_squares = False
        self.save_lsts_YW_steps = False

        self.energy_window = None
        self.get_ECSCM = False
        self.alphanumeric = None
       
        ### update attributes to **kwargs
        for key, value in kwargs.items():
            if key == 'options': # catch legacy implementation
                for key1, val1 in value.items():
                    setattr(self, key1, val1)
            setattr(self, key, value)


        if self.alphanumeric is None:
            self.alphanumeric = []


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

    experiment: Experimental_Model
    experimental_data: Optional[Union[DataFrame,ndarray]] = None
    experimental_covariance: Optional[dict] = None
    energy_grid: Optional[arraytype_id] = None
    template: Optional[str] = None # outdated and should not be used in most cases

    initial_parameter_uncertainty: Optional[float] = 1.0

    ECSCM_experiment: Optional[Experimental_Model] = None

    


@dataclass
class SammyOutputData:
    pw: Union[DataFrame, List[DataFrame]]
    par: DataFrame
    chi2: Union[float, List[float]]
    chi2n: Union[float, List[float]]
    pw_post: Optional[Union[DataFrame, List[DataFrame]]] = None
    par_post: Optional[DataFrame] = None
    chi2_post: Optional[Union[float, List[float]]] = None
    chi2n_post: Optional[Union[float, List[float]]] = None
    derivatives: Optional[ndarray] = None

    ECSCM: Optional[DataFrame] = None 
    est_df: Optional[DataFrame] = None
    




### New scheme


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

    minibatch   :   bool = True
    minibatches :   int  = 4

    external_resonance_indices: Optional[list] = None

    LevMar: bool = True
    LevMarV: float = 1.5
    LevMarVd: float = 5.0
    minF:   float = 1e-5
    maxF:   float = 10
    
    initial_parameter_uncertainty: float = 1.0





