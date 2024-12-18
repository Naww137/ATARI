
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
        return str(vars(self))




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
    alphanumeric = []

    


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
    covariance_data_at_theory: Optional[dict] = None
    covariance_data_at_theory_post: Optional[dict] = None
    
    total_derivative_evaluations = 0

    




### New scheme
### should eventually move to a parent class for sammyINP and child classes for specific iteration schemes

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
    experimental_covariance: list[Union[dict, str]] #= None

    experiments_no_pup: Optional[list[Experimental_Model]] = None

    idc_at_theory : bool = False
    measurement_models : Optional[list] = None

    external_resonance_indices: Optional[list] = None

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

    minibatch   :   bool = False
    minibatches :   int  = 4

    LevMar: bool = False
    LevMarV: float = 1.5
    LevMarVd: float = 5.0
    minF:   float = 1e-5
    maxF:   float = 10
    
    initial_parameter_uncertainty: float = 1.0



@dataclass
class SolverOPTs:
    _solver = "base"

    max_steps       : int       = 1
    step_threshold  : float     = 0.01

    LevMar          : bool      = False
    LevMarV         : float     = 1.5
    LevMarVd        : float     = 5.0
    minF            : float     = 1e-5
    maxF            : float     = 10

    idc_at_theory   : bool      = False

@dataclass
class SolverOPTs_YW(SolverOPTs):
    _solver = "YW"

    initial_parameter_uncertainty   : float     = 0.1
    iterations                      : int       = 2
    step_threshold_lag              : int       = 1
    autoelim_threshold              : Optional[float] = None

    LS                              : bool      = False
    batch_fitpar                    : bool      = False
    batch_fitpar_ifit               : int       = 10
    steps_per_batch                 : int       = 1
    batch_fitpar_random             : bool      = False
    minibatch                       : bool      = False
    minibatches                     : int       = 4

@dataclass
class SolverOPTs_EXT(SolverOPTs):
    _solver = "EXT"

    alpha           : float     = 1e-3
    solution_mode   : str       = "LMa"

    minibatch       : bool      = False
    batch_size      : int       = 10
    patience        : int       = 25
    beta_1          : float     = 0.9
    beta_2          : float     = 0.999
    epsilon         : float     = 1e-8

    lasso           : bool      = False
    lasso_parameters: dict = field(default_factory=lambda: {"lambda":1, "gamma":0, "weights":None})
    
    ridge           : bool      = False
    ridge_parameters: dict = field(default_factory=lambda: {"lambda":1, "gamma":0, "weights":None})

    elastic_net     : bool      = False
    elastic_net_parameters: dict = field(default_factory=lambda: {"lambda":1, "gamma":0, "alpha":0.7})

@dataclass
class SammyInputDataEXT:
    """
    Input data for sammy run.

    This object holds at minimum the particle pair description and a resonance ladder.
    An appropriate energy grid must also be supplied either in a DataFrame with experimental data or standalone as a series or array.
    The other attributes hold information about the data, experiment, and the initial parameter uncertainty.
    """
    particle_pair: Particle_Pair
    resonance_ladder: DataFrame

    datasets : list[DataFrame]
    experiments: list[Experimental_Model]  # sammy_interface only needs title and template outside of write_saminp
    experiments_no_pup: list[Experimental_Model]
    experimental_covariance: Optional[list[Union[dict, str]]] #= None

    idc_at_theory : bool = False
    measurement_models : Optional[list] = None
    # V_projection: Optional[ndarray] = None

    external_resonance_indices: Optional[list] = None
    cap_norm_unc: float = 0.0
    remove_V: bool = False
    V_is_inv: bool = False
    Vinv: Optional[ndarray] = None
    D: Optional[ndarray] = None


    ### alternatively could have nested class for solver options
    # solver_options: SolverOPTs_EXT = field(default_factory=SolverOPTs_EXT)
    max_steps       : int       = 1
    step_threshold  : float     = 0.01

    LevMar          : bool      = False
    LevMarV         : float     = 1.5
    LevMarVd        : float     = 5.0
    minF            : float     = 1e-6
    maxF            : float     = 1e-2

    alpha           : float     = 1e-3
    solution_mode   : str      = "LMa"
    
    minibatch       : bool      = False
    batch_size      : int       = 10
    patience        : int       = 25
    beta_1          : float     = 0.9
    beta_2          : float     = 0.999
    epsilon         : float     = 1e-8

    lasso           : bool      = False
    lasso_parameters: dict = field(default_factory=lambda: {"lambda":1, "gamma":0, "weights":None})
    
    ridge           : bool      = False
    ridge_parameters: dict = field(default_factory=lambda: {"lambda":1, "gamma":0, "weights":None})

    elastic_net     : bool      = False
    elastic_net_parameters: dict = field(default_factory=lambda: {"lambda":1, "gamma":0, "alpha":0.7})
    


