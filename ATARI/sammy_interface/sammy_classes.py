
from typing import Optional, Union, List, Any
from dataclasses import dataclass, field
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model
from pandas import DataFrame, Series
from numpy import ndarray


# from ATARI.utils.stats import chi2_val

# @dataclass
# class SammyRunTimeOptions:
#     """
#     Runtime options for sammy. 

#     This object holds many options for how sammy should be used.
#     Running sammy with this interface is dependent on the supply of a template input file that is used to handle the extensive list of options when running sammy (i.e., spin group definitions, experimental corrections). 
#     The options here fall into two primary categories:
#     1) simple options that can be toggled on/off without significant change to the input (i.e., reaction model, run bayes).
#     2) automated approaches such as recursion, least squares, simultaneous or sequential fitting, etc.

#     There are several input templates preloaded with the ATARI package, but the user can supply one as well. 
#     """
#     path_to_SAMMY_exe: str
#     shell: str = 'zsh'
#     sammy_runDIR: str = 'SAMMY_runDIR'
#     keep_runDIR: bool = False
#     Print: bool = False

#     model: str = 'XCT'
#     reaction: str = 'total'
#     solve_bayes: bool = False
#     inptemplate: str = "noexp_1sg.inp"
#     inpname: str = "sammy.inp"
#     title: str = "default title"
#     get_ECSCM: bool = False

#     alphanumeric: list = field(default_factory=lambda: [])
#     energy_window: Optional[float] = None
#     recursive: bool = False
#     recursive_opt: dict = field(default_factory=lambda: {"threshold":0.01,
#                                                         "iterations": 5,
#                                                         "print":False}      )


class SammyRunTimeOptions:

    def __init__(self, sammyexe: str, options={}, alphanumeric=None):
        default_options = {
            # 'sh'               : 'zsh',
            'sammy_runDIR'      : 'SAMMY_runDIR', # the run directory for SAMMY input and output files
            'keep_runDIR'       : False,
            'Print'             : False,

            # What to calculate:
            'derivatives'       : False, # calculate the derivatives
            'bayes'             : False, # calculate Bayesian posteriors

            # What to use when calculating:
            'iterations'        : 2,     # the number of iterations of bayes
            'bayes_scheme'      : None,  # the scheme for bayes (MW, IQ, or NV)
            'use_least_squares' : False, # uses least squares if true

            'energy_window'     : None, 
            'get_ECSCM'         : False,
            'ECSCM_rxn'         : 'total'
        }
        options = update_dict(default_options, options)
        self.options = options

        self.path_to_SAMMY_exe = sammyexe
        # self.shell =  options["sh"]
        self.sammy_runDIR =  options["sammy_runDIR"]
        self.keep_runDIR = options["keep_runDIR"]
        self.Print =  options["Print"]
        
        self.derivatives = options["derivatives"]
        self.bayes = options["bayes"]

        self.iterations = options["iterations"]
        self.bayes_scheme = options["bayes_scheme"]
        self.use_least_squares = options["use_least_squares"]

        self.energy_window = options["energy_window"]
        self.get_ECSCM = options["get_ECSCM"]
        self.ECSCM_rxn = options["ECSCM_rxn"]

        if alphanumeric is None:
            alphanumeric = []
        self.alphanumeric = alphanumeric

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

    datasets : List[DataFrame]
    experimental_covariance: Optional[List[Union[dict, str]]]
    experiments: List[Experimental_Model]  # sammy_interface only needs title and template outside of write_saminp

    max_steps: int = 1
    iterations: int = 2
    step_threshold: float = 0.01
    autoelim_threshold: Optional[float] = None

    LS: bool = False
    LevMar: bool = True
    LevMarV: float = 1.5
    LevMarVd: float = 5.0
    minF:   float = 1e-5
    maxF:   float = 10
    initial_parameter_uncertainty: float = 1.0




