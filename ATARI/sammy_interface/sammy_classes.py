
from typing import Optional, Union
from dataclasses import dataclass, field
from ATARI.syndat.particle_pair import Particle_Pair
from pandas import DataFrame, Series
from numpy import ndarray
# from ATARI.utils.stats import chi2_val

@dataclass
class SammyRunTimeOptions:
    """
    Runtime options for sammy. 

    This object holds many options for how sammy should be used.
    Running sammy with this interface is dependent on the supply of a template input file that is used to handle the extensive list of options when running sammy (i.e., spin group definitions, experimental corrections). 
    The options here fall into two primary categories:
    1) simple options that can be toggled on/off without significant change to the input (i.e., reaction model, run bayes).
    2) automated approaches such as recursion, least squares, simultaneous or sequential fitting, etc.

    There are several input templates preloaded with the ATARI package, but the user can supply one as well. 
    """
    path_to_SAMMY_exe: str
    shell: str = 'zsh'
    sammy_runDIR: str = 'SAMMY_runDIR'
    keep_runDIR: bool = False

    model: str = 'XCT'
    reaction: str = 'total'
    solve_bayes: bool = False
    inptemplate: str = "noexp_1sg.inp"
    inpname: str = "sammy.inp"
    title: str = "default title"
    get_ECSCM: bool = False

    alphanumeric: list = field(default_factory=lambda: [])
    energy_window: Optional[float] = None
    recursive: bool = False
    recursive_opt: dict = field(default_factory=lambda: {"threshold":0.01,
                                                        "iterations": 5,
                                                        "print":False}      )


arraytype_id = Union[Series, ndarray]

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
    experimental_data: Optional[DataFrame] = None
    experimental_cov: Optional[DataFrame] = None
    energy_grid: Optional[arraytype_id] = None
    target_thickness: Optional[float] = None
    temp: Optional[float] = None
    FP: Optional[float] = None
    frac_res_FP: Optional[float] = None
    initial_parameter_uncertainty: Optional[float] = 1.0


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

    datasets : list
    dataset_titles : list
    reactions : list
    templates : list

    steps: int = 200
    iterations: int = 2
    threshold: float = 0.001

    target_thickness: Optional[float] = None
    temp: Optional[float] = None
    FP: Optional[float] = None
    frac_res_FP: Optional[float] = None
    initial_parameter_uncertainty: Optional[float] = 1.0


@dataclass
class SammyOutputData:
    pw: DataFrame
    par: DataFrame
    # chi2: float
    par_post: Optional[DataFrame] = None
    # chi2_post: Optional[float] = None
    derivatives: Optional[ndarray] = None

    ECSCM: Optional[DataFrame] = None 
    est_df: Optional[DataFrame] = None
    


    # @property
    # def chi2(self, rxn, post):
    #     theo = self.pw[f"theo_{rxn}"]
        



    


    # @property
    # def chi2(self):
    #     return chi2_val(self.pw.theo)
        

# @dataclass
# class SammyOutputData:
#     pw: DataFrame
#     par: DataFrame
#     chi2: