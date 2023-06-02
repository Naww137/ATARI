
from typing import Optional
from dataclasses import dataclass
from ATARI.syndat.particle_pair import Particle_Pair
from pandas import DataFrame
from numpy import ndarray


@dataclass
class SammyRunTimeOptions:
    path_to_SAMMY_exe: str
    model: str = 'XCT'
    reaction: str = 'total'
    experimental_corrections: str = 'all_exp'
    solve_bayes: bool = False
    one_spingroup: bool = False
    energy_window: Optional[float] = None
    sammy_runDIR: str = 'SAMMY_runDIR'
    keep_runDIR: bool = False
    shell: str = 'zsh'


@dataclass
class SammyInputData:
    particle_pair: Particle_Pair
    resonance_ladder: DataFrame
    experimental_data: Optional[DataFrame] = None
    experimental_cov: Optional[DataFrame] = None
    energy_grid: Optional[ndarray] = None
    initial_parameter_uncertainty: Optional[float] = 1.0
