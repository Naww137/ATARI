import os
import numpy as np
from ATARI import PiTFAll as pf
import pandas as pd
import h5py
from dataclasses import dataclass
from typing import List


@dataclass
class fit():

    name                :   str
    integral_FoMs       :   pd.DataFrame
    integral_par_FoMs   :   pd.DataFrame
    bv_window           :   pd.DataFrame
    bv_pw_xs_df         :   pd.DataFrame
    bv_pw_trans_df      :   pd.DataFrame
    syndat_data         :   pd.DataFrame
    printout            :   str

    def __post_init__(self):
        self.joined_df = pd.concat([self.integral_FoMs, 
                                    self.integral_par_FoMs,
                                    self.bv_window,
                                    self.syndat_data],      axis=1)

