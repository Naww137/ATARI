#%%
import numpy as np
import pandas as pd
import os
import h5py
import scipy.stats as sts

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.resonance_statistics import make_res_par_avg
from ATARI.syndat.sample_resparms import sample_resonance_ladder

# from ATARI.theory.xs import SLBW
# from ATARI.theory.scattering_params import FofE_recursive
# from ATARI.theory.scattering_params import gstat
# from ATARI.utils.datacontainer import DataContainer
# from ATARI.utils.atario import fill_resonance_ladder
# from ATARI.utils.stats import chi2_val

#%%
ac = 0.81271  # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity
l_max = 1       # highest order l-wave to consider

E_min_max = [75, 125]
energy_grid = E_min_max

input_options = {'Add Noise': True,
                'Calculate Covariance': False,
                'Compression Points':[],
                'Grouping Factors':None}

experiment_parameters = {'bw': {'val':0.1024,   'unc'   :   0},
                         'n':  {'val':0.067166,     'unc'   :0}}

exp = Experiment(energy_grid, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)


for Gg_DOF in [10,100,1000,10000]:


    res_par_avg = make_res_par_avg(D_avg = 8.79, 
                                Gn_avg= 46.4, 
                                n_dof = 1, 
                                Gg_avg = 64.0, 
                                g_dof = Gg_DOF, 
                                print = False)

    spin_groups = [ (3.0,1,0) ]
    average_parameters = {'3.0':res_par_avg}

    # resonance_ladder = sample_resonance_ladder(exp.energy_domain, spin_groups, average_parameters, 
    #                                                             use_fudge=False)

    Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                    input_options={},
                                    spin_groups=spin_groups,
                                    average_parameters=average_parameters )   

    case_file = f'/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/varyGg/GgDOF_{Gg_DOF}.hdf5'
    dataset_range = (0, 500)
    samples_not_generated = generate(Ta_pair, exp, 
                                            'syndat_SLBW', 
                                            dataset_range, 
                                            case_file,
                                            fixed_resonance_ladder=None, 
                                            open_data=None,
                                            vary_Erange=None,
                                            use_hdf5=True,
                                            overwrite = False)


# %%
