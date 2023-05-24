#%%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.xs import SLBW
# %%


ac = 0.81271  # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity
l_max = 1       # highest order l-wave to consider


spin_groups = [ (3.0,1,0) ]
average_parameters = pd.DataFrame({ 'dE'    :   {'3.0':8.79, '4.0':4.99},
                                    'Gg'    :   {'3.0':64.0, '4.0':64.0},
                                    'gn2'    :   {'3.0':46.4, '4.0':35.5}  })

Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={},
                                spin_groups=spin_groups,
                                average_parameters=average_parameters )   


E_min_max = [550, 600]
energy_grid = E_min_max

input_options = {'Add Noise': True,
                'Calculate Covariance': True,
                'Compression Points':[],
                'Grouping Factors':None}

experiment_parameters = {'bw': {'val':0.0256,   'unc'   :   0},
                         'n':  {'val':0.06,     'unc'   :0}}

# initialize experimental setup
exp = Experiment(energy_grid, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)


resonance_ladder = Ta_pair.sample_resonance_ladder(energy_grid, spin_groups, average_parameters)
true, _, _ = SLBW(exp.energy_domain, Ta_pair, resonance_ladder)
df_true = pd.DataFrame({'E':exp.energy_domain, 'theo_trans':np.exp(-exp.redpar.val.n*true)})

exp.run(df_true)



theo_par = parameters.factory.create('theo_par')
pw_exp = pointwise.factory.create('pw_exp', energy_grid = df_true.E)

theo_par.add_theoretical_parameters(Ta_pair, resonance_ladder)
theo_par.add_experimental_parameters(exp.redpar.val.n)

pw_exp.add_model(theo_par)
pw_exp.add_experimental_data(exp.trans, exp.CovT, 1e-2, exp.redpar.val.n, 0)
###
### Now use the DataCOntainer with the hdf5 MMDA structure
###
# case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/TestFMReduction.hdf5'
# dataset_range = (0, 10)
# samples_not_generated = generate(Ta_pair, exp, 
#                                         'syndat_SLBW', 
#                                         dataset_range, 
#                                         case_file,
#                                         fixed_resonance_ladder=None, 
#                                         open_data=None,
#                                         vary_Erange=None,
#                                         use_hdf5=True,
#                                         overwrite = False
#                                                                     )

#%%





#%%
