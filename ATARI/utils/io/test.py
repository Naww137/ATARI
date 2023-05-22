
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


# E_min_max = [590, 600]
# energy_grid = E_min_max

# input_options = {'Add Noise': True,
#                 'Calculate Covariance': True,
#                 'Compression Points':[],
#                 'Grouping Factors':None}

# experiment_parameters = {'bw': {'val':0.0256,   'unc'   :   0},
#                          'n':  {'val':0.06,     'unc'   :0}}

# # initialize experimental setup
# exp = Experiment(energy_grid, 
#                         input_options=input_options, 
#                         experiment_parameters=experiment_parameters)


# resonance_ladder = Ta_pair.sample_resonance_ladder(energy_grid, spin_groups, average_parameters)
# true, _, _ = SLBW(exp.energy_domain, Ta_pair, resonance_ladder)
# df_true = pd.DataFrame({'E':exp.energy_domain, 'theo_trans':np.exp(-exp.redpar.val.n*true)})

# exp.run(df_true)

#%%
from dataclasses import dataclass
import ATARI.atari_io.hdf5 as io
from ATARI.utils.io import parameters
from ATARI.utils.io import pointwise
from ATARI.utils.misc import fine_egrid 

from datacontainer import DataContainer
from pointwise import Pointwise_Container
from parameters import TheoreticalParameters
from parameters import ExperimentalParameters
from parameters import Estimates


case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/TestFMReduction.hdf5'

# read hdf
casenum = 3
theo_resladder = pd.read_hdf(case_file, f'sample_{casenum}/theo_par')
exp_pw, exp_cov = io.read_experimental(case_file, casenum)

exp_par = ExperimentalParameters(0.06, 0, 1e-2)
theo_par = TheoreticalParameters(Ta_pair, theo_resladder)
est_par = TheoreticalParameters(Ta_pair, theo_resladder)


from ATARI.utils.misc import fine_egrid 
pwfine = pd.DataFrame({'E':fine_egrid(exp_pw.E,100)})
pw = Pointwise_Container(exp_pw, pwfine)
pw.add_experimental(exp_pw, exp_cov, exp_par)
pw.add_model(theo_par, exp_par, 'theo')


dc = DataContainer(pw, exp_par, theo_par)
dc.add_estimate(est_par, 'est')

print(dc.pw.exp)


# class DataContainerConstructor:
#     def __init__():
        


# pw_exp.add_model(theo_par, 'test')
# pw_exp.add_experimental_data(exp.trans, exp.CovT, 1e-2, exp.redpar.val.n, 0)

figure()
# plot(pw.fine.E, pw.fine.theo_xs)
plot(pw.exp.E, pw.exp.theo_xs)
errorbar(pw.exp.E, pw.exp.exp_xs, yerr=pw.exp.exp_xs_unc, fmt='.', capsize=2)
show()
close()

#%%

# import pandas as pd

# test1 = pd.DataFrame({'E': [1,2,3], 'xs':[3,3,3]})
# test2 = pd.DataFrame({'E': [1,2,3], 'xs':[3,3,3], 'T':[4,3,2]})

# merge_keys = list(set(test1.columns).intersection(test2.columns))
# print(pd.merge(test1,test2, on=merge_keys))

#%%

# config = {
#     'theoretical_df': df_true,
#     'experimental': [2,2,2],
#     'data': [1,2,3]
# }


# theo = factory.create('theo', particle_pair=Ta_pair, resonance_ladder=resonance_ladder)
# theo.test()

# local = data.factory.create('LOCAL', **config)
# local.test_connection()

# spotify2 = data.services.get('SPOTIFY', **config)
# print(f'id(spotify) == id(spotify2): {id(spotify) == id(spotify2)}')

# %%
