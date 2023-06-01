
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

from ATARI.utils.misc import fine_egrid 

from ATARI.utils.io.experimental_parameters import BuildExperimentalParameters_fromDIRECT, DirectExperimentalParameters
from ATARI.utils.io.theoretical_parameters import BuildTheoreticalParameters_fromHDF5, BuildTheoreticalParameters_fromATARI, DirectTheoreticalParameters
from ATARI.utils.io.pointwise_container import BuildPointwiseContainer_fromHDF5, BuildPointwiseContainer_fromATARI, DirectPointwiseContainer
from ATARI.utils.io.data_container import BuildDataContainer_fromBUILDERS, DirectDataContainer


### Build data objects from atari 

# build theoretical parameters
resonance_ladder = pd.DataFrame({'E':[], 'Gg':[], 'Gnx':[], 'chs':[], 'lwave':[], 'J':[], 'J_ID':[]})
director = DirectTheoreticalParameters()
builder_theo_par = BuildTheoreticalParameters_fromATARI('test', resonance_ladder, Ta_pair)
director.builder = builder_theo_par
director.build_product()
# theo_par = builder_theo_par.product

# build experimental parameters
director = DirectExperimentalParameters()
builder_exp_par = BuildExperimentalParameters_fromDIRECT(0.05, 0, 1e-2)
director.builder = builder_exp_par
director.build_product()
# exp_par = builder_exp_par.product

# build pointwise data
pwfine = pd.DataFrame({'E':fine_egrid([5,20],10)})
pw_exp = pd.DataFrame({'E':[5,20], 'exp_trans': [0.8,0.8]})
CovT = pd.DataFrame(np.array([[0.1,0], [0,0.1]]), index=pw_exp.E, columns= pw_exp.E)
CovT.index.name = None

director = DirectPointwiseContainer()
builder_pw = BuildPointwiseContainer_fromATARI(pw_exp, CovT=CovT, ppeV=10)
director.builder = builder_pw
director.build_lite_w_CovT()
# pw = builder_pw.product
# pw.add_model(theo_par, exp_par)


director = DirectDataContainer()
builder = BuildDataContainer_fromBUILDERS(
    builder_pw,
    builder_exp_par,
    [builder_theo_par]
    )
director.builder = builder
# director.build_product()
# dc = builder.product
dc = director.construct()
# dc.pw.add_model()

print(dc.pw.exp)
# print(dc.pw.CovT)


### write to hdf5
casenum = 1
case_file = './test.hdf5'
dc.to_hdf5(case_file, casenum)


### Buld from hdf5
director = DirectTheoreticalParameters()
builder_theo_par_h5 = BuildTheoreticalParameters_fromHDF5('test', case_file, casenum, Ta_pair)
director.builder = builder_theo_par_h5
director.build_product()


director = DirectPointwiseContainer()
builder_pw_h5 = BuildPointwiseContainer_fromHDF5(case_file, casenum)
director.builder = builder_pw_h5
director.build_lite_w_CovT()

## Can get the product result directly and call dc director.build or can wait to get product and call dc director.construct
## similar option above
# theo_par_h5 = builder_theo_par_h5.product
# pw_h5 = builder_pw_h5.product


director = DirectDataContainer()
builder = BuildDataContainer_fromBUILDERS(
    builder_pw_h5,
    builder_exp_par,
    [builder_theo_par_h5]
    )
director.builder = builder
# director.build_product()
# dc_h5 = builder.product
dc_h5 = director.construct()

print(dc_h5.pw.exp)


# # pw_exp.add_model(theo_par, 'test')
# # pw_exp.add_experimental_data(exp.trans, exp.CovT, 1e-2, exp.redpar.val.n, 0)

# figure()
# # plot(pw.fine.E, pw.fine.theo_xs)
# plot(pw.exp.E, pw.exp.theo_xs)
# errorbar(pw.exp.E, pw.exp.exp_xs, yerr=pw.exp.exp_xs_unc, fmt='.', capsize=2)
# show()
# close()

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
