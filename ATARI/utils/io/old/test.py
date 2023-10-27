
#%%
import numpy as np
import pandas as pd

from theory.particle_pair import Particle_Pair
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

#%%

from ATARI.utils.io.experimental_parameters import BuildExperimentalParameters_fromDIRECT, DirectExperimentalParameters
from ATARI.utils.io.theoretical_parameters import BuildTheoreticalParameters_fromHDF5, BuildTheoreticalParameters_fromATARI, DirectTheoreticalParameters
from ATARI.utils.io.pointwise_container import BuildPointwiseContainer_fromHDF5, BuildPointwiseContainer_fromATARI, DirectPointwiseContainer
from ATARI.utils.io.data_container import BuildDataContainer_fromBUILDERS, BuildDataContainer_fromOBJECTS, DirectDataContainer

#%% testing using the builder.construct method

print()
print('Now testing using the builder.construct methods')
print()

### Build data objects from atari 

# build theoretical parameters
resonance_ladder = pd.DataFrame({'E':[], 'Gg':[], 'Gnx':[], 'chs':[], 'lwave':[], 'J':[], 'J_ID':[]})
builder_theo_par = BuildTheoreticalParameters_fromATARI('test', resonance_ladder, Ta_pair)
theo_par = builder_theo_par.construct()

# build experimental parameters
builder_exp_par = BuildExperimentalParameters_fromDIRECT(0.05, 0, 1e-2)
exp_par = builder_exp_par.construct()

# build pointwise data
from ATARI.utils.misc import fine_egrid 
pwfine = pd.DataFrame({'E':fine_egrid([5,20],10)})
pw_exp = pd.DataFrame({'E':[5,20], 'exp_trans': [0.8,0.8]})
CovT = pd.DataFrame(np.array([[0.1,0], [0,0.1]]), index=pw_exp.E, columns= pw_exp.E)
CovT.index.name = None

builder_pw = BuildPointwiseContainer_fromATARI(pw_exp, CovT=CovT, ppeV=10)
pw = builder_pw.construct_full()
pw.add_model(theo_par, exp_par)

# build data container
builder_dc = BuildDataContainer_fromOBJECTS(pw, exp_par, [theo_par])
dc = builder_dc.construct()

# dc.pw.add_model(theo_par, exp_par)

print(dc.pw.exp)
# print(dc.pw.CovT)


# ### write to hdf5
casenum = 1
case_file = './test.hdf5'
dc.to_hdf5(case_file, casenum)


# ### Buld from hdf5
builder_theo_par_h5 = BuildTheoreticalParameters_fromHDF5('test', case_file, casenum, Ta_pair)
theo_par_h5 = builder_theo_par_h5.construct()

builder_pw_h5 = BuildPointwiseContainer_fromHDF5(case_file, casenum)
pw_h5 = builder_pw_h5.construct_lite_w_CovT()

builder_dc_h5 = BuildDataContainer_fromOBJECTS(pw, exp_par, [theo_par])
dc_h5 = builder_dc_h5.construct()

print(dc_h5.pw.exp)


#%% testing using the director class
print()
print('Now testing using the director class')
print()
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
from ATARI.utils.misc import fine_egrid 
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
director.build_product()
dc = builder.product
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
director.build_product()
dc_h5 = builder.product
# dc_h5 = director.construct()

print(dc_h5.pw.exp)
