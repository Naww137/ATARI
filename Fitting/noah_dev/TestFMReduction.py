# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import h5py
import scipy.stats as sts

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.xs import SLBW
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.theory.scattering_params import gstat
from ATARI.utils.datacontainer import DataContainer
from ATARI.utils.atario import fill_resonance_ladder

from numpy.linalg import inv
from scipy.linalg import block_diag

from scipy.optimize import lsq_linear
from qpsolvers import solve_qp
from scipy.optimize import linprog

import functions as fn 


# %%
# %matplotlib widget

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


case_file = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/TestFMReduction.hdf5'
dataset_range = (0, 10)
samples_not_generated = generate(Ta_pair, exp, 
                                        'syndat_SLBW', 
                                        dataset_range, 
                                        case_file,
                                        fixed_resonance_ladder=None, 
                                        open_data=None,
                                        vary_Erange=None,
                                        use_hdf5=True,
                                        overwrite = True
                                                                    )

# %%
import ATARI.atari_io.hdf5 as io
from ATARI.utils.io import parameters
from ATARI.utils.io import pointwise
from ATARI.utils.misc import fine_egrid 

casenum = 3
theo_resladder = pd.read_hdf(case_file, f'sample_{casenum}/theo_par')
exp_pw, exp_cov = io.read_experimental(case_file, casenum)

print(type(theo_resladder))  # Print the type of the `data` object
# Further inspection of the `data` object, such as printing the first few rows
print(theo_resladder.head())

exp_par = parameters.ExperimentalParameters(exp.redpar.val.n, exp.redpar.unc.n, 1e-2)
theo_par = parameters.Parameters(Ta_pair, theo_resladder, exp_par)

pwfine = pd.DataFrame({'E':fine_egrid(energy_grid,100)})
pw = pointwise.Pointwise_Container(exp_pw, pwfine)

pw.add_experimental(exp_pw, exp_cov, exp_par)
pw.add_model(theo_par, 'theo')


figure()
# plot(pw.fine.E, pw.fine.theo_xs)
plot(pw.exp.E, pw.exp.theo_xs)
errorbar(pw.exp.E, pw.exp.exp_xs, yerr=pw.exp.exp_xs_unc, fmt='.', capsize=2)
# ylim([-max_xs*.1, max_xs*1.25])


# %%



