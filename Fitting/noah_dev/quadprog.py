# %%
import numpy as np
import pandas as pd
import os
from matplotlib.pyplot import *
import h5py

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
import ATARI.atari_io.hdf5 as io
from ATARI.theory.xs import SLBW
from ATARI.theory.scattering_params import FofE_recursive

from numpy.linalg import inv

from scipy.optimize import lsq_linear
from qpsolvers import solve_qp

import functions as fn 


# %%

def gn2G(row):
    S, P, phi, k = FofE_recursive([row.E], Ta_pair.ac, Ta_pair.M, Ta_pair.m, row.lwave)
    Gnx = 2*np.sum(P)*row.gnx2
    return Gnx.item()

def G2gn(row):
    S, P, phi, k = FofE_recursive([row.E], Ta_pair.ac, Ta_pair.M, Ta_pair.m, row.lwave)
    gnx2 = row.Gnx/2/np.sum(P)
    return gnx2.item()


#%%
ac = 0.81271  # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity
l_max = 1       # highest order l-wave to consider


Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={})



spin_groups = [ (3.0,1,0)] # , (4.0,1,[0]) ]
average_parameters = pd.DataFrame({ 'dE'    :   {'3.0':8.79, '4.0':4.99},
                                    'Gg'    :   {'3.0':64.0, '4.0':64.0},
                                    'gn2'    :   {'3.0':46.4, '4.0':35.5}  })


E_min_max = [560, 580]
energy_grid = E_min_max

input_options = {'Add Noise': True,
                'Calculate Covariance': True,
                'Compression Points':[],
                'Grouping Factors':None}

experiment_parameters = {'bw': {'val':0.0256,    'unc'   :   0}}

# initialize experimental setup
exp = Experiment(energy_grid, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)


# resonance_ladder = Ta_pair.sample_resonance_ladder(E_min_max, spin_groups, average_parameters, False)
resonance_ladder = pd.DataFrame({'E':[570], 'J':[3.0], 'chs':[1.0], 'lwave':[0.0], 'J_ID':[1.0], 'gnx2':[100], 'Gg':[750]})
Gnx = resonance_ladder.apply(lambda row: gn2G(row), axis=1)
resonance_ladder['Gtot'] = Gnx + resonance_ladder['Gg']

print(resonance_ladder)

true, _, _ = SLBW(exp.energy_domain, Ta_pair, resonance_ladder)
df_true = pd.DataFrame({'E':exp.energy_domain, 'theo_trans':np.exp(-exp.redpar.val.n*true)})

exp.run(df_true)

# figure()
# plot(exp.trans.E, exp.trans.exp_trans, '.')
# plot(exp.theo.E, exp.theo.theo_trans)
# axhline(0)

# %%


average_parameters.loc[:,['Gn']] = average_parameters['gn2']/12.5

Elam_features, Gtot_features = fn.get_parameter_grid(energy_grid, average_parameters, '3.0', 1e-1, 1e-2)
# Elam_features, Gtot_features = [565, 569, 570, 571, 575], [.7585]
# print(Elam_features, Gtot_features)
print(len(Elam_features)*len(Gtot_features))

E = np.sort(exp.energy_domain)
Resonance_Matrix, potential_scattering = fn.get_resonance_matrix(E, Ta_pair, Elam_features, Gtot_features)
nfeatures = np.shape(Resonance_Matrix)[1]

# sort everything
CovT = np.flip(exp.CovT)
exp.trans.sort_values('E', inplace=True)
exp.theo.sort_values('E', inplace=True)

# convert to xs - linear error propagation is not exact
exp, CovXS, Lxs = fn.convert_2_xs(exp, CovT)

# %%

# setup
b = (np.array(exp.trans.exp_xs)-potential_scattering).flatten()
A = Resonance_Matrix

lb, ub = fn.get_bound_arrays(nfeatures, 0, 1)

# Cast into linear least squares
bp = Lxs.T @ b
Ap = Lxs.T @ A
# Cast into quadratic program 
P = A.T @ inv(CovXS) @ A
q = - A.T @ inv(CovXS) @ b


# solve 
res_ls = lsq_linear(Ap, bp, bounds=(lb,ub), 
                            lsmr_tol='auto', 
                            lsq_solver='lsmr',
                            max_iter = 500, 
                            verbose=1)

qp_res = solve_qp(P, q, G=None, h=None, A=None, b=None, lb=lb, ub=ub, solver="cvxopt")


# %%
# print(Elam_features)
# print(res_ls.x)
# print(qp_res)

figure()
errorbar(exp.trans.E, exp.trans.exp_xs, yerr=np.sqrt(np.diag(CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
plot(E, Resonance_Matrix@res_ls.x+potential_scattering.flatten(), lw=5, color='cornflowerblue')
plot(E, Resonance_Matrix@qp_res+potential_scattering.flatten(), color='k')
show()


# %% Now add constraints

# %%
# from ATARI.utils.stats import chi2_val

# print(sum(qp_res))

# basename = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/figures/'
# x_constrained = []
# ifig = 0
# chi2 = []

# for fac in np.linspace(0,1,100):
#     G = np.ones(len(qp_res))
#     h = np.array([[sum(qp_res)*fac]])
#     qp_x_c = solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, solver="cvxopt")

#     figure()
#     errorbar(exp.trans.E, exp.trans.exp_xs, yerr=np.sqrt(np.diag(CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
#     plot(E, Resonance_Matrix@qp_x_c+potential_scattering.flatten(), color='k')
#     title(f'Factor: {fac}')
#     savefig(os.path.join(basename,'constrain_w_giffigs', f'fig_{ifig}.png'))
#     close()
#     ifig += 1
    
#     chi2.append(chi2_val(Resonance_Matrix@qp_x_c+potential_scattering.flatten(), exp.trans.exp_xs, CovXS).flatten())
#     x_constrained.append(qp_x_c)
    

# # %%
# figure()
# scatter(np.linspace(0,1,100), chi2)
# xlabel('factor')
# ylabel(r'$\chi^2$')
# title(r'$\chi^2$ of fit with changing fraction of initial weights')
# savefig(os.path.join(basename, 'constrain_w_chi2fac.png'))

# # %%
# # index = 2
# # figure()
# # errorbar(exp.trans.E, exp.trans.exp_xs, yerr=np.sqrt(np.diag(CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
# # # plot(E, Resonance_Matrix@qp_x+potential_scattering.flatten(), lw=5, color='cornflowerblue')
# # plot(E, Resonance_Matrix@x_constrained[index]+potential_scattering.flatten(), color='k')

# # %%
# import imageio
# images = []
# for job_number in range(len(x_constrained)-1, 0 , -1):
#     images.append(imageio.imread(os.path.join(basename, 'constrain_w_giffigs', f'fig_{job_number}.png')))
# imageio.mimsave(os.path.join(basename, 'constrain_w.gif'), images)

# %%



