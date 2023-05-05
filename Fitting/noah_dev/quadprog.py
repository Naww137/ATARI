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
from ATARI.utils.atario import fill_resonance_ladder

from numpy.linalg import inv

from scipy.optimize import lsq_linear
from qpsolvers import solve_qp

import functions as fn 


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


E_min_max = [550, 600]
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
true, _, _ = SLBW(exp.energy_domain, Ta_pair, resonance_ladder)
df_true = pd.DataFrame({'E':exp.energy_domain, 'theo_trans':np.exp(-exp.redpar.val.n*true)})

exp.run(df_true)

#%%
from ATARI.utils.datacontainer import DataContainer
from ATARI.theory.experimental import trans_2_xs

dc = DataContainer()
dc.add_theoretical(Ta_pair, resonance_ladder)
threshold_0T = 1e-2
dc.add_experimental(exp, threshold=threshold_0T)

max_xs, _ = trans_2_xs(threshold_0T, dc.n)
# print(dc.theo_resonance_ladder)
# print(dc.pw_exp)

# figure()
# plot(exp.trans.E, exp.trans.exp_trans, '.')
# plot(exp.theo.E, exp.theo.theo_trans)
# axhline(0)

# %%


average_parameters.loc[:,['Gn']] = average_parameters['gn2']/12.5

Elam_features, Gtot_features = fn.get_parameter_grid(energy_grid, average_parameters, '3.0', 1e-1, 1e-2)
# Elam_features, Gtot_features = [565, 569, 570, 571, 575], [.7585]
# print(Elam_features, Gtot_features)
Gtot_features = [0.755, 0.758, 0.760]
print(len(Elam_features)*len(Gtot_features))

Resonance_Matrix, potential_scattering = fn.get_resonance_feature_bank(dc.pw_exp.E, dc.particle_pair, Elam_features, Gtot_features)
nfeatures = np.shape(Resonance_Matrix)[1]

# # sort everything
# CovT = np.flip(exp.CovT)
# exp.trans.sort_values('E', inplace=True)
# exp.theo.sort_values('E', inplace=True)

# # convert to xs - linear error propagation is not exact
# exp, CovXS, Lxs = fn.convert_2_xs(exp, CovT)

# %%

# setup
b = (np.array(dc.pw_exp.exp_xs)-potential_scattering).flatten()
A = Resonance_Matrix

lb, ub = fn.get_bound_arrays(nfeatures, 0, 1)

# Cast into linear least squares
Lxs = np.linalg.cholesky(inv(dc.CovXS))
bp = Lxs.T @ b
Ap = Lxs.T @ A
# Cast into quadratic program 
P = A.T @ inv(dc.CovXS) @ A
q = - A.T @ inv(dc.CovXS) @ b


# solve 
res_ls = lsq_linear(Ap, bp, bounds=(lb,ub), 
                            lsmr_tol='auto', 
                            lsq_solver='lsmr',
                            max_iter = 500, 
                            verbose=1)

qp_res = solve_qp(P, q, G=None, h=None, A=None, b=None, lb=lb, ub=ub, 
                                                            solver="cvxopt",
                                                            verbose=False,
                                                            abstol=1e-10,
                                                            reltol=1e-10,
                                                            feastol= 1e-8,
                                                            maxiters = 500)


# # %%
# print(Elam_features)
print(res_ls.x)
print(qp_res)

figure()
errorbar(dc.pw_exp.E, dc.pw_exp.exp_xs, yerr=np.sqrt(np.diag(dc.CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
plot(dc.pw_exp.E, Resonance_Matrix@res_ls.x+potential_scattering.flatten(), lw=5, color='cornflowerblue')
plot(dc.pw_exp.E, Resonance_Matrix@qp_res+potential_scattering.flatten(), color='k')
show()


# %% Now add constraints

# basename = '/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/figures/'
x_constrained = []
for fac in np.linspace(0,1,5):
    G = np.ones(len(qp_res))
    h = np.array([[sum(qp_res)*fac]])

    # qp_x_c = solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, solver="cvxopt")
    qp_x_c = solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, 
                                                        solver="cvxopt",
                                                        verbose=False,
                                                        abstol=1e-10,
                                                        reltol=1e-10,
                                                        feastol= 1e-8,
                                                        maxiters = 500)
    if qp_x_c is None:
        qp_x_c = solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, 
                                                        solver="cvxopt",
                                                        verbose=False,
                                                        abstol=1e-9,
                                                        reltol=1e-9,
                                                        feastol= 1e-7,
                                                        maxiters = 500)
    
    x_constrained.append(qp_x_c)

ws_vs_factor = np.array(x_constrained)
    

#%%





# %%

inonzero = []
nonzero_w = []; threshold = 1e-10
izero_init = np.argwhere(qp_res<threshold).flatten()
num_revived = []

for ifac, fac in enumerate(np.linspace(0,1,5)):
    current_w = ws_vs_factor[ifac]
    izero_thisw = np.argwhere(current_w<threshold)
    nonzero_thisf = np.argwhere(current_w>threshold).flatten()

    inonzero.append(len(nonzero_thisf))
    nonzero_w.append(current_w[nonzero_thisf])
    true_if_revived = [init not in izero_thisw for init in izero_init]
    num_revived.append(np.count_nonzero(true_if_revived))

#%%

figure()
plot(np.linspace(0,1,5), num_revived,'.', label='Revived')
plot(np.linspace(0,1,5), inonzero,'.', label='Non-zero')
legend()
# axhline(1)
xlabel('Fraction of Unconstrained Weight')
ylabel('Number of Features (Res)')

#%%

i = 3
final = np.argwhere([ws_vs_factor[i]>threshold])
# figure(figsize=(10,5))
# # scatter(1, np.log10(qp_res[final].flatten()[0]), color='orange')
# # scatter(1, np.log10(qp_res[final].flatten()[1]), color='blue')
# out = plot(factors.T[:,ws_vs_factor[i]>threshold], np.log10(ws_vs_factor[:,ws_vs_factor[i]>threshold]))
# ylabel('Log10(w)')
# xlabel('Factor')

est_resonance_ladder = fn.get_resonance_ladder_from_matrix(ws_vs_factor[i][ws_vs_factor[i]>threshold], Elam_features, Gtot_features, threshold)
dc.add_estimate(est_resonance_ladder, 'est')
print(dc.pw_exp)

# figure()
# plot()

#%%

# figure()
# scatter(np.linspace(0,1,100), chi2)
# xlabel('factor')
# ylabel(r'$\chi^2$')
# title(r'$\chi^2$ of fit with changing fraction of initial weights')
# savefig(os.path.join(basename, 'constrain_w_chi2fac.png'))

# %%
# index = 15
# figure()
# errorbar(dc.pw_exp.E, dc.pw_exp.exp_xs, yerr=np.sqrt(np.diag(dc.CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
# # plot(E, Resonance_Matrix@qp_x+potential_scattering.flatten(), lw=5, color='cornflowerblue')
# plot(dc.pw_exp.E, Resonance_Matrix@x_constrained[index]+potential_scattering.flatten(), color='k')

# # %%
# import imageio
# images = []
# for job_number in range(len(x_constrained)-1, 0 , -1):
#     images.append(imageio.imread(os.path.join(basename, 'constrain_w_giffigs', f'fig_{job_number}.png')))
# imageio.mimsave(os.path.join(basename, 'constrain_w.gif'), images)

# %%



