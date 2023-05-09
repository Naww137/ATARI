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

experiment_parameters = {'bw': {'val':0.0256,   'unc'   :   0},
                         'n':  {'val':0.06,     'unc'   :0}}

# initialize experimental setup
exp = Experiment(energy_grid, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)


resonance_ladder = Ta_pair.sample_resonance_ladder(E_min_max, spin_groups, average_parameters, False)
# resonance_ladder = pd.DataFrame({'E':[570], 'J':[3.0], 'chs':[1.0], 'lwave':[0.0], 'J_ID':[1.0], 'gnx2':[100], 'Gg':[750]})
# resonance_ladder = pd.DataFrame({'E':[575], 'J':[3.0], 'chs':[1.0], 'lwave':[0.0], 'J_ID':[1.0], 'gnx2':[5000], 'Gg':[75]})
true, _, _ = SLBW(exp.energy_domain, Ta_pair, resonance_ladder)
df_true = pd.DataFrame({'E':exp.energy_domain, 'theo_trans':np.exp(-exp.redpar.val.n*true)})

exp.run(df_true)

from ATARI.utils.datacontainer import DataContainer
from ATARI.theory.experimental import trans_2_xs

dc = DataContainer()
dc.add_theoretical(Ta_pair, resonance_ladder)
threshold_0T = 1e-2
dc.add_experimental(exp, threshold=threshold_0T)

max_xs, _ = trans_2_xs(threshold_0T, dc.n)

figure()
plot(dc.pw_fine.E, dc.pw_fine.theo_xs)
errorbar(dc.pw_exp.E, dc.pw_exp.exp_xs, yerr=dc.pw_exp.exp_xs_unc, fmt='.', capsize=2)
ylim([-max_xs*.1, max_xs*1.25])


# %% [markdown]
# ## Full feature bank and unconstrained solve

# %%
# Gts = []
# for i in range(1000):
#     resonance_ladder = Ta_pair.sample_resonance_ladder(E_min_max, spin_groups, average_parameters, False)
#     dc.add_theoretical(Ta_pair, resonance_ladder)
#     Gts.extend(np.array(resonance_ladder.Gt))

# print(min(Gts))
# print(max(Gts))

# %%
average_parameters.loc[:,['Gn']] = average_parameters['gn2']/12.5
Elam_features, Gtot_features = fn.get_parameter_grid(energy_grid, average_parameters, '3.0', 1e-1, 1e-1)
# Gtot_features = [0.755, 0.758, 0.760]

# get resonance feature matrix
Resonance_Matrix, potential_scattering = fn.get_resonance_feature_bank(dc.pw_exp.E, dc.particle_pair, Elam_features, Gtot_features)
nfeatures = np.shape(Resonance_Matrix)[1]
print(nfeatures)

# %%

def solve_qp_w_unconstrained(exp_xs, cov_xs, potential_scattering, max_xs, feature_matrix):

    nfeatures = np.shape(feature_matrix)[1]
    
    # remove nan values in xs and cov for solver
    b, cov, pscat, A, index_0T = fn.remove_nan_values(np.array(exp_xs), np.array(cov_xs), potential_scattering, feature_matrix)
    b = b-pscat

    # get bounds and constraints
    lb, ub = fn.get_bound_arrays(nfeatures, 0, 1)
    G, h = fn.get_0Trans_constraint(feature_matrix, max_xs, index_0T)

    # Cast into quadratic program 
    P = A.T @ inv(cov) @ A
    q = - A.T @ inv(cov) @ b

    # solve linear program
    # lp_res = linprog(q, A_ub=G, b_ub=h, bounds=np.array([lb, ub]).T)
    # solve quadratic program
    wresult_unconstrained = solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, 
                                                                solver="cvxopt",
                                                                verbose=False,
                                                                abstol=1e-12,
                                                                reltol=1e-12,
                                                                feastol= 1e-8,
                                                                maxiters = 100)
    return wresult_unconstrained


def get_qp_inputs(exp_xs, cov_xs, potential_scattering, max_xs, feature_matrix):
    nfeatures = np.shape(feature_matrix)[1]
    
    # remove nan values in xs and cov for solver
    b, cov, pscat, A, index_0T = fn.remove_nan_values(np.array(exp_xs), np.array(cov_xs), potential_scattering, feature_matrix)
    b = b-pscat

    # get bounds and constraints
    lb, ub = fn.get_bound_arrays(nfeatures, 0, 1)
    G, h = fn.get_0Trans_constraint(feature_matrix, max_xs, index_0T)

    # Cast into quadratic program 
    P = A.T @ inv(cov) @ A
    q = - A.T @ inv(cov) @ b

    return P, q, G, h, lb, ub, index_0T

# unconstrained_w = solve_qp_w_unconstrained(dc.pw_exp.exp_xs, dc.CovXS, potential_scattering.flatten(), max_xs, Resonance_Matrix)

P, q, G, h, lb, ub, index_0T = get_qp_inputs(dc.pw_exp.exp_xs, dc.CovXS, potential_scattering.flatten(), max_xs, Resonance_Matrix)


# %%
# unconstrained_w = solve_qp(P, q, G=G, h=h, A=None, b=None, lb=lb, ub=ub, 
#                                                                 solver="cvxopt",
#                                                                 verbose=True,
#                                                                 abstol=1e-10,
#                                                                 reltol=1e-10,
#                                                                 feastol= 1e-7,
#                                                                 maxiters = 100)

# %%
# solve linear program
w_threshold = 1e-10

lp_unconstrained_w = linprog(q, A_ub=G, b_ub=h, bounds=np.array([lb, ub]).T)

print(np.sum(lp_unconstrained_w.x))
np.count_nonzero(lp_unconstrained_w.x>w_threshold)

# %%
### Constrain and solve
w_constraint = 10
G_wc = np.vstack([G,np.ones(len(P))])
h_wc = np.append(h, w_constraint)
lp_constrained_w = linprog(q, A_ub=G_wc, b_ub=h_wc, bounds=np.array([lb, ub]).T)

print(np.sum(lp_constrained_w.x))
print(np.count_nonzero(lp_constrained_w.x>w_threshold))

# %%
### reduce and solve
full_feature_matrix = Resonance_Matrix.copy()
index_w_surviving = np.argwhere(lp_constrained_w.x>w_threshold).flatten()
reduced_feature_matrix = full_feature_matrix[:, index_w_surviving]

P_reduced, q_reduced, G_reduced, h_reduced, lb_reduced, ub_reduced, index_0T_reduced = get_qp_inputs(dc.pw_exp.exp_xs, 
                                                                                     dc.CovXS, potential_scattering.flatten(), 
                                                                                     max_xs, reduced_feature_matrix )

# unconstrained_w_reduced = solve_qp(P_reduced, q_reduced, G=G_reduced, h=h_reduced, A=None, b=None, lb=lb_reduced, ub=ub_reduced, 
#                                                                                                 solver="cvxopt",
#                                                                                                 verbose=True,
#                                                                                                 abstol=1e-10,
#                                                                                                 reltol=1e-10,
#                                                                                                 feastol= 1e-7,
#                                                                                                 maxiters = 100)

# %%
# print(resonance_ladder)
# unconstrained_est_resladder = fn.get_resonance_ladder_from_feature_bank(lp_unconstrained_w.x, Elam_features, Gtot_features, 1e-10)
# dc.add_estimate(unconstrained_est_resladder, est_name='est_unconstrained')

constrained_est_resladder = fn.get_resonance_ladder_from_feature_bank(lp_constrained_w.x, Elam_features, Gtot_features, 1e-10)
dc.add_estimate(constrained_est_resladder, est_name='est_constrained')


# %%

figure()
errorbar(dc.pw_exp.E, dc.pw_exp.exp_xs, yerr=np.sqrt(np.diag(dc.CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
# plot(dc.pw_exp.E, Resonance_Matrix@lp_unconstrained_w.x+potential_scattering.flatten(), lw=2, color='purple')
plot(dc.pw_exp.E, Resonance_Matrix@lp_constrained_w.x+potential_scattering.flatten(), lw=2, color='purple')
# plot(dc.pw_exp.E, Resonance_Matrix@unconstrained_w+potential_scattering.flatten(), color='blue')
# plot(dc.pw_exp.E, reduced_feature_matrix@unconstrained_w_reduced+potential_scattering.flatten(), color='blue')


# plot(dc.pw_fine.E, dc.pw_fine.est_unconstrained_xs)
# plot(dc.pw_fine.E, dc.pw_fine.est_constrained_xs)

ylim([-5, max_xs*1.5])
show()





# %% [markdown]
# ### Now perform bisection method to find a weight constraint for 1-5 resonances
# 
# Need to define lower threshold for weight constraint because if there is a blackout resonance, the problem becomes infeasible if the constrained weights don't allow the blackout constraint to be met.
# 
# 
# 

# # %%
# def solve_qp_w_constraint(P, q, G, h, lb, ub, w_constraint, w_threshold=1e-10):
#     G_wc = np.vstack([G,np.ones(len(P))])
#     h_wc = np.append(h, w_constraint)
#     qp_res_c = solve_qp(P, q, G=G_wc, h=h_wc, A=None, b=None, lb=lb, ub=ub, 
#                                                         solver="cvxopt",
#                                                         verbose=False,
#                                                         abstol=1e-10,
#                                                         reltol=1e-10,
#                                                         feastol= 1e-8,
#                                                         maxiters = 100) 
#     return qp_res_c, np.count_nonzero(qp_res_c>w_threshold)

# def get_minimum_weight_factor(unconstrained_weight):
#     failed = True
#     print(f'Unconstrained weight: {unconstrained_weight}')
#     for fac in np.linspace(0,1,50):
#         w = unconstrained_weight*fac
#         try:
#             res, nonzero = solve_qp_w_constraint(P, q, G, h, lb, ub, w)
#             failed = False
#         except:
#             print(f'Failed at weight: {w}')

#         if not failed:
#             break
        
#     return w, nonzero

# def secant_method(f, x0, y0, x1, offset, tol=1e-5, n=0):
#     # increment counter
#     n += 1

#     # calculate weights and number of resonances at endpoints
#     qp_res, numres = f(x1)
#     y1 = numres-offset

#     # calculate next root approximation
#     xn = x1 - y1 * ((x1 - x0) / (y1 - y0))
#     # if nan or inf (bc same y val) divide by non-zero  #TODO: find the best normalization here, 1 works but maybe something else is faster?
#     # if np.isnan(xn) or np.isinf(xn):
#     #     xn = x1 - y1 * ((x1 - x0) /1)

#     # check tolerance condition - could do list comprehension here to get w_constraints corrresponding to multiple different resonances in one go
#     if -tol < y1 < tol:
#         return xn, n, qp_res
    
#     # recursive call with updated interval
#     return secant_method(f, x1, y1, xn, offset, tol=tol, n=n)



# def validate_interval(f, x0, x1):
#     return f(x0) * f(x1) < 0

# # solve for root using bisection method
# # def bisection_method(f, interval, tol):
# #     """
# #     param f: find root for function
# #     param interval: within range
# #     param tol: root accuracy tolerance
# #     """

# #     # extract interval start and end points
# #     x0, x1 = interval[0], interval[1]

# #     # check interval can be used to solve for root
# #     if not validate_interval(f, x0, x1):
# #         return

# #     # iterations required to find the root within a specified error bound
# #     n = error_bound(x0, x1, tol)

# #     counter = 1

# #     # iterate over error bound
# #     while True:

# #         # calculate root approximation
# #         root_approx = x0 + ((x1 - x0) / 2)

# #         # evaluate y at current estimate
# #         y = f(root_approx)

# #         # check tolerance condition
# #         if -tol < y < tol:
# #             # check that error bound actually worked
# #             print(counter, n)

# #             # return root approximation
# #             return root_approx

# #         # check if next segment is left of bisection
# #         if validate_interval(f, x0, root_approx):
# #             x1 = root_approx
# #         else:
# #             x0 = root_approx

# #         # increment counter
# #         counter += 1


# def bisection_method(f, x0, y0, x1, offset, tol=1e-5, n=0):
#     # increment counter
#     n += 1

#     # calculate weights and number of resonances at endpoints
#     qp_res, numres = f(x1)
#     y1 = numres-offset

#     # calculate next root approximation
#     xn = x0 + ((x1 - x0) / 2)

#     # check tolerance condition - could do list comprehension here to get w_constraints corrresponding to multiple different resonances in one go
#     if -tol < y1 < tol:
#         return xn, n, qp_res
    
#     # recursive call with updated interval
#     return secant_method(f, x1, y1, xn, offset, tol=tol, n=n)

# # %%
# # iw_keep = np.argwhere(qp_res>w_threshold).flatten()
# # new_full_feature_matrix = full_feature_matrix[:, iw_keep]
# # y0, y1 = number_of_surviving_resonances(P, q, G, h, 0, w_threshold), number_of_surviving_resonances(P, q, G, h, np.sum(qp_res), w_threshold)

# full_feature_matrix = Resonance_Matrix
# w_threshold = 1e-10
# def func(w):
#     return solve_qp_w_constraint(P, q, G, h, lb, ub, w, w_threshold)

# unconstrained_weight = np.sum(unconstrained_w)
# if any(index_0T):
#     min_w, min_nonzero = get_minimum_weight_factor(unconstrained_weight)
# else:
#     min_w = 0; min_nonzero = 0

# target_numres = 5
# w_constraint, niter, qp_res = secant_method(func, min_w, min_nonzero-target_numres, unconstrained_weight*0.4, target_numres, tol=1e-2, n=0)
# # w_constraint, niter, qp_res = secant_method(func, unconstrained_weight*0.2, np.count_nonzero(unconstrained_w>w_threshold), min_w, target_numres, tol=1e-2, n=0)

# # %%


# # %%
# # x_constrained = []

# # for fac in np.linspace(0,0.6,200):
# #     G_wc = np.vstack([G,np.ones(len(unconstrained_w))])
# #     h_wc = np.append(h, unconstrained_weight*fac)
# #     res = solve_qp(P, q, G=G_wc, h=h_wc, A=None, b=None, lb=lb, ub=ub, 
# #                                                         solver="cvxopt",
# #                                                         verbose=False,
# #                                                         abstol=1e-10,
# #                                                         reltol=1e-10,
# #                                                         feastol= 1e-8,
# #                                                         maxiters = 500)
# #     x_constrained.append(res)

# x_constrained = np.load('ws_vs_wconstraint.npy')
# x_nonzero = []
# for each in x_constrained:
#     try:
#         non0 = np.count_nonzero(each>w_threshold)
#     except:
#         non0 = np.nan
#     x_nonzero.append(non0)

# figure()
# plot(np.linspace(0,0.6,200), x_nonzero, '.')
# # plot( x_nonzero,np.linspace(0,0.6,200), '.')
# # plot(np.linspace(0,0.6,200), np.linspace(0,0.6,200)*)

# # %%
# print(np.count_nonzero(unconstrained_w>w_threshold))
# print(unconstrained_weight)
# print(min_nonzero)
# print(min_w)

# # w_constraint, niter, qp_res = secant_method(func, min_w, -min_nonzero, unconstrained_weight, target_numres, tol=1e-2, n=0)

# # %%
# # constrained_res, nonzero = solve_qp_w_constraint(P, q, G, h, lb, ub, min_w*1.2, w_threshold)
# constrained_res, nonzero = solve_qp_w_constraint(P, q, G, h, lb, ub, w_constraint, w_threshold)
# print(nonzero)

# # %%
# est_resladder = fn.get_resonance_ladder_from_feature_bank(constrained_res, Elam_features, Gtot_features, w_threshold)
# dc.add_estimate(est_resladder)

# figure()

# errorbar(dc.pw_exp.E, dc.pw_exp.exp_xs, yerr=np.sqrt(np.diag(dc.CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
# # plot(dc.pw_exp.E, Resonance_Matrix@lp_res.x+potential_scattering.flatten(), lw=2, color='purple')
# # plot(dc.pw_exp.E, Resonance_Matrix@constrained_res+potential_scattering.flatten(), color='blue')
# plot(dc.pw_fine.E, dc.pw_fine.theo_xs, 'g')
# plot(dc.pw_fine.E, dc.pw_fine.est_xs)

# scatter(np.array(dc.pw_exp.E)[index_0T], np.ones(len(index_0T))*10)
# ylim([-5, max_xs*1.5])

# # %%
# ### reduce feature bank

# index_w_surviving = np.argwhere(constrained_res>w_threshold).flatten()
# reduced_feature_matrix = full_feature_matrix[:, index_w_surviving]

# P_reduced, q_reduced, G_reduced, h_reduced, lb_reduced, ub_reduced, index_0T_reduced = get_qp_inputs(dc.pw_exp.exp_xs, 
#                                                                                      dc.CovXS, potential_scattering.flatten(), 
#                                                                                      max_xs, reduced_feature_matrix )

# unconstrained_w_reduced = solve_qp(P_reduced, q_reduced, G=G_reduced, h=h_reduced, A=None, b=None, lb=lb_reduced, ub=ub_reduced, 
#                                                                                                 solver="cvxopt",
#                                                                                                 verbose=False,
#                                                                                                 abstol=1e-12,
#                                                                                                 reltol=1e-12,
#                                                                                                 feastol= 1e-8,
#                                                                                                 maxiters = 100)

# # %%
# np.count_nonzero(unconstrained_w_reduced>w_threshold)

# # %%
# dc.add_estimate(est_resladder, est_name='est_unconstrained')

# figure()

# errorbar(dc.pw_exp.E, dc.pw_exp.exp_xs, yerr=np.sqrt(np.diag(dc.CovXS)), fmt='.', ecolor='r', color='k', capsize=1, ms=2)
# # plot(dc.pw_exp.E, reduced_feature_matrix@unconstrained_w_reduced+potential_scattering.flatten(), color='blue')
# plot(dc.pw_fine.E, dc.pw_fine.theo_xs, 'g')
# plot(dc.pw_fine.E, dc.pw_fine.est_unconstrained_xs)

# scatter(np.array(dc.pw_exp.E)[index_0T], np.ones(len(index_0T))*10)
# ylim([-5, max_xs*1.5])

# # %%



