
import numpy as np
import pandas as pd
import scipy.stats as sts

from scipy.linalg import block_diag
from numpy.linalg import inv

from ATARI.theory.scattering_params import FofE_recursive
from ATARI.theory.scattering_params import gstat

from operator import itemgetter
from itertools import groupby

from classes import FeatureBank

# ================================================================================================
# Functions for setting up the feature bank and constriants
# ================================================================================================

def get_parameter_grid(energy_grid, average_parameters, spin_group, dE, dGt):

    # get range of total widths from chi2 cdfs on Gn and Gg
    Gtot_avg = average_parameters['Gn'][spin_group] + average_parameters['Gg'][spin_group]
    maxGn = sts.chi2.ppf(0.99, 1, loc=0, scale=average_parameters['Gn'][spin_group])/1
    maxGg = sts.chi2.ppf(0.99, 1000, loc=0, scale=average_parameters['Gg'][spin_group])/1000
    minGg = sts.chi2.ppf(0.01, 1000, loc=0, scale=average_parameters['Gg'][spin_group])/1000
    max_Gtot = maxGn+maxGg
    min_Gtot = minGg
    max_Gtot = max_Gtot/1e3
    min_Gtot = min_Gtot/1e3

    # allow Elambda to be just outside of the window
    max_Elam = max(energy_grid) + Gtot_avg/10e3
    min_Elam = min(energy_grid) - Gtot_avg/10e3

    # define grid
    Elam_features = np.arange(min_Elam, max_Elam, dE)
    Gtot_features = np.arange(min_Gtot, max_Gtot, dGt/1e3)

    return Elam_features, Gtot_features


def get_resonance_feature_bank(E, particle_pair, Elam_features, Gtot_features):
    
    E = np.array(E)
    number_of_resonances = len(Elam_features)*len(Gtot_features)
    Resonance_Matrix = np.zeros((len(E), number_of_resonances))
    feature_pairs = []

    lwave = 0
    _, P, phi, k = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    g = gstat(3.0, particle_pair.I, particle_pair.i)
    kinematic_constant = (4*np.pi*g/k**2)
    potential_scattering = kinematic_constant * np.sin(phi)**2 

    for iElam, Elam in enumerate(Elam_features):
        for iGtot, Gtot in enumerate(Gtot_features):
            # Gtot = Gtot/1e3*1e3 
            feature_pairs.append([Elam,Gtot])

            _, PElam, _, _ = FofE_recursive([Elam], particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
            PPElam = P/PElam
            # A_column =  kinematic_constant * Gtot * ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
            #                                         -(Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  
            A_column =  kinematic_constant* ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
                                            - (Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  
            
            vertical_index = (iElam)*len(Gtot_features) + iGtot
            Resonance_Matrix[:, vertical_index] = A_column

    return Resonance_Matrix, potential_scattering, np.array(feature_pairs)


# 


def get_bound_arrays(nfeat, bounds):
    return np.ones(nfeat)*bounds[0], np.ones(nfeat)*bounds[1]


# def convert_2_xs(exp, CovT):
#     xs_theo = (-1/exp.redpar.val.n)*np.log(exp.theo.theo_trans)
#     xs_exp = (-1/exp.redpar.val.n)*np.log(exp.trans.exp_trans)
#     exp.theo['theo_xs'] = xs_theo
#     exp.trans['exp_xs'] = xs_exp
#     dXi_dn = (1/exp.redpar.val.n**2) * np.log(exp.theo.theo_trans)
#     dXi_dT = (-1/exp.redpar.val.n) * (1/exp.theo.theo_trans)
#     Jac = np.vstack((np.diag(dXi_dT),dXi_dn))
#     Cov = block_diag(CovT,exp.redpar.unc.n**2)
#     CovXS = Jac.T @ Cov @ Jac
#     Lxs = np.linalg.cholesky(inv(CovXS))
#     return exp, CovXS, Lxs

# def get_0Trans_constraint(A, max_xs, index_0Trans, E):
#     constraint = np.array([max_xs]*len(E))
#     constraint[index_0Trans] = -constraint[index_0Trans]
#     constraint_mat = A.copy()
#     constraint_mat[index_0Trans, :] = -constraint_mat[index_0Trans, :]
#     return constraint_mat, constraint

# def get_0Trans_constraint(A, max_xs, index_0Trans):
#     constraint = - np.array([max_xs]*len(index_0Trans))
#     constraint_mat = -A.copy()[index_0Trans, :]
#     return constraint_mat, constraint

def get_0Trans_constraint(exp_E, index_0T, max_xs, particle_pair, feature_pairs):
    
    # group consecutive ranges of 0T
    consecutive_ranges = []
    for k,g in groupby(enumerate(index_0T),lambda x:x[0]-x[1]):
        group = (map(itemgetter(1),g))
        group = list(map(int,group))
        consecutive_ranges.append((group[0],group[-1]))

    # create a fine grid in energy where we have blackout resonances
    fineE_0T = []
    for each_group in consecutive_ranges:
        Emin = exp_E[each_group[0]]
        Emax = exp_E[each_group[1]]
        fineE = np.linspace(Emin, Emax, max(int(abs(Emax-Emin)*5e1),1)  )
        fineE_0T.extend(fineE)
    
    # calculate resonance matrix on fine E grid for constraint
    E = np.array(fineE_0T)
    number_of_resonances = np.shape(feature_pairs)[0]
    Constraint_Matrix = np.zeros((len(E), number_of_resonances))
    Constraint = - np.array([max_xs]*len(E))

    lwave = 0
    _, P, phi, k = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    g = gstat(3.0, particle_pair.I, particle_pair.i)
    kinematic_constant = (4*np.pi*g/k**2)

    for ifeature, resonance in enumerate(feature_pairs):

        Elam = resonance[0]
        Gtot = resonance[1]

        _, PElam, _, _ = FofE_recursive([Elam], particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
        PPElam = P/PElam
        # A_column =  kinematic_constant * Gtot * ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
        #                                         -(Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) ) 
        A_column =  kinematic_constant* ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
                                            - (Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  

        Constraint_Matrix[:, ifeature] = - A_column

    return Constraint_Matrix ,Constraint 



def remove_nan_values(full_xs, full_cov, full_pscat, full_feature_matrix):
    index_0T = np.argwhere(np.isnan(full_xs)).flatten()
    index_finiteT = np.argwhere(np.isfinite(full_xs)).flatten()

    cov = full_cov.copy()[index_finiteT, :]
    cov = cov[:, index_finiteT]

    xs = full_xs[index_finiteT]
    pscat = full_pscat[index_finiteT]

    feature_matrix = full_feature_matrix[index_finiteT, :]

    return xs, cov, pscat, feature_matrix, index_0T

# from typing import Protocol
# class get_qp_inputsProtocol(Protocol):
#     @

# def get_qp_inputs(exp_E, exp_xs, cov_xs, max_xs, particle_pair, feature_bank: FeatureBank):
#     nfeatures = np.shape(feature_bank.feature_matrix)[1]
    
#     # remove nan values in xs and cov for solver
#     b, cov, pscat, A, index_0T = remove_nan_values(exp_xs, cov_xs, feature_bank.potential_scattering, feature_bank.feature_matrix)
#     b = b-pscat

#     # get bounds and constraints
#     lb, ub = get_bound_arrays(nfeatures, 0, 1)
#     G, h = get_0Trans_constraint(exp_E, index_0T, max_xs, particle_pair, feature_bank.feature_pairs)

#     # Cast into quadratic program 
#     P = A.T @ inv(cov) @ A
#     q = - A.T @ inv(cov) @ b

#     return P, q, G, h, lb, ub, index_0T


def get_reduced_features(full_feature_matrix, solution_ws, w_threshold, feature_pairs):
    index_w_surviving = np.argwhere(solution_ws>w_threshold).flatten()
    reduced_solw = solution_ws[index_w_surviving]
    reduced_feature_matrix = full_feature_matrix[:, index_w_surviving]
    reduced_feature_pairs = feature_pairs[index_w_surviving,:]
    return reduced_feature_matrix, reduced_feature_pairs, reduced_solw

# ================================================================================================
# Functions for decoding the feature bank
# ================================================================================================

def get_resonance_ladder_from_feature_bank(weights, Elam_features, Gtot_features, threshold):
    feature_indices = np.argwhere(weights>threshold).flatten()
    resonances = []
    for ifeature in feature_indices:
        Efeature_index, Gfeature_index = divmod(ifeature, len(Gtot_features))
        Elam = Elam_features[Efeature_index]
        Gt = Gtot_features[Gfeature_index]*1e3
        w = weights[ifeature]
        Gnx = Gt*w
        Gg = Gt-Gnx
        resonances.append([Elam, Gt, Gnx, Gg, w])
    resonance_ladder = pd.DataFrame(resonances, columns=['E', 'Gt', 'Gnx', 'Gg', 'w'])
    return resonance_ladder




# def calculate_integral_FoMs(weights, Elam_features, Gtot_features, threshold, datacon):
#     est_resonance_ladder = get_resonance_ladder_from_matrix(weights, Elam_features, Gtot_features, threshold)
#     est_resonance_ladder = fill_resonance_ladder(est_resonance_ladder, datacon.particle_pair, J=3.0, chs=1.0, lwave=0.0, J_ID=1.0)

#     xs = Resonance_Matrix@ws_vs_factor[9]+potential_scattering.flatten()
#     trans = np.exp(-exp.redpar.val.n*xs)

#     fineE = fine_egrid(datacon.pw_exp.E, 1e2)
#     est_xs_tot, _, _ = SLBW(fineE, datacon.particle_pair, est_resonance_ladder)
#     theo_xs_tot, _, _ = SLBW(fineE, datacon.particle_pair, datacon.theo_resonance_ladder)
#     MSE = trapezoid((est_xs_tot-theo_xs_tot)**2, fineE)
#     bias = est_xs_tot-theo_xs_tot

#     return MSE, bias



## Cholesky decomposition unit test
# A = np.array([[1,2],[1,2]])
# b = np.array([1,1])
# x = np.array([1,1])
# C = np.array([[2,1],[1,2]])
# L = np.linalg.cholesky(inv(C))
# Ap = L.T@A
# bp = L.T@b

# print((Ap@x-bp)@(Ap@x-bp).T)
# print((A@x-b)@inv(C)@(A@x-b).T)
# (Ap@x-bp)@(Ap@x-bp).T == (A@x-b)@inv(C)@(A@x-b).T



# ================================================================================================
# Functions runnning the linear or quadratic programs
# ================================================================================================