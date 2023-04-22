
import numpy as np
import scipy.stats as sts

from scipy.linalg import block_diag
from numpy.linalg import inv

from ATARI.theory.scattering_params import FofE_recursive
from ATARI.theory.scattering_params import gstat



def get_parameter_grid(energy_grid, average_parameters, spin_group, dE, dGt):

    Gtot_avg = average_parameters['Gn'][spin_group] + average_parameters['Gg'][spin_group]
    max_Gtot = sts.chi2.ppf(0.99, 1, loc=0, scale=average_parameters['Gn'][spin_group])/1 + average_parameters['Gg'][spin_group]
    min_Gtot = average_parameters['Gg'][spin_group]
    max_Gtot = max_Gtot*1e-3
    min_Gtot = min_Gtot*1e-3

    max_Elam = max(energy_grid) + Gtot_avg/10e3
    min_Elam = min(energy_grid) - Gtot_avg/10e3

    Elam_features = np.arange(min_Elam, max_Elam, dE)
    Gtot_features = np.arange(min_Gtot, max_Gtot, dGt)

    return Elam_features, Gtot_features


def get_resonance_matrix(E, particle_pair, Elam_features, Gtot_features):
    
    number_of_resonances = len(Elam_features)*len(Gtot_features)
    Resonance_Matrix = np.zeros((len(E), number_of_resonances))

    lwave = 0
    _, P, phi, k = FofE_recursive(E, particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
    g = gstat(3.0, particle_pair.I, particle_pair.i)
    kinematic_constant = (4*np.pi*g/k**2)
    potential_scattering = kinematic_constant * np.sin(phi)**2

    for iElam, Elam in enumerate(Elam_features):
        for iGtot, Gtot in enumerate(Gtot_features):

            _, PElam, _, _ = FofE_recursive([Elam], particle_pair.ac, particle_pair.M, particle_pair.m, lwave)
            PPElam = P/PElam
            A_column =  kinematic_constant * Gtot * ( Gtot*PPElam**2*np.cos(2*phi) /4 /((Elam-E)**2+(Gtot*PPElam/2)**2) 
                                                    -(Elam-E)*PPElam*np.sin(2*phi) /2 /((Elam-E)**2+(Gtot*PPElam/2)**2) )  

            vertical_index = (iElam)*len(Gtot_features) + iGtot
            Resonance_Matrix[:, vertical_index] = A_column

    return Resonance_Matrix, potential_scattering


def convert_2_xs(exp, CovT):

    xs_theo = (-1/exp.redpar.val.n)*np.log(exp.theo.theo_trans)
    xs_exp = (-1/exp.redpar.val.n)*np.log(exp.trans.exp_trans)

    exp.theo['theo_xs'] = xs_theo
    exp.trans['exp_xs'] = xs_exp

    dXi_dn = (1/exp.redpar.val.n**2) * np.log(exp.theo.theo_trans)
    dXi_dT = (-1/exp.redpar.val.n) * (1/exp.theo.theo_trans)

    Jac = np.vstack((np.diag(dXi_dT),dXi_dn))
    Cov = block_diag(CovT,exp.redpar.unc.n**2)
    CovXS = Jac.T @ Cov @ Jac

    Lxs = np.linalg.cholesky(inv(CovXS))

    return exp, CovXS, Lxs


def get_bound_arrays(nfeat, lb, ub):
    return np.ones(nfeat)*lb, np.ones(nfeat)*ub




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