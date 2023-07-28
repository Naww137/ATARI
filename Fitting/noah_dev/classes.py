
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

from scipy.optimize import lsq_linear
from qpsolvers import solve_qp
from scipy.optimize import linprog
from numpy.linalg import inv
from scipy.linalg import block_diag

import functions as fn
from ATARI.utils.io.data_container import DataContainer


def get_resonance_ladder_from_feature_pairs(weights, feature_pairs):
    Elam = feature_pairs[:,0]
    Gt = feature_pairs[:,1]*1e3
    weights = weights.flatten()
    # Gnx = Gt*weights
    Gn = weights*1e3
    Gg = Gt-Gn
    resonances = np.array([Elam, Gt, Gn, Gg, weights])
    resonance_ladder = pd.DataFrame(resonances.T, columns=['E', 'Gt', 'Gn', 'Gg', 'w'])
    return resonance_ladder

@dataclass
class FeatureBank:
    feature_matrix: np.ndarray
    feature_pairs: np.ndarray
    potential_scattering: np.ndarray
    nfeatures: int
    w_bounds: tuple
    solution_ws: Optional[np.ndarray] = None

    @property
    def model(self):
        return self.feature_matrix@self.solution_ws+self.potential_scattering
    
    def get_parameter_solution(self):
        return get_resonance_ladder_from_feature_pairs(self.solution_ws, self.feature_pairs)

@dataclass
class MatrixInputs:
    P: np.ndarray
    q: np.ndarray
    G: np.ndarray
    h: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    index_0T: np.ndarray
    
@dataclass
class QPopt:
    solver: str = "cvxopt"
    verbose: bool = False
    abstol: float = 1e-12
    reltol: float = 1e-12
    feastol: float = 1e-8
    maxiters: float =  100


class Solvers:

    @staticmethod
    def solve_quadratic_program(inputs: MatrixInputs, qpopt: QPopt) -> Optional[np.ndarray] :

        sol_w = solve_qp(inputs.P, inputs.q, G=inputs.G, h=inputs.h, A=None, b=None, lb=inputs.lb, ub=inputs.ub, 
                                                                                                    solver=qpopt.solver,
                                                                                                    verbose=qpopt.verbose,
                                                                                                    abstol=qpopt.abstol,
                                                                                                    reltol=qpopt.reltol,
                                                                                                    feastol=qpopt.feastol,
                                                                                                    maxiters=qpopt.maxiters
                                                                                                    )
                                                                                                
        return sol_w.reshape(-1,1)
    
    @staticmethod
    def solve_linear_program(inputs: MatrixInputs):
        # if inputs.ub is None:
        #     ub = np.ones_like(inputs.lb)*1000
        # else:
        #     ub = inputs.ub
        sol_w = linprog(inputs.q, A_ub=inputs.G, b_ub=inputs.h, bounds=np.array([inputs.lb, inputs.ub]).T)
        return sol_w.x
        


class ProblemHandler:

    def __init__(self, w_threshold):
        self.w_threshold = w_threshold
        

    def get_FeatureBank(self, dc: DataContainer, ElFeatures:np.ndarray, GtFeatures: np.ndarray):

        feature_matrix, potential_scattering, feature_pairs = fn.get_resonance_feature_bank(dc.pw.exp.E, dc.theoretical_parameters['true'].particle_pair, ElFeatures, GtFeatures)
        nfeatures = np.shape(feature_matrix)[1]

        return FeatureBank(feature_matrix, feature_pairs, potential_scattering.reshape((-1,1)), nfeatures, [0,feature_pairs[:,1]*0.999] )


    def get_MatrixInputs(self, dc: DataContainer, feature_bank: FeatureBank):
        nfeatures = np.shape(feature_bank.feature_matrix)[1]

        # remove nan values in xs and cov for solver
        b, cov, pscat, A, index_0T = fn.remove_nan_values(np.array(dc.pw.exp.exp_xs), np.array(dc.pw.CovXS), feature_bank.potential_scattering, feature_bank.feature_matrix)
        b = b.reshape((-1,1))-pscat

        # get bounds and constraints
        lb, ub = fn.get_bound_arrays(nfeatures, feature_bank.w_bounds)
        G, h = fn.get_0Trans_constraint(np.array(dc.pw.exp.E), index_0T, dc.experimental_parameters.max_xs, dc.theoretical_parameters['true'].particle_pair, feature_bank.feature_pairs)

        # Cast into quadratic program 
        P = A.T @ inv(cov) @ A
        q = - A.T @ inv(cov) @ b

        return MatrixInputs(P, q, G, h, lb, ub, index_0T)
    

    def get_ConstrainedMatrixInputs(self, matrix_inputs: MatrixInputs, w_constraint: float):
        G_wc = np.vstack([matrix_inputs.G, np.ones(len(matrix_inputs.P))])
        h_wc = np.append(matrix_inputs.h, w_constraint)
        return MatrixInputs(matrix_inputs.P, matrix_inputs.q, G_wc, h_wc, matrix_inputs.lb, matrix_inputs.ub, matrix_inputs.index_0T)


    def get_MinSolvableWeight(self, nfeatures: int, inputs: MatrixInputs):
        c = np.ones(nfeatures)
        # if inputs.ub is None:
        #     ub = np.ones_like(inputs.lb)*1000
        # else:
        #     ub = inputs.ub
        lp_minw_unred = linprog(c, A_ub=inputs.G, b_ub=inputs.h, bounds=np.array([inputs.lb, inputs.ub]).T)
        return np.sum(lp_minw_unred.x)


    def reduce_FeatureBank(self, bank: FeatureBank, sol_w: np.ndarray):
        feature_matrix, feature_pairs, reduced_solw = fn.get_reduced_features(bank.feature_matrix, sol_w, self.w_threshold, bank.feature_pairs)
        nfeatures = np.shape(feature_matrix)[1]
        return FeatureBank(feature_matrix, feature_pairs, bank.potential_scattering, nfeatures,  [0,feature_pairs[:,1]*0.999] ), reduced_solw


