from typing import List
from copy import copy, deepcopy
import numpy as np
import pandas as pd

# from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.sammy_interface.sammy_classes import SolverOPTs_YW, SammyRunTimeOptions
from ATARI.utils.datacontainers import Evaluation_Data
from ATARI.AutoFit.sammy_interface_bindings import Solver
from ATARI.sammy_interface.sammy_misc import get_idc_at_theory
from ATARI.utils.stats import add_normalization_uncertainty_to_covariance

__doc__ = """
...
"""

def split_train_data(eval_data:Evaluation_Data, K_folds:int=5, rng:np.random.Generator=None, seed:int=None):
    """
    ...
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    # Selecting test/train indices sets:
    train_indices = [[] for ifold in range(K_folds)]
    test_indices  = [[] for ifold in range(K_folds)]
    for iset, dataset in enumerate(eval_data.datasets):
        indices = np.arange(len(dataset))
        print(f'dataset #{iset} | # datapoints = {len(dataset)}')
        rng.shuffle(indices)
        folds = np.array_split(indices, K_folds)
        # print('FOLDS:')
        # print(folds)
        for ifold in range(K_folds):
            test_idx = np.sort(folds[ifold])
            train_idx = np.sort(np.concatenate([folds[i] for i in range(K_folds) if i != ifold]))
            train_indices[ifold].append(train_idx)
            test_indices [ifold].append(test_idx )

    # Getting training evaluation data:
    eval_data_train_folds = []
    for ifold in range(K_folds):
        eval_data_train_fold = deepcopy(eval_data)
        eval_data_train_fold = eval_data_train_fold.select_datapoints(train_indices[ifold])
        eval_data_train_folds.append(eval_data_train_fold)

    return eval_data_train_folds, test_indices, train_indices


def evaluate_chi2_test(res_ladder:pd.DataFrame, solver_test:Solver,
                       test_indices:np.ndarray, train_indices:np.ndarray):
    """
    ...
    """

    # Preparing solver:
    solver_test = copy(solver_test)
    solver_test.set_bayes(False)

    # Es = [dataset['E']   for dataset in solver_test.sammyINP.datasets]
    # Ds = [dataset['exp'] for dataset in solver_test.sammyINP.datasets]
    Ds = solver_test.sammyINP.datasets
    sammy_out = solver_test.fit(resonance_ladder=res_ladder)
    Ts = sammy_out.pw
    idcs = get_idc_at_theory(solver_test.sammyINP, solver_test.sammyRTO, sammy_out.par)
    chi2_eff = []
    for iset, (D, T, idc, test_idx, train_idx) in enumerate(zip(Ds, Ts, idcs, test_indices, train_indices)):
        
        # Pointwise data:
        D.sort_values(by='E', inplace=True)
        D.reset_index(drop=True, inplace=True)

        # Covariance:
        if "theory" in idc.keys(): # capture pw True
            diag_stat = np.diag(D["exp_unc"].values**2)
            # print(f'SET #{iset}. CAP IDC:')
            # print(diag_stat)
            # print()
            # print(idc["theory"]["true"].values)
            # print()
            # print(solver_test.cap_norm_unc**2)
            V = add_normalization_uncertainty_to_covariance(diag_stat, idc["theory"]["true"].values, solver_test.cap_norm_unc)
            # V = add_normalization_uncertainty_to_covariance(diag_stat, D["exp"].values, solver_test.cap_norm_unc)
        elif not idc:
            raise ValueError('IDC not provided.')
        else:
            # print(f'SET #{iset}. NO CAP IDC')
            cov_sys = idc['Cov_sys']
            diag_stat = idc["diag_stat"].sort_values("E")
            # diag_stat.reset_index(drop=True)
            Jac_sys = idc["Jac_sys"].sort_index(axis=1).values
            # print('DIAG STAT:')
            # print(np.diag(diag_stat.values[:,0]))
            # print(Jac_sys.T @ cov_sys @ Jac_sys)
            V = np.diag(diag_stat.values[:,0]) + Jac_sys.T @ cov_sys @ Jac_sys
            # print(V)

        # Calculate chi2 effective:
        # print('RES LADDER:')
        # print(res_ladder)
        # print()
        print('\nPW THEORETICAL:')
        print(T[['E','theo_xs', 'theo_trans']])
        print()
        # print('EXP DATA:')
        # print(D)
        # print()
        chi2_eff.append(find_chi2_eff(D, T, V, test_idx, train_idx))
    return chi2_eff

def find_chi2_eff(D:pd.DataFrame, T:pd.DataFrame, V:np.ndarray,
                  test_indices:np.ndarray, train_indices:np.ndarray):
    """
    ...
    """
    # print('TEST INDICES:')
    # print(test_indices)
    # print()
    if T['theo_trans'].isnull().all():
        theo_col = 'theo_xs' # capture
    else:
        theo_col = 'theo_trans' # transmission
    T_test  = T.loc[test_indices , theo_col].to_numpy()
    T_train = T.loc[train_indices, theo_col].to_numpy()
    D_test  = D.loc[test_indices , 'exp'].to_numpy()
    D_train = D.loc[train_indices, 'exp'].to_numpy()
    # E_test  = T.loc[test_indices , 'E'].to_numpy()
    # E_train = T.loc[train_indices, 'E'].to_numpy()
    # D_test  = D[D['E'].isin(E_test ), 'exp'].to_numpy()
    # D_train = D[D['E'].isin(E_train), 'exp'].to_numpy()
    # print()
    # print('T_TEST:')
    # print(T_test)
    # print()
    # print('D_TEST:')
    # print(D_test)
    # print()
    V_test_test   = V[np.ix_(test_indices ,test_indices )]
    V_test_train  = V[np.ix_(test_indices ,train_indices)]
    V_train_train = V[np.ix_(train_indices,train_indices)]
    # print('V_train_train:')
    # print(V)
    # print(V_train_train)
    # print()
    T_eff = T_test + V_test_train @ np.linalg.solve(V_train_train, (D_train-T_train))
    V_eff = V_test_test - V_test_train @ np.linalg.solve(V_train_train, V_test_train.T)
    delta = D_test - T_eff
    print('\nT EFF:')
    print(T_eff)
    print('\nV EFF:')
    print(V_eff)
    chi2_eff = delta.T @ np.linalg.solve(V_eff, delta)
    return chi2_eff

def find_CV_scores(fold_results, use_MAD:bool=False):
    """
    ...
    """

    CV_test_scores = {}
    CV_train_scores = {}
    for Nres, fold_result in fold_results.items():
        obj_test    = np.array(fold_result.obj_test   )
        ndata_test  = np.array(fold_result.ndata_test )
        obj_train   = np.array(fold_result.obj_train  )
        ndata_train = np.array(fold_result.ndata_train)
        objn_test  = obj_test  / ndata_test
        objn_train = obj_train / ndata_train
        K_folds = len(obj_test)
        if use_MAD:
            CV_test_score_mean  = np.median(objn_test)
            CV_test_score_std   = 1.4826*np.median(np.abs(objn_test - CV_test_score_mean), axis=0)
            CV_train_score_mean = np.median(objn_train)
            CV_train_score_std  = 1.4826*np.median(np.abs(objn_train - CV_train_score_mean), axis=0)
        else:
            CV_test_score_mean  = np.mean(objn_test)
            CV_test_score_std   = np.std(objn_test) / np.sqrt(K_folds)
            CV_train_score_mean = np.mean(objn_train)
            CV_train_score_std  = np.std(objn_train) / np.sqrt(K_folds)
        CV_test_scores[Nres]  = {'mean':CV_test_score_mean , 'std':CV_test_score_std }
        CV_train_scores[Nres] = {'mean':CV_train_score_mean, 'std':CV_train_score_std}
        print('\n\nOBJECTIVES:')
        print(objn_test)
        print('CV TEST:')
        print(CV_test_scores[Nres])
    return CV_test_scores, CV_train_scores

def find_model_complexity(CV_scores:dict, use_1std_rule:bool=True):
    """
    ...
    """

    # Sorting keys in increasing order:
    CV_scores = dict(sorted(CV_scores.items()))

    # Finding the minimum case:
    Nres_min = None
    CV_score_min = {'mean': np.inf, 'std': None}
    for Nres, CV_score in CV_scores.items():
        if CV_score_min['mean'] > CV_score['mean']:
            Nres_min = Nres
            CV_score_min = CV_score
    
    # Applying one standard deviation rule where applicable:
    if use_1std_rule:
        CV_1std_limit = CV_score_min['mean'] + CV_score_min['std']
        for Nres, CV_score in CV_scores.items():
            if CV_score['mean'] < CV_1std_limit:
                Nres_selected = Nres
                break
    else:
        Nres_selected = Nres_min

    return Nres_selected