from typing import List
from copy import deepcopy
import numpy as np
import pandas as pd

# from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.utils.datacontainers import Evaluation_Data
from ATARI.AutoFit.external_fit import get_Ds_Vs
# from ATARI.sammy_interface.sammy_misc import get_idc_at_theory

__doc__ = """
...
"""

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
            CV_test_score_std   = np.std(objn_test, ddof=1) / np.sqrt(K_folds)
            CV_train_score_mean = np.mean(objn_train)
            CV_train_score_std  = np.std(objn_train, ddof=1) / np.sqrt(K_folds)
        CV_test_scores[Nres]  = {'mean':CV_test_score_mean , 'std':CV_test_score_std }
        CV_train_scores[Nres] = {'mean':CV_train_score_mean, 'std':CV_train_score_std}
    return CV_test_scores, CV_train_scores

# def find_CV_scores(fold_results, use_MAD:bool=False):
#     """
#     ...
#     """

#     CV_test_scores = {}
#     CV_train_scores = {}
#     for Nres, fold_result in fold_results.items():
#         obj_test    = np.array(fold_result.obj_test   )
#         ndata_test  = np.array(fold_result.ndata_test )
#         obj_train   = np.array(fold_result.obj_train  )
#         ndata_train = np.array(fold_result.ndata_train)
#         objn_test  = obj_test  / ndata_test
#         objn_train = obj_train / ndata_train
#         if use_MAD:
#             raise NotImplementedError('Weighted folds have not been implemented with MAD statistics.')
#         else:
#             K_folds = len(obj_test)
#             CV_test_score_mean = np.sum(ndata_test * objn_test) / sum(ndata_test)
#             CV_test_score_std  = np.sqrt(np.sum(ndata_test * (objn_test - CV_test_score_mean)**2) / ((K_folds-1) * np.sum(ndata_test)))
#             CV_train_score_mean = np.sum(ndata_train * objn_train) / sum(ndata_train)
#             CV_train_score_std  = np.sqrt(np.sum(ndata_train * (objn_train - CV_train_score_mean)**2) / ((K_folds-1) * np.sum(ndata_train)))

#         CV_test_scores[Nres]  = {'mean': CV_test_score_mean , 'std':CV_test_score_std }
#         CV_train_scores[Nres] = {'mean': CV_train_score_mean, 'std':CV_train_score_std}
#     return CV_test_scores, CV_train_scores

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




def _partition_weights_multiple_datasets(eigvals_sets:List[np.ndarray], K_folds:int):
    """
    ...
    """

    # Organizing and sorting eigenvalues:
    num_sets = len(eigvals_sets)
    eigvals = []
    set_indices = []
    eig_indices = []
    for set_idx, eigval_arr in enumerate(eigvals_sets):
        for eigval_idx, eigval in enumerate(eigval_arr):
            eigvals.append(eigval)
            set_indices.append(set_idx)
            eig_indices.append(eigval_idx)
    sorting_indices = np.argsort(eigvals)[::-1] # sorted from highest precision to lowest precision

    # Partitioning Algorithm:
    weights_folds = np.zeros((K_folds,), dtype=float)
    folds_groups = [[[] for ifold in range(K_folds)] for iset in range(num_sets)]
    for idx_original in sorting_indices:
        eigval = eigvals[idx_original]
        set_idx = set_indices[idx_original]
        eig_idx = eig_indices[idx_original]
        if eigval <= 0.0:
            break # Ignore negative eigenvalues
        fold_idx = np.argmin(weights_folds)
        # print('fold_idx', fold_idx, folds_weight)
        folds_groups[set_idx][fold_idx].append(eig_idx)
        weights_folds[fold_idx] += eigval

    return folds_groups, weights_folds

def split_data_into_folds(Vi_eigvals, Vi_eigvecs, folds_groups):
    """
    ...
    """

    Vi_test_folds = []
    Vi_train_folds = []
    for fold_groups in folds_groups:
        Vi_eigvals_test_fold = Vi_eigvals[fold_groups]
        Vi_eigvecs_test_fold = Vi_eigvecs[:,fold_groups]
        Vi_eigvals_train_fold = np.delete(Vi_eigvals, fold_groups)
        Vi_eigvecs_train_fold = np.delete(Vi_eigvecs, fold_groups, 1)

        print('Eigensizes:')
        print(Vi_eigvals_test_fold.shape, Vi_eigvecs_test_fold.shape)

        Vi_test_fold = Vi_eigvecs_test_fold @ np.diag(Vi_eigvals_test_fold) @ Vi_eigvecs_test_fold.T
        Vi_test_folds.append(Vi_test_fold)

        Vi_train_fold = Vi_eigvecs_train_fold @ np.diag(Vi_eigvals_train_fold) @ Vi_eigvecs_train_fold.T
        Vi_train_folds.append(Vi_train_fold)

    return Vi_train_folds, Vi_test_folds

def populate_evaluation_data(evaluation_data_base:Evaluation_Data, Ds:List[np.ndarray], Vis:List[np.ndarray], Es:List[np.ndarray]):
    """
    ...
    """

    datasets = []
    covariances = []
    for D, Vi, E in zip(Ds, Vis, Es):
        dataset = {'E': E,
                   'exp': D,
                   'exp_unc': [None]*len(E)}
        dataset = pd.DataFrame(dataset)
        datasets.append(dataset)
        covariance = {'Covi': pd.DataFrame(Vi, columns=E, index=E)} # precision matrix provided for covariance matrix
        covariances.append(covariance)

    evaluation_data = Evaluation_Data(experimental_titles        = evaluation_data_base.experimental_titles,
                                      experimental_models        = evaluation_data_base.experimental_models,
                                      datasets                   = datasets,
                                      covariance_data            = covariances,
                                      measurement_models         = None,
                                      experimental_models_no_pup = evaluation_data_base.experimental_models_no_pup)
    return evaluation_data

def split_correlated_datasets_into_folds(evaluation_data:Evaluation_Data, cap_norm_unc:float=0.0384200, K_folds:int=5):
    """
    Splits the experimental data into multiple folds, assuming a fit that is close enough to true.

    Parameters
    ----------
    evaluation_data : Evaluation_Data
        The evaluation data for the problem.
    cap_norm_unc : float
        The normalization uncertainty for capture experiments. Default is 0.03842.
    K_folds : int
        The number of folds to split into. Default is 5.
    
    Returns
    -------
    evaluation_data_train_folds : list[Evaluation_Data]
        The evaluation data for training for each fold.
    evaluation_data_test_folds : list[Evaluation_Data]
        The evaluation data for testing for each fold.
    weights_folds : list[float]
        Weights, representing the amount of information between each fold.
    """

    if not isinstance(evaluation_data, Evaluation_Data):
        raise TypeError('"evaluation_data" must be a "Evaluation_Data" object.')

    # Extracting data:
    datasets = [dataset for dataset in evaluation_data.datasets]
    covariance_data = [cov for cov in evaluation_data.covariance_data]
    Ds, Vs = get_Ds_Vs(datasets, covariance_data, normalization_uncertainty=cap_norm_unc, idc_at_theory=True)
    print('# of datapoints:', len(Ds))
    
    # Getting Eigenvalues:
    Vi_vals = []; Vi_vecs = []
    for V in Vs:
        Vval, Vvec = np.linalg.eigh(V)
        Vi_val = 1.0 / Vval # we want the eigenvalues of the precision matrix, not the covariance matrix
        Vi_vec = Vvec
        Vi_vals.append(Vi_val); Vi_vecs.append(Vi_vec)
        print('# of eigvals for dataset:', len(Vi_val))

    # Partitioning data for equal-variance folds:
    folds_groups_all_datasets, weights_folds = _partition_weights_multiple_datasets(Vi_vals, K_folds)

    print(f'Weights for each CV fold:\n{weights_folds}')
    print('Fold Indices', folds_groups_all_datasets)
    print()
    
    # Splitting data into folds:
    Vi_train_all_datasets_folds = [[] for ifold in range(K_folds)]
    Vi_test_all_datasets_folds  = [[] for ifold in range(K_folds)]
    for Vi_val, Vi_vec, folds_groups_dataset in zip(Vi_vals, Vi_vecs, folds_groups_all_datasets):
        Vi_train_folds, Vi_test_folds = split_data_into_folds(Vi_val, Vi_vec, folds_groups_dataset)
        for ifold in range(K_folds):
            Vi_train_all_datasets_folds[ifold].append(Vi_train_folds[ifold])
            Vi_test_all_datasets_folds[ifold].append(Vi_test_folds[ifold])
    
    # # Populating evaluation data for each fold:
    # evaluation_data_train_folds = []
    # evaluation_data_test_folds  = []
    # for ifold in range(K_folds):
    #     evaluation_data_train_fold = populate_evaluation_data(evaluation_data, Ds, Vi_train_all_datasets_folds[ifold], Es)
    #     evaluation_data_train_folds.append(evaluation_data_train_fold)
    #     evaluation_data_test_fold = populate_evaluation_data(evaluation_data, Ds, Vi_test_all_datasets_folds[ifold], Es)
    #     evaluation_data_test_folds.append(evaluation_data_test_fold)

    return Ds, Vi_train_all_datasets_folds, Vi_test_all_datasets_folds, weights_folds