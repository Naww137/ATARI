from copy import copy
import numpy as np

__doc__ = """
...
"""

# def find_CV_scores(fold_test_scores:dict, use_MAD:bool=False):
#     """
#     ...
#     """

#     CV_scores = {}
#     for Nres, fold_test_scores_case in fold_test_scores.items():
#         if use_MAD:
#             CV_score     = np.median(fold_test_scores_case)
#             CV_score_std = 1.4826*np.median(np.abs(fold_test_scores_case - CV_score), axis=0)
#         else:
#             CV_score     = np.mean(fold_test_scores_case)
#             CV_score_std = np.std(fold_test_scores_case)/np.sqrt(len(fold_test_scores_case))
#         CV_scores[Nres] = {'mean': CV_score, 'std':CV_score_std}
#     return CV_scores

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
        if use_MAD:
            raise NotImplementedError('Weighted folds have not been implemented with MAD statistics.')
        else:
            K_folds = len(obj_test)
            CV_test_score_mean = np.sum(ndata_test * objn_test) / sum(ndata_test)
            CV_test_score_std  = np.sqrt(np.sum(ndata_test * (objn_test - CV_test_score_mean)**2) / ((K_folds-1) * np.sum(ndata_test)))
            CV_train_score_mean = np.sum(ndata_train * objn_train) / sum(ndata_train)
            CV_train_score_std  = np.sqrt(np.sum(ndata_train * (objn_train - CV_train_score_mean)**2) / ((K_folds-1) * np.sum(ndata_train)))

        CV_test_scores[Nres]  = {'mean': CV_test_score_mean , 'std':CV_test_score_std }
        CV_train_scores[Nres] = {'mean': CV_train_score_mean, 'std':CV_train_score_std}
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

def _partition_weights(eigvals, K_folds:int):
    """
    ...
    """

    folds_weight = [0.0]*K_folds
    folds_groups = [[]]*K_folds
    for eig_idx, eigval in enumerate(eigvals):
        fold_idx = np.argmin(folds_weight)
        folds_groups[fold_idx].append(eig_idx)
        folds_weight[fold_idx] += eigval
    return folds_groups, folds_weight

def split_correlated_folds(D, V, T, K_folds:int):
    """
    ...
    """

    # NOTE: CHECK THIS WITH NOAH!!!

    # Eigen-decomposition:
    eigvals, eigvecs = np.linalg.eigh(V)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[::-1,::-1]

    # Partition folds:
    folds_groups, folds_weight = _partition_weights(eigvals=eigvals, K_folds=K_folds)

    # Generate data and data covariances for each fold:
    D_test_folds = []
    V_test_folds = []
    D_train_folds = []
    V_train_folds = []
    for fold_groups in folds_groups:
        eigvals_test_fold = eigvals[fold_groups]
        eigvecs_test_fold = eigvecs[:,fold_groups]
        eigvals_train_fold = np.delete(eigvals, fold_groups)
        eigvecs_train_fold = np.delete(eigvecs, [], fold_groups)

        D_test_fold = T + eigvecs_test_fold @ (eigvecs_test_fold.T @ (D - T))
        D_test_folds.append(D_test_fold)
        V_test_fold = eigvecs_test_fold @ eigvals_test_fold @ eigvecs_test_fold.T
        V_test_folds.append(V_test_fold)

        D_train_fold = T + eigvecs_train_fold @ (eigvecs_train_fold.T @ (D - T))
        D_train_folds.append(D_train_fold)
        V_train_fold = eigvecs_train_fold @ eigvals_train_fold @ eigvecs_train_fold.T
        V_train_folds.append(V_train_fold)

    return D_test_folds, V_test_folds, D_train_folds, V_train_folds, folds_weight

