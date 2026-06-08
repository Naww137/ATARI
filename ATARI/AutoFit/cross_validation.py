from typing import List
from copy import copy, deepcopy
import numpy as np
import pandas as pd
from numpy import newaxis as NA

# from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.utils.datacontainers import Evaluation_Data
from ATARI.utils.misc import psd_solve
from ATARI.AutoFit.external_fit import get_Ds_Vs
from ATARI.AutoFit.sammy_interface_bindings import Solver

__doc__ = """
...
"""

#%% ===============================================================================================
#   How to Split Datasets:
# =================================================================================================

def get_train_test_over_datasets(evaluation_data:Evaluation_Data):
    """
    Splits datasets into training set and testing set where testing set is the single dataset.
    Returns two lists of new evaluation data instances, one with training data only and the other
    with testing data only.

    Parameters
    ----------
    evaluation_data : Evaluation_Data
        The experimental data to be split.

    Returns
    -------
    evaluation_data_train_sets : List[Evaluation_Data]
        The evaluation data for the training sets
    evaluation_data_test_sets  : List[Evaluation_Data]
        The evaluation data for the testing/validation sets
    """

    k_folds = len(evaluation_data.datasets)
    if k_folds <= 3:
        print(UserWarning(f'{k_folds} datasets is a very low number of datasets to split experimental data by dataset. \
                          It is suggested that the user use non-factorized cross-validation.'))

    evaluation_data_train_sets = []
    evaluation_data_test_sets  = []
    for i_test in range(k_folds):
        experimental_titles_train, experimental_titles_test = [], []
        experimental_models_train, experimental_models_test = [], []
        datasets_train           , datasets_test            = [], []
        covariance_data_train    , covariance_data_test     = [], []
        if evaluation_data.measurement_models is None:
            measurement_models_train, measurement_models_test = None, None
        else:
            measurement_models_train, measurement_models_test = [], []
        if evaluation_data.experimental_models_no_pup is None:
            experimental_models_no_pup_train, experimental_models_no_pup_test = None, None
        else:
            experimental_models_no_pup_train, experimental_models_no_pup_test = [], []

        for i_dset in range(k_folds):
            if i_dset == i_test:
                experimental_titles_test .append(evaluation_data.experimental_titles[i_dset])
                experimental_models_test .append(evaluation_data.experimental_models[i_dset])
                datasets_test            .append(evaluation_data.datasets           [i_dset])
                covariance_data_test     .append(evaluation_data.covariance_data    [i_dset])
                if evaluation_data.measurement_models:
                    measurement_models_test.append(evaluation_data.measurement_models[i_dset])
                if evaluation_data.measurement_models:
                    experimental_models_no_pup_test.append(evaluation_data.experimental_models_no_pup[i_dset])
            else:
                experimental_titles_train.append(evaluation_data.experimental_titles[i_dset])
                experimental_models_train.append(evaluation_data.experimental_models[i_dset])
                datasets_train           .append(evaluation_data.datasets           [i_dset])
                covariance_data_train    .append(evaluation_data.covariance_data    [i_dset])
                if evaluation_data.measurement_models:
                    measurement_models_train.append(evaluation_data.measurement_models[i_dset])
                if evaluation_data.measurement_models:
                    experimental_models_no_pup_train.append(evaluation_data.experimental_models_no_pup[i_dset])

        eval_data_train = Evaluation_Data(tuple(experimental_titles_train), tuple(experimental_models_train), tuple(datasets_train), tuple(covariance_data_train), measurement_models=measurement_models_train, experimental_models_no_pup=experimental_models_no_pup_train)
        eval_data_test  = Evaluation_Data(tuple(experimental_titles_test ), tuple(experimental_models_test ), tuple(datasets_test ), tuple(covariance_data_test ), measurement_models=measurement_models_test , experimental_models_no_pup=experimental_models_no_pup_test )
    
        evaluation_data_train_sets.append(eval_data_train)
        evaluation_data_test_sets .append(eval_data_test )

    return evaluation_data_train_sets, evaluation_data_test_sets

def get_train_test_non_factorized(evaluation_data:Evaluation_Data, k_folds:int,
                                  rng:np.random.Generator=None, seed:int=None):
    """
    ...
    """

    if rng is None:
        rng = np.random.default_rng(seed)

    # Selecting test/train indices sets:
    train_indices = [[] for i_fold in range(k_folds)]
    test_indices  = [[] for i_fold in range(k_folds)]
    for iset, dataset in enumerate(evaluation_data.datasets):
        indices = list(dataset.index)
        rng.shuffle(indices)
        folds = np.array_split(indices, k_folds)
        for i_fold in range(k_folds):
            test_idx  = sorted(folds[i_fold].tolist())
            train_idx = sorted(np.concatenate([folds[i] for i in range(k_folds) if i != i_fold]).tolist())
            train_indices[i_fold].append(train_idx)
            test_indices [i_fold].append(test_idx )

    # Getting train/test evaluation data:
    eval_data_train_folds = []
    eval_data_test_folds  = []
    for i_fold in range(k_folds):
        eval_data_train_fold = deepcopy(evaluation_data)
        eval_data_test_fold  = deepcopy(evaluation_data)
        datasets_train    = [];    covariances_train = []
        datasets_test     = [];    covariances_test  = []
        for iset in range(len(evaluation_data.datasets)):
            dataset_all    = evaluation_data.datasets[iset]
            covariance_all = evaluation_data.covariance_data[iset]

            dataset_train = dataset_all.drop(index=test_indices[i_fold][iset])
            dataset_test = dataset_all#.loc[test_indices[i_fold][iset]]
            test_energies = dataset_test.E[test_indices[i_fold][iset]].tolist()
            if covariance_all in (None, {}):
                covariance_train, covariance_test = covariance_all, covariance_all
            else:
                # Train:
                diag_stat = covariance_all["diag_stat"].drop(index=test_energies)
                Jac_sys = covariance_all["Jac_sys"].drop(columns=test_energies)
                cov_sys = covariance_all["Cov_sys"]
                assert len(diag_stat['var_stat'].values) == Jac_sys.values.shape[1]
                covariance_train = {"diag_stat": diag_stat, 'Jac_sys': Jac_sys, "Cov_sys": cov_sys}
                # Test:
                diag_stat = covariance_all["diag_stat"]#.loc[test_energies]
                Jac_sys = covariance_all["Jac_sys"]#.loc[:,test_energies]
                cov_sys = covariance_all["Cov_sys"]
                assert len(diag_stat['var_stat'].values) == Jac_sys.values.shape[1]
                covariance_test = {"diag_stat": diag_stat, 'Jac_sys': Jac_sys, "Cov_sys": cov_sys}
                
                # # Test:
                # dataset_test = dataset_all.sort_values("E").drop(index=test_indices[i_fold][iset])
                # datasets_test.append(dataset_test)
                # diag_stat = covariance_all["diag_stat"].sort_values("E")
                # diag_stat.reset_index(drop=True)[test_indices[i_fold][iset]]
                # Jac_sys = covariance_all["Jac_sys"].sort_index(axis=1)[test_indices[i_fold][iset]]
                # cov_sys = covariance_all["Cov_sys"]
                # covariance_test = {"diag_stat": diag_stat, 'Jac_sys':Jac_sys, "Cov_sys": cov_sys}
                # # Train:
                # dataset_train = dataset_all.sort_values("E").drop(index=test_indices[i_fold][iset])
                # datasets_train.append(dataset_train)
                # diag_stat = covariance_all["diag_stat"].sort_values("E")
                # diag_stat.reset_index(drop=True).drop(index=test_indices[i_fold][iset])
                # Jac_sys = covariance_all["Jac_sys"].sort_index(axis=1).drop(index=test_indices[i_fold][iset])
                # cov_sys = covariance_all["Cov_sys"]
                # covariance_train = {"diag_stat": diag_stat, 'Jac_sys': Jac_sys, "Cov_sys": cov_sys}
            datasets_train.append(dataset_train)
            datasets_test .append(dataset_test)
            covariances_train.append(covariance_train)
            covariances_test .append(covariance_test )

        # Getting training/test data:
        eval_data_train_fold.datasets = datasets_train           ;      eval_data_test_fold.datasets = datasets_test
        eval_data_train_fold.covariance_data = covariances_train ;      eval_data_test_fold.covariance_data = covariances_test
        eval_data_train_folds.append(eval_data_train_fold)       ;      eval_data_test_folds.append(eval_data_test_fold)

    return eval_data_train_folds, eval_data_test_folds, test_indices, train_indices

#%% ===============================================================================================
#   Chi-squared Calculation:
# =================================================================================================

def evaluate_chi2s(res_ladder:pd.DataFrame, solver_test:Solver,
                   train_indices_list:np.ndarray, test_indices_list:np.ndarray):
    """
    ...
    """

    # Preparing Solver:
    solver = copy(solver_test)
    solver.set_bayes(False)

    # Getting Data:
    data_list = solver.sammyINP.datasets
    sammy_out = solver.fit(resonance_ladder=res_ladder)
    fit_list = sammy_out.pw
    if solver.sammyINP.idc_at_theory:
        idc_list = solver.get_idc_at_theory(res_ladder)
    else:
        idc_list = solver.sammyINP.experimental_covariance
    reaction_list = [solver.sammyINP.experiments[idx_dset].reaction for idx_dset in range(len(data_list))]
    cov_list = []
    for data, idc in zip(data_list, idc_list):
        if idc in (None, {}):
            data_unc = data['exp_unc']
            cov = np.diag(data_unc*data_unc)
        else:
            diag_stat = idc["diag_stat"].sort_values("E")
            diag_stat.reset_index(drop=True)
            var_stat = diag_stat['var_stat'].values
            cov_sys = idc['Cov_sys']
            Jac_sys = idc["Jac_sys"].sort_index(axis=1).values
            cov = np.diag(var_stat) + Jac_sys.T @ cov_sys @ Jac_sys
        indices = data.index
        cov = pd.DataFrame(cov, index=indices, columns=indices)
        cov_list.append(cov)

    # Calculating Chi-squared Values:
    Ndatas_train = [];       Ndatas_test  = []
    chi2s_train = [];       chi2s_test  = [];       chi2s_eff   = []
    for dset_idx, (data, fit_all, cov, test_indices, train_indices, reaction) in enumerate(zip(data_list, fit_list, cov_list, train_indices_list, test_indices_list, reaction_list)):
        fit_all.index = data.index
        assert all(np.isclose(np.array(fit_all.E), np.array(data.E), rtol=1e-4, atol=1e-6))
        if reaction == 'transmission':      fit = fit_all['theo_trans']
        else:                               fit = fit_all['theo_xs']
        Ndata_train, Ndata_test, chi2_train, chi2_test, chi2_eff, chi2_all = find_chi2_eff(fit, data.exp, cov, test_indices, train_indices)
        Ndatas_train.append(Ndata_train);    Ndatas_test.append(Ndata_test)
        chi2s_train .append(chi2_train) ;    chi2s_test .append(chi2_test) ;    chi2s_eff.append(chi2_eff)
    
        # Checking against solver chi2 values:
        Ndata_sammy = len(sammy_out.pw[dset_idx])
        Ndata_atari = Ndata_test + Ndata_train
        assert np.isclose(Ndata_atari, Ndata_sammy, rtol=1e-5, atol=1e-3), f'The number of datasets calculated by the non-factorized solver is different than the number of datasets evaluated by SAMMY ({Ndata_atari} and {Ndata_sammy})'
        chi2_sammy = sammy_out.chi2[dset_idx]
        chi2_atari = np.sum(chi2_all)
        assert np.isclose(chi2_atari, chi2_sammy, rtol=1e-3, atol=1e3), f'The chi-squared calculated by the non-factorized solver is different than the chi-squared evaluated by SAMMY ({chi2_atari} and {chi2_sammy})'

    return Ndatas_train, Ndatas_test, chi2s_train, chi2s_test, chi2s_eff,

def find_chi2_eff(fit:np.ndarray, data:np.ndarray, cov:np.ndarray,
                  train_indices:np.ndarray, test_indices:np.ndarray):
    """
    ...
    """

    # Splitting Test/Train:
    fit_all         = fit.values
    fit_train       = fit [train_indices].values
    fit_test        = fit [test_indices ].values
    data_all        = data.values
    data_test       = data[test_indices ].values
    data_train      = data[train_indices].values
    cov_all         = cov.values
    cov_test_test   = cov.loc[ test_indices, test_indices].values
    cov_test_train  = cov.loc[ test_indices,train_indices].values
    cov_train_train = cov.loc[train_indices,train_indices].values

    # Calculating Train Chi-squared:
    Ndata_train = len(train_indices)
    delta_train = data_train - fit_train
    Vid_train = psd_solve(cov_train_train, delta_train)
    chi2_train = delta_train.T @ Vid_train

    # Calculating Test Chi-squared:
    Ndata_test = len(test_indices)
    delta_test = data_test - fit_test
    chi2_test = delta_test.T @ psd_solve(cov_test_test, delta_test)

    # Calculating Chi-squared for all data:
    delta_all = data_all - fit_all
    chi2_all  = delta_all.T @ psd_solve(cov_all, delta_all)

    # Calculating Effective Test Chi-squared:
    fit_eff = fit_test      - cov_test_train @ Vid_train
    cov_eff = cov_test_test - cov_test_train @ psd_solve(cov_train_train, cov_test_train.T)
    delta_eff = data_test - fit_eff
    chi2_eff = delta_eff.T @ psd_solve(cov_eff, delta_eff)

    return Ndata_train, Ndata_test, chi2_train, chi2_test, chi2_eff, chi2_all

#%% ===============================================================================================
#   Splitting CV Scores:
# =================================================================================================

def find_CV_scores(fold_results, use_MAD:bool=False):
    """
    ...
    """

    objn_tests  = []
    objn_trains = []
    for Nres, fold_result in fold_results.items():
        obj_test    = np.array(fold_result.obj_test   )
        ndata_test  = np.array(fold_result.ndata_test )
        obj_train   = np.array(fold_result.obj_train  )
        ndata_train = np.array(fold_result.ndata_train)
        objn_test  = obj_test  / ndata_test
        objn_train = obj_train / ndata_train
        objn_tests .append(objn_test )
        objn_trains.append(objn_train)
    objn_tests  = np.array(objn_tests )
    objn_trains = np.array(objn_trains)

    # K_folds = len(obj_test)
    if use_MAD:
        CV_test_score_means  = []
        CV_test_score_stds   = []
        CV_train_score_means = []
        CV_train_score_stds  = []
        for CV_test_score_mean, CV_train_score_mean, objn_test, objn_train in zip(CV_test_score_means, CV_train_score_means, objn_tests, objn_trains):
            CV_test_score_mean  = np.median(objn_test )
            CV_train_score_mean = np.median(objn_train)
            CV_test_score_std  = 1.4826*np.median(np.abs(objn_test  - CV_test_score_mean ), axis=0)
            CV_train_score_std = 1.4826*np.median(np.abs(objn_train - CV_train_score_mean), axis=0)
            CV_test_score_means .append(CV_test_score_mean )
            CV_train_score_means.append(CV_train_score_mean)
            CV_test_score_stds  .append(CV_test_score_std  )
            CV_train_score_stds .append(CV_train_score_std )
        CV_test_score_cov  = np.diag(np.array(CV_test_score_stds )**2)
        CV_train_score_cov = np.diag(np.array(CV_train_score_stds)**2)
    else:
        CV_test_score_means , CV_test_score_cov  = calculate_mean_and_cov(objn_tests )
        CV_train_score_means, CV_train_score_cov = calculate_mean_and_cov(objn_trains)

    Nres = np.array(list(fold_results.keys()))
    return Nres, CV_test_score_means, CV_test_score_cov, CV_train_score_means, CV_train_score_cov

def calculate_mean_and_cov(y):
    K_folds = y.shape[1]
    y_mean = np.mean(y, axis=1)
    res = y - y_mean[:,NA]
    y_cov = np.sum(res[:,NA,:]*res[NA,:,:], axis=2) / (K_folds*(K_folds-1))
    return y_mean, y_cov

def find_model_complexity(Nres_array, CV_score_means, CV_score_cov, use_1std_rule:bool=True, use_1disc_rule:bool=False, disc_thres:float=1.0):
    """
    ...
    """

    # Sorting Nres in increasing order:
    order = np.argsort(Nres_array)
    Nres_array     = Nres_array    [order]
    CV_score_means = CV_score_means[order]
    CV_score_cov   = CV_score_cov  [np.ix_(order,order)]

    # Finding the minimum case:
    Nres_min = None
    idx_min = np.argmin(CV_score_means)
    Nres_min = Nres_array[idx_min]
    CV_score_mean_min = CV_score_means[idx_min]
    CV_score_var_min  = CV_score_cov  [idx_min,idx_min]
    
    # Applying one standard deviation rule where applicable:
    if   use_1disc_rule:
        for idx, (Nres, CV_score_mean) in enumerate(zip(Nres_array, CV_score_means)):
            if idx == idx_min:
                Nres_selected = Nres_min
                break
            CV_score_var   = CV_score_cov[idx,idx]
            CV_score_cross = CV_score_cov[idx,idx_min]
            discrepancy = (CV_score_mean-CV_score_mean_min)/np.sqrt(CV_score_var_min + CV_score_var - 2*CV_score_cross)
            if discrepancy < disc_thres:
                Nres_selected = Nres
                break
    elif use_1std_rule:
        CV_1std_limit = CV_score_mean_min + disc_thres*np.sqrt(CV_score_var_min)
        for idx, (Nres, CV_score_mean) in enumerate(zip(Nres_array, CV_score_means)):
            if CV_score_mean < CV_1std_limit:
                Nres_selected = Nres
                break
    else:
        Nres_selected = Nres_min
    return Nres_selected









# def find_model_complexity(CV_scores:dict, use_1std_rule:bool=True, use_1disc_rule:bool=False):
#     """
#     ...
#     """

#     # Sorting keys in increasing order:
#     CV_scores = dict(sorted(CV_scores.items()))

#     # Finding the minimum case:
#     Nres_min = None
#     CV_score_min = {'mean': np.inf, 'std': None}
#     for Nres, CV_score in CV_scores.items():
#         if CV_score_min['mean'] > CV_score['mean']:
#             Nres_min = Nres
#             CV_score_min = CV_score
    
#     # Applying one standard deviation rule where applicable:
#     if use_1std_rule:
#         CV_1std_limit = CV_score_min['mean'] + CV_score_min['std']
#         for Nres, CV_score in CV_scores.items():
#             if CV_score['mean'] < CV_1std_limit:
#                 Nres_selected = Nres
#                 break
#     else:
#         Nres_selected = Nres_min

#     return Nres_selected




# def _partition_weights_multiple_datasets(eigvals_sets:List[np.ndarray], K_folds:int):
#     """
#     ...
#     """

#     # Organizing and sorting eigenvalues:
#     num_sets = len(eigvals_sets)
#     eigvals = []
#     set_indices = []
#     eig_indices = []
#     for set_idx, eigval_arr in enumerate(eigvals_sets):
#         for eigval_idx, eigval in enumerate(eigval_arr):
#             eigvals.append(eigval)
#             set_indices.append(set_idx)
#             eig_indices.append(eigval_idx)
#     sorting_indices = np.argsort(eigvals)[::-1] # sorted from highest precision to lowest precision

#     # Partitioning Algorithm:
#     weights_folds = np.zeros((K_folds,), dtype=float)
#     folds_groups = [[[] for ifold in range(K_folds)] for iset in range(num_sets)]
#     for idx_original in sorting_indices:
#         eigval = eigvals[idx_original]
#         set_idx = set_indices[idx_original]
#         eig_idx = eig_indices[idx_original]
#         if eigval <= 0.0:
#             break # Ignore negative eigenvalues
#         fold_idx = np.argmin(weights_folds)
#         # print('fold_idx', fold_idx, folds_weight)
#         folds_groups[set_idx][fold_idx].append(eig_idx)
#         weights_folds[fold_idx] += eigval

#     return folds_groups, weights_folds

# def split_data_into_folds(Vi_eigvals, Vi_eigvecs, folds_groups):
#     """
#     ...
#     """

#     Vi_test_folds = []
#     Vi_train_folds = []
#     for fold_groups in folds_groups:
#         Vi_eigvals_test_fold = Vi_eigvals[fold_groups]
#         Vi_eigvecs_test_fold = Vi_eigvecs[:,fold_groups]
#         Vi_eigvals_train_fold = np.delete(Vi_eigvals, fold_groups)
#         Vi_eigvecs_train_fold = np.delete(Vi_eigvecs, fold_groups, 1)

#         print('Eigensizes:')
#         print(Vi_eigvals_test_fold.shape, Vi_eigvecs_test_fold.shape)

#         Vi_test_fold = Vi_eigvecs_test_fold @ np.diag(Vi_eigvals_test_fold) @ Vi_eigvecs_test_fold.T
#         Vi_test_folds.append(Vi_test_fold)

#         Vi_train_fold = Vi_eigvecs_train_fold @ np.diag(Vi_eigvals_train_fold) @ Vi_eigvecs_train_fold.T
#         Vi_train_folds.append(Vi_train_fold)

#     return Vi_train_folds, Vi_test_folds

# def populate_evaluation_data(evaluation_data_base:Evaluation_Data, Ds:List[np.ndarray], Vis:List[np.ndarray], Es:List[np.ndarray]):
#     """
#     ...
#     """

#     datasets = []
#     covariances = []
#     for D, Vi, E in zip(Ds, Vis, Es):
#         dataset = {'E': E,
#                    'exp': D,
#                    'exp_unc': [None]*len(E)}
#         dataset = pd.DataFrame(dataset)
#         datasets.append(dataset)
#         covariance = {'Covi': pd.DataFrame(Vi, columns=E, index=E)} # precision matrix provided for covariance matrix
#         covariances.append(covariance)

#     evaluation_data = Evaluation_Data(experimental_titles        = evaluation_data_base.experimental_titles,
#                                       experimental_models        = evaluation_data_base.experimental_models,
#                                       datasets                   = datasets,
#                                       covariance_data            = covariances,
#                                       measurement_models         = None,
#                                       experimental_models_no_pup = evaluation_data_base.experimental_models_no_pup)
#     return evaluation_data

# def split_correlated_datasets_into_folds(evaluation_data:Evaluation_Data, cap_norm_unc:float=0.0384200, K_folds:int=5):
#     """
#     Splits the experimental data into multiple folds, assuming a fit that is close enough to true.

#     Parameters
#     ----------
#     evaluation_data : Evaluation_Data
#         The evaluation data for the problem.
#     cap_norm_unc : float
#         The normalization uncertainty for capture experiments. Default is 0.03842.
#     K_folds : int
#         The number of folds to split into. Default is 5.
    
#     Returns
#     -------
#     evaluation_data_train_folds : list[Evaluation_Data]
#         The evaluation data for training for each fold.
#     evaluation_data_test_folds : list[Evaluation_Data]
#         The evaluation data for testing for each fold.
#     weights_folds : list[float]
#         Weights, representing the amount of information between each fold.
#     """

#     if not isinstance(evaluation_data, Evaluation_Data):
#         raise TypeError('"evaluation_data" must be a "Evaluation_Data" object.')

#     # Extracting data:
#     datasets = [dataset for dataset in evaluation_data.datasets]
#     covariance_data = [cov for cov in evaluation_data.covariance_data]
#     Ds, Vs = get_Ds_Vs(datasets, covariance_data, normalization_uncertainty=cap_norm_unc, idc_at_theory=True)
#     print('# of datapoints:', len(Ds))
    
#     # Getting Eigenvalues:
#     Vi_vals = []; Vi_vecs = []
#     for V in Vs:
#         Vval, Vvec = np.linalg.eigh(V)
#         Vi_val = 1.0 / Vval # we want the eigenvalues of the precision matrix, not the covariance matrix
#         Vi_vec = Vvec
#         Vi_vals.append(Vi_val); Vi_vecs.append(Vi_vec)
#         print('# of eigvals for dataset:', len(Vi_val))

#     # Partitioning data for equal-variance folds:
#     folds_groups_all_datasets, weights_folds = _partition_weights_multiple_datasets(Vi_vals, K_folds)

#     print(f'Weights for each CV fold:\n{weights_folds}')
#     print('Fold Indices', folds_groups_all_datasets)
#     print()
    
#     # Splitting data into folds:
#     Vi_train_all_datasets_folds = [[] for ifold in range(K_folds)]
#     Vi_test_all_datasets_folds  = [[] for ifold in range(K_folds)]
#     for Vi_val, Vi_vec, folds_groups_dataset in zip(Vi_vals, Vi_vecs, folds_groups_all_datasets):
#         Vi_train_folds, Vi_test_folds = split_data_into_folds(Vi_val, Vi_vec, folds_groups_dataset)
#         for ifold in range(K_folds):
#             Vi_train_all_datasets_folds[ifold].append(Vi_train_folds[ifold])
#             Vi_test_all_datasets_folds[ifold].append(Vi_test_folds[ifold])
    
#     # # Populating evaluation data for each fold:
#     # evaluation_data_train_folds = []
#     # evaluation_data_test_folds  = []
#     # for ifold in range(K_folds):
#     #     evaluation_data_train_fold = populate_evaluation_data(evaluation_data, Ds, Vi_train_all_datasets_folds[ifold], Es)
#     #     evaluation_data_train_folds.append(evaluation_data_train_fold)
#     #     evaluation_data_test_fold = populate_evaluation_data(evaluation_data, Ds, Vi_test_all_datasets_folds[ifold], Es)
#     #     evaluation_data_test_folds.append(evaluation_data_test_fold)

#     return Ds, Vi_train_all_datasets_folds, Vi_test_all_datasets_folds, weights_folds