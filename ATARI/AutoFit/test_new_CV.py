import sys
sys.path.insert(0, '/home/wfritsc1/ATARI/ATARI')
from cross_validation import find_CV_scores, find_model_complexity
from ATARI.AutoFit.auto_fit_updated_ import CrossValidationOUT
import numpy as np

folds_data = {}
obj_train_values = obj_test_values = chi2_train_values = chi2_test_values = [35.0, 36.0, 45.0, 43.0]
ndata_test_values = ndata_train_values = [25, 25, 25, 25]
folds_data[3] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)
obj_train_values = obj_test_values = chi2_train_values = chi2_test_values = [30.0, 33.0, 41.0, 40.0]
ndata_test_values = ndata_train_values = [25, 25, 25, 25]
folds_data[4] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)
obj_train_values = obj_test_values = chi2_train_values = chi2_test_values = [29.5, 30.2, 41.6, 37.0]
ndata_test_values = ndata_train_values = [25, 25, 25, 25]
folds_data[5] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)
obj_train_values = obj_test_values = chi2_train_values = chi2_test_values = [30.0, 32.8, 42.0, 39.2]
ndata_test_values = ndata_train_values = [25, 25, 25, 25]
folds_data[6] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)
obj_train_values = obj_test_values = chi2_train_values = chi2_test_values = [30.1, 32.9, 41.9, 39.4]
ndata_test_values = ndata_train_values = [25, 25, 25, 25]
folds_data[7] = CrossValidationOUT(chi2_test=chi2_test_values,   obj_test=obj_test_values,   ndata_test=ndata_test_values,
                                      chi2_train=chi2_train_values, obj_train=obj_train_values, ndata_train=ndata_train_values)

Nres_array, CV_test_score_means, CV_test_score_cov, CV_train_score_means, CV_train_score_cov = find_CV_scores(folds_data, use_MAD=False)
        
### Get cardinality from CV results:
Nres_selected = find_model_complexity(Nres_array, CV_test_score_means, CV_test_score_cov, use_1std_rule=True, use_1disc_rule=True, disc_thres=1.0)