import numpy as np
import pandas as pd
from copy import deepcopy
from ATARI.utils.atario import update_dict
from ATARI.syndat.general_functions import *
from ATARI.theory.experimental import e_to_t, t_to_e


from ATARI.ModelData.structuring import parameter, vector_parameter
from ATARI.syndat.data_classes import syndatOPT




# ========================================================================================
#            Handler class
# ========================================================================================

class Poor_Mans_Model:
    """
    This model is a low-fidelity measurement model, used when limited information is provided.
    """

    def __init__(self, covariance_data:dict):
        self._covariance_data = covariance_data

    @property
    def covariance_data(self):
        return self._covariance_data
    @covariance_data.setter
    def covariance_data(self, covariance_data):
        self._covariance_data = covariance_data

    def __repr__(self):
        string = 'Measurement model (data reduction) parameters:\n'
        string += self.covariance_data
        return string
    
    def sample_model_parameters(self, rng:np.random.Generator=None, seed:int=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        cov_sys = self.covariance_data['Cov_sys']
        size = cov_sys.shape[0]
        model_parameters = rng.multivariate_normal(np.zeros((size,)), cov_sys)
        return model_parameters
    
    def generate_raw_data(self,
                          pw_true:pd.DataFrame,
                          true_model_parameters,
                          rng:np.random.Generator=None,
                          seed:int=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        print('tmp', true_model_parameters.shape)
        # print('stat', self.covariance_data['diag_stat']['var_stat'].values)
        var_stat = self.covariance_data['diag_stat']
        var_stat.sort_index(inplace=True)
        stat_part = rng.normal(scale=np.sqrt(var_stat['var_stat'].values))
        print('stat', stat_part.shape)
        jac = self.covariance_data['Jac_sys'].T
        jac.sort_index(inplace=True)
        syst_part = jac @ true_model_parameters
        print('jac', self.covariance_data['Jac_sys'].to_numpy().shape)
        unc_part = stat_part + syst_part

        # Interpolating to true grid:
        E_idc = self.covariance_data['diag_stat'].index
        print(E_idc)
        pw_true.sort_values(by='E', inplace=True)
        print(pw_true['E'].values)
        print('true', pw_true['true'].values.shape)
        exp = pw_true['true'].values + unc_part
        raw_data = {'E':pw_true['E'].values, 'exp':exp}
        return raw_data
    
    def reduce_raw_data(self, raw_data):
        var_stat = self.covariance_data['diag_stat']
        var_stat.sort_index().to_numpy()
        jac = self.covariance_data['Jac_sys']
        jac = jac.sort_index().to_numpy()
        print(self.covariance_data['diag_stat'].values[:,0])
        exp_unc = np.sqrt(self.covariance_data['diag_stat'].values[:,0] + np.diag(jac.T @ self.covariance_data['Cov_sys'] @ jac))
        print(exp_unc.shape)
        data = pd.DataFrame({'E':raw_data['E'], 'exp':raw_data['exp'], 'exp_unc':exp_unc})
        return data, self.covariance_data, raw_data