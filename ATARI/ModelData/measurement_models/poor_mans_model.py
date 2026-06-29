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

    def __init__(self, covariance_data:dict, model_parameters=None):
        self._covariance_data = covariance_data
        if model_parameters is None:
            model_parameters = np.zeros((len(covariance_data['Cov_sys']),))
        self._model_parameters = model_parameters

    @property
    def covariance_data(self):
        return self._covariance_data
    @covariance_data.setter
    def covariance_data(self, covariance_data):
        self._covariance_data = covariance_data

    @property
    def model_parameters(self):
        return self._model_parameters
    @model_parameters.setter
    def model_parameters(self, model_parameters):
        self._model_parameters = model_parameters

    def __repr__(self):
        string = 'Measurement model (data reduction) parameters:\n'
        string += 'Covariance Data:'
        string += repr(self.covariance_data)
        return string
    
    def sample_model_parameters(self, rng:np.random.Generator=None, seed:int=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        cov_sys = self.covariance_data['Cov_sys']
        size = cov_sys.shape[0]
        model_parameters = rng.multivariate_normal(mean=np.zeros((size,)), cov=cov_sys)
        return model_parameters
    
    def generate_raw_data(self,
                          pw_true:pd.DataFrame,
                          true_model_parameters,
                          options:syndatOPT,
                          rng:np.random.Generator=None,
                          seed:int=None):
        if rng is None:
            rng = np.random.default_rng(seed)
        var_stat = self.covariance_data['diag_stat']
        var_stat.sort_index(inplace=True)
        stat_part = rng.normal(scale=np.sqrt(var_stat['var_stat'].values))
        jac = self.covariance_data['Jac_sys']
        jac = np.array(jac.sort_index()).T
        syst_part = jac @ true_model_parameters
        unc_part = np.array(stat_part + syst_part)

        # Interpolating to true grid:
        pw_true.sort_values(by='E', inplace=True)
        exp = pw_true['true'].values + unc_part
        raw_data = {'E':pw_true['E'].values, 'exp':exp}
        return raw_data
    
    def reduce_raw_data(self, raw_data, options:syndatOPT):
        
        # Getting data:
        data = pd.DataFrame({'E':raw_data['E'], 'exp':raw_data['exp']})

        if options.calculate_covariance:
            # Get diagonal experimental uncertainty:
            var_stat = self.covariance_data['diag_stat']
            var_stat.sort_index().to_numpy()
            jac = self.covariance_data['Jac_sys']
            jac = jac.sort_index().to_numpy()
            exp_unc = np.sqrt(self.covariance_data['diag_stat'].values[:,0] + np.diag(jac.T @ self.covariance_data['Cov_sys'] @ jac))
            data['exp_unc'] = exp_unc

            # Updating covariance data:
            cov_data = self.covariance_data


            if options.explicit_covariance:
                cov_sys = jac.T @ self.covariance_data['Cov_sys'] @ jac
                cov = np.diag(var_stat) + cov_sys
                cov_data['Cov'] = cov
        else:
            cov_data = {}

        return data, cov_data, raw_data
    
    def sample_true_model_parameters(self, true_model_parameters:dict, rng:np.random.Generator=None, seed:int=None):
        return self.model_parameters

    def truncate_energy_range(self, new_energy_range):
        minE = float(min(new_energy_range))
        maxE = float(max(new_energy_range))
        filtered_cov = {}
        if 'Cov_sys' in self._covariance_data.keys():
            filtered_cov["diag_stat"] = self._covariance_data['diag_stat'].loc[np.array(self._covariance_data['diag_stat'].index>=minE) & np.array(self._covariance_data['diag_stat'].index<=maxE)]
            filtered_cov["Jac_sys"]   = self._covariance_data['Jac_sys'].loc[:,np.array(self._covariance_data['Jac_sys'].columns>=minE) & np.array(self._covariance_data['Jac_sys'].columns<=maxE)]
            filtered_cov["Cov_sys"]   = self._covariance_data['Cov_sys']
        else:
            raise ValueError("Filtering not implemented for explicit cov yet")
        self._covariance_data = filtered_cov
    
    def select_data_points(self, energies:list):
        filtered_cov = {}
        if 'Cov_sys' in self._covariance_data.keys():
            filtered_cov["diag_stat"] = self._covariance_data['diag_stat'].loc[energies]
            filtered_cov["Jac_sys"]   = self._covariance_data['Jac_sys'].loc[:,energies]
            filtered_cov["Cov_sys"]   = self._covariance_data['Cov_sys']
        else:
            raise ValueError("Filtering not implemented for explicit cov yet")
        self._covariance_data = filtered_cov