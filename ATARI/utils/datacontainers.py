
from ATARI.utils import hdf5 as h5io
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy, copy
import h5py
import pandas as pd


@dataclass
class Evaluation_Data:
    """
    Data class to hold data used in an evaluation, i.e.,  experimental titles/models, experimental pointwise data, and covariance information.
    """

    experimental_titles     : tuple
    experimental_models     : tuple
    datasets                : tuple
    covariance_data         : tuple


    @classmethod
    def from_hdf5(cls, experimental_titles, experimental_models, sample_file, isample):
        """
        Construct an Evaluation_Data instance from data hdf5 file.


        Parameters
        ----------
        experimental_titles : _type_
            _description_
        experimental_models : _type_
            _description_
        sample_file : _type_
            _description_
        isample : bool
            _description_

        Returns
        -------
        _type_
            _description_
        """

        datasets = []
        covariance_data = []
        for title in experimental_titles:
            pw_reduced_df, cov_data = h5io.read_pw_reduced(sample_file, isample, title)
            datasets.append(pw_reduced_df)
            covariance_data.append(cov_data)

        return cls(tuple(experimental_titles), tuple(experimental_models), tuple(datasets), tuple(covariance_data))


    def to_hdf5(self, filepath, isample, overwrite=True):
        """
        Writes the Evaluation data to an hdf5 file under sample_<isample>/exp_dat_<title>

        Parameters
        ----------
        filepath : str
            Path to the .hdf5 file.
        isample : int
            Integer sample number to determine top level in hdf5.
        overwrite : bool, optional
            Option to overwrite data existing with the same isample and title (False is not yet implemented), by default True

        Raises
        ------
        ValueError
            _description_
        """
        sample_group = f'sample_{isample}'
        for title, data, cov in zip(self.experimental_titles, self.datasets, self.covariance_data):

            ### if exp dataframe already exists, just add to it
            h5f = h5py.File(filepath, "a")
            if sample_group in h5f:
                if f"exp_dat_{title}" in h5f[sample_group]:
                    exists = True
                else:
                    exists = False
            h5f.close()
            
            if exists:
                pw_reduced_df, _ = h5io.read_pw_reduced(filepath, isample, title)
                model_keys = [each for each in data.keys() if each not in ["E", "exp", "exp_unc"] ]
                if all([k in pw_reduced_df.keys() for k in model_keys]):
                    print(f"Experimental pointwise models {model_keys} already exist in dataframe for {title}, overwriting")
                data = pw_reduced_df.merge(data, how='inner', validate="1:1")
            h5io.write_pw_reduced(filepath, isample, title, data, cov_data=cov)



    def to_matrix_form(self):
        # functions to convert to matrix form for external solves (see IFB_dev/fit_w_derivative)
        pass


    def truncate(self, energy_range):
        """
        Truncates all evaluation data to a specific energy region

        Parameters
        ----------
        energy_range : _type_
            _description_

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        min_max_E = (min(energy_range), max(energy_range))
        datasets = []
        for d in copy(self.datasets):
            datasets.append(d.loc[(d.E>min_max_E[0]) & (d.E<min_max_E[1])])
        
        covariance_data = []
        for exp_cov in copy(self.covariance_data):
            filtered_cov = {}
            if not exp_cov:
                pass
            elif 'Cov_sys' in exp_cov.keys():
                filtered_cov["diag_stat"] = exp_cov['diag_stat'].loc[(exp_cov['diag_stat'].index>=min_max_E[0]) & (exp_cov['diag_stat'].index<=min_max_E[1])]
                filtered_cov["Jac_sys"] = exp_cov["Jac_sys"].loc[:,(exp_cov['Jac_sys'].columns>=min_max_E[0]) & (exp_cov['Jac_sys'].columns<=min_max_E[1])]
                filtered_cov["Cov_sys"] = exp_cov['Cov_sys']
            else: # 'CovT' in exp_cov.keys() or 'CovY' in exp_cov.keys():
                raise ValueError("Filtering not implemented for explicit cov yet")
        
            covariance_data.append(filtered_cov)

        # TODO: truncate experiment energy/tof grid and range
        experiments = deepcopy(self.experimental_models)

        return Evaluation_Data(tuple(self.experimental_titles), tuple(experiments), tuple(datasets), tuple(covariance_data))




@dataclass
class Evaluation:
    """
    Dataclass to hold information about an evaluation (model), i.e., model title, resonance parameters, and pointwise reconstructions.
    """

    title                   : str
    respar                  : pd.DataFrame
    
    pw_theoretical          : Optional[pd.DataFrame] = None
    evaluation_data         : Optional[Evaluation_Data] = None
    chi2                    : Optional[pd.Series] = None


    @classmethod
    def from_pkl(title, filepath):
        pass
        # needs to be specific to AutoFitOUT pkl? Not sure if that object is stable yet

    @classmethod
    def from_hdf5(cls, title, filepath, isample):
        respar = h5io.read_par(filepath, isample, title)
        # could also look for existing experimental and theoretical pointwise and chi2
        return cls(title, respar)
    

    @staticmethod
    def reduce_sammyOUT_pw(sammy_pw_list, experimental_models):
        """
        Reduces the list of pointwise dataframes from sammyOUT_YW to only columns: exp, exp_unc, and theo.
        This is useful to do before instantiating an Evaluation data object with experimental pointwise data for an evaluation model.
        Performing this function first reduces clutter in the naming scheme particularly once saved to an hdf5 file.

        Parameters
        ----------
        sammy_pw_list : _type_
            _description_
        experimental_models : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        pw_models = copy(sammy_pw_list)
        for i, exp_df in enumerate(pw_models):
            exp_df.drop(columns=["theo_xs_bayes", "theo_trans_bayes"], inplace=True)
            if experimental_models[i].reaction == "capture":
                exp_df.drop(columns=["exp_trans", "exp_trans_unc", "theo_trans"], inplace=True)
                exp_df.rename(columns={"exp_xs":"exp", "exp_xs_unc":"exp_unc", "theo_xs":"theo"}, inplace=True)
            elif experimental_models[i].reaction == "transmission":
                exp_df.drop(columns=["exp_xs", "exp_xs_unc", "theo_xs"], inplace=True)
                exp_df.rename(columns={"exp_trans":"exp", "exp_trans_unc":"exp_unc", "theo_trans":"theo"}, inplace=True)
        return pw_models

    @staticmethod
    def chi2_list_to_series(chi2_list, evaluation_title, experimental_titles):
        """
        Converts a chi2 list (usually from sammyOUT class) to a properly formatted series for the Evaluation class.

        Parameters
        ----------
        chi2_list : _type_
            _description_
        evaluation_title : _type_
            _description_
        experimental_titles : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return pd.Series(chi2_list, index=experimental_titles, name=evaluation_title)

    def to_hdf5(self, filepath, isample, overwrite=True):
        sample_group = f'sample_{isample}'
        chi2_exists = False
        h5f = h5py.File(filepath, "a")
        if sample_group in h5f:
            if f'par_{self.title}' in h5f[sample_group]:
                if overwrite:
                    print(f"Parameters titled {self.title} already exists in {sample_group}, overwriting")
                else:
                    raise ValueError("Overwrite=False is not implemented yet")
            if f'chi2' in h5f[sample_group]:
                chi2_exists = True
        h5f.close()

        h5io.write_par(filepath, isample, self.respar, self.title)

        if self.pw_theoretical is not None:
            pass # write pw theo

        if self.evaluation_data is not None:
            for exp_df in self.evaluation_data.datasets:
                mapper={}
                for key in exp_df.keys():
                    if key not in ["E", "exp", "exp_unc"]:
                        mapper[key] = f"{key}_{self.title}"
                exp_df.rename(columns=mapper, inplace=True)
            self.evaluation_data.to_hdf5(filepath, isample)

        if chi2_exists:
            df = pd.read_hdf(filepath, f"sample_{isample}/chi2")
            if self.title in df.keys():
                print(f"Chi2 for model {self.title} already exists, overwriting")
                df.drop(columns=self.title, inplace=True)
            df = df.join(self.chi2, validate='1:1')
            df.to_hdf(filepath, f"sample_{isample}/chi2")
        else:
            pd.DataFrame(self.chi2).to_hdf(filepath, f"sample_{isample}/chi2")



    def truncate(self, energy_range, E_external_resonances=0):
        min_max_E = (min(energy_range), max(energy_range))
        respar = copy(self.respar)
        respar = respar[(respar.E>=min_max_E[0]-E_external_resonances) & (respar.E<=min_max_E[1]+E_external_resonances)]
        # if self.evaluation_data is not None:
            # eval_data = self.evaluation_data.truncate()
        return Evaluation(self.title, respar)

