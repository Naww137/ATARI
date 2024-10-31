
from ATARI.utils import hdf5 as h5io
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy, copy
import h5py
import pandas as pd
from ATARI.sammy_interface.sammy_functions import get_endf_parameters
from ATARI.AutoFit.functions import update_vary_resonance_ladder


def get_train_test_from_list(list_object, i_exclude, itotal):
    train = [deepcopy(list_object[i]) for i in range(itotal) if i != i_exclude] 
    test = [list_object[i_exclude]]
    return train, test



@dataclass
class Evaluation_Data:
    """
    Data class to hold data used in an evaluation, i.e.,  experimental titles/models, experimental pointwise data, and covariance information.
    """

    experimental_titles     : tuple
    experimental_models     : tuple
    datasets                : tuple
    covariance_data         : tuple
    
    measurement_models      : Optional[tuple]    = None
    experimental_models_no_pup : Optional[tuple] = None


    @property
    def N(self):
        return sum([len(each) for each in self.datasets])

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
                for k in model_keys:
                    if k in pw_reduced_df.keys():
                        print(f"Experimental pointwise models {model_keys} already exist in dataframe for {title}, overwriting")
                        pw_reduced_df.drop(columns=k, inplace=True)
                if pw_reduced_df.merge(data, how='inner', validate="1:1").empty:
                    raise ValueError(f"One of the following columns does not match in {isample}, {title}: E, exp, exp_unc") 
                data = pw_reduced_df.merge(data, how='inner', validate="1:1")
            h5io.write_pw_reduced(filepath, isample, title, data, cov_data=cov)



    def to_matrix_form(self):
        # functions to convert to matrix form for external solves (see IFB_dev/fit_w_derivative)
        pass

    def get_train_test_over_datasets(self, i_test):
        """
        Splits datasets into training set and testing set where testing set is the single dataset determined by i_test arguement.
        Returns two new evaluation data instances, one with training data only and the other with testing data only.

        Parameters
        ----------
        i_test : int
            Index of dataset to be isolated in the test set.

        Returns
        -------
        _type_
            _description_
        """
        itotal = len(self.datasets)

        experimental_titles_train, experimental_titles_test = get_train_test_from_list(self.experimental_titles, i_test, itotal)
        experimental_models_train, experimental_models_test = get_train_test_from_list(self.experimental_models, i_test, itotal)
        datasets_train, datasets_test = get_train_test_from_list(self.datasets, i_test, itotal)
        covariance_data_train, covariance_data_test = get_train_test_from_list(self.covariance_data, i_test, itotal)
        
        if self.measurement_models:
            measurement_models_train, measurement_models_test = get_train_test_from_list(self.measurement_models, i_test, itotal)
            measurement_models_train, measurement_models_test = tuple(measurement_models_train), tuple(measurement_models_test)
        else:
            measurement_models_train, measurement_models_test = None, None
            
        if self.experimental_models_no_pup:
            experimental_models_no_pup_train, experimental_models_no_pup_test = get_train_test_from_list(self.experimental_models_no_pup, i_test, itotal)
            experimental_models_no_pup_train, experimental_models_no_pup_test = tuple(experimental_models_no_pup_train), tuple(experimental_models_no_pup_test)
        else:
            experimental_models_no_pup_train, experimental_models_no_pup_test = None, None

        eval_data_train = Evaluation_Data(tuple(experimental_titles_train), tuple(experimental_models_train), tuple(datasets_train), tuple(covariance_data_train), measurement_models=measurement_models_train, experimental_models_no_pup=experimental_models_no_pup_train)
        eval_data_test = Evaluation_Data(tuple(experimental_titles_test), tuple(experimental_models_test), tuple(datasets_test), tuple(covariance_data_test), measurement_models=measurement_models_test, experimental_models_no_pup=experimental_models_no_pup_test)
        
        return eval_data_train, eval_data_test
    

    def truncate(self, energy_range):
        """
        Truncates all evaluation data to a specific energy domain.
        Returns new evaluation data instance.

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

        experiments = deepcopy(self.experimental_models)
        for exp in experiments:
            exp.truncate_energy_range(energy_range)

        if self.experimental_models_no_pup:
            experiments_no_pup = deepcopy(self.experimental_models_no_pup)
            for exp_no_pup in experiments_no_pup:
                exp_no_pup.truncate_energy_range(energy_range)
            experiments_no_pup = tuple(experiments_no_pup)
        else:
            experiments_no_pup = None

        if self.measurement_models:
            measurement_models = deepcopy(self.measurement_models)
            for meas in measurement_models:
                if meas:
                    meas.truncate_energy_range(energy_range)
            measurement_models = tuple(measurement_models)
        else:
            measurement_models = None
        

        return Evaluation_Data(tuple(self.experimental_titles), tuple(experiments), tuple(datasets), tuple(covariance_data), measurement_models=measurement_models, experimental_models_no_pup=experiments_no_pup)




@dataclass
class Evaluation:
    """
    Dataclass to hold information about an evaluation (model), i.e., model title, resonance parameters, and pointwise reconstructions.
    """

    title                   : str
    respar                  : pd.DataFrame

    external_resonance_indices: Optional[list] = None
    
    pw_theoretical          : Optional[pd.DataFrame] = None
    evaluation_data         : Optional[Evaluation_Data] = None
    chi2                    : Optional[pd.Series] = None
    

    @property
    def resonance_ladder(self):
        return self.respar
    
    @classmethod
    def from_pkl(title, filepath):
        pass
        # needs to be specific to AutoFitOUT pkl? Not sure if that object is stable yet

    @classmethod
    def from_hdf5(cls, title, filepath, isample, experimental_titles=None, experimental_models=None):
        respar = h5io.read_par(filepath, isample, title)
        try:
            chi2 = pd.read_hdf(filepath, key=f"sample_{isample}/chi2")[title]
        except:
            chi2 = None
        # # could also look for existing experimental and theoretical pointwise and chi2 - have to see if this evalutations title is in the dataframe columns
        # if experimental_titles is not None and experimental_models is not None:
        #     h5f = h5py.File(filepath, 'r')
        #     for each in experimental_titles:
        #     h5f.close()
        # from_hdf5(cls, experimental_titles, experimental_models, filepath, isample)
        return cls(title, respar, chi2=chi2)
    
    @classmethod
    def from_ENDF6(cls, title, matnum, filepath, sammyRTO):
        sammyRTO = copy(sammyRTO)
        resonance_ladder = get_endf_parameters(filepath, matnum, sammyRTO)
        return cls(title, resonance_ladder)
    
    @classmethod
    def from_samout(cls, title, samout, post=True, external_resonance_indices=None):
        if post:
            return cls(title, respar=samout.par_post, chi2=samout.chi2_post, external_resonance_indices=external_resonance_indices)
        else:
            return cls(title, respar=samout.par, chi2=samout.chi2, external_resonance_indices=external_resonance_indices)
    

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

    def to_hdf5(self, filepath, isample, overwrite=True, experimental_titles = None):

        chi2_for_eval = True
        if isinstance(self.chi2, list):
            if experimental_titles is None:
                raise ValueError("Please provide ordered experimental titles or change self.chi2 to series")
            self.chi2 = self.chi2_list_to_series(self.chi2, self.title, experimental_titles)
        elif self.chi2 is None:
            chi2_for_eval = False

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

        if chi2_for_eval:
            if chi2_exists:
                df = pd.read_hdf(filepath, f"sample_{isample}/chi2")
                if self.title in df.keys():
                    print(f"Chi2 for model {self.title} already exists, overwriting")
                    df.drop(columns=self.title, inplace=True)
                df = df.join(self.chi2, validate='1:1')
                df.to_hdf(filepath, key=f"sample_{isample}/chi2")
            else:
                pd.DataFrame(self.chi2).to_hdf(filepath, key=f"sample_{isample}/chi2")



    def truncate(self, energy_range, external_resonance_energy_buffer=0, reset_index=True, inplace=False):
        if inplace: respar = self.respar
        else: respar = copy(self.respar)

        min_max_E = (min(energy_range), max(energy_range))
        
        respar = respar[(respar.E>=min_max_E[0]-external_resonance_energy_buffer) & (respar.E<=min_max_E[1]+external_resonance_energy_buffer)]

        # if self.evaluation_data is not None:
            # eval_data = self.evaluation_data.truncate()
        if reset_index:
            respar.reset_index(inplace=True, drop=True)

        external_respar = respar[(respar.E<=min_max_E[0]) | (respar.E>=min_max_E[1])]
        external_resonance_indices = external_respar.index.to_list()

        if inplace: 
            self.external_resonance_indices = external_resonance_indices
            self.respar = respar
        else: 
            return Evaluation(self.title, respar, external_resonance_indices=external_resonance_indices)
        
    def update_vary_on_resonance_ladder(self,varyE, varyGg, varyGn1):
        self.respar = update_vary_resonance_ladder(self.respar, varyE=varyE, varyGg=varyGg, varyGn1=varyGn1)

