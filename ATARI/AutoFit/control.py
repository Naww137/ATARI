
import pandas as pd
import numpy as np
from ATARI.syndat.control import Syndat_Control, Syndat_Model
from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT, InitialFBOUT
from ATARI.AutoFit.chi2_eliminator_v2 import elim_OPTs, eliminator_by_chi2, eliminator_OUTput
from ATARI.AutoFit.spin_group_selection import SpinSelectOPT, SpinSelect, SpinSelectOUT
from typing import Optional
from ATARI.sammy_interface import sammy_classes

from ATARI.utils import hdf5 as h5io
from dataclasses import dataclass
from typing import Optional
from copy import deepcopy, copy


@dataclass
class Evaluation_Data:
    """
    Data class to hold all data used in an evaluation, i.e., experimental pointwise data, covariance information, and experimental model classes.
    """

    datasets                : tuple
    covariance_data         : tuple
    experimental_models     : tuple
    par_true                : Optional[pd.DataFrame]


    @classmethod
    def from_hdf5(cls, experimental_models, experimental_titles, sample_file, isample):
        """
        Construct an Evaluation_Data instance from data hdf5 file.


        Parameters
        ----------
        experimental_models : _type_
            _description_
        experimental_titles : _type_
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

        try: # need better logic to check for true
            par_true = h5io.read_par(sample_file, isample, 'true')
        except:
            par_true = None

        datasets = []
        covariance_data = []
        for title in experimental_titles:
            pw_reduced_df, cov_data = h5io.read_pw_reduced(sample_file, isample, title)
            datasets.append(pw_reduced_df)
            covariance_data.append(cov_data)

        return cls(tuple(datasets), tuple(covariance_data), tuple(experimental_models), par_true)


    def to_hdf5(self, sample_file, isample):
        # save to hdf5 isample folder
        pass

    def to_matrix_form(self):
        # functions to convert to matrix form for external solves (see IFB_dev/fit_w_derivative)
        pass

    def truncate(self, energy_range, E_external_resonances = 0):
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
            else:# 'CovT' in exp_cov.keys() or 'CovY' in exp_cov.keys():
                raise ValueError("Filtering not implemented for explicit cov yet")
        
            covariance_data.append(filtered_cov)

        # truncate experiments and par_true
        experiments = deepcopy(self.experimental_models)
        par_true = copy(self.par_true)
        par_true = par_true[(par_true.E>=min_max_E[0]-E_external_resonances) & (par_true.E<=min_max_E[1]+E_external_resonances)]

        return Evaluation_Data(tuple(datasets), tuple(covariance_data), tuple(experiments), par_true)



class AutoFitOPT:
    def __init__(self,
                 featurebankOPT: Optional[InitialFBOPT] = None,
                 eliminateOPT: Optional[elim_OPTs]      = None,
                 spinselectOPT: Optional[SpinSelectOPT] = None,
                 **kwargs
                 ):
        
        if featurebankOPT is None:
            self.featurebankOPT = InitialFBOPT()
        else:
            self.featurebankOPT = featurebankOPT
        if eliminateOPT is None:
            self.elimOPT = elim_OPTs()
        else:
            self.eliminateOPT = eliminateOPT
        if spinselectOPT is None:
            self.spinselectOPT = SpinSelectOPT()
        else:
            self.spinselectOPT = spinselectOPT


        # for key, value in kwargs.items():
        #     setattr(self, key, value)



class AutoFitOUT:
    def __init__(self, 
                 initialFBOUT: Optional[InitialFBOUT] = None,
                 eliminateOUT: Optional[eliminator_OUTput] = None,
                 eliminateFiltered: Optional[dict] = None,
                 spinselectOUT: Optional[SpinSelectOUT] = None,
                 total_time: Optional[float]= None
                 ):
        
        self.initial = initialFBOUT
        self.eliminate = eliminateOUT
        self.eliminate_filtered = eliminateFiltered
        self.spinselect = spinselectOUT
        self.total_time = total_time


class AutoFit_Control:

    def __init__(self,
                 sammy_rto: sammy_classes.SammyRunTimeOptions,
                 autofitOPT: Optional[AutoFitOPT]
                 ):
        
        self.sammy_rto = sammy_rto

        if autofitOPT is None:
            self.autofitOPT = AutoFitOPT()
        else:
            self.autofitOPT = autofitOPT

        self.initial_fit = InitialFB(self.autofitOPT.featurebankOPT)
        self.eliminator = eliminator_by_chi2(options = self.autofitOPT.eliminateOPT)
        self.spinselect = SpinSelect(self.autofitOPT.spinselectOPT)

    
    def fit(self, 
            particle_pair,
            energy_range,
            datasets,
            experiments,
            covariance_data,
            sammy_rto_fit
            ):
        
        ### initial feature bank solve
        initial_out = self.initial_fit.fit(particle_pair,
                                           energy_range,
                                           datasets,
                                           experiments,
                                           covariance_data,
                                           sammy_rto_fit
                                           )
        
        initial_out_final_par = initial_out[-1].par_post
        assert isinstance(initial_out_final_par, pd.DataFrame)
        initial_out_final_par["varyGg"] = np.zeros(len(initial_out_final_par))
        # need to get side resonances here 


        ### eliminator solve
        eliminate_out = self.eliminator.eliminate(initial_out_final_par)

        # return models 5 res - 20 res 

        ### spin group selector





    def train(self, training_data):
        if isinstance(training_data, Syndat_Control):
            # sample syndat each time
            # option to save each sample
            pass
        elif isinstance(training_data, str):
            # read saved training data from hdf5 file
            pass
            
        # do training
        # update internal hyperparameters (dchi2 for AIC, starting widths)



    def validate(self, validation_data):
        if isinstance(validation_data, Syndat_Control):
            # sample syndat each time
            # option to save each sample
            pass
        elif isinstance(validation_data, str):
            # read saved training data from hdf5 file
            pass
            
        # do validation on trained hyperparameters
        # report performance





