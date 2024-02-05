
import pandas as pd
import numpy as np
from ATARI.syndat.control import Syndat_Control, Syndat_Model
from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT, InitialFBOUT
from ATARI.AutoFit.chi2_eliminator_v2 import elim_OPTs, eliminator_by_chi2, eliminator_OUTput
from ATARI.AutoFit.spin_group_selection import SpinSelectOPT, SpinSelect, SpinSelectOUT
from typing import Optional
from ATARI.sammy_interface import sammy_classes

class Evaluation_Data:
    def __init__(self):
        self.pw_data = []
        self.covariance_data = []
        self.experimental_models = []

    def add_dataset(self,
                    pointwise_data,
                    covariance_data,
                    experimental_model
                    ):
        self.pw_data.append(pointwise_data)
        self.covariance_data.append(covariance_data)
        self.experimental_models.append(experimental_model)



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





