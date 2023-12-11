
import pandas as pd

from ATARI.syndat.control import Syndat_Control, Syndat_Model

class autofitOPT:
    def __init__(self):
        pass



# options = 

class AutoFit_Control:

    def __init__(self):
        pass
    


    
    def fit(self, data):
        # determine data type
        # fit with trained and validated hyperparameters
        pass




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





