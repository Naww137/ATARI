
import pandas as pd

class autofitOPT:
    def __init__(self):
        pass


class feature_bank:
    def __init__(self):
        pass

class eliminate_resonances:
    def __init__(self):
        pass

# options = 

class autofit:

    def __init__(self):
        pass

    def initial_FB(self, saminp, samrto, options) -> pd.DataFrame:
        
        if options ==0: 
            out = Oleksii_FB_initial()

        elif options ==1:
            out = Noah_FB_initial()




