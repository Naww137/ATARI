


class syndatOUT:
    def __init__(self, **kwargs):
        self._par_true = None
        self._pw_reduced = None
        self._pw_raw = None
        self._covariance_data = {}

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def par_true(self):
        return self._par_true
    @par_true.setter
    def par_true(self, par_true):
        self._par_true = par_true
        
    @property
    def pw_raw(self):
        return self._pw_raw
    @pw_raw.setter
    def pw_raw(self, pw_raw):
        self._pw_raw = pw_raw

    @property
    def pw_reduced(self):
        return self._pw_reduced
    @pw_reduced.setter
    def pw_reduced(self, pw_reduced):
        self._pw_reduced = pw_reduced

    @property
    def covariance_data(self):
        return self._covariance_data
    @covariance_data.setter
    def covariance_data(self, covariance_data):
        self._covariance_data = covariance_data





class syndatOPT:
    """
    Options and settings for a single syndat case.
    
    Parameters
    ----------
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    Attributes
    ----------
    sampleRES : bool
        Sample a new resonance ladder with each sample.
    sample_counting_noise : bool = False
        Option to sample counting statistic noise for data generation, if False, no statistical noise will be sampled.
    calculate_covariance : bool = True
        Indicate whether to calculate off-diagonal elements of the data covariance matrix.
    explicit_covariance : bool = False
        Indicate whether to return explicit data covariance elements or the decomposed statistical and systematic covariances with systematic derivatives.
    sampleTURP : bool
        Option to sample true underlying measurement model (data-reduction) parameters for data generation.
    sampleTNCS : bool
        Option to sample true neutron count spectrum for data generation.
    smoothTNCS : bool
        Option to use a smoothed function for the true neutron count spectrum for data generation.
    save_raw_data : bool
        Option to save raw count data, if False, only the reduced transmission data will be saved.
    """
    def __init__(self, **kwargs):
        self._sampleRES = True
        self._sample_counting_noise = True
        self._calculate_covariance = False
        self._explicit_covariance = False
        self._sampleTURP = True
        self._sampleTNCS = True
        self._smoothTNCS = False
        self._save_raw_data = False


        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        string=''
        for prop in dir(self):
            if not callable(getattr(self, prop)) and not prop.startswith('_'):
                string += f"{prop}: {getattr(self, prop)}\n"
        return string
    
    @property
    def sampleRES(self):
        return self._sampleRES
    @sampleRES.setter
    def sampleRES(self, sampleRES):
        self._sampleRES = sampleRES
        
    @property
    def sampleTURP(self):
        return self._sampleTURP
    @sampleTURP.setter
    def sampleTURP(self, sampleTURP):
        self._sampleTURP = sampleTURP

    @property
    def sampleTNCS(self):
        return self._sampleTNCS
    @sampleTNCS.setter
    def sampleTNCS(self, sampleTNCS):
        self._sampleTNCS = sampleTNCS

    @property
    def smoothTNCS(self):
        return self._smoothTNCS
    @smoothTNCS.setter
    def smoothTNCS(self, smoothTNCS):
        self._smoothTNCS = smoothTNCS

    @property
    def sample_counting_noise(self):
        return self._sample_counting_noise
    @sample_counting_noise.setter
    def sample_counting_noise(self, sample_counting_noise):
        self._sample_counting_noise = sample_counting_noise

    @property
    def calculate_covariance(self):
        return self._calculate_covariance
    @calculate_covariance.setter
    def calculate_covariance(self, calculate_covariance):
        self._calculate_covariance = calculate_covariance
    
    @property
    def explicit_covariance(self):
        return self._explicit_covariance
    @explicit_covariance.setter
    def explicit_covariance(self, explicit_covariance):
        self._explicit_covariance = explicit_covariance

    @property
    def save_raw_data(self):
        return self._save_raw_data
    @save_raw_data.setter
    def save_raw_data(self, save_raw_data):
        self._save_raw_data = save_raw_data


