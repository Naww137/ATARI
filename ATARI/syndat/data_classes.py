
import ATARI.utils.hdf5 as h5io
import h5py



class syndatOUT:
    def __init__(self, **kwargs):
        self._title = None
        self._par_true = None
        self._pw_reduced = None
        self._pw_raw = None
        self._covariance_data = {}

        self._true_model_parameters = None

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def title(self):
        return self._title
    @title.setter
    def title(self, title):
        self._title = title
        
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

    def to_hdf5(self, filepath, isample):
        sample_group = f'sample_{isample}'
        
        ### check existing samples
        check_par_true = False
        h5f = h5py.File(filepath, "a")
        if sample_group in h5f:
            if 'par_true' in h5f[sample_group]:
                check_par_true = True
            if self.title in h5f[sample_group]:
                raise ValueError(f"Dataset titled {self.title} already exists in {sample_group}")
        h5f.close()
        
        ### actually write things
        if check_par_true:
            existing_true_par = h5io.read_par(filepath, isample, "true")
            if existing_true_par.equals(self.par_true):
                pass
            else:
                raise ValueError(f"par_true is already written to {sample_group} but does not agree with par_true in {self.title}")
        h5io.write_pw_reduced(filepath, isample, self.title, self.pw_reduced, cov_data=self.covariance_data)
        h5io.write_par(filepath, isample, self.par_true, "true")


    @staticmethod
    def from_hdf5(filepath, isample, title):  # Could make this also title, the loop over title outside of this dataclass
        # sample_group = f'sample_{isample}'
        ### check existing samples
        # h5f = h5py.File(filepath, "r")
        # if sample_group in h5f:
        #     keys = h5f[sample_group].keys()
        #     titles = ['_'.join(each.split('_')[2:]) for each in keys if each not in ['par_true']]
        # h5f.close()

        par_true = h5io.read_par(filepath, isample, 'true')
        pw_reduced_df, cov_data = h5io.read_pw_reduced(filepath, isample, title)
        syndat_out = syndatOUT(title = title, par_true = par_true, pw_reduced = pw_reduced_df, pw_raw = None, covariance_data = cov_data)
        # syndat_out_list = []
        # for title in titles:
        #     pw_reduced_df, cov_data = h5io.read_pw_reduced(filepath, isample, title)
        #     syndat_out = syndatOUT(title = title, par_true = par_true, pw_reduced = pw_reduced_df, pw_raw = None, covariance_data = cov_data)
        #     syndat_out_list.append(syndat_out)
            
        return syndat_out




class syndatOPT:
    """
    Options and settings for a single syndat case.
    
    Parameters
    ----------
    **kwargs : dict, optional
        Any keyword arguments are used to set attributes on the instance.

    Attributes
    ----------
    sampleRES : bool = True
        Sample a new resonance ladder with each sample.
    sample_counting_noise : bool = True
        Option to sample counting statistic noise for data generation, if False, no statistical noise will be sampled.
    calculate_covariance : bool = True
        Indicate whether to calculate off-diagonal elements of the data covariance matrix.
    explicit_covariance : bool = False
        Indicate whether to return explicit data covariance elements or the decomposed statistical and systematic covariances with systematic derivatives.
    sampleTMP : bool = True
        Option to sample true underlying measurement model (data-reduction) parameters for data generation.
    sampleTNCS : bool = True
        Option to sample true neutron count spectrum for data generation.
    smoothTNCS : bool = False
        Option to use a smoothed function for the true neutron count spectrum for data generation.
    save_raw_data : bool = False
        Option to save raw count data, if False, only the reduced transmission data will be saved.
    force_zero_to_1 : bool = True
        Option to force sampled 0-counts to counts of 1. 
        This option is true by default as an artifact of the synthetic data methodology. 
        In actuallity, un-grouped time bins (on the order of ns widths) may have 0-counts, but the experimentalist will determine a grouping structure that avoids this.
        Because they synthetic data methodology samples counting statistics on the grouped bin structure, 0-counts are unrealistic.
        The combination of settings, particularly energy grid and linac triggers, can cause 0-counts.
        This is an indication that your settings are likely not realistic, however, you should still be able to generate usable data and 0-counts will cause downstream issues.
    """
    def __init__(self, **kwargs):
        self._sampleRES = True
        self._sample_counting_noise = True
        self._calculate_covariance = True
        self._explicit_covariance = False
        self._sampleTMP = True
        self._sampleTNCS = True
        self._smoothTNCS = False
        
        # self._save_raw_data = False
        # self._saveTMP = False

        self._force_zero_to_1 = True


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
    def sampleTMP(self):
        return self._sampleTMP
    @sampleTMP.setter
    def sampleTMP(self, sampleTMP):
        self._sampleTMP = sampleTMP

    # @property
    # def saveTMP(self):
    #     return self._saveTMP
    # @saveTMP.setter
    # def saveTMP(self, saveTMP):
    #     self._saveTMP = saveTMP

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

    @property
    def force_zero_to_1(self):
        return self._force_zero_to_1
    @force_zero_to_1.setter
    def force_zero_to_1(self, force_zero_to_1):
        self._force_zero_to_1 = force_zero_to_1


