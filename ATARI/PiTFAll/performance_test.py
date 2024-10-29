import os
import h5py
from ATARI.syndat.data_classes import syndatOUT
from ATARI.PiTFAll import fnorm
import ATARI.utils.hdf5 as h5io
import ATARI.utils.plotting as myplot


class PerformanceTest:

    def __init__(self, 
                 filepath, 
                #  energy_region,
                 syndat_sample_filepath = None
                 ):
        
        if os.path.isfile(filepath):
            pass
        else:
            h5f = h5py.File(filepath, "w")
            h5f.close()
            
        self.filepath = filepath
        # self.energy_region = energy_region
        self.loaded_data = {}

        if syndat_sample_filepath is not None:
            self.add_synthetic_data_samples(syndat_sample_filepath, hdf5=True)

    def clear_loaded_data(self):
        self.loaded_data = {}

    def load_data(self,
                  par_true = True,
                  doppler_theo = True,
                  exp_dat = False):
        pass
        # check for already loaded data
        # load new data

    def add_synthetic_data_samples(self, 
                                   syndat_sample_filepath, 
                                   hdf5=True,
                                   keep_pars = True,
                                   keep_exp = False
                                   ):
        """
        Adds syndat samples of true resonance parameters and experimental datasets to the Performance Test instance.
        Currently only implemented for HDF5 samples.

        Parameters
        ----------
        syndat_sample_filepath : str
            Path to the syndat samples hdf5 file.
        hdf5 : bool, optional
            _description_, by default True

        Raises
        ------
        ValueError
            hdf5 False is not yet implemented
        """
        if hdf5:
            h5f = h5py.File(syndat_sample_filepath, "r")
            isample_keys = [each for each in h5f.keys()]
            h5f.close()
            h5f = h5py.File(self.filepath, "r")
            isample_keys_new = [i.split('_')[1] for i in isample_keys if i not in h5f.keys()]
            h5f.close()
            for i in isample_keys_new:
                syndat_out_list = syndatOUT.from_hdf5(syndat_sample_filepath, i) # load each sample/title with .from_hdf5
                for out in syndat_out_list:
                    out.to_hdf5(self.filepath, i)

                    
                #     if keep_exp:
                #         # self.loaded_data[i][f"exp_dat_{out.title}"] = {"pw_reduced":out.pw_reduced}
                #         raise ValueError("Not Implemented")
                # if keep_pars:
                #     self.loaded_data[f"sample_{i}"] = {"par_true":syndat_out_list[0].par_true}
                
        else:
            raise ValueError("Not yet implemented for hdf5=False")
    

    def calculate_doppler_theoretical(self, 
                                      sammy_exe, 
                                      particle_pair, 
                                      energy_range,
                                      model = "true",
                                      temperature = 300,
                                      template =  os.path.realpath(os.path.join(os.path.dirname(__file__), "../sammy_interface/sammy_templates/dop_2sg.inp")),
                                      reactions = ['elastic', 'capture']
                                      ):

        h5f = h5py.File(self.filepath, "r")
        isamples = [isample_key.split('_')[1] for isample_key in h5f.keys() if f'par_{model}' in h5f[isample_key]]
        h5f.close()
        
        for i in isamples:
            resonance_ladder = h5io.read_par(self.filepath, i, model)
            data_frame = fnorm.calc_theo_broad_xs_for_all_reaction(sammy_exe,
                                            particle_pair, 
                                            resonance_ladder, 
                                            energy_range,
                                            temperature,
                                            template, 
                                            reactions)
            


        
