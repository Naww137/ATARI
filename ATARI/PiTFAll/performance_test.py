import os
import numpy as np
import syndat
import subprocess
from ATARI import PiTFAll as pf
import pandas as pd
import h5py


class performance_test():

    def __init__(self,
                    dataset_range,
                    case_file,
                    input_options={}):

        ### Gather inputs
        self.dataset_range = dataset_range
        self.case_file = case_file

        ### Default options
        default_options = { 'Overwrite Syndats'    :   False, 
                            'Overwrite Fits'       :   False,
                            'Use HDF5'             :   True,
                            'Vary Erange'          :   None
                                                                    } 

        ### redefine options dictionary if any input options are given
        options = default_options
        for old_parameter in default_options:
            if old_parameter in input_options:
                options.update({old_parameter:input_options[old_parameter]})
        for input_parameter in input_options:
            if input_parameter not in default_options:
                raise ValueError('User provided an unrecognized input option')
        self.options = options

        ### Gather options
        # self.overwrite_syndat = options['Overwrite Syndats']
        # self.overwrite_fits = options['Overwrite Fits']
        # self.use_hdf5 = options['Use HDF5']
        # self.vary_Erange = options['Vary Erange']

###
    def __repr__(self):
        return f"This performance test has the following options:\nINSERT OPTIONS HERE"


###
    def check_case_directory(self,case_directory):
        if os.path.isdir(case_directory):
            pass
        else:
            os.mkdir(case_directory)
        return




# ===========================================================================================
#   METHODS FOR GENERATING SYNTHETIC DATA
# ===========================================================================================


###
    def generate_syndats(self, particle_pair, experiment, 
                                solver='syndat_SLBW'
                                                        ):
        ### generate syndats
        samples_not_generated = syndat.MMDA.generate(particle_pair, experiment, 
                                                    solver, 
                                                    self.dataset_range, 
                                                    self.case_file,
                                                    fixed_resonance_ladder=None, 
                                                    open_data=None,
                                                    vary_Erange=self.options['Vary Erange'],
                                                    use_hdf5=self.options['Use HDF5'],
                                                    overwrite=self.options['Overwrite Syndats']
                                                                                                )
        ### Get test-level statistics
        FoMs = []
        for i in range(min(self.dataset_range), max(self.dataset_range)):
            FoMs_sample = pf.sample_case.analyze_syndat(self.case_file, i)
            FoMs.append(FoMs_sample)

        sample_data_df = pd.DataFrame(FoMs, columns=['isample',
                                                     'NumRes',
                                                     'NumEpts',
                                                     'theo_exp_SE',
                                                     'theo_avg_gnx2',
                                                     'theo_avg_Gg', 
                                                     'theo_min_gnx2', 
                                                     'theo_min_Gg', 
                                                     'theo_max_gnx2', 
                                                     'theo_max_Gg'])
        sample_data_df.set_index('isample', inplace=True)
                        
        if self.options['Use HDF5']:
            sample_data_df.to_hdf(self.case_file, 'test_stats/sample_data')
        else:
            self.check_case_directory(os.path.join(self.case_file,'test_stats/'))
            sample_data_df.to_csv(os.path.join(self.case_file,'test_stats/sample_data.csv'))

        if not self.options['Overwrite Syndats']:
            samples_to_be_run = np.setdiff1d(np.arange(0,max(self.dataset_range)), np.array(samples_not_generated))
            min_value = min(samples_to_be_run, default=max(self.dataset_range))

            out = f"User chose to NOT overwrite previously generated datasets in the file {self.case_file}.\n\
Samples  {min(self.dataset_range)}-{min_value} already existed.\n\
Samples {min_value}-{max(self.dataset_range)} were generated.\n\
If Syndat generation settings were changed these files should be overwriten."
            
        else:
            out = ""

        return sample_data_df, out







# ===========================================================================================
#   METHODS FOR ANALYZING FITS
# ===========================================================================================

###
    def analyze_printout(self, integral_FoMs_df):
        mean_fit_exp_chi2dof, std_fit_exp_chi2dof = np.mean(integral_FoMs_df.fit_exp_chi2dof), np.std(integral_FoMs_df.fit_exp_chi2dof)
        mean_fit_theo_MSE, std_fit_theo_MSE = np.mean(integral_FoMs_df.fit_theo_MSE), np.std(integral_FoMs_df.fit_theo_MSE)
        printout = f"The mean/std of the fit to experimental chi2/dof is {mean_fit_exp_chi2dof} +/- {std_fit_exp_chi2dof} in transmission space.\n\
The mean/std of the fit to theorectical MSE is {mean_fit_theo_MSE} +/- {std_fit_theo_MSE} in cross section space."
        return printout
    
###
    def analyze(self, particle_pair, experiment, fit_name):
        
        ### Loop over isamples
        integral_pw_FoMs  = []
        integral_par_FoMs  = []
        bv_pw_inwindow = [] 
        for i in range(min(self.dataset_range), max(self.dataset_range)):
            integral_pw_FoMs_sample, integral_par_FoMs_sample, bv_pw_inwindow_sample = pf.sample_case.analyze_fit(self.case_file, i, experiment, particle_pair, fit_name)
            bv_pw_inwindow.append(bv_pw_inwindow_sample)
            integral_pw_FoMs.append(integral_pw_FoMs_sample)
            integral_par_FoMs.append(integral_par_FoMs_sample)

        ### create dataframes
        integral_pw_FoMs_df = pd.DataFrame(integral_pw_FoMs, columns=['isample',
                                                                    'fit_theo_MSE'    ,
                                                                    'fit_exp_SE'      ,
                                                                    'fit_exp_chi2'    , 
                                                                    'fit_exp_chi2dof' , 
                                                                    'theo_exp_SE'     , 
                                                                    'theo_exp_chi2'   , 
                                                                    'theo_exp_chi2dof',
                                                                    'tfit'] )
        
        integral_par_FoMs_df = pd.DataFrame(integral_par_FoMs, columns=['isample',
                                                                    'est_card', 
                                                                    'est_avg_Gg', 
                                                                    'est_avg_gnx2',
                                                                    'est_min_Gg', 
                                                                    'est_min_gnx2', 
                                                                    'est_max_Gg', 
                                                                    'est_max_gnx2'] )
        
        bv_pw_inwindow_df = pd.DataFrame(bv_pw_inwindow, columns=['isample',
                                                                    'WE_midpoint',
                                                                    'window_bias_xs',
                                                                    'window_bias_trans', 
                                                                    'window_variance_xs', 
                                                                    'window_variance_trans'] )
        
        integral_pw_FoMs_df.set_index('isample', inplace=True)
        integral_par_FoMs_df.set_index('isample', inplace=True)
        bv_pw_inwindow_df.set_index('isample', inplace=True)

        if self.options['Use HDF5']:
            integral_pw_FoMs_df.to_hdf(self.case_file, 'test_stats/integral_pw_FoMs')
            # TODO: write integral par FoMs !!!
            sample_data_df = pd.read_hdf(self.case_file, 'test_stats/sample_data')   # read out sample data so it can be returned with self.analyze
        else:
            integral_pw_FoMs_df.to_csv(os.path.join(self.case_file, 'test_stats/integral_pw_FoMs.csv'))
            sample_data_df = pd.read_csv(os.path.join(self.case_file, 'test_stats/sample_data.csv'))

        printout = self.analyze_printout(integral_pw_FoMs_df)

        return integral_pw_FoMs_df, integral_par_FoMs_df, bv_pw_inwindow_df, sample_data_df, printout



















# ===========================================================================================
#   METHODS FOR GENERATING FITS
# ===========================================================================================


###
    def generate_fits(self, run_local,
                            path_to_application_exe = None, 
                            path_to_fitting_script  = None):
        
        ### Check input
        if run_local:
            if (path_to_application_exe is None) or (path_to_fitting_script is None):
                raise ValueError('User chose to run fitting algorithm locally but did not supply the proper paths.')
            else:
                pass

        ### prepare case_file.hdf5 or csv's based on overwrite option and run_fitting_algorithm() - if not run_local this function does nothing
        if self.options['Use HDF5']:
            samples_not_being_run = self.prepare_and_fit_hdf5(run_local, path_to_application_exe,  path_to_fitting_script)
        else:
            samples_not_being_run = self.prepare_and_fit_csv(run_local, path_to_application_exe,  path_to_fitting_script)

        # TODO: if not running locally, generate a jobarray.sh with appropriate isamples and print breif instructions
        if not run_local:
            samples_to_be_run = np.setdiff1d(np.arange(0,max(self.dataset_range)), np.array(samples_not_being_run))
            min_value = min(samples_to_be_run, default=max(self.dataset_range))

            out = f"User chose to NOT run the fitting algorithm locally.\n\
The data file {self.case_file} has been prepared based on the selected overwrite options.\n\
Please run samples {min_value}-{max(self.dataset_range)}"

        else:
            out = ""
                        
        return out

###
    # def read_csv_to_hdf5(directory):
    #     # dataset_range = (0,11)
    #     # directory = '/Users/noahwalton/Documents/GitHub/ATARI/Fitting'
    #     # case_file = 'perf_test_baron.hdf5'
    #     for i in range(min(dataset_range), max(dataset_range)):
    #         csv_2_hdf5(directory, case_file, i, 'baron')

    #     return

###
    def prepare_and_fit_csv(self, run_local, path_to_application_exe,  path_to_fitting_script):

        samples_not_being_run = []
        for i in range(min(self.dataset_range), max(self.dataset_range)):
            sample_directory = os.path.join(self.case_file, f'sample_{i}')
            # check for isample directory
            if os.path.isdir(sample_directory):
                # check for syndat
                syndat_pw = os.path.join(sample_directory, f'syndat_pw.csv')
                syndat_par = os.path.join(sample_directory, f'syndat_par.csv')
                if os.path.isfile(syndat_pw) and os.path.isfile(syndat_par):
                    pass
                else:
                    raise ValueError(f'Syndat in sample directory {os.path.abspath(sample_directory)} does not exist.')
                # check for fits
                fit_pw = os.path.join(sample_directory, f'fit_pw.csv')
                fit_par = os.path.join(sample_directory, f'fit_par.csv')
                if os.path.isfile(fit_pw) and os.path.isfile(fit_par):
                    if self.options['Overwrite Fits']:
                        out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.options['Use HDF5'])
                    else:
                        samples_not_being_run.append(i)
                else:
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.options['Use HDF5'])
            else:
                raise ValueError(f'Sample directory {os.path.abspath(sample_directory)} does not exist.')
            
        return samples_not_being_run
    

###
    def prepare_and_fit_hdf5(self, run_local, path_to_application_exe,  path_to_fitting_script):

        samples_not_being_run = []
        for i in range(min(self.dataset_range), max(self.dataset_range)):
            f = h5py.File(self.case_file, "r+")
            sample_group = f'sample_{i}'
            if sample_group in f:
                # check for syndat
                if ('exp_pw' in f[sample_group]) and ('theo_par' in f[sample_group]):
                    pass
                else:
                    raise ValueError(f'Syndat in sample group {sample_group} does not exist in {self.case_file}.')
                # if both exist, either overwrite or return
                if ('est_par' in f[sample_group]):
                    if self.options['Overwrite Fits']:
                        del f[sample_group]['est_par']
                        out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.options['Use HDF5'])
                    else:
                        samples_not_being_run.append(i)
                # if only one exists must have been an error - delete and re-run fit
                elif 'fit_pw' in f[sample_group]:
                    del f[sample_group]['fit_pw']
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.options['Use HDF5'])
                elif 'fit_par' in f[sample_group]:
                    del f[sample_group]['fit_par']
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.options['Use HDF5'])
                # if neither exist, run fit
                else:
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.options['Use HDF5'])
            # raise error if sample group does not exist
            else:
                raise ValueError(f'Sample group {sample_group} does not exist in {self.case_file}.')
        # close hdf5 file    
        f.close()
                
        return samples_not_being_run

###
    def run_fitting_algorithm(self, case_file, isample, run_local,
                                    path_to_fitting_script, path_to_application_exe,
                                    use_hdf5,
                                    solver=None
                                                                                                ):
        if use_hdf5:
            if run_local:
                fitting_script = os.path.splitext(os.path.basename(path_to_fitting_script))[0]
                fitting_script_directory = os.path.dirname(path_to_fitting_script)

                out = subprocess.run([f'{path_to_application_exe}', '-nodisplay', '-batch', f'{fitting_script}("{case_file}",{isample})'],
                                        cwd=f'{fitting_script_directory}' , check=False, encoding="utf-8", capture_output=True)
            else:
                out = 0
        else:
            raise NotImplementedError("Need to update run_local capabilities for use_hdf5=False")
            # if run_local:
            #     fitting_script = os.path.splitext(os.path.basename(path_to_fitting_script))[0]
            #     fitting_script_directory = os.path.dirname(path_to_fitting_script)

            #     out = subprocess.run([f'{path_to_application_exe}', '-nodisplay', '-batch', f'{fitting_script}("{case_file}",{isample}, {use_hdf5})'],
            #                             cwd=f'{fitting_script_directory}' , check=False, encoding="utf-8", capture_output=True)
            # else:
            #     out = 0

        return out
