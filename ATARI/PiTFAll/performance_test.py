import os
import numpy as np
import syndat
import subprocess
from ATARI import PiTFAll as pf
import pandas as pd
import h5py


class performance_test():

    def __init__(self,
                    number_of_datasets,
                    case_file,
                    input_options={}):

        ### Gather inputs
        self.number_of_datasets = number_of_datasets
        self.case_file = case_file

        ### Default options
        default_options = { 'Overwrite Syndats'    :   False, 
                            'Overwrite Fits'       :   False,
                            'Use HDF5'             :   True
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
        self.overwrite_syndat = options['Overwrite Syndats']
        self.overwrite_fits = options['Overwrite Fits']
        self.use_hdf5 = options['Use HDF5']

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

###
    def generate_syndats(self, particle_pair, experiment, 
                                solver='syndat_SLBW'
                                                        ):
        ### generate syndats
        syndat.MMDA.generate(particle_pair, experiment, 
                                solver, 
                                self.number_of_datasets, 
                                self.case_file,
                                fixed_resonance_ladder=None, 
                                open_data=None,
                                use_hdf5=self.use_hdf5,
                                overwrite=self.overwrite_syndat
                                                                            )
        ### Get test-level statistics
        NumRes = []
        NumEpts = []
        theo_exp_SE = []
        for i in range(self.number_of_datasets):
            Res, Epts, te_SE = pf.sample_case.analyze_syndat(self.case_file, i)
            NumRes.append(Res)
            NumEpts.append(Epts)
            theo_exp_SE.append(te_SE)

        sample_data_df = pd.DataFrame(  {'NumRes'   :   NumRes,
                                         'NumEpts'  :   NumEpts,
                                         'theo_exp_SE': theo_exp_SE})
        
        if self.use_hdf5:
            sample_data_df.to_hdf(self.case_file, 'test_stats/sample_data')
        else:
            self.check_case_directory(os.path.join(self.case_file,'test_stats/'))
            sample_data_df.to_csv(os.path.join(self.case_file,'test_stats/sample_data.csv'))

        return sample_data_df



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
        if self.use_hdf5:
            samples_not_being_run = self.prepare_and_fit_hdf5(run_local, path_to_application_exe,  path_to_fitting_script)
        else:
            samples_not_being_run = self.prepare_and_fit_csv(run_local, path_to_application_exe,  path_to_fitting_script)

        # TODO: if not running locally, generate a jobarray.sh with appropriate isamples and print breif instructions
        if not run_local:
            samples_to_be_run = np.setdiff1d(np.arange(0,self.number_of_datasets), np.array(samples_not_being_run))
            min_value = min(samples_to_be_run, default=self.number_of_datasets)

            out = f"User chose to NOT run the fitting algorithm locally. \
The data file {self.case_file} has been prepared based on the selected overwrite options. \
Please run samples {min_value}-{self.number_of_datasets}"
                        
        return out


###
    def prepare_and_fit_csv(self, run_local, path_to_application_exe,  path_to_fitting_script):

        samples_not_being_run = []
        for i in range(self.number_of_datasets):
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
                    if self.overwrite_fits:
                        out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.use_hdf5)
                    else:
                        samples_not_being_run.append(i)
                else:
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.use_hdf5)
            else:
                raise ValueError(f'Sample directory {os.path.abspath(sample_directory)} does not exist.')
            
        return samples_not_being_run
    

###
    def prepare_and_fit_hdf5(self, run_local, path_to_application_exe,  path_to_fitting_script):

        samples_not_being_run = []
        for i in range(self.number_of_datasets):
            f = h5py.File(self.case_file, "r+")
            sample_group = f'sample_{i}'
            if sample_group in f:
                # check for syndat
                if ('syndat_pw' in f[sample_group]) and ('syndat_par' in f[sample_group]):
                    pass
                else:
                    raise ValueError(f'Syndat in sample group {sample_group} does not exist in {self.case_file}.')
                # if both exist, either overwrite or return
                if ('fit_pw' in f[sample_group]) and ('fit_par' in f[sample_group]):
                    if self.overwrite_fits:
                        del f[sample_group]['fit_pw']
                        del f[sample_group]['fit_par']
                        out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.use_hdf5)
                    else:
                        samples_not_being_run.append(i)
                # if only one exists must have been an error - delete and re-run fit
                elif 'fit_pw' in f[sample_group]:
                    del f[sample_group]['fit_pw']
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.use_hdf5)
                elif 'fit_par' in f[sample_group]:
                    del f[sample_group]['fit_par']
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.use_hdf5)
                # if neither exist, run fit
                else:
                    out = self.run_fitting_algorithm(self.case_file, i, run_local, path_to_fitting_script, path_to_application_exe, self.use_hdf5)
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


###
    def analyze(self, particle_pair):
        
        ### build integral figures of merit
        fit_theo_MSE = []
        fit_exp_SE = []; fit_exp_chi2 = []; fit_exp_chi2dof = []
        theo_exp_SE = []; theo_exp_chi2 = []; theo_exp_chi2dof = []
        for i in range(self.number_of_datasets):
            # analyze the case
            FoM = pf.sample_case.analyze_fit(self.case_file, i, particle_pair)
            # append key FoMs
            fit_theo_MSE.append(FoM.fit_theo.SE) 
            fit_exp_SE.append(FoM.fit_exp.SE); fit_exp_chi2.append(FoM.fit_exp.Chi2); fit_exp_chi2dof.append(FoM.fit_exp['Chi2/dof']) 
            theo_exp_SE.append(FoM.theo_exp.SE); theo_exp_chi2.append(FoM.theo_exp.Chi2) ; theo_exp_chi2dof.append(FoM.theo_exp['Chi2/dof']) 

        integral_FoMs_df = pd.DataFrame(  { 'fit_theo_MSE'      :   fit_theo_MSE    ,
                                            'fit_exp_SE'        :   fit_exp_SE      ,
                                            'fit_exp_chi2'      :   fit_exp_chi2    , 
                                            'fit_exp_chi2dof'   :   fit_exp_chi2dof , 
                                            'theo_exp_SE'       :   theo_exp_SE     , 
                                            'theo_exp_chi2'     :   theo_exp_chi2   , 
                                            'theo_exp_chi2dof'  :   theo_exp_chi2dof    })

        if self.use_hdf5:
            integral_FoMs_df.to_hdf(self.case_file, 'test_stats/integral_FoMs')
            sample_data_df = pd.read_hdf(self.case_file, 'test_stats/sample_data')   # read out sample data so it can be returned with self.analyze
        else:
            integral_FoMs_df.to_csv(os.path.join(self.case_file, 'test_stats/integral_FoMs.csv'))
            sample_data_df = pd.read_csv(os.path.join(self.case_file, 'test_stats/sample_data.csv'))

        return integral_FoMs_df, sample_data_df


