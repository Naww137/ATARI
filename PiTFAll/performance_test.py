import os
import numpy as np
import syndat
import subprocess
import PiTFAll as pf
import pandas as pd
import matplotlib.pyplot as plt


class performance_test():

    def __init__(self,
                    number_of_datasets,
                    case_directory,
                    input_options={}):

        ### Gather inputs
        self.number_of_datasets = number_of_datasets
        self.case_directory = case_directory

        ### Default options
        default_options = { 'Overwrite Syndats'    :   False, 
                            'Overwrite Fits'       :   False
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
    

    def __repr__(self):
        return f"This performance test has the following options:\nINSERT OPTIONS HERE"




    def generate(self, Ta_pair, exp, 
                                solver='syndat_SLBW'):

        syndat.MMDA.generate(Ta_pair, exp, 
                                solver, 
                                self.number_of_datasets, 
                                self.case_directory,
                                fixed_resonance_ladder=None, 
                                open_data=None,
                                overwrite=self.overwrite_syndat
                                                                            )

        return


    def fit(self, path_to_application_exe, path_to_fitting_script):

        for i in range(self.number_of_datasets):
            
            sample_directory = os.path.join(self.case_directory, f'sample_{i}')

            # check for isample directory
            if os.path.isdir(sample_directory):

                # check for syndat
                syndat_pw = os.path.join(sample_directory, f'syndat_{i}_pw.csv')
                syndat_par = os.path.join(sample_directory, f'syndat_{i}_par.csv')
                if os.path.isfile(syndat_pw) and os.path.isfile(syndat_par):
                    pass
                else:
                    raise ValueError(f'Syndat in sample directory {os.path.abspath(sample_directory)} does not exist.')

                # check for fits
                fit_pw = os.path.join(sample_directory, f'fit_{i}_pw.csv')
                fit_par = os.path.join(sample_directory, f'fit_{i}_par.csv')
                if os.path.isfile(fit_pw) and os.path.isfile(fit_par):
                    if self.overwrite_fits:
                        out = self.run_fitting_algorithm(syndat_pw, syndat_par, path_to_fitting_script, path_to_application_exe)
                    else:
                        out = "Did not call subprocess for fitting - fits already existed"
                else:
                    out = self.run_fitting_algorithm(syndat_pw, syndat_par, path_to_fitting_script, path_to_application_exe)

            else:
                raise ValueError(f'Sample directory {os.path.abspath(sample_directory)} does not exist.')

        return out
    

    def run_fitting_algorithm(self, syndat_pw, syndat_par,
                                    path_to_fitting_script, path_to_application_exe,
                                    solver=None
                                                                                                ):

        fitting_script = os.path.splitext(os.path.basename(path_to_fitting_script))[0]
        fitting_script_directory = os.path.dirname(path_to_fitting_script)

        out = subprocess.run([f'{path_to_application_exe}', '-nodisplay', '-batch', f'{fitting_script}("{syndat_pw}","{syndat_par}")'],
                                cwd=f'{fitting_script_directory}' , check=False, encoding="utf-8", capture_output=True)

        return out

    
    def analyze(self):

        # initialize case array
        case_array = []
        for i in range(self.number_of_datasets):
            case = pf.sample_case(self.case_directory, i)
            case.analyze()
            case_array.append(case)
        self.case_array = case_array

        # integral analysis (stats for each sample over all energy points)
        SE_df = pd.DataFrame()
        chi_square_df = pd.DataFrame()
        chi_square_perdof_df = pd.DataFrame()
        for case in case_array:
            icase = case.icase
            if icase == 0:
                SE_df = case.FoMs.loc['SE']
                SE_df.name = icase
                chi_square_df = case.FoMs.loc['Chi2']
                chi_square_df.name = icase
                chi_square_perdof_df = case.FoMs.loc['Chi2/dof']
                chi_square_perdof_df.name = icase
            else:
                SE = case.FoMs.loc['SE']
                SE.name = icase
                chi_square = case.FoMs.loc['Chi2']
                chi_square.name = icase
                chi_square_perdof = case.FoMs.loc['Chi2/dof']
                chi_square_perdof.name = icase

                SE_df = pd.merge(SE_df,SE, right_index=True,left_index=True)
                chi_square_df = pd.merge(chi_square_df,chi_square, right_index=True,left_index=True)
                chi_square_perdof_df = pd.merge(chi_square_perdof_df,chi_square_perdof, right_index=True,left_index=True)

        # now transpose and instantiate all
        self.SE_df = SE_df.T
        self.chi_square_df = chi_square_df.T
        self.chi_square_perdof_df = chi_square_perdof_df.T


        return




    def plot_trans(self, icase):

        case = self.case_array[icase]

        plt.figure()
        plt.errorbar(case.pw_data.E, case.pw_data.exp_trans, yerr=case.pw_data.exp_trans_unc, zorder=0, 
                                     fmt='.', color='k', linewidth=1, markersize=3, capsize=2, label='exp')

        plt.plot(case.pw_data.E, case.pw_data.theo_trans, lw=1, color='g', label='sol', zorder=2)
        plt.plot(case.pw_data.E, case.pw_data.est_trans, lw=1, color='r', label='est')

        plt.ylim([-.1, 1])
        plt.xscale('log')
        plt.xlabel('Energy'); plt.ylabel('Transmission')
        plt.legend()

        return