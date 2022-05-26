#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:44:18 2022

@author: noahwalton
"""

import sys
import os
import shutil
import pandas as pd



# =============================================================================
#  User inputs
# =============================================================================

TrueNumPeaks = 3 
number_of_samples = 2

jobs_submitted_perstep_perparm = 2



loop_peaks = [3]
loop_energies = [500]

# =============================================================================
# loop_peaks = [3, 4, 5, 6, 7, 8]
# loop_energies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]
# =============================================================================

create_parmdir_and_smplfiles = True
submit_jobarrays = False


# =============================================================================
# read template files in as strings
# =============================================================================
matlab_template_file = open(os.path.join(os.getcwd(),'perf_test_template.m'), "r")
matlab_template_string = matlab_template_file.read()
matlab_template_file.close()

sub_jobarray_template_file = open(os.path.join(os.getcwd(),'sub_jobarray_template.sh'))
sub_jobarray_template =  sub_jobarray_template_file.read()
sub_jobarray_template_file.close()
job_array_specifications = f'1-{number_of_samples}%{jobs_submitted_perstep_perparm}'



# =============================================================================
#  create appropriate directories
# =============================================================================

if create_parmdir_and_smplfiles:
    
    case_basename = 'perf_test_3true'
    case_directory = os.path.join(os.getcwd(),case_basename)

    if os.path.isdir(case_directory):
        _ = 0
    else:
        os.mkdir(case_basename)
    
    for peak in loop_peaks:
        for epts in loop_energies:
        
            parm_name = f'p_{peak}_{epts}_e'
            parm_directory = os.path.join(case_directory, f'{parm_name}')
            if os.path.isdir(parm_directory):
                _ = 0
            else:
                os.mkdir(parm_directory)
                    
            for isample in range(1,number_of_samples+1):
                
                output_name = f'{parm_name}_smpl_{isample}.out'
                figure_name = f'{parm_name}_smpl_{isample}.png'
                
                new_matlab_string = matlab_template_string
                new_matlab_string = new_matlab_string.replace('%%%TrueNumPeaks%%%', str(TrueNumPeaks))
                new_matlab_string = new_matlab_string.replace('%%%NumPeaks%%%', str(peak))
                new_matlab_string = new_matlab_string.replace('%%%EnergyPoints%%%', str(epts))
                new_matlab_string = new_matlab_string.replace('%%%output_filename%%%', output_name)
                new_matlab_string = new_matlab_string.replace('%%%main_directory%%%', os.getcwd())
                new_matlab_string = new_matlab_string.replace('%%%figure_filename%%%', figure_name)
                
                with open(os.path.join(parm_directory, f'{parm_name}_smpl_{isample}.m'), 'w') as matscript:
                    matscript.write(new_matlab_string)
                    matscript.close()
                                           
            # create a job array submit file to run all samples for each parm case
            new_sub_jobarray_template = sub_jobarray_template
            new_sub_jobarray_template = new_sub_jobarray_template.replace('%%%parm_name%%%', parm_name)
            new_sub_jobarray_template = new_sub_jobarray_template.replace('%%%array_specs%%%', job_array_specifications)
            with open(os.path.join(parm_directory, f'{parm_name}_jobarray.sh'), 'w') as subscript:
                subscript.write(new_sub_jobarray_template)
                subscript.close()
                
       
        
        
# =============================================================================
# run job array within each (peak, epts)
# =============================================================================
if submit_jobarrays:
    
    for peak in loop_peaks:
        for epts in loop_energies:
            
            parm_name = f'p_{peak}_{epts}_e'
            parm_directory = os.path.join(case_directory, f'{parm_name}')
            os.system(f"ssh -t necluster.ne.utk.edu 'cd {parm_directory} ; qsub {parm_name}_jobarray.sh'")



# =============================================================================
# if jobs are done, add to average/distribution of outputs for that (peak. epts) case
# =============================================================================











