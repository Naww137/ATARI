#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:36:03 2022

@author: noahwalton
"""




# =============================================================================
#       USER INPUT
# =============================================================================

case_basename = 'slbw_5L_allexp'
number_of_cases = 100
number_of_levels = 5

par_template = 'template_5L.par'
inp_template = 'template_allexp.inp'


create_synthetic_data = True

run_sammy_wo_bayes = True

run_bayes_with_baron_suggested_parameters = False




# =============================================================================
# for debugging this code, call individual sub-methods instead of running all at oncc
# =============================================================================

import sammy_interface
si = sammy_interface.sammy_interface() # initialize sammy interface

if create_synthetic_data:
    # create sammy scripts for synthetic data
    [csd_summary_stats, csd_warnings] = si.create_synthetic_data(case_basename, number_of_cases, number_of_levels, run_sammy_wo_bayes, 'true_parameters.csv', par_template, inp_template)
    
if run_bayes_with_baron_suggested_parameters:
    [rb_summary_stats, rb_warnings] = si.run_bayes(case_basename, number_of_cases, number_of_levels, 'baron_parameters.csv', par_template, inp_template)
    
    
    
    
    
    
# =============================================================================
#  print out gathered statistics and warnings - save to run_output file
# =============================================================================
import os
clear = open(os.path.join(os.getcwd(),"run_output.txt"), "w")
clear.close()
f = open(os.path.join(os.getcwd(),"run_output.txt"), "a")
print('Hello!\nPlease find below the warnings and run statistics for your most recent sammy_interface run', file=f)
print('==========================================================================================', file=f)

# warnings printed first
if create_synthetic_data:
    print(file=f); print('create_synthetic_data routine produced the following case warnings:', file=f); print('--------', file=f); print(file=f)
    print("\n".join("{}\t{}".format(k, v) for k, v in csd_warnings.items()), file=f)
    
if run_bayes_with_baron_suggested_parameters:
    print(file=f); print('run_bayes routine produced the following case warnings:', file=f); print('--------', file=f); print(file=f)
    print("\n".join("{}\t{}".format(k, v) for k, v in rb_warnings.items()), file=f)


# stats printed last
if create_synthetic_data:
    print(file=f); print('create_synthetic_data routine ran with the following statistics:', file=f); print('--------', file=f); print(file=f)
    for stat in csd_summary_stats:
        print(stat, file=f)
    
if run_bayes_with_baron_suggested_parameters:
    print(file=f); print('run_bayes routine ran with the following statistics:', file=f); print('--------', file=f); print(file=f)
    for stat in rb_summary_stats:
        print(stat, file=f)
    
f.close()



# =============================================================================
# import os
# sammy_par_filename = 'sammy_bayes.par'
# case_name = case_basename + '_case' + str(32)
# case_directory = os.path.join(os.getcwd(),case_name)
# si.write_par_file_from_template(32, case_directory, par_template, sammy_par_filename, 'baron_parameters.csv');
# =============================================================================

    
