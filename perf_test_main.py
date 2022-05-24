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


TrueNumPeaks = 3 
number_of_samples = 3

loop_peaks = [3, 4, 5, 6, 7, 8]
loop_energies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000]



# =============================================================================
#  create appropriate directories
# =============================================================================
case_basename = 'perf_test_3true'
case_directory = os.path.join(os.getcwd(),case_basename)

if os.path.isdir(case_directory):
    _ = 0
else:
    os.mkdir(case_basename)


    
for peak in loop_peaks:
    for epts in loop_energies:
    
        parm_name = str(peak) + '_peaks_' + str(epts) + '_epts'
        parm_directory = os.path.join(case_directory, f'{parm_name}')
        if os.path.isdir(parm_directory):
            _ = 0
        else:
            os.mkdir(parm_directory)
                
        for isample in range(1,number_of_samples+1):
            
            output_name = 'parm_name.out'
            # create matlab script from template file
            
    

# =============================================================================
# run job array within each (peak, epts)
# =============================================================================

# =============================================================================
# if jobs are done, add to average/distribution of outputs for that (peak. epts) case
# =============================================================================











