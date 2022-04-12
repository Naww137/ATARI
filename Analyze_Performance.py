#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:10:42 2022

@author: noahwalton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

data_directory = "/Users/noahwalton/Library/Mobile Documents/com~apple~CloudDocs/Research Projects/Resonance Fitting/sammy/"
case_directory = os.path.join(data_directory,'slbw_testing_1L_allexp') # case 51 didn't run, 
has_bayes_solutions = True

true_parameters = pd.read_csv(case_directory+'/true_parameters.csv') #pd.read_csv(os.path.realpath("True_Parameters_(E,Gg,Gn).csv"))
baron_parameters = pd.read_csv(case_directory+'/baron_parameters.csv') #pd.read_csv(os.path.realpath("Baron_parameters.csv"))




# put baron width parameters into meV
baron_parameters['Gg1'] = baron_parameters['Gg1']#.apply(lambda x: x*1e3)
baron_parameters['Gn1'] = baron_parameters['Gn1']#.apply(lambda x: x*1e3)

plt.scatter(true_parameters['Gn1'],baron_parameters['Gn1'])
plt.xlabel('True Gn'); plt.ylabel('Baron Pred. Gn')
plt.title('With ms experimental corrections')
#plt.ylim([17500,28000])
#plt.ylim([45000,48550])



baron_error = abs(true_parameters-baron_parameters)/true_parameters*100

if has_bayes_solutions:
    bayes_parameters = pd.read_csv(case_directory+"/bayes_parameters.csv")
    bayes_error = abs(true_parameters-bayes_parameters)/true_parameters*100


# put energy error in terms of total width
min_vec_baron = []; max_vec_baron = []
min_vec_bayes = []; max_vec_bayes = []
for each in true_parameters:
    min_vec_baron.append(min(baron_error[each]))    
    max_vec_baron.append(max(baron_error[each]))
    if has_bayes_solutions:
        max_vec_bayes.append(max(bayes_error[each]))
        min_vec_bayes.append(min(bayes_error[each]))
    

if has_bayes_solutions:
    summary_statistics = pd.DataFrame({'Min Baron Error' : min_vec_baron[1:],'Max Baron Error' : max_vec_baron[1:], \
                                       'Min Bayes Error': min_vec_bayes[1:], 'Max Bayes Error': max_vec_bayes[1:]}, \
                                          index = ['E', 'Gg', 'Gn'])

    #display(summary_statistics)

