#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:38:36 2022

@author: noahwalton
"""
import sys
import os
import shutil
import pandas as pd


class sammy_interface:
    
    def __init__(self):
        
        print()
        print("Hi! Welcome to Sammy Interface")
        print()
        
        
        
        

# =============================================================================
#               sub-method
# =============================================================================
    def pad_res_parms(self, parm, length):
        
        if len(parm) > length:
            difference = len(parm) - length 
            decimals = len(parm.split('.')[1])
            parm = float(parm)
            padded_parm = str(round(float(parm),decimals-difference))
            if len(padded_parm) < length:
                padded_parm = padded_parm.ljust(length,'0')
            else:
                _=0
        else:
            padded_parm = parm.ljust(length,'0')
            
        return padded_parm
    
# =============================================================================
#           sub-method
# =============================================================================
    def write_inp_file_from_template(self, case_directory, inp_template, sammy_inp_filename, solve_bayes):
        with open(os.path.realpath(inp_template),'r') as f:
            inp_template_lines = f.readlines()
            f.close()
        with open(os.path.join(case_directory,sammy_inp_filename),'w') as f:
            for line in inp_template_lines:
                if line.startswith('%%% solve bayes ? %%%'):
                    if solve_bayes:
                        f.write('# do not solve bayes equation\n')
                    else:
                        f.write('do not solve bayes equation\n')
                else:
                    f.write(line)
            f.close()       
            
# =============================================================================
#             sub-method
# =============================================================================
    def write_par_file_from_template(self, icase, case_directory, number_of_levels, par_template, sammy_par_filename, csv):
        with open(os.path.realpath(par_template),'r') as f:
            par_template_lines = f.readlines()
            f.close()
            
        sub_method_warning = []
        parameters = pd.read_csv(os.path.realpath(csv), index_col=0)
        
        #!!! add warning if icase does not exit in csv
        case_parms = parameters.loc[icase]
        
        if len(case_parms) != number_of_levels*3:
            sub_method_warning.append('User input number of resonances per case does not match true parameter file')
                
        for ilevel in range(1,number_of_levels+1):
            energy = str(case_parms[f'E{ilevel}']); energy = self.pad_res_parms(energy,11)
            Gg = str(case_parms[f'Gg{ilevel}']); Gg = self.pad_res_parms(Gg,10)
            Gn = str(case_parms[f'Gn{ilevel}']); Gn = self.pad_res_parms(Gn,10)
            
            if float(energy) == 0 or float(Gg) == 0 or float(Gn) == 0:
                sub_method_warning.append('One or more ResParms == 0')
            
            par_template_lines[ilevel-1] = par_template_lines[ilevel-1].replace(f'%%%E_level{ilevel}%%%', energy)
            par_template_lines[ilevel-1] = par_template_lines[ilevel-1].replace(f'%%%_Gg_{ilevel}%%%', Gg)
            par_template_lines[ilevel-1] = par_template_lines[ilevel-1].replace(f'%%%_Gn_{ilevel}%%%', Gn)
        
        with open(os.path.join(case_directory,sammy_par_filename), 'w') as par_file:
            for line in par_template_lines:
                par_file.write(line)
            par_file.close()     
            
        return sub_method_warning
# =============================================================================
#             sub-method
# =============================================================================

    def write_awk_shell_script(self, icase, case_directory):
        with open(os.path.join(case_directory,'awk.sh'),'w') as awksh:
            awksh.write("""#!/bin/bash

#PBS -V
#PBS -l nodes=1:ppn=1
#PBS -q fill

cd ${PBS_O_WORKDIR}

awk '{print "     " $1 "     " $4 "      " $4}' SAMMY.LST > syndat_"""+str(icase))

# =============================================================================
#             sub-method
# =============================================================================
    def write_qsub_shell_script(self, icase, case_directory):
        with open(os.path.join(case_directory,f'qsub_{icase}.sh'), 'w') as f:
            f.write("""#!/bin/bash

#PBS -V
#PBS -l nodes=1:ppn=1
#PBS -q fill

cd ${PBS_O_WORKDIR}

/home/nwalton1/my_sammy/SAMMY/sammy/build/install/bin/sammy < piped_sammy_commands.sh""")


# =============================================================================
#       sub-method 
# =============================================================================
    def run_awk(self,case_basename,first_case,last_case):
        
        # delete awk.sh.* files - these files indicate that qsub job has completed
        for icase in range(first_case,last_case+1):
            wildcard_path = f'/home/nwalton1/my_sammy/{case_basename}/{case_basename}_case{icase}/awk.sh.*'
            os.system(f'rm {wildcard_path}')
            
        # run awk.sh files   
        irunawk = 0
        run_cases = range(first_case,last_case+1)
        for i in run_cases:
            directory = f'/home/nwalton1/my_sammy/{case_basename}/{case_basename}_case{i}'
            os.system("ssh -t necluster.ne.utk.edu 'cd "+directory+" ; qsub awk.sh'")
            irunawk += 1
            # ssh -t necluster.ne.utk.edu 'cd /home/nwalton1/my_sammy/slbw_testing/slbw_fitting_case1/ ; /home/nwalton1/my_sammy/SAMMY/sammy/build/install/bin/sammy < slbw_fitting_case1.sh'
        print(); print(f'submitted {irunawk} awk scripts'); print()
        
        
        # wait on all cases to complete running - looking for qsub_icase.sh.o file
        running_awk = True
        print(); print('Waiting for awk to run'); print()
        while running_awk:
            case_run_bool = []
            for icase in range(first_case,last_case+1):
                directory = os.path.join(os.getcwd(), f'{case_basename}_case{icase}')
                
                idone_file = 0
                for file in os.listdir(directory):
                    if file.startswith('awk.sh.o'):
                        idone_file += 1
                    else:
                        _ = 0
                        
                if idone_file > 0:
                    case_run_bool.append(False)
                else:
                    case_run_bool.append(True)
                    
            if any(case_run_bool):
                continue
            else:
                running_awk = False
            icases_still_running = case_run_bool.count(True)
            print(f'Waiting on {icases_still_running} to complete') #!!! this could be done better - only prints this when all are complete for some reason
            



# =============================================================================
#       sub-method
# =============================================================================
    def copy_syndat(self,case_basename,first_case,last_case):
        if os.path.isdir(os.path.join(f'/home/nwalton1/my_sammy/{case_basename}','synthetic_data')):
            _ = 0
        else:
            os.mkdir(os.path.join(f'/home/nwalton1/my_sammy/{case_basename}','synthetic_data'))
        run_cases = range(1,last_case+1); icopy = 0
        for i in run_cases:
            
            shutil.copy(os.path.join(f'/home/nwalton1/my_sammy/{case_basename}',case_basename+f'_case{i}',f'syndat_{i}'), os.path.join(f'/home/nwalton1/my_sammy/{case_basename}','synthetic_data'))
            #os.system("scp nwalton1@necluster.ne.utk.edu:/home/nwalton1/my_sammy/slbw_testing_noexp/slbw_1L_noexp_case1/syndat_{i} /Users/noahwalton/research_local/resonance_fitting/synthetic_data")
            icopy += 1
            # ssh -t necluster.ne.utk.edu 'cd /home/nwalton1/my_sammy/slbw_testing/slbw_fitting_case1/ ; /home/nwalton1/my_sammy/SAMMY/sammy/build/install/bin/sammy < slbw_fitting_case1.sh'
        print(); print(f'copied {icopy} synthetic data files'); print()
        
        

# =============================================================================
#       sub-methd
# =============================================================================
    def run_sammy_and_wait(self, case_basename, number_of_cases):
        
        # delete qsub_icase.sh.* files - these files indicate that qsub job has completed
        for icase in range(1,number_of_cases+1):
            wildcard_path = f'/home/nwalton1/my_sammy/{case_basename}/{case_basename}_case{icase}/qsub_{icase}.sh.*'
            os.system(f'rm {wildcard_path}')
            
        # run sammy with bayes for all files created
        irunsammy = 0
        for icase in range(1,number_of_cases+1):
            directory = f'/home/nwalton1/my_sammy/{case_basename}/{case_basename}_case{icase}'
            os.system("ssh -t necluster.ne.utk.edu 'cd "+directory+f" ; qsub qsub_{icase}.sh'")
            irunsammy += 1
            
        # wait on all cases to complete running - looking for qsub_icase.sh.o file
        running_sammy = True
        print(); print('Waiting for sammy to run'); print()
        while running_sammy:
            case_run_bool = []
            for icase in range(1,number_of_cases+1):
                directory = os.path.join(os.getcwd(), f'{case_basename}_case{icase}')
                
                idone_file = 0
                for file in os.listdir(directory):
                    if file.startswith(f'qsub_{icase}.sh.o'):
                        idone_file += 1
                    else:
                        _ = 0
                        
                if idone_file > 0:
                    case_run_bool.append(False)
                else:
                    case_run_bool.append(True)
                    
            if any(case_run_bool):
                continue
            else:
                running_sammy = False
            icases_still_running = case_run_bool.count(True)
            print(f'Waiting on {icases_still_running} to complete') #!!! this could be done better - only prints this when all are complete for some reason
            
            return irunsammy
            

# =============================================================================
#       sub-method
# =============================================================================
    def pull_parameters_from_bayes_updated_par(self, case_basename, number_of_cases):
        my_df = pd.DataFrame({'E':[], 'Gg':[], 'Gn':[]})
        cases_that_did_not_run = []
        for icase in range(1,number_of_cases+1):
            directory = f'/home/nwalton1/my_sammy/{case_basename}/{case_basename}_case{icase}'
            bayes_updated_par_file = os.path.join(directory,'SAMMY.PAR')
            if os.path.isfile(bayes_updated_par_file):
                with open(bayes_updated_par_file,'r') as f:
                    lines = f.readlines()
                    splitline = lines[0].split()
                    case_df = pd.DataFrame({'E':splitline[0], 'Gg':splitline[1], 'Gn':splitline[2]}, index=[icase])
            else:
                cases_that_did_not_run.append(icase)
                case_df = pd.DataFrame({'E':0, 'Gg':0, 'Gn':0}, index=[icase])
            
            my_df = pd.concat([my_df,case_df])
        
        if len(cases_that_did_not_run) > 0:
            print(); print('The following cases did not update with bayes');
            print(cases_that_did_not_run)
            
        sorted_df = my_df.sort_index()
        sorted_df.to_csv(os.path.join(os.getcwd(),'bayes_parameters.csv'))



  
# =============================================================================
#         PRIMARY METHOD
# =============================================================================
    def create_synthetic_data(self, case_basename, number_of_cases, number_of_levels, run_sammy, Truecsv, par_template, inp_template):
        
        method_warnings = {}
        if type(case_basename) != str:
            print("WARNING: basename given for case is not of type: str")
            sys.exit()
        if type(number_of_cases) != int:
            print("WARNING: number of cases given is not of type: int")
            sys.exit()
            
        
        cwd = os.getcwd()
        copied_E_struct = 0; inputs_created = 0; pipefiles_created = 0; par_created = 0
        
        for icase in range(1,number_of_cases+1):
            
            case_name = case_basename + '_case' + str(icase)
            case_directory = os.path.join(cwd,case_name)
            
            if os.path.isdir(case_directory):
                _ = 0
            else:
                os.mkdir(case_directory)
            
            sammy_inp_filename = 'sammy_syndat.inp'
            sammy_par_filename = 'sammy_syndat.par'
            E_structure_filename = 'E_structure'
            piped_commands_filename = 'piped_sammy_commands.sh'
            
            # perform submethod actions and return sub-method warnings
            self.write_inp_file_from_template(case_directory, inp_template, sammy_inp_filename, False); inputs_created += 1
            smw2 = self.write_par_file_from_template(icase, case_directory, number_of_levels, par_template, sammy_par_filename, Truecsv); par_created += 1
            self.write_qsub_shell_script(icase, case_directory)
            self.write_awk_shell_script(icase, case_directory)
            
            # copy estructure file into case directory
            shutil.copy(os.path.join(cwd,E_structure_filename), case_directory); copied_E_struct += 1
            
            # create piped_sammy_commands.sh
            with open(os.path.join(case_directory, piped_commands_filename) , 'w') as pipefile:
                line1 = sammy_inp_filename
                line2 = sammy_par_filename
                line3 = E_structure_filename
                pipefile.write(line1 + '\n' + line2 + '\n' + line3 + '\n\n'); pipefiles_created += 1
                
            # gather warnings
            case_warnings = [smw2]
            method_warnings[f'case_{icase}'] = case_warnings
        
        if run_sammy:
            print();print('going to run sammy to create synthetic data'); print()
            irunsammy = self.run_sammy_and_wait(case_basename, number_of_cases)
            self.run_awk(case_basename,1,number_of_cases)
            self.copy_syndat(case_basename,1,number_of_cases)
        else:
            irunsammy = 0
            
        # summarizy statistics
        statistics = [f'Copied {copied_E_struct} energy structure data files to unique directores',
                      f'Created {inputs_created} input files',
                      f'Created {par_created} parameter files',
                      f'Created {pipefiles_created} piped sammy command shell files',
                      f'Submitted {irunsammy} SAMMY jobs not solving bayes']

        
        return [statistics, method_warnings]

# =============================================================================
#       PRIMARY METHOD
# =============================================================================
    def run_bayes(self, case_basename, number_of_cases, number_of_levels, Baroncsv, par_template, inp_template):
        
        method_warnings = {}
        if type(case_basename) != str:
            print("WARNING: basename given for case is not of type: str")
            sys.exit()
        if type(number_of_cases) != int:
            print("WARNING: number of cases given is not of type: int")
            sys.exit()
            
        
        cwd = os.getcwd()
        inputs_created = 0; pipefiles_created = 0; par_created = 0; irunsammy =0
        
        
        for icase in range(1,number_of_cases+1):
            
            case_name = case_basename + '_case' + str(icase)
            case_directory = os.path.join(cwd,case_name)
            if os.path.isdir(case_directory):
                _ = 0
            else:
                print('Tried running bayes in non-existent case directory'); sys.exit()
            
            sammy_inp_filename = 'sammy_bayes.inp'
            sammy_par_filename = 'sammy_bayes.par'
            syndat_filename = 'syndat_'+str(icase)
            piped_commands_filename = 'piped_sammy_commands.sh'
            
            # write input file in case directory from template
            self.write_inp_file_from_template(case_directory, inp_template, sammy_inp_filename, True); inputs_created += 1
            # write par file in case directory from template
            smw2 = self.write_par_file_from_template(icase, case_directory, number_of_levels, par_template, sammy_par_filename, Baroncsv); par_created += 1
            
            # write qsub script only if one does not exist
            if os.path.isfile(os.path.join(case_directory,f'qsub_{icase}.sh')):
                _ = 0 
            else:
                self.write_qsub_shell_script(icase, case_directory)
            
            # create piped_sammy_commands.sh
            with open(os.path.join(case_directory, piped_commands_filename) , 'w') as pipefile:
                line1 = sammy_inp_filename
                line2 = sammy_par_filename
                line3 = syndat_filename
                pipefile.write(line1 + '\n' + line2 + '\n' + line3 + '\n\n'); pipefiles_created += 1
                
            # gather case warnings
            case_warnings = [smw2]
            method_warnings[f'case_{icase}'] = case_warnings
            
        print();print('going to run sammy to solve bayes'); print()
        irunsammy = self.run_sammy_and_wait(case_basename, number_of_cases)
        
        # gather summary statistics
        # summarizy statistics
        statistics = [f'submitted {irunsammy} SAMMY jobs solving bayes',
                      f'Created {inputs_created} input files',
                      f'Created {par_created} parameter files',
                      f'Overwrote {pipefiles_created} piped sammy command shell files']
        
        # read out bayesian updated parameters from SAMMY.PAR files
        self.pull_parameters_from_bayes_updated_par(case_basename, number_of_cases)
        
        #!!! could include an iterative sammy loop here if we haven't converged to true values or some level of SE 

        return [statistics, method_warnings]


# =============================================================================
# #!!! can remove this method
# # =============================================================================
# #       PRIMARY METHOD  
# # =============================================================================
#     def run_sammy(self,case_basename,first_case,last_case):
#         irunsammy = 0
#         run_cases = range(first_case,last_case+1)
#         for i in run_cases:
#             directory = f'/home/nwalton1/my_sammy/{case_basename}/{case_basename}_case{i}'
#             os.system("ssh -t necluster.ne.utk.edu 'cd "+directory+f" ; qsub qsub_{i}.sh'")
#             irunsammy += 1
#             # ssh -t necluster.ne.utk.edu 'cd /home/nwalton1/my_sammy/slbw_testing/slbw_fitting_case1/ ; /home/nwalton1/my_sammy/SAMMY/sammy/build/install/bin/sammy < slbw_fitting_case1.sh'
#         print(); print(f'submitted {irunsammy} SAMMY Jobs'); print()
# =============================================================================





    

            



