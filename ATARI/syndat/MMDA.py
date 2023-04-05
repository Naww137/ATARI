#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 11:26:07 2022

@author: noahwalton
"""

import os
import syndat
import shutil
import pandas as pd
import numpy as np
from pandas import HDFStore
import h5py
from syndat.scattering_theory import SLBW


###
def check_case_file(case_file):
    if os.path.isfile(case_file):
        if case_file.endswith('.hdf5'):
            pass
        else:
            raise ValueError("User supplied an unknown file type, must have extension .hdf5")
    elif os.path.isdir(case_file):
        raise ValueError("User supplied True for 'use_hdf5' but supplied a path to a directory rather than an hdf5 file.")
    else:
        print(f'User provided case file does not exist, making file at : {os.path.abspath(case_file)}')
    return

###
def check_case_directory(case_directory):
    if os.path.isdir(case_directory):
        pass
    elif os.path.isfile(case_directory):
        raise ValueError("User supplied False for 'use_hdf5' but supplied a path to a file rather than a directory.")
    else:
        print(f'User provided data directory does not exist, making directory at : {os.path.abspath(case_directory)}')
        os.mkdir(case_directory)
    return

###
def check_sample_directory(sample_directory):
    if os.path.isdir(sample_directory):
        pass
    else:
        os.mkdir(sample_directory)
    return

###
def random_energy_domain(Erange_total, maxres, probability, average_spacing):

    prob_comb = np.power(probability, (1/maxres))
    val = np.sqrt(2/np.pi) * np.sqrt(np.log(1/(1 - 2*prob_comb))) # (2*np.sqrt(np.log(1/(1 - prob_lessthan_comb))))/np.sqrt(np.pi)
    eV_spacing = val*average_spacing
    window_size = eV_spacing*maxres

    random_window_start = np.random.default_rng().uniform(min(Erange_total),max(Erange_total)-window_size,1)
    window_energy_domain = np.append(random_window_start, random_window_start+window_size)

    return window_energy_domain

###
def fine_egrid(energy):
    minE = min(energy); maxE = max(energy)
    n = int((maxE - minE)*1e2)
    new_egrid = np.linspace(minE, maxE, n)
    return new_egrid

###
def compute_theoretical(solver, experiment, particle_pair, resonance_ladder, theo_pw):

    if theo_pw:
        energy_grid = fine_egrid(experiment.energy_domain)
    else:
        energy_grid = experiment.energy_domain

    if solver == 'syndat_SLBW':
        xs_tot, xs_scat, xs_cap = syndat.scattering_theory.SLBW(energy_grid, particle_pair, resonance_ladder)
        n = experiment.redpar.val.n  # atoms per barn or atoms/(1e-12*cm^2)
        trans = np.exp(-n*xs_tot)

    elif solver == 'sammy_SLBW':
        raise ValueError("Need to implement ability to compute theoretical transmission with solver other that 'syndat' (SLBW)")
    elif solver == 'sammy_RM':
        raise ValueError("Need to implement ability to compute theoretical transmission with solver other that 'syndat' (SLBW)")
    else:
        raise ValueError("Need to implement ability to compute theoretical transmission with solver other that 'syndat' (SLBW)")
    
    if theo_pw:
        theoretical_df = pd.DataFrame({'E':energy_grid, 'theo_xs':xs_tot})
    else:
        theoretical_df = pd.DataFrame({'E':energy_grid, 'theo_trans':trans})

    return theoretical_df

###
def sample_syndat(particle_pair, experiment, solver,
                    open_data, fixed_resonance_ladder, vary_Erange):

    # either sample resonance ladder or set it to fixed resonance ladder
    if fixed_resonance_ladder is None:
        if vary_Erange is not None:
            new_energy_domain = random_energy_domain(vary_Erange['fullrange'], vary_Erange['maxres'], vary_Erange['prob'], particle_pair.average_parameters['dE']['3.0'])
            experiment.def_self_energy_grid(new_energy_domain)
        else:
            pass
        resonance_ladder = particle_pair.sample_resonance_ladder(experiment.energy_domain, particle_pair.spin_groups, particle_pair.average_parameters)
    else:
        if vary_Erange:
            raise ValueError("Options to vary the energy range were provided, but so was a fixed resonance ladder.")
        resonance_ladder = fixed_resonance_ladder

    # Compute expected xs or transmission
    theoretical_df = compute_theoretical(solver, experiment, particle_pair, resonance_ladder, False)

    # run the experiment
    experiment.run(theoretical_df, open_data)

    # Build data frame for export
    pw_df = pd.DataFrame({  'E':experiment.trans.E, 
                            'theo_trans':experiment.theo.theo_trans,
                            'exp_trans':experiment.trans.exp_trans
                                                                    })
    
    CoV = experiment.CovT

    return resonance_ladder, pw_df, CoV

###
def sample_and_write_syndat(case_file, isample, particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange, use_hdf5):

    # sample theoretical parameters and compute transmission on experimentaal fine-grid theoretical pw cross section
    resonance_ladder, exp_pw_df, CovT = sample_syndat(particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange)
    theo_pw_df = compute_theoretical(solver, experiment, particle_pair, resonance_ladder, True)

    # write data
    if use_hdf5:
        exp_pw_df.to_hdf(case_file, f"sample_{isample}/exp_pw")
        theo_pw_df.to_hdf(case_file, f"sample_{isample}/theo_pw")
        resonance_ladder.to_hdf(case_file, f"sample_{isample}/theo_par") 
        pd.DataFrame(CovT, index=np.array(exp_pw_df.E), columns=exp_pw_df.E).to_hdf(case_file, f"sample_{isample}/exp_cov")
    else:
        exp_pw_df.to_csv(os.path.join(case_file,f'sample_{isample}','exp_pw'), index=False)
        theo_pw_df.to_csv(os.path.join(case_file,f'sample_{isample}','theo_pw'), index=False) 
        resonance_ladder.to_csv(os.path.join(case_file,f'sample_{isample}','theo_par'), index=False)
        pd.DataFrame(CovT, index=np.array(exp_pw_df.E), columns=exp_pw_df.E).to_csv(os.path.join(case_file,f'sample_{isample}','exp_cov'))

    return



def generate(particle_pair, experiment, 
                solver, 
                dataset_range, 
                case_file,
                fixed_resonance_ladder=None, 
                open_data=None,
                vary_Erange = None,
                use_hdf5=True,
                overwrite=True
                                                            ):
    """
    Generate multiple synthetic experimental datasets in a convenient directory structure. 

    _extended_summary_

    Parameters
    ----------
    particle_pair : syndat.particle_pair.particle_pair
        Syndat particle pair object
    experiment : syndat.experiment.experiment
        Syndat experiment object
    solver : str
        Which solver will be used to calculate theoretical cross section/transmission.
    dataset_range : int
        Min/Max samples to generate.
    case_file : str
        Full path to hdf5 file or top level case directory where data will be stored.
    fixed_resonance_ladder : DataFrame or None
        If a DataFrame is given, resonance parameters will not be sampled. Must be given in proper Syndat format.
    open_data : DataFrame or None
        If a DataFrame is given, this sample out spectra (open) will be fixed for each generated sample. Must be given in proper Syndat format.
    use_hdf5 : bool
        If True, generated data will be stored in an hdf5 case file. If False, generated data will be stored in csv's in a nested directory structure.
    overwrite : bool
        Option to overwrite existing syndat data.
    """

    # check inputs
    if vary_Erange is not None:
        if not np.all([key in vary_Erange for key in ['fullrange', 'maxres','prob']]):
            raise ValueError("User supplied a vary_Erange input without proper dict_keys. Requires keys: ['fullrange', 'maxres','prob'].")

    samples_not_being_generated = [] 

    # use hdf5 file to store data for this case
    if use_hdf5:
        # check for exiting test case file
        check_case_file(case_file)
        # loop over given number of samples
        for i in range(min(dataset_range), max(dataset_range)):
            h5f = h5py.File(case_file, "a")
            sample_group = f'sample_{i}'
            if sample_group in h5f:
                if ('exp_pw' in h5f[sample_group]) and ('theo_pw' in h5f[sample_group]) and ('theo_par' in h5f[sample_group]):
                    if overwrite:
                        h5f.close()
                        sample_and_write_syndat(case_file, i, particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange, use_hdf5)
                    else:
                        h5f.close()
                        samples_not_being_generated.append(i)
                else:
                    h5f.close()
                    sample_and_write_syndat(case_file, i, particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange, use_hdf5)
            else:
                h5f.close()
                sample_and_write_syndat(case_file, i, particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange, use_hdf5)

    # use nested directory structure and csv's to store data
    else:
        # check for case directory
        check_case_directory(case_file)
        for i in range(min(dataset_range), max(dataset_range)):
            # check for sample directory
            sample_directory = os.path.join(case_file,f'sample_{i}')
            check_sample_directory(sample_directory)
            # check for existing syndat in sample_directory
            syndat_pw = os.path.join(sample_directory, 'exp_pw.csv')
            syndat_par = os.path.join(sample_directory, 'theo_par.csv')
            if os.path.isfile(syndat_pw) and os.path.isfile(syndat_par):
                if overwrite:
                    sample_and_write_syndat(case_file, i, particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange, use_hdf5)
                else:
                    samples_not_being_generated.append(i)
            else:
                sample_and_write_syndat(case_file, i, particle_pair, experiment, solver, open_data, fixed_resonance_ladder, vary_Erange, use_hdf5)



    return samples_not_being_generated





#  =============----------------------===========================


def wrapped_sammy_file_creator(number_of_realizations, case_directory, Estruct, \
                               I, i, l_wave_max,  
                               RRR_Erange, 
                               Davg, Gavg, 
                               Gavg_swave, 
                               print_out,
                                   save_csv):
    
    estruct_created = 0; inputs_created = 0; par_created = 0
    
    for irealize in range(1,number_of_realizations+1):
        
        realization_dir = os.path.join(case_directory, f'realization_{irealize}/')
        
        if os.path.isdir(realization_dir):
# in here I could look for existing sammy files and have an option to overwrite or keep
            _ = 0
        else:
            os.mkdir(realization_dir)
            
    #     sample resparms
        Jn_ladders, Jp_ladders = syndat.spin_groups.sample_all_Jpi(I, i, l_wave_max,  
                            RRR_Erange, 
                            Davg, Gavg, 
                            Gavg_swave, 
                            print_out,
                            save_csv, 
                            realization_dir)
    
    #   create necessary sammy files
        syndat.sammy_interface.create_sammyinp(os.path.join(realization_dir,'sammy.inp')); inputs_created+=1
        syndat.sammy_interface.create_sammypar(Jn_ladders, Jp_ladders,os.path.join(realization_dir,'sammy.par')); par_created+=1
    #   could maybe sample a paremter for energy structure, i.e. detector deadtime
        syndat.sammy_interface.write_estruct_file(Estruct, os.path.join(realization_dir,'estruct')); estruct_created+=1
    
    report_string = f'Report for wrapped sammy file creator:\n\
{estruct_created} Energy structure files created\n\
{inputs_created} sammy.inp files created\n\
{par_created} sammy.par files created'
                    
    print();print(report_string); print()
                    
    return report_string




def run_sammy_and_wait(case_directory, case_basename, number_of_cases):
        
    # delete qsub_icase.sh.* files - these files indicate that qsub job has completed
    for isample in range(1,number_of_cases+1):
        wildcard_path = os.path.join(case_directory, case_basename, f'{case_basename}_smpl_{isample}/qsub_{isample}.sh.*')
        os.system(f'rm {wildcard_path}')
        
    # run sammy with bayes for all files created
    irunsammy = 0
    for isample in range(1,number_of_cases+1):
        directory = os.path.join(case_directory, case_basename,f'{case_basename}_smpl_{isample}')
        os.system("ssh -t necluster.ne.utk.edu 'cd "+directory+f" ; qsub qsub_{isample}.sh'")
        irunsammy += 1
        
    # wait on all cases to complete running - looking for qsub_icase.sh.o file
    running_sammy = True
    print(); print('Waiting for sammy to run'); print()
    while running_sammy:
        case_run_bool = []
        for isample in range(1,number_of_cases+1):
            directory = os.path.join(case_directory, case_basename, f'{case_basename}_smpl_{isample}')
            
            idone_file = 0
            for file in os.listdir(directory):
                if file.startswith(f'qsub_{isample}.sh.o'):
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
        isamples_still_running = case_run_bool.count(True)
        print(f'Waiting on {isamples_still_running} to complete') #!!! this could be done better - only prints this when all are complete for some reason
        
    return irunsammy


def copy_syndat(case_directory,case_basename,first_case,last_case):
    if os.path.isdir(os.path.join(case_directory, case_basename,'synthetic_data')):
        _ = 0
    else:
        os.mkdir(os.path.join(case_directory, case_basename,'synthetic_data'))
    run_cases = range(1,last_case+1); icopy = 0
    for i in run_cases:
        shutil.copy(os.path.join(case_directory,case_basename,case_basename+f'_smpl_{i}',f'syndat_{i}'), os.path.join(case_directory,case_basename,'synthetic_data'))
        #os.system("scp nwalton1@necluster.ne.utk.edu:/home/nwalton1/my_sammy/slbw_testing_noexp/slbw_1L_noexp_case1/syndat_{i} /Users/noahwalton/research_local/resonance_fitting/synthetic_data")
        icopy += 1
        # ssh -t necluster.ne.utk.edu 'cd /home/nwalton1/my_sammy/slbw_testing/slbw_fitting_case1/ ; /home/nwalton1/my_sammy/SAMMY/sammy/build/install/bin/sammy < slbw_fitting_case1.sh'
    print(); print(f'copied {icopy} synthetic data files'); print()



def write_qsub_shell_script(isample, sample_directory):
    with open(os.path.join(sample_directory,f'qsub_{isample}.sh'), 'w') as f:
        f.write("""#!/bin/bash

#PBS -V
#PBS -l nodes=1:ppn=1
#PBS -q fill

cd ${PBS_O_WORKDIR}

/home/nwalton1/my_sammy/SAMMY/sammy/build/install/bin/sammy < piped_sammy_commands.sh""")



def create_sammy_runfiles(case_basename, samples, energy, ladder_sample_function, inp_template_file, run_sammy,
                            run_directory=os.getcwd()):
    """
    Creates directories and SAMMY runfiles for a number of sample data cases.

    This function will create a directory and all files necessary to run SAMMY for each sample. 
    Currently this funciton is setup to generate theoretical cross sections from resonance parameters
    that can then be put through the syndat methodology to generate experimental noise.

    The directory strucure is:
    - cwd
        - case_basename
            - case_basename_smpl_#
                - sammy runfiles for sample #

    A function for gennerating a resonance ladder must be supplied allowing the user to generate very problem specific resonance ladders.

    Parameters
    ----------
    case_basename : string
        Name of the main directory for which this set of synthetic data will live. This folder is created within the directory that this script is run from.
    samples : int
        Number of sample cases
    energy : array-like
        Energy grid for the calculation
    ladder_sample_function : function
        Function that when called samples a resonance ladder and outputs (dataframe, samtools array).
    inp_template_file : string
        Full path to the template sammy.inp file
    run_sammy : bool
        Boolean option to run sammy or not.
    """
    if os.path.isdir(os.path.join(run_directory, f'{case_basename}')):
        pass
    else:
        os.mkdir(os.path.join(run_directory, f'{case_basename}'))

    for i in range(samples):

        sample_df, sample_array = ladder_sample_function()

        sample_name = case_basename + '_smpl_' + str(i)
        sample_directory = os.path.join(run_directory, f'{case_basename}/{sample_name}')
        
        if os.path.isdir(sample_directory):
            pass
        else:
            os.mkdir(sample_directory)
        
        sammy_inp_filename = 'sammy_syndat.inp'
        sammy_par_filename = 'sammy_syndat.par'
        estruct_filename = 'estruct'
        piped_commands_filename = 'piped_sammy_commands.sh'
        
        # write necessary sammy runfiles
        syndat.sammy_interface.write_estruct_file(energy, os.path.join(sample_directory,estruct_filename))
        syndat.sammy_interface.create_sammyinp(filename=os.path.join(sample_directory,sammy_inp_filename), template=inp_template_file)
        syndat.sammy_interface.samtools_fmtpar(sample_array, os.path.join(sample_directory,sammy_par_filename))
        
        # write qsub shell script and piped sammy input shell script
        write_qsub_shell_script(i, sample_directory)
        with open(os.path.join(sample_directory, piped_commands_filename) , 'w') as pipefile:
            line1 = sammy_inp_filename
            line2 = sammy_par_filename
            line3 = estruct_filename
            pipefile.write(line1 + '\n' + line2 + '\n' + line3 + '\n\n')

    if run_sammy:
        print();print('going to run sammy to create synthetic data'); print()
        irunsammy = syndat.MMDA.run_sammy_and_wait(run_directory, case_basename, samples)
        syndat.MMDA.copy_syndat(run_directory,case_basename,1,samples)
    else:
        irunsammy = 0






