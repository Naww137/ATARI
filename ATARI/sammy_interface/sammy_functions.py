

import numpy as np
import os
import shutil
from pathlib import Path
from ATARI.theory import scattering_params
import pandas as pd
import subprocess

# module_dirname = os.path.dirname(__file__)

# =============================================================================
# 
# =============================================================================
def readlst(filepath):
    """
    Reads a sammy .lst or .dat file.

    _extended_summary_

    Parameters
    ----------
    filepath : str
        Full path to the .lst or .dat file.

    Returns
    -------
    DataFrame
        DataFrame with headers.
    """
    df = pd.read_csv(filepath, delim_whitespace=True, names=['E','exp_dat','exp_dat_unc','theo_xs','theo_xs_bayes','exp_trans','exp_trans_unc','theo_trans', 'theo_trans_bayes'])
    return df


# =============================================================================
# 
# =============================================================================
def samtools_fmtpar(a, filename, template):
    
    # print("WARNING: check parameter file created - formatting in sammy_interface.samtools_fmtpar could be more robust")
    
    with open(template, 'r') as f:
        template_lines = f.readlines()
    f.close()
    
    with open(filename,'w+') as f:
        for line in template_lines:
            
            if line.startswith("%%%ResParms%%%"):
                for row in a:
                    # for some reason, the first format string has had problems with the width definition,
                    #if using a different sized energy range (digits before decimal) this may become an issue
                    f.write(f'{row[0]:0<11.4f} {row[1]:0<10f} {row[2]:0<10f} {row[3]:0<10f} {row[4]:0<10f} ')
                    f.write(f'{row[5]:1g} {row[6]:1g} {row[7]:1g} {row[8]:1g} {row[9]:1g} {row[10]:1g}\n')
            else:        
                f.write(line)
        f.close()
        
    return

def write_sampar(df, pair, vary_parm, filename, 
                                    template = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'templates', 'sammy_template_RM_only.par'))):
                                # template = os.path.join(os.path.dirname(module_dirname), "templates/sammy_template_RM_only.par") ):
    """
    Writes a formatted sammy.par file.

    This function writes a formatted sammy parameter file from a DataFrame containing resonance parameters.
    The DataFrame may have

    Parameters
    ----------
    df : DataFrame
        _description_
    pair : object
        The pair object is of the particle_pair class in Syndat describing the incident and target particles.
    vary_parm : Bool
        Option to set vary parameters (0 or 1) in sammy.par file.
    filename : str
        Filepath/name of sammy.par file being created.
    template : str, optional
        Filepath to template file for sammy.par. Included because of the different options for input to sammy, by default os.path.realpath("../templates/sammy_template_RM_only.par")
    """

    def gn2G(row):
        S, P, phi, k = scattering_params.FofE_recursive([row.E], pair.ac, pair.M, pair.m, row.lwave)
        Gnx = 2*np.sum(P)*row.gnx2
        return Gnx.item()

    if df.empty:
        samtools_array = []
    else:
        
        if "Gnx" not in df:
            df['Gnx'] = df.apply(lambda row: gn2G(row), axis=1)
        else:
            pass
        
        # force to 0.001 if Gnx == 0
        df.loc[df['Gnx']==0.0, 'Gnx'] = 0.0001

        par_array = np.array([df.E, df.Gg, df.Gnx]).T
        zero_neutron_widths = 5-(len(par_array[0]))
        zero_neutron_array = np.zeros([len(par_array),zero_neutron_widths])
        par_array = np.insert(par_array, [5-zero_neutron_widths], zero_neutron_array, axis=1)

        #vary_parm = True
        if vary_parm:
            binary_array = np.hstack((np.ones([len(par_array),5-zero_neutron_widths]), np.zeros([len(par_array),zero_neutron_widths])))
        else:
            binary_array = np.zeros([len(par_array),5])

        samtools_array = np.insert(par_array, [5], binary_array, axis=1)
        if np.any([each is None for each in df.J_ID]):
            raise ValueError("NoneType was passed as J_ID in the resonance ladder")
        j_array = np.array([df.J_ID]).T
        samtools_array = np.hstack((samtools_array, j_array))

    samtools_fmtpar(samtools_array, filename, template)
    
    return
        
      
# =============================================================================
# 
# =============================================================================
def write_estruct_file(Energies, filename):
    # print("WARNING: if 'twenty' is not specified in sammy.inp, the data file format will change.\nSee 'sammy_interface.write_estruct_file'")
    if np.all([isinstance(E,float) for E in Energies]):
        pass
    else:
        Energies = np.array([float(E) for E in Energies])
    with open(filename,'w') as f:
        for ept in Energies:
            f.write(f'{ept:0<19} {1.0:<19} {1.0:0<7}\n')
        f.close()
    return


# =============================================================================
# 
# =============================================================================
def write_samdat(exp_pw, exp_cov, filename):
    """
    Writes a formatted sammy.dat file.

    These DataFrame keys required correspond to the "trans" DataFrame generated by Syndat.
    This is an attribute of the experiment object once the experiment is "run".
    The uncertainty on the experimental data point must be absolute for this formatting as it corresponds to the
    sammy format with "use twenty significant digits" or "twenty" in the input file. If this flag is not in the
    sammy input file, the uncertainty should not be absolute and the formatting (space between variables) changes.

    Parameters
    ----------
    transmission_data : DataFrame
        DataFrame containing experimental transmission data, requires columns ["E", "exp_trans", and "exp_trans_unc"].
    filename : str
        Filepath/name for the sammy.dat file being created.

    Raises
    ------
    ValueError
        Energy column not in DataFrame.
    ValueError
        Experimental transmission column not in DataFrame.
    ValueError
        Experimental transmission uncertainty column not in DataFrame.
    """
    # print("WARNING: if 'twenty' is not specified in sammy.inp, the data file format will change.\nSee 'sammy_interface.write_estruct_file'")
    exp_pw['exp_trans_unc'] = np.sqrt(np.diag(exp_cov))
    iterable = exp_pw.sort_values('E', axis=0, ascending=True).to_numpy(copy=True)
    cols = exp_pw.columns
    if 'E' not in cols:
        raise ValueError("transmission data passed to 'write_expdat_file' does not have the column 'E'")
    if 'exp_trans' not in cols:
        raise ValueError("transmission data passed to 'write_expdat_file' does not have the column 'exp'")

    iE = cols.get_loc('E')
    iexp = cols.get_loc('exp_trans')
    idT = cols.get_loc('exp_trans_unc')
    

    with open(filename,'w') as f:
        for each in iterable:
            f.write(f'{each[iE]:0<19f} {each[iexp]:0<19f} {each[idT]:0<19f}\n')
        f.close()



# =============================================================================
#         
# =============================================================================
def create_samtools_array_from_J(Jn_ladders, Jp_ladders):
    
    J = Jn_ladders + Jp_ladders
    samtools_array = np.empty((0,11))
    for ij, j_df in enumerate(J):
        
        j_array = np.array(j_df)
    
        levels = j_array.shape[0]
        zero_neutron_widths = 5-j_array.shape[1] # accepts 3 neutron widths
        
        # zeros for additional neutron widths plus zeros for all binary "vary parameter options
        j_inp_array = np.concatenate( [j_array, np.zeros((levels, zero_neutron_widths+5)), np.full((levels,1),ij+1)] , axis=1)
        
        samtools_array  = np.concatenate([samtools_array, j_inp_array], axis=0)
    
    
    return samtools_array





# =============================================================================
# Could update this to do more to the input file, i.e. input energy range
# =============================================================================
def create_sammyinp(filename='sammy.inp', \
                    template=os.path.join(Path(os.path.dirname(__file__)).parents[0],'templates/sammy_template.inp') ):
    
    with open(template, 'r') as f:
        template_lines = f.readlines()
    f.close()
    
    with open(filename,'w+') as f:
        for line in template_lines:
            f.write(line)
        f.close()
        
    return
    
# =============================================================================
# 
# =============================================================================
def read_sammy_par(filename, calculate_average):
    """
    Reads sammy.par file and calculates average parameters.

    _extended_summary_

    Parameters
    ----------
    filename : _type_
        _description_
    calculate_average : bool
        Whether or not to calculate average parameters.

    Returns
    -------
    DataFrame
        Contains the average parameters for each spin group
    DataFrame
        Contains all resonance parameters for all spin groups
    """

    energies = []; spin_group = []; nwidth = []; gwidth = []
    with open(filename,'r') as f:   
        readlines = f.readlines()
        in_res_dat = True
        for line in readlines:   

            try:
                float(line.split()[0]) # if first line starts with a float, parameter file starts directly with resonance data
            except:
                in_res_dat = False # if that is not the case, parameter file has some keyword input before line "RESONANCES" then resonance parameters

            if line.startswith(' '):
                in_res_dat = False  
            if in_res_dat:
                if line.startswith('-'):
                    continue #ignore negative resonances
                else:
                    splitline = line.split()
                    energies.append(float(splitline[0]))
                    gwidth.append(float(splitline[1]))
                    nwidth.append(float(splitline[2]))
                    spin_group.append(float(splitline[-1]))
            if line.startswith('RESONANCE'):
                in_res_dat = True

                
                
    Gg = np.array(gwidth); Gn = np.array(nwidth); E = np.array(energies); jspin = np.array(spin_group)
    df = pd.DataFrame([E, Gg, Gn, jspin], index=['E','Gg','Gn','jspin']); df = df.transpose()
    
    if calculate_average:
        #avg_widths = df.groupby('jspin', as_index=False)['Gg','Gn'].mean() 
        gb = df.groupby('jspin')    
        list_of_dfs=[gb.get_group(x) for x in gb.groups]
        
        avg_df = pd.DataFrame(index=df['jspin'].unique(),columns=['dE','Gg','Gn'])

        for ij, jdf in enumerate(list_of_dfs):
            avg_df['dE'][ij+1]=jdf['E'].diff().mean()
            avg_df['Gg'][ij+1]=jdf['Gg'].mean()
            avg_df['Gn'][ij+1]=jdf['Gn'].mean()
    else:
        avg_df = ''

    return avg_df, df
    


# =============================================================================
# 
# ============================================================================= 
def copy_template_to_runDIR(exp, file, target_dir):
    if os.path.splitext(file)[0] != 'sammy':
        file_out = ''.join(('sammy', os.path.splitext(file)[1]))
    else:
        file_out = file
    shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sammy_templates', f'{exp}', file), 
                os.path.join(target_dir,file_out))
    return

def write_saminp(model, particle_pair, reaction, bayes, filepath):
    ac = particle_pair.ac*10
    with open(filepath,'r') as f:
        old_lines = f.readlines()
    with open(filepath,'w') as f:
        for line in old_lines:
            if line.startswith("%%%formalism%%%"):
                f.write(f'{model}\n')
            elif line.startswith('%%%scattering_radius%%%'):
                f.write(f'  {ac: <8}  0.067166                       0.00000          \n')
            elif line.startswith('%%%reaction%%%'):
                f.write(f'{reaction}\n')
            elif line.startswith('%%%bayes%%%'):
                if bayes:
                    f.write('SOLVE BAYES EQUATIONS\n')
                elif not bayes:
                    f.write('DO NOT SOLVE BAYES EQUATIONS')
            else:
                f.write(line)
           





def make_runDIR(sammy_runDIR):
    if os.path.isdir(sammy_runDIR):
        pass
    else:
        os.mkdir(sammy_runDIR)

def fill_runDIR_with_templates(sammy_runDIR, one_spingroup, experimental_corrections):
    # fill temporary sammy_runDIR with runtime appropriate template files
    if one_spingroup:
        copy_template_to_runDIR(experimental_corrections, 'sammy_1spin.inp', sammy_runDIR)
    else:
        copy_template_to_runDIR(experimental_corrections, 'sammy.inp', sammy_runDIR)
    copy_template_to_runDIR(experimental_corrections, 'sammy.par', sammy_runDIR)


from ATARI.sammy_interface.sammy_classes import SammyInputData, SammyRunTimeOptions


def update_template_files(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):

    # write experimental data if you have it, else write using the estructure
    if sammy_INP.experimental_data is not None:
        write_samdat(sammy_INP.experimental_data, sammy_INP.experimental_cov, os.path.join(sammy_RTO.sammy_runDIR,'sammy.dat'))
    elif sammy_INP.energy_grid is not None:
        write_estruct_file(sammy_INP.energy_grid, os.path.join(sammy_RTO.sammy_runDIR,'estruct'))
    else:
        raise ValueError("Please provide either experimental data or an energy grid in SammyInputData")

    # edit copied runtime template files
    write_saminp(sammy_RTO.model, sammy_INP.particle_pair, sammy_RTO.reaction, sammy_RTO.solve_bayes, os.path.join(sammy_RTO.sammy_runDIR, 'sammy.inp'))
    write_sampar(sammy_INP.resonance_ladder, sammy_INP.particle_pair, sammy_RTO.solve_bayes, os.path.join(sammy_RTO.sammy_runDIR,"sammy.par"))


def write_shell_script(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):
    with open(os.path.join(sammy_RTO.sammy_runDIR, 'pipe.sh'), 'w') as f:
        f.write('sammy.inp\nsammy.par\nsammy.dat')
        if sammy_RTO.energy_window is None:
            f.write('\n')
        elif sammy_INP.experimental_data is not None:
            iter = np.arange(np.floor(min(sammy_INP.experimental_data.E)),np.ceil(max(sammy_INP.experimental_data.E))+sammy_RTO.energy_window,sammy_RTO.energy_window)
            if len(iter) >= 50:
                raise ValueError("To many energy windows supplied, please solve in less sections")
            string = ''
            for ie in range(len(iter)-1):
                string += f'{int(iter[ie])}. {int(iter[ie+1])}.\n'
            f.write(f' {string}')
        else:
            raise ValueError("An energy window input was provided but no experimental data.")


def run_sammy(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):

    # setup sammy runtime files
    make_runDIR(sammy_RTO.sammy_runDIR)
    fill_runDIR_with_templates(sammy_RTO.sammy_runDIR, sammy_RTO.one_spingroup, sammy_RTO.experimental_corrections)
    update_template_files(sammy_INP, sammy_RTO)
    write_shell_script(sammy_INP, sammy_RTO)

    # run sammy and wait for completion with subprocess
    runsammy_process = subprocess.run(
                                    [f"{sammy_RTO.shell}", "-c", f"{sammy_RTO.path_to_SAMMY_exe}<pipe.sh"], 
                                    cwd=os.path.realpath(sammy_RTO.sammy_runDIR),
                                    capture_output=True
                                    )
    # if len(runsammy_process.stderr) > 0:
    #     raise ValueError(f'SAMMY did not run correctly\n\nSAMMY error given was: {runsammy_process.stderr}')

    # read output  and delete sammy_runDIR
    lst_df = readlst(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.LST'))
    if sammy_RTO.solve_bayes:
        par_df = pd.read_csv(os.path.join(sammy_RTO.sammy_runDIR, 'sammy.par'), skipfooter=2, delim_whitespace=True, usecols=[0,1,2,6], names=['E', 'Gg', 'Gnx','J_ID'], engine='python')
    else:
        par_df = sammy_INP.resonance_ladder
    if not sammy_RTO.keep_runDIR:
        shutil.rmtree(sammy_RTO.sammy_runDIR)

    return lst_df, par_df
