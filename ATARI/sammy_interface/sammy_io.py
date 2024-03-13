

import numpy as np
import os
import shutil
from pathlib import Path
# from ATARI.theory import scattering_params
import pandas as pd
# import subprocess
from copy import copy

# from ATARI.utils.atario import fill_resonance_ladder
from ATARI.theory.scattering_params import FofE_recursive
# from ATARI.utils.stats import chi2_val

from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyOutputData, SammyInputDataYW

from typing import Optional, Union
from ATARI.sammy_interface.sammy_classes import SammyInputData, SammyRunTimeOptions


# module_dirname = os.path.dirname(__file__)



# =============================================================================
#  Readers
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
    if filepath.endswith('.LST') or filepath.endswith('.lst'):
        df = pd.read_csv(filepath, delim_whitespace=True, names=['E','exp_xs','exp_xs_unc','theo_xs','theo_xs_bayes','exp_trans','exp_trans_unc','theo_trans', 'theo_trans_bayes'])
        if df.index.equals(pd.RangeIndex(len(df))):
            pass
        else:
            df = pd.read_csv(filepath, delim_whitespace=True, names=['E','exp_xs','exp_xs_unc','theo_xs','theo_xs_bayes','exp_trans','exp_trans_unc','theo_trans', 'theo_trans_bayes', 'other'])
    else:
        df = pd.read_csv(filepath, delim_whitespace=True, names=['E','exp','exp_unc'])
    return df

def readpar(filepath):
    """
    Reads a sammy .par file.

    Parameters
    ----------
    filepath : str
        Full path to the .par file you want to read

    Returns
    -------
    DataFrame
        DataFrame with appropriately names columns
    """
    column_widths = [11, 11, 11, 11, 11, 2, 2, 2, 2, 2, 2]
    data = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:

            if line.isspace():
                break # end of resonance region

            row = []
            start = 0
            for iw, width in enumerate(column_widths):
                value = line[start:start+width].strip()
                if value == '':
                    value = None
                else:
                    # if iw == 0: # energy can be negative
                        # value = float(value)
                    # else: # widths cannots
                    try:
                        value = float(value)
                    except:
                        sign='+'
                        splitvals = value.split('-')
                        if splitvals[0] == '':
                            splitvals = splitvals[1::]
                            sign = '-'

                        if len(splitvals) == 1:
                            splitvals = splitvals[0].split('+')
                            if splitvals[0] == '':
                                splitvals = splitvals[1::]
                                sign = '+'
                            joiner = 'e+'

                        else:
                            joiner = 'e-'

                        if sign == '-':
                            value = -float(joiner.join(splitvals))
                        else:
                            value = float(joiner.join(splitvals))
                            

                row.append(value)
                start += width
            data.append(row)
    df = pd.DataFrame(data, columns=['E', 'Gg', 'Gn1', 'Gn2', 'Gn3', 'varyE', 'varyGg', 'varyGn1', 'varyGn2', 'varyGn3', 'J_ID'])
    return df.dropna(axis=1)


def read_ECSCM(file_path):
    """
    Reads a sammy generated ECSCM in as a pandas dataframe.

    Parameters
    ----------
    file_path : str
        Path to the sammy generated SAMCOV.PUB file.

    Returns
    -------
    _type_
        _description_
    """

    data = pd.read_csv(file_path, delim_whitespace=True, skiprows=3, header=None)
    df_tdte = data.iloc[:,0:3]
    df_tdte.columns = ["theo", "theo_unc", "E"]

    # assert top rows == left columns
    dftest = pd.read_csv(file_path, delim_whitespace=True, nrows=3, header=None)
    dftest=dftest.T
    dftest.columns = ["theo", "theo_unc", "E"]
    assert(np.all(dftest == df_tdte))

    dfcov = data.iloc[:, 3:]

    return df_tdte, dfcov

# =============================================================================
# Sammy Parameter File
# =============================================================================
def format_float(value, width, sep=''):
    formatted_value = f'{abs(value):0<15f}'  

    if value < 0:
        formatted_value = f'{sep}-' + formatted_value
    else:
        formatted_value = f'{sep} ' + formatted_value

    if len(formatted_value) > width:
        formatted_value = formatted_value[:width]

    return formatted_value


def fill_sammy_ladder(df, particle_pair, vary_parm=False, J_ID=None):
    df = copy(df)
    def gn2G(row):
        _, P, _, _ = FofE_recursive([row.E], particle_pair.ac, particle_pair.M, particle_pair.m, row.lwave)
        Gn = 2*np.sum(P)*row.gn2
        return Gn.item()
    
    def nonzero_ifvary(row):
        for par in ["E", "Gg", "Gn1", "Gn2", "Gn3"]:
            if row[f"vary{par}"] == 1.0 and row[par] < 1e-5:
                row[par] = 1e-5
        return row

    cols = df.columns.values
    if "Gn1" not in cols:
        if "Gn" not in cols:
            if "gn2" not in cols:
                raise ValueError("Neutron width (Gn1, Gn, gn2) not in parameter dataframe.")
            else:
                df["Gn1"] = df.apply(lambda row: gn2G(row), axis=1)
        else:
            df.rename(columns={"Gn":"Gn1"}, inplace=True)
    if "Gg" not in cols:
        if "Gt" not in cols:
            raise ValueError("Gg nor Gt in parameter dataframe")
        else:
            df['Gg'] = df['Gt'] - df['Gn1']

    for vary in ["varyE", "varyGg", "varyGn1", "varyGn2", "varyGn3"]:
        if vary not in cols:
            if vary_parm:
                df[vary] = np.ones(len(df))
            else:
                df[vary] = np.zeros(len(df))

    if "Gn2" not in cols:
        df["Gn2"] = np.zeros(len(df))
        df["varyGn2"] = np.zeros(len(df))
    if "Gn3" not in cols:
        df["Gn3"] = np.zeros(len(df))
        df["varyGn3"] = np.zeros(len(df))

    # if a parameter (width) is zero and it is varied then make it 1e-5
    df = df.apply(nonzero_ifvary, axis=1)

    # must have J_ID
    if "J_ID" not in cols:
        if J_ID is None:
            raise ValueError("J_ID not in ladder nor provided as input")
        else:
            df["J_ID"] = J_ID

    return df

def check_sampar_inputs(df):
    # check for same energy and same spin group
    return 

def write_sampar(df, pair, initial_parameter_uncertainty, filename, vary_parm=False, template=None):
                                    # template = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'templates', 'sammy_template_RM_only.par'))):
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
    initial_parameter_uncertainty : float
        Global initial parameter uncertainty for Bayes solve.
    filename : str
        Filepath/name of sammy.par file being created.
    template : str, optional
        Filepath to template file for sammy.par. Included because of the different options for input to sammy, by default None and the SAMMY parameter file will only contain resonance parameters.
    """

    if df.empty:
        par_array = []
    else:
        df = fill_sammy_ladder(df, pair, vary_parm)
        par_array = np.array(df[['E', 'Gg', 'Gn1', 'Gn2', 'Gn3', 'varyE', 'varyGg', 'varyGn1', 'varyGn2', 'varyGn3', 'J_ID']])

        if np.any([each is None for each in df.J_ID]):
            raise ValueError("NoneType was passed as J_ID in the resonance ladder")

    widths = [11, 11, 11, 11, 11, 2, 2, 2, 2, 2, 2]
    with open(filename, 'w') as file:
        for row in par_array:
            for icol, val in enumerate(row):
                column_width = widths[icol]
                formatted_value = format_float(val, column_width)
                file.write(formatted_value)
            file.write('\n')
        file.write(f'\n{initial_parameter_uncertainty}\n')
        
    
    return
        
      

# ################################################ ###############################################
# sammy data file
# ################################################ ###############################################

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
    exp_pw : DataFrame
        DataFrame containing experimental measurement data, requires columns ["E", "exp"].
    exp_cov: DataFrame or None
        DataFrame containting experimental measurement covariance.
    filename : str
        Filepath/name for the sammy.dat file being created.

    Raises
    ------
    ValueError
        Energy column not in DataFrame.
    ValueError
        Experimental measurement column not in DataFrame.
    ValueError
        Experimental measurement uncertainty column not in DataFrame.
    """

    if 'exp' not in exp_pw:
        if 'exp_trans' in exp_pw:
            exp_pw.rename(columns={'exp_trans':'exp', 'exp_trans_unc': 'exp_unc'}, inplace=True)
        elif 'exp_dat' in exp_pw:
            exp_pw.rename(columns={'exp_dat':'exp', 'exp_dat_unc': 'exp_unc'}, inplace=True)
        else:
            ValueError("Data passed to 'write_expdat_file' does not have the column 'exp'")

    # print("WARNING: if 'twenty' is not specified in sammy.inp, the data file format will change.\nSee 'sammy_interface.write_estruct_file'")
    if 'exp_unc' not in exp_pw:
        exp_pw['exp_unc'] = np.sqrt(np.diag(exp_cov))
    
    iterable = exp_pw.sort_values('E', axis=0, ascending=True).to_numpy(copy=True)
    cols = exp_pw.columns
    if 'E' not in cols:
        raise ValueError("Data passed to 'sammy_functions.write_expdat_file' does not have the column 'E'")
    if 'exp' not in cols:
        raise ValueError("Data passed to 'sammy_functions.write_expdat_file' does not have the column 'exp'")

    iE = cols.get_loc('E')
    iexp = cols.get_loc('exp')
    idT = cols.get_loc('exp_unc')
    

    with open(filename,'w') as f:
        for each in iterable:
            f.write(f'{each[iE]:0<19f} {each[iexp]:0<19f} {each[idT]:0<19f}\n')
        f.close()


def write_idc(filepath, J, C, stat):
    with open(filepath, 'w+') as f:
        f.write(f"NUmber of data-reduction parameters = {J.shape[0]} \n\n")
        f.write(f"FREE-FORMAt partial derivatives\n")
        width = 13 #[11, 11, 11, 11, 11, 2, 2, 2, 2, 2, 2]

        for E, data in J.items():
            formatted_E = format_float(E, width)
            f.write(formatted_E)
            formatted_data_stat_unc = format_float(np.sqrt(stat.loc[E, 'var_stat']), width, sep=' ')
            f.write(formatted_data_stat_unc)
            for derivative in data:
                formatted_derivative = format_float(derivative, width, sep=' ')
                f.write(formatted_derivative)
            f.write('\n')

        f.write("\nUNCERTAINTies on data- reduction parameters\n")
        for sys_uncertainty in np.sqrt(np.diag(C)):
            formatted_uncertainty = format_float(sys_uncertainty, width, sep=' ')
            f.write(formatted_uncertainty)
        f.write("\n\n")
        
        f.write("CORRELATIOns for data-reduction parameters")
        Dinv = np.diag(1 / np.sqrt(np.diag(C))) 
        Corr = Dinv @ C @ Dinv
        for i, row in enumerate(Corr):
            for corr in row[0:i]:
                formatted_correlation = format_float(corr, width, sep=' ')
                f.write(formatted_correlation)
            f.write("\n")
            


# ################################################ ###############################################
# Sammy Input file
# ################################################ ###############################################

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

def write_saminp(filepath   :   str, 

                bayes       :   bool,
                iterations  :   int,

                formalism   :   str,
                isotope     :   str,
                M           :   float,
                ac          :   float,

                reaction    :   str,
                energy_range,
                temp        :   tuple,
                FP          :   tuple,
                n           :   tuple,

                alphanumeric:   bool    = None,
                use_IDC     :   bool    = False,
                # resolution_function = True,
                ):
    
    if alphanumeric is None:
        alphanumeric = []
    # if use_ecscm_reaction:
    #     reaction = rto.ECSCM_rxn
    # else:
    #     reaction = experimental_model.reaction
    # ac = sammy_INP.particle_pair.ac*10  
    broadening = True
    
    if bayes:
        bayes_cmd = "SOLVE BAYES EQUATIONS"
    else:
        bayes_cmd = "DO NOT SOLVE BAYES EQUATIONS"
    if use_IDC:
            alphanumeric.append("USER-SUPPLIED IMPLICIT DATA COVARIANCE MATRIX")
    
    alphanumeric = [formalism, bayes_cmd] + alphanumeric

    with open(filepath,'r') as f:
        old_lines = f.readlines()

    with open(filepath,'w') as f:
        for line in old_lines:

            if "broadening is not wa" in line.lower() or np.any([each.lower().startswith("broadening is not wa") for each in alphanumeric]):
                broadening = False

            if line.startswith("%%%alphanumeric%%%"):
                for cmd in alphanumeric:
                    f.write(f'{cmd}\n')
            
            elif line.startswith("%%%card2%%%"):
                f.write(f"{isotope: <9} {M:<9.8} {float(min(energy_range)):<9.8} {float(max(energy_range)):<9.8}      {iterations: <5} \n")


            elif line.startswith('%%%card5/6%%%'):
                if broadening:
                    f.write(f'  {float(temp[0]):<8.7}  {float(FP[0]):<8.7}  {float(FP[1]):<8.7}        \n')
                else:
                    pass

            elif line.startswith('%%%card7%%%'): #ac*10 because sqrt(bn) -> fm for sammy 
                f.write(f'  {float(ac):<8.7}  {float(n[0]):<8.7}                       0.00000          \n')

            elif line.startswith('%%%card8%%%'):
                f.write(f'{reaction}\n')

            else:
                f.write(line)


           

def make_runDIR(sammy_runDIR):
    if os.path.isdir(sammy_runDIR):
        pass
    else:
        os.mkdir(sammy_runDIR)

def fill_runDIR_with_templates(input_template, input_name, sammy_runDIR):

    if os.path.basename(input_template) == input_template:
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sammy_templates', input_template)
    else:
        template_path = input_template

    shutil.copy(template_path, os.path.join(sammy_runDIR, input_name))



# ################################################ ###############################################
# MISC
# ################################################ ###############################################



def remove_resolution_function_from_template(template_filepath):
    file = open(template_filepath, 'r')
    readlines = file.readlines()
    file.close()
    file = open(template_filepath, 'w')
    sg_started = False
    end = False
    for line in readlines:
        if sg_started:
            if line == '\n':
                end = True
        if line.startswith("%%%card8%%%"):
            sg_started = True
        if not end:
            file.write(line)
    file.close()


