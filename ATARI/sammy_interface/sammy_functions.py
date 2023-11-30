

import numpy as np
import os
import shutil
from pathlib import Path
# from ATARI.theory import scattering_params
import pandas as pd
import subprocess
from copy import copy

# from ATARI.utils.atario import fill_resonance_ladder
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.utils.stats import chi2_val

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
def format_float(value, width):
    # Convert the value to a string with the desired precision
    formatted_value = f'{abs(value):0<12f}'  # Adjust the precision as needed

    # Check if the value is negative
    if value < 0:
        # If negative, add a negative sign to the formatted value
        formatted_value = '-' + formatted_value
    else:
        # If positive, add a space to the left of the formatted value
        formatted_value = ' ' + formatted_value

    # Check if the formatted value exceeds the desired width
    if len(formatted_value) > width:
        # If the value is too large, truncate it to fit the width
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

def write_saminp(filepath, 
                model,
                experiment,
                rto,
                alphanumeric = []):
    
    # ac = sammy_INP.particle_pair.ac*10  
    broadening = True
    
    if rto.bayes:
        bayes_cmd = "SOLVE BAYES EQUATIONS"
    else:
        bayes_cmd = "DO NOT SOLVE BAYES EQUATIONS"
    alphanumeric = [model.formalism, bayes_cmd] + alphanumeric

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
                f.write(f"{model.isotope: <9} {model.M: <9} {float(min(experiment.energy_range)): <9} {float(max(experiment.energy_range)): <9}      {rto.options['iterations']: <5} \n")


            elif line.startswith('%%%card5/6%%%'):
                if broadening:
                    f.write(f'  {float(experiment.parameters["temp"][0]):<8}  {float(experiment.parameters["FP"][0]):<8}  {float(experiment.parameters["FP"][1]):<8}        \n')
                else:
                    pass

            elif line.startswith('%%%card7%%%'):
                f.write(f'  {float(model.ac):<8}  {float(experiment.parameters["n"][0]):<8}                       0.00000          \n')

            elif line.startswith('%%%card8%%%'):
                f.write(f'{experiment.reaction}\n')

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
# Workflow
# ################################################ ###############################################


def write_shell_script(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions, use_RPCM=False):
    with open(os.path.join(sammy_RTO.sammy_runDIR, 'pipe.sh'), 'w') as f:
        f.write('sammy.inp\nSAMMY.PAR\nsammy.dat\n')

        if sammy_RTO.energy_window is None:
            if use_RPCM:
                f.write("SAMMY.COV")
            else:
                pass
        
        # energy windowed solves
        elif sammy_INP.experimental_data is not None:
            iter = np.arange(np.floor(np.min(sammy_INP.experimental_data.E)),np.ceil(np.max(sammy_INP.experimental_data.E))+sammy_RTO.energy_window,sammy_RTO.energy_window)
            if len(iter) >= 50:
                raise ValueError("To many energy windows supplied, please solve in less sections")
            string = ''
            for ie in range(len(iter)-1):
                string += f'{int(iter[ie])}. {int(iter[ie+1])}.\n'
            f.write(f' {string}')

        else:
            raise ValueError("An energy window input was provided but no experimental data.")




def runsammy_shellpipe(sammy_RTO: SammyRunTimeOptions):
    # run sammy and wait for completion with subprocess
    runsammy_process = subprocess.run(
                                    [f"sh", "-c", f"{sammy_RTO.path_to_SAMMY_exe}<pipe.sh"], 
                                    cwd=os.path.realpath(sammy_RTO.sammy_runDIR),
                                    capture_output=True
                                    )
    if len(runsammy_process.stderr) > 0:
        print(f'SAMMY gave the following warning or error: {runsammy_process.stderr}')

    return




# ################################################ ###############################################
# Read endf files
# ################################################ ###############################################


def replace_matnum(filepath, matnum):
    with open(filepath, 'r') as f:
        s = f.read()
        s = s.replace("%%%MAT%%%", str(matnum))
    with open(filepath, 'w') as f:
        f.write(s)

def get_endf_parameters(endf_file, matnum, sammyRTO: SammyRunTimeOptions):

    make_runDIR(sammyRTO.sammy_runDIR)
    fill_runDIR_with_templates("readendf.inp", "sammy.inp", sammyRTO.sammy_runDIR)

    shutil.copy(endf_file, os.path.join(sammyRTO.sammy_runDIR, os.path.basename(endf_file)))

    write_estruct_file([10,100,1000], os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    replace_matnum(os.path.join(sammyRTO.sammy_runDIR,'sammy.inp'), matnum)

    with open(os.path.join(sammyRTO.sammy_runDIR, "pipe.sh"), 'w') as f:
        f.write(f"sammy.inp\n{os.path.basename(endf_file)}\nsammy.dat\n\n")

    runsammy_shellpipe(sammyRTO)
    resonance_ladder = readpar(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF.PAR"))
    # could also read endf spin groups here! 

    with open(os.path.join(sammyRTO.sammy_runDIR, "SAMNDF.INP"), 'r') as f:
        s = f.read()
    with open("SAMNDF.INP", 'w') as f:
        f.write(s) 

    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)

    return resonance_ladder










# ################################################ ###############################################
# run sammy with NV or IQ scheme
# ################################################ ###############################################



def get_ECSCM(sammyRTO, sammyINP):

    # update_input_files(sammy_INP, sammy_RTO)
    fill_runDIR_with_templates(sammyINP.template, "sammy.inp", sammyRTO.sammy_runDIR)
    energy_grid = np.linspace(min(sammyINP.experimental_data.E), max(sammyINP.experimental_data.E), 498) # if more than 498 datapoints then I need a new reader!
    write_estruct_file(energy_grid, os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"), 
                 sammyINP.particle_pair, 
                 sammyINP.experiment, 
                 sammyRTO,
                 alphanumeric=["CROSS SECTION COVARIance matrix is wanted"])
    
    write_shell_script(sammyINP, sammyRTO, use_RPCM=True)
    runsammy_shellpipe(sammyRTO)

    df, cov = read_ECSCM(os.path.join(sammyRTO.sammy_runDIR, "SAMCOV.PUB"))

    return df, cov
    




def execute_sammy(sammy_RTO:SammyRunTimeOptions):
    runsammy_shellpipe(sammy_RTO)
    lst_df = readlst(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.LST'))
    par_df = readpar(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.PAR'))
    return lst_df, par_df



def delta_chi2(lst_df):
    chi2_prior = chi2_val(lst_df.theo_xs, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
    chi2_posterior = chi2_val(lst_df.theo_xs_bayes, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
    return chi2_prior - chi2_posterior


def recursive_sammy(pw_prior, par_prior, sammy_INP: SammyInputData, sammy_RTO: SammyRunTimeOptions, itter=0):

    if itter >= sammy_RTO.recursive_opt["iterations"]:
        return pw_prior, par_prior
    
    # sammy_INP.resonance_ladder = par_posterior
    write_sampar(par_prior, sammy_INP.particle_pair, sammy_INP.initial_parameter_uncertainty, os.path.join(sammy_RTO.sammy_runDIR,"SAMMY.PAR"), vary_parm=sammy_RTO.solve_bayes)
    pw_posterior, par_posterior = execute_sammy(sammy_RTO)
    
    Dchi2 = delta_chi2(pw_posterior)
    if sammy_RTO.recursive_opt["print"]:
        print(Dchi2)

    if Dchi2 <= sammy_RTO.recursive_opt["threshold"]:
        return pw_prior, par_prior
    
    return recursive_sammy(pw_posterior, par_posterior, sammy_INP, sammy_RTO, itter + 1)



def run_sammy(sammyINP: SammyInputData, sammyRTO:SammyRunTimeOptions):

    sammyINP.resonance_ladder = copy(sammyINP.resonance_ladder)

    #### setup 
    make_runDIR(sammyRTO.sammy_runDIR)

    if isinstance(sammyINP.experimental_data, pd.DataFrame):
        write_samdat(sammyINP.experimental_data, sammyINP.experimental_cov, os.path.join(sammyRTO.sammy_runDIR,'sammy.dat'))
    else:
        write_estruct_file(sammyINP.energy_grid, os.path.join(sammyRTO.sammy_runDIR,"sammy.dat"))

    write_sampar(sammyINP.resonance_ladder, sammyINP.particle_pair, sammyINP.initial_parameter_uncertainty,os.path.join(sammyRTO.sammy_runDIR, 'SAMMY.PAR'))
    fill_runDIR_with_templates(sammyINP.template, "sammy.inp", sammyRTO.sammy_runDIR)
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"sammy.inp"), sammyINP.particle_pair, sammyINP.experiment, sammyRTO)
    write_shell_script(sammyINP, sammyRTO, use_RPCM=False)

    lst_df, par_df = execute_sammy(sammyRTO)

    sammy_OUT = SammyOutputData(pw=lst_df, 
                        par=sammyINP.resonance_ladder,
                        chi2=[])#,
                        # chi2=chi2_val(lst_df.theo_xs, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc)))

    #### need to update for recursive sammy
    # if sammyRTO.recursive == True:
    #     if sammyRTO.solve_bayes == False:
    #         raise ValueError("Cannot run recursive sammy with solve bayes set to false")
    #     lst_df, par_df = recursive_sammy(lst_df, par_df, sammyINP, sammyRTO)

    if sammyRTO.bayes:
        # sammy_OUT.chi2_post = chi2_val(lst_df.theo_xs_bayes, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
        sammy_OUT.pw=lst_df
        sammy_OUT.par_post = par_df

        if sammyRTO.get_ECSCM:
            est_df, ecscm = get_ECSCM(sammyRTO, sammyINP)
            sammy_OUT.ECSCM = ecscm
            sammy_OUT.est_df = est_df

    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)

    return sammy_OUT





########################################################## ###############################################
# run sammy with the YW scheme
########################################################## ###############################################

def make_inputs_for_YW(sammyINPYW: SammyInputDataYW, sammyRTO:SammyRunTimeOptions):
    temp_RTO = copy(sammyRTO)
    #### make files for each dataset YW generation
    for exp, tem in zip(sammyINPYW.experiments, sammyINPYW.templates):  # fix this !!

        temp_RTO.bayes = True
        temp_RTO.options["bayes"] = True
        fill_runDIR_with_templates(tem, f"{exp.title}_initial.inp", temp_RTO.sammy_runDIR)
        write_saminp(os.path.join(temp_RTO.sammy_runDIR, f"{exp.title}_initial.inp"), sammyINPYW.particle_pair, exp, temp_RTO,
                                    alphanumeric=["yw"])

        fill_runDIR_with_templates(tem, f"{exp.title}_iter.inp", temp_RTO.sammy_runDIR)
        write_saminp(os.path.join(temp_RTO.sammy_runDIR, f"{exp.title}_iter.inp"), sammyINPYW.particle_pair, exp, temp_RTO,
                                    alphanumeric=["yw","Use remembered original parameter values"])

        temp_RTO.bayes = False
        temp_RTO.options["bayes"] = False
        fill_runDIR_with_templates(tem, f"{exp.title}_plot.inp", temp_RTO.sammy_runDIR)
        write_saminp(os.path.join(temp_RTO.sammy_runDIR, f"{exp.title}_plot.inp"), sammyINPYW.particle_pair, exp, temp_RTO,
                                    alphanumeric=[])
    
    ### options for least squares
    if sammyINPYW.LS:
        alphanumeric_LS_opts = ["USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Take baby steps with Least-Squares method"]
    else:
        alphanumeric_LS_opts = []

    #### make files for solving bayes reading in each YW matrix  -- # TODO: I should define a better template/exp here
    fill_runDIR_with_templates(tem, "solvebayes_initial.inp", temp_RTO.sammy_runDIR)
    temp_RTO.bayes = True
    temp_RTO.options["bayes"] = True
    write_saminp(os.path.join(temp_RTO.sammy_runDIR,"solvebayes_initial.inp"), sammyINPYW.particle_pair, exp, temp_RTO,   
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "Remember original parameter values"]+alphanumeric_LS_opts)

    fill_runDIR_with_templates(tem, "solvebayes_iter.inp" , temp_RTO.sammy_runDIR)
    write_saminp(os.path.join(temp_RTO.sammy_runDIR, "solvebayes_iter.inp"), sammyINPYW.particle_pair, exp, temp_RTO,
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "Use remembered original parameter values"]+alphanumeric_LS_opts )



def make_data_for_YW(datasets, experiments, rundir):
    if np.all([isinstance(i, pd.DataFrame) for i in datasets]):
        real = True
    else:
        if np.any([isinstance(i,pd.DataFrame) for i in datasets]):
            raise ValueError("It looks like you''ve mixed dummy energy-grid data and real data")
        real = False
    for d, exp in zip(datasets, experiments):
        if real:
            write_samdat(d, None, os.path.join(rundir,f"{exp.title}.dat"))
            write_estruct_file(d.E, os.path.join(rundir,"dummy.dat"))
        else:
            write_estruct_file(d, os.path.join(rundir,f"{exp.title}.dat"))
            # write_estruct_file(d, os.path.join(rundir,"dummy.dat"))


def make_YWY0_bash(dataset_titles, sammyexe, rundir):
    cov = ''
    par = 'results/step$1.par'
    inp_ext = 'initial'
    with open(os.path.join(rundir, "YWY0.sh") , 'w') as f:
        ### Copy final iteration result to step + 1 result
        # f.write(f"\n\n\n\n############## Copy Iteration Result ###########\nplus_one=$(( $1 + 1 ))\nhead -$(($(wc -l < iterate/bayes_iter{iterations}.par) - 1)) iterate/bayes_iter{iterations}.par > results/step$plus_one.par\n\nrm REMORI.PAR\n")
        for ds in dataset_titles:
            title = f"{ds}_iter0"
            f.write(f"##################################\n# Generate YW for {ds}\n")
            f.write(f"{sammyexe}<<EOF\n{ds}_{inp_ext}.inp\n{par}\n{ds}.dat\n{cov}\n\nEOF\n")
            f.write(f"""mv -f SAMMY.LPT "iterate/{title}.lpt" \nmv -f SAMMY.ODF "iterate/{title}.odf" \nmv -f SAMMY.LST "iterate/{title}.lst" \nmv -f SAMMY.YWY "iterate/{title}.ywy" \n""")    
        f.write("################# read chi2 #######################\n#\n")
        for ds in dataset_titles:
            f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" iterate/{ds}_iter0.lpt)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $chi2_string_{ds}")
        f.write(""""\n""")
        
    with open(os.path.join(rundir, "BAY0.sh") , 'w') as f:
        dataset_inserts = "\n".join([f"iterate/{ds}_iter0.ywy" for ds in dataset_titles])
        title = f"bayes_iter1"
        f.write(f"#######################################################\n# Run Bayes for {title}\n#\n#######################################################\n")
        f.write(f"{sammyexe}<<eod\nsolvebayes_{inp_ext}.inp\n{par}\ndummy.dat\n{dataset_inserts}\n\n{cov}\n\neod\n")
        f.write(f"""mv -f SAMMY.LPT iterate/{title}.lpt \nmv -f SAMMY.PAR iterate/{title}.par \nmv -f SAMMY.COV iterate/{title}.cov \nrm -f SAM*\n""")


def make_YWYiter_bash(dataset_titles, sammyexe, rundir):
    cov = f"iterate/bayes_iter$1.cov"
    par = f"iterate/bayes_iter$1.par"
    inp_ext = 'iter'
    with open(os.path.join(rundir,"YWYiter.sh"), 'w') as f:
        for ds in dataset_titles:
            title = f"{ds}_iter$1"
            f.write(f"##################################\n# Generate YW for {ds}\n")
            f.write(f"{sammyexe}<<EOF\n{ds}_{inp_ext}.inp\n{par}\n{ds}.dat\n{cov}\n\nEOF\n")
            f.write(f"""mv -f SAMMY.LPT "iterate/{title}.lpt" \nmv -f SAMMY.ODF "iterate/{title}.odf" \nmv -f SAMMY.LST "iterate/{title}.lst" \nmv -f SAMMY.YWY "iterate/{title}.ywy" \n""")    
    with open(os.path.join(rundir,"BAYiter.sh"), 'w') as f:
        dataset_inserts = "\n".join([f"iterate/{ds}_iter$1.ywy" for ds in dataset_titles])
        title = f"bayes_iter$plus_one"
        f.write(f"#######################################################\n# Run Bayes for {title}\n#\n#######################################################\n")
        f.write(f"{sammyexe}<<eod\nsolvebayes_{inp_ext}.inp\n{par}\ndummy.dat\n{dataset_inserts}\n\n{cov}\n\neod\n")
        f.write("plus_one=$(( $1 + 1 ))\n")
        f.write(f"""mv -f SAMMY.LPT iterate/{title}.lpt \nmv -f SAMMY.PAR iterate/{title}.par \nmv -f SAMMY.COV iterate/{title}.cov \nrm -f SAM*\n""")


def make_final_plot_bash(dataset_titles, sammyexe, rundir):
    with open(os.path.join(rundir, "plot.sh") , 'w') as f:
        for ds in dataset_titles:
            f.write(f"##################################\n# Plot for {ds}\n")
            f.write(f"{sammyexe}<<EOF\n{ds}_plot.inp\nresults/step$1.par\n{ds}.dat\n\nEOF\n")
            f.write(f"""mv -f SAMMY.LPT "results/{ds}.lpt" \nmv -f SAMMY.ODF "results/{ds}.odf" \nmv -f SAMMY.LST "results/{ds}.lst" \n\n""")    
        f.write("################# read chi2 #######################\n#\n")
        for ds in dataset_titles:
            f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/{ds}.lpt | tail -n 1)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
        f.write("""\necho "$1""")
        for ds in dataset_titles:
            f.write(f" $chi2_string_{ds}")
        f.write(""""\n""")
    

def setup_YW_scheme(sammyRTO, sammyINPyw): 

    try:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    except:
        pass

    os.mkdir(sammyRTO.sammy_runDIR)
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "results"))
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "iterate"))

    make_data_for_YW(sammyINPyw.datasets, sammyINPyw.experiments, sammyRTO.sammy_runDIR)
    write_sampar(sammyINPyw.resonance_ladder, sammyINPyw.particle_pair, sammyINPyw.initial_parameter_uncertainty, os.path.join(sammyRTO.sammy_runDIR, "results/step0.par"))

    make_inputs_for_YW(sammyINPyw, sammyRTO)
    dataset_titles = [exp.title for exp in sammyINPyw.experiments]
    make_YWY0_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR)
    make_YWYiter_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR)
    make_final_plot_bash(dataset_titles, sammyRTO.path_to_SAMMY_exe, sammyRTO.sammy_runDIR)
    



def iterate_for_nonlin_and_update_step_par(iterations, step, rundir):
    runsammy_bay0 = subprocess.run(
                            ["sh", "-c", f"./BAY0.sh {step}"], cwd=os.path.realpath(rundir),
                            capture_output=True
                            )

    for i in range(1, iterations+1):
        runsammy_ywy0 = subprocess.run(
                                    ["sh", "-c", f"./YWYiter.sh {i}"], cwd=os.path.realpath(rundir),
                                    capture_output=True
                                    )

        runsammy_bay0 = subprocess.run(
                                    ["sh", "-c", f"./BAYiter.sh {i}"], cwd=os.path.realpath(rundir),
                                    capture_output=True
                                    )

    # Move par file from final iteration 
    out = subprocess.run(
        ["sh", "-c", 
        f"""head -$(($(wc -l < iterate/bayes_iter{iterations}.par) - 1)) iterate/bayes_iter{iterations}.par > results/step{step+1}.par"""],
        cwd=os.path.realpath(rundir), capture_output=True)
    

def run_YWY0_and_get_chi2(rundir, step):
    runsammy_ywy0 = subprocess.run(
                                ["sh", "-c", f"./YWY0.sh {step}"], 
                                cwd=os.path.realpath(rundir),
                                capture_output=True, text=True
                                )
    i_chi2s = [float(s) for s in runsammy_ywy0.stdout.split('\n')[-2].split()]
    i=i_chi2s[0]; chi2s=i_chi2s[1:] 
    return i, [c for c in chi2s]+[np.sum(chi2s)]


def update_fudge_in_parfile(rundir, step, fudge):
    out = subprocess.run(
        ["sh", "-c", 
        f"""head -$(($(wc -l < results/step{step}.par) - 1)) results/step{step}.par > results/temp;
echo "{np.round(fudge,11)}" >> "results/temp"
mv "results/temp" "results/step{step}.par" """],
        cwd=os.path.realpath(rundir), capture_output=True)





def step_until_convergence_YW(sammyRTO, sammyINPyw):
    istep = 0
    chi2_log = []
    fudge = sammyINPyw.initial_parameter_uncertainty
    rundir = os.path.realpath(sammyRTO.sammy_runDIR)
    criteria="max steps"
    if sammyRTO.Print:
        print(f"Stepping until convergence\nchi2 values\nstep fudge: {[exp.title for exp in sammyINPyw.experiments]+['sum']}")
    while istep<sammyINPyw.max_steps:
        i, chi2_list = run_YWY0_and_get_chi2(rundir, istep)
        if istep>=1:

            # Levenberg-Marquardt algorithm
            if sammyINPyw.LevMar:
                assert(sammyINPyw.LevMarV>1)

                if chi2_list[-1] < chi2_log[istep-1][-1]:
                    fudge *= sammyINPyw.LevMarV
                    fudge = min(fudge,sammyINPyw.maxF)
                else:
                    if sammyRTO.Print:
                        print(f"Repeat step {int(i)}, \tfudge: {[exp.title for exp in sammyINPyw.experiments]+['sum']}")

                    while True:
                        fudge /= sammyINPyw.LevMarV
                        fudge = max(fudge, sammyINPyw.minF)
                        update_fudge_in_parfile(rundir, istep-1, fudge)
                        iterate_for_nonlin_and_update_step_par(sammyINPyw.iterations, istep-1, rundir)
                        i, chi2_list = run_YWY0_and_get_chi2(rundir, istep)

                        if sammyRTO.Print:
                            print(f"\t\t{np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")

                        if chi2_list[-1] < chi2_log[istep-1][-1] or fudge==sammyINPyw.minF:
                            break
                        else:
                            pass
            
            # convergence check
            Dchi2 = chi2_log[istep-1][-1] - chi2_list[-1]
            if Dchi2 < sammyINPyw.step_threshold:
                if Dchi2 < 0:
                    criteria = f"Chi2 increased, taking solution {istep-1}"
                    if sammyINPyw.LevMar and fudge==sammyINPyw.minF:
                        criteria = f"Fudge below minimum value, taking solution {istep-1}"
                    if sammyRTO.Print:
                        print(f"{int(i)}    {np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")
                        print(criteria)
                    return istep-1
                else:
                    criteria = "Chi2 improvement below threshold"
                if sammyRTO.Print:
                    print(f"{int(i)}    {np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")
                    print(criteria)
                return istep
                
        
        chi2_log.append(chi2_list)
        if sammyRTO.Print:
            print(f"{int(i)}    {np.round(float(fudge),3):<5}: {list(np.round(chi2_list,4))}")
        
        update_fudge_in_parfile(rundir, istep, fudge)
        iterate_for_nonlin_and_update_step_par(sammyINPyw.iterations, istep, rundir)

        istep += 1

    # print(criteria)
    return istep-1



def plot_YW(sammyRTO, dataset_titles, i):
    out = subprocess.run(["sh", "-c", f"./plot.sh {i}"], 
                cwd=os.path.realpath(sammyRTO.sammy_runDIR), capture_output=True, text=True
                        )
    par = readpar(os.path.join(sammyRTO.sammy_runDIR,f"results/step{i}.par"))
    lsts = []
    for dt in dataset_titles:
        lsts.append(readlst(os.path.join(sammyRTO.sammy_runDIR,f"results/{dt}.lst")) )
    i_chi2s = [float(s) for s in out.stdout.split('\n')[-2].split()]
    i=i_chi2s[0]; chi2s=i_chi2s[1:] 
    return par, lsts, chi2s



def run_sammy_YW(sammyINPyw, sammyRTO):

    ## need to update functions to just pull titles and reactions from sammyINPyw.experiments
    dataset_titles = [exp.title for exp in sammyINPyw.experiments]
    # sammyINPyw.reactions = [exp.reaction for exp in sammyINPyw.experiments]

    setup_YW_scheme(sammyRTO, sammyINPyw)
    for bash in ["YWY0.sh", "YWYiter.sh", "BAY0.sh", "BAYiter.sh", "plot.sh"]:
        os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'{bash}')}")

    ### get prior
    par, lsts, chi2list = plot_YW(sammyRTO, dataset_titles, 0)
    sammy_OUT = SammyOutputData(pw=lsts, par=par, chi2=chi2list)
    
    ### run bayes
    if sammyRTO.bayes:
        ifinal = step_until_convergence_YW(sammyRTO, sammyINPyw)
        par_post, lsts_post, chi2list_post = plot_YW(sammyRTO, dataset_titles, ifinal)
        sammy_OUT.pw_post = lsts_post
        sammy_OUT.par_post = par_post
        sammy_OUT.chi2_post = chi2list_post


    if not sammyRTO.keep_runDIR:
        shutil.rmtree(sammyRTO.sammy_runDIR)

    return sammy_OUT
