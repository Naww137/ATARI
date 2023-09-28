

import numpy as np
import os
import shutil
from pathlib import Path
from ATARI.theory import scattering_params
import pandas as pd
import subprocess
from copy import copy
# from ATARI.utils.atario import fill_resonance_ladder
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.utils.stats import chi2_val

from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyOutputData, SammyInputDataYW


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
            for width in column_widths:
                value = line[start:start+width].strip()
                if value == '':
                    value = None
                else:
                    try:
                        value = abs(float(value))
                    except:
                        try:
                            # value = float('e-'.join(value.split('-')))
                            value = value.split('-')
                            joiner = 'e-'
                        except:
                            # value = float('e+'.join(value.split('+')))
                            value = value.split('+')
                            joiner = 'e+'
                        if value[0] == '':
                            value = float(joiner.join(value[1::]))
                        else:
                            value = float(joiner.join(value))
                row.append(value)
                start += width
            data.append(row)
    df = pd.DataFrame(data, columns=['E', 'Gg', 'Gn1', 'Gn2', 'Gn3', 'varyE', 'varyGg', 'varyGn1', 'varyGn2', 'varyGn3', 'J_ID'])
    return df.dropna(axis=1)



# =============================================================================
# 
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


def fill_sammy_ladder(df, particle_pair, vary_parm=False):

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


def write_saminp(filepath, 
                model, bayes, 
                reaction, 
                sammy_INP: SammyInputData,
                alphanumeric = []):
    
    ac = sammy_INP.particle_pair.ac*10  
    broadening = True
    alphanumeric_base = ["TWENTY", 
                        "EV", 
                        "GENERATE PLOT FILE AUTOMATICALLY"]
    
    if bayes:
        bayes_cmd = "SOLVE BAYES EQUATIONS"
    else:
        bayes_cmd = "DO NOT SOLVE BAYES EQUATIONS"
    alphanumeric = [model, bayes_cmd] + alphanumeric_base + alphanumeric

    with open(filepath,'r') as f:
        old_lines = f.readlines()

    with open(filepath,'w') as f:
        for line in old_lines:

            if "broadening is not wa" in line.lower() or np.any([each.lower().startswith("broadening is not wa") for each in alphanumeric]):
                broadening = False

            if line.startswith("%%%alphanumeric%%%"):
                for cmd in alphanumeric:
                    f.write(f'{cmd}\n')

            elif line.startswith('%%%card5/6%%%'):
                if broadening:
                    assert(np.any(np.isfinite(np.array([sammy_INP.temp, sammy_INP.FP, sammy_INP.frac_res_FP]))))
                    f.write(f'  {sammy_INP.temp: <8}  {sammy_INP.FP: <8}  {sammy_INP.frac_res_FP: <8}        \n')
                else:
                    pass

            elif line.startswith('%%%card7%%%'):
                assert(isinstance(sammy_INP.target_thickness, float))
                f.write(f'  {ac: <8}  {sammy_INP.target_thickness: <8}                       0.00000          \n')

            elif line.startswith('%%%card8%%%'):
                f.write(f'{reaction}\n')

            else:
                f.write(line)
           





def make_runDIR(sammy_runDIR):
    if os.path.isdir(sammy_runDIR):
        pass
    else:
        os.mkdir(sammy_runDIR)

def fill_runDIR_with_templates(sammy_RTO: SammyRunTimeOptions):

    if os.path.basename(sammy_RTO.inptemplate) == sammy_RTO.inptemplate:
        template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sammy_templates', sammy_RTO.inptemplate)
    else:
        template_path = sammy_RTO.inptemplate

    shutil.copy(template_path, os.path.join(sammy_RTO.sammy_runDIR, sammy_RTO.inpname))
    
    # fill temporary sammy_runDIR with runtime appropriate template files
    # if one_spingroup:
    #     copy_template_to_runDIR(experimental_corrections, 'sammy_1spin.inp', sammy_runDIR)
    # else:
    #     copy_template_to_runDIR(experimental_corrections, 'sammy.inp', sammy_runDIR)
    # copy_template_to_runDIR(experimental_corrections, 'sammy.par', sammy_runDIR)


from ATARI.sammy_interface.sammy_classes import SammyInputData, SammyRunTimeOptions


def update_input_files(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):

    # write experimental data if you have it, else write using the estructure
    if sammy_INP.experimental_data is not None:
        write_samdat(sammy_INP.experimental_data, sammy_INP.experimental_cov, os.path.join(sammy_RTO.sammy_runDIR,'sammy.dat'))
    elif sammy_INP.energy_grid is not None:
        write_estruct_file(sammy_INP.energy_grid, os.path.join(sammy_RTO.sammy_runDIR,'sammy.dat'))
    else:
        raise ValueError("Please provide either experimental data or an energy grid in SammyInputData")

    # edit copied input template file
    # write_saminp(sammy_RTO.model, sammy_INP.particle_pair, sammy_RTO.reaction, sammy_RTO.solve_bayes, os.path.join(sammy_RTO.sammy_runDIR, 'sammy.inp'))
    write_saminp(os.path.join(sammy_RTO.sammy_runDIR, 'sammy.inp'), 
                sammy_RTO.model, sammy_RTO.solve_bayes, 
                sammy_RTO.reaction, sammy_INP, alphanumeric=sammy_RTO.alphanumeric)

    # write parameter file
    write_sampar(sammy_INP.resonance_ladder, sammy_INP.particle_pair, sammy_INP.initial_parameter_uncertainty, os.path.join(sammy_RTO.sammy_runDIR,"SAMMY.PAR"), vary_parm=sammy_RTO.solve_bayes)


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



def runsammy_shellpipe(sammy_RTO: SammyRunTimeOptions):
    # run sammy and wait for completion with subprocess
    runsammy_process = subprocess.run(
                                    [f"{sammy_RTO.shell}", "-c", f"{sammy_RTO.path_to_SAMMY_exe}<pipe.sh"], 
                                    cwd=os.path.realpath(sammy_RTO.sammy_runDIR),
                                    capture_output=True
                                    )
    if len(runsammy_process.stderr) > 0:
        print(f'SAMMY gave the following warning or error: {runsammy_process.stderr}')

    return

# ###############################################
# run sammy with NV or IQ scheme
# ###############################################



def get_ECSCM(sammy_RTO, sammy_INP):

    # update_input_files(sammy_INP, sammy_RTO)
    fill_runDIR_with_templates(sammy_RTO)
    energy_grid = np.linspace(min(sammy_INP.energy_grid), max(sammy_INP.energy_grid), 498)
    write_estruct_file(energy_grid, os.path.join(sammy_RTO.sammy_runDIR,'sammy.dat'))
    write_saminp(os.path.join(sammy_RTO.sammy_runDIR, 'sammy.inp'), 
                sammy_RTO.model, False, 
                sammy_RTO.reaction, sammy_INP, 
                alphanumeric=sammy_RTO.alphanumeric + ["CROSS SECTION COVARIance matrix is wanted"])
    
    write_shell_script(sammy_INP, sammy_RTO, use_RPCM=True)
    runsammy_shellpipe(sammy_RTO)

    df, cov = read_ECSCM(os.path.join(sammy_RTO.sammy_runDIR, "SAMCOV.PUB"))

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



def run_sammy(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):

    sammy_INP.resonance_ladder = copy(sammy_INP.resonance_ladder)

    # setup sammy runtime files
    make_runDIR(sammy_RTO.sammy_runDIR)
    fill_runDIR_with_templates(sammy_RTO)
    update_input_files(sammy_INP, sammy_RTO)
    write_shell_script(sammy_INP, sammy_RTO)

    lst_df, par_df = execute_sammy(sammy_RTO)

    sammy_OUT = SammyOutputData(pw=lst_df, 
                        par=sammy_INP.resonance_ladder)#,
                        # chi2=chi2_val(lst_df.theo_xs, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc)))

    if sammy_RTO.recursive == True:
        if sammy_RTO.solve_bayes == False:
            raise ValueError("Cannot run recursive sammy with solve bayes set to false")
        lst_df, par_df = recursive_sammy(lst_df, par_df, sammy_INP, sammy_RTO)

    if sammy_RTO.solve_bayes:
        # sammy_OUT.chi2_post = chi2_val(lst_df.theo_xs_bayes, lst_df.exp_xs, np.diag(lst_df.exp_xs_unc))
        sammy_OUT.pw=lst_df
        sammy_OUT.par_post = par_df

        if sammy_RTO.get_ECSCM:
            est_df, ecscm = get_ECSCM(sammy_RTO, sammy_INP)
            sammy_OUT.ECSCM = ecscm
            sammy_OUT.est_df = est_df

    if not sammy_RTO.keep_runDIR:
        shutil.rmtree(sammy_RTO.sammy_runDIR)

    return sammy_OUT





#########################################################
# run sammy with the YW scheme
#########################################################


# def copy_inptemp_to_runDIR(template_path, target_path):
#     shutil.copy(template_path, target_path)


def make_inputs_for_YW(sammyINP, sammyRTO, datasets, reactions, templates):

    for ds, rxn, tem in zip(datasets, reactions, templates):

        sammyRTO.template = tem

        sammyRTO.inpname = f"{ds}_initial.inp"# os.path.join(sammyRTO.sammy_runDIR, f"{ds}_initial.inp")
        fill_runDIR_with_templates(sammyRTO)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{ds}_initial.inp"), sammyRTO.model, True, rxn, sammyINP, 
                                    alphanumeric=["yw"])

        sammyRTO.inpname = f"{ds}_iter.inp"# os.path.join(sammyRTO.sammy_runDIR, f"{ds}_iter.inp")
        fill_runDIR_with_templates(sammyRTO)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{ds}_iter.inp"), sammyRTO.model, True, rxn, sammyINP, 
                                    alphanumeric=["yw","Use remembered original parameter values"])

        sammyRTO.inpname = f"{ds}_plot.inp"# os.path.join(sammyRTO.sammy_runDIR, f"{ds}_plot.inp")
        fill_runDIR_with_templates(sammyRTO)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{ds}_plot.inp"), sammyRTO.model, False, rxn, sammyINP, 
                                    alphanumeric=[])

    # copy_inptemp_to_runDIR("/Users/noahwalton/Documents/GitHub/ATARI/examples/sammy_template.inp", os.path.join(sammyRTO.sammy_runDIR, "solvebayes_initial.inp"))
    sammyRTO.inpname = "solvebayes_initial.inp"# os.path.join(sammyRTO.sammy_runDIR, "solvebayes_initial.inp")
    fill_runDIR_with_templates(sammyRTO)
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_initial.inp"), sammyRTO.model, True, "reaction", sammyINP, 
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Remember original parameter values", "Take baby steps with Least-Squares method"])#, "Broadening is not wanted"])


    # copy_inptemp_to_runDIR("/Users/noahwalton/Documents/GitHub/ATARI/examples/sammy_template.inp", os.path.join(sammyRTO.sammy_runDIR, "solvebayes_iter.inp"))
    sammyRTO.inpname = "solvebayes_iter.inp" #os.path.join(sammyRTO.sammy_runDIR, "solvebayes_iter.inp")
    fill_runDIR_with_templates(sammyRTO)
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_iter.inp"), sammyRTO.model, True, "reaction", sammyINP, 
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Use remembered original parameter values", "Take baby steps with Least-Squares method"])#, "Broadening is not wanted"])



def make_bash_script_iterate(iterations, dataset_titles, rundir, sammyexe, shell, save_each_step=True):

    with open(os.path.join(rundir, f"iterate.{shell}"), 'w+') as f:
            f.write("""### Filter out resonances with Gn < 1e-4 meV
par_file=results/step$1.par
temp_file=results/temp.par

awk '{
  # Replace "-" with "e-" in the third column
  gsub(/-/, "e-", $3)

  column3 = $3 + 0

  if (column3 <= -1e-4 || column3 >= 1e-4) {
    print $0
  }
}' "$par_file" > "$temp_file"

mv $temp_file $par_file
\n""")

            for i in range(1, iterations+1):

                stepstr_prior = f'iter{i-1}'
                if i == 1:
                    cov = ''
                    par = 'results/step$1.par'
                    inp_ext = 'initial'
                else:
                    cov = f"iterate/bayes_{stepstr_prior}.cov"
                    par = f"iterate/bayes_{stepstr_prior}.par"
                    inp_ext = 'iter'

                # Write commands for each dataset
                for ds in dataset_titles:
                    title = f"{ds}_{stepstr_prior}"#step{i}
                    f.write(f"##################################\n# Generate YW for {ds}\n")
                    f.write(f"{sammyexe}<<EOF\n{ds}_{inp_ext}.inp\n{par}\n{ds}.dat\n{cov}\n\nEOF\n")
                    f.write(f"""mv -f SAMMY.LPT "iterate/{title}.lpt" \nmv -f SAMMY.ODF "iterate/{title}.odf" \nmv -f SAMMY.LST "iterate/{title}.lst" \nmv -f SAMMY.YWY "iterate/{title}.ywy" \n""")    

                # write commands for bayes solve using all datasets
                dataset_inserts = "\n".join([f"iterate/{ds}_{stepstr_prior}.ywy" for ds in dataset_titles])
                title = f"bayes_iter{i}"
                f.write(f"#######################################################\n# Run Bayes for {title}\n#\n#######################################################\n")
                f.write(f"{sammyexe}<<eod\nsolvebayes_{inp_ext}.inp\n{par}\ndummy.dat\n{dataset_inserts}\n\n{cov}\n\neod\n")
                f.write(f"""mv -f SAMMY.LPT iterate/{title}.lpt \nmv -f SAMMY.PAR iterate/{title}.par \nmv -f SAMMY.COV iterate/{title}.cov \nrm -f SAM*\n""")

            f.write(f"\n\n\n\n############## Copy Iteration Result ###########\nplus_one=$(( $1 + 1 ))\nhead -$(($(wc -l < iterate/bayes_iter{iterations}.par) - 3)) iterate/bayes_iter{iterations}.par > results/step$plus_one.par\n\nrm REMORI.PAR\n")
            
            ### After all steps, write commands to make plot for step
            for ds in dataset_titles:
                f.write(f"\n#######################################################\n# Plotting for {ds} final iteration \n#\n#######################################################\n")
                f.write(f"{sammyexe}<<EOF\n{ds}_plot.inp\nresults/step$plus_one.par\n{ds}.dat\n\nEOF\n\n")

                # write lines to grep chi2
                f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" SAMMY.LPT)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
                f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")
                
                if save_each_step:
                    f.write(f"mv -f SAMMY.LST results/{ds}_step$plus_one.lst\nmv -f SAMMY.LPT results/{ds}_step$plus_one.lpt\n\n")
            
            f.write("rm -f SAM*\n\n")

            # log an return chi2 values
            f.write("""echo "$plus_one""")
            for ds in dataset_titles:
                f.write(f" $chi2_string_{ds}")
            f.write("""" >> chi2.txt\n""")

            f.write("""echo " """)
            for ds in dataset_titles:
                f.write(f" $chi2_string_{ds}")
            f.write(""""\n""")

                # f.write(f"{sammyexe}<<EOF\n{ds}_plot.inp\ninitial.par\n{ds}.dat\n\nEOF\n")
                # f.write(f"mv -f SAMMY.LST {ds}_initial.lst\nrm -f SAM*\n")


def make_data_for_YW(datasets, dataset_titles, rundir):
    for d, dt in zip(datasets, dataset_titles):
        write_samdat(d, None, os.path.join(rundir,f"{dt}.dat"))
        write_estruct_file(d.E, os.path.join(rundir,"dummy.dat"))

def make_bash_script_run(steps, dataset_titles, threshold, sammyexe, shell, rundir):
    with open(os.path.join(rundir, f"run.{shell}"), 'w+') as f:

        ### change cwd and setup chi2 log
        f.write(f"""
# change cwd
cd "$(dirname "$0")"
#### remove existing chi2 log and write new
rm chi2.txt\n""")
        f.write("""echo "chi2" >> chi2.txt\necho "step""")
        for ds in dataset_titles:
            f.write(f" {ds}") 
        f.write("""" >> chi2.txt\n""")

        ### plot prior for each dataset and grep chi2
        for ds in dataset_titles:
            f.write(f"""\n\n#### plot prior for {ds}
out=$({sammyexe}<<EOF
{ds}_plot.inp
results/step0.par
{ds}.dat

EOF
)
mv -f SAMMY.LST results/{ds}_step0.lst
mv -f SAMMY.LPT results/{ds}_step0.lpt
rm -f SAM*\n""")
            f.write(f"""chi2_line_{ds}=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/{ds}_step0.lpt)\nchi2_string_{ds}=$(echo "$chi2_line_{ds}" """)
            f.write("""| awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')\n\n""")

        ### Echo prior to chi2 log
        f.write("""echo "0""")
        for ds in dataset_titles:
            f.write(f" $chi2_string_{ds}")
        f.write("""" >> chi2.txt\n""")

        ### now write iteration script 
        f.write(f"""
\n\n\n#### For each step
#   1. run iteration for nonlinearities 
#   2. OPTIONAL: Generate plot for step

max_iterations={steps}
threshold={threshold}
        
iteration=0
criteria_met=false
criteria="max iterations"
echo "\nIterating until convergence\nchi2 values\n""")
        
        f.write("step ")
        for ds in dataset_titles:
            f.write(f" {ds}") 
        f.write(""" sum"\n""")
        
        # write while loop
        f.write(f"""
while [ $iteration -lt $max_iterations ] && [ "$criteria_met" = false ]; do

    # new chi2 is output from step
    output=$(./iterate.{shell} $iteration)\n""")
        f.write(r"""
    newchi2_str=$(echo "$output" | tail -n 1)
    newchi2_str=$(echo "$newchi2_str" | sed 's/E/e/g; s/+//g')
    newchi2_sum=$(echo "$newchi2_str" | tr -s ' ' '\n' | grep -E '[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?' | paste -sd+ - | bc -l)  

    # oldchi2 is read from chi2 log
    line_with_oldchi2=$(grep -E "^$iteration " chi2.txt)
    line_with_oldchi2=$(echo "$line_with_oldchi2" | awk '{$1=""; sub(/^[[:space:]]+/, "")}1')
    oldchi2_str=$(echo "$line_with_oldchi2" | sed 's/E/e/g; s/+//g')
    oldchi2_sum=$(echo "$oldchi2_str" | tr -s ' ' '\n' | grep -E '[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?' | paste -sd+ - | bc -l)

    # termination criteria on change in chi2
    diff=$(echo "$oldchi2_sum - $newchi2_sum" | bc) 

    # print 
    echo $iteration $oldchi2_str $oldchi2_sum

    # Check if the termination criteria is met (e.g., based on some condition)
    if (( $(echo "$diff < $threshold" |bc -l) )); then # || (( $(echo "$diff > $threshold" | bc -l) )); then
        criteria_met=true
        criteria="improvement below threshold"
    fi
    
    # Increment the iteration counter
    ((iteration++))
    
done

echo "Loop terminated after $iteration iterations because $criteria."
echo $iteration
""")





def setup_YW_scheme(sammyRTO, sammyINP, datasets, dataset_titles, reactions, templates, 
                                                                                steps=10,
                                                                                iterations=5,
                                                                                threshold=0.01):

    ### clean and make directories
    try:
        shutil.rmtree(sammyRTO.sammy_runDIR)
    except:
        pass
    os.mkdir(sammyRTO.sammy_runDIR)
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "results"))
    os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "iterate"))
    # for ds in datasets:
    #     os.mkdir(os.path.join(sammyRTO.sammy_runDIR, "results", ds))

    make_data_for_YW(datasets, dataset_titles, sammyRTO.sammy_runDIR)
    write_sampar(sammyINP.resonance_ladder, sammyINP.particle_pair, 0.1, os.path.join(sammyRTO.sammy_runDIR, "results/step0.par"))

    make_inputs_for_YW(sammyINP, sammyRTO, dataset_titles, reactions, templates)
    make_bash_script_iterate(iterations, dataset_titles, sammyRTO.sammy_runDIR, sammyRTO.path_to_SAMMY_exe, sammyRTO.shell )
    make_bash_script_run(steps, dataset_titles, threshold, sammyRTO.path_to_SAMMY_exe, sammyRTO.shell,  sammyRTO.sammy_runDIR)



def run_YW_scheme(sammyINP_YW: SammyInputDataYW, sammyRTO: SammyRunTimeOptions, resonance_ladder: pd.DataFrame):
    sammyINP_YW.resonance_ladder = resonance_ladder

    setup_YW_scheme(sammyRTO, sammyINP_YW, sammyINP_YW.datasets, sammyINP_YW.dataset_titles, sammyINP_YW.reactions, sammyINP_YW.templates, 
                                                                                    steps=sammyINP_YW.steps,
                                                                                    iterations=sammyINP_YW.iterations,
                                                                                    threshold=sammyINP_YW.threshold)

    os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'iterate.{sammyRTO.shell}')}")
    os.system(f"chmod +x {os.path.join(sammyRTO.sammy_runDIR, f'run.{sammyRTO.shell}')}")

    result = subprocess.check_output(os.path.join(sammyRTO.sammy_runDIR, f'run.{sammyRTO.shell}'), shell=True, text=True)
    ifinal = int(result.splitlines()[-1]) -1
    return ifinal


















# def make_bash_script_run(steps, rundir):
#     with open(os.path.join(rundir, "run.zsh"), 'w+') as f:
#         f.write("""# change cwd
# cd "$(dirname "$0")"\n
# # remove existing chi2 log
# rm chi2.txt\n\n
# #### plot prior
# /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
# trans1_plot.inp
# results/step0.par
# trans1.dat

# EOF
# mv -f SAMMY.LST results/trans1_step0.lst
# mv -f SAMMY.LPT results/trans1_step0.lpt
# rm -f SAM*
# #### For each step
# #   1. run iteration for nonlinearities 
# #   2. OPTIONAL: Generate plot for step\n""")

#         f.write( "for number in {0..$steps$}".replace("$steps$",str(steps)) )

#         f.write("""\ndo
#     ./iterate.zsh $number

#     # #######################################################
#     # # Plotting for step - don't need to do each time
#     # /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
#     # trans1_plot.inp
#     # results/step$number.par
#     # trans1.dat

#     # EOF
#     # mv -f SAMMY.LST results/trans1_step$plus_one.lst
#     # mv -f SAMMY.LPT results/trans1_step$plus_one.lpt
#     # rm -f SAM*


#         # line_with_chi2=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/trans1_step$number.lpt)
#         # chi2_string=$(echo "$line_with_chi2" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
#         # chi2=$(echo "$chi2_string" | bc -l)
#         # # echo "chi2 (float): $chi2"
#         # # # Example: Perform arithmetic operations with chi2
#         # # result=$(echo "$chi2 * 2" | bc -l)
#         # # echo "Result of chi2 * 2: $result"

#         # echo "step $number chi2: $chi2_string" >> chi2.txt

#     done

#     # ./iterate.zsh 0

#     # line_with_chi2=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" SAMMY.LPT)
#     # chi2_string=$(echo "$line_with_chi2" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
#     # chi2=$(echo "$chi2_string" | bc -l)
#     # # echo "chi2 (float): $chi2"
#     # # # Example: Perform arithmetic operations with chi2
#     # # result=$(echo "$chi2 * 2" | bc -l)
#     # # echo "Result of chi2 * 2: $result"

#     # echo "step 0 chi2: $chi2_string" >> chi2.txt
#     # # echo  "$chi2"





#     # ############## Copy Final Result ###########
#     # head -$(($(wc -l < results/bayes_step50.par) - 3)) results/bayes_step50.par > final.par

#     # #######################################################
#     # # Plotting for trans1 Final/Initial
#     # #
#     # #######################################################
#     # /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
#     # trans1_plot.inp
#     # final.par
#     # trans1.dat

#     # EOF
#     # mv -f SAMMY.LST trans1_final.lst
#     # rm -f SAM*

#     # /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
#     # trans1_plot.inp
#     # initial.par
#     # trans1.dat

#     # EOF
#     # mv -f SAMMY.LST trans1_initial.lst
#     # rm -f SAM*""")
        


# =============================================================================
# 
# =============================================================================
# def read_sammy_par(filename, calculate_average):
#     """
#     Reads sammy.par file and calculates average parameters.

#     _extended_summary_

#     Parameters
#     ----------
#     filename : _type_
#         _description_
#     calculate_average : bool
#         Whether or not to calculate average parameters.

#     Returns
#     -------
#     DataFrame
#         Contains the average parameters for each spin group
#     DataFrame
#         Contains all resonance parameters for all spin groups
#     """

#     energies = []; spin_group = []; nwidth = []; gwidth = []
#     with open(filename,'r') as f:   
#         readlines = f.readlines()
#         in_res_dat = True
#         for line in readlines:   

#             try:
#                 float(line.split()[0]) # if first line starts with a float, parameter file starts directly with resonance data
#             except:
#                 in_res_dat = False # if that is not the case, parameter file has some keyword input before line "RESONANCES" then resonance parameters

#             if line.startswith(' '):
#                 in_res_dat = False  
#             if in_res_dat:
#                 if line.startswith('-'):
#                     continue #ignore negative resonances
#                 else:
#                     splitline = line.split()
#                     energies.append(float(splitline[0]))
#                     gwidth.append(float(splitline[1]))
#                     nwidth.append(float(splitline[2]))
#                     spin_group.append(float(splitline[-1]))
#             if line.startswith('RESONANCE'):
#                 in_res_dat = True

                
                
#     Gg = np.array(gwidth); Gn = np.array(nwidth); E = np.array(energies); jspin = np.array(spin_group)
#     df = pd.DataFrame([E, Gg, Gn, jspin], index=['E','Gg','Gn','jspin']); df = df.transpose()
    
#     if calculate_average:
#         #avg_widths = df.groupby('jspin', as_index=False)['Gg','Gn'].mean() 
#         gb = df.groupby('jspin')    
#         list_of_dfs=[gb.get_group(x) for x in gb.groups]
        
#         avg_df = pd.DataFrame(index=df['jspin'].unique(),columns=['dE','Gg','Gn'])

#         for ij, jdf in enumerate(list_of_dfs):
#             avg_df['dE'][ij+1]=jdf['E'].diff().mean()
#             avg_df['Gg'][ij+1]=jdf['Gg'].mean()
#             avg_df['Gn'][ij+1]=jdf['Gn'].mean()
#     else:
#         avg_df = ''

#     return avg_df, df
    


# =============================================================================
# 
# ============================================================================= 
# def copy_template_to_runDIR(experimental_corrections, filename, target_dir):
#     if os.path.splitext(filename)[0] != 'sammy':
#         file_out = ''.join(('sammy', os.path.splitext(filename)[1]))
#     else:
#         file_out = filename
#     shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sammy_templates', f'{experimental_corrections}', filename), 
#                 os.path.join(target_dir,file_out))
#     return