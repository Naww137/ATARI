

import numpy as np
import os
import shutil
from pathlib import Path
from ATARI.theory import scattering_params
import pandas as pd
import subprocess
# from ATARI.utils.atario import fill_resonance_ladder
from ATARI.theory.scattering_params import FofE_recursive
from ATARI.utils.stats import chi2_val

from ATARI.sammy_interface.sammy_classes import SammyRunTimeOptions, SammyInputData, SammyOutputData


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
        df = pd.read_csv(filepath, delim_whitespace=True, names=['E','exp_dat','exp_dat_unc'])
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
                        value = float('e-'.join(value[1::].split('-')))
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


def fill_sammy_ladder(df, particle_pair):

    def gn2G(row):
        _, P, _, _ = FofE_recursive([row.E], particle_pair.ac, particle_pair.M, particle_pair.m, row.lwave)
        Gn = 2*np.sum(P)*row.gn2
        return Gn.item()

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

    return df

def write_sampar(df, pair, vary_parm, initial_parameter_uncertainty, filename, template=None):
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
        samtools_array = []
    else:
        # df = fill_resonance_ladder(df, pair)
        df = fill_sammy_ladder(df, pair)

        ### force to 0.001 if Gnx == 0
        df.loc[df['Gn1']<=1e-5, 'Gn1'] = 1e-5

        par_array = np.array([df.E, df.Gg, df.Gn1]).T
        zero_neutron_widths = 5-(len(par_array[0]))
        zero_neutron_array = np.zeros([len(par_array),zero_neutron_widths])
        par_array = np.insert(par_array, [5-zero_neutron_widths], zero_neutron_array, axis=1)

        if vary_parm:
            binary_array = np.hstack((np.ones([len(par_array),5-zero_neutron_widths]), np.zeros([len(par_array),zero_neutron_widths])))
        else:
            binary_array = np.zeros([len(par_array),5])

        samtools_array = np.insert(par_array, [5], binary_array, axis=1)
        if np.any([each is None for each in df.J_ID]):
            raise ValueError("NoneType was passed as J_ID in the resonance ladder")
        j_array = np.array([df.J_ID]).T
        samtools_array = np.hstack((samtools_array, j_array))

    widths = [11, 11, 11, 11, 11, 2, 2, 2, 2, 2, 2]
    with open(filename, 'w') as file:
        for row in samtools_array:
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
        try:
            exp_pw.rename(columns={'exp_trans':'exp', 'exp_trans_unc': 'exp_unc'}, inplace=True)
        except:
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


def write_saminp(filepath, 
                model, bayes, 
                reaction, 
                sammy_INP: SammyInputData,
                alphanumeric = [],
                alphanumeric_base = ["TWENTY", 
                                    "EV", 
                                    "GENERATE PLOT FILE AUTOMATICALLY"]):
    
    ac = sammy_INP.particle_pair.ac*10
    broadening = True
    
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
                sammy_RTO.reaction, sammy_INP)

    # write parameter file
    write_sampar(sammy_INP.resonance_ladder, sammy_INP.particle_pair, sammy_RTO.solve_bayes, sammy_INP.initial_parameter_uncertainty, os.path.join(sammy_RTO.sammy_runDIR,"SAMMY.PAR"))


def write_shell_script(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):
    with open(os.path.join(sammy_RTO.sammy_runDIR, 'pipe.sh'), 'w') as f:
        f.write('sammy.inp\nSAMMY.PAR\nsammy.dat')
        if sammy_RTO.energy_window is None:
            f.write('\n')
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


def execute_sammy(sammy_RTO:SammyRunTimeOptions):

    # run sammy and wait for completion with subprocess
    runsammy_process = subprocess.run(
                                    [f"{sammy_RTO.shell}", "-c", f"{sammy_RTO.path_to_SAMMY_exe}<pipe.sh"], 
                                    cwd=os.path.realpath(sammy_RTO.sammy_runDIR),
                                    capture_output=True
                                    )
    if len(runsammy_process.stderr) > 0:
        print(f'SAMMY gave the following warning or error: {runsammy_process.stderr}')

    # read output  and delete sammy_runDIR
    lst_df = readlst(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.LST'))
    # if sammy_RTO.solve_bayes:
        # par_df = pd.read_csv(os.path.join(sammy_RTO.sammy_runDIR, 'SAMMY.PAR'), skipfooter=2, delim_whitespace=True, usecols=[0,1,2,6], names=['E', 'Gg', 'Gnx','J_ID'], engine='python')
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
    write_sampar(par_prior, sammy_INP.particle_pair, sammy_RTO.solve_bayes, sammy_INP.initial_parameter_uncertainty, os.path.join(sammy_RTO.sammy_runDIR,"SAMMY.PAR"))
    pw_posterior, par_posterior = execute_sammy(sammy_RTO)
    
    Dchi2 = delta_chi2(pw_posterior)
    if sammy_RTO.recursive_opt["print"]:
        print(Dchi2)

    if Dchi2 <= sammy_RTO.recursive_opt["threshold"]:
        return pw_prior, par_prior
    
    return recursive_sammy(pw_posterior, par_posterior, sammy_INP, sammy_RTO, itter + 1)



def run_sammy(sammy_INP: SammyInputData, sammy_RTO:SammyRunTimeOptions):

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
        sammy_OUT.par_post = par_df

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

        sammyRTO.inpname = os.path.join(sammyRTO.sammy_runDIR, f"{ds}_initial.inp")
        fill_runDIR_with_templates(sammyRTO)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{ds}_initial.inp"), sammyRTO.model, sammyRTO.solve_bayes, rxn, sammyINP, 
                                    alphanumeric=["yw"])

        sammyRTO.inpname = os.path.join(sammyRTO.sammy_runDIR, f"{ds}_iter.inp")
        fill_runDIR_with_templates(sammyRTO)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{ds}_iter.inp"), sammyRTO.model, sammyRTO.solve_bayes, rxn, sammyINP, 
                                    alphanumeric=["yw","Use remembered original parameter values"])

        sammyRTO.inpname = os.path.join(sammyRTO.sammy_runDIR, f"{ds}_plot.inp")
        fill_runDIR_with_templates(sammyRTO)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{ds}_plot.inp"), sammyRTO.model, False, rxn, sammyINP, 
                                    alphanumeric=[])

    # copy_inptemp_to_runDIR("/Users/noahwalton/Documents/GitHub/ATARI/examples/sammy_template.inp", os.path.join(sammyRTO.sammy_runDIR, "solvebayes_initial.inp"))
    sammyRTO.inpname = os.path.join(sammyRTO.sammy_runDIR, "solvebayes_initial.inp")
    fill_runDIR_with_templates(sammyRTO)
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_initial.inp"), sammyRTO.model, sammyRTO.solve_bayes, "reaction", sammyINP, 
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Remember original parameter values", "Take baby steps with Least-Squares method"])#, "Broadening is not wanted"])


    # copy_inptemp_to_runDIR("/Users/noahwalton/Documents/GitHub/ATARI/examples/sammy_template.inp", os.path.join(sammyRTO.sammy_runDIR, "solvebayes_iter.inp"))
    sammyRTO.inpname = os.path.join(sammyRTO.sammy_runDIR, "solvebayes_iter.inp")
    fill_runDIR_with_templates(sammyRTO)
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_iter.inp"), sammyRTO.model, sammyRTO.solve_bayes, "reaction", sammyINP, 
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Use remembered original parameter values", "Take baby steps with Least-Squares method"])#, "Broadening is not wanted"])



def make_bash_script_iterate(iterations, dataset_titles, rundir, sammyexe, save_each_step=True):

    with open(os.path.join(rundir, "iterate.zsh"), 'w+') as f:

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

            f.write(f"\n\n\n\n############## Copy Iteration Result ###########\nplus_one=$(( $1 + 1 ))\nhead -$(($(wc -l < iterate/bayes_iter{iterations}.par) - 3)) iterate/bayes_iter{iterations}.par > results/step$plus_one.par\n\n")
            
            ### After all steps, write commands to make plot for step
            for ds in dataset_titles:
                f.write(f"rm REMORI.PAR\n#######################################################\n# Plotting for {ds} final iteration \n#\n#######################################################\n")
                f.write(f"{sammyexe}<<EOF\n{ds}_plot.inp\nresults/step$plus_one.par\n{ds}.dat\n\nEOF\n\n")

                # write lines to grep chi2
                f.write("""line_with_chi2=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" SAMMY.LPT)
    chi2_string=$(echo "$line_with_chi2" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
    echo "step $plus_one chi2: $chi2_string" >> chi2.txt\n\n""")
                
                if save_each_step:
                    f.write(f"mv -f SAMMY.LST results/{ds}_step$plus_one.lst\nmv -f SAMMY.LPT results/trans1_step$plus_one.lpt\n")

                f.write("rm -f SAM*\n\n")

                # f.write(f"{sammyexe}<<EOF\n{ds}_plot.inp\ninitial.par\n{ds}.dat\n\nEOF\n")
                # f.write(f"mv -f SAMMY.LST {ds}_initial.lst\nrm -f SAM*\n")


def make_data_for_YW(datasets, dataset_titles, rundir):
    for d, dt in zip(datasets, dataset_titles):
        write_samdat(d, None, os.path.join(rundir,f"{dt}.dat"))
        write_estruct_file(d.E, os.path.join(rundir,"dummy.dat"))


def make_bash_script_run(steps, rundir):
    with open(os.path.join(rundir, "run.zsh"), 'w+') as f:
        f.write("""# change cwd
cd "$(dirname "$0")"\n
# remove existing chi2 log
rm chi2.txt\n\n
#### plot prior
/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
trans1_plot.inp
results/step0.par
trans1.dat

EOF
mv -f SAMMY.LST results/trans1_step0.lst
mv -f SAMMY.LPT results/trans1_step0.lpt
rm -f SAM*
#### For each step
#   1. run iteration for nonlinearities 
#   2. OPTIONAL: Generate plot for step\n""")

        f.write( "for number in {0..$steps$}".replace("$steps$",str(steps)) )

        f.write("""\ndo
    ./iterate.zsh $number

    # #######################################################
    # # Plotting for step - don't need to do each time
    # /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
    # trans1_plot.inp
    # results/step$number.par
    # trans1.dat

    # EOF
    # mv -f SAMMY.LST results/trans1_step$plus_one.lst
    # mv -f SAMMY.LPT results/trans1_step$plus_one.lpt
    # rm -f SAM*


        # line_with_chi2=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" results/trans1_step$number.lpt)
        # chi2_string=$(echo "$line_with_chi2" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
        # chi2=$(echo "$chi2_string" | bc -l)
        # # echo "chi2 (float): $chi2"
        # # # Example: Perform arithmetic operations with chi2
        # # result=$(echo "$chi2 * 2" | bc -l)
        # # echo "Result of chi2 * 2: $result"

        # echo "step $number chi2: $chi2_string" >> chi2.txt

    done

    # ./iterate.zsh 0

    # line_with_chi2=$(grep -i "CUSTOMARY CHI SQUARED DIVIDED" SAMMY.LPT)
    # chi2_string=$(echo "$line_with_chi2" | awk '{ for (i=1; i<=NF; i++) if ($i ~ /[0-9]+(\.[0-9]+)?([Ee][+-]?[0-9]+)?/) print $i }')
    # chi2=$(echo "$chi2_string" | bc -l)
    # # echo "chi2 (float): $chi2"
    # # # Example: Perform arithmetic operations with chi2
    # # result=$(echo "$chi2 * 2" | bc -l)
    # # echo "Result of chi2 * 2: $result"

    # echo "step 0 chi2: $chi2_string" >> chi2.txt
    # # echo  "$chi2"





    # ############## Copy Final Result ###########
    # head -$(($(wc -l < results/bayes_step50.par) - 3)) results/bayes_step50.par > final.par

    # #######################################################
    # # Plotting for trans1 Final/Initial
    # #
    # #######################################################
    # /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
    # trans1_plot.inp
    # final.par
    # trans1.dat

    # EOF
    # mv -f SAMMY.LST trans1_final.lst
    # rm -f SAM*

    # /Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy<<EOF
    # trans1_plot.inp
    # initial.par
    # trans1.dat

    # EOF
    # mv -f SAMMY.LST trans1_initial.lst
    # rm -f SAM*""")
        


def setup_YW_scheme(sammyRTO, sammyINP, datasets, dataset_titles, reactions, templates, 
                                                                                steps=10,
                                                                                iterations=5):

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
    write_sampar(sammyINP.resonance_ladder, sammyINP.particle_pair, True, 0.1, os.path.join(sammyRTO.sammy_runDIR, "results/step0.par"))

    make_inputs_for_YW(sammyINP, sammyRTO, dataset_titles, reactions, templates)
    make_bash_script_iterate(iterations, dataset_titles, sammyRTO.sammy_runDIR, sammyRTO.path_to_SAMMY_exe)
    make_bash_script_run(steps, sammyRTO.sammy_runDIR)























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