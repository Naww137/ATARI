
from ATARI.models.particle_pair import Particle_Pair
import pandas as pd
import os
import shutil
from ATARI.sammy_interface.sammy_functions import fill_runDIR_with_templates, write_saminp, write_samdat, write_sampar
import subprocess



def make_inputs_for_YW(sammyINPYW: SammyInputDataYW, sammyRTO:SammyRunTimeOptions):

    #### make files for each dataset YW generation
    for exp, tem in zip(sammyINPYW.experiments, sammyINPYW.templates):  # fix this !!

        sammyRTO.options["bayes"] = True
        fill_runDIR_with_templates(tem, f"{exp.title}_initial.inp", sammyRTO.sammy_runDIR)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{exp.title}_initial.inp"), sammyINPYW.model, exp, sammyRTO, 
                                    alphanumeric=["yw"])

        fill_runDIR_with_templates(tem, f"{exp.title}_iter.inp", sammyRTO.sammy_runDIR)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{exp.title}_iter.inp"), sammyINPYW.model, exp, sammyRTO, 
                                    alphanumeric=["yw","Use remembered original parameter values"])

        sammyRTO.options["bayes"] = False
        fill_runDIR_with_templates(tem, f"{exp.title}_plot.inp", sammyRTO.sammy_runDIR)
        write_saminp(os.path.join(sammyRTO.sammy_runDIR,f"{exp.title}_plot.inp"), sammyINPYW.model, exp, sammyRTO,  
                                    alphanumeric=[])
    
    ### options for least squares
    if sammyINPYW.LS:
        alphanumeric_LS_opts = ["USE LEAST SQUARES TO GIVE COVARIANCE MATRIX", "Take baby steps with Least-Squares method"]
    else:
        alphanumeric_LS_opts = []

    #### make files for solving bayes reading in each YW matrix  -- # TODO: I should define a better template/exp here
    fill_runDIR_with_templates(tem, "solvebayes_initial.inp", sammyRTO.sammy_runDIR)
    sammyRTO.options["bayes"] = True
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_initial.inp"), sammyINPYW.model, exp, sammyRTO,   
                                alphanumeric=["wy", "CHI SQUARED IS WANTED", "Remember original parameter values"]+alphanumeric_LS_opts)

    fill_runDIR_with_templates(tem, "solvebayes_iter.inp" , sammyRTO.sammy_runDIR)
    write_saminp(os.path.join(sammyRTO.sammy_runDIR,"solvebayes_iter.inp"), sammyINPYW.model, exp, sammyRTO, 
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
    # return istep



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





class sammyYW:


    def __init__(self, 
                 particle_pair: Particle_Pair,
                 resonance_ladder: pd.DataFrame,
                 options={}):

        self.particle_pair = particle_pair
        # model: theory
        resonance_ladder: DataFrame

        datasets : list[Union[pd.DataFrame, np.ndarray]]
        templates : list[str]
        experiments: list[experimental_model]

        max_steps: int = 1
        iterations: int = 2
        step_threshold: float = 0.01
        autoelim_threshold: Optional[float] = None

        LS: bool = False
        LevMar: bool = True
        LevMarV: float = 2.0
        # LevMarVd: float = 2.0
        minF:   float = 1e-5
        maxF:   float = 10
        initial_parameter_uncertainty: float = 1.0