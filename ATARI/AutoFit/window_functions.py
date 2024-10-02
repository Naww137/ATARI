import numpy as np
import os
import subprocess
import time

def get_windows(resonance_energies_start, resonances_per_window, data_overlap, external_resonance_buffer, total_data_range, data_overlap_fraction = None, minimum_step=5):

    assert np.all(resonance_energies_start == np.sort(np.array(resonance_energies_start))), "resonances energies must be sorted"
    lower_external_resonances = resonance_energies_start[resonance_energies_start<min(total_data_range)]
    upper_external_resonances = resonance_energies_start[resonance_energies_start>max(total_data_range)]
    internal_resonances = resonance_energies_start[(resonance_energies_start>min(total_data_range)) & (resonance_energies_start<max(total_data_range))]
    window = 'not_last'

    at_least_max_E_data = min(total_data_range) + minimum_step + data_overlap
    at_least_max_E_resonances = at_least_max_E_data + external_resonance_buffer
    i_max_E_resonances = np.searchsorted(resonance_energies_start, at_least_max_E_resonances, 'left')

    if i_max_E_resonances > resonances_per_window:
        print(f"Increasing data range to minimum step, {i_max_E_resonances} resonances in this window")
        max_E_data = at_least_max_E_data
    else:
        i_max_E_resonances = resonances_per_window
        if i_max_E_resonances >= len(resonance_energies_start): # check that we are not beyond the internal resonances
            i_max_E_resonances -= 1
            max_E_data = max(total_data_range)
            window = 'last'
        else:
            max_E_data = resonance_energies_start[i_max_E_resonances]-external_resonance_buffer
    
    resonances = resonance_energies_start[0:i_max_E_resonances+1]
    data_range = np.array([min(total_data_range), max_E_data])
    index_range = np.array([0, i_max_E_resonances+1])

    window_resonance_energies = [resonances]
    window_data_ranges = [data_range]
    window_index_range = [index_range]

    while True:
        
        if window == 'last':
            break

        if data_overlap_fraction is None:
            minimum_of_new_data_range = window_data_ranges[-1][1]-data_overlap
        else:
            buffer = max((np.diff(window_data_ranges[-1]).item()*data_overlap_fraction), data_overlap)
            minimum_of_new_data_range = window_data_ranges[-1][1]-buffer

        # get index of minimum/maximum energy resonances
        min_E_resonances = minimum_of_new_data_range - external_resonance_buffer
        i_min_E_resonances = np.searchsorted(internal_resonances, min_E_resonances, 'left')
        i_max_E_resonances = i_min_E_resonances+resonances_per_window

        if i_max_E_resonances >= len(internal_resonances)-1: # check that we are not beyond the internal resonances
            i_max_E_resonances = len(internal_resonances)-1
            window = 'last'
        else: # check that we meet minimum window energy domain
            at_least_max_E_data = window_data_ranges[-1][1] + minimum_step
            at_least_max_E_resonances = at_least_max_E_data + external_resonance_buffer
            if internal_resonances[i_max_E_resonances] < at_least_max_E_resonances:
                i_max_E_resonances = np.searchsorted(internal_resonances, at_least_max_E_resonances, 'left')
                print(f"Increasing data range to minimum step, {i_max_E_resonances-i_min_E_resonances} resonances in this window")
                max_E_data = at_least_max_E_data
            else:
                max_E_data = internal_resonances[i_max_E_resonances]-external_resonance_buffer
        
        if i_max_E_resonances >= len(internal_resonances)-1:
            i_max_E_resonances = len(internal_resonances)-1
            window = 'last'
            data_range = np.array([minimum_of_new_data_range, max(total_data_range)])
            resonances = np.concatenate([internal_resonances[i_min_E_resonances:], upper_external_resonances])
            index_range = np.array([i_min_E_resonances, i_max_E_resonances+len(upper_external_resonances)+1])
        
        else:
            resonances = internal_resonances[i_min_E_resonances:i_max_E_resonances+1]
            index_range = np.array([i_min_E_resonances, i_max_E_resonances+1])
            data_range = np.array([minimum_of_new_data_range, max_E_data])


            
        # if i_max_E_resonances >= len(internal_resonances)-1:
        #     i_max_E_resonances = len(internal_resonances)-1
        #     window = 'last'

        # # if window == 'last':
        #     # data_range[1] = max(total_data_range)
        #     # resonances = np.concatenate([resonances, upper_external_resonances])
        #     data_range = np.array([minimum_of_new_data_range, max(total_data_range)])
        #     resonances = np.concatenate([internal_resonances[i_min_E_resonances:], upper_external_resonances])
        #     index_range = np.array([i_min_E_resonances, i_max_E_resonances+len(upper_external_resonances)])
        #     # index_range = np.array([i_min_E_resonances, len(resonance_energies_start)-1-len(lower_external_resonances)])


        # else:
        #     at_least_max_E_data = window_data_ranges[-1][1] + minimum_step
        #     at_least_max_E_resonances = at_least_max_E_data + external_resonance_buffer

        #     if internal_resonances[i_max_E_resonances] < at_least_max_E_resonances:
        #         i_max_E_resonances = np.searchsorted(internal_resonances, at_least_max_E_resonances, 'left')
        #         print(f"Increasing data range to minimum step, {i_max_E_resonances-i_min_E_resonances} resonances in this window")
        #         max_E_data = at_least_max_E_data
        #     else:
        #         max_E_data = internal_resonances[i_max_E_resonances]-external_resonance_buffer
            
        #     # define resonances and data_range for window
        #     resonances = internal_resonances[i_min_E_resonances:i_max_E_resonances+1]
        #     index_range = np.array([i_min_E_resonances, i_max_E_resonances+1])
        #     data_range = np.array([minimum_of_new_data_range, max_E_data])



        # append window resonances and data range
        window_resonance_energies.append(resonances)
        window_data_ranges.append(data_range)
        window_index_range.append(len(lower_external_resonances) + index_range)
        
    
    return window_resonance_energies, window_data_ranges, window_index_range


def get_window_dataframes(window_resonances, index_ranges, all_resonances):
    window_dataframes = []
    for wr, ir in zip(window_resonances, index_ranges):
        reslad = all_resonances.iloc[ir[0]:ir[1], :]
        assert(np.all(reslad.E.values == wr))
        window_dataframes.append(reslad)
    return window_dataframes

### poor man's parallel over windows
def write_fitpy(basepath, threads=1, fixed_resonance_indices=[]):
    assert isinstance(fixed_resonance_indices, list)

    with open(os.path.join(basepath, "fit.py"), 'w') as f:
        f.write("""import os\n""")
        
        f.write(f"""os.environ['OMP_NUM_THREADS'] = str({threads})\n""")
        f.write(f"""os.environ['OPENBLAS_NUM_THREADS'] = str({threads})\n""")
        f.write(f"""os.environ['MKL_NUM_THREADS'] = str({threads})\n""")
        f.write(f"""os.environ['VECLIB_MAXIMUM_THREADS'] = str({threads})\n""")
        f.write(f"""os.environ['NUMEXPR_NUM_THREADS'] = str({threads})\n""")

        f.write("""from ATARI.utils.atario import load_general_object, save_general_object\n""")
        f.write("""import pandas as pd\n""")
        f.write("""os.chdir(os.path.dirname(__file__))\n""")
        f.write("""autofit = load_general_object("autofit.pkl")\n""")
        f.write("""data = load_general_object("eval_data.pkl")\n""")
        f.write("""df = pd.read_csv("df.csv", index_col=0)\n""")
        f.write(f"""out = autofit.fit(data, df, fixed_resonance_indices={fixed_resonance_indices})\n""")
        f.write("""save_general_object(out, "out.pkl")\n""")
        
    return




def write_submit_wait(basepath, parallel_processes, parallel_windows, total_windows):

    with open(os.path.join(os.path.realpath(basepath), "jr.sh"), 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("#PBS -V\n")
        f.write("#PBS -q fill\n")
        f.write(f"#PBS -l nodes=1:ppn={parallel_processes}\n")
        f.write(f"#PBS -t 0-{total_windows}%{parallel_windows}\n\n")
        f.write("cd ${PBS_O_WORKDIR}\n")
        f.write("source ~/atarienv/bin/activate\n")
        f.write("python3.9 window_${PBS_ARRAYID}/fit.py\n")
    output = subprocess.run(["sh", "-c", f'ssh -tt necluster.ne.utk.edu "cd {os.path.realpath(basepath)} && qsub jr.sh "'], capture_output=True, text=True, timeout=10)

    ### wait for all jobs to complete
    jobID = output.stdout.split('[')[0]
    while True:
        windows_done = []
        for i in range(total_windows):
            # full_path = os.path.realpath(os.path.join(rto.sammy_runDIR, f"window_{imap[0]}_{imap[1]}"))
            path_to_output = os.path.realpath(os.path.join(basepath, f"jr.sh.e{jobID}-{i}"))
            if os.path.isfile(path_to_output):
                windows_done.append(True)
            else:
                windows_done.append(False)
        if np.all(windows_done):
            break
        else:
            print(f"Window jobs are not done, waiting 60s")
            print(windows_done)
            time.sleep(60)

    return 


def get_windows_static(energy_range_total, 
                       resonance_ladder, 
                       window_size = 10,
                       data_overlap_fraction = 0.25,
                       external_resonance_buffer = 2,
                       maxres_per_window = 50
                       ):

    data_overlap = window_size*data_overlap_fraction
    assert data_overlap_fraction < 0.5, "Data overlap > 50% has not been tested"

    Estart = float(min(energy_range_total))
    Eend = Estart + float(window_size)
    external_resonance_buffer = float(external_resonance_buffer)

    data_ranges = []
    data_ranges_overlap = []
    resonance_dataframes = []
    resonance_dataframes_overlap = []
    maxres_bool = []
    overlap_elim_indices = []
    overlap_fixed_indices = []
    while True:
        
        dr = np.array([Estart, Eend])
        dro = np.array([Eend-data_overlap, Eend])
        
        rr = dr + np.array([-external_resonance_buffer, external_resonance_buffer])
        mask = (resonance_ladder.E>rr[0]) & (resonance_ladder.E<rr[1])
        rdf = resonance_ladder.loc[mask].copy()

        rro = dro + np.array([-external_resonance_buffer, external_resonance_buffer])
        masko = (resonance_ladder.E>rro[0]) & (resonance_ladder.E<rro[1])
        rdfo = resonance_ladder.loc[masko].copy()

        overlap_elim_indices.append(((rdfo.E>dro[0]) & (rdfo.E<dro[1])).index[(rdfo.E>dro[0]) & (rdfo.E<dro[1])])
        overlap_fixed_indices.append(((rdfo.E>dro[0]) & (rdfo.E<dro[1])).index[~((rdfo.E>dro[0]) & (rdfo.E<dro[1]))])

        maxres_bool.append(len(rdf)>maxres_per_window)
        resonance_dataframes.append(rdf)
        data_ranges.append(dr)
        if Eend >= max(energy_range_total):
            break
        resonance_dataframes_overlap.append(rdfo)
        data_ranges_overlap.append(dro)
        
        Estart += window_size-data_overlap
        Eend = Estart + window_size

    if np.any(maxres_bool):
        print(f"Efficiency Warning: {np.count(maxres_bool)} windows have more than {maxres_per_window}, consider reducing window size parameters.")
    
    return data_ranges, data_ranges_overlap, resonance_dataframes, resonance_dataframes_overlap, overlap_fixed_indices
