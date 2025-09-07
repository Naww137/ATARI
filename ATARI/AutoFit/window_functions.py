import numpy as np
import pandas as pd
import os
import subprocess
import time
from copy import copy

from ATARI.theory.resonance_statistics import find_external_levels
from ATARI.AutoFit.sammy_interface_bindings import Solver_factory
from ATARI.sammy_interface.sammy_classes import SolverOPTs, SammyRunTimeOptions
from ATARI.utils.datacontainers import Evaluation_Data
from ATARI.AutoFit.spin_shuffling import minimize_spingroup_shuffling

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
        f.write(f"""os.environ['NUMEXPR_NUM_THREADS'] = str({threads})\n\n""")
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





def get_windows_static2(energy_range_total, 
                       resonance_ladder, 
                       window_size = 10,
                       data_overlap_size = 12.5,
                       data_overlap_fraction = 0.25,
                       external_resonance_buffer = 2,
                       maxres_per_window = 50
                       ):

    a = float(window_size)
    b = float(window_size*data_overlap_fraction)
    c = float(external_resonance_buffer)
    d = data_overlap_size

    assert b+ 2*c < a-b-2*c
    assert d > b+ 2*c
    assert d < a-b-2*c

    Estart = float(min(energy_range_total))
    Eend = Estart + a
    c_array = np.array([-c, c])

    data_ranges = []
    data_ranges_overlap = []
    resonance_dataframes = []
    resonance_dataframes_overlap = []
    maxres_bool = []
    overlap_elim_indices = []
    overlap_fixed_indices = []
    while True:
        
        dr = np.array([Estart, Eend]) # range for stage 1
        Estarto = Eend-(d+b)/2
        dro = np.array([Estarto, Estarto+d]) # range for stage 2 (overlap)

        rr = dr + c_array
        mask = (resonance_ladder.E>rr[0]) & (resonance_ladder.E<rr[1])
        rdf = resonance_ladder.loc[mask].copy()

        rro = dro + c_array
        masko = (resonance_ladder.E>rro[0]) & (resonance_ladder.E<rro[1])
        rdfo = resonance_ladder.loc[masko].copy()

        overlap_elim_indices .append(((rdfo.E>dro[0]) & (rdfo.E<dro[1])).index[  (rdfo.E>dro[0]) & (rdfo.E<dro[1]) ])
        overlap_fixed_indices.append(((rdfo.E>dro[0]) & (rdfo.E<dro[1])).index[~((rdfo.E>dro[0]) & (rdfo.E<dro[1]))])

        maxres_bool.append(len(rdf)>maxres_per_window)
        resonance_dataframes.append(rdf)
        data_ranges.append(dr)
        if Eend >= max(energy_range_total):
            break
        resonance_dataframes_overlap.append(rdfo)
        data_ranges_overlap.append(dro)
        
        Estart += window_size-b
        Eend = Estart + window_size

    if np.any(maxres_bool):
        print(f"Efficiency Warning: {maxres_bool} windows have more than {maxres_per_window}, consider reducing window size parameters.")

    return data_ranges, data_ranges_overlap, resonance_dataframes, resonance_dataframes_overlap, overlap_fixed_indices

def get_windows_stage_1(energy_range_total, 
                        resonance_ladder,
                        particle_pair,
                        window_size:float = 10.0,
                        data_overlap_size:float = 12.5,
                        data_overlap_fraction:float = 0.25,
                        external_resonance_buffer:float = 2.0,
                        maxres_per_window:int = 50,
                        add_external_resonances:bool=True
                       ):

    a = float(window_size)
    b = float(window_size*data_overlap_fraction)
    c = float(external_resonance_buffer)
    d = data_overlap_size

    assert b+ 2*c < a-b-2*c
    assert d > b + 2*c # makes sure the stage 2 range encompasses the stage 1 overlap
    assert d < a - b # makes sure that the stage 2 regions do not overlap

    Estart = float(min(energy_range_total))
    c_array = np.array([-c, c])

    data_ranges = []
    param_ranges = []
    resonance_dataframes = []
    fixed_indices = []
    maxres_bool = []
    while True:
        
        Eend = Estart + a
        data_range = np.array([Estart, Eend]) # range for stage 1

        # Getting internal levels:
        data_range_extended = data_range + c_array
        mask = (resonance_ladder.E>data_range_extended[0]) & (resonance_ladder.E<data_range_extended[1])
        res_ladder_internal = resonance_ladder.loc[mask].copy()

        # Add external levels:
        if add_external_resonances:
            ext_res = find_external_levels(particle_pair, data_range, return_reduced=False)
            ext_res['varyE']   = 0
            ext_res['varyGg']  = 0
            ext_res['varyGn1'] = 1
        else:
            ext_res = pd.DataFrame({'E':[], 'Gg':[], 'Gn1':[], 'varyE':[], 'varyGg':[], 'varyGn1':[], 'J_ID':[]})

        # Combining external and internal levels:
        res_ladder_full = pd.concat((ext_res, res_ladder_internal))
        res_ladder_full.reset_index(drop=True, inplace=True)
        # FIXME: THIS IS A PATCHFIX FOR AN UNDERLYING PROBLEM OF UNKNOWN SOURCE:
        res_ladder_full['Gn2'] = 0.0
        res_ladder_full['Gn3'] = 0.0
        res_ladder_full['varyGn2'] = 0.0
        res_ladder_full['varyGn3'] = 0.0
        fixed_indices.append([i for i in range(len(ext_res))])

        # Add windows:
        maxres_bool.append(len(res_ladder_internal)>maxres_per_window)
        resonance_dataframes.append(res_ladder_full)
        data_ranges.append(data_range)
        param_ranges.append(data_range_extended)
        if Eend >= max(energy_range_total):
            break
        
        Estart += window_size-b

    if np.any(maxres_bool):
        print(f"Efficiency Warning: {maxres_bool} windows have more than {maxres_per_window}, consider reducing window size parameters.")

    return data_ranges, param_ranges, resonance_dataframes, fixed_indices

def get_windows_stage_2(resonance_ladders_internal:list, 
                        data_ranges:list,
                        particle_pair,
                        window_size:float = 10.0,
                        data_overlap_size:float = 12.5,
                        data_overlap_fraction:float = 0.25,
                        external_overlap_resonance_buffer:float = 0.0,
                        maxres_per_window:int = 50,
                        add_external_resonances:bool=True
                        ):

    # a = float(window_size)
    b = float(window_size*data_overlap_fraction)
    # c = float(external_resonance_buffer)
    d = data_overlap_size

    # assert b+ 2*c < a-b-2*c
    # assert d > b+ 2*c
    # assert d < a-b-2*c

    data_ranges_overlap = []
    param_ranges_overlap = []
    resonance_dataframes_overlap = []
    overlap_fixed_indices = []
    for i in range(len(resonance_ladders_internal)-1):
        Eend = data_ranges[i][1]
        Estarto = Eend-(d+b)/2
        Eendo = Estarto+d
        data_range_overlap = np.array([Estarto, Eendo]) # range for stage 2 (overlap)
        data_range_overlap_extended_res = np.array([data_ranges[i][0], data_ranges[i+1][-1]])
        data_range_overlap_extended_data = np.array([Estarto-external_overlap_resonance_buffer,Eend+external_overlap_resonance_buffer])

        # Getting combined dataframe:
        resonance_ladder_low  = resonance_ladders_internal[i]
        resonance_ladder_high = resonance_ladders_internal[i+1]
        res_ladder_comb = pd.concat((resonance_ladder_low,resonance_ladder_high), join='outer', ignore_index=True)

        # Getting fixed indices:
        mask_can_vary = (res_ladder_comb['E'] < data_range_overlap[1]) & (res_ladder_comb['E'] > data_range_overlap[0])
        res_ladder_can_vary = res_ladder_comb.loc[mask_can_vary].copy()
        res_ladder_can_vary.sort_values(by='E', inplace=True)
        res_ladder_can_vary.reset_index(drop=True, inplace=True)
        res_ladder_fixed = res_ladder_comb.loc[~mask_can_vary].copy()
        res_ladder_fixed.sort_values(by='E', inplace=True)
        res_ladder_fixed.reset_index(drop=True, inplace=True)
        vary_cols = [col for col in res_ladder_fixed.columns if col.startswith("vary")]
        res_ladder_fixed[vary_cols] = 0

        # Add external levels:
        if add_external_resonances:
            ext_res = find_external_levels(particle_pair, data_range_overlap_extended_res, return_reduced=False)
            ext_res['varyE']   = 0
            ext_res['varyGg']  = 0
            ext_res['varyGn1'] = 1
        else:
            ext_res = pd.DataFrame({'E':[], 'Gg':[], 'Gn1':[], 'varyE':[], 'varyGg':[], 'varyGn1':[], 'J_ID':[]})
        res_ladder_fixed = pd.concat((ext_res, res_ladder_fixed), join='outer', ignore_index=True)

        # Recombining:
        overlap_fixed_indices.append([i for i in range(len(res_ladder_fixed))])
        resonance_dataframe_overlap = pd.concat((res_ladder_fixed, res_ladder_can_vary), join='outer', ignore_index=True)
        # FIXME: THIS IS A PATCHFIX FOR AN UNDERLYING PROBLEM OF UNKNOWN SOURCE:
        resonance_dataframe_overlap['Gn2'] = 0.0
        resonance_dataframe_overlap['Gn3'] = 0.0
        resonance_dataframe_overlap['varyGn2'] = 0.0
        resonance_dataframe_overlap['varyGn3'] = 0.0
        resonance_dataframes_overlap.append(resonance_dataframe_overlap)
        data_ranges_overlap.append(data_range_overlap_extended_data)
        param_ranges_overlap.append(data_range_overlap_extended_res)

    # Finding and stitching all resonances without overlap:
    all_resonances_wo_overlap = pd.concat(resonance_ladders_internal, join='inner', ignore_index=True)
    for i in range(len(resonance_ladders_internal)-1):
        Eend = data_ranges[i][1]
        Estarto = Eend-(d+b)/2
        Eendo = Estarto+d
        mask_outside_overlap = (all_resonances_wo_overlap['E'] < Estarto) | (all_resonances_wo_overlap['E'] > Eendo)
        all_resonances_wo_overlap = all_resonances_wo_overlap.loc[mask_outside_overlap]
    # FIXME: THIS IS A PATCHFIX FOR AN UNDERLYING PROBLEM OF UNKNOWN SOURCE:
    all_resonances_wo_overlap['Gn2'] = 0.0
    all_resonances_wo_overlap['Gn3'] = 0.0
    all_resonances_wo_overlap['varyGn2'] = 0.0
    all_resonances_wo_overlap['varyGn3'] = 0.0

    return data_ranges_overlap, param_ranges_overlap, resonance_dataframes_overlap, overlap_fixed_indices, all_resonances_wo_overlap

# def get_windows_stage_3(energy_range_total:tuple, 
#                         resonance_ladder:pd.DataFrame,
#                         particle_pair,
#                         window_size:float = 10.0, # eV
#                         data_overlap_fraction:float = 0.25,
#                         parameter_buffer:float = 5.0, # eV
#                         data_buffer:float = 2.0, # eV
#                         ):
    
#     assert 0.0 < data_overlap_fraction < 1.0, 'Overlap fraction must be a number between 0 and 1.'
    
#     Estart = energy_range_total[0]
#     data_ranges = []
#     data_ranges_data_buffer = []
#     resonance_dataframes = []
#     fixed_indices = []
#     while True:
#         Eend = Estart + window_size
#         data_range = np.array([Estart, Eend])
#         if Estart == energy_range_total[0]:
#             data_range_param = np.array([Estart, Eend+parameter_buffer])
#             data_range_data_buffer = np.array([Estart, Eend+data_buffer])
#         else:
#             data_range_param = np.array([Estart-parameter_buffer, Eend+parameter_buffer])
#             data_range_data_buffer = np.array([Estart-data_buffer, Eend+data_buffer])
#         data_ranges.append(data_range)
#         data_ranges_data_buffer.append(data_range_data_buffer)

#         mask_param = (resonance_ladder.E>data_range_param[0]) & (resonance_ladder.E<data_range_param[1])
#         res_ladder_internal = resonance_ladder.loc[mask_param].copy()
#         mask_can_vary = (res_ladder_internal.E < data_range[1]) & (res_ladder_internal.E > data_range[0])
#         res_ladder_fixed = res_ladder_internal.loc[~mask_can_vary].copy()
#         res_ladder_fixed.sort_values(by='E', inplace=True)
#         res_ladder_fixed.reset_index(drop=True, inplace=True)
#         vary_cols = [col for col in res_ladder_fixed.columns if col.startswith("vary")]
#         res_ladder_fixed[vary_cols] = 0
#         ext_res = find_external_levels(particle_pair, data_range_param, return_reduced=False)
#         ext_res['varyE']   = 0
#         ext_res['varyGg']  = 0
#         ext_res['varyGn1'] = 1
#         res_ladder_fixed = pd.concat((ext_res, res_ladder_fixed), join='outer', ignore_index=True)
#         res_ladder_full = pd.concat((res_ladder_fixed, res_ladder_internal))
#         res_ladder_full.reset_index(drop=True, inplace=True)
#         # FIXME: THIS IS A PATCHFIX FOR AN UNDERLYING PROBLEM OF UNKNOWN SOURCE:
#         res_ladder_full['Gn2'] = 0.0
#         res_ladder_full['Gn3'] = 0.0
#         res_ladder_full['varyGn2'] = 0.0
#         res_ladder_full['varyGn3'] = 0.0
#         resonance_dataframes.append(res_ladder_full)

#         fixed_indices_window = [i for i in range(len(res_ladder_fixed))]
#         fixed_indices.append(fixed_indices_window)

#         if Eend >= max(energy_range_total):
#             break

#         Estart += window_size * (1.0 - data_overlap_fraction)

#     return data_ranges, data_ranges_data_buffer, resonance_dataframes, fixed_indices

def execute_stage_3(energy_range_total:tuple, 
                    resonance_ladder:pd.DataFrame,
                    sammy_rto:SammyRunTimeOptions,
                    solver_opts:SolverOPTs,
                    particle_pair,
                    eval_data:Evaluation_Data,

                    num_shuffles:int=10,
                    model_selection:str='chi2',

                    window_size:float = 20.0, # eV
                    data_overlap_fraction:float = 0.25,
                    parameter_buffer:float = 5.0, # eV
                    data_buffer:float = 2.0, # eV
                    ):
    
    assert 0.0 < data_overlap_fraction < 1.0, 'Overlap fraction must be a number between 0 and 1.'
    
    Estart = energy_range_total[0]
    full_ladder = copy(resonance_ladder)
    while True:
        Eend = Estart + window_size
        data_range = np.array([Estart, Eend])
        data_range_param = np.array([max(Estart-parameter_buffer,energy_range_total[0]), min(Eend+parameter_buffer,energy_range_total[1])])
        data_range_data_buffer = np.array([max(Estart-data_buffer,energy_range_total[0]), min(Eend+data_buffer,energy_range_total[1])])

        eval_data_trunc = eval_data.truncate(data_range_data_buffer)
        solver = Solver_factory(sammy_rto, solver_opts._solver, solver_opts, particle_pair, eval_data_trunc)
        # solver_no_bayes = copy(solver)
        # solver_no_bayes.set_bayes(False)

        mask_param = (full_ladder.E>data_range_param[0]) & (full_ladder.E<data_range_param[1])
        res_ladder_internal = full_ladder.loc[mask_param].copy()
        mask_can_vary = (res_ladder_internal.E > data_range[0]) & (res_ladder_internal.E < data_range[1])
        res_ladder_can_vary = res_ladder_internal.loc[mask_can_vary].copy()#.reset_index(drop=True)
        res_ladder_fixed = res_ladder_internal.loc[~mask_can_vary].copy()
        # res_ladder_fixed.sort_values(by='E', inplace=True)
        # res_ladder_fixed.reset_index(drop=True, inplace=True)
        vary_cols = [col for col in res_ladder_fixed.columns if col.startswith("vary")]
        res_ladder_fixed[vary_cols] = 0
        # ext_res = find_external_levels(particle_pair, data_range_param, return_reduced=False)
        # ext_res['varyE']   = 0
        # ext_res['varyGg']  = 0
        # ext_res['varyGn1'] = 1
        # res_ladder_fixed = pd.concat((ext_res, res_ladder_fixed), join='outer', ignore_index=True)
        res_ladder_comb = pd.concat((res_ladder_fixed, res_ladder_can_vary))
        # res_ladder_comb.reset_index(drop=True, inplace=True)
        # FIXME: THIS IS A PATCHFIX FOR AN UNDERLYING PROBLEM OF UNKNOWN SOURCE:
        # res_ladder_comb.drop(columns=[] inplace=True)
        res_ladder_comb['Gn2'] = 0.0
        res_ladder_comb['Gn3'] = 0.0
        res_ladder_comb['varyGn2'] = 0.0
        res_ladder_comb['varyGn3'] = 0.0

        # fixed_indices_window = [i for i in range(len(res_ladder_fixed))]
        # varied_indices_window = [i for i in range(len(res_ladder_fixed), len(res_ladder_comb))]
        fixed_spingroup_indices = res_ladder_comb.index[res_ladder_comb.E < data_range[0]]
        
        # samout = solver_no_bayes.fit(res_ladder_comb, fixed_indices_window)
        # print('chi2:', np.sum(samout.chi2)/eval_data_trunc.N)
        # print('N:', eval_data_trunc.N)
        # print('Data Range:', min(data_range_data_buffer), max(data_range_data_buffer), 'eV')
        particle_pair.resonance_ladder = res_ladder_comb
        particle_pair.energy_range = data_range_param
        spin_shuffle_cases = minimize_spingroup_shuffling(res_ladder_comb, solver, num_shuffles=num_shuffles, window_E_bounds=data_range_param, model_selection=model_selection, fixed_resonance_indices=fixed_spingroup_indices, no_shuffle_indices=fixed_spingroup_indices, verbose=True)
        samout_best = None
        obj_best    = np.inf
        for spin_shuffle_case in spin_shuffle_cases:
            if spin_shuffle_case['obj_value'] < obj_best:
                obj_best = spin_shuffle_case['obj_value']
                samout_best = spin_shuffle_case['sammy_out']
        respar_window = samout_best.par_post
        # mask_full = (full_ladder.E>data_range[0]) & (full_ladder.E<data_range[1])
        # non_vary_cols = [col for col in res_ladder_fixed.columns if not col.startswith("vary")]
        print(res_ladder_comb)
        print(respar_window)
        assert len(respar_window) == len(res_ladder_comb)
        full_ladder.loc[res_ladder_can_vary.index,['E','Gg','Gn1','J_ID']] = respar_window.loc[res_ladder_can_vary.index,['E','Gg','Gn1','J_ID']]
        assert len(full_ladder) == len(resonance_ladder)

        # print(full_ladder)

        # Break condition:
        if Eend >= max(energy_range_total):
            break

        Estart += window_size * (1.0 - data_overlap_fraction)

        full_ladder = full_ladder.loc[:,['E','Gg','Gn1','varyE','varyGg','varyGn1','J_ID']]
    return full_ladder

    # return data_ranges, data_ranges_data_buffer, resonance_dataframes, fixed_indices

# def cut_resonances_outside_window(ladder:pd.DataFrame, energy_range:tuple, resolution_broadening, threshold_factor:float=2.0):
#     ladder_cut = copy(ladder)
#     for idx in ladder_cut.index:
#         tot_width = 1e-3 * (2*ladder_cut.loc[idx,'Gn1'] + ladder_cut.loc[idx,'Gg']) # meV -> eV
#         tot_broad_low  = tot_width + resolution_broadening(energy_range[0])
#         tot_broad_high = tot_width + resolution_broadening(energy_range[1])
#         if   ladder_cut.loc[idx,'E'] - threshold_factor * tot_broad_high > energy_range[1]:
#             ladder_cut.drop(index=idx, inplace=True)
#         elif ladder_cut.loc[idx,'E'] + threshold_factor * tot_broad_low  < energy_range[0]:
#             ladder_cut.drop(index=idx, inplace=True)
#     ladder_cut.reset_index()
#     return ladder_cut