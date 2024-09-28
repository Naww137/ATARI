import numpy as np
import os

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




# # Function to set threading limits
# def set_threading_limits(num_threads):
#     os.environ['OMP_NUM_THREADS'] = str(num_threads)
#     os.environ['OPENBLAS_NUM_THREADS'] = str(num_threads)
#     os.environ['MKL_NUM_THREADS'] = str(num_threads)
#     os.environ['VECLIB_MAXIMUM_THREADS'] = str(num_threads)
#     os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
#     print(f"Threading limits set to {num_threads}")

# def unset_threading_limits():
#     os.environ.pop('OMP_NUM_THREADS', None)
#     os.environ.pop('OPENBLAS_NUM_THREADS', None)
#     os.environ.pop('MKL_NUM_THREADS', None)
#     os.environ.pop('VECLIB_MAXIMUM_THREADS', None)
#     os.environ.pop('NUMEXPR_NUM_THREADS', None)
#     print("Threading limits unset")


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
        f.write("""df = pd.read_csv("df.csv")\n""")
        f.write(f"""out = autofit.fit(data, df, fixed_resonance_indices={fixed_resonance_indices})\n""")
        f.write("""save_general_object(out, "out.pkl")\n""")
        
    return
