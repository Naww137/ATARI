### ADDITIONAL FUNCTIONS for ladders characterization or 
### resonances characterization

import numpy as np
import pandas as pd
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.theory import resonance_statistics
from ATARI.sammy_interface.sammy_classes import SammyInputDataYW, SammyRunTimeOptions, SammyOutputData
import os
import pickle

#     calc return aic, aicc, bic, bicc for models that were fittes using WLS
def calc_AIC_AICc_BIC_BICc_by_fit(
        data: np.array, 
        data_unc: np.array,
        fit: np.array,
        ladder_df: pd.DataFrame,
        precalc_chi2: float = 0):
        
    residuals = data - fit

    n = len(data) # length of the dataset
    k = ladder_df.shape[0] * 3 # num of res params

    if (precalc_chi2==0):
        chi2 = np.sum((residuals / data_unc) ** 2) # chi2 (weighted residual sum of squares)
    else:
        print('Warning - using precalculated chi2 value')
        chi2 = precalc_chi2 # chi2 (weighted residual sum of squares)

    chi2_n = chi2 / n

    aic_wls = 2 * k + chi2 # AIC for WLS!

    aicc_wls = aic_wls + 2 * k * (k + 1) / (n - k - 1)

    # BIC for WLS
    bic_wls = k * np.log(n) + chi2
    bicc_wls = bic_wls +  2* k * (k + 1) / (2 * (n - k - 1))
    
    return aic_wls, aicc_wls, bic_wls, bicc_wls, chi2, chi2_n




def extract_jids_and_keys(Ta_Pair: Particle_Pair):
    """
    Extracts unique J_IDs and corresponding keys from Ta_pair.spin_groups.

    :param ta_pair: Object that has a 'spin_groups' attribute as a dictionary.
    :return: Two lists, one of unique J_IDs and one of corresponding keys.
    """
    j_ids = []
    keys = []
    for key, value in Ta_Pair.spin_groups.items():
        j_id = value['J_ID']
        j_ids.append(j_id)
        keys.append(key)
    return j_ids, keys


def calc_Wigner_LL_by_ladder(ladder_df: pd.DataFrame,
                             Ta_pair: Particle_Pair):
    
    """
    Calculating the -LL value for a given ladder,

    using Wigner PDF, defined as:
    wigner_PDF(x, avg_level_spacing)
        x = x/avg_level_spacing
        return (np.pi/2) * x * np.exp(-np.pi*(x**2)/4)

        TODO: why some normalization in a func?

    for each spin group listed in Ta_pair & returning a sum + dict by group
    """

    all_groups_NLLW = {}

    # select all spin groups listed in Ta_Pair
    separate_j_ids, spin_gr_ids = extract_jids_and_keys(Ta_Pair=Ta_pair)

    # Iterating through unique spin groups and their J_IDs
    for j_id, sp_gr_id in zip(separate_j_ids, spin_gr_ids):
        
        # Access the specific spin group using the key
        spin_group_data = Ta_pair.spin_groups[sp_gr_id]

        # taking resonances from ladder only from one spin group by j_id
        spin_gr_res = ladder_df[ladder_df['J_ID']==j_id]
        
        # taking avg dist for current spin group
        avg_dist = spin_group_data['<D>']

        # energy calculations
        E_values = spin_gr_res.E.to_numpy()
        E_values_sorted = np.sort(E_values)
        
        spacings = np.diff(E_values_sorted)
        # print('Spacings')
        # print(spacings)
        wigner_values = resonance_statistics.wigner_PDF(spacings, avg_dist)
        # print('Wigner values')
        # print(wigner_values)

        neg_log_likelihood_gr = -np.sum(np.log(wigner_values))

        # print(f"J_ID: {j_id}, Key: {sp_gr_id}, \n ")
        # print(f"Spin Group info: \n{ spin_group_data }")
        # print(spin_gr_res.shape[0])
        # print(f'\t NLLW: \t {neg_log_likelihood_gr}')

        all_groups_NLLW[sp_gr_id]=neg_log_likelihood_gr

    total_NLLW = sum(all_groups_NLLW.values())

    # print(all_groups_NLLW)
    # print(total_NLLW)

    return total_NLLW, all_groups_NLLW


# calculation of AIC & all parameters of current solution

def characterize_sol(Ta_pair: Particle_Pair,
                     datasets: list,
                     experiments: list,
                     sol: SammyOutputData, # ! chi2 is calculated inside?
                     covariance_data: list =[]
                     ):
    
    output_dict = {}

    # for each datasets if they are separate
    aic = []
    aicc = []
    chi2_stat = []
    bic = []
    bicc = []

    # Variables for aggregated data
    aggregated_exp_data = []
    aggregated_exp_unc = []
    aggregated_fit = []

    # chi2 & AIC calculation based on the datasets & fits
    for index, ds in enumerate(datasets):
        #check data type
        exp = experiments[index]

        if exp.reaction == "transmission":
            model_key = "theo_trans"
        elif exp.reaction == "capture":
            model_key = "theo_xs"
        else:
            raise ValueError()

        aic_wls, aicc_wls, bic_wls, bicc_wls, chi2, chi2_n = calc_AIC_AICc_BIC_BICc_by_fit(
            data = ds.exp, 
            data_unc = ds.exp_unc,
            fit = sol.pw_post[index][model_key],
            ladder_df = sol.par_post,
            precalc_chi2 = sol.chi2_post[index])
        
        # note, if the data about cov is present - take chi2 values from sol (SammyOutputData) object
        if (len(covariance_data)>0):
            print('Using cov data for chi2 calc')
            chi2 = sol.chi2_post[index]
            chi2_n = sol.chi2n_post[index]
        else:
            print('for chi2 calc using diag unc only')
        
        # Aggregate data for each dataset
        aggregated_exp_data.extend(ds.exp)
        aggregated_exp_unc.extend(ds.exp_unc)
        

        aic.append(aic_wls)
        aicc.append(aicc_wls)
        chi2_stat.append(chi2)
        bic.append(bic_wls)
        bicc.append(bicc_wls)

    # for each dataset - separately - wrong!
    output_dict['aic'] = aic
    output_dict['aicc'] = aicc
    output_dict['bic'] = bic
    output_dict['bicc'] = bicc
    output_dict['chi2_stat'] = chi2_stat

    # recalc entire dataset & calc AICc and BICc values for all datasets as one (!)
    if(len(covariance_data)>0 and (len(sol.chi2_post)>0)):

        #chi2 for all datasets
        precalc_chi2_sum = np.sum(sol.chi2_post)

        k = sol.par_post.shape[0] * 3

        n = len(aggregated_exp_data)

        AICc_entire_ds = 2*k + precalc_chi2_sum + 2*k*(k+1)/(n-k-1)
        BIC_entire_ds = k*np.log(n) + precalc_chi2_sum

        output_dict['aicc_entire_ds'] = AICc_entire_ds
        output_dict['bic_entire_ds'] = BIC_entire_ds

    # Wigner - by spingroups
    NLLW, NLLW_gr = calc_Wigner_LL_by_ladder(ladder_df = sol.par_post,
                             Ta_pair = Ta_pair)

    output_dict['NLLW'] = NLLW_gr

    return output_dict


def save_obj_as_pkl(folder_name:str,
                      file_name:str,
                      obj):
    
    try:
        # Ensure the folder exists
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Combine folder name and file name for the full path
        full_path = os.path.join(folder_name, file_name)
        
        # Save the figure
        with open(full_path, 'wb') as file:
            pickle.dump(obj, file)
        
        # Return True indicating success
        return True
        
    except Exception as e:
        print(f"Error saving pickle: {e}")
        return False


def load_obj_from_pkl(folder_name: str, pkl_fname: str):
    
    full_path = os.path.join(folder_name, pkl_fname)
    
    # Check if file exists
    if not os.path.exists(full_path):
        print(f"Error: {full_path} does not exist.")
        return None
    
    try:
        with open(full_path, 'rb') as file:
            loaded_obj = pickle.load(file)
        return loaded_obj
    except Exception as e:
        print(f"An error occurred while loading the pickle file: {e}")
        return None
    

# deleting small resonances limiting up to N based on Gn1
def reduce_ladder(ladder_df: pd.DataFrame, 
                    Gn1_threshold: float, 
                    vary_list: list = [0, 0, 1],
                    N: int = 0,
                    keep_fixed: bool = True,
                    fixed_side_resonances: pd.DataFrame = pd.DataFrame()
                    ) -> pd.DataFrame:
        
    ladder = ladder_df.copy()

    # Identify fixed resonances
    if (keep_fixed):

        if (fixed_side_resonances.shape[0] > 0):

            # Extract energies from fixed_side_resonances
            energies = fixed_side_resonances["E"].tolist()

            # Find those energies in the ladder dataframe
            fixed_resonances = ladder[ladder["E"].isin(energies)]

            print('Found side resonances (fixed):')
            print(fixed_resonances)
            print(f'Keeping them: {keep_fixed}')

            # Check if all energy values were found
            if len(fixed_resonances) != fixed_side_resonances.shape[0]:
                print("** Error: Not enough matching energy values found for fixed resonances in ladder. **")
                print("Missing energy values:", set(energies) - set(fixed_resonances["E"].tolist()))
                raise ValueError
        else:
            print('*'*40)
            print('Provided info on side resonances:')
            print(fixed_side_resonances)
            print()
            print('Provided ladder:')
            print(ladder)
            print('*'*40)
            print()
            fixed_resonances = pd.DataFrame()

            #raise ValueError('Side resonances has different length..')
    else:
        # just using empty df
        fixed_resonances = pd.DataFrame()


    # If N is provided, sort initial ladder by Gn1 and keep top N (excluding fixed resonances)
    if N > 0:
        variable_resonances = ladder.drop(fixed_resonances.index)
        variable_resonances = variable_resonances.sort_values(by=['Gn1'], ascending=False).head(N)
    else:
        # Apply Gn1 threshold filter (excluding fixed resonances)
        variable_resonances = ladder.drop(fixed_resonances.index)
        variable_resonances = variable_resonances[variable_resonances.Gn1 >= Gn1_threshold]

    # Set column values based on function parameters for variable resonances
    variable_resonances = set_varying_fixed_params(ladder_df=variable_resonances, vary_list=vary_list)

    # Combine fixed and variable resonances
    ladder = pd.concat([fixed_resonances, variable_resonances]).drop_duplicates()
    ladder = ladder.sort_values(by=['E'], ascending=True)

    if ladder.shape[0] != ladder_df.shape[0]: 
        print(f'Reduced number of resonances from {ladder_df.shape[0]} to {ladder.shape[0]}')
        print(f'Threshold used: {Gn1_threshold}')
        print(f'Eliminated {ladder_df.shape[0] - ladder.shape[0]} resonances')

    # reindex?
    ladder.reset_index(drop=True, inplace=True) 
    
    return ladder
    


def set_varying_fixed_params(ladder_df: pd.DataFrame,
                        vary_list: list):
    
    # Check if the length of vary_list is correct
    if len(vary_list) != 3:
        raise ValueError("vary_list must contain exactly 3 elements.")

    # Assign values from vary_list to the specified columns
    columns_to_set = ['varyE', 'varyGg', 'varyGn1']
    for col, value in zip(columns_to_set, vary_list):
        ladder_df[col] = value

    return ladder_df



def find_side_res_df(
        initial_sol_df: pd.DataFrame,
        energy_region: list,
        N_res: int = 2
) -> pd.DataFrame:
    """
    Select N_res resonances from the initial DataFrame, with half (rounded down) from the left and half (rounded up) from the right of the given energy_region.
    
    Parameters:
    initial_sol_df (pd.DataFrame): DataFrame containing the initial solutions with an 'E' column for energy.
    energy_region (list): A list with two elements specifying the energy region [min_energy, max_energy].
    N_res (int): Number of resonances to select, default is 2.

    Returns:
    pd.DataFrame: A DataFrame containing the selected resonances.
    """

    left_res_count = N_res // 2
    right_res_count = N_res - left_res_count

    # Select resonances to the left of the energy region
    left_res = initial_sol_df[initial_sol_df['E'] < np.min(energy_region)]
    left_res = left_res.nlargest(left_res_count, 'E')

    # Select resonances to the right of the energy region
    right_res = initial_sol_df[initial_sol_df['E'] > np.max(energy_region)]
    right_res = right_res.nsmallest(right_res_count, 'E')

    # Combine the selected resonances
    selected_res = pd.concat([left_res, right_res])

    return selected_res
