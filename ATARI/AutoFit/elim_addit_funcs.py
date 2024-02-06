### ADDITIONAL FUNCTIONS for ladders characterization or 
### resonances characterization

import numpy as np
import pandas as pd

from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model

from ATARI.theory import resonance_statistics
from ATARI.utils.misc import fine_egrid
from ATARI.theory.resonance_statistics import sample_RRR_levels

from ATARI.theory.scattering_params import FofE_recursive
from ATARI.utils.atario import fill_resonance_ladder

from ATARI.sammy_interface.sammy_classes import SammyInputDataYW, SammyRunTimeOptions, SammyOutputData, SammyInputData
from ATARI.sammy_interface.sammy_functions import run_sammy
from ATARI.sammy_interface import template_creator

#from ATARI.AutoFit.chi2_eliminator_v2 import eliminator_OUTput

import os
import pickle
import copy
import uuid
import glob

from datetime import datetime

from matplotlib.pyplot import *
from matplotlib import pyplot as plt

from scipy.stats import gaussian_kde, norm


def make_ladder_correct_types(ladder_df: pd.DataFrame) -> pd.DataFrame:
    # Dictionary of desired column types
    desired_types = {
        'E': 'float64',
        'Gg': 'float64',
        'Gn1': 'float64',
        #'gnx2': 'float64',
        'J': 'float64',
        'chs': 'float64',
        'lwave': 'float64',
        'J_ID': 'float64',
        'varyE': 'int',
        'varyGg': 'int',
        'varyGn1': 'int'
    }

    # Make a copy of the DataFrame
    output_ladder = ladder_df.copy()

    # Iterate through the desired columns and types
    for column, dtype in desired_types.items():
        if column in output_ladder.columns:
            # Convert column to desired type
            output_ladder[column] = output_ladder[column].astype(dtype, errors='ignore')
        else:
            # Warn if the column is not in the DataFrame
            print(f"Warning: Column '{column}' not found in the DataFrame.")

    return output_ladder

#     calc return aic, aicc, bic, bicc for models that were fittes using WLS
def calc_AIC_AICc_BIC_BICc_by_fit(
        data: np.ndarray, 
        data_unc: np.ndarray,
        fit: np.ndarray,
        ladder_df: pd.DataFrame,
        precalc_chi2: float = 0):
        
    residuals = data - fit

    n = len(data) # length of the dataset
    k = ladder_df.shape[0] * 3 # num of res params +1(?)

    if (precalc_chi2==0):
        chi2 = np.sum((residuals / data_unc) ** 2) # chi2 (weighted residual sum of squares)
    else:
        print('Warning - using precalculated chi2 value')
        chi2 = precalc_chi2 # chi2 (weighted residual sum of squares)

    chi2_n = chi2 / n

    aic_wls = 2 * k + chi2 # AIC for WLS!
    bic_wls = k * np.log(n) + chi2 # BIC for WLS!

    if (n<=k+1):
        error_msg = f'Number of parameters of the model must be << than number of observations! n = {n}, k = {k}'
        print('!!!')
        print('\t', error_msg)
        print()
        #raise ValueError(error_msg)
        
        bicc_wls = np.inf
        aicc_wls = np.inf

    else:
        aicc_wls = aic_wls + 2 * k * (k + 1) / (n - k - 1)
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


def find_spin_group_info(Ta_Pair: Particle_Pair,
                         j_id: float):
    
    """Getting info from Ta_pair spin groups dict"""
    
    for key, value in Ta_Pair.spin_groups.items():
        # Check if the J_ID matches the input J_ID
        if value.get('J_ID') == j_id:
            return key, value
    return None, None


def find_deleted_res_spingroup_J_ID(ladder1, ladder2):

    # Count J_ID occurrences in both DataFrames
    count_df1 = ladder1['J_ID'].value_counts()
    count_df2 = ladder2['J_ID'].value_counts()

    # Find J_IDs with decreased counts
    deleted_counts = count_df1 - count_df2
    deleted_counts = deleted_counts[deleted_counts > 0]

    return deleted_counts


def norm_log_likelihood(y_hat, mean_Y, std_dev_Y):
    """
    Calculate the negative log likelihood for given values from a normal distribution.

    Parameters:
    y_hats (array-like): The observed values for which to calculate the likelihood.
    mean_Y (float): The mean of the normal distribution.
    std_dev_Y (float): The standard deviation of the normal distribution.

    Returns:
    float: The negative log likelihood.
    """
    # Calculate the likelihood for each y_hat
    likelihoods = norm.pdf(y_hat, mean_Y, std_dev_Y)

    # Calculate the log likelihoods (add a small constant to prevent log(0))
    epsilon = 1e-323
    NLL = - np.log(likelihoods + epsilon)
        
    return likelihoods, NLL



def calc_LL_by_ladder(ladder_df: pd.DataFrame,
                             Ta_pair: Particle_Pair,
                             energy_grid: np.array = np.array([])):
    
    """
    Calculating the -LL value for a given ladder, in a given window

    using Wigner PDF, defined as:
    wigner_PDF(x, avg_level_spacing)
        x = x/avg_level_spacing
        return (np.pi/2) * x * np.exp(-np.pi*(x**2)/4)

        TODO: why some normalization in a func?
    
    + using chi2 pdf with all given params in Ta_pair class

    + using normal pdf with all assumed params for a Gn


    for each spin group listed in Ta_pair & returning a sum + dict by group
    NOTE: only inside the provided window
    """

    ladder_df = ladder_df.copy()

    by_groups_NLLW = {}
    by_groups_NLL_PT_Gn = {}
    by_groups_NLL_PT_Gg = {}

    by_groups_NLL_gn_normal_on_reduced_width_amp = {}
    by_groups_NLLW_all_energy = {}

    # calc the gn values for a given ladder
    # reduced_width_amplitude

    # not squarred
    ladder_df['gn'] = ladder_df.apply(lambda row: convert_gn1_by_Gn1(row, Ta_pair), axis=1)
    
    print(ladder_df)

    # select all spin groups listed in Ta_Pair
    separate_j_ids, spin_gr_ids = extract_jids_and_keys(Ta_Pair=Ta_pair)

    # Iterating through unique spin groups and their J_IDs
    for j_id, sp_gr_id in zip(separate_j_ids, spin_gr_ids):
        
        # Access the specific spin group using the key
        spin_group_data = Ta_pair.spin_groups[sp_gr_id]

        # taking resonances from ladder only from one spin group by j_id
        spin_gr_res = ladder_df[ladder_df['J_ID']==j_id]

        if (len(energy_grid)>0):
            # selecting values with energies between min & max in defined window
            spin_gr_res_internal = spin_gr_res[(spin_gr_res['E'] >= np.min(energy_grid)) & (spin_gr_res['E'] <= np.max(energy_grid))]
        
        # taking avgs for the current spin group
        avg_dist = spin_group_data['<D>']
        avg_Gn = spin_group_data['<Gn>']
        avg_Gg = spin_group_data['<Gg>']

        Gn1_values = spin_gr_res_internal.Gn1.to_numpy()
        Gg_values = spin_gr_res_internal.Gg.to_numpy()

        E_values = spin_gr_res_internal.E.to_numpy()
        E_values_sorted = np.sort(E_values)
        spacings = np.diff(E_values_sorted)

        # the same but for all energy we have in our ladder (without limitation)
        gn_values_allenergy = spin_gr_res.gn.to_numpy()

        # calc likellihood in assumption that gn has normal distr..
        std_dev_gn = np.sqrt(avg_Gn)

        likelihoods_gn, NLL_gn = norm_log_likelihood(gn_values_allenergy, 0, std_dev_gn)


        E_values_allenergy = spin_gr_res.E.to_numpy()
        E_values_allenergy_sorted = np.sort(E_values_allenergy)
        spacings_allenergy = np.diff(E_values_allenergy_sorted)

        # for combined ladder (all levels)
        wigner_values_allenergy = resonance_statistics.wigner_PDF(spacings_allenergy, avg_dist)
        neg_LL_W_gr_allenergy = -np.sum(np.log(wigner_values_allenergy))


        # calculating for current sg
        current_sgr_NLL_PT_Gn = - resonance_statistics.width_LL(resonance_widths=Gn1_values,
                                           average_width = avg_Gn,
                                           dof = spin_group_data['n_dof'])
        
        current_sgr_NLL_PT_Gg = - resonance_statistics.width_LL(resonance_widths=Gg_values,
                                           average_width = avg_Gg,
                                           dof = spin_group_data['g_dof'])

        wigner_values = resonance_statistics.wigner_PDF(spacings, avg_dist)
        neg_sgr_NLLW = -np.sum(np.log(wigner_values))

        # print(f"J_ID: {j_id}, Key: {sp_gr_id}, \n ")
        # print(f"Spin Group info: \n{ spin_group_data }")
        # print(spin_gr_res.shape[0])

        by_groups_NLLW[sp_gr_id] = neg_sgr_NLLW

        by_groups_NLL_PT_Gn[sp_gr_id] = current_sgr_NLL_PT_Gn
        by_groups_NLL_PT_Gg[sp_gr_id] = current_sgr_NLL_PT_Gg

        by_groups_NLLW_all_energy[sp_gr_id] = neg_LL_W_gr_allenergy
        by_groups_NLL_gn_normal_on_reduced_width_amp[sp_gr_id] = np.sum(NLL_gn)


    total_NLLW = sum(by_groups_NLLW.values())
    total_NLL_PT_Gn = sum(by_groups_NLL_PT_Gn.values())
    total_NLL_PT_Gg = sum(by_groups_NLL_PT_Gg.values())

    # for all energy region - not filtering any resonances + assuming normal distribution for gn
    total_NLLW_all_energy = sum(by_groups_NLLW_all_energy.values())
    total_NLL_gn_normal_on_reduced_width_amp = sum(by_groups_NLL_gn_normal_on_reduced_width_amp.values())
    # end for all energy region - not filtering any resonances + assuming normal distribution for gn

    # print(all_groups_NLLW)
    # print(total_NLLW)

    result_dict = {
        'total_NLLW': total_NLLW,
        'by_groups_NLLW': by_groups_NLLW,
        'total_NLL_PT_Gg': total_NLL_PT_Gg,
        'by_groups_NLL_PT_Gg': by_groups_NLL_PT_Gg,
        'total_NLL_PT_Gn': total_NLL_PT_Gn,
        'by_groups_NLL_PT_Gn': by_groups_NLL_PT_Gn,

        'total_NLLW_all_energy': total_NLLW_all_energy,
        'by_groups_NLLW_all_energy': by_groups_NLLW_all_energy,

        'total_NLL_gn_normal_on_reduced_width_amp': total_NLL_gn_normal_on_reduced_width_amp,
        'by_groups_NLL_gn_normal_on_reduced_width_amp': by_groups_NLL_gn_normal_on_reduced_width_amp,
        }

    return result_dict


# calculation of AIC & all parameters of current solution

def characterize_sol(Ta_pair: Particle_Pair,
                     datasets: list,
                     experiments: list,
                     sol: SammyOutputData, # ! chi2 is calculated inside?
                     covariance_data: list =[],
                     energy_grid_2_compare_on: np.array = np.array([]),
                     printout: bool  = True
                     ):
    
    output_dict = {}

    # for each datasets if they are separate
    aic = []
    aicc = []
    all_chi2 = []

    bic = []
    bicc = []

    # Variables for aggregated data
    aggregated_exp_data = []
    aggregated_exp_unc = []

    # for e-range and characterization
    e_range = [np.inf, 0]

    e_range[0] = np.min(energy_grid_2_compare_on)
    e_range[1] = np.max(energy_grid_2_compare_on)

    if (printout):
        print(f'Energy grid for analysis: {e_range}')

        


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
        
        all_chi2.append(chi2)

        bic.append(bic_wls)
        bicc.append(bicc_wls)

    # for each dataset - separately - wrong!
    output_dict['aic'] = aic
    output_dict['aicc'] = aicc
    output_dict['bic'] = bic
    output_dict['bicc'] = bicc
    output_dict['chi2'] = all_chi2

    output_dict['chi2_stat'] = np.sum(all_chi2) / ( len(aggregated_exp_data) - len(sol.par_post) * 3)
    output_dict['chi2_stat_ndat'] = np.sum(all_chi2) / ( len(aggregated_exp_data))
    

    # recalc entire dataset & calc AICc and BICc values for all datasets as one (!)
    if(len(covariance_data)>0 and (len(sol.chi2_post)>0)):

        #chi2 for all datasets
        precalc_chi2_sum = np.sum(sol.chi2_post)

        k = sol.par_post.shape[0] * 3  #+ 1 # estimating the variance

        n = len(aggregated_exp_data)

        AICc_entire_ds = 2*k + precalc_chi2_sum + 2*k*(k+1)/(n-k-1)
        BIC_entire_ds = k*np.log(n) + precalc_chi2_sum

        output_dict['aicc_entire_ds'] = AICc_entire_ds
        output_dict['bic_entire_ds'] = BIC_entire_ds

    # Wigner - by spingroups
    LL_dict = calc_LL_by_ladder(ladder_df = sol.par_post,
                             Ta_pair = Ta_pair,
                             energy_grid=energy_grid_2_compare_on)

    output_dict['NLLW'] = LL_dict['by_groups_NLLW']
    output_dict['NLL_PT_Gn1'] = LL_dict['by_groups_NLL_PT_Gn']
    output_dict['NLL_PT_Gg'] = LL_dict['by_groups_NLL_PT_Gg']

    output_dict['NLL_gn_normal_all_energy'] = LL_dict['by_groups_NLL_gn_normal_on_reduced_width_amp']
    output_dict['NLL_W_all_energy'] = LL_dict['by_groups_NLLW_all_energy']

    joint_prob, prob_by_spin_groups, joint_LL = calc_N_res_probability(Ta_pair=Ta_pair,
                                                             e_range = e_range,
                                                             ladder_df=sol.par_post)
    
    output_dict['N_res_prob_by_spingr'] = prob_by_spin_groups
    output_dict['N_res_joint_prob'] = joint_prob
    output_dict['N_res_joint_LL'] = joint_LL

    return output_dict


def calc_N_res_probability(Ta_pair: Particle_Pair,
                           e_range: list,
                           ladder_df: pd.DataFrame,
                           num_samples: int = 10000):
    
    """Calculating the probability of having N resonances in a given energy window"""

    ladder_df = ladder_df.copy()
    # selecting only elements that are inside the energy window
    ladder_df = ladder_df[(ladder_df['E'] >= e_range[0]) & (ladder_df['E'] <= e_range[1])]

    prob_by_spin_groups = {}

    joint_LL  = 0

    # select all spingroups from Ta_pair
    separate_j_ids, spin_gr_ids = extract_jids_and_keys(Ta_Pair = Ta_pair)

    # Iterating through unique spin groups and their J_IDs
    for j_id, sp_gr_id in zip(separate_j_ids, spin_gr_ids):
        
        # Access the specific spin group using the key
        spin_group_data = Ta_pair.spin_groups[sp_gr_id]

        # taking resonances from ladder only from one spin group by j_id
        spin_gr_resonances = ladder_df[ladder_df['J_ID']==j_id]

        curspingroup_num = spin_gr_resonances.shape[0]

        # Calculate the probability for the current spin group
        prob = calculate_probability_N_res(avg_distance=spin_group_data['<D>'],
                                                                    N_res = curspingroup_num , 
                                                                    num_samples = num_samples, 
                                                                    e_range = e_range)
        
        # to avoid problems with small numbers
        prob = max(np.finfo(float).tiny, prob)

        # loglikellihood
        LL = np.log(prob)
        joint_LL += LL

        prob_by_spin_groups[sp_gr_id]  = prob 

        #print(prob_by_spin_groups[sp_gr_id])
    
    # Update the joint probability
    joint_prob = np.exp(joint_LL)  # Multiplying individual probabilities

    # The joint probability is the product of all individual probabilities
    return joint_prob, prob_by_spin_groups, joint_LL



def calculate_probability_N_res(avg_distance, N_res, num_samples, e_range):
    # Constants
    
    ensemble = 'NNE'  # Assuming Wigner distribution (Nearest Neighbor Ensemble)
    
    # Initialize count for the number of times exactly N_res resonances are found in the range
    count_exact_N_res = 0

    # Run simulations
    for _ in range(num_samples):
        levels, _ = sample_RRR_levels(e_range, avg_distance, ensemble)
        # Count levels within the exact window
        levels_in_window = [lvl for lvl in levels if e_range[0] <= lvl <= e_range[1]]
        if len(levels_in_window) == N_res:
            count_exact_N_res += 1

    # Calculate and return the probability
    probability = count_exact_N_res / num_samples
    
    return probability




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


def separate_side_res_from_inwindow_res(ladder_df: pd.DataFrame, 
                                        fixed_side_resonances: pd.DataFrame):
    
    ladder = ladder_df.copy()

    # Extract energies from fixed_side_resonances
    energies = fixed_side_resonances["E"].tolist()

    # Find those energies in the ladder dataframe
    fixed_resonances = ladder[ladder["E"].isin(energies)]

    # print('Found side resonances (fixed):')
    # print(fixed_resonances)

    # print('*'*40)
    # print('Provided info on side resonances:')
    # print(fixed_side_resonances)
    # print()
    # print('Provided ladder:')
    # print(ladder)
    # print('*'*40)
    # print()

    # Check if all energy values were found
    if len(fixed_resonances) != fixed_side_resonances.shape[0]:
        print("** Error: Not enough matching energy values found for fixed resonances in ladder. **")
        print("Missing energy values:", set(energies) - set(fixed_resonances["E"].tolist()))
        raise ValueError

    inwindow_res_df = ladder.drop(fixed_resonances.index)

    side_res_df = fixed_side_resonances

    return inwindow_res_df, side_res_df



def extract_res_var_params(ladder_df: pd.DataFrame, fixed_side_resonances: pd.DataFrame = pd.DataFrame()):
    """
    Extract varying parameters in resonances from ladder dataframe and fixed side resonances.
    Check if all of them have the same values (varying all or only some of them).
    """
    # Create a copy of the ladder dataframe to work with
    ladder = ladder_df.copy()

    # Extract resonances without sides
    res_wo_sides, _ = separate_side_res_from_inwindow_res(ladder_df = ladder, 
                                 fixed_side_resonances = fixed_side_resonances)
    
    # print('Ladder without defined side resonances:')
    # print(res_wo_sides)

    # print('Side resonances:')
    # print(fixed_side_resonances)

    # Columns to check for varying parameters
    columns_to_check = ['varyE', 'varyGg', 'varyGn1']
    side_res_vary_params = []
    main_res_vary_params = []

    # Check if the values in each column are the same for all rows
    for col in columns_to_check:
        if fixed_side_resonances[col].nunique() == 1:
            # All values are the same
            side_res_vary_params.append(fixed_side_resonances[col].iloc[0])
        else:
            # Different values present
            side_res_vary_params.append(np.inf)

        if res_wo_sides[col].nunique() == 1:
            # All values are the same
            main_res_vary_params.append(res_wo_sides[col].iloc[0])
        else:
            # Different values present
            main_res_vary_params.append(np.inf)

    # # Print extracted varying parameters
    # print('Varying parameters for side resonances:')
    # print(side_res_vary_params)

    # print('Varying parameters for main resonances:')
    # print(main_res_vary_params)

    return main_res_vary_params, side_res_vary_params


# deleting small resonances limiting up to N based on Gn1
def reduce_ladder(ladder_df: pd.DataFrame, 
                    Gn1_threshold: float, 
                    vary_list: list = [0],
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
    if (len(vary_list)==3):
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
        # ladder_df[col] = value
        ladder_df.loc[:, col] = value

    return ladder_df


def get_resonances_from_window(input_res_df: pd.DataFrame, 
                               energy_range: list, 
                               N_side_res: int = 0):
    """
    Select all resonances from the given window bounded by energy_range 
    and add N_side_res resonances to the selection - right and left to the window.

    Parameters:
        input_res_df (pd.DataFrame): DataFrame containing the initial solutions with an 'E' column for energy.
        energy_range (list): A list with two elements specifying the energy region [min_energy, max_energy].
        N_side_res (int): Number of additional resonances to select on each side of the energy window, default is 0.

    Returns:
    pd.DataFrame: A DataFrame containing the selected resonances.
    pd.DataFrame: A DataFrame containing the selected side resonances
    """
    
    # Select resonances within the energy range
    window_res = input_res_df[(input_res_df['E'] >= np.min(energy_range)) & (input_res_df['E'] <= np.max(energy_range))]

    # Select N_side_res resonances from left and right side of the energy window
    if N_side_res > 0:
        left_res = input_res_df[input_res_df['E'] < np.min(energy_range)]
        left_res = left_res.nlargest(N_side_res, 'E')

        right_res = input_res_df[input_res_df['E'] > np.max(energy_range)]
        right_res = right_res.nsmallest(N_side_res, 'E')

        # Combine the selected resonances
        side_res = pd.concat([left_res, right_res])

    return window_res, side_res


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


### calc the SSE for one case




### elimination history funcs (parsing, plotting, etc)

def load_all(savefolder: str,
             hist_pkl_name: str,
             dataset_pkl_name: str):

    hist = load_obj_from_pkl(folder_name=savefolder, pkl_fname=hist_pkl_name)
    case_data_loaded = load_obj_from_pkl(folder_name=savefolder, pkl_fname=dataset_pkl_name)

    energy_range = case_data_loaded['energy_range']

    datasets = case_data_loaded['datasets']
    covariance_data = case_data_loaded['covariance_data']
    experiments = case_data_loaded['experiments']
    true_chars = case_data_loaded['true_chars']
    Ta_pair = case_data_loaded['Ta_pair']

    elim_OPTS_used = case_data_loaded['elim_opts']

    return energy_range, datasets, covariance_data, experiments, true_chars, Ta_pair, hist, elim_OPTS_used





# plotting for comparison of models

# add comparison of the solutions


# little bit modified
def plot_datafits(datasets, experiments, 
    fits=[], fits_chi2=[], f_model_name='fit', f_color='',
    priors=[], priors_chi2=[], pr_model_name='prior', pr_color='',
    true=[], true_chi2=[], t_model_name ='true', t_color = '',
    true_pars = pd.DataFrame(), 
    prior_pars = pd.DataFrame(),
    fit_pars = pd.DataFrame(),
    title: str = '',
    show_spingroups: bool = True,
    fig_size : tuple = (12,9)
    ):

    #ioff()

    colors = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    fig, axes = subplots(2,1, figsize=fig_size, sharex=True)

    for i, exp in enumerate(experiments):
        if exp.reaction == "transmission":
            model_key = "theo_trans"
            iax = 0
        elif exp.reaction == "capture":
            model_key = "theo_xs"
            iax = 1
        else:
            raise ValueError()

        axes[iax].errorbar(datasets[i].E, datasets[i].exp, yerr=datasets[i].exp_unc, zorder=0,
                                                fmt='.', color=f'{colors[i]}', alpha=0.5, linewidth=1.0, markersize=4, capsize=1, label=exp.title)
        
        if len(fits) != 0:
            if (len(fits_chi2) != 0):
                fit_label = f'{f_model_name} {exp.title} ({fits_chi2[i]})'
            else:
                fit_label = f'{f_model_name} {exp.title}'
            
            if (len(f_color)==0):
                fit_color = 'red'
            else:
                fit_color = f_color

            axes[iax].plot(fits[i].E, fits[i][model_key], color=fit_color, zorder=1, lw=1.5, label=fit_label) # colors[i]
        
        if len(priors) != 0:
            if (len(priors_chi2) != 0):
                prior_label = f'{pr_model_name} {exp.title} ({priors_chi2[i]})'
            else:
                prior_label = f'{pr_model_name} {exp.title}'

            if (len(f_color)==0):
                prior_color = 'orange'
            else:
                prior_color = pr_color
            
            axes[iax].plot(priors[i].E, priors[i][model_key], '--', color=prior_color, zorder=0, lw=1.5, label=prior_label)
        
        if (len(t_color)==0):
            true_color = 'green'
        else:
            true_color = t_color

        if len(true) != 0:
            if (len(true_chi2) != 0):
                true_label = f'{t_model_name} {exp.title} ({true_chi2[i]})'
            else:
                true_label = f'{t_model_name} {exp.title}'
            
            axes[iax].plot(true[i].E, true[i][model_key], '-', color=true_color, zorder=1, alpha=0.5, lw=1.5, label=true_label)


    # Set the y-axis limits with additional space for text and capture ymax before changing
    
    y_top_padding = 0.1 
    x_offset = 0.05

    ymax_values = [ax.get_ylim()[1] for ax in axes]  # Store original ymax values for each axis
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + y_top_padding)

    font_size = 8
    y_text_shift = 0.01  # Adjust as needed, related to font size
    y_text_positions = [ymax_values[0], ymax_values[1]]

    # show vertical lines for energies
    
    # fits
    for index, res in fit_pars.iterrows():
        res_E = res.E
        # Add vertical lines at the resonance energies to both subplots
        axes[0].axvline(x=res_E, color=fit_color, linestyle='--', linewidth=0.5, alpha=0.3)
        axes[1].axvline(x=res_E, color=fit_color, linestyle='--', linewidth=0.5, alpha=0.3)

        if (show_spingroups):
            # add txt with
            sp_gr_txt = np.round(int(res.J_ID),0)
            
            y_text_position = ymax  # Position the text at the top of the original y-axis limit
            x_text_position = res_E + x_offset
            
            # Show the text to the right of the line
            for i, ax in enumerate(axes):
                y_text_position = ymax_values[i]  # Use original ymax for text position
                #ax.text(x_text_position, y_text_position, str(sp_gr_txt), color='red', verticalalignment='bottom', fontsize=8)

                ax.text(res_E, y_text_positions[i], str(sp_gr_txt), color=fit_color, verticalalignment='bottom', fontsize=font_size)
                y_text_positions[i] -= y_text_shift


    # the same for theoretical positions
    for index, true_res in true_pars.iterrows():

        true_res_energy = true_res.E
        # Add vertical lines at the resonance energies to both subplots
        axes[0].axvline(x=true_res_energy, color=true_color, linestyle='--', linewidth=0.5, alpha=0.7)
        axes[1].axvline(x=true_res_energy, color=true_color, linestyle='--', linewidth=0.5, alpha=0.7)

        if (show_spingroups):
            # add txt with
            sp_gr_txt = np.round(true_res.J_ID,0)
            y_text_position = ymax  # Position the text at the top of the original y-axis limit
            x_text_position = true_res_energy
            
            # Show the text to the right of the line
            for i, ax in enumerate(axes):
                y_text_position = ymax_values[i]  # Use original ymax for text position
                ax.text(x_text_position , y_text_position, str(sp_gr_txt), color=true_color, verticalalignment='bottom', fontsize=8)

    
    for index, res in prior_pars.iterrows():
        # Add vertical lines at the resonance energies to both subplots
        axes[0].axvline(x=res.E, color=prior_color, linestyle='--', linewidth=0.8, alpha=0.5)
        axes[1].axvline(x=res.E, color=prior_color, linestyle='--', linewidth=0.8, alpha=0.5)

        
    axes[0].set_ylabel("T")
    axes[1].set_ylabel(r"$Y_{\gamma}$")

    # set title
    fig.suptitle(title, fontsize=14)
    
    # additional info if present
    add_title = ''
    if (true_pars.shape[0]>0):
        add_title+=''+r'$N_{'+f'{t_model_name}'+'}$ = '+str(true_pars.shape[0])
        
    if (len(true_chi2)>0):
        add_title += ', ' if (len(add_title)>0) else ''
        add_title+='$\sum_{ds}\chi^2$ = '+str(np.round(np.sum(true_chi2),3))

    if (prior_pars.shape[0]>0):
        add_title += ', ' if (len(add_title)>0) else ''
        add_title += r'$N_{'+f'{pr_model_name}'+'}$ = '+str(prior_pars.shape[0])
    if (len(priors_chi2)>0):
        add_title += ', ' if (len(add_title)>0) else ''
        add_title+='$\sum_{ds}\chi^2$ = '+str(np.round(np.sum(priors_chi2),3))

    if (fit_pars.shape[0]>0):
        add_title += ', ' if (len(add_title)>0) else ''
        add_title+=r'$N_{'+f'{f_model_name}'+'}$ = '+str(fit_pars.shape[0])
    if (len(fits_chi2)>0):
        add_title += ', ' if (len(add_title)>0) else ''
        add_title+='$\sum_{ds}\chi^2$ = '+str(np.round(np.sum(fits_chi2),3))
    
    # end additional info if present
    axes[0].set_title(add_title, fontsize=10)
    

    # ### make it pretty
    for ax in axes:
        # ax.set_xlim([200,250])
        # ax.set_ylim([-0.1,1.1])
        ax.legend(fontsize='xx-small', loc='lower right')

    fig.supxlabel('Energy (eV)')
    fig.tight_layout()

    return fig


def get_level_times(allexp_data: dict, 
                 show_keys: list,
                 settings: dict, 
                 fig_size: tuple = (6, 10), 
                 max_level: int = None,
                 title : str = ''):

    times_dict = {}

    for key in show_keys:
        
        value = allexp_data[key]
        cur_hist = value['hist']

        N_res = []
        level_times = []
        total_times = []

        for level in cur_hist.elimination_history.keys():

            numres = cur_hist.elimination_history[level]['selected_ladder_chars'].par_post.shape[0]
            N_res.append(numres)

            cur_level_time = cur_hist.elimination_history[level]['level_time']
            level_times.append(cur_level_time)

            cur_total_time = cur_hist.elimination_history[level]['total_time']
            total_times.append(cur_total_time)

        times_dict[key] = {
            'N_res': N_res,
            'level_times': level_times,
            'total_times': total_times
        }
    
    #now plot and output
    # TODO: plot
        
    return times_dict






# plotting history
def plot_history(allexp_data: dict, 
                 show_keys: list,
                 settings: dict, 
                 fig_size: tuple = (6, 10), 
                 max_level: int = None,
                 title : str = '',
                 folder_to_save: str = 'data/'):

    
    ioff() # turn interactive plotting off

    # Create the figure and axis objects
    fig, (ax1, ax2, ax3, ax4) = subplots(4, 1, figsize=fig_size, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    # Iterate over each key in the specified show_keys list
    for key in show_keys:

        value = allexp_data[key]

        cur_hist = value['hist']

        Ta_pair = value['Ta_pair']

        theo_ladder = value['true_chars'].par_post
        # energy_grid = fine_egrid(energy = np.array([202, 227]))

        # e_range - getting from dataset

        datasets = value['datasets']
        e_range = [np.inf, 0] # min, max
        for index, ds in enumerate(datasets):
            # energy range - select min and max energy from all datasets we have
            e_range[0] = min(ds.E.min(), e_range[0])
            e_range[1] = max(ds.E.max(), e_range[1])
            # trimming for the closest integer value

        # end e_range - getting from dataset
            
        energy_grid = fine_egrid(energy = np.array([e_range[0], e_range[1]]))


        if max_level is None:
            cur_max_level = max(cur_hist.elimination_history.keys())
        else:
            cur_max_level = min(max(cur_hist.elimination_history.keys()), max_level)

        levels = []
        N_ress = []
        chi2_s = []
        SSE_s = []

        joint_prob_s = []

        gl_min_level = np.min(list(cur_hist.elimination_history.keys()))
        gl_max_level = np.max(list(cur_hist.elimination_history.keys()))

        min_level_passed_test = gl_max_level
        num_res_stop = 0 

        for level in cur_hist.elimination_history.keys():

            if level < cur_max_level:
                break  # Skip levels higher than max_level

            # Retrieve data for the current level
            numres = cur_hist.elimination_history[level]['selected_ladder_chars'].par_post.shape[0]
            chi2 = np.sum(cur_hist.elimination_history[level]['selected_ladder_chars'].chi2_post)

            # passed the test?
            # catch where it stopped to show the vertical line
            pass_test =cur_hist.elimination_history[level]['final_model_passed_test']

            if (pass_test and level<min_level_passed_test):
                min_level_passed_test = level
                num_res_stop = numres

            # end catch where it stopped

            levels.append(level)
            N_ress.append(numres)
            chi2_s.append(chi2)

            total_time = cur_hist.elimination_history[level]['total_time']

            # calc SSE
            calc_xs_fig = True

            df_est, df_theo, resid_matrix, SSE_dict, xs_figure = calc_all_SSE_gen_XS_plot(
                    est_ladder = cur_hist.elimination_history[level]['selected_ladder_chars'].par_post,
                    theo_ladder = theo_ladder,
                    Ta_pair = Ta_pair,
                    settings = settings,
                    energy_grid = energy_grid,
                    reactions_SSE = ['capture', 'elastic'],
                    fig_size = (13,8),
                    calc_fig = calc_xs_fig,
                    fig_yscale='linear'
            )

            if (xs_figure):
                xs_figure.savefig(fname=f'{folder_to_save}xs_{key}_step_{level}.png')

            
            total_SSE = np.round(SSE_dict['SSE_sum_normalized_casewise'][0],1)
            SSE_s.append(total_SSE)

            # probability of N_res
            joint_prob, prob_by_spin_groups, joint_LL = calc_N_res_probability(Ta_pair=Ta_pair,
                                                                     e_range = e_range,
                                                                     ladder_df=cur_hist.elimination_history[level]['selected_ladder_chars'].par_post)
            
            joint_prob_s.append(joint_LL)
            

        # Plot the data for the current key
        ax1.plot(N_ress, chi2_s, marker='o', label=f'$\Delta\chi^2$ = {key}, t = {np.round(total_time,1)} s')
        ax1.legend()

        # Calculate differences in chi2 values and plot
        chi2_diffs = np.diff(chi2_s, prepend=chi2_s[0])
        ax2.plot(N_ress, chi2_diffs, marker='o') 

        ax3.plot(N_ress, SSE_s, marker='o', label=f'{SSE_s[-1]}')  
        ax3.legend()

        ax4.plot(N_ress, joint_prob_s, marker = 'o', label=f'{np.round(joint_prob_s[-1],1)}')
        ax4.legend()

        if (num_res_stop>0):
            ax2.axvline(x=num_res_stop, linestyle='--', linewidth=1.0, alpha=0.5)

    # for true data
    # #plotting true chi2 
    if (len(allexp_data[f'{key}']['true_chars'].chi2_post)>0):
        ax1.axhline(y=sum(allexp_data[f'{key}']['true_chars'].chi2), color='g', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.axhline(y=sum(allexp_data[f'{key}']['true_chars'].chi2_post), color='g', linestyle='--', linewidth=1.5, alpha=0.5)
    # plotting true N_res
        
    if (len(theo_ladder)>0):
        ax4.axvline(x=len(theo_ladder), color='g', linestyle='--', linewidth=1.5, alpha=0.5)
        ax3.axvline(x=len(theo_ladder), color='g', linestyle='--', linewidth=1.5, alpha=0.5)
        ax1.axvline(x=len(theo_ladder), color='g', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.axvline(x=len(theo_ladder), color='g', linestyle='--', linewidth=1.5, alpha=0.5)

    
    
    
    # Set labels and other properties
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)

    ax1.set_ylabel('$\chi^2$')
    #ax2.set_xlabel(r'$N_{res}$')
    ax2.set_ylabel('Change in $\chi^2$')
    ax3.set_ylabel('Normalized SSE')
    ax4.set_ylabel('LL(sol_N_res|avg)')

    ax4.set_xlim((gl_min_level, gl_max_level))

    fig.suptitle(title, fontsize=14)

    ax1.set_title('Change in $\chi^2$', fontsize=10)

    ax4.invert_xaxis()
    
    fig.supxlabel(r'$N_{res}$')

    fig.tight_layout()

    # Return the figure object for further manipulation or display
    return fig


### SSE funcs


# calc theo only broadened
def calc_theo_broadened_xs_for_reactions(
        resonance_ladder: pd.DataFrame,
        Ta_pair: Particle_Pair,
        energy_grid: np.array,
        settings: dict,
        reactions: list = ['capture', 'elastic', 'transmission'],
        ):
    
    rundirname = generate_sammy_rundir_uniq_name(path_to_sammy_temps = settings['path_to_SAMMY_temps'], addit_str='calc_xs_theo') 
    # old value was settings['path_to_SAMMY_temps']+'calc_xs_theo/'
    
    df = pd.DataFrame({"E":energy_grid})
    
    for rxn in reactions:
            
        exp_model_theo = Experimental_Model(title = "theo",
                               reaction = rxn,
                               energy_range = [np.min(energy_grid), np.max(energy_grid)],
                               energy_grid = energy_grid,
                               n = (0.017131,0.0),  
                               FP = (100.14,0.0), 
                               burst = (8, 0.0), 
                               temp = (300.0, 0.0),
                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
        )

        exp_model_theo.sammy_inputs['ResFunc'] = ''

        
        rto = SammyRunTimeOptions(
            sammyexe = settings['path_to_SAMMY_exe'],
            options = {"Print"   :   True,
                    "bayes"   :   False,
                    "keep_runDIR"     : settings['keep_runDIR_SAMMY'],
                    "sammy_runDIR": rundirname
                    }
        )

        # Construct the file path..
        # we need to construct it for each reaction type?? 
        # What the difference then for each reaction type?
        template_filename = os.path.join(settings['running_path'], 'theo.inp') # current working directory!!
        
        if not os.path.exists(template_filename):
            template_creator.make_input_template(template_filename, Ta_pair, exp_model_theo, rto)
        
        sammy_INP= SammyInputData(
            particle_pair = Ta_pair,
            resonance_ladder = resonance_ladder,
            template = template_filename,
            experiment = exp_model_theo,
            experimental_data = exp_model_theo.energy_grid,
            experimental_covariance = None,
            energy_grid = exp_model_theo.energy_grid,
            initial_parameter_uncertainty = None
        )
        sammyOUT = run_sammy(sammy_INP, rto)
        
        if rxn == "capture" and resonance_ladder.empty: ### if no resonance parameters - sammy does not return a column for capture theo xs
            sammyOUT.pw["theo_xs"] = np.zeros(len(sammyOUT.pw))
        if rxn == "transmission": 
            df[f'trans_{rxn}'] = sammyOUT.pw.theo_trans
        df[f'xs_{rxn}'] = sammyOUT.pw.theo_xs
    return df


def build_residual_matrix_dict(
        est_par_list : list, 
        true_par_list : list,
        Ta_pair : Particle_Pair, 
        settings: dict,
        energy_grid : np.array,
        reactions: list = ['capture', 'elastic'],
        print_bool=False
        ):
    """Calculating the residuals disctionaries on the fine grid for provided reactions"""

    energy_grid_fine = fine_egrid(energy=energy_grid)

    # print('building residuals matrix')
    # print('Energy grid fine:')
    # print(len(energy_grid_fine))
    # print(energy_grid_fine)

    ResidualMatrixDict = {}

    for rxn in reactions:
        ResidualMatrixDict[rxn] = []

            # initialize residual matrix dict
    ResidualMatrixDict = {}

    for rxn in reactions:
        ResidualMatrixDict[rxn] = []

    # loop over all cases in est_par and true_par lists
    i = 0 
    for est_ladder, theo_ladder in zip(est_par_list, true_par_list):
        
        for rxn in reactions:

            # # calculating the residuals
            theo_pw_df = calc_theo_broadened_xs_for_reactions(
                resonance_ladder = theo_ladder,
                Ta_pair = Ta_pair,
                energy_grid = energy_grid_fine,
                settings = settings,
                reactions = [rxn],
            )

            est_pw_df = calc_theo_broadened_xs_for_reactions(
                resonance_ladder = est_ladder,
                Ta_pair = Ta_pair,
                energy_grid = energy_grid_fine,
                settings = settings,
                reactions = [rxn],
            )

            # now let's just delete 
            E = est_pw_df.E
            assert(np.all(E == theo_pw_df.E))
            residuals = est_pw_df-theo_pw_df
            residuals["E"] = E

            # append reaction residual
            ResidualMatrixDict[rxn].append(list(residuals[f'xs_{rxn}']))

        if print_bool:
            i += 1
            print(f"Completed Job: {i}")
       
    # convert to numpy array
    for rxn in reactions:
        ResidualMatrixDict[rxn] = np.array(ResidualMatrixDict[rxn])

    return ResidualMatrixDict


# calculating SSE for all and by case
def calculate_SSE_by_cases(ResidualMatrixDict):
    """
    Calculating SSE by case & reaction type.
    & for all reactions
    """
    SSE = {'SSE': {},
           'SSE_normalized':{},
           'SSE_sum_casewise': [],
           'SSE_sum_normalized_casewise': []
    }

    # to normalize calculated SSE (sum for all reactions) by sum of dataset sizes - to get avg SSE_per case per point in energy
    overall_sum = 0
    overall_size = 0

    # Calculate sum of sq. residuals each case for every reaction type
    for rxn in ResidualMatrixDict.keys():

        R = ResidualMatrixDict[rxn]
        SSE['SSE'][rxn] = {}
        SSE['SSE_normalized'][rxn] = {}

        SSE['SSE'][rxn]['casewise'] = [np.sum(R[i]**2) for i in range(R.shape[0])] # for each case by react type

        SSE['SSE_normalized'][rxn]['casewise'] = [np.sum(R[i]**2)/R[i].size for i in range(R.shape[0])] # for each case by react type
        
        SSE['SSE'][rxn]['for_all_cases'] = np.sum(R**2)  # for all cases by react type

        SSE['SSE_normalized'][rxn]['for_all_cases'] = np.sum(R**2)/R.size  # for all cases by react type

        overall_sum += SSE['SSE'][rxn]['for_all_cases']
        overall_size += R.size

    overall_sum_div_size = overall_sum / overall_size
    
    SSE['SSE_normalized']['sum'] = overall_sum_div_size
    SSE['SSE_normalized']['size'] = overall_size

    # calculating for all reactions but per case!
    
    for i in range(R.shape[0]):

        total_case_SSE, total_case_SSE_normalized  = 0, 0
        size_to_norm = 0

        for rxn in ResidualMatrixDict.keys():
            total_case_SSE += SSE['SSE'][rxn]['casewise'][i]

            size_to_norm += ResidualMatrixDict[rxn][i].shape[0]
        
        total_case_SSE_normalized = total_case_SSE / size_to_norm

        SSE['SSE_sum_casewise'].append(total_case_SSE)
        SSE['SSE_sum_normalized_casewise'].append(total_case_SSE_normalized)

    return SSE


# PLOTTING ALL cs calculated data and comparing it to some assumed true sol?

def plot_xs_differences(cand_sol : list,
                        cand_sol_names : list,
                        cand_sol_colors : list,
                        Ta_pair : Particle_Pair,
                        energy_grid : np.array,
                        settings : dict,
                        reactions : list = ['transmission', 'capture', 'elastic'],

                        addit_str : str = ''
                        ):
    """
    given a true solution and a list of candidate solutions - output each type of cs or transmission 
    for a specific reaction to compare on corresponding plots

    """
    cs_calc_res_dfs_list = []

    for sol in cand_sol:
        
        cur_cs_df = calc_theo_broadened_xs_for_reactions(
            resonance_ladder = sol,
            Ta_pair = Ta_pair,
            energy_grid = energy_grid,
            settings = settings,
            reactions = reactions,
        )
        
        # Identifying columns starting with "xs_"
        #xs_columns = [col for col in cur_cs_df.columns if (col.startswith('xs_') or col.startswith('trans_'))]
        xs_columns = [col for col in cur_cs_df.columns if (col.startswith('xs_'))]

        cs_calc_res_dfs_list.append(cur_cs_df)

        
    # Counting the number of xs_ columns to create subplots
    num_xs_columns = len(xs_columns)

    # Creating subplots
    fig_all_reacts, axs = subplots(num_xs_columns, 1, figsize=(8, 3 * num_xs_columns))

    if num_xs_columns == 1:
        axs = [axs]

    # Looping through each column to plot
    for idx, column in enumerate(xs_columns):

        for index_sol, sol in enumerate(cs_calc_res_dfs_list):
            
            axs[idx].plot(sol["E"], sol[column], label=f'{column} {cand_sol_names[index_sol]}')

            # vertical lines where we assume res.
            for index_res, res in cand_sol[index_sol].iterrows():
                axs[idx].axvline(x=res.E, linestyle='--', linewidth=0.5, alpha=0.5, color = cand_sol_colors[index_sol])
        
        # fill between if we have 2 sol in a list
        if len(cs_calc_res_dfs_list) == 2:
            axs[idx].fill_between(cs_calc_res_dfs_list[0].E, 
                                  cs_calc_res_dfs_list[0][column], 
                                  cs_calc_res_dfs_list[1][column], 
                                  color='red', 
                                  alpha=0.5,
                                  label = 'diff')
        # end fill between two lines

        axs[idx].set_title(column)
        axs[idx].set_xlabel("E")
        axs[idx].set_ylabel(column)
        axs[idx].legend(loc='right')
    
    
    # Adding a title for the entire figure
    fig_all_reacts.suptitle(f"Cross-section & Transm. Comparison {addit_str}", fontsize=14)

    tight_layout()

    return fig_all_reacts


def calc_all_SSE_gen_XS_plot(
        est_ladder: pd.DataFrame,
        theo_ladder: pd.DataFrame,
        Ta_pair: Particle_Pair,
        settings: dict,
        energy_grid: np.array,
        reactions_SSE : list = ['capture', 'elastic'],
        fig_size: tuple = (8,6),
        calc_fig: bool = False,
        fig_yscale: str = 'log'
):
    # print('Input Energy grid')
    # print(len(energy_grid))
    # print(energy_grid)
    
    resid_matrix = build_residual_matrix_dict(
        est_par_list = [est_ladder], 
        true_par_list = [theo_ladder], # [jeff_parameters], 
        Ta_pair = Ta_pair, 
        settings = settings,
        energy_grid = energy_grid,
        reactions = reactions_SSE, 
        print_bool = False
        )
    
    SSE_dict = calculate_SSE_by_cases(ResidualMatrixDict = resid_matrix)
    
    # plotting 
    if (calc_fig):
    
        df_est = calc_theo_broadened_xs_for_reactions(
            resonance_ladder = est_ladder,
            Ta_pair = Ta_pair,
            energy_grid = energy_grid,
            settings = settings,
            reactions = ['transmission']
            )
        
        df_theo = calc_theo_broadened_xs_for_reactions(
            resonance_ladder = theo_ladder, # [jeff_parameters], ,
            Ta_pair = Ta_pair,
            energy_grid = energy_grid,
            settings = settings,
            reactions = ['transmission'],
            )
        
        # generating the plot to output
        #ioff()
        figure = plt.figure(figsize = fig_size)
        
        # # Total
        plt.plot(df_est.E, df_est.xs_transmission, label=f'Fit result (N={len(est_ladder)})', color = 'b', alpha=1.0, linewidth=1.0)
        plt.plot(df_theo.E, df_theo. xs_transmission, label=f'Theo (N={len(theo_ladder)})', color = 'r', alpha=1.0, linewidth=1.0)
        
        # just fill in area between two xs
        plt.fill_between(df_est.E, df_est.xs_transmission, df_theo.xs_transmission, 
                    color='darkorange', alpha=0.5, 
                    label=' $SSE_{W}$ = '+ str(np.round(SSE_dict['SSE_sum_normalized_casewise'][0], 2))
                    )
        
        # Add vertical dashed lines for each E value in ladder_df
        for energy in est_ladder.E:
            plt.axvline(x=energy, color='b', linestyle='--', linewidth=0.5, alpha=0.9)

            # Add vertical dashed lines for each E value in ladder_df
        for energy in theo_ladder.E:
            plt.axvline(x=energy, color='r', linestyle='--', linewidth=0.5, alpha=0.9)

        plt.xlabel('Energy, eV')  # Replace with your actual x-axis label
        plt.ylabel('$\sigma_{t}$, barns')  # Replace with your actual y-axis label
        
        # plt.title('')
        figure.suptitle('Fitting results, Cross-section Comparison (Exp. corr.: Doppler Broadening only)')

        plt.legend(loc='upper right')
        plt.grid(color='grey', linestyle='--', linewidth=0.1)
        
        plt.yscale(fig_yscale)
        # plt.xlim(208.5,216)
        # plt.ylim(6,600)
        figure.tight_layout()
    
    
    else:
        figure = False
        df_est = pd.DataFrame()
        df_theo = pd.DataFrame()


    return df_est, df_theo, resid_matrix, SSE_dict, figure



def calc_strength_functions(
        theoretical_df, 
        estimated_df, 
        energy_range, 
        fig_size=(8, 6), 
        create_fig=True):

    
    # Filter dataframes based on the energy range

    filt_theoretical_df = theoretical_df[(theoretical_df['E'] >= energy_range[0]) & (theoretical_df['E'] <= energy_range[1])]
    filt_estimated_df = estimated_df[(estimated_df['E'] >= energy_range[0]) & (estimated_df['E'] <= energy_range[1])]

    # Combine and sort energy values from both datasets
    #combined_energy = np.sort(np.unique(np.concatenate((filt_theoretical_df['E'], filt_estimated_df['E']))))
    combined_energy = fine_egrid(energy_range)

    # Initialize the cumulative sum arrays with zero at the beginning
    cumulative_theo_gn1 = np.zeros(len(combined_energy))
    cumulative_est_gn1 = np.zeros(len(combined_energy))
    cumulative_theo_gg = np.zeros(len(combined_energy))
    cumulative_est_gg = np.zeros(len(combined_energy))

    # Create step functions (ladders) for cumulative sums
    for i, e in enumerate(combined_energy):
        # if i == 0:
        #     continue
      
        cumulative_theo_gn1[i] = filt_theoretical_df[filt_theoretical_df['E'] <= e]['Gn1'].sum()
        cumulative_est_gn1[i] = filt_estimated_df[filt_estimated_df['E'] <= e]['Gn1'].sum()

        cumulative_theo_gg[i] = filt_theoretical_df[filt_theoretical_df['E'] <= e]['Gg'].sum()
        cumulative_est_gg[i] = filt_estimated_df[filt_estimated_df['E'] <= e]['Gg'].sum()

    # Calculate squared differences and their integral

    SSE_Gn1 = np.trapz((cumulative_theo_gn1 - cumulative_est_gn1) ** 2, combined_energy) / (np.max(combined_energy) - np.min(combined_energy))
    SSE_Gg = np.trapz((cumulative_theo_gg - cumulative_est_gg) ** 2, combined_energy) / (np.max(combined_energy) - np.min(combined_energy))

    if create_fig:
        
        fig, axs = plt.subplots(2, 1, figsize=fig_size)

        # Plotting for Gn1
        axs[0].plot(combined_energy, cumulative_theo_gn1, label=f'Theo (N = {filt_theoretical_df.shape[0]})', color='blue')
        axs[0].plot(combined_energy, cumulative_est_gn1, label=f'Est. (N = {filt_estimated_df.shape[0]})', color='red')
        label = r'$\frac{1}{E_{2}-E_{1}} \int_{E_{1}}^{E_{2}} {(\Gamma_{true}-\Gamma_{est})^2} dE $ = '+ f'{SSE_Gn1:.2f}'
        axs[0].fill_between(combined_energy, cumulative_theo_gn1, cumulative_est_gn1, color='purple', alpha=0.3, label = label)
        axs[0].set_xlabel('Energy (E)')
        axs[0].set_ylabel('Cumul. SF for $\Gamma_{n1}$')
        axs[0].legend(loc='upper left')
        axs[0].grid(True)

        # Plotting for Gg
        axs[1].plot(combined_energy, cumulative_theo_gg, label='Theo', color='blue')
        axs[1].plot(combined_energy, cumulative_est_gg, label='Est.', color='red')
        label = r'$\frac{1}{E_{2}-E_{1}} \int_{E_{1}}^{E_{2}} {(\Gamma_{true}-\Gamma_{est})^2} dE $ = '+ f'{SSE_Gg:.2f}'

        axs[1].fill_between(combined_energy, cumulative_theo_gg, cumulative_est_gg, color='purple', alpha=0.3, label = label)
        axs[1].set_xlabel('Energy (E)')
        axs[1].set_ylabel('Cumul. SF for $\Gamma_{\gamma}$')
        axs[1].legend(loc='upper left')
        axs[1].grid(True)

        # add vertical lines for both - dashed, 0.5 linewidth
        for energy in filt_theoretical_df.E:
            axs[0].axvline(x=energy, color='b', linestyle='--', linewidth=0.5, alpha=0.9)
            axs[1].axvline(x=energy, color='b', linestyle='--', linewidth=0.5, alpha=0.9)
        
        for energy in filt_estimated_df.E:
            axs[0].axvline(x=energy, color='r', linestyle='--', linewidth=0.5, alpha=0.9)
            axs[1].axvline(x=energy, color='r', linestyle='--', linewidth=0.5, alpha=0.9)
        # end add vertical lines for both

        plt.tight_layout()

        return SSE_Gg, SSE_Gn1, fig
    else:
        return SSE_Gg, SSE_Gn1, None








def format_time_2_str(time_interval):
    """Reformat time interval in sec to present it in a more nice format for vieweing in
    days, hours, minutes, seconds
    
    Parameters
    ----------
    time_interval : float
        value of time interval in seconds
    ----------

    Outputs
    ----------
    time_components_list : list
        list with time components [days, hours, minutes, seconds]
    formatted_string : str
        string representation of time components but each component is added only if it's > 0
        example: 1 d, 2 h, 12 sec
    ----------
    """

    # Calculate days, hours, minutes, and seconds
    days, remainder = divmod(time_interval, 86400)  # 86400 seconds in a day
    hours, remainder = divmod(remainder, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(remainder, 60)  # 60 seconds in a minute
    
    # Prepare the components for the list
    time_components_list = [int(days), int(hours), int(minutes), int(seconds)]
    
    # Prepare the string components, only adding if value is > 0
    string_components = []
    if days > 0:
        string_components.append(f"{int(days)} d")
    if hours > 0:
        string_components.append(f"{int(hours)} h")
    if minutes > 0:
        string_components.append(f"{int(minutes)} min")
    if seconds > 0:  # Assuming you want to show seconds even if it's 0
        string_components.append(f"{int(seconds)} sec")
    
    # Join the string components
    formatted_string = ", ".join(string_components)

    if (sum(time_components_list)==0):
        formatted_string = '0 sec'
    
    return time_components_list, formatted_string



def update_u_by_p(ladder_full, Ta_pair):
    """
    Recalculating values of u using p in ladder as input.
    """
    # print(ladder_full)
    # print(ladder_full.dtypes)
    
    # ladder_full = fill_resonance_ladder(resonance_ladder=ladder_full,
    #                       particle_pair=Ta_pair,
    #                       J=ladder_full.J,
    #                                                 chs=1,
    #                                                 lwave=0,
    #                                                 J_ID= ladder_full.J_ID )

    # e lambda
    
    #ladder_full['u_e'] = np.sqrt(ladder_full['E'])

    def compute_u_e(row):
        if (row['E'] < 0):
            print(f'Warning! Negative E: {row["E"]}')
            return  - np.sqrt(-row['E'])
        else:
            return  np.sqrt(row['E'])

    ladder_full['u_e'] = ladder_full.apply(compute_u_e, axis=1)
    
    
    # G_gamma
    #ladder_full['u_g'] = np.sqrt(ladder_full['Gg'].values * 1e-3 / 2)

    def compute_u_g(row):
        if (row['Gg'] < 0):
            print(f'Warning! Negative Gg: {row["Gg"]} @ {row["E"]} eV')
            return  - np.sqrt(-row['Gg'] * 1e-3 / 2)
        else:
            return  np.sqrt(row['Gg'] * 1e-3 / 2)

    ladder_full['u_g'] = ladder_full.apply(compute_u_g, axis=1)

    # Gn1 - this requires a more complex calculation
    def compute_u_n(row):
        
        # S_array, P_array, arcphi_array, k = FofE_recursive([row.E], Ta_pair.ac, Ta_pair.M, Ta_pair.m, row.lwave)
        S_array, P_array, arcphi_array, k = FofE_recursive([row.E], Ta_pair.ac, Ta_pair.M, Ta_pair.m, 0)
        Pl = np.sum(P_array)
        if (row['Gn1'] < 0):
            print(f'Warning! Negative Gn: {row["Gn1"]} @ {row["E"]}')
            return  - np.sqrt(-row['Gn1'] / 1000 / (2 * Pl))
        else:
            return  np.sqrt(row['Gn1'] / 1000 / (2 * Pl))
        
    # Use apply with axis=1 to apply function to each row
    ladder_full['u_n1'] = ladder_full.apply(compute_u_n, axis=1)

    return ladder_full



def norm_log_likelihood(y_hat, mean_Y, std_dev_Y):
    """
    Calculate the negative log likelihood for given values from a normal distribution.

    Parameters:
    y_hats (array-like): The observed values for which to calculate the likelihood.
    mean_Y (float): The mean of the normal distribution.
    std_dev_Y (float): The standard deviation of the normal distribution.

    Returns:
    float: The negative log likelihood.
    """
    # Calculate the likelihood for each y_hat
    likelihoods = norm.pdf(y_hat, mean_Y, std_dev_Y)

    # Calculate the log likelihoods (add a small constant to prevent log(0))
    epsilon = 1e-323
    NLL = - np.log(likelihoods + epsilon)
        
    return likelihoods, NLL





def create_solution_comparison_table_from_sol_list():
    """taking list of ladders and producing comparison table """

    return None


# 
# function to update 'Gn1' using recalculation of the most likely gnx2 taken from the mean params of 
# the chi2 distribution for reduced with amplitude = gamma
# gamma^2  has chi2 distr. with 1 dof,
# TODO: check on units!!

def update_Gn1_using_assumed_mode_gnx2(row, Ta_Pair):
    # get the spin group id from the row
    j_id = row['J_ID']

    spin_group_key, current_spin_group_info = find_spin_group_info(Ta_Pair = Ta_Pair, 
                                                                   j_id = j_id)
    
    # mode
    likely_value_gnx2 = max(current_spin_group_info["<Gn>"] - 2, 0)
            
    _, P_array, _, _ = FofE_recursive([row['E']], Ta_Pair.ac, Ta_Pair.M, Ta_Pair.m, 0)
    Pl = np.sum(P_array)
    
    New_GN = 2 * Pl * likely_value_gnx2

    return New_GN



def convert_gn1_by_Gn1(row, Ta_Pair):
         
    _, P_array, _, _ = FofE_recursive([row['E']], Ta_Pair.ac, Ta_Pair.M, Ta_Pair.m, 0)
    Pl = np.sum(P_array)
    
    # mode
    if (row['Gn1']>=0):
        gn = np.sqrt( row['Gn1'] / 2 * Pl ) # TODO: do we need to divide by 1000 here??
    else:
        gn = - np.sqrt( row['Gn1'] / 2 * Pl ) # TODO: do we need to divide by 1000 here??

    return gn


def produce_art_ladder(input_ladder: pd.DataFrame,
                       res_to_add: pd.Series,
                       energy_range: list,
                       Ta_Pair: Particle_Pair):
    
    """Produces ladder based on Ta_pair and average parameters and given ladder with """

    art_res = input_ladder.copy()
    # emptying this dataframe but keeping the structure & columns
    art_res = art_res.iloc[0:0]
    #print(art_res)

    varyE, varyGg, varyGn1 = 0, 0, 0

    # shift in energy
    delta_E = np.max(energy_range) + 5

    for j_id, count in res_to_add.items():

        print(f"J_ID {j_id} was deleted {count} times")

        # adding {count} resonances to art_res with the corresponding J_ID value
        # columns to fill: E - starting from given delta_E + <D> for current spin group,  
        # Gn1, Gg - use average values for all

        spin_group_key, current_spin_group_info = find_spin_group_info(Ta_Pair = Ta_Pair,
                         j_id = j_id)
        #current_spin_group_info = Ta_Pair.spin_groups[spin_group_key]

        # print(current_spin_group_info)
        
        Gg = np.repeat(current_spin_group_info["<Gg>"], count)

        current_Gn = current_spin_group_info["<Gn>"]

        Gn = np.repeat(current_Gn, count)
        Er = np.linspace(start = delta_E, 
                         stop = delta_E + (count - 1) * np.sqrt(2/np.pi) * current_spin_group_info['<D>'],
                         num = count)
        J_ID = np.repeat(current_spin_group_info["J_ID"], count)
        # combine it into a dataframe

        sg_dataframe = pd.DataFrame(
            {"E":Er, 
             "Gg":Gg, 
             "Gn1":Gn, 
             "varyE":np.ones(len(Er))*varyE, 
             "varyGg":np.ones(len(Er))*varyGg, 
             "varyGn1":np.ones(len(Er))*varyGn1 ,
             "J_ID":J_ID}
             )
        
        # update Gn1 everywhere using for each row of sg_dataframe
        sg_dataframe['Gn1'] = sg_dataframe.apply(lambda row: update_Gn1_using_assumed_mode_gnx2(row, Ta_Pair), axis=1)
        sg_dataframe['Gn1'] = 0
        
        art_res = pd.concat([art_res, sg_dataframe], ignore_index = True)

        # del all NANs
        art_res.fillna(0.0, inplace=True)

    # print(art_res)
    # print()

    return art_res

    



def create_solutions_comparison_table_from_hist(hist,
                                                Ta_pair: Particle_Pair,
                     datasets: list,
                     experiments: list,
                     covariance_data: list =[],
                     true_chars: SammyOutputData = None,
                     settings: dict = {},
                     energy_grid_2_compare_on: np.array = np.array([])
                    ):

    # table for analysis of the models - produce chi2
    hist = copy.deepcopy(hist)

    # sums - for all datasets
    is_true = []
    elaps_time = []

    SSE_s = []
    NLLW_s = []
    NLL_Gn1 = []
    NLL_Gg = []

    chi2_s = []
    chi2_stat_s_npar = []
    chi2_stat_s_ndat = []

    OF_alt1 = []
    OF_alt2 = []
    OF_alt3 = []

    OF_alt4 = [] # function values that are calculated by artificial ladder (remaining resonances + added ones)
    sum_NLLW_al = []
    sum_NLL_gn_normal = []

    # SF
    SF_Gn1 = []
    SF_Gg = []
    # end SF


    aicc_s = []
    bicc_s = []
    N_res_s = []
    N_res_art_s = []

    test_pass = []

    N_res_joint_LL = []

    # # incorporate true sol
    true_sol_key = 0

    if ((true_sol_key not in hist.elimination_history.keys()) and true_chars is not None):
        hist.elimination_history[true_sol_key] = {

                'input_ladder' : true_chars.par_post, #? questionable
                'selected_ladder_chars': true_chars, # solution with min chi2 on this level
                'any_model_passed_test': True, # any model on this level of N-1 res..
                'final_model_passed_test': True,
                'level_time': np.inf,
                'total_time': np.inf,
                # adding a mark that this is a true solution
                'assumed_true': True
            }

    # # end incorporate true sol
        
    max_level = np.max(list(hist.elimination_history.keys()))
    previous_ladder = hist.elimination_history[max_level ]['selected_ladder_chars'].par_post
    combined_ladder = previous_ladder

    for level in hist.elimination_history.keys():
        
        # # adding a mark that this is a true solution
        if (level != true_sol_key):
            # adding a mark that this is a true solution
            hist.elimination_history[level]['assumed_true'] = False

        if (level == max_level or level == true_sol_key):
            """This is the initial number of resonances"""
            N_res_initial = hist.elimination_history[level]['selected_ladder_chars'].par.shape[0]
            # empty dataframe
            arificial_res = pd.DataFrame()
        else:
            # produce artificial resonances

            # get the difference with previous remaining resonances - which spin group res. was deleted?
            # compare current ladder vs. previous

            # remaining resonances
            current_ladder = hist.elimination_history[level]['selected_ladder_chars'].par_post

            # we got count of deleted resonances by each J_ID
            deleted_res = find_deleted_res_spingroup_J_ID(ladder1 = previous_ladder, ladder2 = current_ladder)
            
            # print('Deleted count resonances by each group:')
            # print(deleted_res)

            # for j_id, count in deleted_res.items():
            #     print(f"J_ID {j_id} was deleted {count} times")

            art_ladder = produce_art_ladder(input_ladder = current_ladder,
                                            res_to_add = deleted_res,
                                            energy_range = energy_grid_2_compare_on,
                                            Ta_Pair = Ta_pair)
            
            print()
            print('Artificial ladder:')
            print(art_ladder)
            print()

            # adding artificial resonance 
            print()
            combined_ladder = pd.concat([current_ladder, art_ladder], ignore_index=True)
            print('Combined ladder:')
            print(combined_ladder)
            print()


        # # update prev. ladder - to track changes in spin groups
        # previous_ladder = hist.elimination_history[level]['selected_ladder_chars'].par_post

        numres = hist.elimination_history[level]['selected_ladder_chars'].par_post.shape[0]

        numres_combined = combined_ladder.shape[0]

        time_elaps = hist.elimination_history[level]['total_time']

        pass_test = hist.elimination_history[level]['final_model_passed_test']

        is_true.append(hist.elimination_history[level]['assumed_true'])

        elaps_time.append(time_elaps)

        # SSE & strength funcs?
        if (true_chars is not None):
            
            df_est, df_theo, resid_matrix, SSE_dict, xs_figure = calc_all_SSE_gen_XS_plot(
                        est_ladder = hist.elimination_history[level]['selected_ladder_chars'].par_post,
                        theo_ladder = true_chars.par_post,
                        Ta_pair = Ta_pair,
                        settings = settings,
                        energy_grid = energy_grid_2_compare_on,
                        reactions_SSE = ['capture', 'elastic'],
                        fig_size = (13,8),
                        calc_fig = False,
                        fig_yscale='linear'
                )
            
            SSE_s.append(SSE_dict['SSE_sum_normalized_casewise'][0])

            # TODO: strength funcs calculation & fig production
            SSE_Gg, SSE_Gn1, SSE_sf_fig = calc_strength_functions(theoretical_df = true_chars.par_post, 
                                    estimated_df = hist.elimination_history[level]['selected_ladder_chars'].par_post, 
                                    energy_range = [np.min(energy_grid_2_compare_on), np.max(energy_grid_2_compare_on)],
                                    fig_size=(8, 6), 
                                    create_fig=False
                                    )
            
            SF_Gn1.append(SSE_Gn1)
            SF_Gg.append(SSE_Gg)
            # end strength funcs calculation & fig production

        else:
            SSE_s.append(None)
            SF_Gg.append(None)
            SF_Gn1.append(None)

            # print(SSE_dict)

        # print(level)
        # print(f'N_res = {numres}')

        # print(f'level {level}, # of resonances: {numres},  passed the test: {pass_test}')
        cur_ch_dict = characterize_sol(Ta_pair=Ta_pair,
                        datasets = datasets,
                        experiments = experiments,
                        sol = hist.elimination_history[level]['selected_ladder_chars'],
                        covariance_data = covariance_data,
                        energy_grid_2_compare_on = energy_grid_2_compare_on
                        )
        
        # characterizing the combined ladder with side resonances - using Likellihoods

        LL_dict_combined = calc_LL_by_ladder(ladder_df = combined_ladder,
                                             Ta_pair = Ta_pair,
                                             energy_grid = energy_grid_2_compare_on)
        
        sum_NLLW_al.append(LL_dict_combined['total_NLLW_all_energy'])

        sum_NLL_gn_normal.append(LL_dict_combined['total_NLL_gn_normal_on_reduced_width_amp'])
        
        combined_NLL = LL_dict_combined['total_NLL_gn_normal_on_reduced_width_amp'] + LL_dict_combined['total_NLLW_all_energy']

        OF_alt4.append( sum(cur_ch_dict['chi2']) + 2 * combined_NLL    )

        # 

        
        N_res_s.append(numres)
        test_pass.append(pass_test)
        chi2_s.append(sum(cur_ch_dict['chi2']))

        chi2_stat_s_npar.append(cur_ch_dict['chi2_stat'])

        chi2_stat_s_ndat.append(cur_ch_dict['chi2_stat_ndat'])
        
        # aicc_s.append(sum(cur_ch_dict['aicc']))
        # bicc_s.append(sum(cur_ch_dict['bicc']))

        aicc_s.append(cur_ch_dict['aicc_entire_ds'])
        bicc_s.append(cur_ch_dict['bic_entire_ds'])

        NLLW_s.append(sum(cur_ch_dict['NLLW'].values()))

        NLL_Gn1.append(sum(cur_ch_dict['NLL_PT_Gn1'].values()))
        NLL_Gg.append(sum(cur_ch_dict['NLL_PT_Gg'].values()))



        N_res_joint_LL.append(cur_ch_dict['N_res_joint_LL'])

        OF_alt1.append(sum(cur_ch_dict['chi2'])+2 * sum(cur_ch_dict['NLLW'].values()))
        #OF_alt2.append(cur_ch_dict['aicc_entire_ds'] + sum(cur_ch_dict['NLLW'].values()) )
        OF_alt3.append(
            sum(cur_ch_dict['chi2']) + 
            2 * (
                sum(cur_ch_dict['NLLW'].values()) +
                sum(cur_ch_dict['NLL_PT_Gn1'].values())
                # sum(cur_ch_dict['NLL_PT_Gg'].values())
            )
        )

        # levels.append(level)
        # chi2_s.append(np.sum(hist.elimination_history[level]['selected_ladder_chars'].chi2_post))
        # N_ress.append(numres)
        # print(cur_ch_dict )

    # combine to one DataFrame
    table_df = pd.DataFrame.from_dict({
        'AT': is_true,
        'N_res': N_res_s,
        'N_res_joint_LL': N_res_joint_LL,
        'passed': test_pass,
        'sum_chi2': chi2_s,
        'chi2_s': chi2_stat_s_npar,
        'chi2_s_ndat': chi2_stat_s_ndat,
        'SSE': SSE_s,
        'sum_NLLW': NLLW_s,
        'sum_NLL_Gn1': NLL_Gn1,
        'sum_NLL_Gg': NLL_Gg,

        'OF_alt1': OF_alt1,
        'OF_alt3': OF_alt3,

        # for function that contain NLLW - with all resonances (artificial = remaining + art deleted)
        'sum_NLLW_al': sum_NLLW_al,
        'sum_NLL_gn_normal': sum_NLL_gn_normal,
        'OF_alt4': OF_alt4,


        'SF_Gn1': SF_Gn1,
        'SF_Gg': SF_Gg,
        'AICc': aicc_s,
        'BIC': bicc_s,
        'ET': elaps_time, # time
    })

    # calculating deltas in BIC and AICc
    table_df['delta_AICc_best'] = table_df['AICc'] - table_df['AICc'].min()
    table_df['delta_BIC_best'] = table_df['BIC'] - table_df['BIC'].min()
    table_df['delta_chi2_prev'] = table_df['sum_chi2'].diff().fillna(0)
    table_df['delta_chi2_best'] = table_df['sum_chi2'] - table_df['sum_chi2'].min()

    return table_df



def generate_sammy_rundir_uniq_name(path_to_sammy_temps: str, 
                                    case_id: int = None, 
                                    addit_str: str = ''):

    if not os.path.exists(path_to_sammy_temps):
        os.mkdir(path_to_sammy_temps)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")

    # Combine timestamp and random characters
    unique_string = timestamp + str(uuid.uuid4())
    
    # Truncate the string to 100 characters
    unique_string = unique_string[:100]

    if (case_id is not None):
        sammy_rundirname = path_to_sammy_temps+'SAMMY_RD_'+addit_str+'_'+str(case_id)+'_'+unique_string+'/'
    else:
        sammy_rundirname = path_to_sammy_temps+'SAMMY_RD_'+addit_str+'_'+unique_string+'/'

    return sammy_rundirname


# prior_lsts = lsts
def printout_chi2(sammyOUT: SammyOutputData, 
                       addstr :str = 'Solution chi2 values'):
    print(f'{addstr}')

    N_dat = np.sum([df.shape[0] for df in sammyOUT.pw])
    N_par = 3 * sammyOUT.par_post.shape[0]


    print('Chi2:')
    print('\t Prior:')
    print('\t\t', sammyOUT.chi2, np.sum(sammyOUT.chi2))
    print('\t Posterior:')
    print('\t\t', sammyOUT.chi2_post, np.sum(sammyOUT.chi2_post))

    print('Chi2_n:')
    print('\t Prior:')
    print('\t\t', sammyOUT.chi2n, np.sum(sammyOUT.chi2n))
    print('\t Posterior:')
    print('\t\t', sammyOUT.chi2n_post, np.sum(sammyOUT.chi2n_post))

    print('chi2_stat sum/(Ndat-Npar):')
    print(f'\tN_Dat: {N_dat}')
    print(f'\tN_Par: {N_par}')
    print()
    print(f'\t Prior: {np.sum(sammyOUT.chi2) / (N_dat - N_par)}')
    print(f'\t Posterior: {np.sum(sammyOUT.chi2_post) / (N_dat - N_par)}')
    print()
    print('\t Sum/Ndat')
    print(f'\t Prior: {np.sum(sammyOUT.chi2) / (N_dat)}')
    print(f'\t Posterior: {np.sum(sammyOUT.chi2_post) / (N_dat)}')
    print()


    


def load_fit_res(
        fit_res_folder : str,
        case_id : int = 0):
    """Loading fitting results from pkl file (sammyOUT_YW)"""
    
    sammyOUT_YW_obj = load_obj_from_pkl(folder_name = fit_res_folder,
                                                         pkl_fname = f'fit_res_{case_id}.pkl')
    
    return sammyOUT_YW_obj

def load_case_data(cases_folder_name: str,
                   case_id: int = 0):
    
    case_filename = f'sample_{case_id}.pkl'
    gen_params_filename =  f'params_gen.pkl'

    case_data_dict = {}
    
    sample_data = load_obj_from_pkl(folder_name = cases_folder_name,
                                                     pkl_fname = case_filename )

    params_loaded = load_obj_from_pkl(folder_name = cases_folder_name,
                                                     pkl_fname= gen_params_filename)
    
    case_data_dict['params'] = params_loaded
    case_data_dict['sample_data'] = sample_data

    return case_data_dict


def coefficient_of_variation(exp):
    """ Calculate the Coefficient of Variation (CV) of the signal. """
    return np.std(exp) / np.mean(exp)

def mean_uncertainty_percentage(exp, exp_unc):
    """ Calculate the Mean Uncertainty Percentage. """
    return np.mean(exp_unc / np.abs(exp)) * 100

def uncertainty_to_signal_ratio(exp, exp_unc):
    """ Calculate the Uncertainty-to-Signal Ratio (USR). mUSR! """
    return np.mean(exp_unc) / np.mean(exp)

def mean_signal_to_noise_ratio(exp, exp_unc):
    """ Calculate the Signal-to-Noise Ratio (SNR) in dB. """
    signal_power = np.mean(np.square(exp))
    noise_power = np.mean(np.square(exp_unc))
    return 10 * np.log10(signal_power / noise_power)


def calc_SNR(exp, exp_unc):
    """ 
    Calculate the Signal-to-Noise Ratio (SNR) in dB. 
    and mean.
    """
    signal_power = np.square(exp)
    noise_power = np.square(exp_unc)
    SNR_s = 10 * np.log10(signal_power / noise_power)

    mean_SNR = np.mean(SNR_s)

    return SNR_s, mean_SNR






# just to display the struct of a dict
def display_dict_structure(my_dict, indent=0):
    for key, value in my_dict.items():
        value_type = type(value).__name__
        value_shape = ""
        if hasattr(value, "shape"):
            value_shape = f" (shape={value.shape})"
        elif hasattr(value, "__len__"):
            value_shape = f" (size={len(value)})"
        print(f"{' ' * indent}- {key}: {value_type}{value_shape}")
        if isinstance(value, dict):
            display_dict_structure(value, indent=indent+2)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    display_dict_structure(item, indent=indent+2)



def create_comparison_dataframe(SammyOut_YW, cols_to_show, cols_to_keep=['J_ID']):
    """
    Compares parameters between 'par' and 'par_post' in the SammyOut_YW DataFrame.

    Args:
    SammyOut_YW (DataFrame): The original DataFrame containing 'par' and 'par_post' attributes.
    cols_to_show (list): List of columns to compare.
    cols_to_keep (list): List of columns to keep for both dataframes.

    Returns:
    DataFrame: A new DataFrame with comparisons.
    """

    # Select the required columns from 'par' and 'par_post'
    # df1 = SammyOut_YW['par'][cols_to_show + cols_to_keep]
    # df2 = SammyOut_YW['par_post'][cols_to_show + cols_to_keep]

    df1 = getattr(SammyOut_YW, 'par')[cols_to_show + cols_to_keep]
    df2 = getattr(SammyOut_YW, 'par_post')[cols_to_show + cols_to_keep]
    

    # Rename columns for differentiation
    df1_renamed = df1.rename(columns={col: col + '_1' for col in cols_to_show})
    df2_renamed = df2.rename(columns={col: col + '_2' for col in cols_to_show})

    # Concatenate the DataFrames horizontally and keep the specified columns
    result_df = pd.concat([df1_renamed, df2_renamed], axis=1)

    # Calculate differences for specified parameters
    for col in cols_to_show:
        result_df[f'delta_{col}'] = result_df[f'{col}_2'] - result_df[f'{col}_1']

    # Specify the order of columns and reorder the DataFrame
    columns_order = cols_to_keep + [f'{col}_1' for col in cols_to_show] + \
                    [f'{col}_2' for col in cols_to_show] + [f'delta_{col}' for col in cols_to_show]
    result_df = result_df[columns_order]

    return result_df



def analyze_max_min(result_diff_df, col_name):
    res = {
        'mean': np.mean(result_diff_df[col_name]),
        'median': np.median(result_diff_df[col_name]),
        'min': np.min(result_diff_df[col_name]),
        'max': np.max(result_diff_df[col_name]),
        'q_01': np.quantile(a = result_diff_df[col_name], q=0.01),
        'q_99': np.quantile(a = result_diff_df[col_name], q=0.99),
        'q_95': np.quantile(a = result_diff_df[col_name], q=0.95)
    }
    return res 


def plot_multiple_hist(values_list: list, 
                        bins: int, 
                        cumulative: bool, 
                        colors: list, 
                        captions: list, 
                        title: str = '',
                        show_kde: bool = True,
                        stacked: bool = False,
                        show_numbers: bool = False):
    
    # create a new figure and axis
    fig_new, ax_new = plt.subplots()

    # checking the range of values to compare the distr
    min_val = min([min(values) for values in values_list])
    max_val = max([max(values) for values in values_list])
    bins = np.linspace(min_val, max_val, num=bins)

    # create the histograms
    for i, values in enumerate(values_list):
        counts, edges, bars = ax_new.hist(values, 
                                          bins=bins, 
                                          density=False, 
                                          stacked=stacked, 
                                          alpha=0.4, 
                                          cumulative=cumulative, 
                                          color=colors[i], 
                                          label=captions[i])
        
        # kde
        if show_kde: 
            kde = gaussian_kde(values)
            x = bins #np.linspace(min(values), max(values), 50)
            plt.plot(x, kde(x), colors[i], label='KDE '+captions[i])

        if (show_numbers):
            for c in ax_new.containers:
                ax_new.bar_label(c)

    # set the title and legend
    if(len(title)>0):
        ax_new.set_title(title)
    ax_new.legend()

    return fig_new



def get_filenames_and_cases_ids(folder_name, filename_pattern):
    # Create the full pattern including the folder name
    full_pattern = os.path.join(folder_name, filename_pattern)

    # List all files matching the pattern
    matched_files = glob.glob(full_pattern)

    # Sort the files based on the integer part in the filename
    matched_files.sort(key=lambda f: int(os.path.basename(f).split('_')[-1].split('.pkl')[0]))

    # Extract filenames and numbers
    filenames = [os.path.basename(f) for f in matched_files]
    numbers = [int(filename.split('_')[-1].split('.pkl')[0]) for filename in filenames]

    return filenames, numbers


def check_and_change_template_paths(experiments,
                                    current_dir_with_templates):
    
    """checking if the template defined in the experiment.template exists
    and if not - change the path to 
    the same filename but current_dir_with_templates also checking """

    all_success = True

    for experiment in experiments:
        
        current_success = False

        given_filepath = experiment.template

        # Extract the filename 
        filename = os.path.basename(given_filepath)

        alternative_filepath = os.path.join(current_dir_with_templates, filename)

        if not os.path.exists(given_filepath):
            print("WARNING! File does not exist:", given_filepath)

            # check if alternative path exists
            if os.path.exists(alternative_filepath):
                print("Changing the path to existing template:", given_filepath, '->', alternative_filepath)
                experiment.template = alternative_filepath
                current_success = True

            else:
                print('Error, we have no appropriate templates for exp. objects!')
                print(given_filepath)
                print(alternative_filepath)
        
        else:
            current_success = True

        all_success = all_success * current_success
    
    return all_success


        # checkif file exists

    
    return 