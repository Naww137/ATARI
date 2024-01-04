### ADDITIONAL FUNCTIONS for ladders characterization or 
### resonances characterization

import numpy as np
import pandas as pd
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model

from ATARI.theory import resonance_statistics
from ATARI.utils.misc import fine_egrid
from ATARI.theory.resonance_statistics import sample_RRR_levels

from ATARI.sammy_interface.sammy_classes import SammyInputDataYW, SammyRunTimeOptions, SammyOutputData, SammyInputData
from ATARI.sammy_interface.sammy_functions import run_sammy

import os
import pickle

from matplotlib.pyplot import *
from matplotlib import pyplot as plt

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

    if (n<=k+1):

        raise ValueError(f'Number of parameters of the model must be << than number of observations! n = {n}, k = {k}')

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

    # for e-range and characterization

    e_range = [np.inf, 0] # min, max

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
        
        # energy range - select min and max energy from all datasets we have
        e_range[0] = min(ds.E.min(), e_range[0])
        e_range[1] = max(ds.E.max(), e_range[1])

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

        k = sol.par_post.shape[0] * 3  #+ 1 # estimating the variance

        n = len(aggregated_exp_data)

        AICc_entire_ds = 2*k + precalc_chi2_sum + 2*k*(k+1)/(n-k-1)
        BIC_entire_ds = k*np.log(n) + precalc_chi2_sum

        output_dict['aicc_entire_ds'] = AICc_entire_ds
        output_dict['bic_entire_ds'] = BIC_entire_ds

    # Wigner - by spingroups
    NLLW, NLLW_gr = calc_Wigner_LL_by_ladder(ladder_df = sol.par_post,
                             Ta_pair = Ta_pair)

    output_dict['NLLW'] = NLLW_gr

    # propability of having such number of resonances 
    #print(e_range)

    joint_prob, prob_by_spin_groups, joint_LL = calc_N_res_probability(Ta_pair=Ta_pair,
                                                             e_range = e_range,
                                                             ladder_df=sol.par_post)
    
    output_dict['N_res_prob_by_spingr'] = prob_by_spin_groups
    output_dict['N_res_joint_prob'] = joint_prob
    output_dict['N_res_joint_LL'] = joint_LL

    return output_dict


def calc_N_res_probability(Ta_pair: Particle_Pair,
                           e_range: list,
                           ladder_df: pd.DataFrame):
    
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
                                                                    num_samples=10000, 
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


### calc the SSE for one case




### elimination history funcs (parsing, plotting, etc)

def load_all(savefolder: str,
             hist_pkl_name: str,
             dataset_pkl_name: str):

    hist = load_obj_from_pkl(folder_name=savefolder, pkl_fname=hist_pkl_name)
    case_data_loaded = load_obj_from_pkl(folder_name=savefolder, pkl_fname=dataset_pkl_name)

    datasets = case_data_loaded['datasets']
    covariance_data = case_data_loaded['covariance_data']
    experiments = case_data_loaded['experiments']
    true_chars = case_data_loaded['true_chars']
    Ta_pair = case_data_loaded['Ta_pair']

    return datasets, covariance_data, experiments, true_chars, Ta_pair, hist





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

    ioff()

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
                 title : str = ''):

    
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
                    calc_fig = calc_xs_fig
            )

            if (xs_figure):
                xs_figure.savefig(fname=f'/home/fire/py_projects/ATARI_YW_newstruct/ATARI/examples/Ta181_Analysis/data/anim/xs_{key}_step_{level}.png')

            
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

    rundirname = './calc_xs_theo/'
    
    df = pd.DataFrame({"E":energy_grid})
    
    for rxn in reactions:
            


        exp_model_theo = Experimental_Model(title = "theo",
                               reaction = rxn,
                               #energy_range = energy_range_all,
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

        rto = SammyRunTimeOptions(
            sammyexe = settings['path_to_SAMMY_exe'],
            options = {"Print"   :   True,
                    "bayes"   :   False,
                    "keep_runDIR"     : settings['keep_runDIR_SAMMY'],
                    "sammy_runDIR": rundirname
                    }
        )

        # template_creator.make_input_template('theo.inp', Ta_pair, exp_model_theo, rto)
        
        sammy_INP= SammyInputData(
            particle_pair = Ta_pair,
            resonance_ladder = resonance_ladder,
            template = '/home/fire/py_projects/ATARI_YW_newstruct/ATARI/examples/Ta181_Analysis/theo.inp',
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
        reactions: list = ['capture', 'elastic', 'transmission'],
        print_bool=False
        ):
    """Calculating the residuals disctionaries on the fine grid for provided reactions"""

    energy_grid_fine = fine_egrid(energy=energy_grid)

    print('build_residual_matrix_dict')
    print('Energy grid fine:')
    print(len(energy_grid_fine))

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

        total_case_SSE, total_case_SSE_normalized  = 0 ,0
        size_to_norm = 0

        for rxn in ResidualMatrixDict.keys():
            total_case_SSE += SSE['SSE'][rxn]['casewise'][i]

            size_to_norm += ResidualMatrixDict[rxn][i].shape[0]
        
        total_case_SSE_normalized = total_case_SSE / size_to_norm

        SSE['SSE_sum_casewise'].append(total_case_SSE)
        SSE['SSE_sum_normalized_casewise'].append(total_case_SSE_normalized)

    return SSE


def calc_all_SSE_gen_XS_plot(
        est_ladder: pd.DataFrame,
        theo_ladder: pd.DataFrame,
        Ta_pair: Particle_Pair,
        settings: dict,
        energy_grid: np.array,
        reactions_SSE : list = ['capture', 'elastic'],
        fig_size: tuple = (8,6),
        calc_fig: bool = False
):
    print('Input Energy grid')
    print(len(energy_grid))
    
    resid_matrix = build_residual_matrix_dict(
        est_par_list = [est_ladder], 
        true_par_list = [theo_ladder], # [jeff_parameters], 
        Ta_pair = Ta_pair, 
        settings = settings,
        energy_grid = energy_grid,
        reactions = reactions_SSE, 
        print_bool=True
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
        ioff()
        figure = plt.figure(figsize = fig_size)
        
        # # Total
        plt.plot(df_est.E, df_est.xs_transmission, label='Fit result', color = 'b', alpha=1.0, linewidth=1.0)
        plt.plot(df_theo.E, df_theo. xs_transmission, label='Theo', color = 'r', alpha=1.0, linewidth=1.0)
        
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
        plt.yscale('log')
        # plt.xlim(208.5,216)
        # plt.ylim(6,600)
        figure.tight_layout()
    
    else:
        figure = False
        df_est = pd.DataFrame()
        df_theo = pd.DataFrame()


    return df_est, df_theo, resid_matrix, SSE_dict, figure


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
    
    return time_components_list, formatted_string