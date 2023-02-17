"""
all functions
"""

# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

import math
from scipy.ndimage import gaussian_filter1d
from scipy import signal
from scipy import integrate

## all the funcs

# loading CSV transmission data in convenient form
def load_transmission_data_csv(filename: str, n: float) -> pd.DataFrame:
    """
    loading transmission data from csv
    and calculating the crossection values
    """

    # TODO : add logic s.t. this function can read formatted dataframes or path to csv


    # dataframe with transmission
    #filename = isotope_name + '_Emin_10_Emax_1000_2022.10.30_transmission_4980.csv'

    df = pd.read_csv(filename, index_col=0)
    #print(df.head())

    #sorting the df by E and reindexing for naive visualization
    df = df.sort_values(by=['E'], ascending=True)
    
    # resetting index after sorting
    df.reset_index(inplace=True)

    #n = 0.067166 # atoms per barn or atoms/(1e-12*cm^2)

    # adding information about cross section value.
    df['theo_cs'] = np.log(df['theo_trans']) / (-n) # theoretical crosssection
    df['exp_cs'] = np.log(df['exp_trans']) / (-n)

    # calculate tof
    df['tof'] = df['tof'] / 1e6
    
    print(f'Initial data file contains {df.shape[0]} elements')

    # are there NAN elements in exp_cs
    if (df['exp_cs'].isnull().sum()>0):
        print('Warning! There are ', df['exp_cs'].isnull().sum(), ' points in the data file with NAN elements')

    return df

def delete_nan_cs_rows(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    deletes rows in the data with NANs in experimental cross section, 
    where transmission values<=0
    """
    # searching for the transmission points that are below zero..

    #exp_trans_below_zero = input_df[input_df["exp_trans"]<=0]
    #print(f'Number of points below zero: {exp_trans_below_zero.shape[0]} of {input_df.shape[0]}')

    # finding the elements in a dataframe with the None values in a column
    df_wo_NANS = input_df[~input_df['exp_cs'].isnull()]

   
    #resetting the index of the dataframe
    df_wo_NANS.reset_index(drop=True, inplace=True)

    # print('*'*80)
    # print(df_wo_NANS)

    # deleting unnecessary columns after deletion
    df_wo_NANS.drop(['index'], axis=1, inplace = True) 

    # adding a new column with the current row number
    if 'row_num' in df_wo_NANS:
        pass
    else:
        df_wo_NANS.insert(loc=0, column='row_num', value=np.arange(len(df_wo_NANS)))

    print(f'Count in initial: {input_df.shape[0]} \nCount in cleaned {df_wo_NANS.shape[0]}')
    return df_wo_NANS

# plotting ladder and transmission / crosssection
def plot_trcs(
        x_arr: list,
        y_arr: list,
        plotname: str,
        labels: list,
        x_axis_label: str,
        y_axis_label: str,
        ladder: pd.DataFrame):
    
    fig, axs = plt.subplots()
    axs.set_title(plotname)

    colors = ['royalblue', 'red', 'green', 'black']

    for index in range(0, len(x_arr)):

        axs.plot(x_arr[index],
                y_arr[index], 
                marker = "o", 
                markersize = 2, 
                linewidth = 1.0, 
                alpha=0.7, 
                color = colors[index], 
                label = labels[index])

    axs.grid(color ='grey', linestyle ='--', linewidth = 0.2)

    axs.set_xlabel(x_axis_label)
    axs.set_ylabel(y_axis_label)

    axs.legend()

    for index, row in ladder.iterrows():
        x1 = row['E']
        axs.axvline(x = x1, label=r'real positions', color = 'g', linestyle = 'dashed', linewidth = 1.0, alpha=0.5)

def calc_linear_interp_index(df, input_index, input_col='point_number', output_col='E'):
    return np.interp(input_index, df[input_col], df[output_col])


def search_peaks(transm_nonzero: pd.DataFrame, 
                 search_params: dict, 
                 reduce_params: dict,
                 ):
    """
    Searching for peaks in cross section
    """
    cs = transm_nonzero['exp_cs']

    # TODO: add another smoothed cross-section for that?


    #filtering data using gaussian filter
    sigma_filter = cs.std() * search_params['part_of_variance_filter']
    
    dataFiltered = gaussian_filter1d(cs, sigma = sigma_filter)

    all_peaks_indexes_df = signal.find_peaks(
    dataFiltered, 
    prominence = search_params['prominence'], 
    width = search_params['min_width'], # important - in counts!!, not in energy values
    height = search_params['min_height'], 
    distance = 1,
    rel_height = search_params['base_of_peak']  # for signal width estimation using linear interpolation
    )

    print(f'Initial search gives: {len(all_peaks_indexes_df[0])} peaks')

    # creating a dataframe for ALL pulses
    peaks_df_all = pd.DataFrame() # here all the peaks data will be stored
    peaks_list_all = []

    for idx, x in enumerate(all_peaks_indexes_df[0]):
        
        peak_indx_E = calc_linear_interp_index(transm_nonzero, x, input_col='row_num', output_col='E')

        peak_left_border_indx = math.floor(all_peaks_indexes_df[1]["left_ips"][idx]) # to select one point left

        #peak_left_border_indx = all_peaks_indexes_df[1]["left_ips"][idx]

        peak_left_border_E = calc_linear_interp_index(transm_nonzero, peak_left_border_indx, input_col='row_num', output_col='E')

        peak_right_border_indx = math.ceil(all_peaks_indexes_df[1]["right_ips"][idx])
        #peak_right_border_indx = all_peaks_indexes_df[1]["right_ips"][idx]

        peak_right_border_E = calc_linear_interp_index(transm_nonzero, peak_right_border_indx, input_col='row_num', output_col='E')

        peak_width = peak_right_border_indx - peak_left_border_indx
        peak_relative_prom_width_height = all_peaks_indexes_df[1]["width_heights"][idx]

        peak_width_E = peak_right_border_E - peak_left_border_E 

        peak_abs_height = all_peaks_indexes_df[1]['peak_heights'][idx]
        peak_prominence = all_peaks_indexes_df[1]["prominences"][idx]

        peak_simple_sq = peak_abs_height * peak_width_E # simple square of a peak using height and width

        peak_sq = 1 # square under curve
        # to calc square under curve using integration we need to select points of a pulse
        
        peak_points_df = transm_nonzero[(transm_nonzero['E']>=peak_left_border_E) & (transm_nonzero['E']<=peak_right_border_E)]
        peak_numpoints = peak_points_df.shape[0]

        peak_sq = integrate.simpson(peak_points_df['exp_cs'], peak_points_df['E'])

        peak_sq_divE = peak_sq / peak_width_E
        
        # TODO: speed up!
        # it works slow because of this? How to avoid this problem if I need to output a dataframe
    
        peaks_list_all.append(
        {
            'idx_global_num': idx, #pulse index
            'x': x, #pulse peak index
            'peak_E': peak_indx_E, #pulse peak index in E
            'peak_left_border_indx': peak_left_border_indx,
            'peak_right_border_indx': peak_right_border_indx,
            'peak_width_samples': peak_width,
            'peak_relative_prom_width_height': peak_relative_prom_width_height,
            'peak_left_border_E': peak_left_border_E,
            'peak_right_border_E': peak_right_border_E,
            'peak_width_E': peak_width_E,
            'peak_height': peak_abs_height,
            'peak_prominence': peak_prominence,
            'peak_numpoints': peak_numpoints,
            'peak_simple_sq': peak_simple_sq,
            'peak_sq': peak_sq,
            'peak_sq_divE': peak_sq_divE
        })
    
    peaks_df_all = pd.DataFrame.from_dict(peaks_list_all)
    

    ## reducement of the dataframe size using criteria
    cutoff_relative_threshold = reduce_params['cutoff_threshold']

    param_name = reduce_params['param_cutoff_by_name'] # peak_sq - square value of a peak

    param_span = peaks_df_all[param_name].max() - peaks_df_all[param_name].min()

    param_threshold = peaks_df_all[param_name].min() + cutoff_relative_threshold * (param_span)

    # mark only peaks subjected by criteria
    peaks_df_all.loc[peaks_df_all[param_name] >= param_threshold, 'selected'] = 1 
    peaks_df_all.loc[peaks_df_all[param_name] < param_threshold, 'selected'] = 0 

    peaks_df_selected = peaks_df_all[peaks_df_all['selected']==1]
    selected_num = peaks_df_selected.shape[0]

    print(f'Selected {selected_num} peaks from {peaks_df_all.shape[0]}')
    ## end reducement of the initial dataframe

    return peaks_df_selected, peaks_df_all


# plotting the results with the  boxes for convenience

def plot_search_results(transm_df: pd.DataFrame,
                        ladder_df: pd.DataFrame,
                        estimated_pos_df: pd.DataFrame 
                        ):
    """
    visualizing the SELECTED & REAL peak(pole:) positions
    """

    # cs & T

    fig, axs = plt.subplots(2, sharex=True)

    fig.suptitle('Theo and exp Cross-section/T, positions of estimated pulses (missed data removed, filtered)')

    #theoretical cross-section
    axs[0].plot(transm_df['E'], transm_df['theo_cs'], marker = "o", markersize = 1, linewidth = 1.0, alpha=0.8, color = 'g', label = 'Theo', zorder=1)
    axs[1].plot(transm_df['E'], transm_df['theo_trans'], marker = "o", markersize = 1, linewidth = 1.0, alpha=0.8, color = 'g', label = 'Theo')

    #experimental cross-section
    axs[0].plot(transm_df['E'], transm_df['exp_cs'], marker = "o", markersize = 1, linewidth = 1.0, alpha=1, color = 'red', label = 'Exp', zorder=3)
    axs[1].plot(transm_df['E'], transm_df['exp_trans'], marker = "o", markersize = 2, linewidth = 0.5, alpha=0.2, color = 'red', label = 'Exp')


    axs[0].grid(color = 'grey', linestyle = '--', linewidth = 0.2)
    axs[1].grid(color = 'grey', linestyle = '--', linewidth = 0.2)
    
    axs[0].legend()
    axs[1].set_xlabel('E')
    axs[0].set_ylabel('$\sigma(E)$')
    axs[1].set_ylabel('T(E)')
    

    # initial ladder - green vertical lines
    for index, row in ladder_df.iterrows():
        x1 = row['E']
        axs[0].axvline(x = x1, color = 'r', linestyle = 'dashed', linewidth = 0.5, alpha=0.5) 

    for index, row in ladder_df.iterrows():
        x1 = row['E']
        axs[1].axvline(x = x1, color = 'r', linestyle = 'dashed', linewidth = 0.5, alpha=0.5)

    # visualising the pulses width and heights of SELECTED pulses

    for idx, row in estimated_pos_df.iterrows():
        axs[0].hlines(
            y=row['peak_relative_prom_width_height'], 
            xmin=row['peak_left_border_E'], # left border
            xmax=row['peak_right_border_E'], # right border
            color = "blue"
            )
        axs[0].vlines(
            x=row['peak_E'],
            ymin=row["peak_height"]-row["peak_prominence"],
            ymax=row["peak_height"],
            color = 'blue'
            )
    
    # boxes for transmission

    boxes = []

    current_tr_h = transm_df['exp_trans'].max()

    for index,row in estimated_pos_df.iterrows():

        current_rect_left_corner = (row['peak_left_border_E'], 0)
        current_w = row['peak_right_border_E'] - row['peak_left_border_E']
        current_h = current_tr_h

        boxes.append(Rectangle(current_rect_left_corner, current_w, current_h))

    pc = PatchCollection(boxes, facecolor='y', alpha=0.2, edgecolor='black')
    axs[1].add_collection(pc)