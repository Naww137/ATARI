import os
import h5py
from ATARI.syndat.data_classes import syndatOUT
from ATARI.PiTFAll import fnorm
import ATARI.utils.hdf5 as h5io
import pandas as pd
from copy import copy

def add_synthetic_data_samples(
                               performance_test_filepath,
                               syndat_sample_filepath, 
                               isample_max,
                               energy_range,
                               hdf5=True,
                               overwrite=False
                                                ):
    """
    Adds syndat samples of true resonance parameters and experimental datasets to the Performance Test instance.
    Currently only implemented for HDF5 samples.

    Parameters
    ----------
    syndat_sample_filepath : str
        Path to the syndat samples hdf5 file.
    hdf5 : bool, optional
        _description_, by default True

    Raises
    ------
    ValueError
        hdf5 False is not yet implemented
    """
    if os.path.isfile(performance_test_filepath):
        pass
    else:
        h5f = h5py.File(performance_test_filepath, "w")
        h5f.close()
    if os.path.isfile(syndat_sample_filepath):
        pass
    else:
        raise ValueError(f"Syndat sample filepath does not exist: {syndat_sample_filepath}")

    if hdf5:

        ### Check syndat file for sample range and get experiment titles
        h5f = h5py.File(syndat_sample_filepath, "r")
        exp_titles_0 = ['_'.join(each.split('_')[2:]) for each in h5f['sample_0'].keys() if each not in ['par_true']]
        isamples = []
        for isample in range(isample_max):
            isample_key = f"sample_{isample}"
            if isample_key in h5f.keys():
                isamples.append(isample)
                exp_keys = h5f[isample_key].keys()
                exp_titles = ['_'.join(each.split('_')[2:]) for each in exp_keys if each not in ['par_true']]
                assert(exp_titles == exp_titles_0) # make sure all experiment titles are the same
            else:
                print(f"{isample_key} not present in {os.path.basename(syndat_sample_filepath)}")
        h5f.close()

        ### filter isample keys if they already exist in the performance test file
        if overwrite: # TODO: update overwrite capabilities
            # isamples_new = isamples
            # h5f = h5py.File(performance_test_filepath, "r")
            # for i in isamples:
            #     if f"sample_{i}" not in h5f.keys():
            #         pass
            #     else:
            # h5f.close()
            raise ValueError("Not Implemented")
        else:
            h5f = h5py.File(performance_test_filepath, "r")
            isamples_new = [i for i in isamples if f"sample_{i}" not in h5f.keys()]
            h5f.close()

        for i in isamples_new:
            
            for exp in exp_titles:
                syndat_out = syndatOUT.from_hdf5(syndat_sample_filepath, i, exp)
                # filter parameter, pw_reduced, and covariance data frames
                syndat_out.par_true = syndat_out.par_true[(syndat_out.par_true.E>min(energy_range)) & (syndat_out.par_true.E<max(energy_range))]
                syndat_out.par_true.reset_index(inplace=True, drop=True)
                syndat_out.pw_reduced = syndat_out.pw_reduced[(syndat_out.pw_reduced.E>min(energy_range)) & (syndat_out.pw_reduced.E<max(energy_range))]
                syndat_out.pw_reduced.reset_index(inplace=True, drop=True)
                if syndat_out.covariance_data:
                    syndat_out.covariance_data['diag_stat'] = syndat_out.covariance_data['diag_stat'].loc[(syndat_out.covariance_data['diag_stat'].index>min(energy_range)) & (syndat_out.covariance_data['diag_stat'].index<max(energy_range))]
                    syndat_out.covariance_data['Jac_sys'] = syndat_out.covariance_data['Jac_sys'].loc[:, (syndat_out.covariance_data['Jac_sys'].columns>min(energy_range)) & (syndat_out.covariance_data['Jac_sys'].columns<max(energy_range))]
                # write to perf test file
                syndat_out.to_hdf5(performance_test_filepath, i)


    else:
        raise ValueError("Not yet implemented for hdf5=False")
    



def add_model_fit(performance_test_filepath,
                   isample,
                   model_title,
                   par_df,
                   exp_df_dict,
                   overwrite = False
                   ):
    """
    Adds a single model fit to the performance test .hdf5 file.
    Parameter data frame and experimental pointwise dataframes are passed directly to this function, i.e., the user should write their own parser function to pull this data from a saved file as there is no defined structure for output of fitted models.
    The experimental pointwise data frames should be in a dictionary (exp_df_dict) with keys corresponding to the experiment title.
    They should also be filtered to only have two columns, one for energy and one for the model fit. 
    This function will re-title the pointwise model fit based on the supplied model_title arguement and add it as a column to the pw_reduced dataframe for this experiment.

    Parameters
    ----------
    performance_test_filepath : _type_
        _description_
    isample : bool
        _description_
    model_title : _type_
        _description_
    par_df : _type_
        _description_
    exp_df_dict : _type_
        _description_
    overwrite : bool, optional
        _description_, by default False
    """
    if os.path.isfile(performance_test_filepath):
        pass
    else:
        h5f = h5py.File(performance_test_filepath, "w")
        h5f.close()

    sample_key = f"sample_{isample}"

    h5f = h5py.File(performance_test_filepath, "r")
    isample_exists = sample_key in h5f.keys()
    if isample_exists:
        model_exists = f"par_{model_title}" in h5f[sample_key].keys()
    h5f.close()
    
    if isample_exists:
        if model_exists and overwrite or not model_exists:
            h5io.write_par(performance_test_filepath, isample, par_df, model_title)
            for exp, df in exp_df_dict.items():
                df = copy(df)
                df.rename(columns={df.keys()[1]: model_title}, inplace=True)
                pw_reduced_df = pd.read_hdf(performance_test_filepath, f"{sample_key}/exp_dat_{exp}/pw_reduced")
                if model_title in pw_reduced_df.keys():
                    pw_reduced_df.drop(columns=f"{model_title}", inplace=True)
                pw_reduced_df["E"] = pw_reduced_df["E"].round(6)
                df["E"] = df["E"].round(6)
                pw_reduced_df = pw_reduced_df.merge(df, on='E', how='outer', validate='1:1') # if overwrite, remove based on suffix
                pw_reduced_df.to_hdf(performance_test_filepath, f"{sample_key}/exp_dat_{exp}/pw_reduced")
        else:
            print(f"Model {model_title} for sample {isample} exists already, overwrite set to False")

    else:

        h5io.write_par(performance_test_filepath, isample, par_df, model_title)
        for exp, df in exp_df_dict.items():
            df.rename(columns={df.keys()[1]: model_title}, inplace=True)
            df.to_hdf(performance_test_filepath, f"{sample_key}/exp_dat_{exp}/pw_reduced")
    


def add_fine_grid_doppler_only(performance_test_filepath,
                               isample_max,
                               energy_range,
                               sammy_exe,
                               particle_pair,
                               model_title = "true",
                               temperature = 300,
                               template =  os.path.realpath(os.path.join(os.path.dirname(__file__), "../sammy_interface/sammy_templates/dop_2sg.inp")),
                               reactions = ['elastic', 'capture'],
                               overwrite = False,
                               isample_min = 0
                               ):
                               
    # Check if file exists
    if os.path.isfile(performance_test_filepath):
        pass
    else:
        raise ValueError(f"Performance test file does not exist: {performance_test_filepath}")
    
    # get column key mapper
    mapper={}
    for rxn in reactions:
        mapper[rxn] = f"{rxn}_{model_title}"
        
    # see which samples have a theo_pw df already
    h5f = h5py.File(performance_test_filepath, "r")
    isamples = []
    theo_exists_list = []
    par_exists_list =[]
    for isample in range(isample_min, isample_max):
        isample_key = f"sample_{isample}"
        if isample_key in h5f.keys():
            isamples.append(isample)
            theo_exists_list.append(f"theo_pw" in h5f[isample_key].keys())
            par_exists_list.append(f"par_{model_title}" in h5f[isample_key].keys())
    h5f.close()

    for isample, theo_exists, par_exists in zip(isamples, theo_exists_list, par_exists_list):

        if not par_exists:
            print(f"model {model_title} doesn't exist for sample {isample}")
            continue

        if theo_exists:
            theo_df = pd.read_hdf(performance_test_filepath, f'/sample_{isample}/theo_pw')
            
            if f"{reactions[0]}_{model_title}" not in theo_df.keys() or overwrite:
                par_df = h5io.read_par(performance_test_filepath, isample, model_title)
                pw_df = fnorm.calc_theo_broad_xs_for_all_reaction(sammy_exe, particle_pair,  par_df,  energy_range, temperature, template,  reactions)
                pw_df.rename(columns=mapper, inplace=True)
                theo_df = theo_df.merge(pw_df, how='inner', validate="1:1", on="E", suffixes=('_keep', ''))
                theo_df = theo_df[[col for col in theo_df.columns if not col.endswith('_keep')]]
                if theo_df.empty:
                    print(theo_df, pw_df)
                    raise ValueError("theoretical dataframe did not merge properly")
                theo_df.to_hdf(performance_test_filepath, f"sample_{isample}/theo_pw")

        else:
            par_df = h5io.read_par(performance_test_filepath, isample, model_title)
            pw_df = fnorm.calc_theo_broad_xs_for_all_reaction(sammy_exe, particle_pair,  par_df,  energy_range, temperature, template,  reactions)
            pw_df.rename(columns=mapper, inplace=True)
            pw_df.to_hdf(performance_test_filepath, f"sample_{isample}/theo_pw")
