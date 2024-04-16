# cSpell:disable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from scipy.stats import normaltest
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI
from ATARI.ModelData.experimental_model import Experimental_Model

"""Summary
----------
    Description

Parameters
----------
    Name - Type: Text - Default: Text
    
        -Description

Modes
-----
    Name - Type: Text - Default: Text
    
        -Value - Function

Returns
-------
    Name - Type: Text - Default: Text
    
        -Description

Raises
------
    Error - Likely cause: Text
    
        -Description

Outputs
-------
    Text - Cause: Text
    
        -Description
"""



def basic_functionality_test(failure_logging=False):
    """Summary
    ----------
        Tests the function for basic functionality. Allows for more nuanced error correction.

    Raises
    ------
        "Program runs"
        
        "Program runs, but no output data is provided."
    """
    
    
    print("➤ Testing that the code runs")
    print("➤ Expected functionality: The program functions")
    print("")
    
    exp_model=Experimental_Model()
    true_yield = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    
    
    try:
        exp = Capture_Yield_RPI()
    except Exception as error:
        print("⇉ Program breaks when generating Class.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        exp.approximate_unknown_data(exp_model,True)
    except Exception as error:
        print("⇉ Program breaks when aproximating unknown data.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        model_parameters=exp.sample_true_model_parameters({})
    except Exception as error:
        print("⇉ Program breaks when sampling model parameters.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        generated_counts=exp.generate_raw_data(true_yield,exp.model_parameters,False)
    except Exception as error:
        print("⇉ Program breaks when generating count data without sampling.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        generated_counts=exp.generate_raw_data(true_yield,exp.model_parameters,True)
    except Exception as error:
        print("⇉ Program generates count data but breaks when sampling the data.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        generated_counts=exp.generate_raw_data(true_yield,model_parameters,True)
    except Exception as error:
        print("⇉ Program samples count data but breaks when using sampled model parameters.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        generated_yield,covariance_data=exp.reduce_raw_data(true_yield,generated_counts,exp.model_parameters,False)
    except Exception as error:
        print("⇉ Program breaks when reducing count data to yield data.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    try:
        generated_yield,covariance_data=exp.reduce_raw_data(true_yield,generated_counts,model_parameters,False)
    except Exception as error:
        print("⇉ Program samples yield data but breaks when using sampled model parameters.")
        print("")
        print("X Program failed basic functionality test.")
        if(failure_logging):
            print("")
            print("Reported error:")
            print(error)
            print("")
        return 0
    
    print("✓ Program passed basic functionality test.")
    return 1



def data_type_test():
    """_summary_
    """
    
    
    passed=True
    print("➤ Testing returned data")
    print("➤ Expected functionality: Data is returned matching the expected format")
    print("")
    
    exp_model=Experimental_Model()
    true_yield = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    generated_counts=exp.generate_raw_data(true_yield,exp.model_parameters,False)
    generated_yield,covariance_data=exp.reduce_raw_data(true_yield,generated_counts,exp.model_parameters,False)
    
    if(type(generated_counts)==pd.DataFrame):
        if(not(generated_counts.empty)):
            if("ctg_true" in generated_counts):
                if(not(len(generated_counts.ctg_true)==len(exp_model.energy_grid))):
                    print("⇉ True count data is on a different energy grid.")
                    passed=False
            else:
                print("⇉ There is no true count data.")
                passed=False
            if("ctg" in generated_counts):
                if(not(len(generated_counts.ctg)==len(exp_model.energy_grid))):
                    print("⇉ Sampled count data is on a different energy grid.")
                    passed=False
            else:
                print("⇉ There is no sampled count data.")
                passed=False
            if("dctg" in generated_counts):
                if(not(len(generated_counts.dctg)==len(exp_model.energy_grid))):
                    print("⇉ Sampled count uncertainty data is on a different energy grid.")
                    passed=False
            else:
                print("⇉ There is no sampled count uncertainty data.")
                passed=False
        else:
            print("⇉ Count dataframe is empty.")
            passed=False
    else:
        print("⇉ The count dataframe is actually a "+type(generated_counts)+" instead of a dataframe.")
        passed=False
    
    if(type(generated_yield)==pd.DataFrame):
        if(not(generated_yield.empty)):
            if("E" in generated_yield):
                if(not(np.all(np.equal(generated_yield.E.values,exp_model.energy_grid)))):
                    print("⇉ Energy grid data stored with yield does not match input.")
                    passed=False
            else:
                print("⇉ There is no energy grid data stored with yield.")
                passed=False
            if("tof" in generated_yield):
                if(not(np.all(np.equal(generated_yield.tof.values,exp_model.tof_grid)))):
                    print("⇉ Time of flight data stored with yield does not match input.")
                    passed=False
            else:
                print("⇉ There is no time of flight data stored with yield.")
                passed=False
            if("true" in generated_yield):
                if(not(np.all(np.equal(generated_yield.true.values,true_yield.true)))):
                    print("⇉ Output true yield data does not match input true yield data.")
                    passed=False
            else:
                print("⇉ There is no true yield data.")
                passed=False
            if("exp" in generated_yield):
                if(not(len(generated_yield.exp)==len(exp_model.energy_grid))):
                    print("⇉ Sampled yield data is on a different energy grid.")
                    passed=False
            else:
                print("⇉ There is no sampled yield data.")
                passed=False
            if("exp_unc" in generated_yield):
                if(not(len(generated_yield.exp_unc)==len(exp_model.energy_grid))):
                    print("⇉ Sampled yield uncertainty data is on a different energy grid.")
                    passed=False
            else:
                print("⇉ There is no sampled yield uncertainty data.")
                passed=False
        else:
            print("⇉ Yield dataframe is empty.")
            passed=False
    else:
        print("⇉ The yield dataframe is actually a "+type(generated_yield)+" instead of a dataframe.")
        passed=False
    
    if(passed):
        print("✓ Program passed data type test.")
    else:
        print("✘ Program failed data type test.")
    return passed



def paramater_sampling_test():
    """_summary_
    """
    
    print("➤ Tests the sampled paramaters to ensure they are actually different")
    print("➤ Expected functionality: The returned sampled parameters deviate from the input parameters")
    print("")
    
    exp_model=Experimental_Model()
    
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    model_parameters=exp.sample_true_model_parameters({})
    true_model_parameters_dictionary={key:value for key, value in exp.model_parameters.__dict__.items() if not key.startswith('__') and not callable(key)}
    sampled_model_parameters_dictionary={key:value for key, value in model_parameters.__dict__.items() if not key.startswith('__') and not callable(key)}
    
    changed=False
    for key,value in true_model_parameters_dictionary.items():
        if(type(value)==pd.DataFrame):
            #I am not dealing with this right now, good luck
            pass
        else:
            if(not(value==sampled_model_parameters_dictionary[key])):
                changed=True
    
    if(changed):
        print("⇉ Parameter sampling did not change any values.")
        print("")
        print("✘ Program failed parameter sampling test.")
        return False
    else:
        print("✓ Program passed parameter sampling test.")
        return True


def no_sampling_test(failure_logging=False):
    """Summary
    ----------
        Tests the yield function to ensure that the returned values are the same as input when sampling is disabled

    Modes
    -----
        success_logging - type: boolean - default: False
        
            -True  - Prints additional information about the successful run
            
            -False - Suppresses the additional information
        
        failure_logging - type: boolean - default: True
        
            -True  - Prints additional information about the failed run
            
            -False - Suppresses the additional information
    """
     
    print("➤ Testing similarity between return data and input data with sampling disabled")
    print("➤ Expected functionality: The returned data is identical to the input data if all sampling is disabled")
    print("")
    
    
    exp_model=Experimental_Model()
    true_yield = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    generated_counts=exp.generate_raw_data(true_yield,exp.model_parameters,False)
    generated_yield,covariance_data=exp.reduce_raw_data(true_yield,generated_counts,exp.model_parameters,False)
    
    if(np.allclose(generated_yield.exp.values,true_yield.true.values)):
        print("✓ Program passed no sampling test.")
        return True
        
    else:
        print("Reported yield data does not match true yield data.")
        print("")
        print("X Program failed no sampling test.")
        if(failure_logging):
            print("")
            print("Returned values:")
            print(generated_yield.exp.values)
            print("")
            print("True values:")
            print(true_yield.true.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_yield.exp.values,true_yield.true.values))
            print("")
        return False



def sampling_test(failure_logging=False):
    """Summary
    ----------
        Tests the yield function to ensure the code is returning sampled data
    
    Modes
    -----
        success_logging - type: boolean - default: False
        
            -True  - Prints additional information about the successful run
            
            -False - Suppresses the additional information
        
        failure_logging - type: boolean - default: True
        
            -True  - Prints additional information about the failed run
            
            -False - Suppresses the additional information
    """
    
    passed=True
    print("➤ Testing difference between returned data and input data with sampling enabled")
    print("➤ Expected functionality: returned yield is different with any sampling and returned counts are different only if sampling counts")
    print("")
    
    
    exp_model=Experimental_Model()
    true_yield = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    
    #Case 0: no sampling
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    generated_counts_no_sample=exp.generate_raw_data(true_yield,exp.model_parameters,False)
    generated_yield_no_sample,___=exp.reduce_raw_data(true_yield,generated_counts_no_sample,exp.model_parameters,False)
    
    #Case 1: count data is sampled
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    generated_counts_case1=exp.generate_raw_data(true_yield,exp.model_parameters,True)
    generated_yield_case1,___=exp.reduce_raw_data(true_yield,generated_counts_case1,exp.model_parameters,False)
    if(np.allclose(generated_counts_case1.ctg.values,generated_counts_no_sample.ctg.values)):
        print("⇉ Count data is the same with or without count data sampling.")
        print("")
        if(failure_logging):
            print("Sampled data:")
            print(generated_counts_case1.ctg.values)
            print("")
            print("Non-sampled data:")
            print(generated_counts_no_sample.ctg.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_counts_case1.ctg.values,generated_counts_no_sample.ctg.values))
            print("")
        passed=False
    if(np.allclose(generated_yield_case1.exp.values,generated_yield_no_sample.exp.values)):
        print("⇉ Yield data is the same with or without count data sampling.")
        print("")
        if(failure_logging):
            print("Sampled data:")
            print(generated_yield_case1.exp.values)
            print("")
            print("Non-sampled data:")
            print(generated_yield_no_sample.exp.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_yield_case1.exp.values,generated_yield_no_sample.exp.values))
            print("")
        passed=False
    
    #Case 2: model parameters are sampled
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    generated_counts_case2=exp.generate_raw_data(true_yield,exp.model_parameters,False)
    model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0], 
                                                       "trig_bg":exp.model_parameters.trig_bg[0],
                                                       "trig_f":exp.model_parameters.trig_f[0], 
                                                       "trig_bf":exp.model_parameters.trig_bf[0],
                                                       "fn":exp.model_parameters.fn[0],
                                                       "yield_flux":exp.model_parameters.yield_flux.ct})
    generated_yield_case2,___=exp.reduce_raw_data(true_yield,generated_counts_case2,model_parameters,False)
    if(not(np.allclose(generated_counts_case2.ctg.values,generated_counts_no_sample.ctg.values))):
        print("⇉ Count data is differnt with only parameter sampling.")
        print("")
        if(failure_logging):
            print("Sampled data:")
            print(generated_counts_case2.ctg.values)
            print("")
            print("Non-sampled data:")
            print(generated_counts_no_sample.ctg.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_counts_case2.ctg.values,generated_counts_no_sample.ctg.values))
            print("")
        passed=False
    if(np.allclose(generated_yield_case2.exp.values,generated_yield_no_sample.exp.values)):
        print("⇉ Yield data is the same with or without parameter sampling.")
        print("")
        if(failure_logging):
            print("Sampled data:")
            print(generated_yield_case2.exp.values)
            print("")
            print("Non-sampled data:")
            print(generated_yield_no_sample.exp.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_yield_case2.exp.values,generated_yield_no_sample.exp.values))
            print("")
        passed=False
    
    #Case 3: both are sampled
    exp = Capture_Yield_RPI()
    exp.approximate_unknown_data(exp_model,True)
    generated_counts_case3=exp.generate_raw_data(true_yield,exp.model_parameters,True)
    model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0], 
                                                       "trig_bg":exp.model_parameters.trig_bg[0],
                                                       "trig_f":exp.model_parameters.trig_f[0], 
                                                       "trig_bf":exp.model_parameters.trig_bf[0],
                                                       "fn":exp.model_parameters.fn[0],
                                                       "yield_flux":exp.model_parameters.yield_flux.ct})
    generated_yield_case3,___=exp.reduce_raw_data(true_yield,generated_counts_case3,model_parameters,False)
    if(np.allclose(generated_counts_case3.ctg.values,generated_counts_no_sample.ctg.values)):
        print("⇉ Count data is the same with or without count data and parameter sampling.")
        print("")
        if(failure_logging):
            print("Sampled data:")
            print(generated_counts_case3.ctg.values)
            print("")
            print("Non-sampled data:")
            print(generated_counts_no_sample.ctg.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_counts_case3.ctg.values,generated_counts_no_sample.ctg.values))
            print("")
        passed=False
    if(np.allclose(generated_yield_case3.exp.values,generated_yield_no_sample.exp.values)):
        print("⇉ Yield data is the same with or without count data and parameter sampling.")
        print("")
        if(failure_logging):
            print("Sampled data:")
            print(generated_yield_case3.exp.values)
            print("")
            print("Non-sampled data:")
            print(generated_yield_no_sample.exp.values)
            print("")
            print("Comparison test:")
            print(np.allclose(generated_yield_case3.exp.values,generated_yield_no_sample.exp.values))
            print("")
        passed=False
    
    if(passed):
        print("✓ Program passed sampling test.")
    else:
        print("✘ Program failed sampling test.")
    return passed



def monte_carlo_test(convergence_tolerance=0.01,uncertainty_tolerance=0.05,run_count=3000,failure_logging=False):
    """Summary
    ----------
        Tests the yield function to ensure that, given enough iterations, the sampled data converges to the input data

    Raises
    ------
        "The returned values are not close to the mean with a relative tolerance of 1%"
        
        "The returned values are not normally distributed"
    """
    
    print("➤ Testing statistics of function")
    print("➤ Expected functionality: The returned data is normally distributed and it's mean converges to the input data given enough iterations")
    print("")
    
    exp_model=Experimental_Model()
    grid_spacing=len(exp_model.energy_grid)
    true_yield = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    exp = Capture_Yield_RPI()
    
    yields=np.empty((grid_spacing,run_count,2))
    counts=np.empty((grid_spacing,run_count,3))
    for run in range(run_count):
        exp.approximate_unknown_data(exp_model,True)
        generated_counts=exp.generate_raw_data(true_yield,exp.model_parameters,True)
        model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0],
                                                           "trig_bg":exp.model_parameters.trig_bg[0],
                                                           "trig_f":exp.model_parameters.trig_f[0],
                                                           "trig_bf":exp.model_parameters.trig_bf[0],
                                                           "fn":exp.model_parameters.fn[0],
                                                           "yield_flux":exp.model_parameters.yield_flux.ct})
        yield_run,___=exp.reduce_raw_data(true_yield,generated_counts,model_parameters,False)
        yields[:,run,0]=yield_run.exp.values
        yields[:,run,1]=yield_run.exp_unc.values
        counts[:,run,0]=generated_counts.ctg.values
        counts[:,run,1]=generated_counts.dctg.values

    if(np.allclose(np.mean(yields[:,:,0],1),true_yield.true.values,rtol=convergence_tolerance)):
        if(np.allclose(np.std(yields[:,:,0],1),np.mean(yields[:,:,1],1),rtol=uncertainty_tolerance)):
            print("✓ Program passed convergence test.")
            return True
        else:
            print("⇉ Distribution width is different than reported uncertainty.")
            print("")
            print("✘ Program failed convergence test.")
            if(failure_logging):
                print("")
                print("Standard Deviations:")
                print(np.std(yields[:,:,0],1))
                print("")
                print("Reported Uncertainty:")
                print(np.mean(yields[:,:,1],1))
                print("")
                print("Average Difference:")
                print(np.mean(np.abs(np.std(yields[:,:,0],1)-np.mean(yields[:,:,1],1))))
                print("")
                print("Realtive Average Difference:")
                print(np.mean(np.abs(np.std(yields[:,:,0],1)-np.mean(yields[:,:,1],1)))/np.mean(np.mean(yields[:,:,1],1)))
                print("")
                print("Comparisson Value:")
                print(np.allclose(np.std(yields[:,:,0],1),np.mean(yields[:,:,1],1),rtol=uncertainty_tolerance))
                print("")
            return False
            
    else:
        print("⇉ Mean does not converge to true.")
        print("")
        print("✘ Program failed convergence test.")
        if(failure_logging):
            print("")
            print("Mean Values:")
            print(np.mean(yields[:,:,0],1))
            print("")
            print("Average Difference")
            print(np.mean(np.abs(np.mean(yields[:,:,0],1)-true_yield.true.values)))
            print("")
            print("Realtive Average Difference:")
            print(np.mean(np.abs(np.mean(yields[:,:,0],1)-true_yield.true.values))/np.mean(true_yield.true.values))
            print("")
            print("Comparisson Value:")
            print(np.allclose(np.mean(yields[:,:,0],1),true_yield.true.values,rtol=convergence_tolerance))
            print("")
        return False



def covariance_test(failure_logging=False):
    """Summary
    ----------
        Tests the yield function to ensure that, given enough iterations, the sampled data converges to the input data

    Raises
    ------
        "The returned values are not close to the mean with a relative tolerance of 1%"
        
        "The returned values are not normally distributed"
    """
    
    passed=True
    print("➤ Testing covariance functionality")
    print("➤ Expected functionality: When off diagonal values are set to zero in covariance matrix, the diagongal of the matrix should match the variance reported when not using covaraince calculations.")
    print("")
    
    exp_model=Experimental_Model()
    yield_true = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    exp = Capture_Yield_RPI()
    exp.covariance_data={"cr_flux_cov":0,"br_flux_cov":0}
    exp.approximate_unknown_data(exp_model,True)
    generated_counts=exp.generate_raw_data(yield_true,exp.model_parameters,True)
    model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0], 
                                                           "trig_bg":exp.model_parameters.trig_bg[0],
                                                           "trig_f":exp.model_parameters.trig_f[0], 
                                                           "trig_bf":exp.model_parameters.trig_bf[0],
                                                           "fn":exp.model_parameters.fn[0],
                                                           "yield_flux":exp.model_parameters.yield_flux.ct})
    yield_run_T,covariance_data_T=exp.reduce_raw_data(yield_true,generated_counts,model_parameters,True,calculate_covariance_matrix=False)
    yield_run_F,covariance_data_F=exp.reduce_raw_data(yield_true,generated_counts,model_parameters,False)
    full_matrix=covariance_data_T["Cov_Y_jac"].T@covariance_data_T["Cov_Y_cov"]@covariance_data_T["Cov_Y_jac"]
    
    if(not(np.allclose(yield_run_T.exp_unc.values,yield_run_F.exp_unc.values))):
        print("⇉ Reported uncertainty is different between using and not using covariance calculations.")
        print("")
        if(failure_logging):
            print("Reported Uncertainty with calcs:")
            print(yield_run_T.exp_unc.values)
            print("")
            print("Reported Uncertainty without calcs:")
            print(yield_run_F.exp_unc.values)
            print("")
        passed=False
    if(not(np.allclose(np.sqrt(np.diag(full_matrix)),yield_run_T.exp_unc.values))):
        print("⇉ Diagonal of covariance matrix does not match reported uncertainty.")
        print("")
        if(failure_logging):
            print("Diagonal of covariance matrix:")
            print(np.sqrt(np.diag(covariance_data_T["Cov_Y"])))
            print("")
            print("Reported Uncertainty:")
            print(yield_run_T.exp_unc.values)
            print("")
        passed=False
    if(("Cov_Y" in covariance_data_F)or("Cov_Y_jac" in covariance_data_F)or("Cov_Y_cov" in covariance_data_F)):
        print("⇉ Covariance data is included in output when not using covariance calculations.")
        print("")
        passed=False
    
    if(passed):
        print("✓ Program passed covariance test.")
        return True
    else:
        print("✘ Program failed covariance test.")
        return False



def monte_carlo_propogation(run_count):
    exp_model=Experimental_Model()
    grid_spacing=len(exp_model.energy_grid)
    yield_true = pd.DataFrame({'tof':exp_model.tof_grid, 'E':exp_model.energy_grid, 'true':np.ones(len(exp_model.energy_grid))*1000})
    exp = Capture_Yield_RPI()
    
    yields=np.empty((grid_spacing,run_count,2))
    counts=np.empty((grid_spacing,run_count,3))
    for run in range(run_count):
        exp.approximate_unknown_data(exp_model,True)
        generated_counts=exp.generate_raw_data(yield_true,exp.model_parameters,False)
        # model_parameters=exp.sample_true_model_parameters({"trig_g":exp.model_parameters.trig_g[0],
        #                                                    "trig_bg":exp.model_parameters.trig_bg[0],
        #                                                    "trig_f":exp.model_parameters.trig_f[0],
        #                                                    "trig_bf":exp.model_parameters.trig_bf[0],
        #                                                    "fn":exp.model_parameters.fn[0],
        #                                                    "yield_flux":exp.model_parameters.yield_flux.ct})
        yield_run,___=exp.reduce_raw_data(yield_true,generated_counts,exp.model_parameters,False)
        yields[:,run,0]=yield_run.exp.values
        yields[:,run,1]=yield_run.exp_unc.values
        counts[:,run,0]=generated_counts.ctg.values
        counts[:,run,1]=generated_counts.dctg.values
    
    plt.fill_between(exp_model.energy_grid,np.mean(yields[:,:,0],1)+np.std(yields[:,:,0],1),np.mean(yields[:,:,0],1)-np.std(yields[:,:,0],1),color=(0,0.8,0,0.5),label="Monte Carlo Distribution STD")
    plt.fill_between(exp_model.energy_grid,np.mean(yields[:,:,0],1)+np.mean(yields[:,:,1],1),np.mean(yields[:,:,0],1)-np.mean(yields[:,:,1],1),color=(0.2,0.2,0.2,0.5),label="Average Reported Uncertainty")
    plt.plot(exp_model.energy_grid,yield_true.true,color=(0.2,0.2,0.2,1))
    plt.plot(exp_model.energy_grid,np.mean(yields[:,:,0],1),color=(0,0.8,0,1))
    plt.legend()
    plt.show()
    
    print("Yield Distribution STD: {std}".format(std=np.mean(np.std(yields[:,:,0],1))))
    print("Yield Mean Reported Uncertainty: {un}".format(un=np.mean(np.mean(yields[:,:,1],1))))
    
    plt.fill_between(exp_model.energy_grid,np.mean(counts[:,:,0],1)+np.std(counts[:,:,0],1),np.mean(counts[:,:,0],1)-np.std(counts[:,:,0],1),color=(0.6,0,0,0.5),label="Monte Carlo Distribution STD")
    plt.fill_between(exp_model.energy_grid,np.mean(counts[:,:,0],1)+np.mean(counts[:,:,1],1),np.mean(counts[:,:,0],1)-np.mean(counts[:,:,1],1),color=(0.2,0.2,0.2,0.5),label="Average Reported Uncertainty")
    plt.plot(exp_model.energy_grid,np.mean(counts[:,:,0],1),color=(0.6,0,0,1))
    plt.legend()
    plt.show()



print("")
if(basic_functionality_test(failure_logging=True)):
    print("\n\n")
    if(data_type_test()):
        print("\n\n")
        if(paramater_sampling_test()):
            print("\n\n")
            if(no_sampling_test(failure_logging=True)):
                print("\n\n")
                if(sampling_test(failure_logging=True)):
                    print("\n\n")
                    if(monte_carlo_test(failure_logging=True)):
                        print("\n\n")
                        covariance_test(failure_logging=True)
print("")

# monte_carlo_propogation(run_count=100)