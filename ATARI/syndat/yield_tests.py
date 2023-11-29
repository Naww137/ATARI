# cSpell:enable

import numpy as np
import pandas as pd
from scipy.stats import normaltest
from ATARI.syndat.yield_rpi import syndat_Y

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



def basic_functionality_test():
    """Summary
    ----------
        Tests the function for basic functionality. Allows for more nuanced error correction.

    Raises
    ------
        "Program runs"
        
        "Program runs, but no output data is provided."
    """
    
    
    print("Testing that the code runs")
    print("Expected functionality: The program functions and returns data")
    print("")
    
    true_yield = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([1000,1000,1000])})
    
    exp = syndat_Y()
    exp.run(true_yield)
    
    try:
        exp.data.exp
        exp.data.exp_unc
        print("Program runs.")
    except:
        print("Program runs, but no output data is provided.")



def test_mean_converges_to_true():
    """Summary
    ----------
        Tests the yield function to ensure that, given enough iterations, the sampled data converges to the input data

    Raises
    ------
        "The returned values are not close to the mean with a relative tolerance of 1%"
        
        "The returned values are not normally distributed"
    """
    
    
    print("Testing that the mean of returned data converges to input")
    print("Expected functionality: The returned data is normally distributed and it's mean converges to the input data given enough iterations")
    print("")
    
    input_options = {}

    ipert = 1000
    exp_yield = np.zeros([ipert,3])
    exp_yield_unc = np.zeros([ipert,3])
    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([0.8,0.8,0.8])})

    exp = syndat_Y(options=input_options)
    exp.run(df_true)

    for i in range(ipert):
        exp.run(df_true)
        exp_yield[i,:] = np.array(exp.data.exp)
        exp_yield_unc[i,:] = np.array(exp.data.exp_unc)

    true_yield = np.array(exp.data.sort_values('E', ascending=False)["true"])
    assert (np.all(np.isclose(np.mean(exp_yield, axis=0), true_yield, rtol=1e-2))), "The returned values are not close to the mean with a relative tolerance of 1%"
    assert (np.all(normaltest((exp_yield-true_yield)/exp_yield_unc).pvalue>1e-5)), "The returned values are not normally distributed"

    print("Passed test, data is normally distributed and mean of data converges to input")



def no_sampling_returns_same_values(success_logging=False,failure_logging=True):
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
    
    
    print("Testing similarity between return data and input data with sampling disabled")
    print("Expected functionality: The returned data is identical to the input data if all sampling is disabled")
    print("")
    
    
    #Options to turn off all problem sampling
    input_options = {'Sample Counting Noise' : False,
                     'Sample TURP'           : False,
                     'Sample TNCS'           : False}


    #Simple test space for the function
    true_yield = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([1000,1000,1000])})
    
    exp = syndat_Y(options=input_options)
    exp.run(true_yield)
    
    exp_yield = np.array(exp.data.exp)
    
    
    #Tests the similarity of the arrays and prints results
    if(np.array_equal(np.array(true_yield.true), exp_yield)):
        print("Passed test, all returned values are the same as input")
        if(success_logging):
            print("")
            print("Additional Information:")
            print("")
            print(" Input values: ",np.array(true_yield.true))
            print(" Output values:",exp_yield)
            print(" Difference:   ",exp_yield-np.array(true_yield.true))
    else:
        print("Failed test, returned values are not the same")
        if(failure_logging):
            print("")
            print("Additional Information:")
            print("")
            print(" Input values: ",np.array(true_yield.true))
            print(" Output values:",exp_yield)
            print(" Difference:   ",exp_yield-np.array(true_yield.true))



def sampling_returns_different_values(success_logging=False,failure_logging=True):
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
    
    
    print("Testing difference between returned data and input data with sampling enabled")
    print("Expected functionality: returned values are different if any sampling is enabled")
    print("")
    
    
    #Simple test space for the function
    true_yield = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([1000,1000,1000])})
    
    
    #Test with only count rate noise sampling
    #----------------------------------------
    
    #Options to turn on Counting Noise Sample
    input_options = {'Sample Counting Noise' : True,
                     'Sample TURP'           : False,
                     'Sample TNCS'           : False}

    exp = syndat_Y(options=input_options)
    exp.run(true_yield)
    
    CN_exp_yield = np.array(exp.data.exp)
    
    #Tests the similarity of the arrays
    CN_test=int(not(np.array_equal(np.array(true_yield.true), CN_exp_yield)))
    
    
    #Test with only count rate noise sampling
    #----------------------------------------
    
    #Options to turn on Counting Noise Sample
    input_options = {'Sample Counting Noise' : False,
                     'Sample TURP'           : True,
                     'Sample TNCS'           : False}

    exp = syndat_Y(options=input_options)
    exp.run(true_yield)
    
    TURP_exp_yield = np.array(exp.data.exp)
    
    #Tests the similarity of the arrays
    TURP_test=int(not(np.array_equal(np.array(true_yield.true), TURP_exp_yield)))
    
    
    #Test with only count rate noise sampling
    #----------------------------------------
    
    #Options to turn on Counting Noise Sample
    input_options = {'Sample Counting Noise' : False,
                     'Sample TURP'           : False,
                     'Sample TNCS'           : True}

    exp = syndat_Y(options=input_options)
    exp.run(true_yield)
    
    TNCS_exp_yield = np.array(exp.data.exp)
    
    #Tests the similarity of the arrays
    TNCS_test=int(not(np.array_equal(np.array(true_yield.true), TNCS_exp_yield)))
    
    
    #Result Printing
    #---------------
    if(CN_test + TURP_test + TNCS_test == 3):
        print("Passed all three tests, sampling returns different values than input")
        if(success_logging):
            print("")
            print("Additional Information:")
            print("")
            print("Input values:            ",np.array(true_yield.true))
            print("")
            print("Count Rate Output values:",CN_exp_yield)
            print("Count Rate Difference:   ",CN_exp_yield-np.array(true_yield.true))
            print("")
            print("TURP Output values:      ",TURP_exp_yield)
            print("TURP Difference:         ",TURP_exp_yield-np.array(true_yield.true))
            print("")
            print("TNCS Output values:      ",TNCS_exp_yield)
            print("TNCS Difference:         ",TNCS_exp_yield-np.array(true_yield.true))
    else:
        num_failed_tests=3-(CN_test+TURP_test+TNCS_test)
        if(num_failed_tests==1):
            print("Failed",3-(CN_test+TURP_test+TNCS_test),"test.")
        else:
            print("Failed",3-(CN_test+TURP_test+TNCS_test),"tests.")
        print("")
        if(CN_test!=1):
            print("Sampling count rate noise does not return unique values.")
        if(TURP_test!=1):
            print("Sampling TURP noise does not return unique values.")
        if(TNCS_test!=1):
            print("Sampling TNCS noise does not return unique values.")
        if(failure_logging):
            print("")
            print("Additional Information:")
            print("")
            print(" Input values:            ",np.array(true_yield.true))
            if(CN_test!=1):
                print("")
                print(" Count Rate Output values:",CN_exp_yield)
                print(" Count Rate Difference:   ",CN_exp_yield-np.array(true_yield.true))
            if(TURP_test!=1):
                print("")
                print(" TURP Output values:      ",TURP_exp_yield)
                print(" TURP Difference:         ",TURP_exp_yield-np.array(true_yield.true))
            if(TNCS_test!=1):
                print("")
                print(" TNCS Output values:      ",TNCS_exp_yield)
                print(" TNCS Difference:         ",TNCS_exp_yield-np.array(true_yield.true))

print("")
basic_functionality_test()
print("\n\n")
test_mean_converges_to_true()
print("\n\n")
no_sampling_returns_same_values()
print("\n\n")
sampling_returns_different_values()
print("")
