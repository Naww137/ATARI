import numpy as np
import pandas as pd
from scipy.stats import normaltest
from ATARI.syndat.yield_rpi import syndat_Y

extra_logging=False
Failure_logging=True

def test_mean_converges_to_true():
    input_options = {}

    ipert = 10000
    exp_trans = np.zeros([ipert,3])
    exp_trans_unc = np.zeros([ipert,3])
    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([0.8,0.8,0.8])})

    exp = syndat_Y(options=input_options)
    exp.run(df_true)

    for i in range(ipert):
        exp.run(df_true)
        exp_trans[i,:] = np.array(exp.data.exp)
        exp_trans_unc[i,:] = np.array(exp.data.exp_unc)

    true_trans = np.array(exp.data.sort_values('E', ascending=False)["true"])
    assert (np.all(np.isclose(np.mean(exp_trans, axis=0), true_trans, rtol=1e-2)))
    assert (np.all(normaltest((exp_trans-true_trans)/exp_trans_unc).pvalue>1e-5))

    print("Passed test mean_converges_to_true for syndat.yield_rpi.syndat_Y")


def no_sampling_returns_same_values():
    #Options to turn off all problem sampling
    input_options = {'Sample Counting Noise' : False,
                     'Sample TURP'           : False,
                     'Sample TNCS'           : False}



    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([1000,1000,1000])})
    
    exp = syndat_Y(options=input_options)
    exp.run(df_true)
    
    exp_trans = np.array(exp.data.exp)
    
    
    
    #Tests the similarity of the arrays
    if(np.array_equal(np.array(df_true.true), exp_trans)):
        print("Passed no sampling return test, all returned values are the same as input")
        if(extra_logging):
            print("Input values: ",exp_trans)
            print("Output values:",np.array(df_true.true))
            print("Difference:   ",abs(exp_trans-np.array(df_true.true)))
    else:
        print("Returned Values without sampling are not the same")
        if(Failure_logging):
            print("Input values: ",exp_trans)
            print("Output values:",np.array(df_true.true))
            print("Difference:   ",abs(exp_trans-np.array(df_true.true)))

def sampling_returns_different_values():
    #Options to turn off all problem sampling
    input_options = {'Sample Counting Noise' : True,
                     'Sample TURP'           : False,
                     'Sample TNCS'           : False}



    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([1000,1000,1000])})
    
    exp = syndat_Y(options=input_options)
    exp.run(df_true)
    
    exp_trans = np.array(exp.data.exp)
    
    
    
    #Tests the similarity of the arrays
    if(not(np.array_equal(np.array(df_true.true), exp_trans))):
        print("Passed sampling return test, returned values are different")
        if(extra_logging):
            print("Input values: ",exp_trans)
            print("Output values:",np.array(df_true.true))
            print("Difference:   ",abs(exp_trans-np.array(df_true.true)))
    else:
        print("Returned Values with sampling are the same")
        if(Failure_logging):
            print("Input values: ",exp_trans)
            print("Output values:",np.array(df_true.true))
            print("Difference:   ",abs(exp_trans-np.array(df_true.true)))

print("")
no_sampling_returns_same_values()
print("")
sampling_returns_different_values()
print("")
