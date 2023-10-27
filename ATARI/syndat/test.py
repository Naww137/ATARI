import numpy as np
import pandas as pd
from scipy.stats import normaltest
from ATARI.syndat.transmission_rpi import syndat_T



def test_mean_converges_to_true():
    input_options = {'Add Noise': True,
                    'Calculate Covariance': True,
                    'Compression Points':[],
                    'Grouping Factors':None}

    ipert = 10000
    exp_trans = np.zeros([ipert,3])
    exp_trans_unc = np.zeros([ipert,3])
    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([0.8,0.8,0.8])})

    exp = syndat_T()
    exp.run(df_true)

    for i in range(ipert):
        exp.run(df_true)
        exp_trans[i,:] = np.array(exp.data.exp)
        exp_trans_unc[i,:] = np.array(exp.data.exp_unc)

    true_trans = np.array(exp.data.sort_values('E', ascending=False)["true"])
    assert (np.all(np.isclose(np.mean(exp_trans, axis=0), true_trans, rtol=1e-2)))
    assert (np.all(normaltest((exp_trans-true_trans)/exp_trans_unc).pvalue>1e-5))

    print("Passed test mean_converges_to_true for syndat.transmission_rpi.syndat_T")


def no_sampling_returns_same_values():
    return

test_mean_converges_to_true()