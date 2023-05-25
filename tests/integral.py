#%%
import numpy as np
import pandas as pd
from ATARI.syndat.experiment import Experiment
from scipy.stats import normaltest
import unittest


class TestNoiseDistributions(unittest.TestCase):

    def test_mean_converges_to_true(self):
        input_options = {'Add Noise': True,
                        'Calculate Covariance': True,
                        'Compression Points':[],
                        'Grouping Factors':None}

        experiment_parameters = {'bw': {'val':0.0256,    'unc'   :   0}}
        exp = Experiment(energy_domain=None, 
                                input_options=input_options, 
                                experiment_parameters=experiment_parameters)

        ipert = 10000
        exp_trans = np.zeros([ipert,3])
        exp_trans_unc = np.zeros([ipert,3])
        df_true = pd.DataFrame({'E':[10, 1000, 3000], 'theo_trans':np.array([0.8,0.8,0.8])})

        for i in range(ipert):
            exp.run(df_true)
            exp_trans[i,:] = np.array(exp.trans.exp_trans)
            exp_trans_unc[i,:] = np.array(exp.trans.exp_trans_unc)

        theo_trans = np.array(exp.theo.sort_values('E', ascending=False).theo_trans)
        self.assertTrue(np.all(np.isclose(np.mean(exp_trans, axis=0), theo_trans, rtol=1e-2)))
        self.assertTrue(np.all(normaltest((exp_trans-theo_trans)/exp_trans_unc).pvalue>1e-5))

unittest.main()


# %%


