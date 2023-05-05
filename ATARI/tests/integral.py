#%%
import numpy as np
import pandas as pd

from ATARI.syndat.experiment import Experiment

#%%

input_options = {'Add Noise': True,
                'Calculate Covariance': True,
                'Compression Points':[],
                'Grouping Factors':None}

experiment_parameters = {'bw': {'val':0.0256,    'unc'   :   0}}

# initialize experimental setup
exp = Experiment(energy_domain=None, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)

#%%
ipert = 10000
exp_trans = np.zeros([ipert,3])
exp_trans_unc = np.zeros([ipert,3])
df_true = pd.DataFrame({'E':[10, 1000, 3000], 'theo_trans':np.array([0.8,0.8,0.8])})

for i in range(ipert):
    exp.run(df_true)
    exp_trans[i,:] = np.array(exp.trans.exp_trans)
    exp_trans_unc[i,:] = np.array(exp.trans.exp_trans_unc)
E = np.array(exp.trans.E)
theo_trans = np.array(exp.theo.sort_values('E', ascending=False).theo_trans)

# %%
print(E)
print(theo_trans)
print(np.mean(exp_trans, axis=0))
# %%
