#%%
import numpy as np
import pandas as pd
import h5py
from ATARI.PiTFAll.sample_case import csv_2_hdf5
import os
#%%
# dataset_range = (0,5)
# directory = '/home/nwalton1/reg_perf_tests/perf_tests/staticladder'
# case_file = 'perf_test_staticladder.hdf5'


# for fit in os.listdir():

#     if os.path.isdir(fit):

#         fit_directory = os.path.join(directory, fit)

#         splitfit = fit.split('_')
#         matfile = '_'.join([splitfit[0], 'fit', splitfit[-1]])

#         for i in range(min(dataset_range), max(dataset_range)):

#             try:
#                 csv_2_hdf5(fit_directory, os.path.join(directory,case_file), i, f'{fit}_pp')
#             except:

#                 with open(os.path.join(directory, f'resub_job_template.sh'), 'r') as f:
#                     template = f.readlines()
#                     f.close()
#                 with open(os.path.join(fit_directory, f'resub_job_{isamp}.sh'), 'w') as f:
#                     for line in template:
#                         if line.startswith('matlab'):
#                             f.write(f"""matlab -nodisplay -batch "{matfile}('{os.path.join(directory,case_file)}', {i})" \n""")
#                         else:
#                             f.write(line)

#                 print(f'Re-run fit {fit} case {i}')
#%%
import ATARI.atari_io.hdf5 as io
from ATARI.utils.misc import fine_egrid 
from ATARI.utils.io.datacontainer import DataContainer
from ATARI.utils.io.pointwise import PointwiseContainer
from ATARI.utils.io.parameters import TheoreticalParameters, ExperimentalParameters
from ATARI.models.particle_pair import Particle_Pair


ac = 0.81271    # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity [sic: perhaps, angular momentum?]
l_max = 1       # highest order l-wave to consider

spin_groups = [ (3.0,1,0) ] # (4.0,1,0) 
average_parameters = pd.DataFrame({ 'dE'    :   {'3.0':8.79, '4.0':4.99},
                                    'Gg'    :   {'3.0':46.4, '4.0':35.5},
                                    'gn2'    :   {'3.0':64.0, '4.0':64.0}  })

Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                spin_groups=spin_groups, average_parameters=average_parameters,
                                input_options={})

# %%
#case_file = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125.hdf5"
#directory = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125"
case_file = "/home/nwalton1/reg_perf_tests/lasso/E75_125/Ta181_500samples_E75_125.hdf5"
directory = "/home/nwalton1/reg_perf_tests/lasso/E75_125/Ta181_500samples_E75_125"
dataset_range = (0,480)

for isamp in range(min(dataset_range), max(dataset_range)):
    
    try:
        sample_directory = os.path.join(directory, f'sample_{isamp}')

        # read syndat
        exp_cov = pd.read_csv(os.path.join(sample_directory, 'cov.csv'), index_col=0)
        true_par = pd.read_csv(os.path.join(sample_directory, 'ladder.csv'), index_col=0)
        pw_exp = pd.read_csv(os.path.join(sample_directory, 'transm.csv'), index_col=0)
        exp_cov.columns = pw_exp.E
        exp_cov.index = pw_exp.E
        exp_cov.index.name = None

        # read estimate
        est_par = pd.read_csv(os.path.join(sample_directory, 'par_est.csv'), index_col=0)

        # create ATARIO instances
        threshold_0T = 1e-2
        exp_par = ExperimentalParameters(0.067166, 0, threshold_0T)
        theo_par = TheoreticalParameters(Ta_pair, true_par, 'true')
        fit_par = TheoreticalParameters(Ta_pair, est_par, 'fit')

        pwfine = pd.DataFrame({'E':fine_egrid(pw_exp.E,100)})
        pw = PointwiseContainer(pw_exp, pwfine)
        pw.add_experimental(pw_exp, exp_cov, exp_par)
        pw.add_model(theo_par, exp_par)

        dc = DataContainer(pw, exp_par, theo_par, {'fit':fit_par})
        dc.to_hdf5(case_file, isamp)
        print(f"Completed sample {isamp}")

    except:

        with open(os.path.join("/home/nwalton1/reg_perf_tests/lasso/E75_125", f'resub_job_template.sh'), 'r') as f:
            template = f.readlines()
            f.close()
        with open(os.path.join("/home/nwalton1/reg_perf_tests/lasso/E75_125", f'resub_job_{isamp}.sh'), 'w') as f:
            for line in template:
                if line.startswith('python'):
                    f.write(f"python3 run_fitting_alg.py {isamp} \n""")
                else:
                    f.write(line)

        print(f'Re-run sample: {isamp}')



# %%
with h5py.File(case_file, 'r') as f:
    print(f['sample_0'].keys())
    f.close()

# %%
