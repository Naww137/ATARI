#%%
import numpy as np
import pandas as pd
import h5py
import os

from ATARI.utils.io.theoretical_parameters import BuildTheoreticalParameters_fromHDF5, BuildTheoreticalParameters_fromATARI, DirectTheoreticalParameters
from ATARI.syndat.particle_pair import Particle_Pair


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
est_file = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125/iFB_ests.hdf5"
case_file = "/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/SLBW_noexp/lasso/Ta181_500samples_E75_125/Ta181_500samples_E75_125_1.hdf5"
dataset_range = (0,480)

for isamp in range(min(dataset_range), max(dataset_range)):

    # siglevels = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.5]
    # siglevels = np.linspace(0.001, 0.9, 50)
    siglevels = np.logspace(-4,-2, 10)
    siglevels[0] = np.round(siglevels[0], 4)

    # if os.path.isfile(f'/home/nwalton1/reg_perf_tests/lasso/E75_125/outfiles/par_est_iFB_{isamp}_pv_0001.csv'):
    for sig in siglevels:
        sig_str = str(sig).split('.')[1][0:7]
        # est_par = pd.read_csv(os.path.join("/home/nwalton1/reg_perf_tests/lasso/E75_125/outfiles", f'par_est_iFB_{isamp}_pv_{sig_str}.csv'), index_col=0)
        # est_par_builder = BuildTheoreticalParameters_fromATARI(f'par_est_iFB_{isamp}_pv_{sig_str}', est_par, Ta_pair)
        # est_par = est_par_builder.construct()
        est_par_builder = BuildTheoreticalParameters_fromHDF5(f'par_est_iFB_{isamp}_pv_{sig_str}', est_file, isamp, Ta_pair)
        est_par = est_par_builder.construct()
        est_par.to_hdf5(case_file, isamp)
    print(f"Completed sample {isamp}")

    # else:
    #     with open(os.path.join("/home/nwalton1/reg_perf_tests/lasso/E75_125", f'template_resub_job.sh'), 'r') as f:
    #         template = f.readlines()
    #         f.close()
    #     with open(os.path.join("/home/nwalton1/reg_perf_tests/lasso/E75_125", f'resub_job_{isamp}.sh'), 'w') as f:
    #         for line in template:
    #             if line.startswith('python'):
    #                 f.write(f"python3 run_fitting_alg_v2.py {isamp} \n""")
    #             else:
    #                 f.write(line)

    #     print(f'Re-run sample: {isamp}')




# %%
# with h5py.File(case_file, 'r') as f:
#     print(f['sample_14'].keys())
#     f.close()

# %%
