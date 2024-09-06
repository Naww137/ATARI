

import pandas as pd

import os
from ATARI.utils import atario
import h5py
from ATARI.PiTFAll.performance_test import PerformanceTest
import ATARI.PiTFAll.file_handler as pfh


syndat_control = atario.load_general_object("/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/Ta181_Syndat_FullRegion/cluster/SyndatControl_E200_5000.pkl")
particle_pair = syndat_control.particle_pair
sammy_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'
sample_file = os.path.realpath("/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/Ta181_Syndat_FullRegion/cluster/Samples_E200_5000.hdf5")


# sample_file = "/Users/noahwalton/Documents/GitHub/ATARI/tests/test.hdf5"
perf_test_file = "/Users/noahwalton/Documents/GitHub/ATARI/tests/perf_test.hdf5"

# h5f = h5py.File(sample_file, "r")
# isamples = [each.split('_')[1] for each in h5f.keys()]
# h5f.close()
# out_list_0 = syndatOUT.from_hdf5(sample_file, isamples[0])
# print(out_list_0)

os.remove(perf_test_file)
# pf = PerformanceTest(perf_test_file, syndat_sample_filepath=sample_file) # give energy range here
# pf.add_synthetic_data_samples(sample_file)
# pf.calculate_doppler_theoretical(sammy_exe,particle_pair)

pfh.add_synthetic_data_samples(perf_test_file,
                                sample_file,
                                isample_max=5, 
                                energy_range=(200,225))

# print(pd.read_hdf(perf_test_file, "sample_0/exp_dat_cap1mm/pw_reduced"))
par_fit = pd.read_hdf(perf_test_file, "sample_0/par_true")
exp_df_dict = {}
for exp in ["cap1mm", "cap2mm", "trans1mm", "trans3mm", "trans6mm"]:
    exp_df_dict[exp] = pd.read_hdf(perf_test_file, f"sample_0/exp_dat_{exp}/pw_reduced")[["E", "true"]]

# print(pd.read_hdf(perf_test_file, "sample_0/exp_dat_trans1mm/cov_data/Jac_sys"))
# print(pd.read_hdf(perf_test_file, "sample_0/exp_dat_trans1mm/cov_data/diag_stat"))

pfh.add_model_fit(perf_test_file,
                   0,
                   "fit_1",
                   par_fit,
                   exp_df_dict)

# for exp in ["cap1mm", "cap2mm", "trans1mm", "trans3mm", "trans6mm"]:
#     print(pd.read_hdf(perf_test_file, f"sample_10/exp_dat_{exp}/pw_reduced"))

pfh.add_fine_grid_doppler_only(perf_test_file,
                               5,
                               (200,225),
                               sammy_exe,
                               particle_pair,
                               model_title = "true"
                               )

print(pd.read_hdf(perf_test_file, "sample_0/theo_pw"))

pfh.add_fine_grid_doppler_only(perf_test_file,
                               1,
                               (200,225),
                               sammy_exe,
                               particle_pair,
                               model_title = "fit_1"
                               )
print(pd.read_hdf(perf_test_file, "sample_0/theo_pw"))
