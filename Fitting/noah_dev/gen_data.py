import numpy as np
import pandas as pd
import os
# import nuctools
from copy import copy

from ATARI.syndat.particle_pair import Particle_Pair
from ATARI.syndat.experiment import Experiment
from ATARI.syndat.MMDA import generate
from ATARI.theory.xs import SLBW
from ATARI.sammy_interface import sammy_interface, sammy_classes, sammy_functions
from ATARI.theory.resonance_statistics import make_res_par_avg

import ATARI.utils.io.hdf5 as h5io



ac = 0.81271    # scattering radius in 1e-12 cm 
M = 180.948030  # amu of target nucleus
m = 1           # amu of incident neutron
I = 3.5         # intrinsic spin, positive parity
i = 0.5         # intrinsic spin, positive parity
l_max = 1       # highest order l-wave to consider



experiment_parameters = {'bw': {'val':0.1024,    'unc'   :   0}} 
input_options = {'Add Noise': True,
            'Calculate Covariance': False,
            'Sample TURP': True}

energy_grid = [75, 125]
exp = Experiment(energy_grid, 
                        input_options=input_options, 
                        experiment_parameters=experiment_parameters)

Gg_DOF = 10
spin_groups = [ (3.0,1,0) ]
res_par_avg = make_res_par_avg(D_avg = 8.79, 
                            Gn_avg= 0.658, #0.658, 
                            n_dof = 1, 
                            Gg_avg = 64.0, 
                            g_dof = Gg_DOF, 
                            print = False)

average_parameters = {'3.0':res_par_avg}
Ta_pair = Particle_Pair( ac, M, m, I, i, l_max,
                                input_options={},
                                spin_groups=spin_groups,
                                average_parameters=average_parameters )   



sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    model = 'XCT',
    reaction = 'total',
    solve_bayes = False,
    inptemplate= "allexptot_1sg.inp",
    energy_window = None,
    sammy_runDIR = 'SAMMY_runDIR',
    keep_runDIR = False,
    shell = 'zsh'
    )

sammyINP = sammy_classes.SammyInputData(
    particle_pair = Ta_pair,
    resonance_ladder = pd.DataFrame(),
    energy_grid = np.sort(exp.energy_domain),
    # energy_grid = exp.energy_domain,
    temp = 304.5,
    FP=35.185,
    frac_res_FP=0.049600,
    target_thickness=0.067166)



case_file = '/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/fit_w_sammy/data_capcorr.hdf5'  # if using hdf5

dataset_range = (0, 10)

# samples_not_generated = generate(Ta_pair, exp, 
#                                         'sammy', 
#                                         dataset_range, 
#                                         case_file,
#                                         fixed_resonance_ladder=None, 
#                                         open_data=None,
#                                         vary_Erange=None,
#                                         use_hdf5=True,
#                                         overwrite = False,
#                                         sammy_RTO=sammyRTO,
#                                         sammy_INP=sammyINP
#                                                                     )



for isample in range(max(dataset_range)):

    true_resladder = Ta_pair.sample_resonance_ladder(energy_grid, spin_groups, average_parameters)
    sammyINP.resonance_ladder = true_resladder

    # transmission data generation
    sammyRTO.inptemplate = "allexptot_1sg.inp"
    sammyRTO.reaction = "transmission"
    sammyOUT = sammy_functions.run_sammy(sammyINP, sammyRTO)
    exp.run(sammyOUT.pw)

    exp.trans.sort_values(by=["E"], ascending=True, inplace=True)
    exp.trans.reset_index(inplace=True, drop=True)
    exp.trans.drop("tof", axis=1, inplace=True)

    exp.trans.rename(columns={"exp_trans":"exp", "exp_trans_unc":"exp_unc"}, inplace=True)
    exp.trans["true"] = sammyOUT.pw["theo_trans"]


    # capture data generation
    sammyRTO.inptemplate = "allexpcap_1sg.inp"
    sammyRTO.reaction = "capture"
    out_cap = sammy_functions.run_sammy(sammyINP, sammyRTO)

    unc_scale = 0.1
    cap_pw = copy(out_cap.pw[["E"]])
    cap_pw["true"] = out_cap.pw["theo_xs"]
    cap_pw["exp"] = abs(np.random.default_rng().normal(out_cap.pw["theo_xs"], np.sqrt(out_cap.pw["theo_xs"])*unc_scale ))
    cap_pw["exp_unc"] = np.sqrt(cap_pw["exp"])*unc_scale

    # write to file
    h5io.write_par(case_file, isample, true_resladder, 'true')

    datasets = [exp.trans, cap_pw]
    dataset_titles = ["trans1", "cap1"]
    for ds,dt in zip(datasets, dataset_titles):
        h5io.write_pw_exp(case_file, isample, ds, title=dt, CovT=None, CovXS=None)
    

