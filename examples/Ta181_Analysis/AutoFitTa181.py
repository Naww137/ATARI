# %%
import numpy as np
import pandas as pd
import os
import pickle
from ATARI.AutoFit import spin_group_selection
from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.AutoFit.control import AutoFitOUT
from copy import copy

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)


# %% built experimental models

energy_range_all = [201, 210] # 228]

### 1mm capture data
capdat1 = sammy_functions.readlst("./yield_ta1b_unsmooth.dat")
expcap1 = Experimental_Model(title = "cap1mm",
                                reaction ="capture", 
                                energy_range = energy_range_all,
                                n = (0.005631, 0),
                                FP = (45.27, 0.05),
                                burst= (8.0,1.0), 
                                temp= (294.2610, 0.0),
                                channel_widths={
                                    "maxE": [68.20,  122.68, 330.48, 547.57, 199359.52], 
                                    "chw":  [821.40, 410.70, 102.70, 51.30,  25.70],
                                    "dchw": [0.8,    0.8,    0.8,    0.8,    0.8]
                                }
                               )
capdat1 = capdat1[(capdat1.E<max(expcap1.energy_range)) & (capdat1.E>min(expcap1.energy_range))]


### 2mm capture data
capdat2 = sammy_functions.readlst("./yield_ta2_unsmooth.dat")
expcap2 = Experimental_Model(   title = "cap2mm",
                                reaction = "capture", 
                                energy_range = energy_range_all,
                                n = (0.011179, 0.0),
                                FP = (45.27, 0.05),
                                burst = (8.0,1.0),
                                temp = (294.2610, 0.0),
                                channel_widths={
                                    "maxE": [68.20,  122.68, 330.48, 547.57, 199359.52], 
                                    "chw":  [821.40, 410.70, 102.70, 51.30,  25.70],
                                    "dchw": [0.8,    0.8,    0.8,    0.8,    0.8]
                                }
                               )
capdat2 = capdat2[(capdat2.E<max(expcap2.energy_range)) & (capdat2.E>min(expcap2.energy_range))]


### 1mm Transmission data
transdat1 = sammy_functions.readlst("./trans-Ta-1mm.twenty")
transdat1_covfile = './trans-Ta-1mm.idc'
# chw, Emax = get_chw_and_upperE(transdat1.E, 100.14)
exptrans1 = Experimental_Model(title = "trans1mm",
                               reaction = "transmission", 
                               energy_range = energy_range_all,

                                n = (0.00566,0.0),  
                                FP = (100.14,0.0), 
                                burst = (8, 0.0), 
                                temp = (294.2610, 0.0),

                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
                                
                               )
transdat1 = transdat1[(transdat1.E<max(exptrans1.energy_range)) & (transdat1.E>min(exptrans1.energy_range))]

### 3mm transmission data
transdat3 = sammy_functions.readlst("./trans-Ta-3mm.twenty")
transdat3_covfile = "./trans-Ta-3mm.idc"
exptrans3 = Experimental_Model(title = "trans3mm",
                               reaction = "transmission", 
                               energy_range = energy_range_all,

                                n = (0.017131,0.0),  
                                FP = (100.14,0.0), 
                                burst = (8, 0.0), 
                                temp = (294.2610, 0.0),

                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
                                
                               )
transdat3 = transdat3[(transdat3.E<max(exptrans3.energy_range)) & (transdat3.E>min(exptrans3.energy_range))]


### 6mm transmission data
transdat6 = sammy_functions.readlst("./trans-Ta-6mm.twenty")
transdat6_covfile = "./trans-Ta-6mm.idc"
exptrans6 = Experimental_Model(title = "trans6mm",
                               reaction = "transmission", 
                               energy_range = energy_range_all,

                                n = (0.03356,0.0),  
                                FP = (100.14,0.0), 
                                burst = (8, 0.0), 
                                temp = (294.2610, 0.0),

                               channel_widths={
                                    "maxE": [216.16, 613.02, 6140.23], 
                                    "chw": [204.7, 102.4, 51.2],
                                    "dchw": [1.6, 1.6, 1.6]
                                }
                                
                               )
transdat6 = transdat6[(transdat6.E<max(exptrans6.energy_range)) & (transdat6.E>min(exptrans6.energy_range))]


### Not using 12mm measurement for evaluation - this is a validation measurement

# transdat12 = sammy_functions.readlst("/Users/noahwalton/research_local/resonance_fitting/ATARI_workspace/measurement_data/trans-Ta-12mm.dat")
# # transdat12_covfile = Need to generate from sys and stat covariances
# exptrans12 = Experimental_Model(title = "trans12",
#                                 reaction = "transmission",
#                                 energy_range = erange_all,

#                                 sammy_inputs = {
#                                     'alphanumeric'       :   ["BROADENING IS WANTED"],
#                                     'ResFunc'            :   "ORRES"
#                                         },

#                                 n = (0.067166, 0.0),  
#                                 FP = (35.185,0.0), 
#                                 burst = (8,0.0), 
#                                 temp = (294.2610, 0.0),

#                                 channel_widths={
#                                         "maxE": [270], 
#                                         "chw": [102.7],
#                                         "dchw": [0.8]
#                                         },

#                                 additional_resfunc_lines=["WATER 0004 5.6822000 -0.54425 0.07733000", "WATER      0.5000000  0.05000 0.00700000", "LITHI 000  -1.000000  -1.0000 6.00000000", "LITHI      0.1000000  0.10000 0.60000000", "LITHI      166.87839 -28.7093 1.260690", "LITHI      0.2574580 -0.06871 0.004915"]
#                                )

# transdat12 = transdat12[(transdat12.E<max(exptrans12.energy_range)) & (transdat12.E>min(exptrans12.energy_range))]



### setup in zipped lists 
datasets = [capdat1, capdat2, transdat1, transdat3, transdat6]
experiments= [expcap1, expcap2, exptrans1, exptrans3, exptrans6]
covariance_data = [{}, {}, transdat1_covfile, transdat3_covfile, transdat6_covfile]
templates = []
for data, exp in zip(datasets, experiments):
    filepath = f'template_{exp.title}_edited'
    exp.template = os.path.realpath(filepath)



# %% Setup theoretical models

Ta_pair = Particle_Pair(isotope="Ta181",
                        formalism="XCT",
                        ac=8.1271,     # scattering radius
                        M=180.948030,  # amu of target nucleus
                        m=1,           # amu of incident neutron
                        I=3.5,         # intrinsic spin, positive parity
                        i=0.5,         # intrinsic spin, positive parity
                        l_max=2)       # highest order l-wave to consider

Ta_pair.add_spin_group(Jpi='3.0',
                       J_ID=1,
                       D_avg=8.79,
                       Gn_avg=46.5,
                       Gn_dof=1,
                       Gg_avg=64.0,
                       Gg_dof=1000)

Ta_pair.add_spin_group(Jpi='4.0',
                       J_ID=2,
                       D_avg=4.99,
                       Gn_avg=35.5,
                       Gn_dof=1,
                       Gg_avg=64.0,
                       Gg_dof=1000)


#%% autofit function

from ATARI.AutoFit.initial_FB_solve import InitialFB, InitialFBOPT
from ATARI.AutoFit import chi2_eliminator_v2, elim_addit_funcs

def fit(title, maxres, sammy_rto_fit, initialFBopt, spin_group_opt):
    
    ### initial FB
    autofit_initial = InitialFB(initialFBopt)

    initial_out = autofit_initial.fit(Ta_pair,
                                    energy_range_all,
                                    datasets,
                                    experiments,
                                    covariance_data,
                                    sammy_rto_fit)

    start_ladder =  copy(initial_out.final_internal_resonances)
    external_resonances = copy(initial_out.final_external_resonances)
    assert(isinstance(start_ladder, pd.DataFrame))
    assert(isinstance(external_resonances, pd.DataFrame))
    start_ladder['varyGg'] = np.zeros(len(start_ladder))*1
    start_ladder = pd.concat([external_resonances, start_ladder], ignore_index=True)
    external_resonance_indices = [0,1]

    ### do we need to create this sammyINP object here? 
    # can this be done inside the eliminate function with options for all of the fitting parameters?
    # the only information needed is the data, pair, and start_ladder, everything else should be an option for elimOPTs
    elim_sammyINPyw = sammy_classes.SammyInputDataYW(
        particle_pair = Ta_pair,
        resonance_ladder = start_ladder,

        datasets = datasets,
        experimental_covariance=covariance_data,
        experiments = experiments,

        max_steps = 5,
        iterations = 3,
        step_threshold = 0.01,
        autoelim_threshold = None,

        LS = False,
        LevMar = True,
        LevMarV = 1.5,

        minF = 1e-5,
        maxF = 10,
        initial_parameter_uncertainty = 0.05
        )

    ### Eliminate
    elim_opts = chi2_eliminator_v2.elim_OPTs(chi2_allowed = 0,
                                        fixed_resonances_df = external_resonances,
                                        deep_fit_max_iter = 10,
                                        deep_fit_step_thr = 0.1,
                                        start_fudge_for_deep_stage = 0.1,
                                        stop_at_chi2_thr = False
                                        )

    elimi = chi2_eliminator_v2.eliminator_by_chi2(rto=sammy_rto_fit,
                                                sammyINPyw = elim_sammyINPyw , 
                                                options = elim_opts
                                )

    history = elimi.eliminate(ladder_df=start_ladder)
    
    print(f'Eliminated from {history.ladder_IN.shape[0]} res -> {history.ladder_OUT.shape[0]}')
    minkey = min(history.elimination_history.keys())

    ### spin group selection
    spinselector = spin_group_selection.SpinSelect(spin_group_opt)
    minkey = min(history.elimination_history.keys())
    models = [history.elimination_history[i]['selected_ladder_chars'] for i in range(minkey+1, minkey+maxres)]

    spinselect_out = spinselector.fit_multiple_models(models,
                                    [1.0,2.0],
                                    Ta_pair,
                                    datasets,
                                    experiments,
                                    covariance_data,
                                    sammy_rto_fit,
                                    external_resonance_indices)
    
    autofit_out = AutoFitOUT(initial_out, history, spinselect_out)

    file = open(f'./Out_{title}.pkl', 'wb')
    pickle.dump(autofit_out, file)
    file.close()

#%%  Main

sammy_rto_fit = sammy_classes.SammyRunTimeOptions('/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
                                        {"Print":   True,
                                         "bayes":   True,
                                         "keep_runDIR": False,
                                         "sammy_runDIR": "sammy_runDIR_py"
                                         })


initialFBopt = InitialFBOPT(fit_all_spin_groups=False,
                            spin_group_keys = ['3.0'])

spinselectopt = spin_group_selection.SpinSelectOPT()


title = "def_1sg"
maxres = 5
fit(title, maxres, sammy_rto_fit, initialFBopt,spinselectopt)

