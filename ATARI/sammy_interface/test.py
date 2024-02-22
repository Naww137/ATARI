#%%
import pandas as pd
import numpy as np
import os

from ATARI.sammy_interface import sammy_functions, sammy_classes, template_creator
from ATARI.theory.resonance_statistics import chisquare_PDF, wigner_PDF

from ATARI.ModelData.particle import Particle, Neutron
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model

from ATARI.utils.atario import expand_sammy_ladder_2_atari


### Test 
Ta181 = Particle(Z=73, A=181, I=3.5, mass=180.94803, name='Ta181')
Ta_pair = Particle_Pair()
Ta_pair.add_spin_group(Jpi='3.0',
                       J_ID=1,
                       D=8.79,
                       gn2_avg=46.5,
                       gn2_dof=1,
                       gg2_avg=64.0,
                       gg2_dof=1000)
Ta_pair.sample_resonance_ladder()



### sammy
rto = sammy_classes.SammyRunTimeOptions('/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
                                        {"Print":   True,
                                         "bayes":   False,
                                         "keep_runDIR": True,
                                         "sammy_runDIR": "sammy_runDIR"
                                         })

exp_model_T = Experimental_Model()
template_creator.make_input_template('template_T.inp', Ta_pair, exp_model_T, rto)

exp_model_T.template = os.path.realpath('template_T.inp')


sammyINP = sammy_classes.SammyInputData(
    Ta_pair,
    Ta_pair.resonance_ladder,
    os.path.realpath('template_T.inp'),
    exp_model_T,
    energy_grid=exp_model_T.energy_grid
    # experimental_data=data,
    # experimental_covariance = datasample.covariance_data
)

sammyOUT2 = sammy_functions.run_sammy(sammyINP, rto)

print(sammyOUT2.par)
print(sammyOUT2.pw)

data_unc = np.sqrt(sammyOUT2.pw['theo_trans'])/10
data = np.random.default_rng().normal(sammyOUT2.pw['theo_trans'], data_unc)
data_df = pd.DataFrame({'E':sammyOUT2.pw['E'],
                        'exp': data,
                        'exp_unc': data_unc})

rto = sammy_classes.SammyRunTimeOptions('/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
                                        {"Print":   True,
                                         "bayes":   True,
                                         "keep_runDIR": True,
                                         "sammy_runDIR": "sammy_runDIR"
                                         })
sammyINP.experimental_data = data_df

# # try catching error of bayes and no varied inputs
try:
    sammyOUT_fit = sammy_functions.run_sammy(sammyINP, rto)
    raise ValueError()
except:
    pass

sammyINP.resonance_ladder["varyE"] = np.ones(len(sammyINP.resonance_ladder))
sammyINP.resonance_ladder["varyGg"] = np.ones(len(sammyINP.resonance_ladder))
sammyINP.resonance_ladder["varyGn1"] = np.ones(len(sammyINP.resonance_ladder))

sammyOUT_fit = sammy_functions.run_sammy(sammyINP, rto)

print(sammyOUT_fit.pw_post)
print(sammyOUT_fit.par_post)

atari_par_post = expand_sammy_ladder_2_atari(Ta_pair, sammyOUT_fit.par_post)

assert(np.all([each in atari_par_post.keys() for each in ["gg2", "gn2", "Jpi", "L"]]))

print("Passed tests")
# resonance_ladder = pd.DataFrame({'E':[50, 55], 'Gg':[50,50], 'Gnx':[-50,50], 'chs':[1.0,1.0], 'lwave':[0.0,0.0], 'J':[3.0,3.0], 'J_ID':[1,1]})
# pw_exp = pd.DataFrame({'E':[10,100], 'exp_trans': [0.8,0.8], 'exp_trans_unc':[0.1,0.1]})


# sammyRTO = sammy_classes.SammyRunTimeOptions(
#     path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
#     model = 'SLBW',
#     reaction = 'transmission',
#     solve_bayes = False,
#     inptemplate="noexp_1sg.inp",
#     energy_window = None,
#     sammy_runDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SAMMY_runDIR'),
#     keep_runDIR = True,
#     shell = 'zsh'
#     )

# sammyINP = sammy_classes.SammyInputData(
#     particle_pair = Ta_pair,
#     resonance_ladder = resonance_ladder,
#     # energy_grid=np.array(pw_exp.E),
#     experimental_data= pw_exp,
#     # experimental_cov=
#     initial_parameter_uncertainty = 1.0
# )


# # lst, par = sammy_functions.run_sammy(sammyINP, sammyRTO)

# # print(lst)
# # print(par)

# # assert(np.all(lst.E == pw_exp.E))
# # assert(np.all(lst.exp_trans == pw_exp.exp_trans))
# # assert(np.all(lst.exp_trans_unc == pw_exp.exp_trans_unc))

# # assert(np.all(par.E == resonance_ladder.E))
# # assert(np.all(par.Gg == resonance_ladder.Gg))
# # assert(np.all(par.Gn1 == resonance_ladder.Gnx))


# par = sammy_functions.readpar("/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/SAMMY_runDIR/sammy.par")

# print(par)
# %%
