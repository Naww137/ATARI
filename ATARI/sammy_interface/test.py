#%%
import pandas as pd
import numpy as np
import os

from ATARI.sammy_interface import sammy_functions, sammy_classes
from ATARI.syndat.particle_pair import Particle_Pair


ac = 0.81271 
M = 180.948030 
m = 1          
I = 3.5        
i = 0.5        
l_max = 1      
Ta_pair = Particle_Pair( ac, M, m, I, i, l_max)


resonance_ladder = pd.DataFrame({'E':[50, 55], 'Gg':[50,50], 'Gnx':[-50,50], 'chs':[1.0,1.0], 'lwave':[0.0,0.0], 'J':[3.0,3.0], 'J_ID':[1,1]})
pw_exp = pd.DataFrame({'E':[10,100], 'exp_trans': [0.8,0.8], 'exp_trans_unc':[0.1,0.1]})


sammyRTO = sammy_classes.SammyRunTimeOptions(
    path_to_SAMMY_exe = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy',
    model = 'SLBW',
    reaction = 'transmission',
    solve_bayes = False,
    inptemplate="noexp_1sg.inp",
    energy_window = None,
    sammy_runDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SAMMY_runDIR'),
    keep_runDIR = True,
    shell = 'zsh'
    )

sammyINP = sammy_classes.SammyInputData(
    particle_pair = Ta_pair,
    resonance_ladder = resonance_ladder,
    # energy_grid=np.array(pw_exp.E),
    experimental_data= pw_exp,
    # experimental_cov=
    initial_parameter_uncertainty = 1.0
)


# lst, par = sammy_functions.run_sammy(sammyINP, sammyRTO)

# print(lst)
# print(par)

# assert(np.all(lst.E == pw_exp.E))
# assert(np.all(lst.exp_trans == pw_exp.exp_trans))
# assert(np.all(lst.exp_trans_unc == pw_exp.exp_trans_unc))

# assert(np.all(par.E == resonance_ladder.E))
# assert(np.all(par.Gg == resonance_ladder.Gg))
# assert(np.all(par.Gn1 == resonance_ladder.Gnx))


par = sammy_functions.readpar("/Users/noahwalton/Documents/GitHub/ATARI/Fitting/noah_dev/SAMMY_runDIR/sammy.par")

print(par)
# %%
