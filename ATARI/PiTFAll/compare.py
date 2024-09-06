import numpy as np
import pandas as pd
import os
from datetime import datetime

from ATARI.sammy_interface import sammy_classes, sammy_functions
from ATARI.utils.misc import fine_egrid, generate_sammy_rundir_uniq_name
from ATARI.ModelData.experimental_model import Experimental_Model


def get_pw_list_broad_only(sammy_exe,
                                particle_pair, 
                                resonance_ladder, 
                                energy_range,
                                temperature,
                                template, 
                                reactions):

    runDIR = generate_sammy_rundir_uniq_name('./')
    sammyRTO = sammy_classes.SammyRunTimeOptions(sammy_exe,
                             **{"Print"   :   True,
                              "bayes"   :   False,
                              "keep_runDIR"     : False,
                              "sammy_runDIR": runDIR
                              })
    E = fine_egrid(energy_range)
    exp_theo = Experimental_Model(title = "theo",
                                      reaction='total',
                                      temp = (temperature,0),
                                      energy_grid = E,
                                      energy_range = energy_range,
                                      template=template
                                      )
    sammyINP = sammy_classes.SammyInputData(
        particle_pair,
        resonance_ladder,
        template=template,
        experiment=exp_theo,
        energy_grid=E
    )
    
    pw_list = []
    for rxn in reactions:
        sammyINP.experiment.reaction = rxn
        sammyOUT = sammy_functions.run_sammy(sammyINP, sammyRTO)
        # if rxn == "capture" and resonance_ladder.empty: ### if no resonance parameters - sammy does not return a column for capture theo xs
        #     sammyOUT.pw["theo_xs"] = np.zeros(len(sammyOUT.pw))
        # df[rxn+df_column_key_extension] = sammyOUT.pw.theo_xs
        pw_list.append(sammyOUT.pw)
    
    return pw_list




class Compare:

    def __init__(self,
                 samout_est,
                 samout_true,

                 datasets,
                 experiments
                 ):

        # store experimental space models
        self.pw_exp_est = samout_est.pw_post
        self.pw_exp_true = samout_true.pw
        self.pw_exp_fittrue = samout_true.pw_post

        ## store parameter space models
        self.par_est = samout_est.par_post
        self.par_true = samout_true.par
        self.par_fittrue = samout_true.par_post

        ## store experimental data and models
        self.datasets = datasets
        self.experiments = experiments
        
        # convienent definitions
        assert(len(samout_est.pw) == len(samout_true.pw))
        for i in range(len(samout_est.pw)):
            assert(np.all(samout_est.pw[i].E.values == samout_true.pw[i].E.values))
        self.energy_range = (min(samout_est.pw[0].E.values), max(samout_est.pw[0].E.values))
        
        # could also test if all datasets == pw.exp after filtering to energy range of fits

    def get_theo_space_models(self,
                              sammy_exe, 
                              particle_pair, 
                              temperature,
                              template,
                              reactions):
        
        ## store theoretical space models
        self.pw_theo_est = get_pw_list_broad_only(sammy_exe, particle_pair, self.par_est, self.energy_range, temperature, template, reactions)
        self.pw_theo_true = get_pw_list_broad_only(sammy_exe, particle_pair, self.par_true, self.energy_range, temperature, template, reactions)
        self.pw_theo_fittrue = get_pw_list_broad_only(sammy_exe, particle_pair, self.par_fittrue, self.energy_range, temperature, template, reactions)

        self.reactions = reactions

    def get_residuals(self):
        ## reaction residuals in theoretical space for estimate
        residuals = []
        for i, rxn in enumerate(self.reactions):
            residuals.append(self.pw_theo_est[i].theo_xs.values - self.pw_theo_true[i].theo_xs.values)
        self.residuals_theo_est = np.concatenate(residuals)

        ## reaction residuals in theoretical space for fit from true
        residuals = []
        for i, rxn in enumerate(self.reactions):
            residuals.append(self.pw_theo_fittrue[i].theo_xs.values - self.pw_theo_true[i].theo_xs.values)
        self.residuals_theo_fittrue = np.concatenate(residuals)