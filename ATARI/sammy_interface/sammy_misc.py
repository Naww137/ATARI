import pandas as pd
import numpy as np
from ATARI.theory.experimental import e_to_t, t_to_e
from ATARI.syndat.data_classes import syndatOPT
from copy import deepcopy

from typing import Optional
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.sammy_interface import sammy_classes
from ATARI.sammy_interface.sammy_functions import run_sammy
from ATARI.utils.stats import add_normalization_uncertainty_to_covariance


def generate_true_experiment(particle_pair: Optional[Particle_Pair],
                            sammyRTO: Optional[sammy_classes.SammyRunTimeOptions],
                            generate_pw_true_with_sammy: bool,
                            pw_true: Optional[pd.DataFrame] = None,
                            generative_experimental_model: Optional[Experimental_Model] = None
                            ):
        """
        Generates true experimental object using sammy as defined by self.generative_experimental_model.
        Experimental object = theory + experimental corrections (Doppler, resolution, MS, tranmission/yield).

        Parameters
        ----------
        particle_pair : Optional[Particle_Pair]
            _description_
        sammyRTO : Optional[sammy_classes.SammyRunTimeOptions]
            _description_
        generate_pw_true_with_sammy : bool
            _description_
        pw_true : Optional[pd.DataFrame], optional
            _description_, by default None
        """
        
        if generate_pw_true_with_sammy:
            assert sammyRTO is not None
            assert particle_pair is not None
            assert generative_experimental_model is not None

            sammyRTO.bayes = False
            sammyRTO.derivatives = False
            # rto.theoretical = True
            # template = self.generative_experimental_model.template

            if generative_experimental_model.template is None: raise ValueError("Experimental model sammy template has not been assigned")

            sammyINP = sammy_classes.SammyInputData(
                particle_pair,
                particle_pair.resonance_ladder,
                # template= template,
                experiment= generative_experimental_model,
                energy_grid= generative_experimental_model.energy_grid
            )
            sammyOUT = run_sammy(sammyINP, sammyRTO)

            if generative_experimental_model.reaction == "transmission":
                true = "theo_trans"
            else:
                true = "theo_xs"
            assert isinstance(sammyOUT.pw, pd.DataFrame)
            pw_true = sammyOUT.pw.loc[:, ["E", true]]
            pw_true.rename(columns={true: "true"}, inplace=True)

        else:
            assert pw_true is not None
            if np.any(pw_true.true.values == np.nan):
                raise ValueError("'true' in supplied pw true contains a Nan, please make sure you are supplying the correct pw_true")

            pw_true = pw_true
        
        if "tof" not in pw_true:
            pw_true["tof"] = e_to_t(pw_true.E.values, generative_experimental_model.FP[0], True)*1e9+generative_experimental_model.t0[0]
        
        return pw_true



def get_idc_at_theory(sammyINP, sammyRTO, resonance_ladder):
    assert sammyINP.idc_at_theory
    rto_theo = deepcopy(sammyRTO)
    rto_theo.sammy_runDIR = f"{sammyRTO.sammy_runDIR}_theo"
    rto_theo.bayes = False
    rto_theo.keep_runDIR = False
    sammyINP.particle_pair.resonance_ladder = resonance_ladder

    options = syndatOPT(sampleRES = False,sample_counting_noise = False,sampleTMP = False,sampleTNCS = False) 

    covariance_data_at_theory = []
    # try:
    for exp, meas in zip(sammyINP.experiments, sammyINP.measurement_models):
        if meas is None:
            pw_true = generate_true_experiment(particle_pair=sammyINP.particle_pair, sammyRTO=rto_theo, generate_pw_true_with_sammy=True, generative_experimental_model=exp)
            cov = {"theory":pw_true}
            # add_normalization_uncertainty_to_covariance(var)
        else:
            pw_true = generate_true_experiment(particle_pair=sammyINP.particle_pair, sammyRTO=rto_theo, generate_pw_true_with_sammy=True, generative_experimental_model=exp)
            raw_data = meas.generate_raw_data(pw_true, meas.model_parameters, options)
            _, cov, _ = meas.reduce_raw_data(raw_data,  options)
        covariance_data_at_theory.append(cov)
    # except: # weird sammy bug, sometimes need nopup experiments bc it will fail
    #     for exp, meas in zip(sammyINP.experiments_no_pup, sammyINP.measurement_models):
    #         if meas is None:
    #             cov = {}
    #         else:
    #             pw_true = generate_true_experiment(particle_pair=sammyINP.particle_pair, sammyRTO=rto_theo, generate_pw_true_with_sammy=True, generative_experimental_model=exp)
    #             raw_data = meas.generate_raw_data(pw_true, meas.model_parameters, options)
    #             _, cov, _ = meas.reduce_raw_data(raw_data,  options)
    #         covariance_data_at_theory.append(cov)

    return covariance_data_at_theory





