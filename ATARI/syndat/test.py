import numpy as np
import pandas as pd
from scipy.stats import normaltest
from ATARI.ModelData.experimental_model import Experimental_Model
from ATARI.ModelData.particle_pair import Particle_Pair
from ATARI.ModelData.measurement_models.transmission_rpi import Transmission_RPI
from ATARI.ModelData.measurement_models.capture_yield_rpi import Capture_Yield_RPI

from ATARI.syndat.control import syndatOPT, Syndat_Model


def test_mean_converges_to_true_transmissionRPI():

    pair = Particle_Pair()
    exp_model = Experimental_Model(energy_grid = np.array([10, 1000, 3000]),
                                   energy_range =[9, 3001])
    generative = Transmission_RPI()
    reductive = Transmission_RPI()

    synOPT = syndatOPT(sampleRES=False)
    synT = Syndat_Model(exp_model, generative, reductive, synOPT)

    ipert = 5000
    exp_trans = np.zeros([ipert,3])
    exp_trans_unc = np.zeros([ipert,3])
    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([0.5,0.99,0.01])})
    synT.sample(pw_true=df_true, num_samples=ipert)

    for i in range(ipert):
        data = synT.samples[i].pw_reduced
        exp_trans[i,:] = np.array(data.exp)
        exp_trans_unc[i,:] = np.array(data.exp_unc)

    true_trans = np.array(df_true.sort_values('E', ascending=True)["true"])
    assert (np.all(np.isclose(np.mean(exp_trans, axis=0), true_trans, rtol=1e-2)))
    assert (np.all(normaltest((exp_trans-true_trans)/exp_trans_unc).pvalue>1e-5))

    print("Passed test mean_converges_to_true for syndat.transmission_rpi.Transmission_RPI")




def test_mean_converges_to_true_captureRPI():

    pair = Particle_Pair()
    exp_model = Experimental_Model(energy_grid = np.array([10, 1000, 3000]),
                                   energy_range =[9, 3001])
    generative = Capture_Yield_RPI()
    reductive = Capture_Yield_RPI()

    synOPT = syndatOPT(sampleRES=False, calculate_covariance=False)
    synT = Syndat_Model(exp_model, generative, reductive, synOPT)

    ipert = 5000
    exp_trans = np.zeros([ipert,3])
    exp_trans_unc = np.zeros([ipert,3])
    df_true = pd.DataFrame({'E':[10, 1000, 3000], 'true':np.array([0.5,0.8,0.1])})
    synT.sample(pw_true=df_true, num_samples=ipert)

    for i in range(ipert):
        data = synT.samples[i].pw_reduced
        exp_trans[i,:] = np.array(data.exp)
        exp_trans_unc[i,:] = np.array(data.exp_unc)

    true_trans = np.array(df_true.sort_values('E', ascending=True)["true"])
    assert (np.all(np.isclose(np.mean(exp_trans, axis=0), true_trans, rtol=1e-2)))
    assert (np.all(normaltest((exp_trans-true_trans)/exp_trans_unc).pvalue>1e-5))

    print("Passed test mean_converges_to_true for syndat.capture_yield_rpi.Capture_Yield_RPI")


def no_sampling_returns_same_values():
    return



# test_mean_converges_to_true_transmissionRPI()

test_mean_converges_to_true_captureRPI()

# no_sampling_returns_same_values()