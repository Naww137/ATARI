import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest, chi2, norm, ks_1samp, norm
from ATARI.syndat.control import syndatOPT, Syndat_Model
from ATARI.ModelData.experimental_model import Experimental_Model


def noise_distribution_test(syn, print_out=False, ipert=5000, energy_range = [10,3000]):
    """
    Tests the following for data sampling distributions from given measurement model:
        1. the mean of all data samples converges to the true value
        2. the normalized residuals (data-true)/(data_uncertainty) fall on a standard normal distribution
    """

    energy_grid = np.sort(np.random.default_rng().uniform(min(energy_range),max(energy_range),10)) #np.linspace(min(energy_range),max(energy_range),10) # energy below 10 has very low counts due to approximate open spectrum
    df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.1,1,10)})
    df_true.sort_values('E', ascending=True, inplace=True)

    # reset syndat model samples, energy grid, and open neutron spectrum
    syn.clear_samples()
    syn.generative_experimental_model.energy_grid = energy_grid
    syn.generative_experimental_model.energy_range = energy_range
    syn.generative_measurement_model.model_parameters.open_neutron_spectrum = None
    syn.reductive_measurement_model.model_parameters.open_neutron_spectrum = None
    
    # set options for test
    syn.options.sampleRES = False
    syn.options.explicit_covariance = True
    syn.options.calculate_covariance = True

    syn.sample(pw_true=df_true, num_samples=ipert)
    if 'CovT' in syn.samples[0].covariance_data.keys():
        covkey = 'CovT'
    elif 'CovY' in syn.samples[0].covariance_data.keys():
        covkey='CovY'
    else:
        raise ValueError

    chi2_gofs = []
    normres = []
    for i in range(ipert):
        data = syn.samples[i].pw_reduced
        cov = syn.samples[i].covariance_data[covkey].values
        # exp_trans[i,:] = np.array(data.exp)
        res = np.array(data.exp)-df_true.true.values
        normres.extend(res/data.exp_unc.values)
        chi2_gofs.append(res @ np.linalg.inv(cov) @ res.T)

    chi2_dist = chi2(df=len(energy_grid))

    mean_of_residual = np.mean(res, axis=0)
    norm_test_on_residual = normaltest(normres)
    kstest_on_chi2 = ks_1samp(chi2_gofs, chi2_dist.cdf)

    if print_out:
        x = np.linspace(-5,5)
        plt.figure()
        plt.hist(normres, bins=50, density=True)
        plt.plot(x, norm.pdf(x))
        plt.show()

        x = np.linspace(0,50)
        plt.figure()
        plt.hist(chi2_gofs, bins=50, density=True)
        plt.plot(x, chi2_dist.pdf(x))
        plt.show()

        print(f"Mean of residual: {mean_of_residual}")
        print(f"Standard normal test: {norm_test_on_residual}")
        print(f"Chi2 ks test: {kstest_on_chi2}")

    return mean_of_residual, norm_test_on_residual, kstest_on_chi2




def noise_distribution_test2(syn, df_true, print_out=False, ipert=5000):
    """
    Tests the following for data sampling distributions from given measurement model:
        1. the mean of all data samples converges to the true value
        2. the normalized residuals (data-true)/(data_uncertainty) fall on a standard normal distribution
    """
    
    # set options for test
    syn.options.sampleRES = False
    syn.options.explicit_covariance = True
    syn.options.calculate_covariance = True

    syn.sample(pw_true_list=df_true, num_samples=ipert)
    if 'CovT' in syn.samples[0].covariance_data.keys():
        covkey = 'CovT'
    elif 'CovY' in syn.samples[0].covariance_data.keys():
        covkey='CovY'
    else:
        raise ValueError

    chi2_gofs = []
    normres = []
    for i in range(ipert):
        data = syn.samples[i].pw_reduced
        cov = syn.samples[i].covariance_data[covkey].values
        # exp_trans[i,:] = np.array(data.exp)
        res = np.array(data.exp)-df_true.true.values
        normres.extend(res/data.exp_unc.values)
        chi2_gofs.append(res @ np.linalg.inv(cov) @ res.T)

    chi2_dist = chi2(df=len(data))

    mean_of_residual = np.mean(res, axis=0)
    norm_test_on_residual = normaltest(normres)
    kstest_on_chi2 = ks_1samp(chi2_gofs, chi2_dist.cdf)

    if print_out:
        x = np.linspace(-5,5)
        plt.figure()
        plt.hist(normres, bins=50, density=True)
        plt.plot(x, norm.pdf(x))
        plt.show()

        x = np.linspace(0,max(chi2_gofs)+10)
        plt.figure()
        plt.hist(chi2_gofs, bins=50, density=True)
        plt.plot(x, chi2_dist.pdf(x))
        plt.show()

        print(f"Mean of residual: {mean_of_residual}")
        print(f"Standard normal test: {norm_test_on_residual}")
        print(f"Chi2 ks test: {kstest_on_chi2}")

    return mean_of_residual, norm_test_on_residual, kstest_on_chi2



def no_sampling_returns_true_test(generative_model, reductive_model):
    """
    Tests that when nothing is sampled, the measurement model returns the true value.
    """

    synOPT = syndatOPT(sampleRES=False, 
                    sample_counting_noise= False, 
                    calculate_covariance=False, 
                    sampleTMP=False, 
                    smoothTNCS =True)
    
    energy_grid = np.sort(np.random.default_rng().uniform(10,5000,10)) #np.linspace(min(energy_range),max(energy_range),10) # energy below 10 has very low counts due to approximate open spectrum
    df_true = pd.DataFrame({'E':energy_grid, 'true':np.random.default_rng().uniform(0.01,1,10)})
    exp_model = Experimental_Model(energy_grid=energy_grid, energy_range=[10,5000])
    synT = Syndat_Model(generative_experimental_model=exp_model, generative_measurement_model=generative_model, reductive_measurement_model=reductive_model, options=synOPT)

    exp = np.zeros([10,len(energy_grid)])
    exp_unc = np.zeros([10,len(energy_grid)])
    synT.sample(pw_true=df_true, num_samples=10)

    for i in range(10):
        data = synT.samples[i].pw_reduced
        exp[i,:] = np.array(data.exp)
        exp_unc[i,:] = np.array(data.exp_unc)

    return np.sum(abs(exp-df_true.true.values))