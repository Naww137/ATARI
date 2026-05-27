import sys
import unittest

__doc__ = """
This file runs all of the unit tests from the "tests" directory.
"""


sammy_run_path = sys.argv[1]
if sammy_run_path == None:
    sammy_run_path = '/Users/noahwalton/gitlab/sammy/sammy/build/bin/sammy'

from tests import test_resonance_distributions, test_level_spacing_distributions, test_res_generator, test_syndat_functionality, test_measurement_covariance, test_atario, test_mean_parameter_estimation, test_utils_stats #, test_fit_and_eliminate
from tests import test_fit_and_eliminate, test_sammy_interface
from tests import test_samplers, test_pt_bayes, test_wig_bayes, test_wig_sample, test_wig_max_likelihoods

if __name__ == '__main__':
    
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()

    ### general ATARI test suites
    parameter_distribution_test_suite = loader.loadTestsFromModule(test_resonance_distributions)
    level_spacing_distribution_test_suite = loader.loadTestsFromModule(test_level_spacing_distributions)
    resonance_generator_test_suite = loader.loadTestsFromModule(test_res_generator)
    test_mean_parameter_estimation_test_suite = loader.loadTestsFromModule(test_mean_parameter_estimation)
    
    # TODO: add theory_module_test_suite
    syndat_test_suite = loader.loadTestsFromModule(test_syndat_functionality)
    measurement_test_suite = loader.loadTestsFromModule(test_measurement_covariance)

    atario_test_suite = loader.loadTestsFromModule(test_atario)
    stats_test_suite = loader.loadTestsFromModule(test_utils_stats)

    ### TAZ Unit Tests
    samplers_TAZ_test_suite = loader.loadTestsFromModule(test_samplers)
    pt_bayes_test_suite = loader.loadTestsFromModule(test_pt_bayes)
    wig_bayes_test_suite = loader.loadTestsFromModule(test_wig_bayes)
    wig_sample_test_suite = loader.loadTestsFromModule(test_wig_sample)
    wig_max_likelihood_test_suite = loader.loadTestsFromModule(test_wig_max_likelihoods)

    ### those that require sammy
    sammy_test_suite = loader.loadTestsFromModule(test_sammy_interface)
    # fit_and_eliminate_test_suite = loader.loadTestsFromModule(test_fit_and_eliminate)
    # TODO: add fitting_from_theory_test_suite
    # TODO: add autofit_test_suite

    fae_test_suite = loader.loadTestsFromModule(test_fit_and_eliminate)

   
    # print("Running parameter distribution test suite")
    # result = runner.run(parameter_distribution_test_suite)

    # print("Running level-spacing distribution test suite")
    # result = runner.run(level_spacing_distribution_test_suite)

    # print("Running resonance generator test suite")
    # result = runner.run(resonance_generator_test_suite)

    # print("Running mean parameter estimation test suite")
    # result = runner.run(test_mean_parameter_estimation_test_suite)

    # print("Running Syndat test suite")
    # result = runner.run(syndat_test_suite)

    # print("Running measurement covariance test suite")
    # result = runner.run(measurement_test_suite)

    # print("Running atario test suite")
    # result = runner.run(atario_test_suite)

    # print("Running stats test suite")
    # result = runner.run(stats_test_suite)

    # print("Running TAZ test suites")
    # result = runner.run(samplers_TAZ_test_suite)
    # result = runner.run(pt_bayes_test_suite)
    # result = runner.run(wig_bayes_test_suite)
    # result = runner.run(wig_sample_test_suite)
    # result = runner.run(wig_max_likelihood_test_suite)

    print("Now running tests that require SAMMY - if you have not already, go into test files and change your sammy path.")
    print("Running sammy_interface test suite")
    result = runner.run(sammy_test_suite)

    print('Running fit and eliminate test suite')
    result = runner.run(fae_test_suite)

    # print("Running fit_and_eliminate test suite")
    # result = runner.run(fit_and_eliminate_test_suite)



# # The following code will run all tests that fit the pattern. This may be convenient when we get
# # all of the tests in order.
# import unittest
# def run_tests():
#     test_loader = unittest.TestLoader()
#     test_suite = test_loader.discover('tests', pattern='tests_*.py')
#     runner = unittest.TextTestRunner()
#     runner.run(test_suite)

# if __name__ == "__main__":
#     run_tests()
