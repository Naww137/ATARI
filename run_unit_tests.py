import unittest
from tests import test_resonance_distributions, test_sammy_interface, test_res_generator, test_syndat_functionality, test_measurement_covariance, test_atario




if __name__ == '__main__':
    
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()

    ### general ATARI test suites
    parameter_distribution_test_suite = loader.loadTestsFromModule(test_resonance_distributions)
    resonance_generator_test_suite = loader.loadTestsFromModule(test_res_generator)
    
    # TODO: add theory_module_test_suite
    syndat_test_suite = loader.loadTestsFromModule(test_syndat_functionality)
    measurement_test_suite = loader.loadTestsFromModule(test_measurement_covariance)

    atario_test_suite = loader.loadTestsFromModule(test_atario)

    ### those that require sammy
    sammy_test_suite = loader.loadTestsFromModule(test_sammy_interface)
    # TODO: add fitting_from_theory_test_suite
    # TODO: add autofit_test_suite

   
    print("Running parameter distribution test suite")
    result = runner.run(parameter_distribution_test_suite)

    print("Running resonance generator test suite")
    result = runner.run(resonance_generator_test_suite)

    print("Running Syndat test suite")
    result = runner.run(syndat_test_suite)

    print("Running measurement covariance test suite")
    result = runner.run(measurement_test_suite)

    print("Running atario test suite")
    result = runner.run(atario_test_suite)

    print("Now running tests that require SAMMY")
    print("Running sammy_interface test suite")
    result = runner.run(sammy_test_suite)


    
