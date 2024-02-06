import unittest
from tests import test_sammy_interface, test_syndat, test_distributions, test_res_generator




if __name__ == '__main__':
    
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()

    # create test suites
    parameter_distribution_test_suite = loader.loadTestsFromModule(test_distributions)
    resonance_generator_test_suite = loader.loadTestsFromModule(test_res_generator)

    sammy_test_suite = loader.loadTestsFromModule(test_sammy_interface)
    syndat_test_suite = loader.loadTestsFromModule(test_syndat)
   

    print("Running parameter distribution test suite")
    result = runner.run(parameter_distribution_test_suite)

    print("Running resonance generator test suite")
    result = runner.run(resonance_generator_test_suite)

    print("Running SAMMY test suite")
    result = runner.run(sammy_test_suite)

    print("Running Syndat test suite")
    result = runner.run(syndat_test_suite)


    
