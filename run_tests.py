import unittest
from tests import test_sammy_interface, test_syndat, test_distributions, test_res_generator

__doc__ = """
This file runs all of the unit tests from the "tests" directory.
"""




if __name__ == '__main__':
    
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()

    ### general ATARI test suites
    parameter_distribution_test_suite = loader.loadTestsFromModule(test_distributions)
    resonance_generator_test_suite = loader.loadTestsFromModule(test_res_generator)
    # TODO: add theory_module_test_suite
    syndat_test_suite = loader.loadTestsFromModule(test_syndat)

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

    print("Now running tests that require SAMMY")
    print("Running sammy_interface test suite")
    result = runner.run(sammy_test_suite)



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
