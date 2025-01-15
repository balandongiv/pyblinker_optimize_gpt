'''

This script will run all the test in the 'unit_test' directory.
Everything that is in the 'unit_test' directory with the name 'test_*.py' will be run.
'''

import unittest

# Discover and load all tests in the 'tests' directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('unit_test', pattern='test_*.py')

# Run the tests
test_runner = unittest.TextTestRunner(verbosity=2)
test_runner.run(test_suite)