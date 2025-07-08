'''

This script will run all the test in the 'unit_test' directory.
Everything that is in the 'unit_test' directory with the name 'test_*.py' will be run.
s
'''

import unittest
from pathlib import Path
import sys

# Ensure the repository root is on the Python path so that imports
# like ``pyblinkers`` succeed when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Discover and load all tests in the 'tests' directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('unit_test', pattern='test_*.py')

# Run the tests
test_runner = unittest.TextTestRunner(verbosity=2)
test_runner.run(test_suite)