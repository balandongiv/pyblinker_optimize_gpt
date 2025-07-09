"""Execute every unit test in the repository.

This helper script searches both the ``unit_test/blinker_migration`` and
``unit_test/features`` directories for modules matching ``test_*.py`` and runs
them using ``unittest``.  It mirrors what continuous integration would execute
and is useful for local debugging.
"""

from pathlib import Path
import unittest
import multiprocessing
import sys

# Ensure the repository root is on the Python path so that imports
# like ``pyblinkers`` succeed when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Discover and load all tests in the 'tests' directory
test_loader = unittest.TestLoader()
test_suite = test_loader.discover('unit_test', pattern='test_*.py')
def main() -> None:
    """Execute the discovered test suite."""
    multiprocessing.set_start_method("spawn", force=True)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_runner.run(test_suite)


if __name__ == "__main__":
    main()
