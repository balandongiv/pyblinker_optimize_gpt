"""Run the feature extraction test suite.

This convenience script discovers every ``test_*.py`` module under the
``unit_test/features`` directory and executes them using Python's standard
``unittest`` runner.  It is primarily intended for manual debugging during
development.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # Ensure the project root is on the path so imports work when running
    # this script directly.
    sys.path.insert(0, str(base_dir.parent))

    tests_dir = base_dir / "features"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
