"""Run the migration-related test suite.

This script searches the ``unit_test/blinker_migration`` directory for modules
matching ``test_*.py`` and executes them using Python's ``unittest`` framework.
It mirrors CI behaviour and is convenient for local debugging.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(base_dir.parent))

    tests_dir = base_dir / "blinker_migration"
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(tests_dir), pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
