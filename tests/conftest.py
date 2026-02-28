"""
Root conftest for the test suite.

Markers
-------
slow
    Tests that require external data files or long training loops.
    Excluded from the default CI run; opt-in with:  pytest -m slow

Collection guards
-----------------
Any test file that imports a module unavailable in the current environment
(e.g. ``data.pendulum.load_pendulum``) is silently skipped at collection
time so that the rest of the suite can still run.
"""

import importlib

import pytest


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as requiring external data or heavy compute (skipped in CI)",
    )


# ---------------------------------------------------------------------------
# Skip collection of test files whose data dependencies are missing
# ---------------------------------------------------------------------------

_DATA_GUARDED_MODULES = [
    # module string that must be importable for the tests in that file
    ("tests.train.test_train", "data.pendulum.load_pendulum"),
]


def pytest_ignore_collect(collection_path, config):
    """Skip a test file when its external data module cannot be imported."""
    path_str = str(collection_path)
    for _test_module, data_module in _DATA_GUARDED_MODULES:
        if _test_module.replace(".", "/") in path_str.replace("\\", "/"):
            try:
                importlib.import_module(data_module)
            except ImportError:
                return True  # silently skip
    return None
