"""
Root conftest for the test suite.

Markers
-------
slow
    Tests that require external data files or long training loops.
    Excluded from the default CI run; opt-in with:  pytest -m slow

needs_mosek
    Tests that verify tight LMI tolerances or geotorch constraint
    feasibility after _right_inverse initialisation.  They pass reliably
    with MOSEK; they are skipped automatically when only CLARABEL/SCS is
    available (solver precision is then at best ~1e-6 instead of ~1e-9).

Collection guards
-----------------
Any test file that imports a module unavailable in the current environment
(e.g. ``data.pendulum.load_pendulum``) is silently skipped at collection
time so that the rest of the suite can still run.
"""

import importlib
import pytest


# ---------------------------------------------------------------------------
# Solver detection  (session-level, computed once)
# ---------------------------------------------------------------------------

def _best_available_solver() -> str:
    """Return the most accurate CVXPY solver that is installed."""
    for solver in ("MOSEK", "CLARABEL", "SCS"):
        try:
            mod = "mosek" if solver == "MOSEK" else solver.lower()
            importlib.import_module(mod)
            return solver
        except ImportError:
            continue
    return "SCS"  # SCS ships with cvxpy, always present


_SOLVER = _best_available_solver()
_HIGH_PRECISION = _SOLVER in ("MOSEK", "CLARABEL")


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as requiring external data or heavy compute (skipped in CI)",
    )
    config.addinivalue_line(
        "markers",
        (
            "needs_mosek: mark test as requiring high-precision LMI solving "
            "(MOSEK or CLARABEL); skipped automatically when only SCS is available"
        ),
    )


# ---------------------------------------------------------------------------
# Auto-skip needs_mosek tests when solver precision is insufficient
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    skip_low_precision = pytest.mark.skip(
        reason=f"Requires MOSEK or CLARABEL for geotorch init precision "
               f"(available solver: {_SOLVER})"
    )
    for item in items:
        if item.get_closest_marker("needs_mosek") and not _HIGH_PRECISION:
            item.add_marker(skip_low_precision)


# ---------------------------------------------------------------------------
# Skip collection of test files whose data dependencies are missing
# ---------------------------------------------------------------------------

_DATA_GUARDED_FILES = [
    # (substring in path,  required importable module)
    ("tests/train/test_train",   "data.pendulum.load_pendulum"),
    ("tests/models/linear/test_h2linear", "data.pendulum.load_pendulum"),
]


def pytest_ignore_collect(collection_path, config):
    """Skip a test file when its external data module cannot be imported."""
    path_str = str(collection_path).replace("\\", "/")
    for path_fragment, data_module in _DATA_GUARDED_FILES:
        if path_fragment in path_str:
            try:
                importlib.import_module(data_module)
            except ImportError:
                return True   # silently skip the whole file
    return None


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def cvxpy_solver() -> str:
    """Best available CVXPY solver for this session."""
    return _SOLVER


@pytest.fixture(scope="session")
def lmi_atol() -> float:
    """
    Absolute tolerance appropriate for LMI residual checks, scaled to the
    precision of the available solver.

    MOSEK    → 1e-7
    CLARABEL → 1e-5
    SCS      → 1e-2  (rarely reached; most precision tests are skipped)
    """
    return {"MOSEK": 1e-7, "CLARABEL": 1e-5, "SCS": 1e-2}[_SOLVER]
