"""Package-wide isolation of process-level backend state."""

from __future__ import annotations

import pytest

from andrey.core import backend, stats, warning_policy


@pytest.fixture(autouse=True)
def _reset_backend_state():
    """Isolate process-level backend config, warn-once memory, and calibration cache between tests.

    The warn-once set, the ``config`` object, and the calibration cache are process-global, so a
    test that pins a backend, trips an unavailable-backend warning, or seeds a cutoff would
    otherwise leak into later tests in the session. Warn-once memory is also cleared on entry, so
    a warning raised outside a test, in a module fixture, does not hide a test's first warning.
    """
    warning_policy._warned.clear()
    yield
    backend.config._backend = None
    backend.config._n_jobs = None
    backend.config._tie_break = None
    warning_policy._warned.clear()
    backend._dispatch_counts.clear()
    stats._CALIBRATED.clear()
