"""Package-wide isolation of process-level backend state, and the ``--shard`` option."""

from __future__ import annotations

import zlib

import pytest

from andrey.core import backend, stats, warning_policy


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="K/N",
        help="run only part K of N (1 <= K <= N) of the collected tests: the parts are disjoint, "
        "together hold every test, and are of similar size",
    )


def parse_shard(spec: str) -> tuple[int, int]:
    """The part ``K`` and the number of parts ``N`` of a ``K/N`` option value, 1 <= K <= N."""
    try:
        index, count = (int(part) for part in spec.split("/"))
    except ValueError:
        index = count = 0
    if not 1 <= index <= count:
        raise pytest.UsageError(f"--shard takes K/N with 1 <= K <= N, not {spec!r}")
    return index, count


def shard_of(nodeid: str, count: int) -> int:
    """The part, 1 to ``count``, that a test belongs to: a stable hash of its id, so that every
    process and every run agrees."""
    return zlib.crc32(nodeid.encode()) % count + 1


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    spec = config.getoption("--shard")
    if spec is None:
        return
    index, count = parse_shard(spec)
    kept: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        (kept if shard_of(item.nodeid, count) == index else deselected).append(item)
    config.hook.pytest_deselected(items=deselected)
    items[:] = kept


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
