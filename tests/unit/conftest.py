"""Shared unit-test fixtures."""

from __future__ import annotations

import pytest

from andrey.core import backend, stats


@pytest.fixture(autouse=True)
def _reset_backend_state():
    """Isolate process-level backend config, warn-once memory, and calibration cache between tests.

    The warn-once set, the ``config`` object, and the calibration cache are process-global, so a
    test that pins a backend, trips an unavailable-backend warning, or seeds a cutoff would
    otherwise leak into later tests in the session. Warn-once memory is also cleared on entry, so
    warnings raised by tests outside this directory do not hide a first warning here.
    """
    backend._reset_warnings()
    yield
    backend.config._backend = None
    backend.config._num_workers = None
    backend._reset_warnings()
    backend.reset_dispatch_stats()
    stats._CALIBRATED.clear()


def serve_datasets_offline(monkeypatch, cache):
    """Serve dataset downloads from fixtures/, into the empty directory ``cache``, with no network.

    Sachs is the 853-row file, under its own SHA-256; ASIA is the published BIF, under the
    recorded one. Returns the list of URLs fetched, so a test can tell a download from a cache hit.
    """
    import dataclasses
    import gzip
    import hashlib
    import urllib.error
    from pathlib import Path

    from andrey.data import real

    fixtures = Path(__file__).parent / "fixtures"
    sachs = gzip.decompress((fixtures / "sachs-observational.txt.gz").read_bytes())
    served = {"https://example.invalid/sachs.txt": sachs}
    served |= dict.fromkeys(real.DATASETS["asia"].sources, (fixtures / "asia.bif.gz").read_bytes())
    fetched: list[str] = []

    def fetch(url):
        fetched.append(url)
        if url not in served:
            raise urllib.error.URLError(f"no fixture for {url}")
        return served[url]

    remote = dataclasses.replace(
        real.DATASETS["sachs"],
        sources=("https://example.invalid/sachs.txt",),
        sha256=hashlib.sha256(sachs).hexdigest(),
    )
    monkeypatch.setitem(real.DATASETS, "sachs", remote)
    monkeypatch.setattr(real, "fetch", fetch)
    monkeypatch.setenv("ANDREY_DATA_DIR", str(cache))
    return fetched


@pytest.fixture
def offline_datasets(monkeypatch, tmp_path):
    """Dataset downloads served from fixtures/ for one test; see ``serve_datasets_offline``."""
    return serve_datasets_offline(monkeypatch, tmp_path / "cache")


@pytest.fixture(scope="module")
def offline_datasets_module(tmp_path_factory):
    """Dataset downloads served from fixtures/ for a whole module, its module fixtures included."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        yield serve_datasets_offline(monkeypatch, tmp_path_factory.mktemp("cache"))
