"""Refuting tests for materialize-once and for the provenance capture.

They run in ``bench-env``, the canonical generator / ``andrey.data`` host. Scope is correctness on
TINY data (d ≤ 10) only — no large generation, no timing (the harness is validated on a micro-grid
before any large benchmark fires).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from tiny_keys import TINY_KEYS

from andrey_bench import datasets, provenance

# --------------------------------------------------------------------------------------------------
# datasets — materialize once, reload identical.
# --------------------------------------------------------------------------------------------------


def test_materialize_writes_npz_and_registry_reloads_identically(tmp_path):
    registry = datasets.materialize(TINY_KEYS, tmp_path)

    # registry.json present and well-formed.
    registry_path = tmp_path / datasets.REGISTRY_NAME
    assert registry_path.exists()
    on_disk = json.loads(registry_path.read_text())
    assert on_disk == registry
    assert registry["generator"] == "andrey.data.sample_scm"
    assert registry["canonical_dtype"] == "float64"
    assert registry["n_datasets"] == 4

    for key in TINY_KEYS:
        entry = registry["keys"][key.label]

        # .npz file exists and carries BOTH the data and the true-graph endpoint marks.
        npz_path = tmp_path / entry["file"]
        assert npz_path.exists()
        with np.load(npz_path) as z:
            assert set(z.files) == {"data", "graph"}
            data_arr = z["data"]
            graph_arr = z["graph"]

        # Stored canonical dtype + shapes match the registry record.
        assert data_arr.dtype == np.float64
        assert data_arr.shape == (key.n, key.d)
        assert graph_arr.shape == (key.d, key.d)  # true-graph array present
        assert entry["data_shape"] == [key.n, key.d]
        assert entry["graph_shape"] == [key.d, key.d]

        # Reloading gives IDENTICAL bytes to a fresh deterministic generation from andrey.data.
        regen_data, regen_marks, _ = datasets.generate_one(key)
        assert np.array_equal(data_arr, regen_data)
        assert np.array_equal(graph_arr, regen_marks)

        # The load() helper returns the same arrays (f64), and an f32 read losslessly recovers f32.
        loaded, loaded_marks = datasets.load(tmp_path, key)
        assert np.array_equal(loaded, data_arr)
        assert np.array_equal(loaded_marks, graph_arr)
        as_f32, _ = datasets.load(tmp_path, key, dtype="float32")
        assert as_f32.dtype == np.float32
        assert np.array_equal(as_f32, data_arr.astype(np.float32))


def test_materialize_accumulates_across_calls(tmp_path):
    """A benchmark materializes one rung at a time into one directory; the second call must not erase
    the first. The failure it refutes is silent — every ``.npz`` stays on disk while the registry
    describes the last call alone, leaving the earlier datasets with no generator call behind them.
    """
    first, second = TINY_KEYS[0], TINY_KEYS[2]
    early = datasets.materialize([first], tmp_path)
    registry = datasets.materialize([second], tmp_path)

    assert registry["n_datasets"] == 2
    assert set(registry["keys"]) == {first.label, second.label}
    assert registry["keys"][first.label] == early["keys"][first.label]
    assert json.loads((tmp_path / datasets.REGISTRY_NAME).read_text()) == registry
    for key in (first, second):
        assert (tmp_path / key.filename).exists()


def test_materialize_survives_concurrent_calls_into_one_store(tmp_path):
    """Concurrent campaign jobs can materialize datasets in one shared store.

    A temporary file makes renaming safe after a crash. Concurrent writers also need unique
    filenames; a shared name lets one writer rename another's file, causing missing-file errors
    partway through a run on an allocated node.
    """
    import concurrent.futures as cf
    import multiprocessing

    keys = list(TINY_KEYS[:3])
    # Pre-materialize datasets as a campaign does before launching its array. Every call rewrites
    # the registry, even without generating data. Repeat writes to force overlap; one pass per
    # writer finishes too quickly to expose the race.
    datasets.materialize(keys, tmp_path)

    # forkserver rather than fork: forking a multithreaded pytest process can deadlock the child.
    context = multiprocessing.get_context("forkserver")
    with cf.ProcessPoolExecutor(max_workers=4, mp_context=context) as pool:
        registries = list(pool.map(_materialize_into, [(keys, str(tmp_path), 40)] * 4))

    assert all(r["n_datasets"] == len(keys) for r in registries)
    assert json.loads((tmp_path / datasets.REGISTRY_NAME).read_text())["n_datasets"] == len(keys)
    assert not list(tmp_path.glob(".*tmp")), "a temp file was left behind"


def _materialize_into(args) -> dict:
    """Module-level so a process pool can pickle it."""
    keys, out, times = args
    for _ in range(times):
        registry = datasets.materialize(keys, out)
    return registry


def test_a_writer_keeps_the_datasets_a_peer_added_meanwhile(tmp_path, monkeypatch):
    """Two jobs adding different datasets: the one that writes its registry last keeps the other's.

    Each call reads the registry before generating and writes it after, so a writer that did not
    re-read it would replace a peer's new entries with its own older copy.
    """
    first, second = TINY_KEYS[0], TINY_KEYS[1]
    generate = datasets.generate_one

    def peer_finishes_first(key):
        if key.label == first.label:
            monkeypatch.setattr(datasets, "generate_one", generate)
            datasets.materialize([second], tmp_path)
        return generate(key)

    monkeypatch.setattr(datasets, "generate_one", peer_finishes_first)
    datasets.materialize([first], tmp_path)

    recorded = json.loads((tmp_path / datasets.REGISTRY_NAME).read_text())
    assert set(recorded["keys"]) == {first.label, second.label}
    assert recorded["n_datasets"] == 2


def test_a_materialized_file_gets_the_mode_the_umask_allows(tmp_path):
    """`mkstemp` creates files 0600, and a store shared through `ANDREY_BENCH_DATA` is read by other
    users."""
    import os

    umask = os.umask(0)
    os.umask(umask)
    datasets.materialize([TINY_KEYS[0]], tmp_path)
    for path in (tmp_path / TINY_KEYS[0].filename, tmp_path / datasets.REGISTRY_NAME):
        assert path.stat().st_mode & 0o777 == 0o666 & ~umask, path


def test_materialize_refuses_a_registry_of_another_schema(tmp_path):
    """Accumulating onto records of an unknown shape would stamp them with this schema's number."""
    (tmp_path / datasets.REGISTRY_NAME).write_text(json.dumps({"schema": 0, "keys": {}}))
    with pytest.raises(ValueError, match="schema"):
        datasets.materialize([TINY_KEYS[0]], tmp_path)


# --------------------------------------------------------------------------------------------------
# provenance — deterministic env registry.
# --------------------------------------------------------------------------------------------------


def test_provenance_capture_writes_expected_keys(tmp_path):
    ts = "2026-07-19T00:00:00Z"
    prov = provenance.capture(tmp_path, timestamp=ts)

    prov_path = tmp_path / "provenance.json"
    assert prov_path.exists()
    on_disk = json.loads(prov_path.read_text())
    assert on_disk == prov

    for key in ("versions", "blas", "cpu", "git_sha", "timestamp"):
        assert key in prov, f"missing provenance key: {key}"

    assert prov["timestamp"] == ts  # passed through verbatim, no wall-clock read
    assert prov["versions"].get("packages"), "no package set captured"
    assert prov["blas"].get("numpy"), "no numpy linkage captured"
    # The code fingerprint is real, not silently empty.
    assert prov["git_sha"]


def test_a_missing_venv_records_its_blas_as_an_error():
    """A missing venv degrades to a recorded error, never a raise: a benchmark that dies at startup
    because a venv is absent is worse than one that records the absence."""
    missing = provenance._BENCH_VENV.parent / ".venv-does-not-exist"
    assert "error" in provenance._blas_identity(missing)


def test_the_parquet_has_the_metric_columns_without_a_scored_record():
    """A benchmark whose every fit was censored writes the same schema as one where every fit
    scored."""
    from andrey_bench.integration import to_dataframe

    assert {"shd", "skeleton_f1", "arrowhead_f1"} <= set(to_dataframe([]).columns)
