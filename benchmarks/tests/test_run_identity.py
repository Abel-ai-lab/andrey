"""One id per measurement — and never one id for two.

The id covers the environment, not only the dataset and the solution. Two benchmarks differing only in
a wall cap, a memory cap, a device or a thread split are two different measurements, and an id that
omitted those would give them one identity — so whoever pools the records could not tell them apart.

The fields are read off the environment the child was actually handed, which is what makes an
``extra_env`` overlay visible: the overlay is applied last, so an argument-derived id would miss it.
"""

from __future__ import annotations

import pytest

from andrey_bench import runner
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.contracts import RUN_ENV_FIELDS, environment_id
from andrey_bench.datasets import DatasetKey
from andrey_bench.integration import assemble_record

_ENV = {
    "backend": "numpy",
    "mode": "serial",
    "dtype": "float64",
    "device": "cpu",
    "cap_mem_mb": 8192.0,
    "cap_wall_s": 1800.0,
    "warmup": 1,
    "machine_id": "node-a",
    "threads": 16,
    "num_workers": 1,
}

_OTHER = {
    "backend": "numba",
    "mode": "parallel",
    "dtype": "float32",
    "device": "cuda",
    "cap_mem_mb": 4096.0,
    "cap_wall_s": 600.0,
    "warmup": 0,
    "machine_id": "node-b",
    "threads": 1,
    "num_workers": 16,
}


def _key(**overrides) -> DatasetKey:
    base = {
        "topology": "er",
        "functional": "linear",
        "noise": "gaussian",
        "density": 2.0,
        "d": 10,
        "n": 100,
        "seed": 0,
        "standardize": "standardized",
    }
    return DatasetKey(**{**base, **overrides})


def _record(measures, **overrides):
    """One assembled record, at the parent declaration the environment fields are read against."""
    kwargs = {"machine_id": "node-a", "cores": 16, **overrides}
    return assemble_record(_key(), BaselineEmpty(), 0, measures, None, None, **kwargs)


def _measures(**overrides) -> dict:
    base = {
        "status": "ok",
        "wall_s": 1.0,
        "cap_wall_s": 1800.0,
        "cap_mem_mb": 8192.0,
        "warmup": 1,
        "dtype": "float64",
        "env_device": "cpu",
        "env_threads": 16,
        "env_num_workers": 1,
        "env_ges_min_work": None,
        "env_hc_min_work": None,
    }
    return {**base, **overrides}


@pytest.mark.parametrize("field", RUN_ENV_FIELDS)
def test_every_environment_leg_moves_the_id(field):
    assert environment_id({**_ENV, field: _OTHER[field]}) != environment_id(_ENV)


def test_a_missing_leg_is_an_error_not_a_blank():
    """Hashing a missing field as empty would let a new environment axis vanish from the identity."""
    with pytest.raises(KeyError):
        environment_id({k: v for k, v in _ENV.items() if k != "cap_mem_mb"})


def test_an_unset_leg_is_not_the_same_as_an_empty_one():
    """An uncapped run, a blank device string and the literal ``"None"`` are three environments."""
    unset = environment_id({**_ENV, "cap_mem_mb": None})
    blank = environment_id({**_ENV, "cap_mem_mb": ""})
    literal = environment_id({**_ENV, "cap_mem_mb": "None"})
    assert len({unset, blank, literal, environment_id(_ENV)}) == 4


def test_a_numeric_leg_is_compared_by_value_not_by_type():
    """One cap declared ``1800`` and read back ``1800.0`` is one environment, or a requeued rung
    would dedup against nothing. A *string* cap is still a different environment."""
    assert environment_id({**_ENV, "cap_wall_s": 1800}) == environment_id(
        {**_ENV, "cap_wall_s": 1800.0}
    )
    assert environment_id({**_ENV, "cap_wall_s": "1800"}) != environment_id(_ENV)


#: measures key -> a value differing from :func:`_measures`, for every field the runner reports.
_MEASURES_FIELDS = {
    "env_threads": 1,
    "env_num_workers": 8,
    "env_ges_min_work": 5_000,
    "env_hc_min_work": 20_000,
    "env_device": "cuda",
    "cap_mem_mb": 4096.0,
    "cap_wall_s": 60.0,
    "warmup": 0,
    "dtype": "float32",
}


@pytest.mark.parametrize("key", sorted(_MEASURES_FIELDS))
def test_every_measures_leg_reaches_the_id(key):
    """The seam the field guard exists for: :func:`assemble_record` renaming a key the runner reports.

    ``environment_id`` alone cannot catch it — it is handed a dict that was already built. Two
    benchmarks differing only in a wall cap, a thread split or a device must stay two runs, so each
    key is moved through the real record assembly rather than through a hand-built environment.
    """
    base = _record(_measures())
    moved = _record(_measures(**{key: _MEASURES_FIELDS[key]}))
    assert moved.dataset_id == base.dataset_id  # same dataset ...
    assert moved.run_id != base.run_id  # ... measured under a different environment


def test_a_leg_the_runner_stopped_reporting_is_an_error():
    """A dropped key must not hash as ``None``: that collapses every run onto one field value."""
    measures = _measures()
    del measures["env_threads"]
    with pytest.raises(KeyError):
        _record(measures)


def test_the_same_run_re_measured_keeps_its_id():
    """Idempotent re-runs: a requeued rung must dedup against its first attempt, not duplicate it."""
    first = _record(_measures())
    second = _record(_measures(wall_s=1.4))
    assert first.run_id == second.run_id


def test_the_task_is_the_dataset_and_nothing_else():
    """``dataset_id`` is the dataset digest, so it moves with the bytes and with nothing else.

    A record measured on another machine, at another core budget or under another cap is the same dataset
    — that is what makes "the same data, measured twice" a statement the records can express.
    """
    record = _record(_measures())
    assert record.dataset_id == _key().digest
    elsewhere = _record(_measures(), machine_id="node-b", cores=1)
    assert elsewhere.dataset_id == record.dataset_id
    assert elsewhere.run_id != record.run_id


def test_one_dataset_read_at_two_dtypes_is_one_task_and_two_runs():
    """f32 and f64 are the same bytes on disk — one file, one dataset — but not one measurement.

    ``DatasetKey`` cannot separate them by construction, so if ``dtype`` were not an environment field
    the two would share a ``run_id`` and whoever pooled them could not tell them apart.
    """
    f64 = _record(_measures(dtype="float64"))
    f32 = _record(_measures(dtype="float32"))
    assert f64.dataset_id == f32.dataset_id
    assert f64.run_id != f32.run_id
    assert (f64.dtype, f32.dtype) == ("float64", "float32")


def test_the_parent_declaration_survives_a_censored_record():
    """``cores`` and ``device`` are on a timeout record, because timeouts are where a ceiling is read.

    Every child-observed column is ``None`` there, so a record that took them from the child would
    describe the run it failed to make rather than the one that was attempted.
    """
    dead = _record(_measures(status="timeout", env_device="cuda"), cores=8)
    assert dead.status == "timeout"
    assert (dead.cores, dead.device, dead.machine_id) == (8, "cuda", "node-a")
    # ``device`` is the benchmark's, not the backend's: ANDREY_DEVICE reaches every child, so a
    # CPU-only backend measured during a GPU benchmark is still a GPU-benchmark record.
    assert dead.backend != "torch-cuda"


def test_an_env_overlay_moves_the_legs_the_arguments_would_have_missed():
    base = {"OMP_NUM_THREADS": "16", "ANDREY_NUM_WORKERS": "1"}
    assert runner.resolved_env_fields(base) == {
        "env_threads": 16,
        "env_num_workers": 1,
        "env_ges_min_work": None,
        "env_hc_min_work": None,
        "env_device": "cpu",
    }
    # ``extra_env`` is overlaid last in the child env, so these are what the child really gets.
    overlaid = {**base, "OMP_NUM_THREADS": "4", "ANDREY_DEVICE": "cuda"}
    fields = runner.resolved_env_fields(overlaid)
    assert fields["env_threads"] == 4
    assert fields["env_device"] == "cuda"


def test_the_record_carries_the_child_observed_split_not_the_intent():
    """The parent's numbers are in ``run_id``; the columns are what the child came back with."""
    record = _record(_measures(threads=4, num_workers=2, device_id="cpu"))
    assert (record.threads, record.num_workers, record.device_id) == (4, 2, "cpu")
    # A child that never reported leaves them empty rather than echoing the parent's intent.
    dead = _record(_measures(status="timeout"))
    assert (dead.threads, dead.num_workers, dead.device_id) == (None, None, None)
    assert dead.run_id  # ... and the run is still identified
