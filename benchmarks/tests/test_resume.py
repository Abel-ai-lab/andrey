"""Resume: a unit's records are durable when it finishes, and a finished unit is not measured twice.

The property that matters is not "the second run was faster" — it is that an interrupted run plus a
resume is the *same table* as an uninterrupted one. Everything else here defends the two ways that
can quietly stop being true: a part file that exists before it is complete, and a skip key that stops
matching the id the records carry.

``BaselineEmpty`` throughout: the empty graph costs nothing to fit, and none of this is about an
algorithm.
"""

from __future__ import annotations

import os as _os

import pytest
from tiny_keys import TINY_KEYS

from andrey_bench import rundir
from andrey_bench.adapters.baseline_empty import BaselineEmpty
from andrey_bench.integration import run_batch

_ADAPTERS = [BaselineEmpty()]


def _keys(n: int = 1):
    return sorted(TINY_KEYS, key=lambda k: k.d)[:n]


def _batch(data_dir, records_dir, keys, **kwargs):
    return run_batch(
        data_dir,
        _ADAPTERS,
        keys,
        machine_id=kwargs.pop("machine_id", "tiny-cpu"),
        cap_wall_s=120.0,
        repeats=1,
        warmup=0,
        records_dir=records_dir,
        **kwargs,
    )


def test_a_finished_unit_is_written_once_and_then_skipped(tmp_path):
    data, records = tmp_path / "data", tmp_path / "records"

    first = _batch(data, records, _keys())
    parts = sorted(records.glob("*.parquet"))
    assert len(parts) == 1 and first.skipped == 0
    stamped = parts[0].stat().st_mtime_ns

    second = _batch(data, records, _keys())
    assert second.skipped == 1
    assert second.records == []  # nothing re-measured ...
    assert parts[0].stat().st_mtime_ns == stamped  # ... and nothing rewritten


def test_a_different_environment_does_not_skip(tmp_path):
    """The environment digest is in the part name, so another machine is another measurement."""
    data, records = tmp_path / "data", tmp_path / "records"

    _batch(data, records, _keys())
    other = _batch(data, records, _keys(), machine_id="another-node")

    assert other.skipped == 0
    assert len(list(records.glob("*.parquet"))) == 2


def test_an_interrupted_run_plus_a_resume_is_the_uninterrupted_table(tmp_path):
    """The headline invariant, and the reason `skipped` is asserted beside it: a resume that silently
    re-measured everything would produce this same set of ids and prove nothing."""
    whole = _batch(tmp_path / "d1", tmp_path / "r1", _keys(2))

    _batch(tmp_path / "d2", tmp_path / "r2", _keys(1))  # interrupted after one dataset
    resumed = _batch(tmp_path / "d2", tmp_path / "r2", _keys(2))
    assert resumed.skipped == 1

    table = rundir.assemble(tmp_path / "r2")
    assert set(table["run_id"]) == {r.run_id for r in whole.records}


def test_a_part_that_did_not_finish_writing_is_not_a_part(tmp_path, monkeypatch):
    """Use fault injection: a signal cannot reliably target the interval between two syscalls.

    An implementation writing straight to the final path leaves a file the glob would find, and the
    unit would be skipped for a parquet that was never completed.
    """
    data, records = tmp_path / "data", tmp_path / "records"

    # Rebind the name ``rundir`` looks ``replace`` up on, not the attribute of the shared ``os``
    # module: mutating that breaks ``datasets.materialize`` too, and the run would die before it ever
    # reached a part write — passing for a reason that has nothing to do with atomicity.
    class _NoReplace:
        def __getattr__(self, name):
            return getattr(_os, name)

        def replace(self, *_args, **_kwargs):
            raise OSError("killed before the rename")

    monkeypatch.setattr(rundir, "os", _NoReplace())
    with pytest.raises(OSError):
        _batch(data, records, _keys())
    monkeypatch.undo()

    assert list(records.glob("*.parquet")) == []
    assert _batch(data, records, _keys()).skipped == 0  # and the unit runs next time


def test_the_temp_file_is_not_mistaken_for_a_part(tmp_path):
    """A stray temp from a killed run must not join the table."""
    data, records = tmp_path / "data", tmp_path / "records"
    _batch(data, records, _keys())
    (records / ".orphan.parquet.tmp").write_bytes(b"not a parquet")

    assert len(rundir.assemble(records)) == 1


# --------------------------------------------------------------------------------------------------
# start_run: the one rule that decides whether a resume is the same run.
# --------------------------------------------------------------------------------------------------


def test_records_already_there_without_resume_is_refused(tmp_path):
    meta = {"ladder": [6], "repeats": 1}
    rundir.start_run(tmp_path, meta, resume=False)
    (tmp_path / "records" / "x.parquet").write_bytes(b"")

    with pytest.raises(SystemExit, match="--resume"):
        rundir.start_run(tmp_path, meta, resume=False)


def test_a_resume_of_a_different_run_is_refused_and_keeps_the_original(tmp_path):
    """Writing run_meta before the check would destroy what the check reads, and the corrected retry
    would then be refused too — permanently."""
    meta = {"ladder": [6], "repeats": 5}
    rundir.start_run(tmp_path, meta, resume=False)
    (tmp_path / "records" / "x.parquet").write_bytes(b"")

    with pytest.raises(SystemExit, match="repeats"):
        rundir.start_run(tmp_path, {**meta, "repeats": 3}, resume=True)

    assert '"repeats": 5' in (tmp_path / "run_meta.json").read_text()
    rundir.start_run(tmp_path, meta, resume=True)  # the original flags still resume


def test_provenance_is_captured_once(tmp_path):
    """A resume must not restamp the provenance of the build that produced the records here."""
    first = rundir.provenance_once(tmp_path, timestamp="2026-01-01T00:00:00+00:00")
    again = rundir.provenance_once(tmp_path, timestamp="2026-06-06T00:00:00+00:00")

    assert again["timestamp"] == first["timestamp"] == "2026-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------------------------------
# Datasets are reused rather than regenerated — a resume that rebuilt every dataset before it could
# skip a fit would spend the time the skip was for.
# --------------------------------------------------------------------------------------------------


def test_an_existing_dataset_is_reused(tmp_path):
    from andrey_bench import datasets

    datasets.materialize(_keys(), tmp_path)
    npz = next(tmp_path.glob("ds_*.npz"))
    stamped = npz.stat().st_mtime_ns

    datasets.materialize(_keys(), tmp_path)
    assert npz.stat().st_mtime_ns == stamped


def test_a_dataset_whose_recorded_call_disagrees_is_rebuilt(tmp_path):
    """Reuse is keyed on the generator call, not the filename: a file left by different code is not
    the dataset this key asks for."""
    import json

    from andrey_bench import datasets

    registry = datasets.materialize(_keys(), tmp_path)
    npz = next(tmp_path.glob("ds_*.npz"))
    stamped = npz.stat().st_mtime_ns

    label = next(iter(registry["keys"]))
    registry["keys"][label]["generation_call"] = {"n": -1}
    (tmp_path / datasets.REGISTRY_NAME).write_text(json.dumps(registry, indent=2, sort_keys=True))

    datasets.materialize(_keys(), tmp_path)
    assert npz.stat().st_mtime_ns != stamped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
