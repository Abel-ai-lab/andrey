"""The XGES driver's field, and the two switches that can silently change what it measures.

`xges_ges.py` is a benchmark, so most of it is literals and a loop. Two things in it can go wrong
without failing: a `--solutions` typo trims the field and the missing rows read as solutions that
failed, and the parallel work gate reaches solutions that do not have one. Both are checked here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1]


def _driver():
    """Import `xges_ges.py` by path - it is a script beside the package, not inside it."""
    spec = importlib.util.spec_from_file_location("xges_ges_driver", BENCH / "xges_ges.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _driver()


def test_the_field_is_what_the_campaign_says_it_is() -> None:
    """A 2x2 on the penalty: three Andrey backends at each weight, both XGES weights."""
    names = [a.name for a in driver.solutions(1.0)]
    assert names == [
        "andrey.ges.numpy.serial",
        "andrey.ges.numpy.parallel",
        "andrey.ges.numba.serial",
        "andrey.ges.numpy.serial.heavy",
        "andrey.ges.numpy.parallel.heavy",
        "andrey.ges.numba.serial.heavy",
        "causal-learn.ges",
        "xges.ges",
        "xges.ges.shipped",
        "baseline.empty",
    ]
    # Names are the part-file identity, so a duplicate loses a measurement to a resume.
    assert len(set(names)) == len(names)
    # The floor is not optional: every solution list carries it.
    assert "baseline.empty" in names
    # The reference implementation of the same algorithm. Without it a quality difference cannot be
    # attributed to Andrey rather than to GES.
    assert "causal-learn.ges" in names


def test_each_penalty_gets_its_own_names_and_its_own_weight() -> None:
    """The 2x2 only exists if the two weights are distinguishable in the records.

    Sharing names would put both weights in one part file; sharing a weight would make the second
    set of arms a duplicate measurement under a different label. Both are silent.
    """
    field = {a.name: a for a in driver.solutions(1.0)}

    light = [n for n in field if n.startswith("andrey") and not n.endswith(".heavy")]
    heavy = [n for n in field if n.endswith(".heavy")]
    assert len(light) == len(heavy) == 3
    assert {n + ".heavy" for n in light} == set(heavy)

    for name in light:
        assert field[name].params()["lambda_value"] == 1.0
        assert field[name + ".heavy"].params()["lambda_value"] == driver.HEAVY_LAMBDA

    # The weight Andrey's heavy arms carry is the one XGES ships, or the 2x2 is not square.
    assert driver.HEAVY_LAMBDA == field["xges.ges.shipped"].params()["alpha"]
    assert (
        field["xges.ges"].params()["alpha"]
        == field["andrey.ges.numpy.serial"].params()["lambda_value"]
    )


def test_the_work_gate_reaches_only_the_solution_that_has_one() -> None:
    """Declaring the gate on a serial fit or a competitor would claim a knob it does not have."""
    field = driver.solutions(1.0)

    # Undeclared, the shipped default stays in force and the field runs as one group.
    assert driver.groups(field, None) == [(field, None)]

    # Both parallel arms, one per penalty: the gate is a property of the mode, not of the weight.
    gated = {a.name for group, env in driver.groups(field, 12345) if env is not None for a in group}
    assert gated == {"andrey.ges.numpy.parallel", "andrey.ges.numpy.parallel.heavy"}
    assert all(a.mode == "parallel" for a in field if a.name in gated)

    # Every solution still runs exactly once across the groups.
    ran = [a.name for group, _env in driver.groups(field, 12345) for a in group]
    assert sorted(ran) == sorted(a.name for a in field)

    envs = [env for _group, env in driver.groups(field, 12345) if env is not None]
    assert envs == [{"ANDREY_GES_PARALLEL_MIN_WORK": "12345"}]


def test_a_trimmed_field_drops_the_empty_group() -> None:
    """A ladder run without the parallel arm must not launch an empty gated batch."""
    serial_only = [a for a in driver.solutions(1.0) if getattr(a, "mode", "") != "parallel"]
    assert [group for group, _env in driver.groups(serial_only, 12345)] == [serial_only]


def test_an_unknown_solution_name_stops_the_run(capsys: pytest.CaptureFixture) -> None:
    """A typo must fail at launch, not quietly measure a smaller field.

    The rows a trimmed field leaves out are indistinguishable afterwards from a solution that ran
    and failed, so this is the difference between a coverage gap and a finding.
    """
    with pytest.raises(SystemExit):
        driver.main(
            [
                "--out",
                "unused",
                "--ladder",
                "20",
                "--solutions",
                "andrey.ges.numpy.serial",
                "xges.gess",
            ]
        )
    assert "unknown solutions" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--seeds", "0"), ("--repeats", "0"), ("--repeats", "-1"), ("--warmup", "-1")],
)
def test_a_count_that_measures_nothing_stops_the_run(
    flag: str, value: str, capsys: pytest.CaptureFixture
) -> None:
    """No seeds or no timed repeats would finish at once and exit 0 with an empty table."""
    with pytest.raises(SystemExit):
        driver.main(["--out", "unused", "--ladder", "20", flag, value])
    assert flag in capsys.readouterr().err


def test_a_resume_does_not_turn_a_failed_run_green(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """An errored fit from an earlier sitting still fails the sitting that resumes it.

    A resume skips every part file on disk, errored or not, so that failure is in the assembled
    table and nowhere in the resuming sitting's own records.
    """
    from andrey_bench.contracts import RunRecord
    from andrey_bench.integration import BatchResult

    errored = RunRecord(
        run_id="r",
        dataset_id="t",
        name="baseline.empty",
        algorithm="baseline",
        package="baseline",
        package_version="0",
        backend="numpy",
        mode="serial",
        output_type="dag",
        machine_id="m",
        cores=1,
        device="cpu",
        topology="er",
        functional="linear",
        noise="gaussian",
        standardize="standardized",
        density=2.0,
        d=20,
        n=200,
        dtype="float64",
        data_seed=0,
        repeat=0,
        status="error",
        error_text="the fit raised",
    )

    def first_sitting(_store, _group, _keys, *, records_dir, **_kw) -> BatchResult:
        part = driver.rundir.part_path(records_dir, "t", errored.name, "m")
        driver.rundir.write_part(part, [errored])
        return BatchResult(records=[errored])

    monkeypatch.setattr(driver, "check_interpreters", lambda _field: {})
    monkeypatch.setattr(
        driver.rundir, "provenance_once", lambda _out, **_kw: {"git_sha": "test", "blas": {}}
    )
    argv = [
        "--out",
        str(tmp_path / "run"),
        "--ladder",
        "20",
        "--seeds",
        "1",
        "--cores",
        "1",
        "--machine-id",
        "m",
        "--solutions",
        "baseline.empty",
    ]

    monkeypatch.setattr(driver, "run_batch", first_sitting)
    assert driver.main(argv) == 1
    capsys.readouterr()

    monkeypatch.setattr(driver, "run_batch", lambda *_a, **_kw: BatchResult(skipped=1))
    assert driver.main([*argv, "--resume"]) == 1
    assert "failed fits : 1" in capsys.readouterr().out
