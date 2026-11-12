"""The LDP and LD3 adapters: their contract, their start script, and the authors' code on planted
cases.

The contract and script tests need no environment. The tests of the authors' code skip unless
``start_maasch.sh`` has built the environment and its three exports are set; they run the adapters
in that environment's interpreter, as the lane does, because the code's packages are not
bench-env's.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from andrey_bench.adapters import maasch_ldp as maasch  # noqa: E402
from andrey_bench.adjustment_lane import PairAdapter  # noqa: E402
from andrey_bench.contracts import SetupAdapter, SolutionAdapter  # noqa: E402

BENCH = Path(__file__).resolve().parents[1]
PYTHON_ENV = "ANDREY_BENCH_MAASCH_PYTHON"
PAIR = {"treatment": 1, "outcome": 2}
ADAPTERS = {"ldp": maasch.MaaschLDP, "ld3": maasch.MaaschLD3}


@pytest.mark.parametrize("method", sorted(ADAPTERS))
def test_the_adapters_fill_the_pair_contract_with_the_answer_the_andrey_counterpart_gives(method):
    from andrey_bench.adapters.andrey_adjustment import METHODS

    adapter = ADAPTERS[method]()
    assert isinstance(adapter, PairAdapter)
    assert isinstance(adapter, SolutionAdapter) and isinstance(adapter, SetupAdapter)
    row = METHODS[method]
    assert (adapter.estimand, adapter.set_kind) == (row.estimand, row.set_kind)
    assert (adapter.identified_from, adapter.unordered) == (row.identified_from, row.unordered)
    assert adapter.name == f"{method}.authors"
    assert adapter.package == adapter.algorithm == method
    assert adapter.output_type == "effects" and adapter.python_env == PYTHON_ENV
    assert adapter.params() == {"alpha": row.alpha, "indep_test": "fisherz"}
    assert (
        adapter.package_version == {"ldp": maasch.LDP_COMMIT, "ld3": maasch.LD3_COMMIT}[method][:12]
    )


@pytest.mark.parametrize(("method", "variable"), [("ldp", maasch.LDP_ENV), ("ld3", maasch.LD3_ENV)])
def test_a_missing_code_directory_names_the_script_that_fetches_it(
    method, variable, monkeypatch, tmp_path
):
    adapter = ADAPTERS[method]()
    data = np.zeros((10, 4))
    for value in (None, str(tmp_path)):
        if value is None:
            monkeypatch.delenv(variable, raising=False)
        else:
            monkeypatch.setenv(variable, value)
        with pytest.raises(FileNotFoundError, match="start_maasch.sh"):
            adapter.setup()
        with pytest.raises(FileNotFoundError, match="start_maasch.sh"):
            adapter.fit(data, {**adapter.params(), **PAIR})


@pytest.mark.parametrize("method", sorted(ADAPTERS))
def test_the_lane_runs_the_two_tests_the_code_names_and_no_other(method):
    adapter = ADAPTERS[method]()
    with pytest.raises(ValueError, match="kci"):
        adapter.fit(np.zeros((10, 4)), {**adapter.params(), **PAIR, "indep_test": "kci"})


def test_the_start_script_fetches_the_commits_the_adapters_read_and_pins_every_package():
    script = (BENCH / "start_maasch.sh").read_text()
    assert f'LDP_COMMIT="{maasch.LDP_COMMIT}"' in script
    assert f'LD3_COMMIT="{maasch.LD3_COMMIT}"' in script
    assert len(re.findall(r'^LD[P3]_SHA256="[0-9a-f]{64}"$', script, re.MULTILINE)) == 2
    for variable in (maasch.LDP_ENV, maasch.LD3_ENV, PYTHON_ENV):
        assert f"export {variable}=" in script
    lines = [x for x in (BENCH / "requirements-maasch.txt").read_text().splitlines() if x.strip()]
    lines = [x for x in lines if not x.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.]*", x) for x in lines), lines
    names = {x.split("==")[0].lower() for x in lines}
    assert {"numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn"} <= names
    assert {"networkx", "causal-learn"} <= names


# ----- the authors' code, in its own environment -----------------------------------------------


def _reference_env() -> bool:
    return all(os.environ.get(v) for v in (maasch.LDP_ENV, maasch.LD3_ENV, PYTHON_ENV))


needs_maasch = pytest.mark.skipif(
    not _reference_env(), reason="the LDP and LD3 authors' code is absent (start_maasch.sh)"
)

#: Runs one adapter and prints its records and test count as JSON.
_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from andrey_bench.adapters import maasch_ldp as m
from andrey_bench.adjustment_lane import unpack_effects

adapter = {"ldp": m.MaaschLDP, "ld3": m.MaaschLD3}[sys.argv[2]]()
adapter.setup()
params = {**adapter.params(), **json.loads(sys.argv[4])}
records, tests = unpack_effects(adapter.fit(np.load(sys.argv[3]), params))
print(json.dumps({"records": records, "tests": tests}))
"""


def _run(tmp_path: Path, method: str, data: np.ndarray, **settings) -> tuple[dict, int]:
    np.save(tmp_path / "data.npy", data)
    done = subprocess.run(
        [
            os.environ[PYTHON_ENV],
            "-I",
            "-c",
            _SCRIPT,
            str(BENCH),
            method,
            str(tmp_path / "data.npy"),
            json.dumps({**PAIR, **settings}),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    out = json.loads(done.stdout.strip().splitlines()[-1])
    [record] = out["records"]
    return record, out["tests"]


def _partition(n: int = 4000, seed: int = 0, effect: float = 0.9) -> np.ndarray:
    """A confounder ``z1`` (``z1 -> x``, ``z1 -> y``), a cause of the outcome ``z4``, an
    instrument ``z5``, an isolated ``z8``, and ``x -> y``; the columns are ``z1, x, y, z4, z5, z8``.
    """
    rng = np.random.default_rng(seed)
    z1, z4, z5, z8 = (rng.standard_normal(n) for _ in range(4))
    x = 0.9 * z1 + 0.9 * z5 + rng.standard_normal(n)
    y = 0.9 * z1 + 0.9 * z4 + effect * x + rng.standard_normal(n)
    return np.column_stack([z1, x, y, z4, z5, z8])


@needs_maasch
def test_ldp_finds_the_confounder_as_the_valid_set_when_an_instrument_exists(tmp_path):
    record, tests = _run(tmp_path, "ldp", _partition())
    assert record == {
        "treatment": 1,
        "outcome": 2,
        "estimand": "total",
        "nonzero": None,
        "identifiable": True,
        "adjustment_sets": [[0]],
        "set_kind": "valid",
    }
    assert tests > 0


@needs_maasch
def test_ldp_calls_the_effect_not_identifiable_without_an_instrument(tmp_path):
    """The code's Z5 criterion needs an instrument: with the confounder alone it gives no set."""
    record, _ = _run(tmp_path, "ldp", _partition()[:, :3])
    assert record["identifiable"] is False and record["adjustment_sets"] == []
    assert record["set_kind"] is None and record["nonzero"] is None


@needs_maasch
def test_ld3_finds_the_parents_of_the_outcome_and_the_direct_effect_criterion(tmp_path):
    record, tests = _run(tmp_path, "ld3", _partition())
    assert record == {
        "treatment": 1,
        "outcome": 2,
        "estimand": "direct",
        "nonzero": True,
        "identifiable": None,
        "adjustment_sets": [[0, 3]],
        "set_kind": "parents_of_outcome",
    }
    assert tests > 0


@needs_maasch
def test_ld3_reads_no_direct_effect_when_the_treatment_does_not_cause_the_outcome(tmp_path):
    record, _ = _run(tmp_path, "ld3", _partition(effect=0.0))
    assert record["nonzero"] is False and record["adjustment_sets"] == [[0, 3]]


@needs_maasch
@pytest.mark.parametrize("method", sorted(ADAPTERS))
def test_the_chi_square_test_runs_on_discrete_columns(tmp_path, method):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 3, size=(600, 5))
    record, tests = _run(tmp_path, method, data, indep_test="chisq", alpha=0.01)
    assert record["estimand"] == ADAPTERS[method].estimand and tests > 0


@needs_maasch
@pytest.mark.parametrize(("method", "expected"), [("ldp", [[0]]), ("ld3", [[0, 3]])])
def test_the_runner_runs_the_adapters_in_their_own_environment_and_the_lane_reads_the_answer(
    method, expected
):
    from andrey_bench.adjustment_lane import answer_records
    from andrey_bench.runner import run_task

    adapter = ADAPTERS[method]()
    params = {**adapter.params(), **PAIR}
    [measure] = run_task(adapter, _partition(), params, cap_wall_s=300, repeats=1, warmup=0)
    assert measure["status"] == "ok", measure
    [record], tests = answer_records(adapter, np.asarray(measure["adj"]), params)
    assert record["adjustment_sets"] == expected and tests > 0
