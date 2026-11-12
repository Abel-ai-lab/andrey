"""The LDECC adapter: its contract, its start script, and the authors' code on planted cases.

The contract and script tests need no environment. The tests of the authors' code skip unless
``start_ldecc.sh`` has built the environment and its two exports are set; they run the adapter in
that environment's interpreter, as the lane does, because the code's packages are not bench-env's.
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

from andrey_bench.adapters import gupta_ldecc as ldecc  # noqa: E402
from andrey_bench.adjustment_lane import PairAdapter  # noqa: E402
from andrey_bench.contracts import SetupAdapter, SolutionAdapter  # noqa: E402

BENCH = Path(__file__).resolve().parents[1]
PYTHON_ENV = "ANDREY_BENCH_LDECC_PYTHON"
PAIR = {"treatment": 1, "outcome": 3}


def test_the_adapter_fills_the_pair_contract_with_the_answer_the_andrey_counterpart_gives():
    from andrey_bench.adapters.andrey_adjustment import METHODS

    adapter = ldecc.GuptaLDECC()
    assert isinstance(adapter, PairAdapter)
    assert isinstance(adapter, SolutionAdapter) and isinstance(adapter, SetupAdapter)
    row = METHODS["ldecc"]
    assert (adapter.estimand, adapter.set_kind) == (row.estimand, row.set_kind)
    assert (adapter.identified_from, adapter.unordered) == (row.identified_from, row.unordered)
    assert adapter.name == "ldecc.authors" and adapter.package == adapter.algorithm == "ldecc"
    assert adapter.package_version == ldecc.COMMIT[:12]
    assert adapter.output_type == "effects" and adapter.python_env == PYTHON_ENV
    assert adapter.params() == {"alpha": row.alpha, "indep_test": "fisherz"}


def test_a_missing_code_directory_names_the_script_that_fetches_it(monkeypatch, tmp_path):
    adapter = ldecc.GuptaLDECC()
    data = np.zeros((10, 4))
    for value in (None, str(tmp_path)):
        if value is None:
            monkeypatch.delenv(ldecc.CODE_ENV, raising=False)
        else:
            monkeypatch.setenv(ldecc.CODE_ENV, value)
        with pytest.raises(FileNotFoundError, match="start_ldecc.sh"):
            adapter.setup()
        with pytest.raises(FileNotFoundError, match="start_ldecc.sh"):
            adapter.fit(data, {**adapter.params(), **PAIR})


def test_the_code_has_fishers_z_test_only():
    adapter = ldecc.GuptaLDECC()
    with pytest.raises(ValueError, match="chisq"):
        adapter.fit(np.zeros((10, 4)), {**adapter.params(), **PAIR, "indep_test": "chisq"})


def test_the_start_script_fetches_the_commit_the_adapter_reads_and_pins_every_package():
    script = (BENCH / "start_ldecc.sh").read_text()
    assert f'COMMIT="{ldecc.COMMIT}"' in script
    assert re.search(r'^SHA256="[0-9a-f]{64}"$', script, re.MULTILINE)
    assert (
        f"export {ldecc.CODE_ENV}=" in script and f"export {ldecc.GuptaLDECC.python_env}=" in script
    )
    lines = [x for x in (BENCH / "requirements-ldecc.txt").read_text().splitlines() if x.strip()]
    lines = [x for x in lines if not x.startswith("#")]
    assert all(re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.]*", x) for x in lines), lines
    names = {x.split("==")[0].lower() for x in lines}
    assert {
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "networkx",
        "causal-learn",
    } <= names


# ----- the authors' code, in its own environment -----------------------------------------------


def _reference_env() -> bool:
    return bool(os.environ.get(ldecc.CODE_ENV) and os.environ.get(PYTHON_ENV))


needs_ldecc = pytest.mark.skipif(
    not _reference_env(), reason="the LDECC authors' code is absent (start_ldecc.sh)"
)

#: Runs the adapter and, beside it, the code as its notebooks call it; prints both as JSON.
_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import pandas as pd
from andrey_bench.adapters import gupta_ldecc as g
from andrey_bench.adjustment_lane import unpack_effects

X = np.load(sys.argv[2])
pair = json.loads(sys.argv[3])
adapter = g.GuptaLDECC()
adapter.setup()
records, tests = unpack_effects(adapter.fit(X, {**adapter.params(), **pair}))
algorithm, _ = g._reference()
run = algorithm(treatment_node=f"V{pair['treatment']}", outcome_node=f"V{pair['outcome']}",
                alpha=0.05, use_ci_oracle=False, ldecc_do_checks=True)
result = run.run(pd.DataFrame(X, columns=[f"V{i}" for i in range(X.shape[1])]))
code = {k: sorted(result[k]) for k in ("tmt_parents", "tmt_children", "unoriented")}
print(json.dumps({"records": records, "tests": tests, "code": code,
                  "code_tests": run.ci_test_calls["total"]}))
"""


def _run(tmp_path: Path, data: np.ndarray, pair: dict[str, int]) -> dict:
    np.save(tmp_path / "data.npy", data)
    done = subprocess.run(
        [
            os.environ[PYTHON_ENV],
            "-I",
            "-c",
            _SCRIPT,
            str(BENCH),
            str(tmp_path / "data.npy"),
            json.dumps(pair),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def _collider(n: int = 4000, seed: int = 0) -> np.ndarray:
    """``a -> x <- b`` and ``x -> y``: columns ``a, x, b, y``, the treatment is column 1."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    x = 0.9 * a + 0.9 * b + rng.standard_normal(n)
    y = 0.9 * x + rng.standard_normal(n)
    return np.column_stack([a, x, b, y])


def _chain(n: int = 4000, seed: int = 0) -> np.ndarray:
    """``a -> x -> y``: columns ``a, x, y``; every orientation of the two edges but a collider
    is in the class, so the treatment's parents are not determined."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(n)
    x = 0.9 * a + rng.standard_normal(n)
    y = 0.9 * x + rng.standard_normal(n)
    return np.column_stack([a, x, y])


@needs_ldecc
def test_the_code_orients_a_collider_and_the_one_set_is_the_parents(tmp_path):
    out = _run(tmp_path, _collider(), PAIR)
    [record] = out["records"]
    assert record == {
        "treatment": 1,
        "outcome": 3,
        "estimand": "total",
        "nonzero": None,
        "identifiable": True,
        "adjustment_sets": [[0, 2]],
        "set_kind": "local_parents",
    }
    assert out["tests"] == out["code_tests"] > 0
    assert out["code"] == {"tmt_parents": ["V0", "V2"], "tmt_children": ["V3"], "unoriented": []}


@needs_ldecc
def test_a_chain_leaves_the_parents_open_and_the_sets_are_the_local_parent_sets(tmp_path):
    out = _run(tmp_path, _chain(), {"treatment": 1, "outcome": 2})
    [record] = out["records"]
    assert out["code"]["unoriented"] == ["V0", "V2"] and out["code"]["tmt_parents"] == []
    # {a}, {}, and {y} are the parent sets of the class; the outcome is never in a set.
    assert record["adjustment_sets"] == [[], [0]]
    assert record["identifiable"] is False and record["set_kind"] == "local_parents"


@needs_ldecc
def test_the_runner_runs_the_adapter_in_its_own_environment_and_the_lane_reads_its_answer():
    from andrey_bench.adjustment_lane import answer_records
    from andrey_bench.runner import run_task

    adapter = ldecc.GuptaLDECC()
    params = {**adapter.params(), **PAIR}
    [measure] = run_task(adapter, _collider(), params, cap_wall_s=300, repeats=1, warmup=0)
    assert measure["status"] == "ok", measure
    [record], tests = answer_records(adapter, np.asarray(measure["adj"]), params)
    assert record["adjustment_sets"] == [[0, 2]] and tests > 0
