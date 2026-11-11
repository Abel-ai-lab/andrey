"""The Y-structure lane: the driver's records and scores, and the authors' R code.

The driver tests replace ``run_task`` with an in-process fit, so they need no harness environment;
the tests of the authors' code skip unless the R lane and ``start_sbcd_r.sh`` have run, as the other
reference lanes' do.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_r

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import y_structures_cpu  # noqa: E402

from andrey_bench import local_lane as L  # noqa: E402
from andrey_bench import runner, selection_models  # noqa: E402
from andrey_bench.adapters import versteeg_ystructures as sbcd  # noqa: E402


class StandInClaims:
    """A claim solution that claims every true ancestral pair, the best answer there is."""

    name = "stand-in.claims"
    package = "test"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    algorithm = "stand_in"
    output_type = "claims"

    def params(self):
        return {"alpha": 0.01}

    def fit(self, data, params):
        raise AssertionError("the fake run_task answers for this solution")


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """The driver on the fixed graph, its fits run in this process."""
    truths = {}
    for seed in (0, 1):
        draw = selection_models.sample("fixed", n=300, seed=seed)
        truths[draw.biased.tobytes()] = truths[draw.unbiased.tobytes()] = draw.ancestral
    calls = []

    def fit_here(adapter, data, params, **kwargs):
        calls.append((adapter.name, dict(params)))
        ancestral = truths[np.asarray(data).tobytes()]
        scores = np.where(ancestral, 5.0, 0.0)
        scores[-1] = 0.0  # no claim about the context
        scores[:, -1] = 0.0
        return [{"status": "ok", "adj": L.pack_answer(scores, None), "wall_s": 0.01, "cpu_s": 0.01}]

    monkeypatch.setattr(y_structures_cpu, "run_task", fit_here)
    monkeypatch.setattr(y_structures_cpu, "check_interpreters", lambda field: {})
    monkeypatch.setattr(y_structures_cpu, "REFERENCES", [StandInClaims()])
    flags = ["--out", str(tmp_path / "run"), "--settings", "fixed", "--rows", "300", "--seeds", "2"]
    return tmp_path / "run", flags, calls


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]


def test_each_model_is_fitted_on_its_biased_and_its_unbiased_data(lane):
    out, flags, calls = lane
    assert y_structures_cpu.main(flags) == 0
    records = _records(out)
    assert len(records) == len(calls) == 4  # two models, two datasets each
    assert {(r["seed"], r["selection"]) for r in records} == {
        (s, b) for s in (0, 1) for b in ("biased", "unbiased")
    }
    assert all(r["status"] == "ok" and r["n_tests"] is None and r["d"] == 7 for r in records)
    assert all(0 < r["acceptance"] < 1 for r in records)


def test_the_claims_are_scored_against_the_ancestral_pairs_of_the_system_columns(lane):
    out, flags, _ = lane
    y_structures_cpu.main(flags)
    for r in _records(out):
        s = r["scores"]
        assert (s["claims"], s["fp"], s["precision"], s["recall"]) == (4, 0, 1.0, 1.0)
        assert s["tp"] == s["positives"] == 4 and s["average_precision"] == pytest.approx(1.0)
        assert s["pairs"] == 30 and s["base_rate"] == pytest.approx(4 / 30)
        assert len(r["claims"]) == 4 and all(score == 5.0 for *_, score in r["claims"])


def test_the_alpha_reaches_every_solution_that_has_one(lane):
    out, flags, calls = lane
    y_structures_cpu.main(flags + ["--alpha", "0.02"])
    assert {params["alpha"] for _, params in calls} == {0.02}


def test_a_finished_run_is_not_overwritten_and_resumes_without_new_fits(lane):
    out, flags, calls = lane
    y_structures_cpu.main(flags)
    with pytest.raises(SystemExit, match="already holds a run"):
        y_structures_cpu.main(flags)
    assert y_structures_cpu.main(flags + ["--resume"]) == 0
    assert len(calls) == 4 and len(_records(out)) == 4


def test_a_run_is_rescored_from_its_stored_claims(lane):
    out, flags, _ = lane
    y_structures_cpu.main(flags)
    damaged = _records(out)
    for r in damaged:
        r["scores"]["tp"] = -1
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in damaged))
    assert y_structures_cpu.main(["--out", str(out), "--rescore"]) == 0
    assert all(r["scores"]["tp"] == 4 for r in _records(out))


def test_the_papers_settings_and_sample_size():
    assert set(selection_models.SETTINGS) == {"fixed", "small", "large"}
    assert y_structures_cpu.ROWS == (10000,) and y_structures_cpu.ALPHA == 0.01
    assert [a.name for a in y_structures_cpu.solutions()] == ["sbcd.yst", "sbcd.yst_ext"]


def test_the_adapters_name_the_forms_and_the_script_that_builds_their_environment(monkeypatch):
    monkeypatch.delenv(sbcd.CODE_ENV, raising=False)
    yst, ext = sbcd.VersteegYStructures(False), sbcd.VersteegYStructures(True)
    assert (yst.name, ext.name) == ("sbcd.yst", "sbcd.yst_ext")
    assert isinstance(yst, L.ClaimAdapter) and yst.params() == {"alpha": 0.01}
    with pytest.raises(FileNotFoundError, match="start_sbcd_r.sh"):
        yst.setup()


def test_the_start_script_fetches_the_commit_the_adapter_names():
    script = (Path(__file__).resolve().parents[1] / "start_sbcd_r.sh").read_text()
    assert f'COMMIT="{sbcd.COMMIT}"' in script
    assert re.search(r'^SHA256="[0-9a-f]{64}"$', script, re.MULTILINE)
    assert sbcd.SOURCE in script or "R/wrapYStructures.R" in script


def _sbcd_env() -> bool:
    return bool(os.environ.get(sbcd.CODE_ENV))


@needs_r()
@pytest.mark.skipif(not _sbcd_env(), reason="the sbcd authors' code is absent (start_sbcd_r.sh)")
def test_the_authors_code_claims_only_true_ancestry_on_the_fixed_graph():
    adapter = sbcd.VersteegYStructures(False)
    adapter.setup()
    draw = selection_models.sample("fixed", n=10000, seed=0)
    for data, biased in ((draw.biased, True), (draw.unbiased, False)):
        answer = adapter.fit(data, adapter.params())
        scores, tests = L.unpack_answer(answer)
        assert tests is None and scores.shape == (7, 7)
        s = L.claim_scores(scores, draw.ancestral, draw.system)
        assert s["tp"] >= 1 and s["fp"] == 0, (biased, s)
    assert adapter.to_structure(answer).n_nodes == 7


@needs_r()
@pytest.mark.skipif(not _sbcd_env(), reason="the sbcd authors' code is absent (start_sbcd_r.sh)")
def test_the_extended_form_claims_at_least_what_the_full_form_does():
    full, ext = sbcd.VersteegYStructures(False), sbcd.VersteegYStructures(True)
    full.setup()
    draw = selection_models.sample("small", n=10000, seed=1)
    a, _ = L.unpack_answer(full.fit(draw.unbiased, full.params()))
    b, _ = L.unpack_answer(ext.fit(draw.unbiased, ext.params()))
    assert ((a > 0) <= (b > 0)).all()  # every claim of the full form is a claim of the extended one


def _bench_python_ready() -> bool:
    try:
        runner._bench_python()
    except FileNotFoundError:
        return False
    return True


@needs_r()
@pytest.mark.skipif(
    not (_sbcd_env() and _bench_python_ready()), reason="the sbcd code or the harness is absent"
)
def test_the_driver_runs_the_authors_code(tmp_path):
    out = tmp_path / "run"
    flags = ["--out", str(out), "--smoke", "--seeds", "1", "--solutions", "sbcd.yst"]
    assert y_structures_cpu.main(flags) == 0
    records = _records(out)
    assert {r["selection"] for r in records} == {"biased", "unbiased"}
    assert all(r["status"] == "ok" and r["scores"]["claims"] >= 1 for r in records)
