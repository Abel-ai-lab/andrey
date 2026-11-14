"""The MMB-by-MMB lane: the driver's rules for local and global solutions, the authors' code, and the
RFCI arm.

The driver tests replace ``run_task`` with an in-process fit, so they need no harness environment;
the tests of the authors' code and of pcalg skip unless their environments are built
(``start_mmb.sh``, the R lane), as the other reference lanes' do.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_r

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mmb_by_mmb_cpu  # noqa: E402

from andrey_bench import latent_networks, networks, runner  # noqa: E402
from andrey_bench import local_lane as L  # noqa: E402
from andrey_bench.adapters import andrey_mmb_by_mmb as andrey_mmb  # noqa: E402
from andrey_bench.adapters import xie_mmb_by_mmb as xie  # noqa: E402
from andrey_bench.adapters.baseline_empty import BaselineEmpty  # noqa: E402

ASIA = Path(__file__).resolve().parents[2] / "tests" / "unit" / "fixtures" / "asia.bif.gz"
_TAIL, _ARROW, _CIRCLE = 1, 2, 3


def test_the_authors_pag_marks_become_andrey_marks():
    """The code's ``pag[i, j]`` is the mark at ``j``: 1 arrowhead, -1 tail, 2 circle."""
    code = np.zeros((4, 4), dtype=int)
    code[0, 1], code[1, 0] = 1, -1  # 0 -> 1
    code[2, 1], code[1, 2] = 1, 2  # 2 o-> 1
    code[0, 3], code[3, 0] = 1, 1  # 0 <-> 3
    marks = xie.andrey_marks(code)
    assert (marks[0, 1], marks[1, 0]) == (_TAIL, _ARROW)
    assert (marks[2, 1], marks[1, 2]) == (_CIRCLE, _ARROW)
    assert (marks[0, 3], marks[3, 0]) == (_ARROW, _ARROW)
    assert marks.dtype == np.int8 and np.count_nonzero(marks) == 6


def test_only_the_targets_edges_are_kept():
    marks = np.arange(16, dtype=np.int8).reshape(4, 4) % 4
    kept = xie.at_target(marks, 1)
    assert np.count_nonzero(kept[[0, 2, 3]][:, [0, 2, 3]]) == 0
    np.testing.assert_array_equal(kept[1], marks[1])
    np.testing.assert_array_equal(kept[:, 1], marks[:, 1])


def test_the_adapter_names_its_environment_and_the_script_that_builds_it(monkeypatch):
    monkeypatch.delenv(xie.CODE_ENV, raising=False)
    adapter = xie.XieMMBbyMMB()
    assert isinstance(adapter, L.LocalAdapter) and adapter.local
    assert adapter.python_env == "ANDREY_BENCH_MMB_PYTHON"
    assert adapter.params() == {"alpha": 0.01, "max_k": 10}
    with pytest.raises(FileNotFoundError, match="start_mmb.sh"):
        adapter.setup()


def test_the_start_script_fetches_the_files_and_the_commit_the_adapter_reads():
    script = (Path(__file__).resolve().parents[1] / "start_mmb.sh").read_text()
    assert f'COMMIT="{xie.COMMIT}"' in script
    for name in xie.REQUIRED_FILES:
        assert re.search(rf"^[0-9a-f]{{64}} {re.escape(name)}$", script, re.MULTILINE), name
    lock = (Path(__file__).resolve().parents[1] / "requirements-mmb.txt").read_text()
    assert {line.split("==")[0] for line in lock.split()} == {"numpy", "scipy"}


# ----- the driver, with the fit run in this process --------------------------------------------


class PerfectLocal:
    """A local solution that answers with the truth: the stand-in of a method adapter."""

    name = "stand-in.perfect"
    package = "test"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    algorithm = "stand_in"
    output_type = "pag"
    local = True

    def params(self):
        return {"alpha": 0.05}

    def fit(self, data, params):  # replaced by the fake run_task, which knows the truth
        raise AssertionError("the fake run_task answers for this solution")


@pytest.fixture
def lane(tmp_path, monkeypatch):
    """The driver on asia with two hidden nodes, its fits run in this process."""
    folder = tmp_path / "networks"
    folder.mkdir()
    shutil.copy(ASIA, folder / "asia.bif.gz")
    monkeypatch.setenv(networks.CACHE_ENV, str(folder))
    net = networks.load_network("asia")
    truths = {}
    for seed in (0, 1):
        draw = latent_networks.sample(net, n=400, seed=seed, hidden=2)
        truths[draw.data.tobytes()] = latent_networks.truth(draw)
    calls = []

    def fit_here(adapter, data, params, **kwargs):
        calls.append((adapter.name, params.get(L.TARGET_PARAM)))
        if isinstance(adapter, PerfectLocal):
            truth, t = truths[np.asarray(data).tobytes()], params[L.TARGET_PARAM]
            answer = np.zeros_like(truth)
            answer[t], answer[:, t] = truth[t], truth[:, t]
            native = L.pack_answer(answer, 7)
        else:
            native = adapter.fit(data, params)
        return [{"status": "ok", "adj": native, "wall_s": 0.01, "cpu_s": 0.01}]

    monkeypatch.setattr(mmb_by_mmb_cpu, "run_task", fit_here)
    monkeypatch.setattr(mmb_by_mmb_cpu, "check_interpreters", lambda field: {})
    monkeypatch.setattr(mmb_by_mmb_cpu, "REFERENCES", [PerfectLocal()])
    monkeypatch.setattr(mmb_by_mmb_cpu, "ANDREY_ARMS", [])
    flags = ["--out", str(tmp_path / "run"), "--networks", "asia", "--hidden", "2", "--rows", "400"]
    return tmp_path / "run", flags + ["--seeds", "2", "--targets", "2"], calls


def _records(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "records.jsonl").read_text().splitlines()]


def test_a_local_solution_fits_once_per_target_and_a_global_one_once_per_dataset(lane):
    out, flags, calls = lane
    assert mmb_by_mmb_cpu.main(flags) == 0
    # Two datasets, two targets: four fits of the local solution, two of the empty answer.
    assert sum(c[0] == "stand-in.perfect" for c in calls) == 4
    assert [c for c in calls if c[0] == "baseline.empty"] == [("baseline.empty", None)] * 2
    records = _records(out)
    perfect = [r for r in records if r["solution"] == "stand-in.perfect"]
    empty = [r for r in records if r["solution"] == "baseline.empty"]
    assert len(perfect) == 4 and len(empty) == 2
    assert all(r["target"] is not None and list(r["scores"]) == [str(r["target"])] for r in perfect)
    assert all(r["target"] is None and len(r["scores"]) == 2 for r in empty)


def test_the_scores_are_read_at_each_target_against_the_truth(lane):
    out, flags, _ = lane
    mmb_by_mmb_cpu.main(flags)
    for r in _records(out):
        for s in r["scores"].values():
            if r["solution"] == "stand-in.perfect":
                assert s["shd"] == 0 and s["correct_edges"] == s["true_edges"] > 0
                assert r["n_tests"] == 7
            else:
                assert s["shd"] == s["empty_shd"] == s["true_edges"] and s["est_edges"] == 0
                assert r["n_tests"] is None


def test_the_alpha_reaches_every_solution_that_has_one(lane):
    out, flags, _ = lane
    mmb_by_mmb_cpu.main(flags + ["--alpha", "0.02"])
    perfect = [r for r in _records(out) if r["solution"] == "stand-in.perfect"]
    assert {r["params"]["alpha"] for r in perfect} == {0.02}
    assert {r["params"]["target"] for r in perfect} <= set(range(8))


def test_a_finished_run_is_not_overwritten_and_resumes_without_new_fits(lane):
    out, flags, calls = lane
    mmb_by_mmb_cpu.main(flags)
    done = len(calls)
    with pytest.raises(SystemExit, match="already holds a run"):
        mmb_by_mmb_cpu.main(flags)
    assert mmb_by_mmb_cpu.main(flags + ["--resume"]) == 0
    assert len(calls) == done and len(_records(out)) == 6


def test_a_run_is_rescored_from_its_stored_marks(lane):
    out, flags, _ = lane
    mmb_by_mmb_cpu.main(flags)
    before = _records(out)
    for r in before:  # damage the stored scores, so the rescore has something to restore
        for s in r["scores"].values():
            s["shd"] = -1
    (out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in before))
    assert mmb_by_mmb_cpu.main(["--out", str(out), "--rescore"]) == 0
    assert all(s["shd"] >= 0 for r in _records(out) for s in r["scores"].values())


def test_the_parts_of_a_run_split_the_datasets(lane):
    out, flags, _ = lane
    mmb_by_mmb_cpu.main(flags + ["--part", "1", "--parts", "2"])
    assert {r["seed"] for r in _records(out)} == {1}


def test_an_unknown_network_or_solution_is_refused(lane):
    out, flags, _ = lane
    with pytest.raises(SystemExit):
        mmb_by_mmb_cpu.main(["--out", str(out), "--networks", "nonesuch"])
    with pytest.raises(SystemExit):
        mmb_by_mmb_cpu.main(flags + ["--solutions", "nonesuch"])


def test_the_papers_four_networks_and_their_hidden_counts():
    assert mmb_by_mmb_cpu.NETWORKS == ("mildew", "alarm", "win95pts", "andes")
    assert [mmb_by_mmb_cpu.hidden_for(n, None) for n in mmb_by_mmb_cpu.NETWORKS] == [4, 4, 6, 10]
    assert mmb_by_mmb_cpu.hidden_for("pigs", 0) == 0
    with pytest.raises(SystemExit, match="no hidden count"):
        mmb_by_mmb_cpu.hidden_for("pigs", None)
    assert mmb_by_mmb_cpu.solutions()[-1].name == BaselineEmpty().name
    assert [a.name for a in mmb_by_mmb_cpu.ANDREY_ARMS] == [
        "andrey.mmb_by_mmb",
        "andrey.mmb_by_mmb.iamb",
    ]


def test_the_papers_networks_take_its_named_targets_and_others_the_most_connected():
    assert set(latent_networks.PAPER_TARGETS) == set(mmb_by_mmb_cpu.NETWORKS)
    assert [len(t) for t in latent_networks.PAPER_TARGETS.values()] == [4, 4, 5, 7]
    named = latent_networks.PAPER_TARGETS["alarm"]
    others = ("HUB", "A", "B", "C")
    arcs = tuple(("HUB", v) for v in (*named, "A", "B")) + (("A", "B"),)
    alarm = networks.Network("alarm", (*named, *others), arcs)
    assert mmb_by_mmb_cpu.targets_for(alarm, None) == named
    assert mmb_by_mmb_cpu.targets_for(alarm, 2) == ("HUB", "A")
    elsewhere = networks.Network("elsewhere", alarm.nodes, arcs)
    assert mmb_by_mmb_cpu.targets_for(elsewhere, None) == ("HUB", "A")


def test_the_run_records_its_targets_by_name(lane):
    out, flags, _ = lane
    mmb_by_mmb_cpu.main(flags)
    meta = json.loads((out / "run_meta.json").read_text())
    asia = networks.load_network("asia")
    assert meta["targets"] == {"asia": list(latent_networks.default_targets(asia, 2))}
    for r in _records(out):
        assert set(r["target_names"].values()) == set(meta["targets"]["asia"])


@pytest.mark.parametrize("blanket", ["total_conditioning", "iamb"])
def test_andreys_adapter_answers_a_target_of_a_hidden_variable_network_as_the_truth_does(blanket):
    """The fork with a hidden root of the authors' test: Andrey's answer at z is the truth's."""
    net = networks.Network(
        "fork", ("h", "x", "y", "z", "w"), (("h", "x"), ("h", "y"), ("x", "z"), ("w", "z"))
    )
    draw = latent_networks.sample(
        net, n=6000, seed=1, hidden=1, targets=("x", "y", "z", "w"), shuffle=False
    )
    adapter = andrey_mmb.AndreyMMBbyMMB(blanket)
    assert isinstance(adapter, L.LocalAdapter) and adapter.params() == {
        "alpha": 0.01,
        "indep_test": "fisherz",
        "blanket": blanket,
    }
    adapter.setup()
    for target in draw.targets:
        answer, tests = L.unpack_answer(
            adapter.fit(draw.data, {**adapter.params(), L.TARGET_PARAM: target})
        )
        assert tests is not None and tests > 0
        truth = latent_networks.truth(draw)
        assert L.target_edges(answer, target) == L.target_edges(truth, target), target
    assert adapter.to_structure(L.pack_answer(answer)).kind == "pag"


# ----- the references --------------------------------------------------------------------------


def _mmb_env() -> bool:
    return bool(os.environ.get(xie.CODE_ENV))


def _collider(n: int = 4000, seed: int = 0) -> np.ndarray:
    """``a -> t <- b`` and ``t -> c``: the target ``t`` is column 1."""
    rng = np.random.default_rng(seed)
    a, b = rng.standard_normal(n), rng.standard_normal(n)
    t = 0.9 * a + 0.9 * b + rng.standard_normal(n)
    c = 0.9 * t + rng.standard_normal(n)
    return np.column_stack([a, t, b, c])


@pytest.mark.skipif(not _mmb_env(), reason="the MMB-by-MMB authors' code is absent (start_mmb.sh)")
def test_the_authors_code_orients_the_collider_and_agrees_with_its_own_summary():
    X = _collider()
    adapter = xie.XieMMBbyMMB()
    answer, tests = L.unpack_answer(adapter.fit(X, {**adapter.params(), L.TARGET_PARAM: 1}))
    assert tests > 0
    at = L.target_edges(answer, 1)
    assert set(at) == {0, 2, 3}
    # a *-> t <-* b: an arrowhead at the target; t -> c: tail at t, arrowhead at c.
    assert at[0][0] == at[2][0] == _ARROW and at[3] == (_TAIL, _ARROW)
    # The code's own summary of the same run: parents, children, bidirected, and undetermined.
    run = xie._algorithm()(X, 1, 0.01, X.shape[1], 10)
    parents, children, bidirected, undetermined, count = run.mmb_by_mmb()
    assert count == tests
    assert set(parents) == {
        j for j, (mine, theirs) in at.items() if (mine, theirs) == (_ARROW, _TAIL)
    }
    assert set(children) == {
        j for j, (mine, theirs) in at.items() if (mine, theirs) == (_TAIL, _ARROW)
    }
    assert set(bidirected) == {j for j, m in at.items() if m == (_ARROW, _ARROW)}
    assert set(undetermined) == {j for j, m in at.items() if _CIRCLE in m}


@pytest.mark.skipif(not _mmb_env(), reason="the MMB-by-MMB authors' code is absent (start_mmb.sh)")
def test_the_authors_code_answers_a_target_of_a_hidden_variable_network_as_the_truth_does(tmp_path):
    """A seed where the code and the observed PAG agree at the target: the lane scores them."""
    net = networks.Network(
        "fork", ("h", "x", "y", "z", "w"), (("h", "x"), ("h", "y"), ("x", "z"), ("w", "z"))
    )
    draw = latent_networks.sample(
        net, n=6000, seed=1, hidden=1, targets=("x", "y", "z", "w"), shuffle=False
    )
    adapter = xie.XieMMBbyMMB()
    target = draw.targets[2]  # z
    answer, _ = L.unpack_answer(
        adapter.fit(draw.data, {**adapter.params(), L.TARGET_PARAM: target})
    )
    scores = L.target_edge_scores(answer, latent_networks.truth(draw), target)
    assert scores["adj_correct"] == scores["true_edges"] == 2 and scores["extra"] == 0


def _bench_python_ready() -> bool:
    try:
        runner._bench_python()
    except FileNotFoundError:
        return False
    return True


@pytest.mark.skipif(
    not (_mmb_env() and os.environ.get("ANDREY_BENCH_MMB_PYTHON") and _bench_python_ready()),
    reason="the MMB-by-MMB environment or the harness environment is absent",
)
def test_the_driver_runs_the_authors_code_in_its_own_environment(tmp_path, monkeypatch):
    folder = tmp_path / "networks"
    folder.mkdir()
    shutil.copy(ASIA, folder / "asia.bif.gz")
    monkeypatch.setenv(networks.CACHE_ENV, str(folder))
    out = tmp_path / "run"
    flags = ["--out", str(out), "--networks", "asia", "--hidden", "1", "--rows", "800"]
    flags += ["--seeds", "1", "--targets", "1", "--solutions", xie.XieMMBbyMMB.name]
    assert mmb_by_mmb_cpu.main(flags) == 0
    [record] = _records(out)
    assert record["status"] == "ok" and record["n_tests"] > 0
    assert list(record["scores"]) == [str(record["target"])]


@needs_r()
@pytest.mark.skipif(not os.environ.get("ANDREY_BENCH_R_LIBS"), reason="no R library named")
def test_the_rfci_arm_counts_its_tests_and_returns_the_graph_rfci_returns():
    from andrey_bench.adapters.pcalg_rfci import PcalgRFCI
    from andrey_bench.adapters.pcalg_rfci_local import PcalgRFCICounted

    X = _collider()
    counted, plain = PcalgRFCICounted(), PcalgRFCI()
    assert not counted.local and counted.name == plain.name
    counted.setup()
    plain.setup()
    answer = counted.fit(X, counted.params())
    marks, tests = L.unpack_answer(answer)
    assert tests > 0 and marks.shape == (4, 4)
    np.testing.assert_array_equal(marks, plain.fit(X, plain.params()))
    assert counted.to_structure(answer).kind == "pag"
