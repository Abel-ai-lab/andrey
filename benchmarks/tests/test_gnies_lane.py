"""The GnIES lane: its data, its scoring, its reading of the reference, and the score both
packages compute.

The tests that run gnies 0.3.3 skip unless ``ANDREY_BENCH_GNIES_PYTHON`` names the interpreter
``benchmarks/start_gnies.sh`` builds. With it, Andrey's total at the standard BIC equals the
reference's ``GnIESScore`` run to convergence, converted to deviance units, on 20 draws of data,
DAG, and targets; the reference's 0/1 matrix reads as the true class on a fixed chain; the lane's
class enumeration lists the reference's members on 20 random classes; a call past its cap is a
timeout; and one small run scores every arm.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time

import gnies_cpu
import numpy as np
import pytest
from gnies_cases import reference

from andrey.core import ARROW, TAIL
from andrey.search.gnies import NoiseInterventionScore, singleton_cut
from andrey_bench.adapters.gnies_gnies import PYTHON_ENV_VAR, marks, request

needs_reference = pytest.mark.skipif(
    not os.environ.get(PYTHON_ENV_VAR), reason=f"{PYTHON_ENV_VAR} is not set"
)


def _draw(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Six variables, an observational and two other environments of 60-150 rows, each shifted in
    mean; every environment but the first changes the targets' noise variances."""
    rng = np.random.default_rng([6, seed])
    d = 6
    upper = np.triu(rng.random((d, d)) < 0.4, 1).astype(int)
    order = rng.permutation(d)
    dag = upper[np.ix_(order, order)]
    weights = dag * rng.uniform(0.5, 1.0, (d, d)) * rng.choice([-1, 1], (d, d))
    sizes = rng.integers(60, 151, 3)
    targets = sorted(rng.choice(d, size=int(rng.integers(0, 4)), replace=False).tolist())
    blocks, env = [], []
    for k, n in enumerate(sizes):
        sd = np.sqrt(rng.uniform(1, 2, d))
        if k > 0:
            sd[targets] = np.sqrt(rng.uniform(5, 10, len(targets)))
        noise = rng.standard_normal((n, d)) * sd
        blocks.append(noise @ np.linalg.inv(np.eye(d) - weights) + rng.normal(0, 3, d))
        env += [k] * int(n)
    return np.vstack(blocks), np.array(env), dag, targets


def test_marks_read_lone_entries_as_arrows_and_pairs_as_lines():
    icpdag = [[0, 1, 0], [0, 0, 1], [0, 1, 0]]
    m = marks(icpdag)
    assert (m[0, 1], m[1, 0]) == (TAIL, ARROW)
    assert (m[1, 2], m[2, 1]) == (TAIL, TAIL)
    assert m[0, 2] == m[2, 0] == 0


@needs_reference
@pytest.mark.parametrize("seed", range(20))
def test_total_equals_the_converged_reference_score(seed):
    X, env, dag, targets = _draw(seed)
    view = NoiseInterventionScore(X, env, ebic_gamma=0.0).view(targets)
    total = math.fsum(view.score(j, np.flatnonzero(dag[:, j])) for j in range(X.shape[1]))
    answer = reference(X, env, "score", A=dag, I=np.array(targets, dtype=int), iters=[1000])
    assert total == pytest.approx(answer["deviance"][0], abs=1e-6)


@needs_reference
def test_the_reference_matrix_reads_as_the_true_class():
    rng = np.random.default_rng(0)
    env = np.repeat([0, 1], 2000)
    x = rng.standard_normal(4000)
    y = 0.8 * x + rng.standard_normal(4000) * np.where(env == 1, 3.0, 1.0)
    z = 0.7 * y + rng.standard_normal(4000)
    X = np.column_stack([x, y, z])
    answer = reference(X, env, "known", I=np.array([1]))
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[0, 1] = expected[1, 2] = TAIL
    expected[1, 0] = expected[2, 1] = ARROW
    assert np.array_equal(marks(answer["icpdag"]), expected)


def test_the_grid_has_the_designs_in_claim_cells_first():
    in_claim = [c.name for c in gnies_cpu.CELLS if c.claim == "in"]
    assert len(in_claim) == 21
    assert [c.name for c in gnies_cpu.CELLS[:21]] == in_claim
    assert len(gnies_cpu.BY_NAME) == len(gnies_cpu.CELLS)


def test_the_paper_specs_keep_the_stock_draws():
    from andrey.data import SCM, graphs
    from andrey.data.functional import linear
    from andrey.data.noise import gaussian

    graph = graphs.erdos_renyi(10, avg_degree=2.7)
    stock = SCM(graph=graph, functional=linear(weight_range=(0.5, 1.0)), noise=gaussian())
    paper = SCM(
        graph=graph,
        functional=gnies_cpu.PaperLinear(weight_range=(0.5, 1.0)),
        noise=gnies_cpu.PaperGaussian(),
    )
    a, b = stock.sample(n=10, seed=3).params, paper.sample(n=10, seed=3).params
    assert np.array_equal(np.abs(a.weights), b.weights)
    assert np.all((b.noise_scales**2 >= 1.0) & (b.noise_scales**2 <= 2.0))


def test_a_draw_is_the_simulators_draw():
    from andrey.data import SCM, graphs
    from andrey.data.interventions import soft

    cell, seed = gnies_cpu.BY_NAME["A100"], 4
    data = gnies_cpu.draw(cell, seed)
    scm = SCM(
        graph=graphs.erdos_renyi(10, avg_degree=2.7),
        functional=gnies_cpu.PaperLinear(weight_range=(0.5, 1.0)),
        noise=gnies_cpu.PaperGaussian(),
    )
    sigma = scm.sample(n=10, seed=seed, scale="raw").params.noise_scales
    rng = np.random.default_rng([10, gnies_cpu.CELLS.index(cell), seed])
    targets = rng.choice(10, size=(3, 1), replace=False)
    envs = [{}] + [
        {int(j): soft(noise_scale=float(np.sqrt(rng.uniform(5, 10)) / sigma[j])) for j in row}
        for row in targets
    ]
    ds = scm.sample(n=[100] * 4, seed=seed, envs=envs, scale="raw")
    X = (ds.data - ds.data.mean(0)) / ds.data.std(0)
    assert np.array_equal(data.X, X)
    assert np.array_equal(data.env, ds.env)
    assert data.targets == [[]] + [[int(j)] for j in targets[:, 0]]
    assert gnies_cpu.draw(cell, seed).sha256 == data.sha256


def _random_class(seed: int, d: int = 6) -> tuple[np.ndarray, list[int], np.ndarray]:
    """A random DAG as endpoint marks, random targets, and the DAG as a 0/1 matrix."""
    rng = np.random.default_rng([d, seed])
    order = rng.permutation(d)
    upper = np.triu(rng.random((d, d)) < 0.5, 1)
    adjacency = upper[np.ix_(order, order)].astype(np.int8)
    dag = np.zeros((d, d), dtype=np.int8)
    dag[adjacency == 1] = TAIL
    dag[adjacency.T == 1] = ARROW
    targets = sorted(int(j) for j in rng.choice(d, size=int(rng.integers(0, 3)), replace=False))
    return dag, targets, adjacency


@pytest.mark.parametrize("seed", range(10))
def test_every_member_of_a_class_has_that_class(seed):
    dag, targets, adjacency = _random_class(seed)
    cut = singleton_cut(targets, len(dag))
    essential = gnies_cpu.essential_graph(dag, cut)
    members = gnies_cpu.class_members(essential)
    assert any(np.array_equal(m, adjacency) for m in members)
    for member in members:
        as_marks = np.where(member == 1, TAIL, 0) + np.where(member.T == 1, ARROW, 0)
        assert np.array_equal(gnies_cpu.essential_graph(as_marks.astype(np.int8), cut), essential)


def test_scores_of_the_truth_against_itself():
    data = gnies_cpu.draw(gnies_cpu.BY_NAME["A100"], 0)
    truth = gnies_cpu.Truth(data)
    out = gnies_cpu.scores(truth.marks, truth.targets, truth)
    assert out["shd"] == out["target_error"] == out["skeleton_shd"] == 0
    assert out["class_equal"] and (out["tdp"], out["fdp"]) == (1.0, 0.0)
    assert out["gap_common"] == pytest.approx(0.0, abs=1e-6)
    empty = gnies_cpu.scores(np.zeros_like(truth.marks), [], truth)
    assert empty["shd"] == np.count_nonzero(np.triu(truth.marks != 0, 1))
    assert empty["target_error"] == len(truth.targets)
    assert truth.model is not None and not truth.model["flagged"]


def test_a_fit_in_this_process_stops_at_its_cap():
    with pytest.raises(gnies_cpu.FitTimeout):
        with gnies_cpu.capped(0.05):
            time.sleep(1.0)


_MEMBERS = """
import json, sys
import numpy as np
from gnies import utils
request = np.load(sys.argv[1])
A, I = request["A"], set(int(j) for j in request["I"])
members = utils.imec(A, I) if I else utils.mec(A)
print(json.dumps(sorted(np.asarray(m, dtype=int).ravel().tolist() for m in members)))
"""


@needs_reference
@pytest.mark.parametrize("seed", range(20))
def test_class_members_are_the_reference_enumeration(seed, tmp_path):
    dag, targets, adjacency = _random_class(seed)
    essential = gnies_cpu.essential_graph(dag, singleton_cut(targets, len(dag)))
    mine = sorted(m.ravel().tolist() for m in gnies_cpu.class_members(essential))
    path = tmp_path / "request.npz"
    np.savez(path, A=adjacency, I=np.array(targets, dtype=int))
    done = subprocess.run(
        [os.environ[PYTHON_ENV_VAR], "-I", "-c", _MEMBERS, str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert mine == json.loads(done.stdout)


@needs_reference
def test_a_reference_call_past_its_cap_is_a_timeout():
    data = gnies_cpu.draw(gnies_cpu.BY_NAME["A1000"], 0)
    answer = request(data.X, data.env, "fit", cap_s=0.5)
    assert answer["status"] == "timeout"


@needs_reference
def test_a_small_run_scores_every_arm(tmp_path):
    assert gnies_cpu.main(["--out", str(tmp_path), "--cells", "A10", "--seeds", "0"]) == 0
    rows = [json.loads(line) for line in (tmp_path / gnies_cpu.RECORDS).read_text().splitlines()]
    assert [r["arm"] for r in rows] == list(gnies_cpu.ARMS)
    for row in rows:
        assert row["status"] == "ok", row["error_text"]
        assert row["shd"] is not None and row["target_error"] is not None
    assert len({r["data_sha256"] for r in rows}) == 1
