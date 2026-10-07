"""PC traces replay to the returned graph, match the untraced run, and repeat exactly."""

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import andrey
from andrey import data
from andrey.constraint._skeleton import scan_conditioning_sets
from andrey.constraint.pc import _discover_skeleton, pc
from andrey.core.ci import make_indep_test

ALPHA = 0.05
CASES = [(d, density, seed) for d in (6, 12, 25) for density in (2.0, 4.0) for seed in range(4)]


def observations(d, density, seed):
    return data.sample_scm(d=d, n=10 * d, seed=seed, density=density, scale="standardize").data


def replay(trace, d):
    """Rebuild the CPDAG and separating sets from the trace, checking each record on the way."""
    steps = [r["step"] for r in trace]
    assert steps == sorted(steps, key=["test", "collider", "meek"].index)
    tests = [r for r in trace if r["step"] == "test"]
    assert [r["depth"] for r in tests] == sorted(r["depth"] for r in tests)
    adj = np.ones((d, d), dtype=np.int8) - np.eye(d, dtype=np.int8)
    sepsets = {}
    for depth in sorted({r["depth"] for r in tests}):
        snapshot = adj.copy()
        tried, marked = set(), set()
        for r in (r for r in tests if r["depth"] == depth):
            x, y, S = r["x"], r["y"], r["S"]
            assert x < y and snapshot[x, y] and len(S) == depth
            neighbours = [set(np.flatnonzero(snapshot[v]).tolist()) for v in (x, y)]
            assert set(S) <= neighbours[0] - {y} or set(S) <= neighbours[1] - {x}
            assert (x, y, S) not in tried
            tried.add((x, y, S))
            assert r["removed"] == (r["p"] > ALPHA)
            if r["removed"]:
                marked.add((x, y))
                sepsets.setdefault((x, y), set()).update(S)
        for x, y in marked:
            adj[x, y] = adj[y, x] = 0
    for r in trace:
        if r["step"] != "test":
            np.testing.assert_array_equal(r["before"], adj)
            adj = r["after"]
    return adj, {pair: tuple(sorted(s)) for pair, s in sepsets.items()}


def digest(trace):
    """Hash a trace exactly: p-values by their bits, matrices by their marks."""

    def plain(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, float):
            return value.hex()
        return value

    rows = [{key: plain(value) for key, value in record.items()} for record in trace]
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize(("d", "density", "seed"), CASES)
def test_trace_replays_to_the_graph_pc_returns(d, density, seed):
    X = observations(d, density, seed)
    trace = []
    traced = pc(X, alpha=ALPHA, _trace=trace)
    expected = andrey.pc(X, alpha=ALPHA).structure.to_numpy()
    np.testing.assert_array_equal(traced.to_numpy(), expected)
    marks, sepsets = replay(trace, d)
    np.testing.assert_array_equal(marks, expected)
    _, engine_sepsets = _discover_skeleton(X, ALPHA)
    assert sepsets == engine_sepsets


def test_cases_reach_every_skeleton_path_and_step():
    depths, steps = set(), set()
    for case in CASES:
        trace = []
        pc(observations(*case), alpha=ALPHA, _trace=trace)
        depths |= {r["depth"] for r in trace if r["step"] == "test"}
        steps |= {r["step"] for r in trace}
        steps |= {"oriented" for r in trace if r.get("oriented")}
    assert {0, 1, 2, 3} <= depths  # both closed-form passes and the batched scan
    assert steps == {"test", "collider", "meek", "oriented"}


def test_trace_is_identical_across_runs():
    X = observations(25, 4.0, 1)
    first, second = [], []
    pc(X, _trace=first)
    pc(X, _trace=second)
    assert digest(first) == digest(second)
    # A fresh interpreter with a different BLAS thread count records the same trace.
    script = (
        "import sys; sys.path.insert(0, sys.argv[1]); import test_pc_trace as t; "
        "from andrey.constraint.pc import pc; trace = []; "
        "pc(t.observations(25, 4.0, 1), _trace=trace); print(t.digest(trace))"
    )
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    fresh = subprocess.run(
        [sys.executable, "-c", script, os.path.dirname(__file__)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert fresh.stdout.strip() == digest(first)


def test_closed_form_passes_record_the_same_tests_as_the_scan():
    X = observations(25, 4.0, 2)
    fast, scanned = [], []
    fast_result = _discover_skeleton(X, ALPHA, _trace=fast)
    scanned_result = _discover_skeleton(X, ALPHA, prescreen=False, _trace=scanned)
    np.testing.assert_array_equal(fast_result[0], scanned_result[0])

    def keyed(trace):
        return {(r["depth"], r["x"], r["y"], r["S"]): (r["p"], r["removed"]) for r in trace}

    assert len(fast) == len(scanned)
    assert keyed(fast) == keyed(scanned)


def test_tracing_scans_serially_with_the_same_record():
    X = observations(12, 4.0, 0)
    serial, requested = [], []
    _discover_skeleton(X, ALPHA, _trace=serial)
    _discover_skeleton(X, ALPHA, workers=2, _trace=requested)
    assert digest(serial) == digest(requested)


def test_scalar_only_ci_tests_record_the_same_tests():
    test = make_indep_test("fisherz", observations(12, 4.0, 0))
    batched, scalar = [], []
    for ci, trace in ((test, batched), (lambda a, b, S: test(a, b, S), scalar)):
        scan_conditioning_sets(ci, 3, 0, list(range(4, 12)), 2, ALPHA, set(), trace=trace)
    assert len(batched) == 28
    assert scalar == batched
