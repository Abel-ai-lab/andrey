"""Results whose correct value is known independently of any package.

The benchmark cannot be trusted until it can check itself. Everything here has an answer that is
arithmetic or definitional, so a failure means the *harness* is broken rather than a solution being
slow — which is what makes this suite the acceptance gate for every later refactor step.

The whole table is here except ``DatasetKey`` injectivity, which :mod:`test_dataset_identity`
covers. The memory cap has a module of its own (:mod:`test_memory_cap`), and the plain wall-cap kill
lives in ``test_runner.test_wall_cap_timeouts_and_kills_child``; this module covers the part that
test does not say — that the cap covers the warm-up too.

Every check runs on tiny datasets (``d <= 10``) so the suite stays cheap enough to run before any
benchmark, which is the only way an acceptance gate gets used.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

# Andrey endpoint-mark convention: the mark AT node i on edge i-j lives at marks[i, j].
_ARROW = 2


# --------------------------------------------------------------------------------------------------
# Shared fixtures. The committed micro-grid, run once for the whole module.
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_keys():
    from tiny_keys import TINY_KEYS

    return list(TINY_KEYS)


@pytest.fixture(scope="module")
def micro_run(tmp_path_factory, tiny_keys):
    """The two analytic solutions over the committed tiny datasets, through the real pipeline.

    Materialize -> subprocess fit -> ``to_structure`` in the parent -> score. Both solutions have
    known answers, so one pass covers the oracle record and the empty-baseline record at once.
    """
    from andrey_bench.adapters.baseline_empty import BaselineEmpty
    from andrey_bench.integration import run_batch
    from andrey_bench.oracle import OracleGES

    out = tmp_path_factory.mktemp("micro")
    result = run_batch(
        out,
        [OracleGES(), BaselineEmpty()],
        tiny_keys,
        machine_id="tiny-cpu",
        cap_wall_s=120.0,
        repeats=1,
        warmup=0,
    )
    assert not result.errors, result.errors
    return out, result


def _truth_of(out_dir, key):
    """``(data, GraphStructure)`` read back from the materialized file — the DGP, not an estimate."""
    from andrey import GraphStructure
    from andrey_bench import datasets

    data, marks = datasets.load(out_dir, key)
    return data, GraphStructure.from_numpy(marks, kind="dag")


def _edge_count(truth) -> int:
    """The truth's edge count ``m``, counted from the materialized marks.

    Never recomputed from ``d * density / 2``: Andrey's ER draw is fixed-``m``
    (``round(density * d / 2)``), but that ``round`` and the ``min(..., n_pairs)`` clamp make the
    formula disagree with the graph it claims to describe. Read the answer; do not re-derive it.
    """
    marks = np.asarray(truth.to_numpy())
    present = (marks != 0) | (marks.T != 0)
    return int(np.triu(present, k=1).sum())


def _records_named(result, name):
    records = [r for r in result.records if r.name == name]
    assert records, f"no records for {name!r}"
    return records


# --------------------------------------------------------------------------------------------------
# Oracle — truth in, SHD 0 out.
# --------------------------------------------------------------------------------------------------


def test_the_oracle_handed_the_truth_scores_zero(micro_run, tiny_keys):
    """SHD = 0 on every dataset. A failure is marshalling, alignment, kind-tagging or scoring plumbing.

    On its own this proves less than it looks: the oracle projects with ``to_cpdag`` and the scorer
    projects the truth with the same ``to_cpdag``, and SHD(f(T), f(T)) = 0 for *any* deterministic
    ``f``. An identity projection would pass here and score every real solution wrong; the
    ``to_cpdag`` goldens in Andrey's own suite (``tests/metrics/test_graph.py``) refuse it.

    Arrowhead F1 witnesses orientation only on a truth whose CPDAG has an arrowhead, so at least
    one must: an estimate transposed on its way back would then fail.
    """
    import andrey.metrics as M

    out, result = micro_run
    records = _records_named(result, "oracle.ges")
    assert len(records) == len(tiny_keys)
    for record in records:
        assert record.status == "ok", record
        assert record.metrics["shd"] == 0, record
        assert record.metrics["skeleton_f1"] == 1.0, record
        assert record.metrics["arrowhead_f1"] == 1.0, record
        # The runner injects the truth into the fit's params; the record must not carry it.
        assert record.params == "{}", record.params
    projected = [M.to_cpdag(_truth_of(out, key)[1]).to_numpy() for key in tiny_keys]
    assert any((np.asarray(marks) == _ARROW).any() for marks in projected)


# --------------------------------------------------------------------------------------------------
# Empty baseline — SHD is the truth's edge count, exactly. Analytic; no package involved.
# --------------------------------------------------------------------------------------------------


def test_the_empty_graph_scores_exactly_the_truths_edge_count(micro_run, tiny_keys):
    """SHD charges one edit per differing pair and ``dag2cpdag`` preserves the skeleton, so the
    empty graph's distance to the truth is ``m`` whether the truth is scored as a DAG or a CPDAG.

    The F1s have a known answer here too, and this is the only place they have one: no edges means
    no predicted arrowheads, so recall is 0 wherever the truth CPDAG has an arrowhead to recall. The
    oracle's ``== 1.0`` on truth-in is satisfied by a metric hardwired to 1.0; a floor of 0.0 is not.
    Where the projected truth is fully undirected there is nothing to recall, and that 0/0 scores
    1.0 — Andrey's convention, pinned here rather than assumed.

    Joining the records back to their datasets on ``dataset_id`` is also the claim that ``dataset_id`` **is**
    the dataset digest: a record that could not find its truth here would be identified by something
    other than the data it was measured on.
    """
    import andrey.metrics as M

    out, result = micro_run
    by_task = {key.digest: key for key in tiny_keys}
    records = _records_named(result, "baseline.empty")
    assert len(records) == len(tiny_keys)

    for record in records:
        _data, truth = _truth_of(out, by_task[record.dataset_id])
        m = _edge_count(truth)
        assert m > 0, "a truth with no edges makes this check vacuous"
        assert record.status == "ok", record
        assert record.metrics["shd"] == m, (record.metrics, m, record.d)
        assert record.metrics["skeleton_f1"] == 0.0, record.metrics
        # Counted from the projected truth per dataset, never from a list of which ones have
        # orientation — that would rot the moment the datasets are reseeded.
        arrows = int((np.asarray(M.to_cpdag(truth).to_numpy()) == _ARROW).sum())
        assert record.metrics["arrowhead_f1"] == (1.0 if arrows == 0 else 0.0), (
            record.metrics,
            arrows,
        )


# --------------------------------------------------------------------------------------------------
# Repeat determinism — one output hash across independent repeat processes.
# --------------------------------------------------------------------------------------------------


def test_a_solution_returns_one_answer_across_repeats(tmp_path, tiny_keys):
    """Three isolated subprocess repeats of a real solution, one ``output_hash``.

    A second hash means process isolation or seeding is broken.
    """
    from andrey_bench.adapters.andrey_ges import AndreyGES
    from andrey_bench.integration import run_batch

    adapter = AndreyGES(name="andrey.ges.numpy.serial", backend="numpy", mode="serial")
    result = run_batch(
        tmp_path,
        [adapter],
        tiny_keys[-1:],
        machine_id="tiny-cpu",
        cap_wall_s=120.0,
        repeats=3,
        warmup=0,
    )
    assert not result.errors, result.errors
    assert len(result.records) == 3
    assert all(r.status == "ok" for r in result.records), result.records

    assert len({r.output_hash for r in result.records}) == 1, [
        r.output_hash for r in result.records
    ]
    # The record publishes the params this solution ran with, not a harness-wide policy.
    assert {r.params for r in result.records} == {
        json.dumps(dict(adapter.params()), sort_keys=True)
    }


# --------------------------------------------------------------------------------------------------
# Serial vs parallel — same solution, same answer, and the pool provably ran.
# --------------------------------------------------------------------------------------------------


def test_the_parallel_path_returns_the_serial_answer(tmp_path, tiny_keys):
    """Identical ``output_hash``, with BLAS threads pinned equal and proof the pool engaged.

    Both halves are load-bearing. Without the pin this flakes on BLAS-thread non-determinism, and
    the harness hands the two modes different splits by design (``runner.split_cores``), so the
    comparison is made through ``run_task`` at ``threads=1`` on both sides rather than through the
    benchmark path. Without engagement proof it passes vacuously: at ``d = 10`` the shipped
    ``ANDREY_GES_PARALLEL_MIN_WORK`` gate keeps the pool shut and the check degenerates to
    serial-vs-serial. The gate is forced to 0 and ``child_cpu_s`` is what says a pool really burned
    CPU — the serial side is asserted near zero so the number is a discriminator, not decoration.
    """
    from andrey_bench import datasets
    from andrey_bench.adapters.andrey_ges import AndreyGES
    from andrey_bench.runner import run_task

    key = tiny_keys[-1]
    datasets.materialize([key], tmp_path)
    path = datasets.data_path(tmp_path, key)
    # Applied to BOTH sides, so mode and worker count are the only things that differ.
    force_pool = {"ANDREY_GES_PARALLEL_MIN_WORK": "0"}

    def run(adapter, num_workers):
        return run_task(
            adapter,
            path,
            {},
            cap_wall_s=120.0,
            repeats=1,
            warmup=0,
            threads=1,  # the pin: equal BLAS threads on both sides
            num_workers=num_workers,
            extra_env=force_pool,
        )[0]

    serial = run(AndreyGES(name="andrey.ges.numpy.serial", backend="numpy", mode="serial"), 1)
    parallel = run(AndreyGES(name="andrey.ges.numpy.parallel", backend="numpy", mode="parallel"), 2)

    assert serial["status"] == "ok" and parallel["status"] == "ok", (serial, parallel)
    assert parallel["child_cpu_s"] > 0.05, parallel  # the pool ran
    assert serial["child_cpu_s"] < 0.05, serial  # ... and the serial side spawned nothing
    assert parallel["threads"] == serial["threads"] == 1
    assert parallel["output_hash"] == serial["output_hash"]


# --------------------------------------------------------------------------------------------------
# Sortability — two checks on the DGP readback path, which never sees a solution's estimate.
# --------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sortability_pairs(tmp_path_factory):
    """One seed, two ``standardize`` coordinates, materialized side by side."""
    from andrey_bench import datasets, scoring

    def key(standardize):
        return datasets.DatasetKey(
            topology="er",
            functional="linear",
            noise="gaussian",
            density=2.0,
            d=10,
            n=200,
            seed=0,
            standardize=standardize,
        )

    out = tmp_path_factory.mktemp("sortability")
    keys = {name: key(name) for name in ("standardized", "raw")}
    datasets.materialize(list(keys.values()), out)

    measured = {}
    for name, k in keys.items():
        data, truth = _truth_of(out, k)
        assert _edge_count(truth) > 0
        measured[name] = scoring.sortability(data, truth)
    return measured


def test_a_standardized_dataset_has_varsortability_of_exactly_one_half(sortability_pairs):
    """``SCM.sample`` z-scores post-hoc, which drives varsortability to *exactly* 0.5 — every
    variance ratio is a tie, and ties score 0.5 by definition. So gate on equality.

    Read a failure as *the ``standardize`` coordinate never reached the data*, not as a DGP leak.
    """
    assert sortability_pairs["standardized"]["varsortability"] == 0.5


def test_a_raw_dataset_reads_the_truths_orientation_the_right_way_round(sortability_pairs):
    """The only check here that observes the truth's **orientation** at all.

    Raw linear-Gaussian data accumulates variance along the causal order, so varsortability is high.
    Reversing every edge maps ``v -> 1 - v``, so a raw dataset reading ~0.1 says the truth adjacency was
    transposed on the way out of storage — an error the standardized check is invariant to, since a
    tie survives a transpose unchanged.
    """
    assert sortability_pairs["raw"]["varsortability"] >= 0.9


# --------------------------------------------------------------------------------------------------
# The wall cap covers imports + warm-up + fit, not the fit alone.
# --------------------------------------------------------------------------------------------------


class StubSlowFirstCallAdapter:
    """A solution whose *first* fit is slow and whose every later fit is instant.

    The Tetrad shape: JVM boot, class loading and a first JIT all land in the warm-up call, and the
    timed fit that follows is cheap. One module-level flag per process is enough, because the runner
    gives every repeat a fresh interpreter — so call one is always the warm-up when ``warmup=1``.
    """

    name = "stub.slow-warmup"
    algorithm = "ges"
    package = "stub"
    package_version = "0"
    backend = "numpy"
    mode = "serial"
    output_type = "dag"

    def to_structure(self, native):  # pragma: no cover - runner never calls this
        raise NotImplementedError

    def fit(self, data, params):
        global _FIRST_CALL_DONE
        if not _FIRST_CALL_DONE:
            _FIRST_CALL_DONE = True
            time.sleep(1.2)
        return np.zeros((4, 4), dtype=np.int8)


_FIRST_CALL_DONE = False


def test_the_wall_cap_covers_the_warmup_not_only_the_timed_fit():
    """A cap between the warm-up and the fit kills the run: ``timeout``, never ``ok``.

    Deliberate, not incidental. ``cap_wall_s`` is a cap on the whole child — imports, warm-up and
    fit — while ``wall_s`` measures the fit alone, so a solution with a long warm-up and a fast fit
    times out on work that never appears in any recorded duration.

    The generous-cap control is what makes that the claim: it shows the same stub's *timed* fit is
    effectively instant, so the kill can only have come from the warm-up. (The plain
    long-fit-under-a-short-cap case is already covered by
    ``test_runner.test_wall_cap_timeouts_and_kills_child``.)
    """
    from andrey_bench.runner import run_task

    data = np.zeros((16, 4), dtype=np.float64)

    generous = run_task(StubSlowFirstCallAdapter(), data, {}, cap_wall_s=30.0, repeats=1, warmup=1)[
        0
    ]
    assert generous["status"] == "ok", generous
    assert generous["warmup_s"] >= 1.1, generous  # the warm-up is where the time goes ...
    assert generous["wall_s"] < 0.2, generous  # ... and the timed fit is essentially free

    capped = run_task(StubSlowFirstCallAdapter(), data, {}, cap_wall_s=0.6, repeats=1, warmup=1)[0]
    assert capped["status"] == "timeout", capped
    # The upper bound is the half that discriminates: the kill landed *inside* the 1.2 s warm-up,
    # so the cap is a deadline on the whole child rather than a check made between its phases.
    assert 0.6 <= capped["elapsed_at_kill_s"] < 1.2, capped
    assert "adj" not in capped  # censored on a fit that would have finished instantly


class StubSetupFastWarmupHangingFitAdapter:
    """Setup takes about 1 s and the first fit (the warm-up) 0.2 s; every later fit hangs.

    A uniform per-call cost would race the child's imports, which come out of the same cap.
    """

    name, algorithm, package, package_version = "stub.slowfit", "ges", "stub", "0"
    backend, mode, output_type = "numpy", "serial", "dag"

    def __init__(self):
        self.calls = 0

    def to_structure(self, native):  # pragma: no cover - the runner never calls this
        return native

    def params(self):
        return {}

    def setup(self):
        time.sleep(1.0)

    def fit(self, data, params):
        self.calls += 1
        time.sleep(0.2 if self.calls == 1 else 600.0)
        return np.zeros((4, 4), dtype=np.int8)


def test_a_censored_record_reports_the_setup_and_warmup_it_finished():
    """A killed fit still reports the stages its child finished before the kill.

    A warm-up is a complete untimed fit, so its duration bounds the fit by measurement rather than
    by the cap.
    """
    from andrey_bench.runner import run_task

    data = np.zeros((16, 4), dtype=np.float64)
    # Allow time for child startup and setup. A shorter cap could kill setup on a busy node,
    # measuring interpreter startup instead of the fit timeout.
    capped = run_task(
        StubSetupFastWarmupHangingFitAdapter(), data, {}, cap_wall_s=30.0, repeats=1, warmup=1
    )[0]

    assert capped["status"] == "timeout", capped
    assert "wall_s" not in capped, capped  # the fit itself never returned ...
    assert capped["setup_s"] is not None and capped["setup_s"] >= 0.9, capped
    # ... but setup and one full warm-up fit were still measured.
    assert capped["setup_s"] + 0.2 <= capped["warmup_s"] < 30.0, capped
