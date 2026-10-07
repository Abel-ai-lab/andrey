"""Parallel ensemble generation determinism.

Refutations: a task regenerates from its (scm, seed); an ensemble is byte-identical across worker
counts and submission order and across the ``fork`` and ``spawn`` start methods; single-task
regeneration holds. All are deterministic (fixed seeds).
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

import andrey.data as data


def _ensemble(n=200):
    scms = [
        data.SCM(
            graph=data.graphs.erdos_renyi(12, 2.0),
            functional=data.functional.linear(),
            noise=data.noise.gaussian(),
        ),
        data.SCM(
            graph=data.graphs.scale_free(12, 2),
            functional=data.functional.additive_noise(),
            noise=data.noise.uniform(),
        ),
    ]
    return data.Ensemble(scms=scms, seeds=range(4), n=n)


def _hashes(datasets):
    return [ds.data.tobytes() for ds in datasets]


def test_task_regenerates_from_scm_and_seed():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(10, 2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    a = scm.sample(n=200, seed=3)
    b = data.generate(data.Ensemble(scms=[scm], seeds=[3], n=200))[0]
    assert np.array_equal(a.data, b.data)
    assert a.graph == b.graph


def test_serial_equals_parallel():
    ens = _ensemble()
    serial = _hashes(data.generate(ens, workers=1))
    parallel = _hashes(data.generate(ens, workers=4))
    assert serial == parallel


def test_order_independence():
    """Shuffling the task order does not change any task's result (pure per-task generation)."""
    ens = _ensemble()
    forward = {
        (_scm_id(scm), s): ds.data.tobytes()
        for (scm, s), ds in zip(ens.tasks, data.generate(ens, workers=1))
    }
    rev = data.Ensemble(scms=list(ens.scms)[::-1], seeds=list(ens.seeds)[::-1], n=ens.n)
    for (scm, s), ds in zip(rev.tasks, data.generate(rev, workers=2)):
        assert forward[(_scm_id(scm), s)] == ds.data.tobytes()


def _scm_id(scm):
    return (scm.graph.name, scm.functional.name, scm.noise.name)


def test_result_count_and_order():
    ens = _ensemble()
    out = data.generate(ens, workers=1)
    assert len(out) == len(ens.tasks) == 2 * 4
    assert all(isinstance(ds, data.CausalDataset) for ds in out)
    for (scm, seed), ds in zip(ens.tasks, out):  # the k-th result is the k-th task's dataset
        own = scm.sample(n=ens.n, seed=seed)
        assert np.array_equal(ds.data, own.data)
        assert ds.graph == own.graph


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_fork_and_spawn_identical(method):
    """Both start methods give identical output (no reliance on inherited RNG state)."""
    ens = _ensemble(n=100)
    baseline = _hashes(data.generate(ens, workers=1))
    ctx = mp.get_context(method)
    from concurrent.futures import ProcessPoolExecutor

    from andrey.data.generate import _run

    tasks = [(scm, ens.n, s, np.float64) for scm, s in ens.tasks]
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        got = [ds.data.tobytes() for ds in pool.map(_run, tasks)]
    assert got == baseline
