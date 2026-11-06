"""Parallel ensemble generation determinism.

Refutations: a task regenerates from its (scm, seed); an ensemble is byte-identical across worker
counts and submission order and across the ``fork`` and ``spawn`` start methods; single-task
regeneration holds. All are deterministic (fixed seeds). A script without a ``__main__`` guard runs
under ``fork``.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pytest

import andrey
import andrey.data as data

pytestmark = pytest.mark.filterwarnings("ignore::andrey.DeprecatedArgumentWarning")


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


@pytest.mark.parametrize("scale", [None, "raw", "unitless"])
def test_task_regenerates_from_scm_and_seed(scale):
    """A task is the sample of its scm and seed, at the default scale or the one given."""
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(10, 2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    kwargs = {} if scale is None else {"scale": scale}
    a = scm.sample(n=200, seed=3, **kwargs)
    b = data.generate(data.Ensemble(scms=[scm], seeds=[3], n=200), **kwargs)[0]
    assert np.array_equal(a.data, b.data)
    assert a.graph == b.graph
    assert b.provenance["config"]["scale"] == (scale or "standardize")


@pytest.mark.filterwarnings("default::andrey.DeprecatedArgumentWarning")
def test_generate_and_ensemble_warn_that_they_stop_working_in_0_16():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(6, 2.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
    )
    with pytest.warns(andrey.DeprecatedArgumentWarning, match=r"Ensemble .* 0\.16\.0"):
        ens = data.Ensemble(scms=[scm], seeds=[0], n=50)
    with pytest.warns(andrey.DeprecatedArgumentWarning, match=r"generate .* 0\.16\.0"):
        (ds,) = data.generate(ens)
    assert np.array_equal(ds.data, scm.sample(n=50, seed=0).data)


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

    tasks = [(scm, ens.n, s, np.float64, "standardize") for scm, s in ens.tasks]
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as pool:
        got = [ds.data.tobytes() for ds in pool.map(_run, tasks)]
    assert got == baseline


# --- the start method of the worker pool ----------------------------------------------------------

import os  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

_SCRIPT_HEAD = """\
import andrey.data as data

scm = data.SCM(
    graph=data.graphs.erdos_renyi(8, 2.0),
    functional=data.functional.linear(),
    noise=data.noise.gaussian(),
)
ens = data.Ensemble(scms=[scm], seeds=range(4), n=100)
"""


def _run_script(tmp_path, body):
    script = tmp_path / "gen.py"  # a real file: forkserver and spawn re-import ``__main__`` by path
    script.write_text(_SCRIPT_HEAD + body)
    src_dir = str(pathlib.Path(data.__file__).resolve().parents[2])
    pythonpath = os.pathsep.join(filter(None, [src_dir, os.environ.get("PYTHONPATH")]))
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": pythonpath},
        cwd=tmp_path,
        timeout=300,
    )


@pytest.mark.parametrize("guarded", [False, True])
def test_script_without_main_guard(tmp_path, guarded):
    """A script runs ``generate`` on workers unguarded under ``fork``, and guarded under any method.

    ``forkserver`` and ``spawn`` re-import ``__main__`` in each worker, so a script run under them
    needs the guard.
    """
    method = mp.get_context().get_start_method()
    if not guarded and method != "fork":
        pytest.skip(f"{method!r} re-imports __main__, so the script needs the guard")
    call = "print(len(data.generate(ens, workers=2)))"
    body = f'if __name__ == "__main__":\n    {call}\n' if guarded else f"{call}\n"
    proc = _run_script(tmp_path, body)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["4"]
