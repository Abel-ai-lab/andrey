"""Each Andrey warning has its category, warns once per message, and filters by category or base."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import warnings

import numpy as np
import pytest

import andrey
from andrey.cli import main
from andrey.core import backend, ci, warning_policy

pc_module = importlib.import_module("andrey.constraint.pc")
calm_module = importlib.import_module("andrey.search.calm")


class _AlwaysDependent:
    """A CI test that keeps every edge, so PC reaches conditioning size two on any input."""

    def __init__(self, data):
        pass

    def __call__(self, x, y, condition_set=()):
        return 0.0


def _backend_fallback(monkeypatch):
    monkeypatch.setattr(backend, "_is_available", lambda name: False)
    return lambda: backend.resolve_bitset(backend="numba")


def _pc_expensive_pass(monkeypatch):
    monkeypatch.setitem(ci._CI_REGISTRY, "always-dependent", _AlwaysDependent)
    monkeypatch.setattr(pc_module, "_EXPENSIVE_PASS_TESTS", 90)  # size 2 on six nodes is 90 tests
    data = np.zeros((20, 6))
    return lambda: andrey.pc(data, indep_test="always-dependent")


def _calm_on_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return calm_module._require_torch


def _default_pool_cutoff(monkeypatch):
    from andrey.search import _parallel_ges

    monkeypatch.delenv("ANDREY_GES_PARALLEL_MIN_WORK", raising=False)
    monkeypatch.setattr(_parallel_ges, "_DEFAULT_MIN_WORK", 300)  # six variables reach a tenth
    data = np.random.default_rng(0).standard_normal((60, 6))

    def fit():
        with andrey.config(num_workers=2):
            andrey.ges(data)

    return fit


def _experimental(monkeypatch):
    data = np.random.default_rng(0).standard_normal((50, 3))
    return lambda: andrey.hc(data)


TRIGGERS = {
    "backend fallback": (_backend_fallback, andrey.BackendFallbackWarning),
    "PC expensive pass": (_pc_expensive_pass, andrey.PerformanceWarning),
    "CALM on CPU": (_calm_on_cpu, andrey.PerformanceWarning),
    "default pool cutoff": (_default_pool_cutoff, andrey.PerformanceWarning),
    "experimental method": (_experimental, andrey.ExperimentalWarning),
}


def _andrey_warnings(caught):
    return [w for w in caught if issubclass(w.category, andrey.AndreyWarning)]


@pytest.mark.parametrize("name", TRIGGERS)
def test_warns_once_with_its_category_at_the_callers_line(name, monkeypatch):
    make, category = TRIGGERS[name]
    call = make(monkeypatch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        call()
        call()
    [warning] = _andrey_warnings(caught)
    assert warning.category is category
    assert warning.filename == __file__


@pytest.mark.parametrize("name", TRIGGERS)
def test_an_error_filter_on_the_category_fails_every_call(name, monkeypatch):
    make, category = TRIGGERS[name]
    call = make(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error", category)
        for _ in range(2):  # a call the filter stopped does not use up the warning
            with pytest.raises(category):
                call()


def test_ignoring_the_base_class_silences_every_warning(monkeypatch):
    has_torch = importlib.util.find_spec("torch") is not None
    calls = [
        make(monkeypatch) for make, _ in TRIGGERS.values() if has_torch or make is not _calm_on_cpu
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.simplefilter("ignore", andrey.AndreyWarning)
        for call in calls:
            call()
    assert not _andrey_warnings(caught)
    assert len(calls) >= 3


def test_every_category_is_a_user_warning_under_the_base():
    for category in (
        andrey.ExperimentalWarning,
        andrey.PerformanceWarning,
        andrey.BackendFallbackWarning,
    ):
        assert issubclass(category, andrey.AndreyWarning)
    assert issubclass(andrey.AndreyWarning, UserWarning)


def test_pytest_w_flag_names_a_category(tmp_path):
    """``pytest -W error::andrey.<Category>`` fails a test that raises that category.

    Python's own ``-W`` flag cannot name a category outside the standard library, so pytest's flag
    (or ``warnings.filterwarnings`` in code) is how a test suite escalates one category.
    """
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_probe.py").write_text(
        "import warnings, andrey\n"
        "def test_performance():\n"
        "    warnings.warn('slow', andrey.PerformanceWarning)\n"
        "def test_experimental():\n"
        "    warnings.warn('new', andrey.ExperimentalWarning)\n"
    )
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-W",
            "error::andrey.PerformanceWarning",
            str(tmp_path / "test_probe.py"),
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert "1 failed, 1 passed" in run.stdout, run.stdout + run.stderr
    assert "andrey.core.warning_policy.PerformanceWarning: slow" in run.stdout


def test_the_command_line_reports_a_performance_warning_in_its_json(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(pc_module, "_EXPENSIVE_PASS_TESTS", 1)
    rng = np.random.default_rng(0)
    factor = rng.standard_normal((300, 1))
    data = factor + 0.3 * rng.standard_normal((300, 6))  # every pair stays dependent
    path = tmp_path / "data.csv"
    np.savetxt(path, data, delimiter=",")
    assert main(["run", "pc", "--data", str(path)]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    categories = {w["category"] for w in json.loads(out)["run"]["warnings"]}
    assert categories == {"PerformanceWarning"}


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_a_forked_child_can_warn_while_another_thread_held_the_lock():
    """A child forked while a thread is inside ``warn_once`` gets a free lock, not a held one."""
    held, release = threading.Event(), threading.Event()

    def hold():
        with warning_policy._lock:
            held.set()
            release.wait()

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # forking a threaded process
            pid = os.fork()
        if pid == 0:
            os._exit(0 if warning_policy._lock.acquire(blocking=False) else 1)
        _, status = os.waitpid(pid, 0)
    finally:
        release.set()
        holder.join()
    assert os.waitstatus_to_exitcode(status) == 0
