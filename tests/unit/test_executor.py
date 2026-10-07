"""Executor axis: ``num_workers`` over a process pool. The serial and parallel paths agree."""

from __future__ import annotations

import gc
import io
import os
import warnings

import pytest

import andrey
from andrey.core import backend


def _square(x):  # top-level so it is picklable for the process pool
    return x * x


def test_num_workers_invalid_env_names_the_variable(monkeypatch):
    monkeypatch.setenv("ANDREY_NUM_WORKERS", "four")
    with pytest.raises(ValueError, match="ANDREY_NUM_WORKERS"):
        backend.worker_count()


def test_parallel_map_serial_matches_process():
    items = list(range(20))
    serial = backend.parallel_map(_square, items, num_workers=1)
    parallel = backend.parallel_map(_square, items, num_workers=2)
    assert serial == parallel == [x * x for x in items]  # same values, input order preserved


def test_worker_count_resolution(monkeypatch):
    monkeypatch.delenv("ANDREY_NUM_WORKERS", raising=False)
    assert backend.worker_count() == 1  # serial default
    assert backend.worker_count(-1) == backend.usable_cpus()
    with andrey.config(num_workers=3):
        assert backend.worker_count() == 3


def test_parallel_map_single_item_stays_serial():
    # A lone item never spawns a pool (no pickling / start-up cost for trivial work).
    assert backend.parallel_map(_square, [7], num_workers=4) == [49]


# ---- usable_cpus: the machine is not the budget ----------------------------------------------


def test_usable_cpus_applies_the_cgroup_quota(monkeypatch):
    """The quota has to reach the answer: a container's cap is invisible to both other limits."""
    monkeypatch.setattr(backend, "_cgroup_cpu_quota", lambda: 1)
    assert backend.usable_cpus() == 1


@pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="no affinity mask off Linux")
def test_usable_cpus_respects_the_affinity_mask():
    """SLURM and ``taskset`` restrict the mask; cpu_count keeps reporting the whole node."""
    assert backend.usable_cpus() <= len(os.sched_getaffinity(0))


def _fake_cgroup(monkeypatch, tmp_path, *, proc_self, files):
    """Stand up a fake cgroup tree: ``files`` maps a dir under tmp_path to its cpu.max contents."""
    for rel, body in files.items():
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "cpu.max").write_text(body)
    real_open = open

    def fake_open(path, *a, **kw):
        p = str(path)
        if p == "/proc/self/cgroup":
            return io.StringIO(proc_self)
        if p.startswith("/sys/fs/cgroup/cpu/"):  # v1 tree absent in these fixtures
            raise OSError("no cgroup v1")
        if p.startswith("/sys/fs/cgroup"):
            return real_open(str(tmp_path) + p[len("/sys/fs/cgroup") :], *a, **kw)
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)


@pytest.mark.parametrize(
    "cpu_max,expected",
    [
        ("400000 100000", 4),  # a 4-CPU container
        ("50000 100000", 1),  # half a CPU floors to 1, never 0
        ("max 100000", None),  # unlimited: not a constraint
    ],
)
def test_cgroup_quota_shapes(monkeypatch, tmp_path, cpu_max, expected):
    """A container caps a quota, which neither cpu_count nor the affinity mask reflects."""
    _fake_cgroup(monkeypatch, tmp_path, proc_self="0::/\n", files={".": cpu_max})
    assert backend._cgroup_cpu_quota() == expected


def test_cgroup_quota_reads_the_nested_group_not_the_root(monkeypatch, tmp_path):
    """With the host hierarchy visible, the root is unlimited and the real group is capped."""
    _fake_cgroup(
        monkeypatch,
        tmp_path,
        proc_self="0::/system.slice/andrey.service\n",
        files={
            ".": "max 100000",
            "system.slice": "max 100000",
            "system.slice/andrey.service": "200000 100000",
        },
    )
    assert backend._cgroup_cpu_quota() == 2


def test_cgroup_quota_takes_the_tightest_ancestor(monkeypatch, tmp_path):
    """A quota anywhere up the chain applies, so an ancestor can bind tighter than the leaf."""
    _fake_cgroup(
        monkeypatch,
        tmp_path,
        proc_self="0::/parent/child\n",
        files={".": "max 100000", "parent": "200000 100000", "parent/child": "800000 100000"},
    )
    assert backend._cgroup_cpu_quota() == 2  # the parent's 2 CPUs, not the child's 8


def test_cgroup_quota_absent_is_not_a_limit(monkeypatch):
    """No cgroup at all (macOS, Windows, bare metal) must not clamp the count to 1."""

    def fake_open(path, *a, **kw):
        raise OSError("no cgroup")

    monkeypatch.setattr("builtins.open", fake_open)
    assert backend._cgroup_cpu_quota() is None


def test_cgroup_quota_reads_close_their_files(tmp_path):
    """Each read closes its file: an open handle left to the garbage collector warns."""
    (tmp_path / "cpu.max").write_text("200000 100000")
    (tmp_path / "cpu.cfs_quota_us").write_text("200000")
    (tmp_path / "cpu.cfs_period_us").write_text("100000")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        assert backend._quota_v2(str(tmp_path)) == 2
        assert backend._quota_v1(str(tmp_path)) == 2
        gc.collect()
    leaks = [w for w in caught if issubclass(w.category, ResourceWarning)]
    assert not [w for w in leaks if str(tmp_path) in str(w.message)]  # this test's files only


def _fake_cgroup_v1(monkeypatch, tmp_path, *, proc_self, files):
    """A cgroup **v1** tree: ``files`` maps a dir under the v1 cpu mount to its quota/period."""
    for rel, (quota, period) in files.items():
        d = tmp_path / rel
        d.mkdir(parents=True, exist_ok=True)
        (d / "cpu.cfs_quota_us").write_text(quota)
        (d / "cpu.cfs_period_us").write_text(period)
    real_open = open
    v1_root = "/sys/fs/cgroup/cpu/"

    def fake_open(path, *a, **kw):
        p = str(path)
        if p == "/proc/self/cgroup":
            return io.StringIO(proc_self)
        if p.startswith(v1_root):
            return real_open(str(tmp_path) + p[len(v1_root) - 1 :], *a, **kw)
        if p.startswith("/sys/fs/cgroup"):  # v2 tree absent on a v1 host
            raise OSError("no cgroup v2")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", fake_open)


# A v1 host writes per-controller lines and no ``0::`` line; a v2 host writes only ``0::``. Reading
# a path for the generation the host is not running is the case that has to stay non-fatal.
_V1_PROC = "4:cpu,cpuacct:/slurm/job_1/step_0\n2:cpuset:/slurm/job_1/step_0\n1:name=systemd:/\n"
_V2_PROC = "0::/\n"


def test_cgroup_quota_on_a_v1_only_host(monkeypatch, tmp_path):
    """SLURM nodes are cgroup v1: no ``0::`` line, so only the controller paths are readable."""
    files = {"slurm/job_1/step_0": ("400000", "100000")}
    _fake_cgroup_v1(monkeypatch, tmp_path, proc_self=_V1_PROC, files=files)
    assert backend._cgroup_cpu_quota() == 4


def test_cgroup_rel_paths_leaves_the_absent_generation_empty(monkeypatch, tmp_path):
    """One generation per host: the other's path is empty, never unbound."""
    _fake_cgroup_v1(monkeypatch, tmp_path, proc_self=_V1_PROC, files={".": ("max", "100000")})
    assert backend._cgroup_rel_paths() == ("", "/slurm/job_1/step_0")
    _fake_cgroup(monkeypatch, tmp_path, proc_self=_V2_PROC, files={".": "max 100000"})
    assert backend._cgroup_rel_paths() == ("/", "")


@pytest.mark.parametrize("proc_self", [_V1_PROC, _V2_PROC], ids=["v1-only", "v2-only"])
def test_usable_cpus_survives_either_cgroup_generation(monkeypatch, tmp_path, proc_self):
    """``num_workers=-1`` through the public path never hits an unbound name."""
    if proc_self is _V1_PROC:
        _fake_cgroup_v1(monkeypatch, tmp_path, proc_self=proc_self, files={".": ("max", "100000")})
    else:
        _fake_cgroup(monkeypatch, tmp_path, proc_self=proc_self, files={".": "max 100000"})
    assert backend.usable_cpus() >= 1
    assert backend.worker_count(-1) == backend.usable_cpus()
