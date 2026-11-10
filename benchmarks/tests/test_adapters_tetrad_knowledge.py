"""Background knowledge, FCI-JCI, BOSS-FCI, and GRaSP-FCI through both Tetrad drivers, and pcalg's
FCI-JCI.

A context tier must orient every edge at a context variable away from it, a forbidden edge must
never appear, knowledge must leave the adjacencies of these small graphs as they were, each fit's
in-JVM ``search_s`` must be positive and at most the whole call's time, and every PAG must hold
valid marks. causal-cmd's fits run where ``ANDREY_BENCH_TETRAD_JAR`` names its jar, the continuous
driver's where ``ANDREY_BENCH_TETRAD_GIN_JAR`` names tetrad-lib's, and pcalg's where R runs.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_r

from andrey import GraphStructure
from andrey.core import ARROW, CIRCLE, TAIL
from andrey_bench.adapters import tetrad_cmd, tetrad_gfci, tetrad_gin, tetrad_pag, tetrad_search
from andrey_bench.adapters.pcalg_fci_jci import PcalgFCIJCI


def _unavailable(jar_env: str) -> str | None:
    """Why no fit on the jar ``jar_env`` names can run here, or ``None`` when one can."""
    jar = os.environ.get(jar_env, "")
    java = os.environ.get(tetrad_cmd.JAVA_ENV) or shutil.which("java")
    if not jar or not Path(jar).is_file():
        return f"no jar (${jar_env})"
    if not java:
        return "no java"
    modules = subprocess.run([java, "--list-modules"], capture_output=True, text=True)
    if modules.returncode != 0 or "jdk.compiler" not in modules.stdout:
        return f"{java} has no jdk.compiler module"
    return None


def _needs(jar_env: str) -> pytest.MarkDecorator:
    return pytest.mark.skipif(_unavailable(jar_env) is not None, reason=_unavailable(jar_env) or "")


needs_causal_cmd = _needs(tetrad_cmd.JAR_ENV)
needs_tetrad_lib = _needs(tetrad_gin.JAR_ENV)

#: causal-cmd's PAG searches, then the continuous driver's.
CMD = [
    tetrad_pag.TetradCmdFCI(),
    tetrad_gfci.TetradGFCI(),
    tetrad_pag.TetradBFCI(),
    tetrad_pag.TetradCmdGraspFCI(),
]
#: The continuous driver's searches take knowledge only without the rules for selection bias.
LIB = [tetrad_search.TetradFCI(), tetrad_search.TetradBossFCI(), tetrad_search.TetradGraspFCI()]
NO_SELECTION_BIAS = {"excludeSelectionBias": True}
CONTEXT = 4


def _context_data(n: int = 1000) -> np.ndarray:
    """A binary context ``x4 -> x0`` ahead of the chain ``x0 -> x1``, ``x1 -> x2``, ``x1 -> x3``."""
    rng = np.random.default_rng(0)
    c = rng.integers(0, 2, n).astype(float)
    x0 = c + rng.standard_normal(n)
    x1 = x0 + rng.standard_normal(n)
    return np.column_stack([x0, x1, x1 + rng.standard_normal(n), x1 + rng.standard_normal(n), c])


def _collider_data(n: int = 1000) -> np.ndarray:
    """``x0 -> x2 <- x1`` and ``x2 -> x3``, so without knowledge FCI orients ``x2 --> x3``."""
    rng = np.random.default_rng(1)
    x0, x1 = rng.standard_normal(n), rng.standard_normal(n)
    x2 = x0 + x1 + rng.standard_normal(n)
    return np.column_stack([x0, x1, x2, x2 + rng.standard_normal(n)])


def _fit(adapter, data: np.ndarray, **params) -> tuple[np.ndarray, float]:
    """The fit's PAG marks and the seconds the whole call took."""
    adapter.setup()
    start = time.perf_counter()
    marks = adapter.fit(data, {**adapter.params(), **params})
    return marks, time.perf_counter() - start


def _assert_valid_pag(marks: np.ndarray) -> None:
    assert set(np.unique(marks).tolist()) <= {0, TAIL, ARROW, CIRCLE}
    assert np.array_equal(marks != 0, (marks != 0).T)
    assert not marks.diagonal().any()
    GraphStructure.from_numpy(marks, kind="pag").validate()


def _adjacency(marks: np.ndarray) -> np.ndarray:
    return marks != 0


# --- the knowledge file -------------------------------------------------------------------------


def test_the_context_goes_in_tier_1_and_every_other_column_in_tier_2() -> None:
    text = tetrad_cmd.knowledge_text(5, [4, 1], forbidden=[[0, 2]], required=[[2, 3]])
    lines = text.splitlines()
    assert lines[lines.index("addtemporal") + 1 :][:2] == ["1 x4 x1", "2 x0 x2 x3"]
    assert lines[lines.index("forbiddirect") + 1] == "x0 x2"
    assert lines[lines.index("requiredirect") + 1] == "x2 x3"


def test_no_knowledge_writes_no_file_and_a_bad_column_is_refused() -> None:
    assert tetrad_cmd.knowledge_of({"alpha": 0.05, "context_vars": []}, 5) is None
    for bad in ({"context_vars": [5]}, {"context_vars": [0, 0]}, {"forbidden": [[0, 7]]}):
        with pytest.raises(ValueError):
            tetrad_cmd.knowledge_of(bad, 5)
    with pytest.raises(ValueError):
        tetrad_cmd.knowledge_text(2, [0, 1])  # no system column left


def test_a_fit_without_knowledge_runs_without_the_file(monkeypatch) -> None:
    seen = []

    def run(command, **kwargs):
        seen.append(command)
        out = command[command.index("--out") + 1]
        Path(out, "fit_out.txt").write_text(
            "Categorical columns: \nSearch seconds: 0.25\n\nGraph Edges:\n1. x0 o-> x1\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(tetrad_cmd, "_java", lambda: "java")
    monkeypatch.setattr(tetrad_cmd, "_jar", lambda: Path("causal-cmd.jar"))
    monkeypatch.setattr(tetrad_cmd.subprocess, "run", run)
    adapter = tetrad_pag.TetradBFCI()
    data = _context_data(50)
    adapter.fit(data, adapter.params())
    adapter.fit(data, {**adapter.params(), "context_vars": [CONTEXT]})
    assert ["--knowledge" in c for c in seen] == [False, True]
    assert all("--experimental" in c for c in seen)
    assert adapter.search_s == 0.25


@pytest.mark.parametrize("adapter", [*CMD, *LIB], ids=lambda a: a.name)
def test_possible_d_separation_runs_where_the_search_has_it(adapter) -> None:
    """causal-cmd's fci and gfci take ``possibleMsepDone``; its bfci and grasp-fci and 7.6.10's
    BOSS-FCI and GRaSP-FCI have no possible-d-separation step."""
    if isinstance(adapter, tetrad_cmd._TetradCmd):
        named = "--possibleMsepDone" in adapter._arguments(adapter.params())
        assert named == (adapter.algorithm in ("fci", "gfci"))
    else:
        assert adapter.settings({}).get("doPossibleDsep", False) == (adapter.algorithm == "fci")


def test_a_false_boolean_never_reaches_causal_cmd() -> None:
    """causal-cmd turns a boolean on whenever it is named, whatever value follows."""
    assert tetrad_pag.flags({"useBes": False, "completeRuleSetUsed": True, "depth": -1}) == [
        "--completeRuleSetUsed",
        "--depth",
        "-1",
    ]


def test_fci_jci_needs_context_columns() -> None:
    for adapter in (tetrad_pag.TetradCmdFCIJCI(), tetrad_search.TetradFCIJCI(), PcalgFCIJCI()):
        with pytest.raises(ValueError, match="context_vars"):
            adapter.fit(_context_data(50), adapter.params())


@pytest.mark.parametrize("adapter", LIB, ids=lambda a: a.name)
def test_tetrad_lib_refuses_knowledge_it_would_not_orient_by(adapter) -> None:
    with pytest.raises(ValueError, match="excludeSelectionBias"):
        adapter.settings({"context_vars": [CONTEXT]})
    assert adapter.settings({"context_vars": [CONTEXT], **NO_SELECTION_BIAS})


# --- causal-cmd's driver ------------------------------------------------------------------------


def _driver(*args: str) -> subprocess.CompletedProcess:
    java = os.environ.get(tetrad_cmd.JAVA_ENV) or shutil.which("java")
    with tempfile.TemporaryDirectory() as tmp:
        command = [
            java,
            f"-Djava.util.prefs.userRoot={tmp}",
            "--class-path",
            os.environ[tetrad_cmd.JAR_ENV],
            str(tetrad_cmd.DRIVER),
            *args,
        ]
        return subprocess.run(command, capture_output=True, text=True, cwd=tmp)


@needs_causal_cmd
def test_tetrads_tiers_forbid_every_system_to_context_edge_and_nothing_else() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp, "knowledge.txt")
        file.write_text(tetrad_cmd.knowledge_text(5, [3, 4]))
        done = _driver("knowledge", str(file))
    assert done.returncode == 0, done.stderr[-500:]
    forbidden = {tuple(line.split()[1:]) for line in done.stdout.splitlines()}
    system, context = ["x0", "x1", "x2"], ["x3", "x4"]
    assert forbidden == {(s, c) for s in system for c in context}


@needs_causal_cmd
def test_bfci_runs_only_as_an_experimental_algorithm() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp, "data.csv")
        np.savetxt(csv, _context_data(200), delimiter=",", header="x0,x1,x2,x3,x4", comments="")
        base = ["search", "--data-type", "continuous", "--dataset", str(csv)]
        base += ["--delimiter", "comma", "--out", tmp, "--algorithm", "bfci"]
        base += ["--test", "fisher-z-test", "--score", "sem-bic-score"]
        assert _driver(*base).returncode != 0
        assert _driver(*base, "--experimental").returncode == 0


@needs_causal_cmd
@pytest.mark.parametrize("adapter", CMD, ids=lambda a: a.name)
def test_a_causal_cmd_context_tier_orients_edges_out_of_the_context(adapter) -> None:
    data = _context_data()
    off, _ = _fit(adapter, data)
    on, seconds = _fit(adapter, data, context_vars=[CONTEXT])
    for marks in (off, on):
        _assert_valid_pag(marks)
    assert 0 < adapter.search_s <= seconds
    assert np.array_equal(_adjacency(off), _adjacency(on))
    assert _adjacency(on)[CONTEXT].any()
    for s in np.flatnonzero(_adjacency(on)[CONTEXT]):
        assert on[s, CONTEXT] == ARROW  # the system end of every context edge is an arrowhead
    assert not np.array_equal(off, on)


@needs_causal_cmd
@pytest.mark.parametrize("adapter", CMD, ids=lambda a: a.name)
def test_a_causal_cmd_forbidden_edge_never_appears(adapter) -> None:
    data = _collider_data()
    marks, _ = _fit(adapter, data, forbidden=[[2, 3]])
    _assert_valid_pag(marks)
    assert not (marks[2, 3] == TAIL and marks[3, 2] == ARROW)


@needs_causal_cmd
def test_the_fci_jci_solution_is_fci_with_the_context_tier() -> None:
    data = _context_data()
    jci, _ = _fit(tetrad_pag.TetradCmdFCIJCI(), data, context_vars=[CONTEXT])
    fci, _ = _fit(tetrad_pag.TetradCmdFCI(), data, context_vars=[CONTEXT])
    assert np.array_equal(jci, fci)


# --- tetrad-lib's continuous driver -------------------------------------------------------------


@needs_tetrad_lib
@pytest.mark.parametrize("adapter", LIB, ids=lambda a: a.name)
def test_a_tetrad_lib_context_tier_orients_edges_out_of_the_context(adapter) -> None:
    data = _context_data()
    off, _ = _fit(adapter, data, **NO_SELECTION_BIAS)
    on, seconds = _fit(adapter, data, context_vars=[CONTEXT], **NO_SELECTION_BIAS)
    for marks in (off, on):
        _assert_valid_pag(marks)
    assert 0 < adapter.search_s <= seconds
    assert np.array_equal(_adjacency(off), _adjacency(on))
    assert _adjacency(on)[CONTEXT].any()
    for s in np.flatnonzero(_adjacency(on)[CONTEXT]):
        assert on[s, CONTEXT] == ARROW
    assert not np.array_equal(off, on)


@needs_tetrad_lib
@pytest.mark.parametrize("adapter", LIB, ids=lambda a: a.name)
def test_a_tetrad_lib_forbidden_edge_never_appears(adapter) -> None:
    marks, _ = _fit(adapter, _collider_data(), forbidden=[[2, 3]], **NO_SELECTION_BIAS)
    _assert_valid_pag(marks)
    assert not (marks[2, 3] == TAIL and marks[3, 2] == ARROW)


@needs_tetrad_lib
def test_the_tetrad_lib_fci_jci_solution_leaves_out_selection_bias() -> None:
    data = _context_data()
    jci, _ = _fit(tetrad_search.TetradFCIJCI(), data, context_vars=[CONTEXT])
    fci, _ = _fit(tetrad_search.TetradFCI(), data, context_vars=[CONTEXT], **NO_SELECTION_BIAS)
    assert np.array_equal(jci, fci)


@needs_tetrad_lib
def test_tetrad_lib_stops_on_knowledge_for_a_search_without_it() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        csv, file = Path(tmp, "data.csv"), Path(tmp, "knowledge.txt")
        np.savetxt(csv, _context_data(200), delimiter=",", header="x0,x1,x2,x3,x4", comments="")
        file.write_text(tetrad_cmd.knowledge_text(5, [CONTEXT]))
        grasp = tetrad_search.TetradGRaSP()
        flags = [
            p for k, v in grasp.settings({}).items() for p in (f"--{k}", tetrad_search._flag(v))
        ]
        java = os.environ.get(tetrad_cmd.JAVA_ENV) or shutil.which("java")
        command = [java, f"-Djava.util.prefs.userRoot={tmp}", "--class-path"]
        command += [os.environ[tetrad_gin.JAR_ENV], str(tetrad_search.DRIVER), "search"]
        command += [grasp.command]
        command += [str(csv), "0", str(Path(tmp, "out.txt"))]
        assert subprocess.run([*command, *flags], capture_output=True).returncode == 0
        knowing = [*command, "--knowledge", str(file), *flags]
        assert subprocess.run(knowing, capture_output=True).returncode != 0


# --- pcalg --------------------------------------------------------------------------------------


@needs_r()
def test_pcalg_fci_jci_orients_edges_out_of_the_context() -> None:
    from andrey_bench.adapters.pcalg_fci import PcalgFCI

    data = _context_data()
    adapter = PcalgFCIJCI()
    marks, _ = _fit(adapter, data, context_vars=[CONTEXT])
    plain, _ = _fit(PcalgFCI(), data)
    _assert_valid_pag(marks)
    assert np.array_equal(_adjacency(marks), _adjacency(plain))
    assert _adjacency(marks)[CONTEXT].any()
    for s in np.flatnonzero(_adjacency(marks)[CONTEXT]):
        # JCI assumptions 1 and 2: no system variable causes or shares a hidden cause with it.
        assert (marks[CONTEXT, s], marks[s, CONTEXT]) == (TAIL, ARROW)
    assert adapter.to_structure(marks).kind == "pag"


@needs_r()
def test_pcalg_fci_jci_reads_the_context_by_column() -> None:
    """The same data with the columns reversed gives the same PAG, reversed."""
    data = _context_data()
    order = np.arange(data.shape[1])[::-1]
    marks, _ = _fit(PcalgFCIJCI(), data, context_vars=[CONTEXT])
    flipped, _ = _fit(
        PcalgFCIJCI(), data[:, order], context_vars=[int(np.flatnonzero(order == CONTEXT)[0])]
    )
    assert np.array_equal(flipped, marks[np.ix_(order, order)])
