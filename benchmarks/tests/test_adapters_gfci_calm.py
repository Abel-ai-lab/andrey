"""The GFCI and CALM lanes: the Tetrad PAG reader, the settings each pair runs, and short fits.

Tetrad's GFCI runs where ``ANDREY_BENCH_TETRAD_JAR`` names the causal-cmd jar; causal-learn's
CALM and Andrey's need torch. The CALM fits here shorten the optimizer to a few hundred steps, so
they check the conversion, not the answer.
"""

from __future__ import annotations

import inspect
import os
import pickle
import shutil
from types import SimpleNamespace

import calm_cpu
import gfci_latent
import numpy as np
import pytest

from andrey import GraphStructure
from andrey.core import ARROW, CIRCLE, TAIL
from andrey_bench import datasets
from andrey_bench.adapters import tetrad_cmd, tetrad_gfci
from andrey_bench.adapters.andrey_calm import DEFAULTS, PAPER_SCHEDULE, AndreyCALM
from andrey_bench.adapters.andrey_gfci import AndreyGFCI
from andrey_bench.adapters.causal_learn_calm import CausalLearnCALM

needs_tetrad = pytest.mark.skipif(
    not os.environ.get(tetrad_cmd.JAR_ENV)
    or not (os.environ.get(tetrad_cmd.JAVA_ENV) or shutil.which("java")),
    reason=f"no causal-cmd jar (${tetrad_cmd.JAR_ENV}) or no java",
)
#: A few hundred optimizer steps: enough to reach the conversion, far from a converged answer.
_SHORT = {"subproblem_iter": 200, "max_outer": 3}


def test_every_solution_has_its_own_name() -> None:
    for solutions in (gfci_latent.SOLUTIONS, calm_cpu.SOLUTIONS):
        names = [a.name for a in solutions]
        assert len(set(names)) == len(names)


# --- GFCI ------------------------------------------------------------------------------------------


def test_both_gfci_solutions_run_one_setting() -> None:
    mine, theirs = AndreyGFCI(), tetrad_gfci.TetradGFCI()
    assert (mine.algorithm, mine.output_type) == (theirs.algorithm, theirs.output_type)
    mine_params = mine.params()
    assert mine_params.pop("ebic_gamma") == 0.0  # Tetrad scores the standard BIC
    assert mine_params == theirs.params()


def test_the_parallel_gfci_returns_the_serial_pag(monkeypatch) -> None:
    from andrey.search import _parallel_ges

    monkeypatch.setenv("ANDREY_NUM_WORKERS", "2")
    monkeypatch.setenv("ANDREY_GES_PARALLEL_MIN_WORK", "0")  # every GES pass fans out
    pools = []
    get = _parallel_ges._LazyWorkerSet.get
    monkeypatch.setattr(
        _parallel_ges._LazyWorkerSet, "get", lambda self: pools.append(self) or get(self)
    )
    data, _, _, _ = datasets.generate_one(gfci_latent.ladder_keys([20], 1)[0])
    serial, parallel = AndreyGFCI(), AndreyGFCI("parallel")
    assert parallel.params() == serial.params()
    expected = serial.fit(data, serial.params())
    assert not pools
    np.testing.assert_array_equal(parallel.fit(data, parallel.params()), expected)
    assert pools


def test_the_tetrad_reader_keeps_every_endpoint() -> None:
    text = (
        "Graph Nodes:\nx0;x1;x2;x3;x4\n\nGraph Edges:\n1. x0 o-o x1\n2. x1 o-> x2\n3. x3 <-o x2\n"
        "4. x2 <-> x4\n5. x0 --> x3\n6. x4 <-- x1\n7. x3 --- x4\n\nGraph Attributes:\nScore: 1.0\n"
    )
    marks = tetrad_gfci.read_pag(text, 5)
    expected = {
        (0, 1): CIRCLE, (1, 0): CIRCLE,
        (1, 2): CIRCLE, (2, 1): ARROW,
        (3, 2): ARROW, (2, 3): CIRCLE,
        (2, 4): ARROW, (4, 2): ARROW,
        (0, 3): TAIL, (3, 0): ARROW,
        (4, 1): ARROW, (1, 4): TAIL,
        (3, 4): TAIL, (4, 3): TAIL,
    }  # fmt: skip
    found = {(int(i), int(j)): int(marks[i, j]) for i, j in zip(*np.nonzero(marks))}
    assert found == expected
    with pytest.raises(ValueError, match="unexpected"):
        tetrad_gfci.read_pag("Graph Edges:\n1. x0 o-> x1 dd nl\n", 2)


def test_the_gfci_lane_hides_one_node_in_ten() -> None:
    keys = gfci_latent.ladder_keys([20], 2)
    assert [(k.d, k.n, k.latents) for k in keys] == [(20, 200, 2), (20, 200, 2)]
    data, marks, _, _ = datasets.generate_one(keys[0])
    assert data.shape == (200, 18) and marks.shape == (20, 20)


def test_tetrad_reads_the_test_level_and_penalty_from_the_params() -> None:
    values = tetrad_gfci.settings({"alpha": 0.01, "lambda_value": 2.0})
    assert (values["alpha"], values["penaltyDiscount"]) == (0.01, 2.0)


def test_the_tetrad_gfci_fit_runs_the_driver_and_reports_its_search_time(monkeypatch) -> None:
    """``fit`` starts the driver once on continuous data, and the search time is the driver's."""
    commands = []
    pag = "Graph Edges:\n1. x0 o-> x1\n\nGraph Attributes:\nScore: 1.0\n"

    def run(command, **kwargs):
        commands.append(command)
        out = command[command.index("--out") + 1], command[command.index("--prefix") + 1]
        (tetrad_cmd.Path(out[0]) / f"{out[1]}_out.txt").write_text(
            f"Categorical columns: \nSearch seconds: 0.25\n\nGraph Nodes:\nx0;x1\n\n{pag}"
        )
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(tetrad_cmd, "_java", lambda: "java")
    monkeypatch.setattr(tetrad_cmd, "_jar", lambda: tetrad_cmd.Path("causal-cmd.jar"))
    monkeypatch.setattr(tetrad_cmd.subprocess, "run", run)
    adapter = pickle.loads(pickle.dumps(tetrad_gfci.TetradGFCI()))  # as the runner sends it
    adapter.warmup = 2
    marks = adapter.fit(np.zeros((10, 2)), adapter.params())
    (command,) = commands
    assert command[command.index("--warmup") + 1] == "2"
    assert command[command.index("--data-type") + 1] == "continuous"
    assert "--numCategories" not in command
    assert command[command.index("--algorithm") + 1] == "gfci"
    assert adapter.search_s == 0.25
    assert (marks[0, 1], marks[1, 0]) == (CIRCLE, ARROW)


@needs_tetrad
def test_a_tetrad_fit_gives_the_causal_cmd_pag() -> None:
    """The driver's search returns the PAG ``java -jar`` returns for the same settings."""
    adapter = tetrad_gfci.TetradGFCI()
    adapter.setup()
    for key in gfci_latent.ladder_keys([10, 20], 2):
        data, _, _, _ = datasets.generate_one(key)
        marks = adapter.fit(data, adapter.params())
        assert adapter.search_s > 0
        np.testing.assert_array_equal(marks, tetrad_gfci.run_causal_cmd(data, adapter.params()))
        warm = pickle.loads(pickle.dumps(adapter))
        warm.warmup = 1  # a warm-up search in the same JVM leaves the PAG as it was
        np.testing.assert_array_equal(warm.fit(data, adapter.params()), marks)
    structure = adapter.to_structure(marks)
    assert structure.kind == "pag" and structure.n_nodes == data.shape[1]
    structure.validate()


# --- CALM ------------------------------------------------------------------------------------------


def test_the_default_schedule_is_andreys() -> None:
    from andrey.search.calm import calm

    signature = inspect.signature(calm).parameters
    defaults = {k: v for k, v in DEFAULTS.items() if k != "seed"}
    assert {k: signature[k].default for k in defaults} == defaults


def test_both_packages_run_the_paper_schedule() -> None:
    """Every setting causal-learn exposes has its value in Andrey's paper-schedule solution."""
    theirs = CausalLearnCALM().params()
    assert {k: PAPER_SCHEDULE[k] for k in theirs} == theirs
    assert AndreyCALM("paper").params() == PAPER_SCHEDULE
    with pytest.raises(ValueError, match="schedule"):
        AndreyCALM("fast")


def test_andrey_calm_forwards_the_schedule(monkeypatch) -> None:
    import andrey

    calls = []
    marks = np.array([[0, 1], [2, 0]], dtype=np.int8)

    def fit(x, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(structure=GraphStructure.from_numpy(marks, kind="dag"))

    monkeypatch.setattr(andrey, "calm", fit)
    np.testing.assert_array_equal(AndreyCALM("paper").fit(np.zeros((10, 2)), {}), marks)
    assert calls == [PAPER_SCHEDULE]


@pytest.mark.parametrize("adapter", [AndreyCALM(), CausalLearnCALM()], ids=lambda a: a.name)
def test_a_short_calm_fit_gives_a_dag(adapter) -> None:
    pytest.importorskip("torch")
    if adapter.package == "causal-learn":
        pytest.importorskip("causallearn")
    data, _, _, _ = datasets.generate_one(calm_cpu.ladder_keys([5], 1)[0])
    adapter.setup()
    short = {k: v for k, v in _SHORT.items() if k in adapter.params()}
    structure = adapter.to_structure(np.asarray(adapter.fit(data, short)))
    assert structure.kind == "dag" and structure.n_nodes == 5
    structure.validate()
