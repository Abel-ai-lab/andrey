"""The continuous-data Tetrad lanes: the pairs they run, the settings the driver reads, and fits.

Each pair must run one algorithm with one setting; every setting an adapter passes must be one the
Java driver reads for that algorithm, and the driver must read no other; Tetrad's BIC penalty must
equal Andrey's; and the driver's graph text must read exactly. Tetrad's fits run where
``ANDREY_BENCH_TETRAD_GIN_JAR`` names tetrad-lib's shaded jar and ``java`` has the ``jdk.compiler``
module.
"""

from __future__ import annotations

import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tetrad_cpu
import tetrad_fci_latent

from andrey import GraphStructure
from andrey.core.ci import make_indep_test
from andrey.core.score import local_score_bic
from andrey_bench import datasets
from andrey_bench.adapters import tetrad_search
from andrey_bench.adapters.tetrad_cmd import JAVA_ENV, read_search_s
from andrey_bench.adapters.tetrad_gin import JAR_ENV
from andrey_bench.contracts import SolutionAdapter

TETRAD = [a for a in tetrad_cpu.SOLUTIONS if a.package == "tetrad"] + [
    a for a in tetrad_fci_latent.SOLUTIONS if a.package == "tetrad"
]
DRIVER = tetrad_search.DRIVER.read_text()


def _unavailable() -> str | None:
    """Why no Tetrad fit can run here, or ``None`` when one can."""
    jar = os.environ.get(JAR_ENV, "")
    java = os.environ.get(JAVA_ENV) or shutil.which("java")
    if not jar or not Path(jar).is_file():
        return f"no tetrad-lib jar (${JAR_ENV})"
    if not java:
        return "no java"
    modules = subprocess.run([java, "--list-modules"], capture_output=True, text=True)
    if modules.returncode != 0 or "jdk.compiler" not in modules.stdout:
        return f"{java} has no jdk.compiler module"
    return None


needs_tetrad = pytest.mark.skipif(_unavailable() is not None, reason=_unavailable() or "")


def _data(d: int = 20, n: int = 200, seed: int = 0, latents: bool = False) -> np.ndarray:
    coordinates = {**tetrad_cpu.SLICE, "d": d, "n": n, "seed": seed}
    if latents:
        coordinates["latents"] = datasets.latents_for(tetrad_fci_latent.REGIME, d)
    return datasets.generate_one(datasets.DatasetKey(**coordinates))[0]


@pytest.mark.parametrize("adapter", TETRAD, ids=lambda a: a.name)
def test_an_adapter_meets_the_harness_contract(adapter) -> None:
    assert isinstance(adapter, SolutionAdapter)
    json.dumps(adapter.params(), sort_keys=True)
    clone = pickle.loads(pickle.dumps(adapter))
    assert (clone.name, clone.params()) == (adapter.name, adapter.params())


def test_every_solution_has_its_own_name_and_an_empty_baseline() -> None:
    for solutions in (tetrad_cpu.SOLUTIONS, tetrad_fci_latent.SOLUTIONS):
        names = [a.name for a in solutions]
        assert len(set(names)) == len(names)
        assert "baseline.empty" in names


@pytest.mark.parametrize("pair", list(tetrad_cpu.PAIRS))
def test_both_packages_run_one_algorithm_with_one_setting(pair: str) -> None:
    mine, theirs = tetrad_cpu.PAIRS[pair]
    assert (mine.algorithm, mine.output_type) == (theirs.algorithm, theirs.output_type)
    shared = mine.params().keys() & theirs.params().keys()
    assert shared  # a pair with nothing in common compares nothing
    assert {k for k in shared if mine.params()[k] != theirs.params()[k]} == set()


def test_the_fci_pair_runs_one_algorithm_with_one_setting() -> None:
    mine, theirs = tetrad_fci_latent.SOLUTIONS[:2]
    assert (
        (mine.algorithm, mine.output_type)
        == (theirs.algorithm, theirs.output_type)
        == (
            "fci",
            "pag",
        )
    )
    shared = mine.params().keys() & theirs.params().keys()
    assert {"alpha", "indep_test", "collider_rule"} <= shared
    assert {k for k in shared if mine.params()[k] != theirs.params()[k]} == set()


def test_every_tetrad_solution_is_paired() -> None:
    paired = {a.name for pair in tetrad_cpu.PAIRS.values() for a in pair}
    assert {a.name for a in tetrad_cpu.SOLUTIONS} - paired == {"baseline.empty"}


# --- the settings -----------------------------------------------------------------------------


def _read_by_driver() -> dict[str, set[str]]:
    """The setting names the driver reads for each algorithm, from its source."""
    body = DRIVER.split("static Graph run(", 1)[1].split("static String upper(", 1)[0]
    score = set(re.findall(r's\.\w+\("(\w+)"\)', DRIVER.split("static SemBicScore score(", 1)[1]))
    read = {}
    for case in re.split(r'\n            case "', body)[1:]:
        name = case.split('"', 1)[0]
        names = set(re.findall(r's\.\w+\("(\w+)"\)', case))
        read[name] = names | score if "score(data, s)" in case else names
    return read


@pytest.mark.parametrize("adapter", TETRAD, ids=lambda a: a.name)
def test_an_adapter_passes_exactly_the_settings_the_driver_reads(adapter, monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert set(adapter.settings({})) == _read_by_driver()[adapter.command]


def test_the_penalty_discount_is_andreys_lambda() -> None:
    for adapter in (tetrad_search.TetradFGES(), tetrad_search.TetradBOSS()):
        assert adapter.settings({})["penaltyDiscount"] == adapter.params()["lambda_value"]
        assert adapter.settings({"lambda_value": 2.5})["penaltyDiscount"] == 2.5
    assert tetrad_search.TetradGRaSP().settings({"lambda_value": 3.0})["penaltyDiscount"] == 3.0


def test_fges_runs_uncapped_and_a_cap_is_refused() -> None:
    fges = tetrad_search.TetradFGES()
    assert fges.settings({})["maxDegree"] == -1
    with pytest.raises(ValueError, match="degree"):
        fges.settings({"max_parents": 4})


def test_a_test_or_score_tetrad_does_not_run_is_refused() -> None:
    with pytest.raises(ValueError, match="fisherz"):
        tetrad_search.TetradPC().settings({"indep_test": "kci"})
    with pytest.raises(ValueError, match="local_score_BIC"):
        tetrad_search.TetradFGES().settings({"score_func": "local_score_BDeu"})


def test_the_thread_pools_follow_the_fits_cores(monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert tetrad_search.TetradFGES().settings({})["numThreads"] == 3
    assert tetrad_search.TetradBOSS().settings({})["numThreads"] == 3
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert tetrad_search.TetradFGES().settings({})["numThreads"] == 1


def test_the_fci_skeleton_search_gets_the_uncached_test() -> None:
    """Tetrad's cache is a plain ``HashMap`` that its parallel skeleton search corrupts."""
    assert "fci.search(new Fas(test))" in DRIVER


def test_the_driver_gets_booleans_and_numbers_as_it_reads_them() -> None:
    assert [tetrad_search._flag(v) for v in (True, False, -1, 0.05, "sepsets")] == [
        "true",
        "false",
        "-1",
        "0.05",
        "sepsets",
    ]


# --- the graph text ---------------------------------------------------------------------------

_TEXT = "Search seconds: 0.125\n\nGraph Nodes:\nx0;x1;x2;x3\n\nGraph Edges:\n{}\n"


def test_tetrad_7_6_10_graph_text_reads_as_a_cpdag() -> None:
    text = _TEXT.format("1. x0 --> x2\n2. x1 --- x2\n3. x3 --> x2")
    expected = np.zeros((4, 4), dtype=np.int8)
    expected[0, 2] = expected[1, 2] = expected[2, 1] = expected[3, 2] = 1
    adapter = tetrad_search.TetradPC()
    assert np.array_equal(adapter.read(text, 4), expected)
    assert read_search_s(text) == 0.125
    assert adapter.to_structure(expected).kind == "cpdag"


def test_tetrad_7_6_10_graph_text_reads_as_a_pag() -> None:
    from andrey.core import ARROW, CIRCLE, TAIL

    text = _TEXT.format("1. x0 o-> x2\n2. x1 <-> x2\n3. x3 --- x2")
    marks = tetrad_search.TetradFCI().read(text, 4)
    assert (marks[0, 2], marks[2, 0]) == (CIRCLE, ARROW)
    assert (marks[1, 2], marks[2, 1]) == (ARROW, ARROW)
    assert (marks[3, 2], marks[2, 3]) == (TAIL, TAIL)
    assert tetrad_search.TetradFCI().to_structure(marks).kind == "pag"


def test_a_pag_never_passes_as_a_cpdag() -> None:
    with pytest.raises(ValueError, match="bidirected"):
        tetrad_search.TetradPC().read(_TEXT.format("1. x0 <-> x1"), 4)


# --- fits -------------------------------------------------------------------------------------


def _run_driver(*args: str) -> subprocess.CompletedProcess:
    java = os.environ.get(JAVA_ENV) or shutil.which("java")
    with tempfile.TemporaryDirectory() as tmp:
        command = [
            java,
            f"-Djava.util.prefs.userRoot={tmp}",
            "--class-path",
            os.environ[JAR_ENV],
            str(tetrad_search.DRIVER),
            *args,
        ]
        return subprocess.run(command, capture_output=True, text=True, cwd=tmp)


def _write_csv(csv: Path, data: np.ndarray) -> None:
    header = ",".join(f"x{j}" for j in range(data.shape[1]))
    np.savetxt(csv, data, delimiter=",", fmt="%.17g", header=header, comments="")


def _probe(mode: str, data: np.ndarray, first: str | None, queries: list[list[int]]) -> list[str]:
    """The driver's ``ptest`` or ``score`` output lines for ``queries``."""
    with tempfile.TemporaryDirectory() as tmp:
        csv, file = Path(tmp, "data.csv"), Path(tmp, "queries.txt")
        _write_csv(csv, data)
        file.write_text("\n".join(" ".join(map(str, q)) for q in queries))
        done = _run_driver(mode, str(csv), *([first] if first else []), str(file))
    assert done.returncode == 0, done.stderr[-500:]
    return done.stdout.strip().splitlines()


@needs_tetrad
@pytest.mark.parametrize("adapter", TETRAD, ids=lambda a: a.name)
def test_a_tetrad_fit_gives_its_graph_and_the_warm_up_leaves_it_as_it_was(adapter) -> None:
    data = _data(latents=adapter.command == "fci")
    adapter.setup()
    native = adapter.fit(data, adapter.params())
    assert adapter.search_s > 0
    structure = adapter.to_structure(native)
    assert structure.kind == adapter.output_type and structure.n_nodes == data.shape[1]
    warm = pickle.loads(pickle.dumps(adapter))
    warm.warmup = 1  # a warm-up search in the same JVM leaves the graph as it was
    assert np.array_equal(warm.fit(data, warm.params()), native)


@needs_tetrad
def test_the_driver_stops_on_a_missing_or_an_unknown_setting() -> None:
    adapter = tetrad_search.TetradPC()
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp, "data.csv")
        _write_csv(csv, _data())
        flags = [p for k, v in adapter.settings({}).items() for p in (f"--{k}", str(v).lower())]
        out = str(Path(tmp, "out.txt"))
        assert _run_driver("search", "pc", str(csv), "0", out, *flags).returncode == 0
        assert _run_driver("search", "pc", str(csv), "0", out, *flags[2:]).returncode != 0
        assert (
            _run_driver("search", "pc", str(csv), "0", out, *flags, "--extra", "1").returncode != 0
        )


@needs_tetrad
def test_tetrads_fisher_z_p_values_are_andreys() -> None:
    data = _data()
    queries = [[0, 1], [3, 7, 2], [5, 9, 1, 4], [12, 3, 8, 0, 15]]
    test = make_indep_test("fisherz", data)
    for line, query in zip(_probe("ptest", data, None, queries), queries, strict=True):
        p = float(line.rsplit("p=", 1)[1])
        assert p == pytest.approx(test(query[0], query[1], query[2:]), abs=1e-9, rel=1e-6)


@needs_tetrad
@pytest.mark.parametrize("penalty", [1.0, 4.0])
def test_tetrads_bic_penalty_is_andreys_lambda(penalty: float) -> None:
    """A parent set's score gain over the empty set is minus Andrey's, at equal coefficients."""
    data = _data()
    cov, n = np.cov(data, rowvar=False), data.shape[0]
    cases = [(4, [0]), (4, [0, 3]), (9, [1, 2, 5]), (9, [1, 2, 5, 6, 7]), (15, [2])]
    queries = [[node] for node, _ in cases] + [[node, *parents] for node, parents in cases]
    lines = _probe("score", data, str(penalty), queries)
    tetrad = [float(line.rsplit("score=", 1)[1]) for line in lines]
    for i, (node, parents) in enumerate(cases):
        gain = tetrad[len(cases) + i] - tetrad[i]
        andrey = local_score_bic(cov, n, node, parents, penalty_discount=penalty) - local_score_bic(
            cov, n, node, [], penalty_discount=penalty
        )
        assert gain == pytest.approx(-andrey, abs=1e-6)


@needs_tetrad
@pytest.mark.parametrize("seed", range(3))
def test_the_pc_skeletons_agree(seed: int) -> None:
    """The stable skeleton search finds the same adjacencies; the orientations may differ."""
    from andrey_bench.adapters.andrey_pc import AndreyPC

    data = _data(seed=seed)
    mine, theirs = AndreyPC(), tetrad_search.TetradPC()
    mine.setup()
    theirs.setup()
    a = np.asarray(mine.fit(data, mine.params()))
    t = theirs.fit(data, theirs.params())
    assert np.array_equal((a != 0) | (a != 0).T, (t != 0) | (t != 0).T)
    assert isinstance(mine.to_structure(a), GraphStructure)
