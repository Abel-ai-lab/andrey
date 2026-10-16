"""Tetrad's PC and FGES on mixed data, run through the causal-cmd jar and a one-file Java driver.

causal-cmd 1.12.0 parses ``--numCategories`` but never stores it, so its own command line reads
every column as continuous. ``TetradDriver.java``, beside this module, takes causal-cmd's argument
parser, parameters, and algorithm factory from the jar and replaces only the reader with one that
applies ``--numCategories``. ``fit`` writes the data to a CSV with a header, starts ``java`` once
on the driver's source with causal-cmd's arguments, and reads the graph back from the text file it
writes. The driver times the search call alone inside the JVM, after any warm-up searches in the
same JVM, and ``fit`` reports that time as ``search_s``, which the record keeps as ``wall_s``; the
whole call, with the JVM start, the in-memory compile of the driver, and the CSV round trip, is the
record's ``process_s``. The driver also names the columns it read as categorical, and ``fit``
stops unless they are the lane rule's. Its setting names map to Andrey's:

=====================================  ==========================================================
causal-cmd                             Andrey
=====================================  ==========================================================
``--data-type mixed --numCategories``  ``max_categories``: a column of at most that many distinct
                                       values is categorical
``--test dg-lr-test``                  ``indep_test="degen_gauss"``
``--test cg-lr-test``                  ``indep_test="cond_gauss"``
``--score dg-bic-score``               ``score_func="local_score_degen_gauss"``
``--score cg-bic-score``               ``score_func="local_score_cond_gauss"``
``--alpha``                            ``alpha``
``--penaltyDiscount``                  ``lambda_value``
=====================================  ==========================================================

PC runs order-independent (``--stableFAS``) and has no depth cap, as Andrey's PC does. It keeps
Tetrad's default collider conflict rule (``--conflictRule 1``), under which a later collider
replaces an earlier one; Andrey's PC leaves such an edge undirected. FGES runs with no structure
prior (``--structurePrior 0``), as Andrey's GES has none, and without the discretization option.
Each fit's JVM sees only the cores the runner gives the fit (``-XX:ActiveProcessorCount``).

``ANDREY_BENCH_TETRAD_JAR`` names the jar and ``ANDREY_BENCH_JAVA`` the ``java`` executable, by
default the one on ``PATH``; it needs the ``jdk.compiler`` module, which runs a source file. The
README's "Mixed data" section gives the jar's release, download, and checksum.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

JAR_ENV = "ANDREY_BENCH_TETRAD_JAR"
JAVA_ENV = "ANDREY_BENCH_JAVA"
#: The driver ``java`` runs from source: causal-cmd's search with the categorical columns declared.
DRIVER = Path(__file__).with_name("TetradDriver.java")

_DEFAULT_ALPHA = 0.05
_DEFAULT_LAMBDA = 1.0
_MAX_CATEGORIES = 10
#: Andrey's setting names -> causal-cmd's test and score names.
_TESTS = {"degen_gauss": "dg-lr-test", "cond_gauss": "cg-lr-test"}
_SCORES = {"local_score_degen_gauss": "dg-bic-score", "local_score_cond_gauss": "cg-bic-score"}
#: An edge line of causal-cmd's text graph: ``1. x0 --> x4``.
_EDGE = re.compile(r"^\d+\.\s+x(\d+)\s+(-->|---|<->|<--)\s+x(\d+)\s*$")
#: The driver's output lines naming the columns it read as categorical and timing the search.
_CATEGORICAL = "Categorical columns:"
_SEARCH_S = "Search seconds:"


def _java() -> str:
    java = os.environ.get(JAVA_ENV) or shutil.which("java")
    if not java:
        raise RuntimeError(f"no java: put it on PATH or set ${JAVA_ENV}")
    return java


def _jar() -> Path:
    jar = os.environ.get(JAR_ENV, "").strip()
    if not jar or not Path(jar).is_file():
        raise RuntimeError(f"set ${JAR_ENV} to the causal-cmd jar (README: Mixed data)")
    return Path(jar)


def _processors() -> int:
    """The cores the runner gives this fit, read from the thread pins it sets."""
    try:
        return max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    except ValueError:
        return 1


@functools.cache
def versions() -> str:
    """Name the releases a fit runs: ``causal-cmd 1.12.0 (tetrad-lib 7.6.3), Java 21.0.6``.

    causal-cmd's and tetrad-lib's come from the jar's Maven metadata and Java's from
    ``java -version``; a part whose jar or ``java`` is missing reads ``unknown``.
    """
    found = {"causal-cmd": "unknown", "tetrad-lib": "unknown"}
    jar = Path(os.environ.get(JAR_ENV, "").strip())
    if jar.is_file():
        with zipfile.ZipFile(jar) as archive:
            for artifact in found:
                name = f"META-INF/maven/io.github.cmu-phil/{artifact}/pom.properties"
                if name in archive.namelist():
                    text = archive.read(name).decode()
                    found[artifact] = re.search(r"^version=(.+)$", text, re.M).group(1).strip()
    java = "unknown"
    executable = os.environ.get(JAVA_ENV) or shutil.which("java")
    if executable:
        done = subprocess.run([executable, "-version"], capture_output=True, text=True)
        match = re.search(r'version "([^"]+)"', done.stderr)
        java = match.group(1) if match else java
    return f"causal-cmd {found['causal-cmd']} (tetrad-lib {found['tetrad-lib']}), Java {java}"


def _tagged(text: str, tag: str) -> str:
    """Return the rest of the line in the driver's output header that starts with ``tag``."""
    for line in text.split("\n\n", 1)[0].splitlines():
        if line.startswith(tag):
            return line.removeprefix(tag).strip()
    raise ValueError(f"no {tag!r} line in the driver's output")


def read_categorical(text: str) -> list[str]:
    """Return the column names the driver's output text names as read categorical."""
    names = _tagged(text, _CATEGORICAL)
    return names.split(";") if names else []


def read_search_s(text: str) -> float:
    """Return the seconds the driver's output text gives the timed search call inside the JVM."""
    return float(_tagged(text, _SEARCH_S))


def lane_categorical(x: np.ndarray, max_categories: int) -> list[str]:
    """Name the columns the lane reads as categorical: at most that many distinct whole numbers."""
    return [
        f"x{j}"
        for j in range(x.shape[1])
        if np.all(x[:, j] == np.round(x[:, j])) and np.unique(x[:, j]).size <= max_categories
    ]


def read_graph(text: str, d: int) -> np.ndarray:
    """Return the CPDAG adjacency of causal-cmd's text graph over ``x0..x{d-1}``.

    ``1`` at ``[i, j]`` marks an arrowhead at ``j`` or an undirected end: ``-->`` sets
    ``[i, j]``, ``---`` sets both. Any other mark raises ``ValueError``, so a PAG never passes as a
    CPDAG.
    """
    adjacency = np.zeros((d, d), dtype=np.int8)
    body = text.split("Graph Edges:", 1)[1].split("Graph Attributes", 1)[0]
    for line in body.strip().splitlines():
        if not line.strip():
            continue
        match = _EDGE.match(line.strip())
        if match is None:
            raise ValueError(f"unexpected causal-cmd edge line: {line!r}")
        a, mark, b = int(match.group(1)), match.group(2), int(match.group(3))
        if mark == "<--":
            a, b, mark = b, a, "-->"
        if mark == "<->":
            raise ValueError(f"causal-cmd returned a bidirected edge, not a CPDAG: {line!r}")
        adjacency[a, b] = 1
        if mark == "---":
            adjacency[b, a] = 1
    return adjacency


class _TetradCmd:
    """One causal-cmd algorithm and test or score; ``fit`` runs the driver once.

    ``search_s`` and ``warmup`` follow the harness's convention for a search timed in another
    process (``contracts.SolutionAdapter``): ``fit`` sets ``search_s``, and the worker sets
    ``warmup``, the searches the driver runs untimed before the timed one in the same JVM.
    """

    package: str = "tetrad"
    backend: str = "jvm"
    mode: str = "serial"
    output_type: str = "cpdag"
    search_s: float | None = None
    warmup: int = 0

    @property
    def package_version(self) -> str:
        """causal-cmd's, tetrad-lib's, and Java's releases (:func:`versions`)."""
        return versions()

    def setup(self) -> None:
        """Check ``java``, its compiler module, and the jar before timing.

        The JVM itself starts, and compiles the driver, inside ``fit``.
        """
        modules = subprocess.run(
            [_java(), "--list-modules"], check=True, capture_output=True, text=True
        ).stdout
        if not re.search(r"^jdk\.compiler@", modules, re.M):
            raise RuntimeError(f"{_java()} has no jdk.compiler module to run {DRIVER.name}")
        _jar()

    def _arguments(self, params: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError

    def run(self, data: np.ndarray, params: Mapping[str, Any]) -> str:
        """Write the CSV, run the driver, and return the text file it writes."""
        x = np.asarray(data, dtype=np.float64)
        d = x.shape[1]
        with tempfile.TemporaryDirectory(prefix="andrey_bench_tetrad_") as tmp:
            work = Path(tmp)
            csv = work / "data.csv"
            np.savetxt(
                csv,
                x,
                delimiter=",",
                fmt="%.17g",
                header=",".join(f"x{j}" for j in range(d)),
                comments="",
            )
            command = [
                _java(),
                f"-XX:ActiveProcessorCount={_processors()}",
                f"-Djava.util.prefs.userRoot={work}",  # keeps Java's preferences out of $HOME
                "--class-path",
                str(_jar()),
                str(DRIVER),
                "search",
                "--warmup",
                str(self.warmup),
                "--data-type",
                "mixed",
                "--numCategories",
                str(int(params.get("max_categories", _MAX_CATEGORIES))),
                "--dataset",
                str(csv),
                "--delimiter",
                "comma",
                "--out",
                str(work),
                "--prefix",
                "fit",
                *self._arguments(params),
            ]
            # causal-cmd's logger writes causal-cmd.log into the working directory.
            done = subprocess.run(command, capture_output=True, text=True, cwd=work)
            result = work / "fit_out.txt"
            if done.returncode != 0 or not result.is_file():
                raise RuntimeError(
                    f"{DRIVER.name} exited {done.returncode}: {done.stderr.strip()[-500:]}"
                )
            return result.read_text()

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the driver, set ``search_s`` to its search time, and read its graph back.

        Raises ``RuntimeError`` unless Tetrad read exactly the lane rule's categorical columns.
        """
        x = np.asarray(data, dtype=np.float64)
        text = self.run(x, params)
        self.search_s = read_search_s(text)
        declared = read_categorical(text)
        expected = lane_categorical(x, int(params.get("max_categories", _MAX_CATEGORIES)))
        if declared != expected:
            raise RuntimeError(f"Tetrad read {declared} as categorical, the lane rule {expected}")
        return read_graph(text, x.shape[1])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class TetradPC(_TetradCmd):
    """Tetrad's PC with a mixed-data test, named by Andrey's ``indep_test`` value."""

    algorithm: str = "pc"

    def __init__(self, indep_test: str) -> None:
        self.indep_test = indep_test
        self.name = f"tetrad.pc.{indep_test}"

    def params(self) -> dict[str, Any]:
        """Andrey's names for the settings; ``fit`` maps them to causal-cmd's."""
        return {
            "alpha": _DEFAULT_ALPHA,
            "indep_test": self.indep_test,
            "max_categories": _MAX_CATEGORIES,
        }

    def _arguments(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "--algorithm",
            "pc",
            "--test",
            _TESTS[str(params.get("indep_test", self.indep_test))],
            "--alpha",
            repr(float(params.get("alpha", _DEFAULT_ALPHA))),
            "--stableFAS",
            "--conflictRule",
            "1",
            "--depth",
            "-1",
        ]


class TetradFGES(_TetradCmd):
    """Tetrad's FGES with a mixed-data score, named by Andrey's ``score_func`` value."""

    algorithm: str = "ges"

    def __init__(self, score_func: str) -> None:
        self.score_func = score_func
        self.name = f"tetrad.fges.{score_func.removeprefix('local_score_')}"

    def params(self) -> dict[str, Any]:
        """Andrey's names for the settings; ``fit`` maps them to causal-cmd's."""
        return {
            "score_func": self.score_func,
            "lambda_value": _DEFAULT_LAMBDA,
            "max_categories": _MAX_CATEGORIES,
        }

    def _arguments(self, params: Mapping[str, Any]) -> list[str]:
        return [
            "--algorithm",
            "fges",
            "--score",
            _SCORES[str(params.get("score_func", self.score_func))],
            "--penaltyDiscount",
            repr(float(params.get("lambda_value", _DEFAULT_LAMBDA))),
            "--structurePrior",
            "0",
        ]


def tetrad_adapters() -> list[_TetradCmd]:
    """PC with each mixed-data test, then FGES with each mixed-data score."""
    return [
        TetradPC("degen_gauss"),
        TetradPC("cond_gauss"),
        TetradFGES("local_score_degen_gauss"),
        TetradFGES("local_score_cond_gauss"),
    ]
