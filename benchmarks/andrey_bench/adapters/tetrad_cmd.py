"""Tetrad's PC and FGES on mixed data, run as a black box through the causal-cmd command line.

The released jar is the only interface: ``fit`` writes the data to a CSV with a header, runs
``java -jar <causal-cmd jar>`` once, and reads the graph back from the text file it writes. The
timed region therefore holds the JVM start, the CSV write and read, and the search; the README's
"Mixed data" section says so beside every Tetrad time. Its setting names map to Andrey's:

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

PC runs order-independent (``--stableFAS``), keeps the collider found first
(``--conflictRule 1``), and has no depth cap, as Andrey's PC does. FGES runs with no structure
prior (``--structurePrior 0``), as Andrey's GES has none, and without the discretization option.
Each fit's JVM sees only the cores the runner gives the fit (``-XX:ActiveProcessorCount``).

``ANDREY_BENCH_TETRAD_JAR`` names the jar and ``ANDREY_BENCH_JAVA`` the ``java`` executable, by
default the one on ``PATH``. The README's "Mixed data" section gives the jar's release, download,
and checksum.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

JAR_ENV = "ANDREY_BENCH_TETRAD_JAR"
JAVA_ENV = "ANDREY_BENCH_JAVA"
#: The release the lane was written against; a different jar is recorded through its own version.
CAUSAL_CMD_VERSION = "1.12.0"

_DEFAULT_ALPHA = 0.05
_DEFAULT_LAMBDA = 1.0
_MAX_CATEGORIES = 10
#: Andrey's setting names -> causal-cmd's test and score names.
_TESTS = {"degen_gauss": "dg-lr-test", "cond_gauss": "cg-lr-test"}
_SCORES = {"local_score_degen_gauss": "dg-bic-score", "local_score_cond_gauss": "cg-bic-score"}
#: An edge line of causal-cmd's text graph: ``1. x0 --> x4``.
_EDGE = re.compile(r"^\d+\.\s+x(\d+)\s+(-->|---|<->|<--)\s+x(\d+)\s*$")


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
    """One causal-cmd algorithm and test or score; ``fit`` runs the jar once."""

    package: str = "tetrad"
    package_version: str = f"causal-cmd {CAUSAL_CMD_VERSION}"
    backend: str = "jvm"
    mode: str = "serial"
    output_type: str = "cpdag"

    def setup(self) -> None:
        """Check ``java`` and the jar before timing; the JVM itself starts inside ``fit``."""
        subprocess.run(
            [_java(), "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        _jar()

    def _arguments(self, params: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Write the CSV, run causal-cmd, and read its graph back. The only timed region."""
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
                "-jar",
                str(_jar()),
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
            done = subprocess.run(command, capture_output=True, text=True)
            result = work / "fit_out.txt"
            if done.returncode != 0 or not result.is_file():
                raise RuntimeError(
                    f"causal-cmd exited {done.returncode}: {done.stderr.strip()[-500:]}"
                )
            return read_graph(result.read_text(), d)

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
