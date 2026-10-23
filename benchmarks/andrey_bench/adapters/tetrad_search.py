"""Tetrad's PC, FGES, BOSS, GRaSP, and FCI on continuous data, through one source-run Java driver.

Each solution's ``fit`` writes the data to a CSV file, starts ``java`` once on
``TetradContinuousDriver.java`` against tetrad-lib's shaded jar, and reads the graph back from the
text file the driver writes. The driver calls Tetrad's own search class, with every setting given
on its command line, and runs the warm-up searches the harness asks for in the same JVM before the
timed one. The timed search builds the independence test or the score from the data and runs the
search; the record's ``wall_s`` is that time inside the JVM and its ``process_s`` is the whole
call, with the JVM start, the in-memory compile of the driver, and the CSV round trip. The JVM
sees only the fit's cores (``-XX:ActiveProcessorCount``), and FGES's and BOSS's thread pools use
all of them; PC's skeleton pass and FCI's run on the JVM's common pool, which has as many threads
as the JVM sees processors.

A solution's ``params`` use Andrey's names for the settings it shares with Andrey's adapter, and
Tetrad's setter names for the rest, each written out; :meth:`_TetradSearch.settings` maps them to
the driver's:

======================================  =========================================================
Andrey                                  Tetrad (driver setting)
======================================  =========================================================
``alpha``                               ``alpha``, on ``IndTestFisherZ``
``indep_test="fisherz"``                ``IndTestFisherZ`` (the only test run)
``score_func="local_score_BIC"`` and    ``SemBicScore``, ``ruleType chickering``
``"local_score_BIC_from_cov"``
``lambda_value``                        ``penaltyDiscount``: both score ``n log s2 + lambda k log n``
``max_parents=None``                    ``maxDegree -1``, FGES's cap on a node's degree
``seed``                                ``seed`` of ``PermutationSearch`` and ``Grasp``
``depth``                               GRaSP's ``depth``
``collider_rule="sepsets"``             ``r0ColliderRule sepsets``
======================================  =========================================================

``ANDREY_BENCH_TETRAD_GIN_JAR`` names the jar, the tetrad-lib 7.6.10 shaded jar the GIN solution
runs on, and ``ANDREY_BENCH_JAVA`` the ``java`` executable, by default the one on ``PATH``; it needs
the ``jdk.compiler`` module, which runs a source file. The README's "Tetrad on continuous data"
section gives the jar's download and checksum.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import ANDREY_LAMBDA, SEARCH_SEED
from andrey_bench.adapters.tetrad_cmd import _java, _processors, read_graph, read_search_s
from andrey_bench.adapters.tetrad_gfci import read_pag
from andrey_bench.adapters.tetrad_gin import JAR_ENV, versions

DRIVER = Path(__file__).with_name("TetradContinuousDriver.java")

#: Andrey's setting names -> the driver's, for the settings the two share under different names.
_DRIVER_NAME = {"lambda_value": "penaltyDiscount", "collider_rule": "r0ColliderRule"}

_ALPHA = 0.05
#: The standard BIC penalty, which the GES lanes of the harness run.
_BIC_PENALTY = 1.0
#: The score settings FGES, BOSS, and GRaSP share, by Tetrad's names: no structure prior, the
#: standard BIC rule, and the covariance matrix read once.
_SCORE = {"structurePrior": 0.0, "ruleType": "chickering", "precomputeCovariances": True}


def _jar() -> Path:
    jar = os.environ.get(JAR_ENV, "").strip()
    if not jar or not Path(jar).is_file():
        raise RuntimeError(
            f"set ${JAR_ENV} to tetrad-lib's shaded jar (README: Tetrad on continuous data)"
        )
    return Path(jar)


def _flag(value: Any) -> str:
    """A setting's value as the driver reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, float) else str(value)


class _TetradSearch:
    """One Tetrad search with the settings its subclass lists; ``fit`` runs the driver once.

    ``search_s`` and ``warmup`` follow the harness's convention for a search timed in another
    process (``contracts.SolutionAdapter``): ``fit`` sets ``search_s``, and the worker sets
    ``warmup``, the searches the driver runs untimed before the timed one in the same JVM. A
    subclass sets ``name``, ``algorithm``, ``command``, ``defaults``, and ``labels``, and for a
    graph that is not a CPDAG, ``output_type``, ``read``, and ``to_structure``.
    """

    package: str = "tetrad"
    backend: str = "jvm"
    mode: str = "serial"
    output_type: str = "cpdag"
    name: str
    algorithm: str
    #: The driver's name for the search: Tetrad's class, so FGES for the harness's ``ges``.
    command: str
    #: Every setting the search runs, by the names ``params`` use.
    defaults: dict[str, Any]
    #: The settings that choose the test or the score by Andrey's name, with the one value run.
    labels: dict[str, Any]
    search_s: float | None = None
    warmup: int = 0

    @property
    def package_version(self) -> str:
        """tetrad-lib's and Java's releases (:func:`~andrey_bench.adapters.tetrad_gin.versions`)."""
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

    def params(self) -> dict[str, Any]:
        """Every setting the search runs."""
        return dict(self.defaults)

    def settings(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The driver's settings for ``params``, which override :meth:`params` by name.

        Raises ``ValueError`` for a test or a score the solution does not run.
        """
        merged = {**self.defaults, **params}
        for label, value in self.labels.items():
            if merged.pop(label) != value:
                raise ValueError(f"{self.name} runs {label}={value!r} only")
        return {_DRIVER_NAME.get(key, key): value for key, value in merged.items()}

    def run(self, x: np.ndarray, params: Mapping[str, Any]) -> str:
        """Write the CSV, run the driver, and return the text file it writes."""
        flags = [
            part
            for key, value in self.settings(params).items()
            for part in (f"--{key}", _flag(value))
        ]
        with tempfile.TemporaryDirectory(prefix="andrey_bench_tetrad_") as tmp:
            work = Path(tmp)
            csv, result = work / "data.csv", work / "fit_out.txt"
            header = ",".join(f"x{j}" for j in range(x.shape[1]))
            np.savetxt(csv, x, delimiter=",", fmt="%.17g", header=header, comments="")
            command = [
                _java(),
                f"-XX:ActiveProcessorCount={_processors()}",
                f"-Djava.util.prefs.userRoot={work}",  # keeps Java's preferences out of $HOME
                "--class-path",
                str(_jar()),
                str(DRIVER),
                "search",
                self.command,
                str(csv),
                str(self.warmup),
                str(result),
                *flags,
            ]
            done = subprocess.run(command, capture_output=True, text=True, cwd=work)
            if done.returncode != 0 or not result.is_file():
                raise RuntimeError(
                    f"{DRIVER.name} exited {done.returncode}: {done.stderr.strip()[-500:]}"
                )
            return result.read_text()

    def read(self, text: str, d: int) -> np.ndarray:
        """The driver's graph text as the adapter's native array: a CPDAG's 0/1 adjacency."""
        return read_graph(text, d)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the driver, set ``search_s`` to its search time, and read its graph back."""
        x = np.asarray(data, dtype=np.float64)
        text = self.run(x, params)
        self.search_s = read_search_s(text)
        return self.read(text, x.shape[1])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class TetradPC(_TetradSearch):
    """Tetrad's PC with the Fisher z test: order-independent skeleton, no depth cap.

    Colliders are oriented by the separating sets the skeleton search found, and a collider is
    skipped when it would reverse an arrowhead already placed or close a directed cycle, so the
    first collider in Tetrad's visiting order wins; Andrey's PC keeps the collider whose pair was
    separated with the larger p-value. Tetrad 7.6.10 has no setting for the rule.
    """

    name = "tetrad.pc.fisherz"
    algorithm = command = "pc"
    labels = {"indep_test": "fisherz"}
    defaults = {
        "alpha": _ALPHA,
        "indep_test": "fisherz",
        "depth": -1,
        "fasStable": True,
        "colliderOrientationStyle": "sepsets",
        "allowBidirected": False,
    }


class TetradFGES(_TetradSearch):
    """Tetrad's FGES with the BIC score: no structure prior and no cap on a node's degree."""

    name = "tetrad.fges.bic"
    algorithm = "ges"
    command = "fges"
    labels = {"score_func": "local_score_BIC"}
    defaults = {
        "score_func": "local_score_BIC",
        "lambda_value": _BIC_PENALTY,
        "max_parents": None,
        **_SCORE,
        "symmetricFirstStep": False,
        "faithfulnessAssumed": False,
    }

    def settings(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The driver's settings; ``max_parents=None`` is ``maxDegree -1``, and any cap raises."""
        settings = super().settings(params)
        if settings.pop("max_parents") is not None:
            raise ValueError("FGES caps a node's degree, not its parents; run it uncapped")
        return {**settings, "maxDegree": -1, "numThreads": _processors()}


class TetradBOSS(_TetradSearch):
    """Tetrad's BOSS with the BIC score: one start from the data's column order, no BES."""

    name = "tetrad.boss.bic"
    algorithm = command = "boss"
    labels = {"score_func": "local_score_BIC_from_cov"}
    defaults = {
        "score_func": "local_score_BIC_from_cov",
        "lambda_value": ANDREY_LAMBDA,
        "seed": SEARCH_SEED,
        **_SCORE,
        "useBes": False,
        "numStarts": 1,
        "useDataOrder": True,
    }

    def settings(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The driver's settings, with the thread pool at the fit's cores."""
        return {**super().settings(params), "numThreads": _processors()}


class TetradGRaSP(_TetradSearch):
    """Tetrad's GRaSP with the BIC score: one start from the data's column order."""

    name = "tetrad.grasp.bic"
    algorithm = command = "grasp"
    labels = {"score_func": "local_score_BIC_from_cov"}
    defaults = {
        "score_func": "local_score_BIC_from_cov",
        "lambda_value": ANDREY_LAMBDA,
        "depth": 3,
        "seed": SEARCH_SEED,
        **_SCORE,
        "uncoveredDepth": 1,
        "nonSingularDepth": 1,
        "ordered": False,
        "useScore": True,
        "useRaskuttiUhler": False,
        "useDataOrder": True,
        "allowInternalRandomness": False,
        "numStarts": 1,
    }


class TetradFCI(_TetradSearch):
    """Tetrad's FCI with the Fisher z test: order-independent skeleton, possible-d-separation,
    complete rule set R1-R10, and no cap on conditioning sets or discriminating paths."""

    name = "tetrad.fci.fisherz"
    algorithm = command = "fci"
    output_type = "pag"
    labels = {"indep_test": "fisherz"}
    defaults = {
        "alpha": _ALPHA,
        "indep_test": "fisherz",
        "collider_rule": "sepsets",
        "depth": -1,
        "stable": True,
        "doPossibleDsep": True,
        "completeRuleSetUsed": True,
        "maxDiscriminatingPathLength": -1,
        "guaranteePag": False,
        "excludeSelectionBias": False,
    }

    def read(self, text: str, d: int) -> np.ndarray:
        """The driver's PAG text as endpoint marks."""
        return read_pag(text, d)

    def to_structure(self, native: np.ndarray) -> Any:
        """PAG endpoint marks -> ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
