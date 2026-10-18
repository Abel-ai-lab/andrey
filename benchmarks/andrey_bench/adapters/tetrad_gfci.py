"""Tetrad's GFCI on continuous data, run through the causal-cmd jar and the one-file Java driver.

``TetradGFCI`` is a :class:`~andrey_bench.adapters.tetrad_cmd._TetradCmd`, so it times its search
as the mixed-data solutions do: ``fit`` writes the data to a CSV, starts ``java`` once on
``TetradDriver.java`` with causal-cmd's arguments for GFCI, and reads the PAG back from the text
file the driver writes. The driver runs the warm-up searches the harness asks for in the same JVM,
then times the search call alone, so the record's ``wall_s`` is Tetrad's search inside the JVM and
its ``process_s`` is the whole call, with the JVM start, the compile of the driver, and the CSV
round trip. The JVM sees only the fit's cores (``-XX:ActiveProcessorCount``).
:func:`run_causal_cmd` runs the same settings through ``java -jar``; the lane's tests check that
both return the same PAG. Tetrad's setting names map to Andrey's:

===================================  ============================================================
Tetrad (causal-cmd)                  Andrey
===================================  ============================================================
``fisher-z-test``                    ``indep_test="fisherz"``
``alpha``                            ``alpha``
``sem-bic-score``                    ``score_func="local_score_BIC"``
``penaltyDiscount``                  ``lambda_value``
===================================  ============================================================

Every other setting is set to what Andrey's GFCI does: the BIC score has no structure prior
(``semBicStructurePrior 0``) and the standard penalty rule (``semBicRule 1``); conditioning sets
and discriminating paths have no length cap (``depth -1``, ``maxPathLength -1``); the
possible-d-separation step, the discriminating-path rule, and the complete rule set R1-R10 run
(``possibleMsepDone``, ``doDiscriminatingPathRule``, ``completeRuleSetUsed``). The search keeps
causal-cmd's degree cap of 1,000 (``maxDegree``), which no benchmark graph reaches, and reads the
covariance matrix once (``precomputeCovariances``), which changes its speed, not its answer.

``ANDREY_BENCH_TETRAD_JAR`` names the jar and ``ANDREY_BENCH_JAVA`` the ``java`` executable, by
default the one on ``PATH``; it needs the ``jdk.compiler`` module, which runs a source file. The
README's "GFCI" section gives the jar's release, download, and checksum.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.tetrad_cmd import _jar, _java, _processors, _TetradCmd

_DEFAULT_ALPHA = 0.05
_DEFAULT_LAMBDA = 1.0
#: An edge line of Tetrad's text graph: ``1. x0 o-> x4``.
_EDGE = re.compile(r"^\d+\.\s+x(\d+)\s+([<o-])-([>o-])\s+x(\d+)\s*$")
#: Tetrad's endpoint characters -> Andrey's marks (TAIL=1, ARROW=2, CIRCLE=3).
_MARK = {"-": 1, "<": 2, ">": 2, "o": 3}


def settings(params: Mapping[str, Any]) -> dict[str, float | int | bool]:
    """Tetrad's GFCI settings, by causal-cmd's names, for Andrey's ``params``."""
    return {
        "alpha": float(params.get("alpha", _DEFAULT_ALPHA)),
        "penaltyDiscount": float(params.get("lambda_value", _DEFAULT_LAMBDA)),
        "semBicStructurePrior": 0.0,
        "semBicRule": 1,
        "depth": -1,
        "maxPathLength": -1,
        "maxDegree": 1000,
        "possibleMsepDone": True,
        "doDiscriminatingPathRule": True,
        "completeRuleSetUsed": True,
        "precomputeCovariances": True,
    }


def read_pag(text: str, d: int) -> np.ndarray:
    """Return the endpoint marks of Tetrad's text PAG over ``x0..x{d-1}``.

    ``marks[i, j]`` is the mark at ``i`` on edge ``i-j``: ``x0 o-> x4`` sets a circle at ``x0`` and
    an arrowhead at ``x4``. A line in any other form raises ``ValueError``.
    """
    marks = np.zeros((d, d), dtype=np.int8)
    body = text.split("Graph Edges:", 1)[1].split("Graph Attributes", 1)[0]
    for line in body.strip().splitlines():
        if not line.strip():
            continue
        match = _EDGE.match(line.strip())
        if match is None:
            raise ValueError(f"unexpected Tetrad edge line: {line!r}")
        a, left, right, b = match.groups()
        marks[int(a), int(b)] = _MARK[left]
        marks[int(b), int(a)] = _MARK[right]
    return marks


def _gfci_arguments(params: Mapping[str, Any]) -> list[str]:
    """causal-cmd's arguments for GFCI with the Fisher-Z test and the SEM BIC score."""
    command = [
        "--algorithm",
        "gfci",
        "--test",
        "fisher-z-test",
        "--score",
        "sem-bic-score",
    ]
    for name, value in settings(params).items():
        command += [f"--{name}"] if value is True else [f"--{name}", repr(value)]
    return command


def run_causal_cmd(data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Run the same GFCI through ``java -jar`` on a CSV file, one JVM per call; return its PAG."""
    x = np.asarray(data, dtype=np.float64)
    d = x.shape[1]
    with tempfile.TemporaryDirectory(prefix="andrey_bench_tetrad_") as tmp:
        work = Path(tmp)
        csv = work / "data.csv"
        header = ",".join(f"x{j}" for j in range(d))
        np.savetxt(csv, x, delimiter=",", fmt="%.17g", header=header, comments="")
        command = [
            _java(),
            f"-XX:ActiveProcessorCount={_processors()}",
            f"-Djava.util.prefs.userRoot={work}",
            "-jar",
            str(_jar()),
            "--data-type",
            "continuous",
            "--dataset",
            str(csv),
            "--delimiter",
            "comma",
            "--out",
            str(work),
            "--prefix",
            "fit",
            *_gfci_arguments(params),
        ]
        # causal-cmd writes its log to the working directory.
        done = subprocess.run(command, capture_output=True, text=True, cwd=work)
        result = work / "fit_out.txt"
        if done.returncode != 0 or not result.is_file():
            raise RuntimeError(f"causal-cmd exited {done.returncode}: {done.stderr.strip()[-500:]}")
        return read_pag(result.read_text(), d)


class TetradGFCI(_TetradCmd):
    """Tetrad's GFCI with the Fisher-Z test and the SEM BIC score, through the Java driver."""

    name: str = "tetrad.gfci"
    algorithm: str = "gfci"
    output_type: str = "pag"
    data_type: str = "continuous"

    def params(self) -> dict[str, Any]:
        """Andrey's names for the settings; :func:`settings` maps them to Tetrad's."""
        return {
            "alpha": _DEFAULT_ALPHA,
            "indep_test": "fisherz",
            "score_func": "local_score_BIC",
            "lambda_value": _DEFAULT_LAMBDA,
        }

    def _arguments(self, params: Mapping[str, Any]) -> list[str]:
        return _gfci_arguments(params)

    def read(self, text: str, d: int) -> np.ndarray:
        """The driver's PAG text as endpoint marks."""
        return read_pag(text, d)

    def to_structure(self, native: np.ndarray) -> Any:
        """PAG endpoint marks -> ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
