"""Tetrad's FGES, BOSS, and GRaSP on continuous data with missing values, through one source-run
Java driver.

Tetrad 8.0.0 gives its tests and scores a missing-data policy (``MissingDataPolicy``). Under
``EM_COVARIANCE`` its linear-Gaussian BIC (``SemBicScore``) estimates the mean and the covariance
once by EM under a saturated Gaussian model (``EmCovarianceEstimator``: started from each column's
mean and variance over its observed values, run until no entry moves by more than the tolerance)
and the search runs on that covariance with every row as the sample size. Under ``TESTWISE``, what
the score did before the policies and still does when none is named, each local score reads the
rows complete on the variable and its parents.

:class:`TetradMissing` is one search under one policy:

- ``em_covariance``, the EM arm, named ``tetrad.<search>.em_covariance``;
- ``em_covariance`` with ``ess="mean_pairwise"``, the EM arm with the effective sample size of
  Tetrad's ``mean_pairwise`` mode, named ``tetrad.<search>.em_covariance.mean_pairwise``;
- ``testwise``, test-wise deletion, named ``tetrad.<search>.testwise``;
- ``fail``, which stops on a missing value: the search on complete data, named
  ``tetrad.<search>.complete``.

EM here differs from structural EM (Friedman 1997): the covariance is estimated once, with no graph,
before the search, and is not estimated again from the graph the search returns.

Each ``fit`` writes the data to a CSV file, ``nan`` where a value is missing, starts ``java`` once on
``TetradMissingDriver.java`` against tetrad-lib's shaded jar, and reads the CPDAG back from the text
file the driver writes. The timed region inside the JVM builds the score, the EM estimate with it,
and runs the search; the record's ``wall_s`` is that time and its ``process_s`` the whole call. The
JVM sees only the fit's cores (``-XX:ActiveProcessorCount``).

``params`` are the driver's settings, each named for the Tetrad setter or ``MissingDataSpec``
method it reaches and written out, none left to a default. The score is the standard BIC,
``n log s2 + penaltyDiscount * k log n`` with no structure prior, which Andrey's linear-Gaussian BIC
scores at ``penalty_discount`` equal to ``penaltyDiscount`` and ``ebic_gamma=0.0``. The EM settings
are those of ``MissingDataSpec.of``: no ridge, a tolerance of ``1e-6``, at most 1,000 iterations,
and every row as the effective sample size (``full_n``). The EM arm's ``ess`` names another mode of
``MissingDataSpec.EffectiveSampleSizeMode``: ``mean_pairwise``, the mean over variable pairs of the
rows where both are observed, or ``min_pairwise``, their minimum. The BIC then takes that count as
its sample size, in the likelihood and in the penalty's ``log n``. The search settings are those of
:mod:`~andrey_bench.adapters.tetrad_search`; GRaSP has no seed in 8.0.0 and runs one start from the
data's column order without internal randomness.

``ANDREY_BENCH_TETRAD_MISSING_JAR`` names the jar, which ``benchmarks/start_tetrad_missing.sh``
downloads and checks, and ``ANDREY_BENCH_JAVA`` the ``java`` executable, by default the one on
``PATH``; it needs the ``jdk.compiler`` module, which runs a source file.

References
----------
.. [1] Ramsey, J. D., Zhang, K., Glymour, M., Sanchez Romero, R., Huang, B., Ebert-Uphoff, I.,
   Samarasinghe, S., Barnes, E. A., and Glymour, C. (2018). TETRAD - a toolbox for causal
   discovery. 8th International Workshop on Climate Informatics.
.. [2] Dempster, A. P., Laird, N. M., and Rubin, D. B. (1977). Maximum likelihood from incomplete
   data via the EM algorithm. Journal of the Royal Statistical Society: Series B 39(1), 1-22.
.. [3] Friedman, N. (1997). Learning belief networks in the presence of missing values and hidden
   variables. Proceedings of the 14th International Conference on Machine Learning, 125-133.
"""

from __future__ import annotations

import functools
import os
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.bic_penalty import SEARCH_SEED
from andrey_bench.adapters.tetrad_cmd import _java, _processors, read_graph, read_search_s

JAR_ENV = "ANDREY_BENCH_TETRAD_MISSING_JAR"
DRIVER = Path(__file__).with_name("TetradMissingDriver.java")

#: The standard BIC, Andrey's ``penalty_discount=1.0`` at ``ebic_gamma=0.0``.
PENALTY_DISCOUNT = 1.0
#: The arm's name for each policy the driver reads (``MissingDataPolicy``, lower case).
POLICIES = {"em_covariance": "em_covariance", "testwise": "testwise", "fail": "complete"}
#: The effective sample sizes the driver reads (``EffectiveSampleSizeMode``, lower case) and
#: ``MissingDataSpec.of``'s, ``full_n``. Only ``em_covariance`` reads one.
ESS_MODES = ("full_n", "min_pairwise", "mean_pairwise")
#: The score settings: no structure prior, the standard BIC rule, the covariance matrix read once,
#: and ``MissingDataSpec.of``'s EM settings; an arm's ``ess`` replaces ``essMode``.
_SCORE = {
    "penaltyDiscount": PENALTY_DISCOUNT,
    "structurePrior": 0.0,
    "ruleType": "chickering",
    "precomputeCovariances": True,
    "emRidge": 0.0,
    "emTolerance": 1e-6,
    "emMaxIterations": 1000,
    "essMode": "full_n",
}
#: Each search's settings, and whether it takes the fit's cores as ``numThreads``.
_SEARCH: dict[str, tuple[dict[str, Any], bool]] = {
    "fges": ({"maxDegree": -1, "symmetricFirstStep": False, "faithfulnessAssumed": False}, True),
    "boss": (
        {
            "useBes": False,
            "numStarts": 1,
            "useDataOrder": True,
            "resetAfterBM": False,
            "resetAfterRS": True,
            "seed": SEARCH_SEED,
        },
        True,
    ),
    "grasp": (
        {
            "depth": 3,
            "uncoveredDepth": 1,
            "nonSingularDepth": 1,
            "ordered": False,
            "useScore": True,
            "useRaskuttiUhler": False,
            "useDataOrder": True,
            "allowInternalRandomness": False,
            "numStarts": 1,
        },
        False,
    ),
}
#: The harness's algorithm name for each search: Tetrad's FGES is its ``ges``.
_ALGORITHM = {"fges": "ges", "boss": "boss", "grasp": "grasp"}


def _jar() -> Path:
    jar = os.environ.get(JAR_ENV, "").strip()
    if not jar or not Path(jar).is_file():
        raise RuntimeError(
            f"set ${JAR_ENV} to tetrad-lib's shaded jar (benchmarks/start_tetrad_missing.sh)"
        )
    return Path(jar)


@functools.cache
def versions() -> str:
    """Name the releases a fit runs: ``tetrad-lib 8.0.0, Java 21.0.12``."""
    tetrad = "unknown"
    jar = Path(os.environ.get(JAR_ENV, "").strip())
    if jar.is_file():
        with zipfile.ZipFile(jar) as archive:
            name = "META-INF/maven/io.github.cmu-phil/tetrad-lib/pom.properties"
            if name in archive.namelist():
                found = re.search(r"^version=(.+)$", archive.read(name).decode(), re.M)
                tetrad = found.group(1).strip() if found else tetrad
    java = "unknown"
    try:
        done = subprocess.run([_java(), "-version"], capture_output=True, text=True)
        match = re.search(r'version "([^"]+)"', done.stderr)
        java = match.group(1) if match else java
    except RuntimeError:  # no java
        pass
    return f"tetrad-lib {tetrad}, Java {java}"


def _flag(value: Any) -> str:
    """A setting's value as the driver reads it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value) if isinstance(value, float) else str(value)


class TetradMissing:
    """One Tetrad search with the BIC score under one missing-data policy.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. ``search`` is ``"fges"``, ``"boss"``, or
    ``"grasp"`` and ``policy`` a key of :data:`POLICIES`. ``search_s`` and ``warmup`` follow the
    harness's convention for a search timed in another process
    (``contracts.SolutionAdapter``): ``fit`` sets ``search_s``, and the worker sets ``warmup``,
    the searches the driver runs untimed before the timed one in the same JVM.
    """

    package: str = "tetrad"
    backend: str = "jvm"
    mode: str = "serial"
    output_type: str = "cpdag"
    search_s: float | None = None
    warmup: int = 0

    def __init__(self, search: str, policy: str = "em_covariance", ess: str = "full_n") -> None:
        if search not in _SEARCH:
            raise ValueError(f"search must be one of {sorted(_SEARCH)}, got {search!r}")
        if policy not in POLICIES:
            raise ValueError(f"policy must be one of {sorted(POLICIES)}, got {policy!r}")
        if ess not in ESS_MODES:
            raise ValueError(f"ess must be one of {list(ESS_MODES)}, got {ess!r}")
        if ess != "full_n" and policy != "em_covariance":
            raise ValueError(f"ess {ess!r} is read under em_covariance only, not {policy!r}")
        self.search = search
        self.policy = policy
        self.ess = ess
        self.algorithm = _ALGORITHM[search]
        self.name = f"tetrad.{search}.{POLICIES[policy]}" + ("" if ess == "full_n" else f".{ess}")

    @property
    def package_version(self) -> str:
        """tetrad-lib's and Java's releases (:func:`versions`)."""
        return versions()

    def setup(self) -> None:
        """Check ``java``, its compiler module, and the jar before timing."""
        modules = subprocess.run(
            [_java(), "--list-modules"], check=True, capture_output=True, text=True
        ).stdout
        if not re.search(r"^jdk\.compiler@", modules, re.M):
            raise RuntimeError(f"{_java()} has no jdk.compiler module to run {DRIVER.name}")
        _jar()

    def params(self) -> dict[str, Any]:
        """Every setting the driver reads but ``numThreads``, which is the fit's cores."""
        return {
            "missingDataPolicy": self.policy,
            **_SCORE,
            "essMode": self.ess,
            **_SEARCH[self.search][0],
        }

    def settings(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The driver's settings: ``params`` over :meth:`params`, and the thread count."""
        merged = {**self.params(), **params}
        if _SEARCH[self.search][1]:
            merged["numThreads"] = _processors()
        return merged

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the driver on `data`, ``NaN`` where a value is missing; return 0/1 CPDAG adjacency.

        Raises
        ------
        RuntimeError
            If the driver stops: an unknown setting, a missing value under ``fail``, or an error
            inside Tetrad.
        """
        x = np.asarray(data, dtype=np.float64)
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
                self.search,
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
            text = result.read_text()
        self.search_s = read_search_s(text)
        return read_graph(text, x.shape[1])

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 CPDAG adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


__all__ = [
    "DRIVER",
    "ESS_MODES",
    "JAR_ENV",
    "PENALTY_DISCOUNT",
    "POLICIES",
    "TetradMissing",
    "versions",
]
