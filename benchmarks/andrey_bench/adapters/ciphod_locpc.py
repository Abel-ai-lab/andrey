"""LocPC-CDE adapter for the authors' code (Loranchet and Assaad, 2026), run unmodified.

The code is pyCIPHOD (``CIPHOD/pyCIPHOD``) at commit :data:`COMMIT`, under the MIT license. It is a
measurement reference only: ``benchmarks/start_ciphod.sh`` downloads the archive of the commit,
checks its SHA-256, and builds the environment; the benchmark imports the package from the ``src``
directory of ``$ANDREY_BENCH_CIPHOD`` and copies none of it.

A fit calls ``LocPC_CDE(treatment, outcome, sparsity=alpha, ci_test=...).run(frame)``, the
repository's class for the method (the loop of the ``LocPC.run_locPC_CDE`` its experiment scripts
call), with the repository's Fisher-z test (``FisherZTest``) on continuous data
and its G-square test (``GsqTest``) on binary data, which the authors' discrete experiment passes
as integers. The frame's columns are named ``V000``, ``V001``, ... with a width that keeps the
names' sorted order equal to the columns' order: the code visits and tests variables by sorted
name, so the columns are visited in the order the data has them. The code iterates sets of names
too, so its graph depends on Python's string hash seed; a fit refuses to run unless
``PYTHONHASHSEED`` is fixed (``start_ciphod.sh`` prints ``export PYTHONHASHSEED=0``).

The code returns ``{"identifiability", "adjustment_set"}`` and keeps its last graph in
``algo.local_constraint_based_algorithm.g_hat``. The record is read from that graph, with the
reasons and the fields of Andrey's ``locpc_cde``, so the two are scored alike. In the graph a
directed edge counts as oriented whatever else is stored for the pair (the code's own final check
reads it so; a pair can also sit among the uncertain, double-bar, edges of an earlier hop):

- ``"not adjacent"`` (the treatment is not adjacent to the outcome) or ``"treatment is a child"``
  (the outcome points at it): the effect is zero (``nonzero`` false);
- ``"oriented"`` (every edge at the outcome is directed): identifiable, the set is the outcome's
  parents but the treatment, and the effect is nonzero (the treatment is a parent);
- otherwise ``"not orientable"``: not identifiable, no set, and nonzero when the treatment's edge
  into the outcome is directed, else undecided.

After a zero stop, ``identifiable`` and the set are as for ``"oriented"`` when every edge at the
outcome is directed, else undecided with no set. The code's own ``identifiability`` flag is kept as
the record's ``authors_identifiability``, and the reason as ``reason``. The count of tests is the
code's ``nb_ci_tests``: every test it ran, a query asked again from its cache not counted.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adjustment_lane import (
    OUTCOME_PARAM,
    TREATMENT_PARAM,
    empty_structure,
    pack_effects,
    unpack_effects,
)

#: Names the directory holding the extracted code.
CODE_ENV = "ANDREY_BENCH_CIPHOD"

#: The pinned commit of ``CIPHOD/pyCIPHOD``; ``start_ciphod.sh`` fetches it.
COMMIT = "37a62c9f2052a793243a81b1a0b728eb33fbace1"

#: The files a fit imports, relative to the code directory.
REQUIRED_FILES = (
    "src/pyciphod/causal_discovery/local/local_constraint_based.py",
    "src/pyciphod/utils/stat_tests/independence_tests.py",
)

#: The code's settings: its default level and, as the README calls it, Fisher's z test.
DEFAULTS: dict[str, Any] = {"alpha": 0.05, "indep_test": "fisherz"}

#: The tests of the lane's cells and the class of the repository that runs each.
TESTS: dict[str, str] = {"fisherz": "FisherZTest", "gsq": "GsqTest"}

#: Why the code stopped, in the order its graph is read: the labels of Andrey's ``locpc_cde``.
REASONS: tuple[str, ...] = ("not adjacent", "treatment is a child", "oriented", "not orientable")

#: The extra fields of a record, kept in the packed answer's count row after the count.
_AUTHORS_FLAG, _REASON = 1, 2


def _code_dir() -> Path:
    raw = os.environ.get(CODE_ENV, "").strip()
    directory = Path(raw)
    if not raw or not all((directory / name).is_file() for name in REQUIRED_FILES):
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} does not hold the pyCIPHOD files; run benchmarks/start_ciphod.sh"
        )
    return directory


def _import() -> Any:
    """The code's module, imported from the downloaded ``src`` directory."""
    source = str(_code_dir() / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from pyciphod.causal_discovery.local import local_constraint_based
    from pyciphod.utils.stat_tests import independence_tests

    return local_constraint_based, independence_tests


def _check_hash_seed() -> None:
    """Refuse to run with a random string hash seed: the code iterates sets of names, so its graph
    can change from one interpreter to the next."""
    if sys.flags.hash_randomization and os.environ.get("PYTHONHASHSEED", "random") in (
        "",
        "random",
    ):
        raise RuntimeError(
            "pyCIPHOD iterates sets of names, so its graph depends on the string hash seed; "
            "export PYTHONHASHSEED=0 (start_ciphod.sh prints it)"
        )


def column_names(d: int) -> list[str]:
    """``V000``, ``V001``, ... for ``d`` columns, wide enough that sorted names are in column
    order."""
    width = max(3, len(str(d - 1)))
    return [f"V{i:0{width}d}" for i in range(d)]


def read_graph(
    treatment: int, outcome: int, directed: set[tuple[int, int]], adjacent: set[int]
) -> tuple[dict[str, Any], str]:
    """The record of the code's final graph and the reason it reads as.

    ``directed`` holds the ``(tail, head)`` positions of its directed edges and ``adjacent`` the
    positions adjacent to the outcome, whatever the edge's kind.
    """
    x, y = treatment, outcome
    parents = sorted(a for a, b in directed if b == y)
    oriented = all((n, y) in directed or (y, n) in directed for n in adjacent)
    if x not in adjacent:
        reason = REASONS[0]
    elif (y, x) in directed:
        reason = REASONS[1]
    elif oriented:
        reason = REASONS[2]
    else:
        reason = REASONS[3]
    nonzero = False if reason in REASONS[:2] else True if (x, y) in directed else None
    identifiable = False if reason == REASONS[3] else True if oriented else None
    record = {
        "treatment": x,
        "outcome": y,
        "estimand": "direct",
        "nonzero": nonzero,
        "identifiable": identifiable,
        "adjustment_sets": [[v for v in parents if v != x]] if identifiable else [],
        "set_kind": "parents_of_outcome" if identifiable else None,
    }
    return record, reason


class CiphodLocPCCDE:
    """The authors' LocPC-CDE at one treatment-outcome pair.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    name: str = "locpc_cde.authors"
    package: str = "pyciphod"
    package_version: str = COMMIT[:12]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "locpc_cde"
    output_type: str = "effects"
    estimand: str = "direct"
    set_kind: str = "parents_of_outcome"
    identified_from: str = "cpdag"
    unordered: bool = False
    #: The tests of the repository the cells can ask for.
    indep_tests = tuple(TESTS)
    python_env: str = "ANDREY_BENCH_CIPHOD_PYTHON"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _import()

    def params(self) -> dict[str, Any]:
        """The code's settings (:data:`DEFAULTS`)."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code for the pair in ``params``; return the packed record and the number of
        tests. The only timed region."""
        settings = {**DEFAULTS, **params}
        if settings["indep_test"] not in TESTS:
            raise ValueError(
                f"pyCIPHOD runs with the tests {sorted(TESTS)}, got indep_test="
                f"{settings['indep_test']!r}"
            )
        if settings.get("max_depth") is not None:
            raise ValueError("pyCIPHOD has no max_depth cap")
        module, tests = _import()
        _check_hash_seed()
        import pandas as pd

        X = np.asarray(data)
        d = X.shape[1]
        discrete = settings["indep_test"] == "gsq"
        names = column_names(d)
        frame = pd.DataFrame(
            X.astype(np.int64) if discrete else X.astype(np.float64), columns=names
        )
        x, y = int(params[TREATMENT_PARAM]), int(params[OUTCOME_PARAM])
        algorithm = module.LocPC_CDE(
            names[x],
            names[y],
            sparsity=float(settings["alpha"]),
            ci_test=getattr(tests, TESTS[settings["indep_test"]]),
        )
        answer = algorithm.run(frame)
        inner = algorithm.local_constraint_based_algorithm
        position = {name: i for i, name in enumerate(names)}
        graph = inner.g_hat
        directed = {(position[a], position[b]) for a, b in graph.get_directed_edges()}
        adjacent = {position[v] for v in graph.get_adjacencies(names[y])}
        record, reason = read_graph(x, y, directed, adjacent)
        packed = pack_effects([record], d, inner.nb_ci_tests)
        packed[-1, _AUTHORS_FLAG] = int(bool(answer["identifiability"]))
        packed[-1, _REASON] = REASONS.index(reason)
        return packed

    def records(
        self, native: np.ndarray, params: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], int | None]:
        """The packed answer's record with the reason and the code's own flag added; untimed."""
        records, n_tests = unpack_effects(native)
        tail = np.asarray(native)[-1]
        for record in records:
            record["reason"] = REASONS[int(tail[_REASON])]
            record["authors_identifiability"] = bool(tail[_AUTHORS_FLAG])
        return records, n_tests

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; the lane scores the records. Untimed."""
        return empty_structure(native)
