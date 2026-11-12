"""SNAP adapter for the authors' code (Schubert, Claassen, and Magliacane, 2025), run unmodified.

The code is ``Matyasch/snap`` (main) at commit :data:`COMMIT`, under the BSD 2-clause license, a
single NumPy file. It is a measurement reference only: ``benchmarks/start_schubert.sh`` downloads
the archive of the commit, checks its SHA-256, and builds the environment; the benchmark imports
``snap.py`` from the directory ``$ANDREY_BENCH_SNAP`` names and copies none of it. The branches of
the paper's experiments are not the reference.

A fit calls ``snap(data=, ci_test=, alpha=, targets=)`` as the repository's demo does, with the
two columns of the pair as the targets and Fisher's z test on the data's correlation matrix
(:class:`~andrey_bench.adapters.schubert_fisherz.FisherZ`) as ``ci_test``. ``max_order`` is the
code's own cap on the order of the tests; the lane runs SNAP(infinity), the code's negative
default. The code returns ``{"amat", "poss_anc"}``: ``amat[i, j] = 1`` for ``i -> j`` and for both
``(i, j)`` and ``(j, i)`` of an undirected edge, over the possible ancestors of the targets, and no
edge at any other node.

The answer is that graph, so ``fit`` returns it as endpoint marks in Andrey's convention
(:mod:`andrey_bench.contracts`) and the test count, and :meth:`SchubertSNAP.records` reads the
records in the parent, where Andrey is installed, as the PC arm of the lane reads PC's CPDAG
(:class:`~andrey_bench.adapters.andrey_adjustment.AndreyPCAdjustment`): the optimal adjustment set
where the graph identifies the total effect, the local parent sets of the treatment (Maathuis et
al., 2009) where it does not, for both directions of the pair. Andrey's adjustment core accepts a
``cpdag`` only when it is the CPDAG of its own class, which the pruned graph need not be, so the
records are read from the graph completed by :func:`andrey.core.orient.cpdag_of_pdag`: the same
skeleton and v-structures, every edge oriented as the class requires. A graph with no consistent
DAG extension decides nothing: both records have no answer.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.schubert_fisherz import FisherZ, check_settings
from andrey_bench.adjustment_lane import OPTIMAL_OR_LOCAL, VARIABLES_PARAM
from andrey_bench.local_lane import pack_answer, unpack_answer

#: Names the directory holding the downloaded code.
CODE_ENV = "ANDREY_BENCH_SNAP"

#: The pinned commit of ``Matyasch/snap``; ``start_schubert.sh`` fetches it.
COMMIT = "432ab5c969048817ff6ea87d5733a26319503059"

#: The files a fit imports, relative to the code directory.
REQUIRED_FILES = ("snap.py",)

#: The authors' settings: the level of the paper's experiments and of the code's demo, and no cap
#: on the order of the tests (SNAP(infinity)).
DEFAULTS: dict[str, Any] = {"alpha": 0.05, "indep_test": "fisherz", "max_order": None}

# Andrey's marks (:mod:`andrey_bench.contracts`).
_TAIL, _ARROW = 1, 2


def _code_dir() -> Path:
    raw = os.environ.get(CODE_ENV, "").strip()
    directory = Path(raw)
    if not raw or not all((directory / name).is_file() for name in REQUIRED_FILES):
        raise FileNotFoundError(
            f"${CODE_ENV}={raw!r} does not hold the SNAP files; run benchmarks/start_schubert.sh"
        )
    return directory


def _snap() -> Any:
    """The repository's ``snap`` function, imported from the downloaded directory."""
    directory = str(_code_dir())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    import snap  # the repository's module of that name

    return snap.snap


def andrey_marks(amat: np.ndarray) -> np.ndarray:
    """The code's ``amat`` as Andrey's endpoint marks: ``marks[i, j]`` is the mark at ``i``.

    ``amat[i, j]`` without ``amat[j, i]`` is ``i -> j``, a tail at ``i`` and an arrowhead at ``j``;
    both is an undirected edge, a tail at both ends.
    """
    edge = np.asarray(amat) != 0
    marks = np.zeros(edge.shape, dtype=np.int8)
    marks[edge] = _TAIL
    marks[edge.T & ~edge] = _ARROW
    return marks


def _undecided(x: int, y: int) -> dict[str, Any]:
    """The record of an answer that decides nothing about ``x`` on ``y``."""
    return {
        "treatment": x,
        "outcome": y,
        "estimand": "total",
        "nonzero": None,
        "identifiable": None,
        "adjustment_sets": [],
        "set_kind": None,
    }


def _read_pair(graph: Any, marks: np.ndarray, x: int, y: int) -> dict[str, Any]:
    """The ``effects`` record of ``x`` on ``y`` read from the completed CPDAG ``graph``."""
    from andrey.core import adjustment as adj

    record = _undecided(x, y)
    if adj.adjustment_set(graph, x, y) is not None:
        optimal = adj.optimal_set(graph, x, y)
        record.update(identifiable=True, nonzero=optimal is not None)
        if optimal is not None:
            record.update(adjustment_sets=[sorted(optimal)], set_kind="optimal")
        return record
    # Not identified: the parent sets the class allows, as the authors' evaluation falls back to
    # local IDA; the outcome is no member, and two sets that differ by it are one. The effect is
    # zero when the outcome is no possible descendant of the treatment, as the authors' evaluation
    # reads the graph before it looks for a set.
    sets = {tuple(sorted(v for v in s if v != y)) for s in adj.local_parent_sets(graph, x)}
    record.update(
        identifiable=False,
        nonzero=None if y in adj.possible_descendants(marks, [x]) else False,
        adjustment_sets=[list(s) for s in sorted(sets)],
        set_kind="local_parents",
    )
    return record


class SchubertSNAP:
    """The authors' SNAP at one pair of targets.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.adjustment_lane.PairAdapter`.
    """

    name: str = "snap.authors"
    package: str = "snap"
    package_version: str = COMMIT[:12]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "snap"
    output_type: str = "effects"
    estimand: str = "total"
    set_kind: str = OPTIMAL_OR_LOCAL
    identified_from: str = "cpdag"
    unordered: bool = True
    #: The code takes a callable test; the lane gives it Fisher's z.
    indep_tests = ("fisherz",)
    python_env: str = "ANDREY_BENCH_SCHUBERT_PYTHON"

    def setup(self) -> None:
        """Import the code outside fit timing."""
        _snap()

    def params(self) -> dict[str, Any]:
        """The authors' settings (:data:`DEFAULTS`)."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the code with the pair in ``params["variables"]`` as its targets; return the
        packed marks of its graph and the number of tests. The only timed region."""
        X = np.asarray(data, dtype=np.float64)
        settings = {**DEFAULTS, **params}
        check_settings(settings, "SNAP")
        pair = [int(v) for v in params[VARIABLES_PARAM]]
        max_order = settings["max_order"]
        run = _snap()
        test = FisherZ(X)
        out = run(
            data=X,
            ci_test=test,
            alpha=float(settings["alpha"]),
            targets=pair,
            max_order=-1 if max_order is None else int(max_order),
        )
        return pack_answer(andrey_marks(out["amat"]), test.n_tests)

    def records(
        self, native: np.ndarray, params: Mapping[str, Any]
    ) -> tuple[list[dict[str, Any]], int | None]:
        """The ``effects`` records of both directions of the pair and the test count; untimed.

        Read from ``native``'s graph completed to a CPDAG, as the module docstring says; where it
        has no consistent DAG extension, both records have no answer.
        """
        from andrey import GraphStructure
        from andrey.core.orient import cpdag_of_pdag

        marks, n_tests = unpack_answer(native)
        i, j = (int(v) for v in params[VARIABLES_PARAM])
        pair = ((i, j), (j, i))
        try:
            completed = cpdag_of_pdag(marks.astype(np.int8))
            graph = GraphStructure.from_numpy(completed, kind="cpdag")
        except ValueError:
            return [_undecided(x, y) for x, y in pair], n_tests
        return [_read_pair(graph, completed, x, y) for x, y in pair], n_tests

    def to_structure(self, native: np.ndarray) -> Any:
        """The graph without edges; the lane scores the records. Untimed."""
        from andrey import GraphStructure

        d = unpack_answer(native)[0].shape[0]
        return GraphStructure.from_numpy(np.zeros((d, d), dtype=np.int8), kind="digraph")
