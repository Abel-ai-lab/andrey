"""Andrey's Markov blanket and parents-and-children learners, for the set-learner lane.

One adapter runs any learner of :data:`LEARNERS`, through the public facade of the same name
(``andrey.iamb``, ``andrey.mmpc``, ...), in the two ways the lane asks for:

- **One target at a time**: ``fit`` runs the learner with ``target=`` for each position in
  ``params["targets"]`` and returns each target's set, the Markov blanket of a ``"mb"`` learner or
  the parents and children of a ``"pc"`` one, with the tests of each search where Andrey reports
  them (``metadata["n_tests"]``).
- **For the whole graph**: without ``targets``, ``fit`` runs the learner for the graph
  ``structure`` names, ``"moral"``, ``"skeleton"``, or ``"cpdag"``.

Every setting is passed, never defaulted: ``alpha``, ``indep_test``, and ``max_depth``, so a
learner whose default differs (HITON's ``max_depth``) still runs at the lane's setting. The same
three settings reach the reference (:mod:`bnlearn_blanket`), and the lane records them in every
row.

To add a learner, name it in :data:`LEARNERS` with the kind of set it learns and the graphs its
facade offers. The adapter finds the function by name, so nothing else changes: a learner the
installed Andrey does not have yet fails when its adapter is built (:func:`available` lists the
ones it has), and the lane's solution list takes every row whose function exists. The reference of
a learner is named in :data:`bnlearn_blanket.COUNTERPART`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version
from andrey_bench.blanket_lane import (
    NO_COUNT,
    STRUCTURE_PARAM,
    TARGETS_PARAM,
    answer_structure,
    pack_answer,
)

_DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class Learner:
    """What the lane needs to know about one learner of Andrey."""

    #: ``"mb"`` for a learner of Markov blankets, ``"pc"`` for one of parents and children.
    kind: str
    #: The graphs the facade's ``structure`` offers, from :data:`~andrey_bench.blanket_lane.STRUCTURES`.
    structures: tuple[str, ...]


#: The learners the lane runs, by the name of their function in ``andrey``. A learner is added by
#: naming it here.
LEARNERS: dict[str, Learner] = {
    "iamb": Learner("mb", ("moral", "skeleton", "cpdag")),
    "inter_iamb": Learner("mb", ("moral", "skeleton", "cpdag")),
    "mmpc": Learner("pc", ("skeleton", "cpdag")),
    "gs": Learner("mb", ("moral", "skeleton", "cpdag")),
    "fast_iamb": Learner("mb", ("moral", "skeleton", "cpdag")),
}


def available() -> list[str]:
    """The learners of :data:`LEARNERS` the installed Andrey has, in table order."""
    import andrey

    return [name for name in LEARNERS if callable(getattr(andrey, name, None))]


def n_tests(output: Any) -> int | None:
    """The tests a fit evaluated, ``None`` where the installed Andrey reports no count."""
    count = getattr(output, "metadata", {}).get("n_tests")
    return None if count is None else int(count)


class AndreyBlanket:
    """Solution adapter for one learner of :data:`LEARNERS`, at one setting.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. ``structure`` is the graph a whole-graph run
    answers with, and ``None`` for a run that answers for targets. Both read ``params``, which
    :meth:`params` fills and a lane adds ``targets`` to.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"

    def __init__(
        self,
        learner: str,
        *,
        indep_test: str,
        alpha: float = _DEFAULT_ALPHA,
        max_depth: int | None = None,
        structure: str | None = None,
    ) -> None:
        if learner not in LEARNERS:
            raise ValueError(f"learner must be one of {sorted(LEARNERS)}, got {learner!r}")
        if structure is not None and structure not in LEARNERS[learner].structures:
            raise ValueError(
                f"{learner} offers the graphs {LEARNERS[learner].structures}, got {structure!r}"
            )
        self.learner = learner
        self.kind = LEARNERS[learner].kind
        self.indep_test = indep_test
        self.alpha = alpha
        self.max_depth = max_depth
        self.structure = structure
        self.name = f"andrey.{learner}"
        self.algorithm = learner
        self.output_type = "cpdag" if structure == "cpdag" else "undirected"

    def setup(self) -> None:
        """Import Andrey and the blanket engine before timing, so no fit pays for them."""
        import importlib

        import andrey

        importlib.import_module("andrey.constraint.blanket")
        if not callable(getattr(andrey, self.learner, None)):
            raise RuntimeError(f"the installed Andrey has no {self.learner!r}")

    def params(self) -> dict[str, Any]:
        """Every setting of the run: the test, ``alpha``, ``max_depth``, the kind, the graph."""
        return {
            "indep_test": self.indep_test,
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "kind": self.kind,
            STRUCTURE_PARAM: self.structure,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the learner; return :func:`~andrey_bench.blanket_lane.pack_answer`'s array.

        With ``params["targets"]``, each target's set is its row of the answer and each search's
        test count is in the counts, :data:`~andrey_bench.blanket_lane.NO_COUNT` where Andrey
        reports none. Without, the answer is the graph ``params["structure"]`` names, as the
        facade returns it, and the count is the fit's. The only timed region.
        """
        import warnings

        import andrey

        function = getattr(andrey, self.learner)
        settings = {
            "indep_test": str(params.get("indep_test", self.indep_test)),
            "alpha": float(params.get("alpha", self.alpha)),
            "max_depth": params.get("max_depth", self.max_depth),
        }
        X = np.asarray(data, dtype=np.float64)
        d = X.shape[1]
        targets = params.get(TARGETS_PARAM)
        with warnings.catch_warnings():
            # A learner not yet supported warns on every call.
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            if targets is None:
                structure = params.get(STRUCTURE_PARAM, self.structure)
                if structure is None:
                    raise ValueError("a whole-graph run needs params['structure']")
                out = function(X, structure=structure, **settings)
                return pack_answer(np.asarray(out.structure.to_numpy()), None, n_tests(out))
            answer = np.zeros((d, d), dtype=np.int8)
            counts = np.full(d, NO_COUNT, dtype=np.int64)
            for t in targets:
                out = function(X, target=int(t), **settings)
                # The result is the star from the target to each member of its set.
                answer[t] = np.asarray(out.structure.to_numpy())[t] != 0
                found = n_tests(out)
                counts[t] = NO_COUNT if found is None else found
        asked = counts[[int(t) for t in targets]]
        total = None if (asked == NO_COUNT).any() else int(asked.sum())
        return pack_answer(answer, counts, total)

    def to_structure(self, native: np.ndarray) -> Any:
        """The answer as a ``GraphStructure``; untimed.

        A whole-graph answer is the graph it is. The sets of several targets are one undirected
        graph, a pair joined when either target holds the other.
        """
        return answer_structure(native, self.structure)


__all__ = ["LEARNERS", "AndreyBlanket", "Learner", "available", "n_tests"]
