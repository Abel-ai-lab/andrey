"""pgmpy's hill climbing on categorical data, with its BDeu score or its multinomial BIC.

``HillClimbSearch`` adds, removes, and reverses one edge at a time from the empty graph while the
score improves, and returns a DAG. The adapter is written against pgmpy 1.1.2, which installs on
Python 3.13 with NumPy 2.5 and pandas 3. Its search is set to match Andrey's ``hc`` with the same
score:

- ``tabu_length=0`` turns off its list of recently changed edges, so the search is plain hill
  climbing, as Andrey's is. pgmpy's default list holds 100.
- ``max_indegree=None`` caps no node's parents, as Andrey's ``hc`` has no cap.
- ``max_iter`` is the accepted-move budget of Andrey's adapter, high enough never to bind.
- ``epsilon=1e-4``, pgmpy's default, ends the search when the best move gains less than that in
  log-likelihood units. Andrey's threshold is ``1e-9`` in deviance units, twice the scale. Both are
  far below the gain of any real move.

``score_func="local_score_BIC_categorical"`` is pgmpy's ``BIC``, the multinomial BIC with penalty
``log(n) / 2`` per parameter, Andrey's score at ``lambda_value=1``: Andrey's scores are ``-2`` times
pgmpy's. pgmpy's penalty has no weight, so any other ``lambda_value`` is refused.

``score_func="local_score_BDeu"`` is pgmpy's ``BDeu`` with ``equivalent_sample_size=1``, where its
default is 10, and Andrey's structure prior, which pgmpy's has none of: each other variable is a
parent with probability ``p = structure_prior / (d - 1)``. The adapter adds the prior
through the hook pgmpy gives a score for it, ``structure_prior_ratio``; adding an edge moves the
log prior by ``log(p / (1 - p))``, removing one by the negative, and reversing one by nothing. Its
scores are then ``-1/2`` times Andrey's.

The columns are named by their integer position. pgmpy breaks ties between equal moves by the
iteration order of a set of column-name pairs; with string names that order changes with
``PYTHONHASHSEED``, and the graph with it, while integer names fix it. Each fit builds its own
score, so no local score from an earlier fit can answer a later one.
"""

from __future__ import annotations

import functools
from math import log
from typing import Any, Mapping

import numpy as np

#: The release the adapter is written against and the bench venv installs (``start.sh``).
PINNED_VERSION = "1.1.2"

_BIC = "local_score_BIC_categorical"
_BDEU = "local_score_BDeu"
#: Andrey's BDeu priors, BIC weight, and move budget, the values of its hill climbing; a test
#: compares them with the ones the Andrey adapter declares.
_DEFAULT_SAMPLE_PRIOR = 1.0
_DEFAULT_STRUCTURE_PRIOR = 1.0
_DEFAULT_LAMBDA = 1.0
_DEFAULT_MAX_ITER = 100_000


def _pgmpy_version() -> str:
    """The installed pgmpy version, from its distribution metadata."""
    from importlib import metadata

    try:
        return metadata.version("pgmpy")
    except metadata.PackageNotFoundError:  # installed in the bench venv only
        return "unknown"


def check_environment() -> dict[str, str]:
    """Report the installed pgmpy version, or stop the run when it is not the pinned one.

    A driver calls it before the first fit, so a missing or different pgmpy fails once instead of
    as an error record for every unit.
    """
    version = _pgmpy_version()
    if version != PINNED_VERSION:
        found = "is not installed" if version == "unknown" else f"is {version}"
        raise SystemExit(
            f"the pgmpy solutions are written for pgmpy {PINNED_VERSION}, and pgmpy {found}: "
            f"run benchmarks/start.sh or `uv pip install pgmpy=={PINNED_VERSION}`"
        )
    return {"pgmpy": version}


@functools.cache
def _prior_bdeu_class() -> type:
    """pgmpy's ``BDeu`` plus the structure prior of Andrey's ``BDeuScore``.

    The prior gives each of the other ``d - 1`` variables probability ``p = expected_parents / (d -
    1)`` of being a parent of a node, ``k log p + (d - 1 - k) log(1 - p)`` for ``k`` parents. It is
    left out when ``p`` is not below 1, as Andrey's is. Built on first use, so importing this module
    does not import pgmpy.
    """
    from pgmpy.structure_score import BDeu

    class PriorBDeu(BDeu):
        def __init__(
            self,
            data: Any,
            equivalent_sample_size: float = _DEFAULT_SAMPLE_PRIOR,
            expected_parents: float = _DEFAULT_STRUCTURE_PRIOR,
            state_names: Any = None,
        ) -> None:
            self.expected_parents = expected_parents
            super().__init__(data, equivalent_sample_size, state_names)

        def _parent_probability(self) -> float:
            others = len(self.variables) - 1
            return self.expected_parents / others if others > 0 else 1.0

        def structure_prior(self, model: Any) -> float:
            p = self._parent_probability()
            if p >= 1.0:
                return 0.0
            edges, d = len(model.edges()), len(self.variables)
            return edges * log(p) + (d * (d - 1) - edges) * log(1.0 - p)

        def structure_prior_ratio(self, operation: str) -> float:
            p = self._parent_probability()
            odds = log(p) - log(1.0 - p) if p < 1.0 else 0.0
            return {"+": odds, "-": -odds}.get(operation, 0.0)

    return PriorBDeu


class PgmpyHC:
    """pgmpy hill climbing with a categorical score; returns a DAG.

    ``score_func`` is Andrey's name for the score: ``"local_score_BIC_categorical"`` (the default)
    or ``"local_score_BDeu"``. Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    package: str = "pgmpy"
    package_version: str = _pgmpy_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "hc"
    output_type: str = "dag"

    def __init__(self, score_func: str = _BIC) -> None:
        if score_func not in (_BIC, _BDEU):
            raise ValueError(f"score_func must be {_BIC!r} or {_BDEU!r}, got {score_func!r}")
        self.score_func = score_func
        self.name = f"pgmpy.hc.{score_func.removeprefix('local_score_')}"

    def setup(self) -> None:
        """Import pgmpy before timing; see `contracts.SetupAdapter`."""
        import pgmpy.causal_discovery  # noqa: F401
        import pgmpy.structure_score  # noqa: F401

        _prior_bdeu_class()

    def params(self) -> dict[str, Any]:
        """The score in Andrey's name, its priors or BIC weight, the move budget, no parent cap."""
        params: dict[str, Any] = {
            "score_func": self.score_func,
            "max_iter": _DEFAULT_MAX_ITER,
            "max_parents": None,
        }
        if self.score_func == _BDEU:
            params["sample_prior"] = _DEFAULT_SAMPLE_PRIOR
            params["structure_prior"] = _DEFAULT_STRUCTURE_PRIOR
        else:
            params["lambda_value"] = _DEFAULT_LAMBDA
        return params

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run HillClimbSearch; return its 0/1 DAG adjacency. The only timed region."""
        import pandas as pd
        from pgmpy.causal_discovery import HillClimbSearch
        from pgmpy.structure_score import BIC

        if params.get("score_func", self.score_func) != self.score_func:
            raise ValueError(f"{self.name} runs score_func={self.score_func!r} only")
        if params.get("max_parents") is not None:
            raise ValueError("pgmpy.hc runs with no parent cap, as Andrey's hc does")
        if float(params.get("lambda_value", _DEFAULT_LAMBDA)) != _DEFAULT_LAMBDA:
            raise ValueError("pgmpy's BIC has a fixed penalty, log(n) / 2 per parameter")
        x = np.asarray(data)
        d = x.shape[1]
        frame = pd.DataFrame(x.astype(np.int64))
        if self.score_func == _BDEU:
            score = _prior_bdeu_class()(
                frame,
                equivalent_sample_size=float(params.get("sample_prior", _DEFAULT_SAMPLE_PRIOR)),
                expected_parents=float(params.get("structure_prior", _DEFAULT_STRUCTURE_PRIOR)),
            )
        else:
            score = BIC(frame)
        search = HillClimbSearch(
            scoring_method=score,
            tabu_length=0,
            max_indegree=None,
            max_iter=int(params.get("max_iter", _DEFAULT_MAX_ITER)),
            return_type="dag",
            show_progress=False,
        )
        dag = search.fit(frame).causal_graph_
        adjacency = np.zeros((d, d), dtype=np.int8)
        for u, v in dag.edges():
            adjacency[u, v] = 1
        return adjacency

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
