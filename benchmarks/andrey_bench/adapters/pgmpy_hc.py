"""pgmpy's hill climbing on categorical data, with its multinomial BIC.

``HillClimbSearch`` adds, removes, and reverses one edge at a time from the empty graph while the
score improves, and returns a DAG. ``tabu_length=0`` turns off its list of recently changed edges,
so the search is plain hill climbing, as Andrey's ``hc`` is. Its ``BIC`` score is the multinomial
BIC, Andrey's ``score_func="local_score_BIC_categorical"`` at ``lambda_value=1``: Andrey's scores
are ``-2`` times pgmpy's. pgmpy stops when the best move improves its score by less than
``epsilon=1e-4``, its default.

pgmpy memoizes local scores by column name across fits in one process, so each fit names its
columns afresh; otherwise a warm-up fit's scores would answer the timed fit on other data.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping

import numpy as np

_SCORE_FUNC = "local_score_BIC_categorical"
_FITS = itertools.count()


def _pgmpy_version() -> str:
    """The installed pgmpy version, from its distribution metadata."""
    from importlib import metadata

    try:
        return metadata.version("pgmpy")
    except metadata.PackageNotFoundError:  # installed in the bench venv only
        return "unknown"


class PgmpyHC:
    """pgmpy hill climbing with the multinomial BIC; returns a DAG.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "pgmpy.hc.BIC_categorical"
    package: str = "pgmpy"
    package_version: str = _pgmpy_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "hc"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import pgmpy before timing; see `contracts.SetupAdapter`."""
        import pgmpy.causal_discovery  # noqa: F401
        import pgmpy.structure_score  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The score, in Andrey's name."""
        return {"score_func": _SCORE_FUNC}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run HillClimbSearch; return its 0/1 DAG adjacency. The only timed region."""
        import pandas as pd
        from pgmpy.causal_discovery import HillClimbSearch
        from pgmpy.structure_score import BIC

        if params.get("score_func", _SCORE_FUNC) != _SCORE_FUNC:
            raise ValueError(f"pgmpy.hc runs score_func={_SCORE_FUNC!r} only")
        x = np.asarray(data)
        d = x.shape[1]
        prefix = f"f{next(_FITS)}_"
        frame = pd.DataFrame(x.astype(np.int64), columns=[f"{prefix}{j}" for j in range(d)])
        search = HillClimbSearch(
            scoring_method=BIC(frame), tabu_length=0, return_type="dag", show_progress=False
        )
        dag = search.fit(frame).causal_graph_
        adjacency = np.zeros((d, d), dtype=np.int8)
        for u, v in dag.edges():
            adjacency[int(u.removeprefix(prefix)), int(v.removeprefix(prefix))] = 1
        return adjacency

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
