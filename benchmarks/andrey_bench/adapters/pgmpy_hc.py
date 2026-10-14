"""pgmpy hill-climbing adapter for categorical data.

`HillClimbSearch` adds, removes, and reverses one edge at a time while the score improves, from the
empty graph, and returns a DAG. `tabu_length=0` turns off its list of recently changed edges, so
the search is plain hill climbing, as Andrey's `hc` is. `score="bic"` is pgmpy's multinomial BIC,
Andrey's `local_score_BIC_discrete` at `lambda_value=1` up to its factor of `-2`; `score="bdeu"` is
BDeu with equivalent sample size 1 and no structure prior, unlike Andrey's.

pgmpy memoizes local scores by column name across calls in one process, so each fit names its
columns afresh; otherwise a warm-up fit's scores would be reused by the timed fit.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping

import numpy as np

_DEFAULT_SCORE = "bic"
_CALLS = itertools.count()


def _pgmpy_version() -> str:
    """The installed pgmpy version (distribution metadata)."""
    from importlib import metadata

    try:
        return metadata.version("pgmpy")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class PgmpyHC:
    """Solution adapter for pgmpy's hill climbing on categorical data.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "pgmpy.hc"
    package: str = "pgmpy"
    package_version: str = _pgmpy_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "hc"
    output_type: str = "dag"

    def setup(self) -> None:
        """Import pgmpy before timing; see `contracts.SetupAdapter`."""
        import pgmpy.estimators  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The multinomial BIC."""
        return {"score": _DEFAULT_SCORE}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run pgmpy HillClimbSearch; return a 0/1 DAG adjacency. The only timed region."""
        import pandas as pd
        from pgmpy.estimators import BIC, BDeu, HillClimbSearch

        x = np.asarray(data)
        d = x.shape[1]
        prefix = f"c{next(_CALLS)}_"
        frame = pd.DataFrame(x.astype(np.int64), columns=[f"{prefix}{j}" for j in range(d)])
        score = str(params.get("score", _DEFAULT_SCORE))
        scorer = BDeu(frame, equivalent_sample_size=1) if score == "bdeu" else BIC(frame)
        dag = HillClimbSearch(frame).estimate(
            scoring_method=scorer, tabu_length=0, show_progress=False
        )
        adjacency = np.zeros((d, d), dtype=np.int8)
        for u, v in dag.edges():
            adjacency[int(u.removeprefix(prefix)), int(v.removeprefix(prefix))] = 1
        return adjacency

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 DAG adjacency -> Andrey ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)
