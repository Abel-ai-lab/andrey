"""causal-learn FCI adapter — the reference constraint+latent competitor.

Calls ``fci(X, independence_test_method="fisherz", alpha=0.05, show_progress=False)``. Mind the
kwarg name: ``fci`` takes ``independence_test_method`` where causal-learn's ``pc`` takes
``indep_test``. Unlike the GES adapter this path needs no ``np.mat`` shim — it never imports
``ScoreUtils``. The output is a PAG, so :meth:`fit` emits Andrey endpoint marks directly and
:meth:`to_structure` rebuilds with ``kind="pag"``; ``structure_from_adjacency`` cannot carry a PAG's
circle endpoints.

``pag.graph[a, b]`` is the endpoint mark **at node a** on edge ``a-b``, the same orientation as
Andrey's ``marks[i, j]``, so the translation is a per-entry value remap with **no transpose**
(verified empirically). Only four marks occur in an FCI PAG:

    causal-learn Endpoint   value        Andrey mark   value
    ---------------------   -----        -----------   -----
    NULL (no edge)            0     -->   NULL            0
    TAIL                     -1     -->   TAIL            1
    ARROW                    +1     -->   ARROW           2
    CIRCLE                   +2     -->   CIRCLE          3
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.causal_learn_convert import remap_to_unsigned

# Equalized constraint+latent-lane pins (Fisher-Z CI test at alpha=0.05); runner params override them.
_DEFAULT_ALPHA = 0.05
_DEFAULT_INDEP_TEST = "fisherz"


def _causal_learn_version() -> str:
    """The installed causal-learn version (distribution metadata; the module exposes none)."""
    from importlib import metadata

    try:
        return metadata.version("causal-learn")
    except metadata.PackageNotFoundError:  # pragma: no cover - installed in bench-env
        return "unknown"


class CausalLearnFCI:
    """Solution adapter for causal-learn's FCI (constraint+latent lane).

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.fci"
    package: str = "causal-learn"
    package_version: str = _causal_learn_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "fci"
    output_type: str = "pag"

    def setup(self) -> None:
        """Import causal-learn before timing; see `contracts.SetupAdapter`.

        The adapter is unpickled in the child, and `fit` imports lazily. Setup excludes that
        import from the first timed fit at `--warmup 0`.
        """
        from causallearn.search.ConstraintBased.FCI import fci  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Fisher-Z at ``alpha=0.05`` — causal-learn's own defaults."""
        return {"alpha": _DEFAULT_ALPHA, "indep_test": _DEFAULT_INDEP_TEST}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn FCI; return Andrey's PAG endpoint-mark array. The timed region.

        ``alpha`` / ``indep_test`` come from ``params``, defaulting to ``0.05`` / ``"fisherz"``;
        ``indep_test`` is forwarded under causal-learn's own kwarg name
        ``independence_test_method``. The returned PAG's signed endpoint matrix is remapped to
        Andrey's ``NULL/TAIL/ARROW/CIRCLE`` marks via the table in the module docstring — no
        transpose, since the two conventions align.
        """
        from causallearn.search.ConstraintBased.FCI import fci as _fci

        alpha = float(params.get("alpha", _DEFAULT_ALPHA))
        indep_test = params.get("indep_test", _DEFAULT_INDEP_TEST)

        X = np.asarray(data, dtype=np.float64)
        pag, _edges = _fci(
            X,
            independence_test_method=indep_test,
            alpha=alpha,
            show_progress=False,
        )

        return remap_to_unsigned(pag.graph).astype(np.uint8)

    def to_structure(self, native: np.ndarray) -> Any:
        """PAG endpoint-mark array -> ``GraphStructure`` (``from_numpy``, ``kind="pag"``); untimed.

        Not ``structure_from_adjacency`` — a 0/1 adjacency cannot carry the PAG's circle endpoints.
        """
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
