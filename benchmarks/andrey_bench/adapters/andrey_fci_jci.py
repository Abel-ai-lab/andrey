"""Andrey FCI-JCI solution adapter for data pooled from several environments.

Calls ``andrey.fci_jci(X, env, alpha=..., indep_test=..., confounded_index=...)``. The data carry the
environment as one more column, which ``params["context_vars"]`` names, as for the reference
packages' FCI-JCI; ``fit`` takes that column out as ``env`` and returns the marks over every column,
the environment included, rebuilt from ``metadata["index_edges"]``: an arrowhead at the variable,
and at the environment a tail (``directed``), a circle (``partially_directed``), or an arrowhead
(``bidirected``). With ``indep_test="cond_gauss"`` the data columns are continuous and the mixed
test reads the environment as categorical.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_cdnod import _andrey_version

_DEFAULTS = {"alpha": 0.05, "indep_test": "fisherz", "confounded_index": True}

# Andrey's endpoint marks: the mark at node i on the edge i-j is marks[i, j].
_TAIL, _ARROW, _CIRCLE = 1, 2, 3
_INDEX_MARK = {"directed": _TAIL, "partially_directed": _CIRCLE, "bidirected": _ARROW}


def with_index(pag: np.ndarray, index_edges: list, at: int) -> np.ndarray:
    """The marks over the data variables and the environment, the environment at column `at`."""
    d = pag.shape[0]
    columns = [j for j in range(d + 1) if j != at]
    marks = np.zeros((d + 1, d + 1), dtype=np.int8)
    marks[np.ix_(columns, columns)] = pag
    for j, kind in index_edges:
        marks[columns[int(j)], at], marks[at, columns[int(j)]] = _ARROW, _INDEX_MARK[kind]
    return marks


class AndreyFCIJCI:
    """Solution adapter for Andrey's FCI-JCI.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "andrey.fci_jci.numpy"
    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "fci_jci"
    output_type: str = "pag"

    def setup(self) -> None:
        """Import Andrey and its FCI-JCI implementation before timing."""
        import andrey  # noqa: F401
        from andrey.constraint.fci_jci import fci_jci  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Andrey's defaults, and no environment column until ``context_vars`` names one."""
        return {**_DEFAULTS, "context_vars": []}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey FCI-JCI; return the marks over every column. The only timed region.

        Raises ``ValueError`` unless ``context_vars`` names exactly one column, the environment.
        """
        import warnings

        import andrey

        settings = {**_DEFAULTS, **params}
        context = [int(j) for j in settings.get("context_vars") or ()]
        if len(context) != 1:
            raise ValueError(f"{self.name} takes one environment column, got {context}")
        x = np.asarray(data, dtype=np.float64)
        at = context[0]
        env = x[:, at]
        X = np.delete(x, at, axis=1)
        mixed = settings["indep_test"] == "cond_gauss"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.fci_jci(
                X,
                env,
                alpha=float(settings["alpha"]),
                indep_test=settings["indep_test"],
                confounded_index=bool(settings["confounded_index"]),
                categorical=[] if mixed else None,
            )
        return with_index(np.asarray(out.structure.to_numpy()), out.metadata["index_edges"], at)

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG over every column from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)
