"""Andrey's Y-structure method, for the lane of ancestral claims under selection bias.

The adapter runs ``andrey.y_structures`` on every column of a dataset, the context in the last one
included, as the authors' code reads them (:mod:`versteeg_ystructures`); the lane scores the claims
among the system columns. Its settings are passed, never defaulted: ``alpha`` and ``indep_test``
(``"fisherz"``, the partial-correlation test of the paper's simulations).

The answer is the ``(d, d)`` matrix of the claims' scores, ``scores[i, j]`` for the claim that
``i`` is an ancestor of ``j`` (:class:`~andrey_bench.local_lane.ClaimAdapter`), and the fit's
``metadata["n_tests"]``. A claim's score is the paper's, which can be ``0`` for an extended
Y-structure whose second variable is independent of the effect; such a claim is written as the
smallest positive number, so the lane counts it.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version
from andrey_bench.local_lane import pack_answer, unpack_answer

#: The settings of the paper's simulations, which the lane passes to every solution.
DEFAULTS = {"alpha": 0.01, "indep_test": "fisherz"}


class AndreyYStructures:
    """``andrey.y_structures`` in its two forms.

    ``extended=False`` is the Y-structure (YSt), ``True`` the extended one (YSt-Ext). Satisfies
    :class:`~andrey_bench.contracts.SolutionAdapter`, :class:`~andrey_bench.contracts.SetupAdapter`,
    and :class:`~andrey_bench.local_lane.ClaimAdapter`.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "y_structures"
    output_type: str = "claims"

    def __init__(self, extended: bool = False) -> None:
        self.extended = extended
        self.name = "andrey.yst_ext" if extended else "andrey.yst"

    def setup(self) -> None:
        """Import Andrey and the engine before timing, so no fit pays for them."""
        import importlib

        import andrey

        importlib.import_module("andrey.constraint.ystructures")
        if not callable(getattr(andrey, "y_structures", None)):
            raise RuntimeError("the installed Andrey has no 'y_structures'")

    def params(self) -> dict[str, Any]:
        """The settings of the paper's simulations (:data:`DEFAULTS`)."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the method on ``data``, the context in the last column; return the claims' scores
        and the test count. The only timed region."""
        import warnings

        import andrey

        settings = {**DEFAULTS, **params}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.y_structures(
                np.asarray(data, dtype=np.float64),
                extended=self.extended,
                alpha=float(settings["alpha"]),
                indep_test=str(settings["indep_test"]),
            )
        d = np.asarray(data).shape[1]
        scores = np.zeros((d, d))
        for record in out.metadata["effects"]:
            score = max(float(record["score"]), np.nextafter(0.0, 1.0))
            scores[int(record["treatment"]), int(record["outcome"])] = score
        return pack_answer(scores, out.metadata.get("n_tests"))

    def to_structure(self, native: np.ndarray) -> Any:
        """The empty graph over the columns: a claim is ancestry, not an edge; untimed."""
        from andrey import GraphStructure

        d = unpack_answer(native)[0].shape[0]
        return GraphStructure.from_numpy(np.zeros((d, d), dtype=np.int8), kind="digraph")
