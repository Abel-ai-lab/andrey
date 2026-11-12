"""Andrey's MMB-by-MMB, for the lane of local PAGs at a target under hidden variables.

The adapter runs ``andrey.mmb_by_mmb`` at the target the lane names (``params["target"]``) on the
dataset's observed columns. Its settings are passed, never defaulted: ``alpha``, ``indep_test``
(``"fisherz"``, the test of the paper's simulations), and ``blanket``, the blanket learner
(``"total_conditioning"``, the paper's, or a grow-shrink learner as a second arm).

The answer is the ``(d, d)`` matrix of endpoint marks at the target, ``marks[i, j]`` the mark at
``i`` (:class:`~andrey_bench.local_lane.LocalAdapter`), and the fit's ``metadata["n_tests"]``.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version
from andrey_bench.local_lane import TARGET_PARAM, pack_answer, unpack_answer

#: The settings of the paper's simulations, which the lane passes to every solution.
DEFAULTS = {"alpha": 0.01, "indep_test": "fisherz"}


class AndreyMMBbyMMB:
    """``andrey.mmb_by_mmb`` with one blanket learner.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and :class:`~andrey_bench.local_lane.LocalAdapter`.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "mmb_by_mmb"
    output_type: str = "pag"
    local: bool = True

    def __init__(self, blanket: str = "total_conditioning") -> None:
        self.blanket = blanket
        suffix = "" if blanket == "total_conditioning" else f".{blanket}"
        self.name = "andrey.mmb_by_mmb" + suffix

    def setup(self) -> None:
        """Import Andrey and the engine before timing, so no fit pays for them."""
        import importlib

        import andrey

        importlib.import_module("andrey.constraint.sequential")
        if not callable(getattr(andrey, "mmb_by_mmb", None)):
            raise RuntimeError("the installed Andrey has no 'mmb_by_mmb'")

    def params(self) -> dict[str, Any]:
        """The settings of the paper's simulations (:data:`DEFAULTS`) and the blanket learner."""
        return {**DEFAULTS, "blanket": self.blanket}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the method at ``params["target"]``; return the packed marks at the target and the
        test count. The only timed region."""
        import warnings

        import andrey

        settings = {**self.params(), **params}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.mmb_by_mmb(
                np.asarray(data, dtype=np.float64),
                target=int(settings[TARGET_PARAM]),
                blanket=str(settings["blanket"]),
                alpha=float(settings["alpha"]),
                indep_test=str(settings["indep_test"]),
            )
        return pack_answer(out.structure.to_numpy(), out.metadata.get("n_tests"))

    def to_structure(self, native: np.ndarray) -> Any:
        """The marks at the target as a PAG ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(unpack_answer(native)[0], kind=self.output_type)
