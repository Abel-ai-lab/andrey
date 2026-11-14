"""Andrey's target-centered methods and its PC, for the lane of categorical networks.

:class:`AndreyTarget` runs ``andrey.mb_by_mb`` or ``andrey.cmb`` at the target the lane names
(``params["target"]``). Its settings are passed, never defaulted: ``alpha``, ``indep_test``
(``"gsq"``, the G-square test both papers run on categorical data), and ``blanket``, the Markov
blanket learner. ``max_depth`` is not passed: each method keeps its own cap, MB-by-MB's
``local_max_depth`` of 5 on the local structures and none for CMB, because a cap on the blanket
learner stops its grow phase before the blanket is whole.

``AndreyPCCounted`` is ``andrey.pc`` with ``gsq``, as ``AndreyPCGsq`` runs it, with the
conditioning-set cap passed and the tests counted. It learns the whole CPDAG once per dataset,
which the lane reads at every target.

Both pack the marks and the test count as :func:`~andrey_bench.local_lane.pack_answer` does.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import AndreyPCGsq, _andrey_version
from andrey_bench.local_lane import TARGET_PARAM, pack_answer, unpack_answer

#: The method's own Markov blanket learner, the paper's.
DEFAULT_BLANKET = {"mb_by_mb": "iamb_forward", "cmb": "hiton_mb"}

#: The level of each paper's runs.
DEFAULT_ALPHA = {"mb_by_mb": 0.05, "cmb": 0.02}

#: The module each method's engine lives in, imported in ``setup`` so no fit pays for it.
_ENGINE = {"mb_by_mb": "andrey.constraint.sequential", "cmb": "andrey.constraint.cmb"}


class AndreyTarget:
    """``andrey.mb_by_mb`` or ``andrey.cmb`` with one blanket learner, fitted once per target.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter`,
    :class:`~andrey_bench.contracts.SetupAdapter`, and
    :class:`~andrey_bench.local_lane.LocalAdapter`. The answer is the ``(d, d)`` marks at the
    target, ``marks[i, j]`` the mark at ``i``, and the fit's ``metadata["n_tests"]``.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    output_type: str = "cpdag"
    local: bool = True

    def __init__(self, method: str, blanket: str | None = None) -> None:
        if method not in DEFAULT_BLANKET:
            raise ValueError(f"method must be one of {sorted(DEFAULT_BLANKET)}, got {method!r}")
        self.method = method
        self.algorithm = method
        self.blanket = DEFAULT_BLANKET[method] if blanket is None else blanket
        suffix = "" if self.blanket == DEFAULT_BLANKET[method] else f".{self.blanket}"
        self.name = f"andrey.{method}{suffix}"

    def setup(self) -> None:
        """Import Andrey and the method's engine before timing, so no fit pays for them."""
        import importlib

        import andrey

        importlib.import_module(_ENGINE[self.method])
        if not callable(getattr(andrey, self.method, None)):
            raise RuntimeError(f"the installed Andrey has no {self.method!r}")

    def params(self) -> dict[str, Any]:
        """The settings of the method's paper: its level, the ``gsq`` test, and the blanket."""
        return {
            "alpha": DEFAULT_ALPHA[self.method],
            "indep_test": "gsq",
            "blanket": self.blanket,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the method at ``params["target"]``; return the packed marks at the target and the
        test count. The only timed region."""
        import warnings

        import andrey

        settings = {**self.params(), **params}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = getattr(andrey, self.method)(
                np.asarray(data, dtype=np.float64),
                target=int(settings[TARGET_PARAM]),
                blanket=str(settings["blanket"]),
                alpha=float(settings["alpha"]),
                indep_test=str(settings["indep_test"]),
            )
        return pack_answer(out.structure.to_numpy(), out.metadata.get("n_tests"))

    def to_structure(self, native: np.ndarray) -> Any:
        """The marks at the target as a CPDAG ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(unpack_answer(native)[0], kind=self.output_type)


class AndreyPCCounted(AndreyPCGsq):
    """``andrey.pc`` with ``gsq``, the conditioning-set cap passed and the tests counted.

    The solution of :class:`~andrey_bench.adapters.andrey_pc.AndreyPCGsq`, so it keeps its name.
    Not local: the driver fits it once per dataset and reads every target's row and column.
    """

    local: bool = False

    def params(self) -> dict[str, Any]:
        """The ``gsq`` test at ``alpha=0.05`` and no cap on the conditioning sets."""
        return {**super().params(), "max_depth": None}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run Andrey PC; return the packed CPDAG marks and the number of tests. The timed region."""
        import warnings

        import andrey
        from andrey.core.ci import counting_tests

        settings = {**self.params(), **params}
        max_depth = settings["max_depth"]
        with warnings.catch_warnings(), counting_tests() as tests:
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = andrey.pc(
                np.asarray(data, dtype=np.float64),
                alpha=float(settings["alpha"]),
                indep_test=str(settings["indep_test"]),
                max_depth=None if max_depth is None else int(max_depth),
            )
        return pack_answer(out.structure.to_numpy(), tests.n)

    def to_structure(self, native: np.ndarray) -> Any:
        """The CPDAG's marks as a ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(unpack_answer(native)[0], kind=self.output_type)
