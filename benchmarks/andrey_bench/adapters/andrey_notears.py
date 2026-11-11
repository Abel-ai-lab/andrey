"""Andrey NOTEARS and DAGMA solution adapter.

:class:`AndreyContinuous` calls ``andrey.notears`` or ``andrey.dagma`` through its public function
at the defaults, with the options named switched on: ``AndreyContinuous("notears")`` is
``andrey.notears``, ``AndreyContinuous("dagma", "standardize")`` is ``andrey.dagma.standardize``,
and ``"robust"`` adds the likelihood path. ``AndreyContinuous("dagma", "paper")`` is
``andrey.dagma.paper``, the fit at :data:`PAPER_LAMBDA1`, the penalty of the DAGMA paper's
experiments. :mod:`~andrey_bench.adapters.zheng_notears` and
:mod:`~andrey_bench.adapters.bello_dagma` are the references, at the same settings.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version

METHODS = ("notears", "dagma")
#: The boolean options an arm can switch on.
OPTIONS = ("standardize", "robust")
#: The ``l1`` penalty of the experiments of Bello et al. (2022); the package default is 0.03.
PAPER_LAMBDA1 = 0.05
#: The arm suffix that sets DAGMA's ``lambda1`` to :data:`PAPER_LAMBDA1`.
PAPER = "paper"


class AndreyContinuous:
    """Andrey's ``notears`` or ``dagma``, with each of `options` set to ``True``.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"
    output_type: str = "dag"

    def __init__(self, method: str, *options: str) -> None:
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        allowed = (*OPTIONS, PAPER) if method == "dagma" else OPTIONS
        if unknown := sorted(set(options) - set(allowed)):
            raise ValueError(f"options of {method} must be among {allowed}, got {unknown}")
        self.method = method
        self.options = tuple(options)
        self.algorithm = method
        self.name = ".".join(("andrey", method, *options))

    def setup(self) -> None:
        """Import Andrey before timing."""
        import andrey  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The options switched on, and ``lambda1`` for the paper's penalty; every other argument
        is the method's default.
        """
        return {
            **{option: True for option in self.options if option != PAPER},
            **({"lambda1": PAPER_LAMBDA1} if PAPER in self.options else {}),
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the method and return the DAG's endpoint marks. The only timed region.

        The methods' ``ExperimentalWarning`` is silenced.
        """
        import warnings

        import andrey

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = getattr(andrey, self.method)(np.asarray(data, dtype=np.float64), **params)
        return np.asarray(out.structure.to_numpy())

    def to_structure(self, native: np.ndarray) -> Any:
        """Endpoint marks to a DAG ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(np.asarray(native), kind=self.output_type)
