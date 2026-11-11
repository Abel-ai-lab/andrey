"""Andrey NOTEARS and DAGMA solution adapter.

:class:`AndreyContinuous` calls ``andrey.notears`` or ``andrey.dagma`` through its public function
at the defaults, with the options named switched on: ``AndreyContinuous("notears")`` is
``andrey.notears``, ``AndreyContinuous("dagma", "standardize")`` is ``andrey.dagma.standardize``,
and ``"robust"`` adds the likelihood path. :mod:`~andrey_bench.adapters.zheng_notears` and
:mod:`~andrey_bench.adapters.bello_dagma` are the references, at the same defaults.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version

METHODS = ("notears", "dagma")
#: The boolean options an arm can switch on.
OPTIONS = ("standardize", "robust")


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
        if unknown := sorted(set(options) - set(OPTIONS)):
            raise ValueError(f"options must be among {OPTIONS}, got {unknown}")
        self.method = method
        self.options = tuple(options)
        self.algorithm = method
        self.name = ".".join(("andrey", method, *options))

    def setup(self) -> None:
        """Import Andrey before timing."""
        import andrey  # noqa: F401

    def params(self) -> dict[str, Any]:
        """The options switched on; every other argument is the method's default."""
        return {option: True for option in self.options}

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
