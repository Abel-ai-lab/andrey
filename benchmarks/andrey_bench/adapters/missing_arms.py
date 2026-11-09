"""The arms of the missing-value lanes: Andrey's ``missing`` options and the baselines.

:class:`AndreyMissing` calls an Andrey method through its public function with ``missing`` set:
``"mvpc"`` on ``pc``, ``"em"`` on the covariance-based searches, or ``None``, the method's default
(test-wise deletion for ``pc``). The same arm on a key without ``missing`` is the complete-data arm.

:class:`Listwise` and :class:`MeanImpute` wrap any solution adapter and change the data before its
fit, so every method of a lane gets the same two baselines:

- listwise deletion drops every row with a missing value; a fit left with fewer than
  :data:`MIN_ROWS` rows raises, so it is a recorded failure and never an empty graph;
- mean imputation fills each continuous column with its observed mean and each categorical column
  (``categorical``) with its observed mode, the smallest value among ties.

The registry entry of a masked dataset records its complete rows, the rows listwise deletion keeps
(:func:`~andrey_bench.datasets.materialize`).

A ``pc`` arm with ``missing="mvpc"`` returns an int8 ``(2, d, d)`` stack: the CPDAG's endpoint marks
and the detected parents of the missingness indicators, ``[1][j, i] = 1`` when column ``j`` is a
parent of column ``i``'s indicator. :meth:`AndreyMissing.detected_parents` hands the second slab to
:func:`~andrey_bench.scoring.missingness_parent_scores`.

References
----------
.. [1] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
   discovery in the presence of missing data. Proceedings of Machine Learning Research 89
   (AISTATS), 1762-1770.
.. [2] Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H. (2022).
   MissDAG: Causal discovery in the presence of missing data with continuous additive noise models.
   Advances in Neural Information Processing Systems 35.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from andrey_bench.adapters.andrey_pc import _andrey_version

#: The fewest complete rows a listwise fit runs on: Fisher's z test of an unconditional pair needs
#: ``n - 3 > 0``. Fewer is a recorded failure.
MIN_ROWS = 4

#: The output kind of each method's public function.
OUTPUT_TYPES = {
    "pc": "cpdag",
    "ges": "cpdag",
    "xges": "cpdag",
    "boss": "cpdag",
    "grasp": "cpdag",
    "hc": "cpdag",
    "tabu": "cpdag",
    "exact_search": "cpdag",
    "notears": "dag",
    "dagma": "dag",
    "calm": "dag",
}


class AndreyMissing:
    """An Andrey method with ``missing`` set, through its public function.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. ``options`` are the method's other keyword
    arguments (``alpha``, ``ebic_gamma``, ``seed``), each part of the arm's name, so
    ``AndreyMissing("ges", "em", ebic_gamma=0.0)`` is ``andrey.ges.em.ebic_gamma0``.
    """

    package: str = "andrey"
    package_version: str = _andrey_version()
    backend: str = "numpy"
    mode: str = "serial"

    def __init__(self, method: str, missing: str | None = None, **options: Any) -> None:
        if method not in OUTPUT_TYPES:
            raise ValueError(f"method must be one of {sorted(OUTPUT_TYPES)}, got {method!r}")
        self.method = method
        self.missing = missing
        self.options = dict(options)
        self.algorithm = method
        self.output_type = OUTPUT_TYPES[method]
        tag = "".join(
            f".{k}{v:g}" if isinstance(v, float) else f".{k}{v}" for k, v in options.items()
        )
        self.name = f"andrey.{method}.{missing or 'default'}{tag}"

    def setup(self) -> None:
        """Import Andrey before timing."""
        import andrey  # noqa: F401

    def params(self) -> dict[str, Any]:
        """``missing`` and the method's other keyword arguments."""
        return {"missing": self.missing, **self.options}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the method on `data`, ``NaN`` where a value is missing; return its endpoint marks.

        ``pc`` with ``missing="mvpc"`` returns the ``(2, d, d)`` stack with the detected parents.
        The methods' ``ExperimentalWarning`` is silenced.
        """
        import warnings

        import andrey

        kwargs = dict(params)
        X = np.asarray(data, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", andrey.ExperimentalWarning)
            out = getattr(andrey, self.method)(X, **kwargs)
        marks = np.asarray(out.structure.to_numpy())
        if self.method != "pc" or kwargs.get("missing") != "mvpc":
            return marks
        found = np.zeros_like(marks, dtype=np.int8)
        for column, parents in (out.metadata.get("missingness_parents") or {}).items():
            for parent in parents:
                found[int(parent), int(column)] = 1
        return np.stack([marks.astype(np.int8), found])

    def to_structure(self, native: np.ndarray) -> Any:
        """Endpoint marks to a ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        marks = np.asarray(native)
        return GraphStructure.from_numpy(
            marks[0] if marks.ndim == 3 else marks, kind=self.output_type
        )

    def detected_parents(self, native: np.ndarray) -> np.ndarray | None:
        """The detected parents of the missingness indicators, or ``None`` for another arm."""
        marks = np.asarray(native)
        return marks[1] if marks.ndim == 3 else None


class _Baseline:
    """A solution adapter run on data that :meth:`prepare` changes first."""

    #: The name's prefix, before the wrapped solution's name.
    prefix: str

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.name = f"{self.prefix}.{inner.name}"
        for field in ("package", "package_version", "backend", "mode", "algorithm", "output_type"):
            setattr(self, field, getattr(inner, field))
        if getattr(inner, "python_env", None):
            self.python_env = inner.python_env

    def setup(self) -> None:
        """The wrapped solution's setup, outside timing."""
        if hasattr(self.inner, "setup"):
            self.inner.setup()

    def params(self) -> dict[str, Any]:
        """The wrapped solution's parameters."""
        return dict(self.inner.params())

    def prepare(self, data: np.ndarray) -> np.ndarray:  # pragma: no cover - subclasses define it
        raise NotImplementedError

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Change the data, then fit the wrapped solution; both are timed."""
        return self.inner.fit(self.prepare(np.asarray(data, dtype=np.float64)), params)

    def to_structure(self, native: np.ndarray) -> Any:
        """The wrapped solution's conversion; untimed."""
        return self.inner.to_structure(native)


class Listwise(_Baseline):
    """Listwise deletion: fit the wrapped solution on the complete rows only."""

    prefix = "listwise"

    def prepare(self, data: np.ndarray) -> np.ndarray:
        """The rows without a missing value.

        Raises
        ------
        ValueError
            If fewer than :data:`MIN_ROWS` rows are complete.
        """
        keep = ~np.isnan(data).any(axis=1)
        kept = int(keep.sum())
        if kept < MIN_ROWS:
            raise ValueError(
                f"listwise deletion kept {kept} of {len(data)} rows, fewer than {MIN_ROWS}"
            )
        return np.ascontiguousarray(data[keep])


class MeanImpute(_Baseline):
    """Mean imputation: each continuous column's mean, each categorical column's mode.

    ``categorical`` marks the categorical columns; ``None`` reads every column as continuous.
    """

    prefix = "impute_mean"

    def __init__(self, inner: Any, categorical: Sequence[bool] | None = None) -> None:
        super().__init__(inner)
        self.categorical = None if categorical is None else [bool(c) for c in categorical]

    def prepare(self, data: np.ndarray) -> np.ndarray:
        """`data` with each missing value filled from its column's observed values.

        Raises
        ------
        ValueError
            If a column has no observed value.
        """
        X = data.copy()
        for j in range(X.shape[1]):
            absent = np.isnan(X[:, j])
            if not absent.any():
                continue
            seen = X[~absent, j]
            if seen.size == 0:
                raise ValueError(f"column {j} has no observed value to impute from")
            if self.categorical is not None and self.categorical[j]:
                values, counts = np.unique(seen, return_counts=True)
                X[absent, j] = values[np.argmax(counts)]
            else:
                X[absent, j] = seen.mean()
        return X


__all__ = ["AndreyMissing", "Listwise", "MeanImpute", "MIN_ROWS", "OUTPUT_TYPES"]
