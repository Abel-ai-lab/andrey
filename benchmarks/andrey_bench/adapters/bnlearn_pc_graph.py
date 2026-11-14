"""bnlearn's ``pc.stable`` CPDAG on categorical data, for the lane of target-centered methods.

The call is :class:`~andrey_bench.adapters.bnlearn_blanket.BnlearnBlanket`'s whole-graph run of
``pc.stable`` with the test ``mi-adf`` (Andrey's ``gsq`` with the sparse-table rule), ``alpha``, and
``max_depth`` as bnlearn's ``max.sx``. The lane reads the CPDAG at every target.

Two things differ from the blanket adapter:

- bnlearn rejects a factor with one level, so the columns that hold one value stay out of the fit
  and have no edges in the answer.
- The answer is packed as :func:`~andrey_bench.local_lane.pack_answer` does: the CPDAG's endpoint
  marks and the fit's test count.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench import blanket_lane
from andrey_bench.adapters.bnlearn_blanket import BnlearnBlanket
from andrey_bench.local_lane import pack_answer, unpack_answer


class BnlearnPCStableGraph(BnlearnBlanket):
    """bnlearn 5.2.1 ``pc.stable`` for the whole graph, over the columns that vary.

    Not local: the driver fits it once per dataset and reads every target's row and column.
    """

    local: bool = False

    def __init__(self, *, alpha: float = 0.05, max_depth: int | None = None) -> None:
        super().__init__(
            "pc.stable", indep_test="gsq", alpha=alpha, max_depth=max_depth, structure="cpdag"
        )

    def params(self) -> dict[str, Any]:
        """The settings of :class:`BnlearnBlanket`, and that columns of one value are left out."""
        return {**super().params(), "drop_constant": True}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run ``pc.stable`` on the columns that hold more than one value; return the packed
        marks of the CPDAG over every column and the fit's test count. The only timed region."""
        data = np.asarray(data)
        d = data.shape[1]
        keep = np.flatnonzero((data != data[0]).any(axis=0))
        marks = np.zeros((d, d), dtype=np.int8)
        total = 0
        if keep.size > 1:
            native = super().fit(np.ascontiguousarray(data[:, keep]), params)
            fitted = blanket_lane.answer_structure(native, "cpdag").to_numpy()
            marks[np.ix_(keep, keep)] = fitted
            total = blanket_lane.unpack_answer(native)[2]
        return pack_answer(marks, total)

    def to_structure(self, native: np.ndarray) -> Any:
        """The CPDAG's marks as a ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(unpack_answer(native)[0], kind=self.output_type)
