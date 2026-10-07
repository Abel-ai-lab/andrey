"""R matrix encodings -> the arrays the benchmark scores.

The R adapters use three encodings, two from the same package:

* `bnlearn::amat` - 0/1 with `amat[i, j] = 1` for `i -> j`; an undirected arc appears both ways.
* pcalg `amat.cpdag` - 0/1 with the edge written at the *child's* row, the transpose of the above.
* pcalg `amat.pag` - 0 / 1 / 2 / 3 for none / circle / arrowhead / tail, and the mark in
  `amat[i, j]` belongs to node `j`, the opposite end from Andrey's `marks[i, j]`.

The R entry points normalize the two 0/1 forms. This module decodes PAGs.
`tests/test_r_encodings.py` checks all three against known graphs.
"""

from __future__ import annotations

import numpy as np

# Andrey endpoint marks, as in `contracts.py`: `marks[i, j]` is the mark at `i` on edge `i-j`.
_NULL, _TAIL, _ARROW, _CIRCLE = 0, 1, 2, 3

#: pcalg `amat.pag` endpoint codes in package order, mapped to Andrey marks.
PAG_CODE_TO_MARK: dict[int, int] = {0: _NULL, 1: _CIRCLE, 2: _ARROW, 3: _TAIL}


def marks_from_amat_pag(amat: np.ndarray) -> np.ndarray:
    """Convert pcalg `amat.pag` to Andrey endpoint marks.

    pcalg stores the mark at node `j` in `amat[i, j]`; Andrey stores the mark at node `i`
    in `marks[i, j]`. Conversion transposes the matrix and remaps the codes. For the collider
    edge `0 o-> 2`, pcalg has `amat[0, 2] = 2` (arrowhead at 2) and `amat[2, 0] = 1`
    (circle at 0); Andrey has `marks[2, 0] = ARROW` and `marks[0, 2] = CIRCLE`.
    """
    a = np.asarray(amat)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"amat must be square 2-D, got {a.shape}")
    codes = np.rint(a).astype(np.int64)
    unknown = set(np.unique(codes)) - set(PAG_CODE_TO_MARK)
    if unknown:
        raise ValueError(f"amat.pag carries codes {sorted(unknown)}; expected 0, 1, 2 or 3")
    marks = np.zeros(a.shape, dtype=np.uint8)
    for code, mark in PAG_CODE_TO_MARK.items():
        marks[codes.T == code] = mark
    np.fill_diagonal(marks, _NULL)
    return marks
