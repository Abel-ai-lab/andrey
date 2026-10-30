"""Multiple-testing correction of a family of p-values.

A method that reports one p-value per candidate link can correct the family for the number of links
tested. ``fdr_method="fdr_bh"`` replaces each p-value with its Benjamini-Hochberg adjusted value,
the name and procedure of tigramite's ``fdr_method``; ``"none"`` leaves the p-values as tested.

References
----------
Benjamini, Y., and Hochberg, Y. (1995). Controlling the false discovery rate: a practical and
powerful approach to multiple testing. Journal of the Royal Statistical Society B 57(1), 289-300.
"""

from __future__ import annotations

import numpy as np

#: The corrections ``fdr_method`` names: none, and the Benjamini-Hochberg step-up procedure.
FDR_METHODS: tuple[str, ...] = ("none", "fdr_bh")


def check_fdr_method(fdr_method: str) -> None:
    """Raise :class:`ValueError` when ``fdr_method`` is not in :data:`FDR_METHODS`."""
    if fdr_method not in FDR_METHODS:
        raise ValueError(f"fdr_method must be one of {list(FDR_METHODS)}, got {fdr_method!r}")


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """The Benjamini-Hochberg adjusted p-values of the 1-D ``p_values``, in their order.

    The ``k``-th smallest of ``m`` p-values becomes the smallest, over the ``k``-th to the largest,
    of ``p * m / rank``, at most ``1``; equal p-values get equal results. Declaring the tests with
    an adjusted p-value at most ``q`` significant holds the expected share of false ones among them
    at or below ``q`` for independent or positively dependent tests (Benjamini and Hochberg 1995).
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = p_values[order] / (np.arange(1, m + 1) / float(m))
    adjusted = np.minimum(np.minimum.accumulate(adjusted[::-1])[::-1], 1.0)
    corrected = np.empty_like(adjusted)
    corrected[order] = adjusted
    return corrected


def corrected_pvalues(
    p_values: np.ndarray, fdr_method: str, excluded: np.ndarray | None = None
) -> np.ndarray:
    """The ``p_values`` array corrected for the number of tests in it.

    ``"none"`` returns ``p_values`` itself. ``"fdr_bh"`` returns a copy whose entries are replaced
    with their :func:`benjamini_hochberg` adjustment over all the entries together, as tigramite's
    ``get_corrected_pvalues`` does. An entry ``excluded`` (a mask of the same shape) is no test:
    it is not counted and keeps its value. Raises :class:`ValueError` for another ``fdr_method``.
    """
    check_fdr_method(fdr_method)
    if fdr_method == "none":
        return p_values
    tested = np.ones(p_values.shape, dtype=bool) if excluded is None else ~excluded
    corrected = p_values.copy()
    corrected[tested] = benjamini_hochberg(p_values[tested])
    return corrected
