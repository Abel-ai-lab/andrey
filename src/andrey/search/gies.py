"""Greedy Interventional Equivalence Search over the linear-Gaussian BIC score.

Interventional generalization of Greedy Equivalence Search (Hauser & Bühlmann 2012). Each target's
local BIC is evaluated only over the rows where that target was not intervened on; with no
interventions every row counts for every target, so the objective, the forward/backward operator
search, and the resulting CPDAG coincide exactly with plain Greedy Equivalence Search. This module
covers that observational case and delegates the search to :func:`andrey.search.ges.ges`.
"""

from __future__ import annotations

import numpy as np

from andrey.core import GraphStructure
from andrey.search.ges import ges


def gies(
    data: np.ndarray, *, score_func: str = "local_score_BIC", lambda_value: float = 1.0
) -> tuple[GraphStructure, float]:
    """Discover a CPDAG by Greedy Interventional Equivalence Search on observational data.

    Delegates to GES's forward (Insert) and backward (Delete) phases; see the module description
    for the observational equivalence.

    Parameters
    ----------
    data : ndarray, shape (n_samples, n_features)
        Data matrix; rows are observations, columns are variables.
    score_func : str
        Local score; ``"local_score_BIC"`` (linear-Gaussian BIC) is supported.
    lambda_value : float
        Weight on the BIC complexity term of the deviance the search minimizes.

    Returns
    -------
    tuple[GraphStructure, float]
        The CPDAG (``kind="cpdag"``) and its total BIC score (lower is better).
    """
    if score_func != "local_score_BIC":
        raise NotImplementedError(f"unsupported score_func {score_func!r}; use 'local_score_BIC'")
    return ges(data, score_func=score_func, lambda_value=lambda_value)
