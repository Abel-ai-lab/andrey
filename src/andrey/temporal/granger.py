"""Linear Granger causality by a cross-validated multivariate lasso VAR fit.

Stacks a time series into a lag-regression design (each column at time ``t`` regressed on all
columns at times ``t-1 .. t-maxlag``) and fits one cross-validated lasso per target variable. The
resulting coefficient matrix is the Granger lag graph: ``coeff[i, j + d*k]`` is the influence of
variable ``j`` at lag ``k+1`` on variable ``i`` at time ``t``, and its nonzero support is the
directed lag adjacency (a zero coefficient means no Granger cause at that lag).

References
----------
Granger (1969). Investigating Causal Relations by Econometric Models and Cross-spectral
Methods. Econometrica 37(3):424-438. https://doi.org/10.2307/1912791
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV


def granger_lasso(data: np.ndarray, *, maxlag: int = 2, cv: int = 5) -> np.ndarray:
    """Fit the lasso VAR and return the Granger lag-coefficient matrix.

    ``data`` is a ``(n_samples, d)`` time series (row = time step). Returns ``coeff`` of shape
    ``(d, d * maxlag)`` where ``coeff[i, j + d*k]`` is the influence of variable ``j`` at lag
    ``k + 1`` on variable ``i``; ``coeff != 0`` is the directed lag adjacency. Each target row is an
    independent ``cv``-fold cross-validated lasso regression on the stacked lagged design.
    """
    data = np.asarray(data, dtype=np.float64)
    n_samples, d = data.shape
    targets = data[maxlag:]
    design = np.hstack([data[maxlag - k : n_samples - k] for k in range(1, maxlag + 1)])
    coeff = np.zeros((d, d * maxlag), dtype=np.float64)
    lasso = LassoCV(cv=cv)
    for i in range(d):
        lasso.fit(design, targets[:, i])
        coeff[i] = lasso.coef_
    return coeff
