"""Express one BIC complexity penalty in each package's units.

causal-learn maximizes the log-likelihood score
``-0.5*n*(1 + log s2) - lam*(|Pa| + 1)*log n``. Andrey minimizes the deviance
``n*log s2 + lam*|Pa|*log n``. Multiplying the causal-learn score by ``-2`` gives
``lam_andrey = 2*lam_cl``. Terms constant across parent sets do not affect the optimum.
"""

from __future__ import annotations

#: causal-learn's BOSS and GRaSP default, matching Tetrad's penalty discount.
CAUSAL_LEARN_LAMBDA = 2.0


def andrey_lambda(causal_learn_lambda: float) -> float:
    """Return the Andrey (deviance-unit) coefficient matching ``causal_learn_lambda``."""
    return 2.0 * causal_learn_lambda


#: The equivalent penalty in Andrey's deviance units.
ANDREY_LAMBDA = andrey_lambda(CAUSAL_LEARN_LAMBDA)

#: Shared seed because BOSS and GRaSP search random variable orders.
SEARCH_SEED = 0
