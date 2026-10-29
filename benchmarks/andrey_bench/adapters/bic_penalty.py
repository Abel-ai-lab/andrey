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

#: Andrey's name for each score the adapters' params name in the harness's recorded vocabulary
#: (Andrey's names from 0.10.0), which every package's adapter shares.
ANDREY_SCORE = {
    "local_score_BIC": "bic",
    "local_score_BIC_from_cov": "bic",
    "local_score_BDeu": "bdeu",
    "local_score_BIC_categorical": "discrete_bic",
    "local_score_cond_gauss": "cond_gauss",
    "local_score_degen_gauss": "degen_gauss",
}


def andrey_score(name: str) -> str:
    """Return Andrey's name for the params' score ``name``; a name Andrey keeps passes through."""
    return ANDREY_SCORE.get(name, name)


#: Shared seed because BOSS and GRaSP search random variable orders.
SEARCH_SEED = 0

#: Andrey's extended-BIC weight in every lane. The reference packages score the standard BIC, so
#: Andrey's linear-Gaussian BIC runs at ``ebic_gamma=0.0`` and both sides of a row optimize one
#: objective.
ANDREY_EBIC_GAMMA = 0.0
