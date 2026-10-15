"""pgmpy's and bnlearn's categorical adapters: the harness contract and one objective per pair.

Each comparator must score or test exactly what Andrey's setting of the same name does: pgmpy's
and bnlearn's multinomial BIC are ``-1/2`` times Andrey's ``local_score_BIC_categorical``,
bnlearn's BDeu with the ``vsp`` prior is ``-1/2`` times Andrey's ``local_score_BDeu`` up to a
constant per node, and bnlearn's ``x2-adf`` and ``mi-adf`` p-values are Andrey's ``chisq`` and
``gsq``. The pgmpy checks run where pgmpy is installed and the bnlearn checks where R is loaded;
these tests live apart from the discrete lane's, which may not skip.
"""

from __future__ import annotations

import json
import pickle
from itertools import combinations

import numpy as np
import pytest
from conftest import needs_r

from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore, CategoricalBICScore
from andrey.data import sample_discrete
from andrey_bench.adapters.bnlearn_discrete import (
    SCORES,
    TESTS,
    BnlearnHCCategorical,
    BnlearnPCStableCategorical,
    BnlearnTabuCategorical,
)
from andrey_bench.adapters.pgmpy_hc import PgmpyHC
from andrey_bench.contracts import SolutionAdapter

ADAPTERS = [
    PgmpyHC(),
    *(BnlearnPCStableCategorical(t) for t in TESTS),
    *(BnlearnHCCategorical(s) for s in SCORES),
    *(BnlearnTabuCategorical(s) for s in SCORES),
]


def _sample(seed: int = 0, d: int = 6, n: int = 400) -> np.ndarray:
    return sample_discrete(d=d, n=n, seed=seed, concentration=0.5).data


def _total(score, adjacency: np.ndarray) -> float:
    """A DAG's total score; ``adjacency[i, j] == 1`` is the edge ``i -> j``."""
    d = len(adjacency)
    return sum(score.score(j, np.flatnonzero(adjacency[:, j]).tolist()) for j in range(d))


_DAG = np.zeros((5, 5), dtype=int)
_DAG[0, 1] = _DAG[1, 2] = _DAG[0, 3] = _DAG[3, 4] = _DAG[2, 4] = 1


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: a.name)
def test_an_adapter_meets_the_harness_contract(adapter: SolutionAdapter) -> None:
    assert isinstance(adapter, SolutionAdapter)
    json.dumps(adapter.params(), sort_keys=True)
    clone = pickle.loads(pickle.dumps(adapter))
    assert (clone.name, clone.params()) == (adapter.name, adapter.params())


def test_every_adapter_has_its_own_name() -> None:
    names = [a.name for a in ADAPTERS]
    assert len(set(names)) == len(names)


def test_an_adapter_names_its_setting_in_andreys_names() -> None:
    for adapter in ADAPTERS:
        setting = adapter.params().get("score_func", adapter.params().get("indep_test"))
        assert setting in (*TESTS, *SCORES)
    with pytest.raises(ValueError, match="score_func"):
        BnlearnHCCategorical("bde")
    with pytest.raises(ValueError, match="indep_test"):
        BnlearnPCStableCategorical("x2-adf")


# --- pgmpy ----------------------------------------------------------------------------------------


def test_pgmpy_bic_is_minus_half_of_andreys() -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pgmpy")
    from pgmpy.base import DAG
    from pgmpy.structure_score import BIC

    x = _sample(2, d=5)
    frame = pd.DataFrame(x.astype(np.int64), columns=[f"v{j}" for j in range(5)])
    dag = DAG([(f"v{i}", f"v{j}") for i, j in zip(*np.nonzero(_DAG), strict=True)])
    dag.add_nodes_from(frame.columns)
    theirs = BIC(frame).score(dag)
    assert -2 * theirs == pytest.approx(_total(CategoricalBICScore(x), _DAG), rel=1e-9)


def test_pgmpy_hill_climbing_improves_andreys_bic_and_reuses_no_earlier_fit() -> None:
    """The DAG found scores below the empty graph on Andrey's BIC, and a fit after one on other
    data in the same process returns what a first fit returns."""
    pytest.importorskip("pgmpy")
    adapter = PgmpyHC()
    adapter.setup()
    first, second = _sample(0), _sample(1)
    adjacency = adapter.fit(first, adapter.params())
    assert set(np.unique(adjacency).tolist()) <= {0, 1}
    adapter.to_structure(adjacency).validate()
    score = CategoricalBICScore(first)
    assert _total(score, adjacency) < _total(score, np.zeros_like(adjacency))
    assert np.array_equal(adapter.fit(second, adapter.params()), PgmpyHC().fit(second, {}))


# --- bnlearn --------------------------------------------------------------------------------------

_R_SCORE = """
andrey_bnlearn_score_categorical <- function(X, A, score) {
    df <- andrey_bnlearn_factors(X)
    colnames(A) <- rownames(A) <- names(df)
    g <- bnlearn::empty.graph(names(df))
    bnlearn::amat(g) <- A
    if (score == "bde") {
        return(bnlearn::score(g, df, type = "bde", iss = 1, prior = "vsp",
                              beta = 1 / (ncol(df) - 1)))
    }
    bnlearn::score(g, df, type = score)
}
andrey_bnlearn_ci_categorical <- function(X, x, y, z, test) {
    df <- andrey_bnlearn_factors(X)
    z <- as.integer(strsplit(z, ",")[[1]])
    if (length(z) == 0) return(bnlearn::ci.test(names(df)[x], names(df)[y], data = df,
                                                test = test)$p.value)
    bnlearn::ci.test(names(df)[x], names(df)[y], names(df)[z], data = df, test = test)$p.value
}
"""


def _r_session():
    from andrey_bench import rsession

    BnlearnHCCategorical("local_score_BDeu").setup()
    session = rsession.session()
    session.define(_R_SCORE)
    return session


@needs_r()
@pytest.mark.parametrize("score_func", list(SCORES))
def test_bnlearn_scores_a_dag_as_andrey_does(score_func: str) -> None:
    """``-2`` times bnlearn's score is Andrey's; for BDeu, bnlearn's ``vsp`` prior leaves out
    ``(d - 1) * log(1 - beta)`` per node, the same for every graph."""
    session = _r_session()
    x = _sample(2, d=5)
    call = session.function("andrey_bnlearn_score_categorical")
    theirs = float(
        call(session.matrix(x), session.matrix(_DAG.astype(float)), SCORES[score_func])[0]
    )
    d = x.shape[1]
    if score_func == "local_score_BDeu":
        theirs += d * (d - 1) * np.log(1 - 1 / (d - 1))
        mine = _total(BDeuScore(x), _DAG)
    else:
        mine = _total(CategoricalBICScore(x), _DAG)
    assert -2 * theirs == pytest.approx(mine, rel=1e-9)


@needs_r()
@pytest.mark.parametrize("indep_test", list(TESTS))
def test_bnlearn_p_values_are_andreys(indep_test: str) -> None:
    session = _r_session()
    x = _sample(4, d=5, n=600)
    mine = ChiSquare(x, g_square=indep_test == "gsq")
    call = session.function("andrey_bnlearn_ci_categorical")
    for a, b in combinations(range(5), 2):
        rest = [c for c in range(5) if c not in (a, b)]
        for size in range(3):
            for s in combinations(rest, size):
                z = ",".join(str(c + 1) for c in s)
                theirs = float(call(session.matrix(x), a + 1, b + 1, z, TESTS[indep_test])[0])
                assert mine(a, b, s) == pytest.approx(theirs, rel=1e-6, abs=1e-12)


@needs_r()
@pytest.mark.parametrize("adapter", ADAPTERS[1:], ids=lambda a: a.name)
def test_a_bnlearn_fit_gives_its_kind_of_graph(adapter) -> None:
    adapter.setup()
    native = adapter.fit(_sample(3), adapter.params())
    structure = adapter.to_structure(native)
    structure.validate()
    assert structure.kind == adapter.output_type and structure.n_nodes == 6
