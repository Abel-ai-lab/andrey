"""pgmpy's and bnlearn's categorical adapters: the harness contract and one objective per pair.

Each comparator must score or test exactly what Andrey's setting of the same name does: pgmpy's
and bnlearn's multinomial BIC are ``-1/2`` times Andrey's ``local_score_BIC_categorical``,
bnlearn's BDeu with the ``vsp`` prior is ``-1/2`` times Andrey's ``local_score_BDeu`` up to a
constant per node, pgmpy's BDeu with the adapter's structure prior is exactly ``-1/2`` times it, and
bnlearn's ``x2-adf`` and ``mi-adf`` p-values are Andrey's ``chisq`` and ``gsq`` except for the
sparse-table rule the adapter documents. Each search setting the adapters declare must take effect.
The pgmpy checks run where pgmpy is installed and the bnlearn checks where R is loaded; these tests
live apart from the discrete lane's, which may not skip.
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from itertools import combinations

import numpy as np
import pytest
from conftest import needs_r

from andrey.core.ci import ChiSquare
from andrey.core.score import BDeuScore, CategoricalBICScore
from andrey.data import sample_discrete
from andrey_bench.adapters import pgmpy_hc
from andrey_bench.adapters.bnlearn_discrete import (
    SCORES,
    TESTS,
    BnlearnHCCategorical,
    BnlearnPCStableCategorical,
    BnlearnTabuCategorical,
)
from andrey_bench.adapters.pgmpy_hc import PgmpyHC
from andrey_bench.contracts import SolutionAdapter

HC_SCORES = list(SCORES)
ADAPTERS = [
    *(PgmpyHC(s) for s in HC_SCORES),
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


def _acyclic(adjacency: np.ndarray) -> bool:
    """Whether ``adjacency`` (``[i, j] == 1`` for ``i -> j``) has no directed cycle."""
    indegree = adjacency.sum(axis=0)
    ready = [j for j in range(len(adjacency)) if indegree[j] == 0]
    seen = 0
    while ready:
        i = ready.pop()
        seen += 1
        for j in np.flatnonzero(adjacency[i]):
            indegree[j] -= 1
            if indegree[j] == 0:
                ready.append(int(j))
    return seen == len(adjacency)


def _improving_move(score, adjacency: np.ndarray, tol: float) -> tuple[str, int, int] | None:
    """An edge addition, removal, or reversal that keeps a DAG and lowers the total score by more
    than ``tol``, or ``None`` when ``adjacency`` is a local optimum of ``score``."""
    base = _total(score, adjacency)
    for i, j in zip(*np.nonzero(~np.eye(len(adjacency), dtype=bool)), strict=True):
        ops = ("remove", "reverse") if adjacency[i, j] else () if adjacency[j, i] else ("add",)
        for op in ops:
            new = adjacency.copy()
            new[i, j] = op == "add"
            if op == "reverse":
                new[j, i] = 1
            if (op == "remove" or _acyclic(new)) and _total(score, new) < base - tol:
                return op, int(i), int(j)
    return None


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


def _pgmpy_dag(frame, adjacency: np.ndarray):
    from pgmpy.base import DAG

    dag = DAG([(int(i), int(j)) for i, j in zip(*np.nonzero(adjacency), strict=True)])
    dag.add_nodes_from(frame.columns)
    return dag


def _frame(x: np.ndarray):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(x.astype(np.int64))


def test_pgmpy_is_the_pinned_release() -> None:
    pytest.importorskip("pgmpy")
    assert pgmpy_hc.check_environment() == {"pgmpy": pgmpy_hc.PINNED_VERSION}
    assert PgmpyHC().package_version == pgmpy_hc.PINNED_VERSION


def test_pgmpy_bic_is_minus_half_of_andreys() -> None:
    pytest.importorskip("pgmpy")
    from pgmpy.structure_score import BIC

    x = _sample(2, d=5)
    theirs = BIC(_frame(x)).score(_pgmpy_dag(_frame(x), _DAG))
    assert -2 * theirs == pytest.approx(_total(CategoricalBICScore(x), _DAG), rel=1e-9)


@pytest.mark.parametrize("d", [4, 5, 9])
def test_pgmpy_bdeu_with_the_structure_prior_is_minus_half_of_andreys(d: int) -> None:
    """pgmpy's BDeu has an equivalent sample size of 10 and no structure prior by default; the
    adapter sets 1 and Andrey's prior, so the two scores agree on the whole graph."""
    pytest.importorskip("pgmpy")
    x = _sample(2, d=d)
    adjacency = np.triu(np.ones((d, d), dtype=int), k=1) * (np.arange(d * d).reshape(d, d) % 3 == 0)
    score = pgmpy_hc._prior_bdeu_class()(_frame(x), equivalent_sample_size=1.0)
    theirs = score.score(_pgmpy_dag(_frame(x), adjacency))
    assert -2 * theirs == pytest.approx(_total(BDeuScore(x), adjacency), rel=1e-9)


def test_pgmpy_bdeu_prior_ratios_are_the_change_in_the_prior_of_the_graph() -> None:
    """The search adds ``structure_prior_ratio`` to each move's gain: it must be the difference of
    the prior of the graph a move leads to and the one it leaves."""
    pytest.importorskip("pgmpy")
    x = _sample(3, d=6)
    score = pgmpy_hc._prior_bdeu_class()(_frame(x), equivalent_sample_size=1.0)
    before = np.zeros((6, 6), dtype=int)
    before[0, 1] = before[1, 2] = 1
    added, removed, flipped = before.copy(), before.copy(), before.copy()
    added[3, 2] = 1
    removed[1, 2] = 0
    flipped[0, 1], flipped[1, 0] = 0, 1
    frame = _frame(x)
    prior = {
        name: score.structure_prior(_pgmpy_dag(frame, adjacency))
        for name, adjacency in {
            "before": before,
            "+": added,
            "-": removed,
            "flip": flipped,
        }.items()
    }
    for operation in ("+", "-", "flip"):
        assert prior[operation] - prior["before"] == pytest.approx(
            score.structure_prior_ratio(operation), abs=1e-12
        )


@pytest.mark.parametrize("score_func", HC_SCORES)
def test_pgmpy_hill_climbing_improves_andreys_score_and_reuses_no_earlier_fit(
    score_func: str,
) -> None:
    """The DAG found scores below the empty graph on Andrey's score, and a fit after one on other
    data in the same process returns what a first fit returns."""
    pytest.importorskip("pgmpy")
    adapter = PgmpyHC(score_func)
    adapter.setup()
    first, second = _sample(0), _sample(1)
    adjacency = adapter.fit(first, adapter.params())
    assert set(np.unique(adjacency).tolist()) <= {0, 1}
    adapter.to_structure(adjacency).validate()
    score = BDeuScore(first) if score_func == "local_score_BDeu" else CategoricalBICScore(first)
    assert _total(score, adjacency) < _total(score, np.zeros_like(adjacency))
    assert np.array_equal(
        adapter.fit(second, adapter.params()), PgmpyHC(score_func).fit(second, {})
    )


_PGMPY_FIT = """
import sys, numpy as np
from andrey.data import sample_discrete
from andrey_bench.adapters.pgmpy_hc import PgmpyHC
x = sample_discrete(d=12, n=800, seed=3).data
for score_func in ("local_score_BDeu", "local_score_BIC_categorical"):
    print(PgmpyHC(score_func).fit(x, {}).tobytes().hex())
"""


def test_pgmpy_returns_the_same_graph_whatever_the_hash_seed() -> None:
    """Equal moves are broken by set order, which string column names would tie to the seed."""
    pytest.importorskip("pgmpy")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    runs = [
        subprocess.run(
            [sys.executable, "-c", _PGMPY_FIT],
            env={**env, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    ]
    assert runs[0] == runs[1] == runs[2]


@pytest.mark.parametrize("score_func", HC_SCORES)
def test_pgmpy_stops_at_its_move_budget(score_func: str) -> None:
    pytest.importorskip("pgmpy")
    adapter = PgmpyHC(score_func)
    assert adapter.fit(_sample(0), {**adapter.params(), "max_iter": 1}).sum() == 1


def test_pgmpy_refuses_the_settings_it_cannot_run() -> None:
    pytest.importorskip("pgmpy")
    bic = PgmpyHC("local_score_BIC_categorical")
    with pytest.raises(ValueError, match="penalty"):
        bic.fit(_sample(0), {**bic.params(), "lambda_value": 2.0})
    with pytest.raises(ValueError, match="parent cap"):
        bic.fit(_sample(0), {**bic.params(), "max_parents": 4})
    with pytest.raises(ValueError, match="score_func"):
        bic.fit(_sample(0), {"score_func": "local_score_BDeu"})
    with pytest.raises(ValueError, match="score_func"):
        PgmpyHC("local_score_BIC_from_cov")


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


def _tests(session, x: np.ndarray, test: str, size: int, count: int = 60):
    """``count`` conditional tests of up to ``size`` variables on `x`: ``(a, b, s, p)``, with p
    from bnlearn."""
    call = session.function("andrey_bnlearn_ci_categorical")
    d = x.shape[1]
    rng = np.random.default_rng(0)
    for _ in range(count):
        a, b, *s = (int(c) for c in rng.permutation(d)[: 2 + int(rng.integers(0, size + 1))])
        z = ",".join(str(c + 1) for c in s)
        yield a, b, tuple(s), float(call(session.matrix(x), a + 1, b + 1, z, test)[0])


@needs_r()
@pytest.mark.parametrize("indep_test", list(TESTS))
def test_bnlearn_declares_a_sparse_table_independent_and_otherwise_agrees_with_andrey(
    indep_test: str,
) -> None:
    """The ``-adf`` tests return ``p = 1`` when ``n < 5 * r * c * q``, as the adapter documents;
    every other table's p-value is Andrey's."""
    session = _r_session()
    x = _sample(4, d=9, n=600)
    mine = ChiSquare(x, g_square=indep_test == "gsq")
    categories = [len(np.unique(x[:, j])) for j in range(x.shape[1])]
    sparse = dense = 0
    for a, b, s, theirs in _tests(session, x, TESTS[indep_test], size=5):
        cells = categories[a] * categories[b] * int(np.prod([categories[c] for c in s]))
        if len(x) < 5 * cells:
            sparse += 1
            assert theirs == 1.0 and mine(a, b, s) < 1.0
        else:
            dense += 1
            assert mine(a, b, s) == pytest.approx(theirs, rel=1e-6, abs=1e-12)
    assert sparse and dense


@needs_r()
@pytest.mark.parametrize("indep_test", list(TESTS))
def test_bnlearn_without_the_degrees_of_freedom_adjustment_is_not_andreys_test(
    indep_test: str,
) -> None:
    """The reason the adapter runs ``x2-adf`` and ``mi-adf``, not ``x2`` and ``mi``."""
    session = _r_session()
    x = _sample(4, d=9, n=600)
    mine = ChiSquare(x, g_square=indep_test == "gsq")
    plain = {"chisq": "x2", "gsq": "mi"}[indep_test]
    differs = [
        s
        for a, b, s, theirs in _tests(session, x, plain, size=2)
        if mine(a, b, s) != pytest.approx(theirs, rel=1e-6, abs=1e-12)
    ]
    assert differs


@needs_r()
@pytest.mark.parametrize(
    "adapter", [a for a in ADAPTERS if a.package == "bnlearn"], ids=lambda a: a.name
)
def test_a_bnlearn_fit_gives_its_kind_of_graph(adapter) -> None:
    adapter.setup()
    native = adapter.fit(_sample(3), adapter.params())
    structure = adapter.to_structure(native)
    structure.validate()
    assert structure.kind == adapter.output_type and structure.n_nodes == 6


@needs_r()
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("score_func", HC_SCORES)
def test_bnlearn_hill_climbing_stops_where_andreys_score_cannot_improve(
    score_func: str, seed: int
) -> None:
    """With every setting matched, bnlearn's climb ends at a graph no single edge move improves on
    Andrey's scale. Andrey's climb can stop at a different such graph: each breaks exact score ties
    its own way."""
    x = sample_discrete(d=8, n=2000, seed=seed).data
    adapter = BnlearnHCCategorical(score_func)
    adapter.setup()
    adjacency = adapter.fit(x, adapter.params())
    score = BDeuScore(x) if score_func == "local_score_BDeu" else CategoricalBICScore(x)
    assert _improving_move(score, adjacency, 1e-9 * abs(_total(score, adjacency))) is None


@needs_r()
@pytest.mark.parametrize("score_func", HC_SCORES)
def test_bnlearn_stops_at_its_move_budget(score_func: str) -> None:
    adapter = BnlearnHCCategorical(score_func)
    adapter.setup()
    assert adapter.fit(_sample(0), {**adapter.params(), "max_iter": 1}).sum() == 1


@needs_r()
def test_bnlearn_refuses_a_parent_cap() -> None:
    adapter = BnlearnHCCategorical("local_score_BIC_categorical")
    adapter.setup()
    with pytest.raises(ValueError, match="parent cap"):
        adapter.fit(_sample(0), {**adapter.params(), "max_parents": 4})
