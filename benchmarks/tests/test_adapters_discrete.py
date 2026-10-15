"""The categorical-data adapters and the discrete driver.

Each comparison in ``discrete_cpu.py`` must run one objective in every package: the same test with
the same degrees of freedom, or the same score with the same prior. The R checks compare bnlearn's
score of a fixed DAG with Andrey's; the others need no R.
"""

from __future__ import annotations

import json
import warnings

import discrete_cpu
import numpy as np
import pytest
from conftest import needs_r

import andrey
from andrey.core.discrete import BDeuScore, DiscreteBICScore
from andrey.data.discrete import DiscreteBN
from andrey.data.graphs import erdos_renyi
from andrey_bench.adapters.bnlearn_discrete import BnlearnDiscreteHC, BnlearnDiscretePC
from andrey_bench.adapters.causal_learn_boss import CausalLearnBOSS
from andrey_bench.adapters.causal_learn_ges import CausalLearnGES
from andrey_bench.adapters.pgmpy_hc import PgmpyHC


def _sample(seed: int = 0, d: int = 6, n: int = 400) -> np.ndarray:
    """State codes from a random discrete network with 2 to 3 states per variable."""
    bn = DiscreteBN(graph=erdos_renyi(d=d, avg_degree=2.0), levels=(2, 3), concentration=0.5)
    return bn.sample(n=n, seed=seed).data


def _total(score, adjacency: np.ndarray) -> float:
    return sum(
        score.score(j, np.flatnonzero(adjacency[:, j]).tolist()) for j in range(len(adjacency))
    )


@pytest.mark.parametrize("adapter", [CausalLearnGES(), CausalLearnBOSS(), PgmpyHC()], ids=str)
def test_a_reference_adapter_fits_categorical_data(adapter) -> None:
    x = _sample()
    adapter.setup()
    params = (
        {"score": "bic"} if isinstance(adapter, PgmpyHC) else {"score_func": "local_score_BDeu"}
    )
    native = np.asarray(adapter.fit(x, params))
    assert native.shape == (6, 6) and set(np.unique(native).tolist()) <= {0, 1}
    adapter.to_structure(native).validate()


def test_pgmpy_hill_climbing_maximizes_andreys_multinomial_bic() -> None:
    """pgmpy's BIC is Andrey's up to the factor -2, so its DAG scores no worse than the empty one,
    and a second fit in the same process on other data is not answered from the first's cache."""
    adapter = PgmpyHC()
    adapter.setup()
    first, second = _sample(0), _sample(1)
    adjacency = adapter.fit(first, adapter.params())
    score = DiscreteBICScore(first)
    assert _total(score, adjacency) <= _total(score, np.zeros_like(adjacency))
    fresh = PgmpyHC().fit(second, adapter.params())
    assert np.array_equal(adapter.fit(second, adapter.params()), fresh)


def test_the_comparisons_match_their_settings() -> None:
    for method in discrete_cpu.METHODS:
        rows = discrete_cpu.solutions(method, 10, with_r=True)
        settings = [json.dumps(p, sort_keys=True) for _, p in rows]
        assert len(rows) == len({a.name for a, _ in rows}), method
        if method.startswith("PC-"):
            test = method.removeprefix("PC-")
            assert {"indep_test": test} in [{"indep_test": p.get("indep_test")} for _, p in rows]
            bnlearn = {"chisq": "x2-adf", "gsq": "mi-adf"}[test]
            assert any(p.get("test") == bnlearn for _, p in rows), settings
        if method.endswith("-BDeu"):
            assert all(p.get("score_func", "local_score_BDeu") == "local_score_BDeu"
                       for _, p in rows if "score" not in p)  # fmt: skip
            assert all(p["score"] == "bde" for _, p in rows if "score" in p)


@needs_r()
def test_bnlearn_bde_with_vsp_is_andreys_bdeu() -> None:
    from andrey_bench import rsession

    x = _sample(2, d=5)
    dag = np.zeros((5, 5), dtype=int)
    dag[0, 1] = dag[1, 2] = dag[0, 3] = dag[3, 4] = dag[2, 4] = 1
    adapter = BnlearnDiscreteHC()
    adapter.setup()
    session = rsession.session()
    session.define("""
    andrey_bnlearn_bde_score <- function(X, A) {
        df <- andrey_bnlearn_factors(X)
        colnames(A) <- rownames(A) <- names(df)
        g <- bnlearn::empty.graph(names(df))
        bnlearn::amat(g) <- A
        bnlearn::score(g, df, type = "bde", iss = 1, prior = "vsp", beta = 1 / (ncol(df) - 1))
    }
    """)
    theirs = float(
        session.function("andrey_bnlearn_bde_score")(session.matrix(x), session.matrix(dag))[0]
    )
    # bnlearn's vsp prior is k log(beta / (1 - beta)) per node; Andrey's adds (d - 1) log(1 - beta),
    # the same for every graph, so both rank graphs alike.
    d = x.shape[1]
    constant = d * (d - 1) * np.log(1 - 1 / (d - 1))
    assert -2 * (theirs + constant) == pytest.approx(_total(BDeuScore(x), dag), rel=1e-9)


@needs_r()
@pytest.mark.parametrize("test", ["x2-adf", "mi-adf"])
def test_bnlearn_pc_runs_on_factor_columns(test) -> None:
    adapter = BnlearnDiscretePC()
    adapter.setup()
    native = adapter.fit(_sample(3), {"alpha": 0.05, "test": test})
    adapter.to_structure(native).validate()


def test_the_driver_records_every_fit(tmp_path) -> None:
    """Without R: PC with the chi-square test and the empty graph on the smoke datasets."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        assert (
            discrete_cpu.main(
                ["--out", str(tmp_path), "--smoke", "--no-r", "--methods", "PC-chisq", "Empty"]
            )
            == 0
        )
    records = [json.loads(line) for line in (tmp_path / "records.jsonl").read_text().splitlines()]
    assert {r["status"] for r in records} == {"ok"}
    assert {(r["dataset"], r["solution"]) for r in records} == {
        (ds, s)
        for ds in ("asia", "er10")
        for s in ("andrey.pc.numpy", "causal-learn.pc", "baseline.empty")
    }
    assert all("shd" in r and "cpdag_hash" in r for r in records)
    # Andrey's and causal-learn's PC agree on ASIA's smoke sample.
    pc = {r["solution"]: r["cpdag_hash"] for r in records if r["dataset"] == "asia"}
    assert pc["andrey.pc.numpy"] == pc["causal-learn.pc"]
