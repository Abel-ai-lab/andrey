"""Ancestor, non-ancestor, and latent claims in the searches that read them.

The PAG methods set the claims as marks before Zhang's rules and keep the latent pairs; with
d-separation in place of the tests, every mark they set must be the true MAG's. LPCMCI reads the
claims per lag, and DirectLiNGAM's order search follows Shimizu et al. (2011, Section 3.4).
"""

from __future__ import annotations

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import Knowledge
from andrey.constraint import fci as fci_engine
from andrey.constraint.fci import check_path_claims, knowledge_marks
from andrey.core import ARROW, CIRCLE, LATENT, NULL, OBSERVED, TAIL, GraphStructure, ci
from andrey.core.knowledge import KnowledgeError, resolve
from andrey.data.latent import DSepOracle, marginal
from andrey.lingam import pwling
from andrey.search import gfci as engines
from andrey.temporal.lpcmci import lpcmci as lpcmci_engine


def _latent_dag(seed: int) -> tuple[nx.DiGraph, int]:
    """A random DAG over 5 to 8 observed variables and 1 to 3 hidden ones, each hidden one a parent
    of two or three observed ones: ``(dag, d)``, the observed nodes ``0..d-1`` first."""
    rng = np.random.default_rng(seed)
    d, h = int(rng.integers(5, 9)), int(rng.integers(1, 4))
    dag = nx.DiGraph()
    dag.add_nodes_from(range(d + h))
    order = rng.permutation(d)
    for i in range(d):
        for j in range(i + 1, d):
            if rng.random() < 2.0 / (d - 1):
                dag.add_edge(int(order[i]), int(order[j]))
    for hidden in range(d, d + h):
        for child in rng.choice(d, int(rng.integers(2, 4)), replace=False):
            dag.add_edge(hidden, int(child))
    return dag, d


def _true_mag(dag: nx.DiGraph, d: int) -> np.ndarray:
    n = dag.number_of_nodes()
    marks = np.zeros((n, n), dtype=np.int8)
    for a, b in dag.edges:
        marks[a, b], marks[b, a] = TAIL, ARROW
    types = np.array([OBSERVED] * d + [LATENT] * (n - d), dtype=np.int8)
    return marginal(
        GraphStructure.from_numpy(marks, kind="dag", node_types=types), "mag"
    ).to_numpy()


def _true_claims(dag: nx.DiGraph, d: int, seed: int) -> Knowledge:
    """About a third of the true ancestor and non-ancestor facts, and every latent pair."""
    rng = np.random.default_rng([seed, 1])
    pairs = [(a, b) for a in range(d) for b in range(d) if a != b]
    ancestors = [p for p in pairs if nx.has_path(dag, *p) and rng.random() < 0.3]
    non_ancestors = [p for p in pairs if not nx.has_path(dag, *p) and rng.random() < 0.3]
    latent = {
        tuple(sorted((int(a), int(b))))
        for hidden in range(d, dag.number_of_nodes())
        for a in dag.successors(hidden)
        for b in dag.successors(hidden)
        if a != b
    }
    return Knowledge(ancestors=ancestors, non_ancestors=non_ancestors, latent=sorted(latent))


def _oracle(monkeypatch: pytest.MonkeyPatch, dag: nx.DiGraph, x: np.ndarray) -> None:
    """Register d-separation in ``dag`` as the test ``dsep_oracle`` over ``x``'s columns, in any
    order."""

    def oracle(columns):
        return DSepOracle(
            dag,
            nodes=[
                next(j for j in range(x.shape[1]) if np.array_equal(c, x[:, j])) for c in columns.T
            ],
        )

    monkeypatch.setitem(ci._CI_REGISTRY, "dsep_oracle", oracle)


def _wrong_marks(found: np.ndarray, mag: np.ndarray) -> np.ndarray:
    """The marks ``found`` sets, a tail or an arrowhead, that the true MAG lacks."""
    return (found != NULL) & (found != CIRCLE) & (found != mag)


@pytest.mark.parametrize("seed", range(40))
def test_fci_with_true_claims_sets_only_true_marks(seed, monkeypatch):
    dag, d = _latent_dag(seed)
    mag = _true_mag(dag, d)
    knowledge = _true_claims(dag, d, seed)
    x = np.random.default_rng(seed).standard_normal((50, d))
    _oracle(monkeypatch, dag, x)
    constraints = resolve(knowledge, d, None, method="fci", ancestral_tiers=True)
    assert constraints is not None and constraints.latent is not None
    for possible_dsep in ("fast", "order_free"):
        found = fci_engine.fci(
            x, indep_test="dsep_oracle", possible_dsep=possible_dsep, constraints=constraints
        ).to_numpy()
        np.testing.assert_array_equal(found != NULL, mag != NULL)
        assert not _wrong_marks(found, mag).any()
        assert (found != NULL)[constraints.latent].all()
        check_path_claims(found, constraints.ancestor, constraints.non_ancestor)


def _minimal_imap(dag: nx.DiGraph, d: int) -> GraphStructure:
    """The CPDAG of the minimal I-map of the observed distribution in a causal order of ``dag``:
    the pattern a score search finds in the large-sample limit."""
    order = [v for v in nx.topological_sort(dag) if v < d]
    separated = DSepOracle(dag)
    marks = np.zeros((d, d), dtype=np.int8)
    for k, child in enumerate(order):
        before = order[:k]
        for parent in before:
            rest = [v for v in before if v != parent]
            if not separated(child, parent, rest):
                marks[parent, child], marks[child, parent] = TAIL, ARROW
    from andrey.core.orient import dag2cpdag, to_structure

    return to_structure(dag2cpdag(marks), kind="cpdag")


@pytest.mark.parametrize("name", ["gfci", "boss_fci", "grasp_fci"])
def test_score_then_test_methods_with_true_claims_set_only_true_marks(name, monkeypatch):
    """The score phase returns the minimal I-map's CPDAG; on each graph where the method without
    claims sets only true marks, it still does with the claims, and keeps every latent pair."""
    checked = 0
    for seed in range(30):
        dag, d = _latent_dag(seed)
        mag = _true_mag(dag, d)
        x = np.random.default_rng(seed).standard_normal((50, d))
        _oracle(monkeypatch, dag, x)
        pattern = _minimal_imap(dag, d)
        for search in ("ges", "boss", "grasp"):
            monkeypatch.setattr(engines, search, lambda *args, p=pattern, **kwargs: (p, 0.0))
        bare = getattr(engines, name)(x, indep_test="dsep_oracle").to_numpy()
        if _wrong_marks(bare, mag).any():
            continue
        constraints = resolve(_true_claims(dag, d, seed), d, None, method=name)
        assert constraints is not None and constraints.latent is not None
        found = getattr(engines, name)(x, indep_test="dsep_oracle", constraints=constraints)
        found = found.to_numpy()
        assert not _wrong_marks(found, mag).any(), seed
        assert (found != NULL)[constraints.latent].all()
        check_path_claims(found, constraints.ancestor, constraints.non_ancestor)
        checked += 1
    assert checked >= 15


@pytest.fixture
def independent() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.standard_normal((400, 4)), columns=list("abcd"))


def _fit(name, data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "fci_jci":
            return andrey.fci_jci(data, np.repeat([0.0, 1.0], len(data) // 2), **kwargs)
        return getattr(andrey, name)(data, **kwargs)


@pytest.mark.parametrize("name", ["fci", "fci_jci", "gfci", "boss_fci", "grasp_fci"])
def test_a_latent_pair_stays_with_its_claimed_marks(name, independent):
    """No test removes a latent pair, even one no score phase finds, and the claims set its marks:
    with neither variable an ancestor of the other, the edge is ``a <-> b``."""
    knowledge = Knowledge(latent=[("a", "b")], non_ancestors=[("a", "b"), ("b", "a")])
    marks = _fit(name, independent, knowledge=knowledge).structure.to_numpy()
    assert marks[0, 1] == ARROW and marks[1, 0] == ARROW
    assert (marks[2:4, :4] == NULL).all()
    marks = _fit(name, independent, knowledge=Knowledge(latent=[("c", "d")])).structure.to_numpy()
    assert marks[2, 3] != NULL and marks[3, 2] != NULL


def test_an_ancestor_claim_on_an_adjacent_pair_directs_it():
    marks = np.full((3, 3), CIRCLE, dtype=np.int8)
    np.fill_diagonal(marks, NULL)
    constraints = resolve(
        Knowledge(ancestors=[(0, 1)], non_ancestors=[(2, 0)]), 3, None, method="fci"
    )
    knowledge_marks(marks, constraints)
    assert (marks[0, 1], marks[1, 0]) == (TAIL, ARROW)
    assert marks[2, 0] == ARROW and marks[0, 2] == CIRCLE
    clash = resolve(Knowledge(ancestors=[(0, 1)], forbidden=[(0, 1)]), 3, None, method="fci")
    with pytest.raises(KnowledgeError, match="arrowhead at 0"):
        knowledge_marks(np.where(np.eye(3, dtype=bool), NULL, CIRCLE).astype(np.int8), clash)


def test_a_claim_the_pag_contradicts_raises(independent):
    with pytest.raises(ValueError, match="no possibly directed path from 'a' to 'b'"):
        _fit("fci", independent, knowledge=Knowledge(ancestors=[("a", "b")]))
    chain = np.zeros((3, 3), dtype=np.int8)
    chain[0, 1] = chain[1, 2] = TAIL
    chain[1, 0] = chain[2, 1] = ARROW
    with pytest.raises(KnowledgeError, match=r"\(0, 2\) in non_ancestors, but the PAG has a"):
        check_path_claims(chain, None, np.eye(3, k=2, dtype=bool))
    check_path_claims(chain, np.eye(3, k=2, dtype=bool), np.eye(3, k=-2, dtype=bool))


@pytest.mark.parametrize(
    "name", ["fci", "gfci", "lpcmci", "direct_lingam", "multi_group_direct_lingam"]
)
def test_an_ancestor_from_a_later_tier_raises_before_the_fit(name, independent):
    claim = ("b", "a", 1) if name == "lpcmci" else ("b", "a")
    knowledge = Knowledge(tiers=[["a"], ["b"]], ancestors=[claim])
    data = [independent, independent] if name == "multi_group_direct_lingam" else independent
    with pytest.raises(ValueError, match="later tier is no ancestor of an earlier one"):
        _fit(name, data, knowledge=knowledge)


# ---- DirectLiNGAM (Shimizu et al. 2011, Section 3.4) ---------------------------------------------


def test_the_order_search_reads_lemmas_3_and_4():
    precedence = np.zeros((4, 4), dtype=bool)
    precedence[0, 1] = True  # Lemma 4: 1 has a claimed ancestor, so it is not exogenous
    np.testing.assert_array_equal(
        pwling.eligible(precedence, None, [0, 1, 2, 3]), [True, False, True, True]
    )
    non_ancestor = np.zeros((4, 4), dtype=bool)
    non_ancestor[[0, 1, 3], 2] = True  # Lemma 3: no other variable is an ancestor of 2
    np.testing.assert_array_equal(
        pwling.eligible(precedence, non_ancestor, [0, 1, 2, 3]), [False, False, True, False]
    )
    # Among 0, 1, and 3 only, the claims show nothing exogenous: every candidate not excluded stays.
    np.testing.assert_array_equal(pwling.eligible(precedence, non_ancestor, [0, 1, 3]), [1, 0, 1])


@pytest.fixture
def chain() -> pd.DataFrame:
    """x -> y -> z with uniform errors, and w on its own."""
    rng = np.random.default_rng(3)
    e = rng.uniform(-1, 1, (3000, 4))
    x = e[:, 0]
    y = 0.8 * x + e[:, 1]
    z = 0.8 * y + e[:, 2]
    return pd.DataFrame({"x": x, "y": y, "z": z, "w": e[:, 3]})


@pytest.mark.parametrize("name", ["direct_lingam", "multi_group_direct_lingam"])
def test_order_claims_shape_the_dag(name, chain):
    data = [chain[:1500], chain[1500:]] if name == "multi_group_direct_lingam" else chain

    def fit(knowledge):
        out = _fit(name, data, knowledge=knowledge)
        return out[0] if isinstance(out, list) else out

    # One-way non-ancestor claims around a cycle order no pair, so they raise nothing.
    cyclic = Knowledge(non_ancestors=[("x", "w"), ("w", "z"), ("z", "x")])
    assert fit(cyclic).metadata["unmet_knowledge"] == []
    # Claims that make z exogenous, against the data: Lemma 3 puts z first, and Lemma 5 gives it
    # no parent, so the DAG meets them.
    exogenous = Knowledge(non_ancestors=[("x", "z"), ("y", "z"), ("w", "z")])
    out = fit(exogenous)
    assert not (out.structure.to_numpy()[2] == ARROW).any()
    assert out.metadata["unmet_knowledge"] == []
    met = fit(Knowledge(ancestors=[("x", "z")]))
    assert met.metadata["unmet_knowledge"] == []
    unmet = fit(Knowledge(ancestors=[("w", "x")]))
    assert unmet.metadata["unmet_knowledge"] == [["ancestors", "w", "x"]]


# ---- LPCMCI --------------------------------------------------------------------------------------


def _series(seed: int = 1, steps: int = 600) -> np.ndarray:
    """x drives y one step later, y drives z in the same step."""
    rng = np.random.default_rng(seed)
    x = np.zeros((steps, 3))
    for t in range(1, steps):
        x[t, 0] = 0.6 * x[t - 1, 0] + rng.standard_normal()
        x[t, 1] = 0.5 * x[t - 1, 1] + 0.6 * x[t - 1, 0] + rng.standard_normal()
        x[t, 2] = 0.5 * x[t - 1, 2] + 0.5 * x[t, 1] + rng.standard_normal()
    return x


def test_lpcmci_sets_the_claimed_marks_on_the_links_it_keeps():
    data = _series()
    knowledge = Knowledge(ancestors=[(1, 2, 0)], non_ancestors=[(2, 0, 0)], latent=[(0, 2, 0)])
    constraints = andrey.core.knowledge.resolve_lagged(knowledge, 3, None, 2, method="lpcmci")
    result = lpcmci_engine(data, tau_max=2, constraints=constraints)
    assert (result.start_marks[0, 1, 2], result.end_marks[0, 1, 2]) == (TAIL, ARROW)
    assert result.end_marks[0, 0, 2] == ARROW  # the cause 2 of the non-ancestor claim
    bare = lpcmci_engine(data, tau_max=2)
    assert bare.start_marks[0, 0, 2] == NULL  # x and z are separated given y
    assert result.start_marks[0, 0, 2] != NULL  # the latent link stays


def test_lpcmci_claims_need_their_lag():
    data = _series()
    for knowledge, message in (
        (Knowledge(ancestors=[(0, 1)]), "without a lag"),
        (Knowledge(latent=[(0, 1, 3)]), "at lag 3"),
        (Knowledge(ancestors=[(0, 1, 0)], forbidden=[(0, 1, 0)]), "arrowhead at the cause"),
        (Knowledge(latent=[(0, 1, 1)], forbidden=[(0, 1, 1)]), "leaves out"),
        (Knowledge(ancestors=[(0, 1, 1)], non_ancestors=[(0, 1, 1)]), "non_ancestors lists"),
    ):
        with pytest.raises(ValueError, match=message):
            _fit("lpcmci", data, tau_max=2, knowledge=knowledge)
    with pytest.raises(ValueError, match="no possibly directed path from 2 at t-2 to 0 at t"):
        _fit("lpcmci", data, tau_max=2, knowledge=Knowledge(ancestors=[(2, 0, 2)]))
