"""Latent-confounder tests.

Refutations: a latent SCM drops exactly the latent columns, flags them in
node_types, keeps observed nodes first; latents are genuine confounders (>= 2 observed children);
too few confounders raises.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey.data as data
from andrey.core.structure import LATENT


def test_latent_indexing_and_dropping():
    d, k = 30, 4
    ds = data.sample_scm(graph="erdos_renyi", d=d, n=200, seed=0, density=3.0, latents=k)
    assert ds.data.shape == (200, d - k)  # latent columns dropped
    types = ds.graph.node_types
    assert types is not None
    assert int((types == LATENT).sum()) == k  # exactly k latents flagged
    latent_idx = np.flatnonzero(types == LATENT)
    assert np.all(latent_idx >= d - k)  # observed-first: latents occupy the last indices
    assert ds.provenance["config"]["latents"] == k


def test_latents_are_genuine_confounders():
    """Non-vacuity: every latent has >= 2 children in the full truth (else it projects away)."""
    d, k = 40, 5
    ds = data.sample_scm(d=d, n=64, seed=1, density=3.0, latents=k)
    latent_idx = set(np.flatnonzero(ds.graph.node_types == LATENT).tolist())
    out_degree = np.bincount(ds.params.edges[:, 0], minlength=d)
    assert all(out_degree[i] >= 2 for i in latent_idx)


def test_observed_graph_aligns_with_data():
    ds = data.sample_scm(d=30, n=300, seed=3, density=3.0, latents=4)
    og = ds.observed_graph
    n_obs = ds.data.shape[1]
    assert og.n_nodes == n_obs  # scoring target matches the data's columns
    assert og.kind == "dag" and og.node_types is None
    # the subgraph induced on the observed nodes: every observed edge, none touching a latent
    edges = {(int(i), int(j)) for i, j in ds.params.edges}
    observed = {(i, j) for i, j in edges if i < n_obs and j < n_obs}
    assert observed and observed != edges
    assert set(og.oriented_edges()) == {(i, j, "directed") for i, j in observed}


def test_reproducible_with_latents():
    scm = data.SCM(
        graph=data.graphs.erdos_renyi(30, 3.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
        latents=3,
    )
    a, b = scm.sample(n=100, seed=5), scm.sample(n=100, seed=5)
    assert np.array_equal(a.data, b.data)
    assert a.graph == b.graph


def test_too_few_confounders_raises():
    """A tiny/sparse graph without enough >=2-child nodes raises rather than silently
    degenerating."""
    with pytest.raises(ValueError, match="latent confounder"):
        data.sample_scm(d=4, n=50, seed=0, density=1.0, latents=3)


def test_no_latents_unchanged():
    """latents=0 leaves the causally-sufficient path byte-identical."""
    a = data.sample_scm(d=12, n=200, seed=0)
    b = data.SCM(
        graph=data.graphs.erdos_renyi(12, 4.0),
        functional=data.functional.linear(),
        noise=data.noise.gaussian(),
        latents=0,
    ).sample(n=200, seed=0)
    assert np.array_equal(a.data, b.data)
    assert a.graph.node_types is None


# ---- MAG marginal -------------------------------------------------------------------------------


def _labeled_dag(parents, children, d, latent_nodes):
    from andrey.core.structure import LATENT as _L
    from andrey.core.structure import OBSERVED as _O
    from andrey.data.graphs import dag_truth

    nt = np.array([_L if i in latent_nodes else _O for i in range(d)], dtype=np.int8)
    return dag_truth(np.array(parents), np.array(children), d, node_types=nt)


def test_mag_pure_confounder_is_bidirected():
    """L -> a, L -> b (L latent): the observed marginal is a <-> b (bidirected = hidden common
    cause)."""
    from andrey.core.structure import ARROW

    g = _labeled_dag(
        parents=[2, 2], children=[0, 1], d=3, latent_nodes={2}
    )  # 0,1 observed; 2 latent
    mag = data.latent.marginal(g)
    assert mag.kind == "pag"
    assert mag.n_nodes == 2
    assert tuple(mag.endpoints(0, 1)) == (ARROW, ARROW)  # bidirected


def test_mag_confounder_plus_direct_collapses_to_directed():
    """L->a, L->b, a->b: a is an ancestor of b, so the MAG is a -> b (one edge per pair, not
    two)."""
    from andrey.core.structure import ARROW, TAIL

    g = _labeled_dag(parents=[2, 2, 0], children=[0, 1, 1], d=3, latent_nodes={2})
    mag = data.latent.marginal(g)
    assert tuple(mag.endpoints(0, 1)) == (TAIL, ARROW)  # a -> b


def test_mag_nonvacuity_and_roundtrip():
    """A latent SCM's MAG has >= 1 bidirected edge, is kind=pag, and round-trips through the CSR."""
    from andrey.core.structure import GraphStructure

    ds = data.sample_scm(d=20, n=64, seed=3, density=3.0, latents=3)
    mag = data.latent.marginal(ds.graph)
    assert mag.kind == "pag"
    assert mag.n_nodes == ds.data.shape[1]  # over observed nodes
    dense = mag.to_numpy()
    assert np.array_equal(dense != 0, (dense != 0).T)  # endpoint-mark support is symmetric
    # >= 1 bidirected edge (ARROW/ARROW off-diagonal)
    n = mag.n_nodes
    bidirected = sum(
        1 for i in range(n) for j in range(i + 1, n) if tuple(mag.endpoints(i, j)) == (2, 2)
    )
    assert bidirected >= 1
    # lossless CSR round-trip
    assert GraphStructure.from_numpy(mag.to_numpy(), kind="pag") == mag


def test_mag_no_latents_is_identity():
    ds = data.sample_scm(d=12, n=32, seed=0)
    assert data.latent.marginal(ds.graph) is ds.graph  # nothing to project


def test_mag_max_nodes_guard_raises():
    ds = data.sample_scm(d=30, n=32, seed=0, density=3.0, latents=2)
    with pytest.raises(ValueError, match="max_nodes"):
        data.latent.marginal(ds.graph, max_nodes=5)


def test_marginal_requires_observed_first():
    """marginal raises (not silently wrong) if observed nodes are not indexed first."""

    # latent node 0 first -> not observed-first
    g = _labeled_dag(parents=[0, 0], children=[1, 2], d=3, latent_nodes={0})
    with pytest.raises(ValueError, match="observed nodes indexed first"):
        data.latent.marginal(g)


# ---- true PAG (MAG-first oracle FCI) ------------------------------------------------------------


def test_pag_covered_paths_leave_circles():
    """The five-node pcalg dsepTest oracle PAG keeps A o-> C and M o-> C."""
    from andrey.core.structure import ARROW, CIRCLE

    # B -> M, B -> A, M -> A, M -> C, A -> C, X -> C.
    b, m, a, c, x = range(5)
    dag = _labeled_dag(
        parents=[b, b, m, m, a, x], children=[m, a, a, c, c, c], d=5, latent_nodes=set()
    )
    pag = data.latent.marginal(dag, target="pag")
    assert tuple(pag.endpoints(a, c)) == (CIRCLE, ARROW)
    assert tuple(pag.endpoints(m, c)) == (CIRCLE, ARROW)

    # Full oracle golden: the triangle stays o-o and C receives three arrowheads.
    expected = np.zeros((5, 5), dtype=np.int8)
    for u, v in [(b, m), (b, a), (m, a), (m, c), (a, c), (x, c)]:
        expected[u, v] = expected[v, u] = CIRCLE
    expected[c, [m, a, x]] = ARROW
    np.testing.assert_array_equal(pag.to_numpy(), expected)


def test_pag_orients_unshielded_collider():
    """PAG orients a *-> b <-* c collider (arrowheads at b); an unidentified pair stays o-o."""
    from andrey.core.structure import ARROW, CIRCLE

    # collider a(0)->b(1)<-c(2), a/c non-adjacent; latent L(5) confounds x(3),y(4)
    g = _labeled_dag(parents=[0, 2, 5, 5], children=[1, 1, 3, 4], d=6, latent_nodes={5})
    pag = data.latent.marginal(g, target="pag")
    assert pag.kind == "pag" and pag.n_nodes == 5
    assert pag.endpoints(0, 1)[1] == ARROW  # arrowhead at b from a
    assert pag.endpoints(2, 1)[1] == ARROW  # arrowhead at b from c
    assert tuple(pag.endpoints(3, 4)) == (CIRCLE, CIRCLE)  # confounded pair unidentified


def test_pag_no_tail_tail_invariant():
    """No selection variables -> the PAG has no undirected (TAIL-TAIL) edge."""
    from andrey.core.structure import TAIL

    for seed in range(5):
        ds = data.sample_scm(d=15, n=32, seed=seed, density=3.0, latents=2)
        m = data.latent.marginal(ds.graph, target="pag").to_numpy()
        assert not np.any((m == TAIL) & (m.T == TAIL))


def test_pag_invariant_marks_match_the_mag():
    """The MAG is a member of the PAG's class, so every non-circle PAG mark equals the MAG's.

    Run over many seeds -- this also guards against an unsound orientation (for example, a
    spurious tail) leaking in, which would show up as a resolved PAG mark that disagrees with the
    true MAG.
    """
    from andrey.core.structure import CIRCLE, NULL

    for seed in range(12):
        ds = data.sample_scm(d=15, n=32, seed=seed, density=3.0, latents=2)
        mag = data.latent.marginal(ds.graph, target="mag").to_numpy()
        pag = data.latent.marginal(ds.graph, target="pag").to_numpy()
        resolved = (pag != CIRCLE) & (pag != NULL)  # invariant (oriented) marks
        assert np.array_equal(pag[resolved], mag[resolved]), seed


def test_pag_permutation_equivariant():
    """Relabeling observed nodes transforms the PAG equivariantly (catches order-dependence)."""
    ds = data.sample_scm(d=10, n=32, seed=1, density=3.0, latents=2)
    base = data.latent.marginal(ds.graph, target="pag").to_numpy()
    n_obs, d = base.shape[0], ds.graph.n_nodes
    lat = set(np.flatnonzero(ds.graph.node_types == LATENT).tolist())
    pi = np.arange(d)
    pi[:n_obs] = np.random.default_rng(0).permutation(n_obs)  # permute observed only; latents fixed
    e = ds.params.edges
    perm_pag = data.latent.marginal(
        _labeled_dag(pi[e[:, 0]], pi[e[:, 1]], d, latent_nodes=lat), target="pag"
    ).to_numpy()
    assert np.array_equal(perm_pag[np.ix_(pi[:n_obs], pi[:n_obs])], base)


def test_pag_roundtrip_and_gate():
    from andrey.core.structure import GraphStructure

    ds = data.sample_scm(d=12, n=32, seed=0, density=3.0, latents=2)
    pag = data.latent.marginal(ds.graph, target="pag")
    assert GraphStructure.from_numpy(pag.to_numpy(), kind="pag") == pag  # lossless CSR round-trip
    with pytest.raises(ValueError, match="max_nodes"):
        data.latent.marginal(ds.graph, target="pag", max_nodes=3)
    with pytest.raises(ValueError, match="target must be"):
        data.latent.marginal(ds.graph, target="cpdag")


def test_pag_no_latents_is_cpdag():
    """target='pag' on a causally-sufficient DAG returns its CPDAG (a v-structure), not the DAG."""
    from andrey.core.structure import ARROW

    g = _labeled_dag(parents=[0, 1], children=[2, 2], d=3, latent_nodes=set())  # collider 0->2<-1
    cpdag = data.latent.marginal(g, target="pag")
    assert cpdag is not g  # not the identity early-return
    assert (
        cpdag.endpoints(0, 2)[1] == ARROW and cpdag.endpoints(1, 2)[1] == ARROW
    )  # oriented collider
