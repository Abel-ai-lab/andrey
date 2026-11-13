"""Recovery tests for ``local_isa_ling`` (Dai et al. 2024).

Exact checks run the post-processing on the population ISA solution ``W = A^{-1}_{S,S}`` of a known
model (``A = (I - B)^{-1}``, ``S`` the target and its blanket), whose rows are shuffled and whose
subspaces are mixed by random invertible blocks, as an ISA leaves them: Algorithm 1 must return
the edges into the target and its children of every model of the LiNG equivalence class
(Theorem 3), the stable one first when cycles are disjoint (Corollary 1), and Algorithm 3 the true
edges of a DAG. Data checks run the method on the paper's examples and simulation: weights drawn
from ``U([-0.9, -0.5] u [0.5, 0.9])`` and noise ``U(-c, c)^5`` with ``c`` from ``U(0.75, 1.25)``.

``B[child, parent]`` holds the weight of ``parent -> child`` throughout, as in ``X = B X + E``.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

import andrey
from andrey.lingam.local import admissible, algorithm1, algorithm3
from andrey.metrics import target_oracle

TOL = 1e-9


def blanket(B: np.ndarray, t: int) -> list[int]:
    """Parents, children, and the children's other parents of ``t``."""
    children = np.flatnonzero(B[:, t])
    mb = set(np.flatnonzero(B[t])) | set(children)
    for c in children:
        mb |= set(np.flatnonzero(B[c]))
    mb.discard(t)
    return sorted(int(j) for j in mb)


def local_answer(B: np.ndarray, t: int) -> dict:
    """The edges into ``t`` and into each child of ``t`` in ``B``, with their weights."""
    heads = [t, *(int(j) for j in np.flatnonzero(B[:, t]))]
    return {(int(k), j): float(B[j, k]) for j in heads for k in np.flatnonzero(B[j])}


def equivalence_class(B: np.ndarray, t: int) -> list[tuple[dict, bool]]:
    """``local_answer`` of every model of B's LiNG equivalence class, with its stability: every row
    permutation of ``I - B`` with a nonzero diagonal, scaled to ones there (Lacerda et al. 2008)."""
    d = len(B)
    found = []
    for perm in itertools.permutations(range(d)):
        W = (np.eye(d) - B)[list(perm)]
        if np.abs(np.diag(W)).min() < TOL:
            continue
        other = np.eye(d) - W / np.diag(W)[:, None]
        other[np.abs(other) < TOL] = 0.0
        found.append((local_answer(other, t), bool(np.abs(np.linalg.eigvals(other)).max() < 1)))
    return found


def population(B: np.ndarray, t: int, rng: np.random.Generator):
    """The ISA solution of the target's blanket in the population, rows shuffled and each
    subspace mixed, its subspaces, the local columns, and the target's local position."""
    d = len(B)
    A = np.linalg.inv(np.eye(d) - B)
    S = sorted([t, *blanket(B, t)])
    rest = [j for j in range(d) if j not in S]
    W = np.linalg.inv(A[np.ix_(S, S)])
    W[np.abs(W) < TOL] = 0.0
    hidden = np.abs(W @ A[np.ix_(S, rest)]) > TOL  # the outside noises each component mixes
    lt = S.index(t)
    singles = set(np.flatnonzero(W[:, lt]))
    groups = [[i] for i in range(len(S))]
    for i, j in itertools.combinations(range(len(S)), 2):
        if i in singles or j in singles or not (hidden[i] & hidden[j]).any():
            continue
        a = next(g for g in groups if i in g)
        b = next(g for g in groups if j in g)
        if a is not b:
            a.extend(b)
            groups.remove(b)
    order = rng.permutation(len(S))
    mixed, parts, at = np.zeros_like(W), [], 0
    for rows in groups:
        k = len(rows)
        block = rng.uniform(0.5, 2.0, (k, k)) * rng.choice([-1, 1], (k, k))
        new = sorted(int(r) for r in order[at : at + k])
        mixed[new] = block @ W[rows]
        parts.append(new)
        at += k
    return mixed, sorted(parts), S, lt


def named(edges: dict, S: list[int]) -> dict:
    return {(S[k], S[j]): w for (k, j), w in edges.items()}


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(abs(a[e] - b[e]) < 1e-7 for e in a)


def random_cyclic(rng: np.random.Generator) -> np.ndarray:
    d = int(rng.integers(4, 8))
    B = np.zeros((d, d))
    for _ in range(int(rng.integers(d, 2 * d))):
        i, j = rng.choice(d, 2, replace=False)
        B[i, j] = rng.uniform(0.5, 0.9) * rng.choice([-1, 1])
    return B


def disjoint_cycles(rng: np.random.Generator) -> np.ndarray:
    """Blocks in a random order, each a node or a simple cycle, joined by edges from earlier blocks
    to later ones: every cycle is disjoint from the others."""
    d = int(rng.integers(4, 8))
    order, blocks, at = rng.permutation(d), [], 0
    while at < d:
        k = int(rng.choice([1, 1, 2, 3]))
        blocks.append([int(v) for v in order[at : at + k]])
        at += k
    B = np.zeros((d, d))
    for block in blocks:
        if len(block) > 1:
            for a, b in zip(block, block[1:] + block[:1]):
                B[b, a] = rng.uniform(0.5, 0.9) * rng.choice([-1, 1])
    for x, y in itertools.combinations(range(len(blocks)), 2):
        if rng.random() < 0.4:
            B[rng.choice(blocks[y]), rng.choice(blocks[x])] = rng.uniform(0.5, 0.9) * rng.choice(
                [-1, 1]
            )
    return B


def models(W: np.ndarray, parts, lt: int, S) -> list:
    return [(named(m.edges, S), m.stable) for m in algorithm1(W, parts, lt, TOL)]


# --- the paper's examples, in the population ----------------------------------------------------


def chain_model(edges: dict[tuple[int, int], float], d: int) -> np.ndarray:
    B = np.zeros((d, d))
    for (parent, child), w in edges.items():
        B[child, parent] = w
    return B


def test_example_1_two_signals_of_one_hidden_cause_form_one_subspace():
    """Fig. 1(i): X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4 with T = X4: X2 and X3 share X1's noise,
    one two-dimensional subspace, and the answer is the two edges into X4."""
    B = chain_model({(0, 1): 0.7, (0, 2): -0.6, (1, 3): 0.8, (2, 3): 0.5}, 4)
    W, parts, S, lt = population(B, 3, np.random.default_rng(0))
    assert sorted(len(p) for p in parts) == [1, 2]
    assert models(W, parts, lt, S) == [({(1, 3): 0.8, (2, 3): 0.5}, True)] or same(
        models(W, parts, lt, S)[0][0], {(1, 3): 0.8, (2, 3): 0.5}
    )


def test_example_2_reads_the_edges_into_the_target_from_the_ISA_matrix():
    """Fig. 1(ii): the ISA matrix [[1,0,0],[-ac,1,0],[-b,-d,1]] of S = (X1, X3, X4) is not
    I - B_SS, yet its target row gives the edges X1 -> X4 (b) and X3 -> X4 (d)."""
    a, b, c, d = 0.6, -0.7, 0.8, 0.5
    W = np.array([[1, 0, 0], [-a * c, 1, 0], [-b, -d, 1]])
    (edges, _), *rest = models(W[[2, 0, 1]], [[0], [1], [2]], 2, [1, 3, 4])
    assert not rest
    assert same(edges, {(1, 4): b, (3, 4): d})


def test_example_3_a_cycle_outside_the_blanket_leaves_the_target_row():
    """Fig. 1(iii): X1 <-> X2 cycle, X2 -> X3; the ISA matrix [[1 - ab, 0], [-c, 1]] of
    S = (X2, X3) has no one on X2's diagonal, and the answer is X2 -> X3 (c)."""
    a, b, c = 0.6, 0.7, -0.8
    W = np.array([[1 - a * b, 0], [-c, 1]])
    assert [m[0] for m in models(W, [[0], [1]], 1, [2, 3])] == [{(2, 3): c}]


def test_example_4_only_invertible_blocks_are_admissible():
    """Fig. 2: with the subspace {X1, X2} mixed, a permutation with a nonzero diagonal (ii) puts
    a singular block on it; the one admissible assignment gives the true edges."""
    a, b, c, d, e = 0.6, -0.7, 0.8, 0.5, -0.9
    rows = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, -c, 0],
            [-a, 0, 1, 0, 0],
            [0, 0, -b, 1, 0],
            [0, -e, -d, 0, 1],
        ],
        dtype=float,
    )
    W = rows.copy()
    W[:2] = np.array([[1.0, 2.0], [-1.5, 0.5]]) @ rows[:2]
    parts = [[0, 1], [2], [3], [4]]
    assert admissible(W, parts, TOL) == [[(0, 1), (2,), (3,), (4,)]]
    # (ii): rows 1, 2 to columns X2, X4 is a nonzero diagonal but a rank-one block
    assert np.linalg.matrix_rank(W[np.ix_([0, 1], [1, 3])]) == 1
    (edges, stable), *rest = models(W, parts, 2, [1, 2, 3, 4, 5])
    assert not rest and stable
    assert same(edges, {(1, 3): a, (3, 4): b, (3, 5): d, (2, 5): e})


#: Fig. 3: X6 -> X1, X6 -> X2, X6 -> X7, X1 -> X2, X1 -> X4 (b), X2 -> X3 (a), X3 -> X4 (c),
#: X3 -> X5 (d), X5 -> X7, X7 -> X2; the cycle X2 -> X3 -> X5 -> X7 -> X2 passes the target X3.
#: Positions are the paper's numbers less one.
FIGURE_3 = {
    (5, 0): 0.7,
    (5, 1): -0.6,
    (5, 6): 0.8,
    (0, 1): 0.5,
    (0, 3): 0.8,
    (1, 2): 0.9,
    (2, 3): -0.7,
    (2, 4): 0.6,
    (4, 6): 0.7,
    (6, 1): -0.5,
}


def test_example_5_returns_both_models_of_the_cycle_through_the_target():
    """Fig. 3: two models, the true one and the one with the cycle reversed, whose edges at X3
    are X5 -> X3 (1/d) and X3 -> X2 (1/a); the stable one first."""
    B = chain_model(FIGURE_3, 7)
    W, parts, S, lt = population(B, 2, np.random.default_rng(1))
    assert S == [0, 1, 2, 3, 4]
    found = models(W, parts, lt, S)
    a, b, c, d = 0.9, 0.8, -0.7, 0.6
    true = {(1, 2): a, (2, 3): c, (0, 3): b, (2, 4): d}
    reversed_ = {(4, 2): 1 / d, (2, 1): 1 / a, (2, 3): c, (0, 3): b}
    assert len(found) == 2
    assert same(found[0][0], true) and found[0][1]
    assert same(found[1][0], reversed_) and not found[1][1]


# --- the population: Theorem 3, Corollary 1, Algorithm 3 ---------------------------------------


@pytest.mark.parametrize("seed", range(4))
def test_algorithm1_returns_the_local_part_of_every_equivalent_model(seed):
    """Theorem 3 on random cyclic graphs of 4 to 7 variables: the answers equal the local parts
    of the LiNG equivalence class, whatever the rows' order and the subspaces' mixing."""
    rng = np.random.default_rng(seed)
    checked = 0
    while checked < 15:
        B = random_cyclic(rng)
        if abs(np.linalg.det(np.eye(len(B)) - B)) < 1e-6:
            continue
        t = int(rng.integers(len(B)))
        if not blanket(B, t):
            continue
        W, parts, S, lt = population(B, t, rng)
        got = [edges for edges, _ in models(W, parts, lt, S)]
        want = [edges for edges, _ in equivalence_class(B, t)]
        assert all(any(same(g, w) for w in want) for g in got)
        assert all(any(same(g, w) for g in got) for w in want)
        checked += 1


@pytest.mark.parametrize("seed", range(4))
def test_with_disjoint_cycles_the_one_stable_model_is_the_truth(seed):
    """Corollary 1: with disjoint cycles and a stable model, exactly one answer is stable, it
    comes first, and it is the true one."""
    rng = np.random.default_rng(100 + seed)
    checked = 0
    while checked < 15:
        B = disjoint_cycles(rng)
        t = int(rng.integers(len(B)))
        if np.abs(np.linalg.eigvals(B)).max() >= 1 or not blanket(B, t):
            continue
        W, parts, S, lt = population(B, t, rng)
        found = models(W, parts, lt, S)
        assert [stable for _, stable in found].count(True) == 1
        assert found[0][1] and same(found[0][0], local_answer(B, t))
        checked += 1


@pytest.mark.parametrize("seed", range(3))
def test_algorithm3_reads_a_dag_and_agrees_with_the_target_oracle(seed):
    """Appendix C on random DAGs: the sink peeling returns the true edges, whose edges at the
    target are those of ``andrey.metrics.target_oracle``."""
    rng = np.random.default_rng(200 + seed)
    checked = 0
    while checked < 15:
        d = int(rng.integers(4, 9))
        B = np.tril(rng.uniform(0.5, 0.9, (d, d)) * rng.choice([-1, 1], (d, d)), -1)
        B *= rng.random((d, d)) < 0.4
        t = int(rng.integers(d))
        if not blanket(B, t):
            continue
        W, parts, S, lt = population(B, t, rng)
        edges = algorithm3(np.where(np.abs(W) > TOL, W, 0.0), lt)
        assert edges is not None
        truth = local_answer(B, t)
        assert same(named(edges, S), truth)
        oracle = target_oracle(B.T != 0, t, "dag").to_numpy()
        at_target = {e for e in truth if t in e}
        assert at_target == {
            (i, j)
            for i, j in itertools.permutations(range(d), 2)
            if oracle[i, j] == 1 and oracle[j, i] == 2
        }
        checked += 1


# --- data ----------------------------------------------------------------------------------------


def sample(B: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Rows of the equilibrium ``X = (I - B)^{-1} E`` with the paper's noise."""
    c = rng.uniform(0.75, 1.25, len(B))
    E = rng.uniform(-c, c, size=(n, len(B))) ** 5
    return np.linalg.solve(np.eye(len(B)) - B, E.T).T


def fit(X, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        return andrey.local_isa_ling(X, **kwargs)


def answer(out) -> dict:
    W = out.weighted_adjacency
    return {(int(i), int(j)): float(W[i, j]) for i, j in zip(*np.nonzero(W))}


@pytest.fixture(scope="module")
def figure_3_fit():
    B = chain_model(FIGURE_3, 7)
    return B, fit(sample(B, 3000, np.random.default_rng(3)), target=2)


def test_figure_3_from_data_finds_both_models_and_the_stable_one_first(figure_3_fit):
    B, out = figure_3_fit
    assert out.structure.kind == "digraph"
    assert out.metadata["markov_blanket"] == [0, 1, 3, 4]
    truth = local_answer(B, 2)
    assert answer(out).keys() == truth.keys()
    assert all(abs(answer(out)[e] - truth[e]) < 0.15 for e in truth)
    stable = [m for m in out.metadata["equivalent_models"] if m["stable"]]
    assert len(out.metadata["equivalent_models"]) == 2 and len(stable) == 1
    reversed_ = {(e[0], e[1]) for e in out.metadata["equivalent_models"][1]["edges"]}
    assert reversed_ == {(4, 2), (2, 1), (2, 3), (0, 3)}


def test_figure_3_lists_the_target_sets_of_the_first_model(figure_3_fit):
    _, out = figure_3_fit
    assert out.metadata["parents"] == [1]
    assert out.metadata["children"] == [3, 4]
    assert out.metadata["equivalent_models"][0]["edges"] == [
        [k, j, pytest.approx(w)] for (k, j), w in sorted(answer(out).items())
    ]


def er_dag(d: int, degree: float, rng: np.random.Generator) -> np.ndarray:
    B = np.tril(rng.random((d, d)) < degree / (d - 1), -1).astype(float)
    B *= rng.uniform(0.5, 0.9, (d, d)) * rng.choice([-1, 1], (d, d))
    order = rng.permutation(d)
    return B[np.ix_(order, order)]


def local_shd(found: dict, truth: dict) -> int:
    """Missing, extra, and reversed edges of the answer against the true local edges."""
    missing = {e for e in truth if e not in found}
    extra = {e for e in found if e not in truth}
    reversed_ = {e for e in extra if e[::-1] in missing}
    return len(missing) + len(extra) - len(reversed_)


@pytest.mark.parametrize("acyclic", [False, True])
def test_on_the_papers_acyclic_simulation_it_beats_the_empty_answer(acyclic):
    """Erdos-Renyi DAG over 20 variables, mean degree 3, 2,000 rows: over three targets the
    local SHD is well below the empty answer's, and the weights lie near the true ones."""
    rng = np.random.default_rng(7)
    B = er_dag(20, 3, rng)
    X = sample(B, 2000, rng)
    targets = [t for t in range(20) if 2 <= len(blanket(B, t)) <= 7][:3]
    errors = empty = 0
    for t in targets:
        out = fit(X, target=t, acyclic=acyclic)
        truth = local_answer(B, t)
        found = answer(out)
        errors += local_shd(found, truth)
        empty += len(truth)
        close = [abs(found[e] - truth[e]) < 0.2 for e in truth if e in found]
        assert sum(close) >= 0.8 * len(close)
    assert errors <= empty // 4
