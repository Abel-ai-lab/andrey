"""Adjustment sets for the total effect of a treatment on an outcome, read from a graph."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from andrey import GraphStructure
from andrey.core import adjustment as adj
from andrey.core.structure import ARROW, TAIL
from andrey.metrics import adjustment_scores, to_cpdag


def _graph(labels: str, edges: str, kind: str = "dag") -> GraphStructure:
    """A labeled graph from ``"A>B B-C D<>E F*>G F**H"``: ``>`` directed, ``-`` undirected,
    ``<>`` bidirected, ``*>`` circle-arrow, ``**`` circle-circle."""
    names = labels.split()
    ix = {name: i for i, name in enumerate(names)}
    marks = np.zeros((len(names), len(names)), dtype=int)
    glyphs = {"<>": (2, 2), "*>": (3, 2), "**": (3, 3), ">": (1, 2), "-": (1, 1)}
    for edge in edges.split():
        for glyph, (near, far) in glyphs.items():
            if glyph in edge:
                a, b = edge.split(glyph)
                marks[ix[a], ix[b]], marks[ix[b], ix[a]] = near, far
                break
    return GraphStructure.from_numpy(marks, kind=kind, labels=tuple(names))


def _dag_of(D: np.ndarray, labels: tuple[str, ...] | None = None) -> GraphStructure:
    marks = np.zeros(D.shape, dtype=np.int8)
    marks[D] = TAIL
    marks[D.T] = ARROW
    return GraphStructure.from_numpy(marks, kind="dag", labels=labels)


def _random_dag(rng: np.random.Generator, d: int, p: float) -> np.ndarray:
    order = rng.permutation(d)
    D = np.zeros((d, d), dtype=bool)
    for a, b in itertools.combinations(range(d), 2):
        if rng.random() < p:
            D[order[a], order[b]] = True
    return D


def _subsets(nodes: list[int]) -> list[list[int]]:
    return [list(s) for k in range(len(nodes) + 1) for s in itertools.combinations(nodes, k)]


def _members(cpdag: GraphStructure) -> list[GraphStructure]:
    """Every DAG of a CPDAG's class, by orienting its undirected edges every way."""
    marks = cpdag.to_numpy()
    lines = [(i, j) for i, j in zip(*np.nonzero(np.triu(adj.undirected(marks))), strict=True)]
    found = []
    for flips in itertools.product((False, True), repeat=len(lines)):
        D = adj.directed(marks)
        for (i, j), flip in zip(lines, flips, strict=True):
            D[i, j], D[j, i] = not flip, flip
        if _acyclic(D) and np.array_equal(to_cpdag(D.astype(int)).to_numpy(), marks):
            found.append(_dag_of(D, cpdag.labels))
    return found


def _acyclic(D: np.ndarray) -> bool:
    return not any(i in adj.reach(D, np.flatnonzero(D[i]).tolist()) for i in range(len(D)))


def _linear(rng: np.random.Generator, D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The covariance and the total effects of a linear model on ``D`` with unit error
    variances: ``x_j = sum_i B[i, j] x_i + e_j``."""
    d = len(D)
    B = D * rng.uniform(0.5, 1.5, (d, d)) * rng.choice((-1, 1), (d, d))
    total = np.linalg.inv(np.eye(d) - B)
    return total.T @ total, total


def _coefficient(cov: np.ndarray, x: int, y: int, z: list[int]) -> float:
    cols = [x, *z]
    return float(np.linalg.solve(cov[np.ix_(cols, cols)], cov[cols, y])[0])


def _variance(cov: np.ndarray, x: int, y: int, z: list[int]) -> float:
    """The asymptotic variance of the coefficient of ``x``, up to the sample size: the residual
    variance of ``y`` given ``x`` and ``z`` over that of ``x`` given ``z``."""

    def residual(v: int, given: list[int]) -> float:
        if not given:
            return float(cov[v, v])
        g = cov[np.ix_(given, given)]
        return float(cov[v, v] - cov[v, given] @ np.linalg.solve(g, cov[given, v]))

    return residual(y, [x, *z]) / residual(x, z)


def test_validity_is_the_regression_coefficient_in_linear_models():
    # A valid set's population coefficient is the total effect, in every linear model on the
    # DAG; an invalid set's differs for generic weights.
    rng = np.random.default_rng(1)
    cases = 0
    for _ in range(20):
        d = int(rng.integers(4, 7))
        D = _random_dag(rng, d, rng.uniform(0.3, 0.7))
        dag = _dag_of(D)
        cov, total = _linear(rng, D)
        for x, y in itertools.permutations(range(d), 2):
            for z in _subsets([v for v in range(d) if v not in (x, y)]):
                recovered = abs(_coefficient(cov, x, y, z) - total[x, y]) < 1e-9
                assert dag.is_adjustment_set(x, y, z) == recovered
                cases += 1
    assert cases > 5000


def test_a_cpdag_answers_as_every_dag_of_its_class():
    rng = np.random.default_rng(2)
    pairs = amenable = 0
    for _ in range(20):
        d = int(rng.integers(4, 7))
        cpdag = to_cpdag(_random_dag(rng, d, rng.uniform(0.3, 0.8)).astype(int))
        members = _members(cpdag)
        for x, y in itertools.permutations(range(d), 2):
            sets = _subsets([v for v in range(d) if v not in (x, y)])
            for z in sets:
                every = all(m.is_adjustment_set(x, y, z) for m in members)
                assert cpdag.is_adjustment_set(x, y, z) == every
            exists = any(all(m.is_adjustment_set(x, y, z) for m in members) for z in sets)
            found = adj.adjustment_set(cpdag, x, y)
            assert (found is not None) == exists
            assert found is None or cpdag.is_adjustment_set(x, y, found)
            # Existence is amenability with the outcome not a parent of the treatment.
            parent = adj.directed(cpdag.to_numpy())[y, x]
            assert exists == (adj.amenable(cpdag, x, y) and not parent)
            pairs += 1
            amenable += exists
    assert pairs > 300 and 0.2 < amenable / pairs < 0.9


def test_the_optimal_set_is_the_class_s_valid_and_most_precise():
    # Henckel et al. (2022): in a class with an adjustment set, the optimal set is the same in
    # every DAG, valid in each, and no valid set gives a smaller asymptotic variance.
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(40):
        d = int(rng.integers(4, 7))
        D = _random_dag(rng, d, rng.uniform(0.3, 0.7))
        cpdag = to_cpdag(D.astype(int))
        members = _members(cpdag)
        cov, _ = _linear(rng, D)
        dag = _dag_of(D)
        for x, y in itertools.permutations(range(d), 2):
            optimal = cpdag.optimal_adjustment_set(x, y)
            if adj.adjustment_set(cpdag, x, y) is None:
                assert optimal is None
                continue
            # The same in every DAG, None too: the treatment is an ancestor in all or in none.
            assert all(m.optimal_adjustment_set(x, y) == optimal for m in members)
            if optimal is None:
                continue
            assert all(m.is_adjustment_set(x, y, optimal) for m in members)
            best = _variance(cov, x, y, optimal)
            for z in _subsets([v for v in range(d) if v not in (x, y)]):
                if dag.is_adjustment_set(x, y, z):
                    assert best <= _variance(cov, x, y, z) * (1 + 1e-9)
            checked += 1
    assert checked > 50


def test_the_local_parent_sets_are_the_parent_sets_of_the_class():
    # Maathuis et al. (2009): every clique of undirected neighbors gives the parents of x in
    # some DAG of the class, and every DAG gives one of them.
    rng = np.random.default_rng(4)
    for _ in range(60):
        d = int(rng.integers(3, 7))
        cpdag = to_cpdag(_random_dag(rng, d, rng.uniform(0.3, 0.8)).astype(int))
        members = _members(cpdag)
        for x in range(d):
            seen = sorted({tuple(m.parents(x)) for m in members})
            assert [tuple(s) for s in adj.local_parent_sets(cpdag, x)] == seen


def test_answers_do_not_depend_on_node_order_or_labels():
    rng = np.random.default_rng(5)
    for _ in range(8):
        d = int(rng.integers(4, 7))
        D = _random_dag(rng, d, 0.5)
        names = tuple(f"v{i}" for i in rng.permutation(d))
        order = rng.permutation(d)
        for graph in (_dag_of(D, names), to_cpdag(D.astype(int))):
            marks = graph.to_numpy()
            graph = GraphStructure.from_numpy(marks, kind=graph.kind, labels=names)
            moved = GraphStructure.from_numpy(
                marks[np.ix_(order, order)], kind=graph.kind, labels=tuple(names[i] for i in order)
            )
            for x, y in itertools.permutations(names, 2):
                assert graph.optimal_adjustment_set(x, y) == _sorted_like(
                    moved.optimal_adjustment_set(x, y), names
                )
                for z in _subsets([v for v in names if v not in (x, y)])[:4]:
                    assert graph.is_adjustment_set(x, y, z) == moved.is_adjustment_set(x, y, z)


def _sorted_like(found: list | None, names: tuple[str, ...]) -> list | None:
    return None if found is None else sorted(found, key=names.index)


def test_the_set_must_leave_out_the_pair_and_name_nodes():
    g = _graph("A X B Y", "A>X A>B B>Y X>Y")
    with pytest.raises(ValueError, match="neither the treatment nor the outcome"):
        g.is_adjustment_set("X", "Y", ["Y"])
    with pytest.raises(ValueError):
        g.is_adjustment_set("X", "Q", [])
    with pytest.raises(ValueError, match="two nodes"):
        g.optimal_adjustment_set("X", "X")


@pytest.mark.parametrize("kind", ["digraph", "undirected"])
def test_kinds_without_a_criterion_raise(kind):
    edges = "A>B" if kind == "digraph" else "A-B"
    g = _graph("A B", edges, kind=kind)
    with pytest.raises(ValueError, match="defined for graphs of kind"):
        g.is_adjustment_set("A", "B", [])


def test_an_mpdag_is_not_read_as_a_cpdag():
    # A - B - C with knowledge A -> B: Meek's rules direct B -> C, so the graph is an MPDAG.
    g = _graph("A B C", "A>B B>C", kind="cpdag")
    with pytest.raises(ValueError, match="MPDAG"):
        g.is_adjustment_set("A", "C", [])


def test_a_dag_s_optimal_set_and_a_zero_effect():
    g = _graph("A X B Y", "A>X A>B B>Y X>Y")
    assert g.optimal_adjustment_set("X", "Y") == ["B"]
    assert g.optimal_adjustment_set("Y", "X") is None  # Y is not an ancestor of X
    assert g.optimal_adjustment_set("X", "Y", index=True) == [2]


def test_adjustment_scores_follow_their_definition():
    g = _graph("A X B Y", "A>X A>B B>Y X>Y")
    s = adjustment_scores(["A"], g, "X", "Y")
    assert (s["valid"], s["optimal"], s["precision"], s["recall"], s["f1"]) == (
        True,
        ["B"],
        0.0,
        0.0,
        0.0,
    )
    s = adjustment_scores(["A", "B"], g, "X", "Y")
    assert (s["valid"], s["precision"], s["recall"]) == (True, 0.5, 1.0)
    assert s["f1"] == pytest.approx(2 / 3)
    s = adjustment_scores([], g, "X", "Y")
    assert (s["valid"], s["precision"], s["recall"], s["f1"]) == (False, 1.0, 0.0, 0.0)
    # No optimal set: the treatment is not an ancestor of the outcome.
    s = adjustment_scores([], g, "Y", "X")
    assert s["optimal"] is None and np.isnan(s["precision"]) and np.isnan(s["f1"])
    # Against the class: X - Y could point either way, so no set is valid in every DAG.
    lone = to_cpdag(np.array([[0, 1], [0, 0]]))
    s = adjustment_scores([], lone, 0, 1)
    assert s["valid"] is False and s["optimal"] is None


# Worked examples of Perković et al. (2018); "o-o" edges are written "**", "o->" edges "*>".
FIG_1A = _graph("I A B Z X Y", "I-A A-B A-Z Z-B I>X A>X Z>X Z>Y B>Y X>Y", kind="cpdag")
FIG_3A = _graph("V1 X V2 Y", "V1**X X**V2 V2**Y X**Y", kind="pag")
FIG_3B = _graph("V1 X V2 Y", "X>V1 X>V2 V2>Y X>Y", kind="mag")
FIG_3C = _graph("V1 X V2 Y", "V1>X X>V2 V2>Y X>Y", kind="mag")
FIG_4A = _graph("V1 V2 V3 V4 X Y", "V1*>X V2*>X V3*>X X>Y X>V4 V3>V4 V4*>Y V3>Y", kind="pag")
FIG_4B = _graph("V1 V2 V3 V4 X Y", "V1*>X V2*>X X>V4 X<>V3 V3<>V4 V4>Y V3>Y", kind="pag")
FIG_5 = _graph("V1 V2 X1 X2 Y", "V1>X1 V2>X1 V2>Y X1>Y Y<>X2", kind="mag")


def _valid_sets(graph: GraphStructure, x: str, y: str) -> list[set[str]]:
    others = [v for v in graph.labels if v not in (x, y)]
    return [set(z) for z in _subsets(others) if graph.is_adjustment_set(x, y, z)]


def test_the_cpdag_of_figure_1a():
    # Example 2: amenable, Forb = {Y}, and every superset of {Z, A} or {Z, B} is valid; Example 5:
    # the constructive set {I, A, Z, B} is one.
    assert adj.amenable(FIG_1A, 4, 5)
    assert adj.forbidden(FIG_1A, 4, 5) == {5}
    expected = [
        z for z in map(set, _subsets(["I", "A", "B", "Z"])) if {"A", "Z"} <= z or {"B", "Z"} <= z
    ]
    assert sorted(map(sorted, _valid_sets(FIG_1A, "X", "Y"))) == sorted(map(sorted, expected))
    assert adj.adjustment_set(FIG_1A, 4, 5) == [0, 1, 2, 3]


def test_the_graphs_of_figure_3():
    # Example 3: the PAG is not amenable (X o-o Y), nor is M1 as a MAG (X -> Y is invisible),
    # though M1 read as a DAG is; M2 is, and the empty set is valid there.
    assert not adj.amenable(FIG_3A, 1, 3) and _valid_sets(FIG_3A, "X", "Y") == []
    assert not adj.amenable(FIG_3B, 1, 3) and _valid_sets(FIG_3B, "X", "Y") == []
    as_dag = GraphStructure.from_numpy(FIG_3B.to_numpy(), kind="dag", labels=FIG_3B.labels)
    assert adj.amenable(as_dag, 1, 3) and as_dag.is_adjustment_set("X", "Y", [])
    assert adj.amenable(FIG_3C, 1, 3) and FIG_3C.is_adjustment_set("X", "Y", [])


def test_the_pags_of_figure_4():
    # Example 4: both PAGs are amenable with Forb = {V4, Y}; P1's valid sets are {V3} with any
    # of V1 and V2, and P2 has none, so amenability does not make a set exist (Example 6).
    for pag in (FIG_4A, FIG_4B):
        assert adj.amenable(pag, 4, 5) and adj.forbidden(pag, 4, 5) == {3, 5}
    expected = [{"V3"}, {"V1", "V3"}, {"V2", "V3"}, {"V1", "V2", "V3"}]
    assert sorted(map(sorted, _valid_sets(FIG_4A, "X", "Y"))) == sorted(map(sorted, expected))
    assert adj.adjustment_set(FIG_4A, 4, 5) == [0, 1, 2]
    assert _valid_sets(FIG_4B, "X", "Y") == [] and adj.adjustment_set(FIG_4B, 4, 5) is None


def test_the_mag_of_figure_5():
    # {V2} is an adjustment set for (X1, Y): X1 -> Y is visible through V1 -> X1.
    assert FIG_5.is_adjustment_set("X1", "Y", ["V2"])
    assert not FIG_5.is_adjustment_set("X1", "Y", [])


def _simple_paths(adjacent: np.ndarray, start: int, end: int | None = None) -> list[list[int]]:
    """Every path (distinct nodes) from ``start``, to ``end`` when it is given."""
    found, stack = [], [[start]]
    while stack:
        path = stack.pop()
        if len(path) > 1 and (end is None or path[-1] == end):
            found.append(path)
            if end is not None:
                continue
        for w in np.flatnonzero(adjacent[path[-1]]).tolist():
            if w not in path:
                stack.append([*path, w])
    return found


def _verbatim(graph: GraphStructure, x: int, y: int):
    """The generalized adjustment criterion for ``(x, y)`` as Definition 4 of Perković et al.
    (2018) states it, path by path: amenability, the forbidden set, and blocking of every proper
    non-causal path of definite status; a function of the candidate set. A CPDAG's undirected
    edge reads as ``o-o``, as there."""
    circle = 3
    m = graph.to_numpy().astype(int)
    m[(m == TAIL) & (m.T == TAIL)] = circle
    adjacent = m != 0
    out = [list(_simple_paths(adjacent, v)) for v in range(len(m))]

    def possibly_causal(p: list[int]) -> bool:
        return all(m[a, b] != ARROW for a, b in itertools.pairwise(p))

    def directed_path(p: list[int]) -> bool:
        return all(m[a, b] == TAIL and m[b, a] == ARROW for a, b in itertools.pairwise(p))

    def visible(a: int, b: int) -> bool:
        if graph.kind in ("dag", "cpdag"):
            return True
        for p in out[a]:  # p = a, ..., v: a collider path from v into a
            v, inner = p[-1], p[1:-1]
            if v == b or adjacent[v, b] or m[a, p[1]] != ARROW:
                continue
            triples = zip(p, p[1:], p[2:], strict=False)
            if all(m[u, s] == ARROW and m[u, t] == ARROW for s, u, t in triples) and all(
                m[u, b] == TAIL and m[b, u] == ARROW for u in inner
            ):
                return True
        return False

    paths = _simple_paths(adjacent, x, y)
    causal = [p for p in paths if possibly_causal(p)]
    if not all(m[x, p[1]] == TAIL and m[p[1], x] == ARROW and visible(x, p[1]) for p in causal):
        return lambda z: False
    on_causal = {w for p in causal for w in p[1:]}
    forbidden = on_causal | {p[-1] for w in on_causal for p in out[w] if possibly_causal(p)}
    descendants = [{v} | {p[-1] for p in out[v] if directed_path(p)} for v in range(len(m))]
    rules = []  # per proper non-causal path of definite status: (colliders, non-colliders)
    for p in paths:
        if possibly_causal(p):
            continue
        colliders, others, definite = [], [], True
        for a, j, b in zip(p, p[1:], p[2:], strict=False):
            if m[j, a] == ARROW and m[j, b] == ARROW:
                colliders.append(j)
            elif (
                m[j, a] == TAIL
                or m[j, b] == TAIL
                or (m[j, a] == circle and m[j, b] == circle and not adjacent[a, b])
            ):
                others.append(j)
            else:
                definite = False
        if definite:
            rules.append((colliders, others))

    def criterion(z: list[int]) -> bool:
        given = set(z)
        if given & forbidden:
            return False
        for colliders, others in rules:
            if all(descendants[j] & given for j in colliders) and not given & set(others):
                return False  # an open path
        return True

    return criterion


def _hidden_graphs(seed: int, count: int) -> list[tuple[GraphStructure, list[str]]]:
    """Random DAGs with one or two hidden nodes, each with the labels of its observed nodes."""
    rng = np.random.default_rng(seed)
    found = []
    for _ in range(count):
        d = int(rng.integers(6, 9))
        names = tuple(f"v{i}" for i in range(d))
        dag = _dag_of(_random_dag(rng, d, rng.uniform(0.25, 0.6)), names)
        hidden = rng.choice(d, size=int(rng.integers(1, 3)), replace=False).tolist()
        found.append((dag, [names[i] for i in hidden]))
    return found


@pytest.mark.parametrize("kind", ["cpdag", "mag", "pag"])
def test_the_criterion_is_definition_4_path_by_path(kind):
    checked = valid = 0
    for dag, hidden in _hidden_graphs(6, 4 if kind == "cpdag" else 12):
        if kind == "cpdag":
            graph = to_cpdag(dag)
            graph = GraphStructure.from_numpy(graph.to_numpy(), kind="cpdag", labels=dag.labels)
        else:
            graph = dag.to_mag(latent=hidden) if kind == "mag" else dag.to_pag(latent=hidden)
        n = graph.n_nodes
        for x, y in itertools.permutations(range(n), 2):
            criterion = _verbatim(graph, x, y)
            for z in _subsets([v for v in range(n) if v not in (x, y)]):
                answer = graph.is_adjustment_set(x, y, z)
                assert answer == criterion(z)
                checked += 1
                valid += answer
    assert checked > 1000 and valid > 50


def test_a_set_valid_in_the_pag_is_valid_in_its_mag_and_dag():
    # Soundness against the generating DAG: the PAG's sets hold in the MAG, whose sets hold in
    # the DAG with its hidden nodes.
    found = 0
    for dag, hidden in _hidden_graphs(7, 8):
        mag, pag = dag.to_mag(latent=hidden), dag.to_pag(latent=hidden)
        names = pag.labels
        for x, y in itertools.permutations(names, 2):
            for z in _subsets([v for v in names if v not in (x, y)]):
                in_pag = pag.is_adjustment_set(x, y, z)
                in_mag = mag.is_adjustment_set(x, y, z)
                assert not in_pag or in_mag
                assert not in_mag or dag.is_adjustment_set(x, y, z)
                found += in_pag
    assert found > 15


# Worked examples of Henckel, Perković and Maathuis (2022).
HPM_1 = _graph("V1 V2 V3 V4 V5 V6", "V1>V2 V1>V4 V2>V5 V3>V5 V4>V5 V4>V6 V5>V6")
HPM_2A = _graph("A B C D X Y", "A>X B>X B>Y C>Y X>D X>Y")
HPM_2B = _graph("A B C D X Y", "A>B C>B A>X D>X X>Y C>Y")
HPM_S7 = _graph(
    "V A1 A2 B1 B2 X M D Y R F",
    "V>X A1>X B1>X A1>A2 B2>B1 X>M X>D M>Y A2>Y R>Y B2>Y Y>F",
)


def _sets(graph: GraphStructure, x: str, y: str) -> list[list[str]]:
    return sorted(sorted(z) for z in _valid_sets(graph, x, y))


def _family(*parts: list[list[str]]) -> list[list[str]]:
    return sorted(sorted(set().union(*combo)) for combo in itertools.product(*parts))


def _nonempty(names: list[str]) -> list[list[str]]:
    return [list(z) for z in _subsets(names) if z]


def test_the_optimal_sets_of_henckel_et_al():
    # Example 3.15 and the supplement's example: the optimal sets, and the valid sets they are
    # chosen from (Examples 3.3 and 3.5).
    assert HPM_1.optimal_adjustment_set("V4", "V6") == ["V2", "V3"]
    assert _sets(HPM_1, "V4", "V6") == _family(_nonempty(["V1", "V2"]), _subsets(["V3"]))
    assert HPM_2A.optimal_adjustment_set("X", "Y") == ["B", "C"]
    assert _sets(HPM_2A, "X", "Y") == _family([["B"]], _subsets(["A", "C", "D"]))
    assert HPM_2B.optimal_adjustment_set("X", "Y") == ["C"]
    assert not HPM_2B.is_adjustment_set("X", "Y", ["B"])  # opens X <- A -> B <- C -> Y
    assert HPM_S7.optimal_adjustment_set("X", "Y") == ["A2", "B2", "R"]
    assert adj.forbidden(HPM_S7, 5, 8) == {6, 8, 10}  # Henckel's forb adds X itself
    expected = _family(_nonempty(["A1", "A2"]), _nonempty(["B1", "B2"]), _subsets(["V", "D", "R"]))
    assert _sets(HPM_S7, "X", "Y") == expected


def test_the_local_parent_sets_of_maathuis_et_al():
    # Figure 2: G_{S->1} is locally valid for S = {}, {X2}, {X4}, not for {X2, X4}.
    cpdag = _graph("X1 X2 X3 X4 Y", "X1-X2 X2-X3 X1-X4 X3>Y X4>Y", kind="cpdag")
    assert adj.local_parent_sets(cpdag, 0) == [[], [1], [3]]


def test_a_mag_or_pag_has_no_optimal_set_here():
    with pytest.raises(ValueError, match="need not exist"):
        FIG_4A.optimal_adjustment_set("X", "Y")
    s = adjustment_scores(["V3"], FIG_4A, "X", "Y")  # validity alone
    assert s["valid"] is True and s["optimal"] is None and np.isnan(s["recall"])


def test_a_set_of_given_nodes_exists_when_the_constructive_one_is_valid():
    # van der Zander et al. (2019), Theorem 2: some adjustment set within R exists exactly when
    # the ancestors of the pair in R that are not forbidden are one; R = the observed nodes.
    rng = np.random.default_rng(8)
    pairs = found_any = 0
    for _ in range(30):
        d = int(rng.integers(4, 8))
        dag = _dag_of(_random_dag(rng, d, rng.uniform(0.3, 0.7)))
        for x, y in itertools.permutations(range(d), 2):
            among = [v for v in range(d) if v not in (x, y) and rng.random() < 0.6]
            exists = any(dag.is_adjustment_set(x, y, z) for z in _subsets(among))
            found = adj.adjustment_set(dag, x, y, among=among)
            assert (found is not None) == exists
            assert found is None or set(found) <= set(among)
            pairs += 1
            found_any += exists
    assert pairs > 500 and 0.2 < found_any / pairs < 0.95
