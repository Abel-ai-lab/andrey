"""Background knowledge: the type, its resolution against the data, and what each method honors."""

from __future__ import annotations

import importlib
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey import Knowledge
from andrey.cli import main
from andrey.constraint.fci import _fci_orient, _remove_by_possible_dsep
from andrey.core import ARROW, NULL, TAIL
from andrey.core.knowledge import KnowledgeError, admits, resolve, resolve_lagged
from andrey.core.orient import dag2cpdag, from_structure, meek
from andrey.core.structure import CIRCLE
from andrey.search.gies import admits as gies_admits
from andrey.spec import REGISTRY

# The searches that take knowledge and return a graph over the columns of a table.
METHODS = [
    name
    for name, spec in REGISTRY.items()
    if spec.knowledge and spec.output.structure_type == "graph"
]
# The time-series searches that take knowledge, over the links between lagged variables.
TEMPORAL = [
    name
    for name, spec in REGISTRY.items()
    if spec.knowledge and spec.output.structure_type == "temporal"
]
# Each method on complete data, and each that accepts missing values on data with a tenth of its
# cells missing as well.
CASES = [
    *(pytest.param(name, False, id=name) for name in METHODS),
    *(
        pytest.param(name, True, id=f"{name}-missing")
        for name in METHODS
        if REGISTRY[name].data.missing
    ),
]
SERIES_CASES = [
    *(pytest.param(name, False, id=name) for name in TEMPORAL),
    *(
        pytest.param(name, True, id=f"{name}-missing")
        for name in TEMPORAL
        if REGISTRY[name].data.missing
    ),
]
PAGS = {name for name, spec in REGISTRY.items() if spec.output.graph_kind == "pag"}
DEPTH = ["pc", "fci", "gfci", "cdnod"]
LABELS = list("abcdef")
# a -> c <- b, c -> d -> e, a -> f.
TRUTH = {("a", "c"), ("b", "c"), ("c", "d"), ("d", "e"), ("a", "f")}
EARLY, LATE = ["a", "b"], ["c", "d", "e", "f"]
KNOWLEDGE = Knowledge(
    required=[("b", "f")],  # b and f are independent
    forbidden=[("c", "d"), ("e", "d")],  # c -> d is a true edge
    tiers=[EARLY, LATE],
)


def _sample(seed: int = 0, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-1.0, 1.0, (n, 6))
    a, b = noise[:, 0], noise[:, 1]
    c = 0.8 * a + 0.8 * b + noise[:, 2]
    d = 0.9 * c + noise[:, 3]
    e = 0.8 * d + noise[:, 4]
    f = 0.8 * a + noise[:, 5]
    return pd.DataFrame(np.column_stack([a, b, c, d, e, f]), columns=LABELS)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _sample()


def _gaps(table, missing: bool):
    """``table`` with a tenth of its cells missing completely at random when ``missing``."""
    if not missing:
        return table
    return table.mask(np.random.default_rng(3).random(table.shape) < 0.1)


def _fit(name: str, data: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "cdnod":
            return andrey.cdnod(data, np.repeat([0.0, 1.0], len(data) // 2), **kwargs)
        if name == "exact_search":
            return andrey.exact_search(data, **kwargs)
        if REGISTRY[name].output.multi:  # the first of two groups, the halves of the rows
            half = len(data) // 2
            return getattr(andrey, name)([data[:half], data[half:]], **kwargs)[0]
        return getattr(andrey, name)(data, **kwargs)


def _marks(out) -> tuple[np.ndarray, list[str]]:
    structure = out.structure
    return from_structure(structure), list(structure.labels)


def _assert_required_forbidden_and_tiers_hold(name: str, marks: np.ndarray, labels: list[str]):
    at = labels.index

    def directed(x, y):
        return marks[at(x), at(y)] == TAIL and marks[at(y), at(x)] == ARROW

    assert directed("b", "f") or "required" not in REGISTRY[name].knowledge, marks
    assert not directed("c", "d") and not directed("e", "d")
    for early in EARLY:
        for late in LATE:
            if marks[at(early), at(late)]:
                # No edge into an earlier tier: an arrowhead at the later end in a PAG, a directed
                # edge otherwise.
                assert marks[at(late), at(early)] == ARROW, (early, late)
                if name not in PAGS:
                    assert directed(early, late), (early, late)
    if name not in PAGS and marks[at("c"), at("d")]:
        assert directed("d", "c")


def _honored(name: str, knowledge: Knowledge) -> Knowledge:
    """``knowledge`` without the parts method ``name`` does not honor."""
    parts = REGISTRY[name].knowledge
    return Knowledge(
        **{part: getattr(knowledge, part) for part in knowledge.parts() if part in parts}
    )


@pytest.mark.parametrize(("name", "missing"), CASES)
def test_required_forbidden_and_tiers_hold(name, missing, data):
    marks, labels = _marks(_fit(name, _gaps(data, missing), knowledge=_honored(name, KNOWLEDGE)))
    _assert_required_forbidden_and_tiers_hold(name, marks, labels)


@pytest.mark.parametrize("in_column_order", [False, True])
@pytest.mark.parametrize("name", ["pc", "cdnod", "fci", "gfci"])
def test_knowledge_holds_whether_the_rules_run_in_rounds_or_one_at_a_time(
    name, in_column_order, data
):
    constraints = resolve(KNOWLEDGE, len(LABELS), LABELS, method=name)
    X = data.to_numpy()
    kwargs = dict(constraints=constraints, in_column_order=in_column_order)
    if name == "cdnod":
        domain = np.repeat([0.0, 1.0], len(X) // 2)
        structure = importlib.import_module("andrey.constraint.cdnod").cdnod(X, domain, **kwargs)
    else:
        module = "andrey.search.gfci" if name == "gfci" else f"andrey.constraint.{name}"
        structure = getattr(importlib.import_module(module), name)(X, **kwargs)
    _assert_required_forbidden_and_tiers_hold(name, from_structure(structure), LABELS)


@pytest.mark.parametrize("columns", [LABELS, LABELS[::-1], list("cfaebd")])
@pytest.mark.parametrize("setting", [{}, {"possible_dsep": "fast"}], ids=["default", "fast"])
@pytest.mark.parametrize("name", ["fci", "gfci"])
def test_knowledge_holds_through_the_possible_dsep_step_in_every_column_order(
    name, setting, columns, data
):
    # The default runs that step on the columns sorted by rank, so the knowledge moves with them.
    marks, labels = _marks(_fit(name, data[columns], knowledge=KNOWLEDGE, **setting))
    _assert_required_forbidden_and_tiers_hold(name, marks, labels)


@pytest.mark.parametrize("by_rank", [None, [3, 1, 0, 2, 4]], ids=["column order", "rank order"])
def test_the_possible_dsep_step_never_removes_a_required_pair(by_rank):
    # A triangle 0, 1, 2 and the edge 2 - 3: node 3 is in the possible-d-sep set of 0 and 1 and in
    # neither one's neighborhood, so a test that finds them independent given {3} removes the edge.
    marks = np.zeros((5, 5), dtype=np.int8)
    for i, j, at_i, at_j in [(0, 1, 3, 1), (0, 2, 2, 2), (1, 2, 1, 3), (2, 3, 2, 3)]:
        marks[i, j], marks[j, i] = at_i, at_j
    keep = np.zeros((5, 5), dtype=bool)
    keep[0, 1] = keep[1, 0] = True
    where = np.arange(5) if by_rank is None else np.argsort(by_rank)  # each node's test index
    pair = {int(where[0]), int(where[1])}
    order = None if by_rank is None else np.array(by_rank)

    def test(x, y, condition_set):
        return 1.0 if {x, y} == pair else 0.0

    unprotected, sepsets = marks.copy(), {}
    _remove_by_possible_dsep(unprotected, test, 0.05, sepsets, order)
    assert unprotected[0, 1] == NULL  # the test would remove it
    # The separating set holds column labels, not ranks.
    assert {frozenset(pair): cond for pair, cond in sepsets.items()} == {frozenset((0, 1)): (3,)}
    protected, sepsets = marks.copy(), {}
    _remove_by_possible_dsep(protected, test, 0.05, sepsets, order, keep=keep)
    np.testing.assert_array_equal(protected, marks)
    assert not sepsets


@pytest.mark.parametrize(("name", "missing"), CASES)
def test_allowed_pairs_hold(name, missing, data):
    allowed = {*TRUTH, ("b", "e")}
    out = _fit(name, _gaps(data, missing), knowledge=Knowledge(allowed=allowed))
    marks, labels = _marks(out)
    adjacent = {
        frozenset((labels[i], labels[j]))
        for i, j in zip(*np.nonzero(np.triu(marks != 0, 1)), strict=True)
    }
    assert adjacent <= {frozenset(pair) for pair in allowed}


@pytest.mark.parametrize(("name", "missing"), CASES)
def test_reordered_columns_relabel_the_result(name, missing, data):
    def edges(frame):
        found = _fit(name, frame, knowledge=_honored(name, KNOWLEDGE)).structure.oriented_edges()
        # An edge with the same mark at both ends lists its ends in column order.
        return sorted(
            (*sorted((a, b)), kind)
            if kind in ("undirected", "bidirected", "circle")
            else (a, b, kind)
            for a, b, kind in found
        )

    table = _gaps(data, missing)
    assert edges(table[LABELS[::-1]]) == edges(table)


@pytest.mark.parametrize("name", METHODS)
def test_empty_knowledge_is_no_knowledge(name, data):
    kept = _fit(name, data, knowledge=Knowledge())
    plain = _fit(name, data)
    assert kept.structure.oriented_edges() == plain.structure.oriented_edges()


@pytest.mark.parametrize("name", ["boss", "grasp"])
def test_bdeu_search_honors_required_forbidden_and_tiers(name):
    rng = np.random.default_rng(0)
    n = 3000
    a, b = rng.integers(0, 2, (2, n))
    c = (a + b + (rng.random(n) < 0.15)) % 3
    d = (c + (rng.random(n) < 0.15)) % 3
    e = rng.integers(0, 2, n)
    frame = pd.DataFrame({"a": a, "b": b, "c": c, "d": d, "e": e}).astype(float)
    knowledge = Knowledge(required=[("b", "e")], forbidden=[("c", "d")], tiers=[["a", "b"], ["c"]])
    out = getattr(andrey, name)(frame, score_func="local_score_BDeu", knowledge=knowledge)
    edges = set(out.structure.oriented_edges())
    assert ("b", "e", "directed") in edges, edges
    assert ("c", "d", "directed") not in edges, edges
    assert not {("c", "a", "directed"), ("c", "b", "directed")} & edges, edges
    free = getattr(andrey, name)(frame, score_func="local_score_BDeu")
    assert ("c", "d", "directed") in free.structure.oriented_edges()


@pytest.mark.parametrize(
    ("name", "setting"),
    [(name, {}) for name in DEPTH]
    + [(name, {"possible_dsep": "fast"}) for name in DEPTH if name in PAGS],
    ids=lambda value: value if isinstance(value, str) else ("fast" if value else "default"),
)
def test_max_depth_caps_the_conditioning_sets(name, setting, monkeypatch):
    rng = np.random.default_rng(1)
    d = 8
    weights = np.tril(rng.uniform(0.5, 1.0, (d, d)) * rng.choice([-1, 1], (d, d)), -1)
    weights *= rng.random((d, d)) < 0.6
    X = rng.standard_normal((1000, d)) @ np.linalg.inv(np.eye(d) - weights).T
    largest: list[int] = []

    class Recorder:
        def __init__(self, test):
            self._test = test

        def __call__(self, x, y, s):
            largest.append(len(s))
            return self._test(x, y, s)

        def batched_call(self, x, y, sets):
            largest.extend(len(s) for s in sets)
            return self._test.batched_call(x, y, sets)

        def marginal_pvalues(self):
            largest.append(0)
            return self._test.marginal_pvalues()

        def first_order_pvalues(self, x, y, candidates):
            largest.append(1)
            return self._test.first_order_pvalues(x, y, candidates)

        def __getattr__(self, attr):
            if attr == "to_seed":  # keep every test in this process
                raise AttributeError(attr)
            return getattr(self._test, attr)

    # The packages re-export functions under their modules' names, so import the modules by path.
    for path in ("andrey.constraint.pc", "andrey.constraint.fci", "andrey.search.gfci"):
        module = importlib.import_module(path)
        make = module.make_indep_test
        monkeypatch.setattr(
            module, "make_indep_test", lambda *a, _make=make, **k: Recorder(_make(*a, **k))
        )
    frame = pd.DataFrame(X)
    _fit(name, frame, **setting)
    assert max(largest) >= 2
    largest.clear()
    _fit(name, frame, max_depth=1, **setting)
    assert max(largest) <= 1


@pytest.mark.parametrize("in_column_order", [False, True])
def test_max_depth_caps_the_conditioning_sets_of_rule_r4(in_column_order):
    # 0 *-> 1 <-* 2 o-o 3 with 1 -> 3 and 0, 3 non-adjacent: R4 tests 0 and 3 given {1, 2} and {1}.
    marks = np.zeros((4, 4), dtype=np.int8)
    for u, v in [(0, 1), (1, 2), (2, 3), (1, 3)]:
        marks[u, v] = marks[v, u] = CIRCLE
    marks[1, 0] = marks[1, 2] = marks[3, 1] = ARROW
    marks[1, 3] = TAIL
    sizes: list[int] = []

    def test(x, y, condition_set):
        sizes.append(len(condition_set))
        return 0.0

    for max_depth, largest in [(None, 2), (1, 1), (0, 0)]:
        sizes.clear()
        _fci_orient(
            marks.copy(), test, 0.05, {}, max_depth=max_depth, in_column_order=in_column_order
        )
        assert max(sizes, default=0) == largest


def test_knowledge_checks_its_shape():
    with pytest.raises(TypeError, match="pairs"):
        Knowledge(required=["ab"])
    with pytest.raises(ValueError, match="self-loop"):
        Knowledge(forbidden=[("a", "a")])
    with pytest.raises(ValueError, match="self-loop"):
        Knowledge(forbidden=[("a", "a", 0)])
    with pytest.raises(TypeError, match="lag"):
        Knowledge(required=[("a", "b", -1)])
    with pytest.raises(TypeError, match="pairs"):
        Knowledge(allowed=[("a", "b", 1)])
    assert Knowledge(forbidden=[("a", "a", 1)]).forbidden == (("a", "a", 1),)
    with pytest.raises(ValueError, match="more than once"):
        Knowledge(tiers=[["a"], ["a", "b"]])
    with pytest.raises(TypeError, match="label or 0-based position"):
        Knowledge(required=[("a", 1.5)])


def test_resolve_reads_labels_and_positions():
    k = Knowledge(required=[("a", 2)], tiers=[["a"], ["b", "c"]])
    c = resolve(k, 3, ["a", "b", "c"], method="pc")
    assert c.required[0, 2] and c.forbidden[2, 0]
    assert c.forbidden[1, 0] and not c.forbidden[0, 1] and not c.forbidden[1, 2]
    assert resolve(Knowledge(), 3, None, method="pc") is None
    assert resolve(None, 3, None, method="pc") is None


@pytest.mark.parametrize(
    ("knowledge", "message"),
    [
        (Knowledge(required=[("z", "a")]), "which data does not have"),
        (Knowledge(required=[(0, 5)]), "data has 3 columns"),
        (Knowledge(required=[("c", "a")], tiers=[["a"], ["c"]]), "also forbids"),
        (Knowledge(required=[("a", "b")], allowed=[("a", "c")]), "also forbids"),
        (Knowledge(required=[("a", "b"), ("b", "c"), ("c", "a")]), "cycle"),
    ],
)
def test_resolve_rejects_contradictions(knowledge, message):
    with pytest.raises(ValueError, match=message):
        resolve(knowledge, 3, ["a", "b", "c"], method="pc")


def test_names_need_column_labels():
    with pytest.raises(ValueError, match="no column names"):
        resolve(Knowledge(required=[("a", "b")]), 3, None, method="pc")


def test_bottom_up_parce_lingam_rejects_required_edges(data):
    with pytest.raises(NotImplementedError, match="required"):
        andrey.bottom_up_parce_lingam(data, knowledge=Knowledge(required=[("a", "c")]))


def test_a_part_a_method_does_not_honor_raises():
    with pytest.raises(NotImplementedError, match="does not honor"):
        resolve(Knowledge(tiers=[[0], [1]]), 2, None, method="m", honors=("required",))


def test_order_searches_reject_tiers_and_required_edges_without_an_order():
    # c -> b -> a is allowed edge by edge, but puts a before c against the tiers.
    k = Knowledge(required=[("c", "b"), ("b", "a")], tiers=[["a"], ["c"]])
    with pytest.raises(ValueError, match="no causal order"):
        resolve(k, 3, ["a", "b", "c"], method="boss").precedence()


def test_from_skeleton_takes_a_result_or_a_matrix(data):
    out = andrey.pc(data)
    k = Knowledge.from_skeleton(out, tiers=[EARLY, LATE])
    assert {frozenset(p) for p in k.allowed} == {
        frozenset((x, y)) for x, y, _ in out.structure.oriented_edges()
    }
    assert Knowledge.from_skeleton(np.array([[0, 1], [0, 0]])).allowed == ((0, 1),)


@pytest.mark.parametrize("in_column_order", [False, True])
def test_meek_rule4_orients_what_knowledge_implies(in_column_order):
    # i -- j, i -- k, i -- m, k -> m -> j, k and j non-adjacent  =>  i -> j.
    i, j, k, m = range(4)
    pdag = np.zeros((4, 4), dtype=np.int8)
    for a, b in ((i, j), (i, k), (i, m)):
        pdag[a, b] = pdag[b, a] = TAIL
    for a, b in ((k, m), (m, j)):
        pdag[a, b], pdag[b, a] = TAIL, ARROW
    order = dict(in_column_order=in_column_order)
    assert meek(pdag, **order)[i, j] == TAIL and meek(pdag, **order)[j, i] == TAIL
    out = meek(pdag, rule4=True, **order)
    assert out[i, j] == TAIL and out[j, i] == ARROW
    out = meek(pdag, rule4=True, ambiguous={(k, i, j)}, **order)
    assert out[i, j] == TAIL and out[j, i] == TAIL


def _pdag(d: int, undirected=(), directed=()) -> np.ndarray:
    pdag = np.zeros((d, d), dtype=np.int8)
    for a, b in undirected:
        pdag[a, b] = pdag[b, a] = TAIL
    for a, b in directed:
        pdag[a, b], pdag[b, a] = TAIL, ARROW
    return pdag


@pytest.mark.parametrize(
    ("pdag", "edge", "triple"),
    [
        # R1: i -> j -- k  =>  j -> k, unless the triple i - j - k is ambiguous.
        (_pdag(3, undirected=[(1, 2)], directed=[(0, 1)]), (1, 2), (2, 1, 0)),
        # R3: i -- j, i -- k, j -> t <- k, i -- t  =>  i -> t, unless j - i - k is ambiguous.
        (
            _pdag(4, undirected=[(0, 1), (0, 2), (0, 3)], directed=[(1, 3), (2, 3)]),
            (0, 3),
            (1, 0, 2),
        ),
    ],
)
@pytest.mark.parametrize("in_column_order", [False, True])
def test_meek_does_not_orient_through_an_ambiguous_triple(pdag, edge, triple, in_column_order):
    a, b = edge
    out = meek(pdag, in_column_order=in_column_order)
    assert out[a, b] == TAIL and out[b, a] == ARROW
    out = meek(pdag, ambiguous={triple}, in_column_order=in_column_order)
    assert out[a, b] == TAIL and out[b, a] == TAIL


def test_meek_rounds_read_only_the_triples_that_are_not_ambiguous():
    # 3 -> 0 -- 2 -> 1 -- 3: R1 gives 0 -> 2 and 1 -> 3 (2 and 3 are non-adjacent), which close the
    # cycle 0 -> 2 -> 1 -> 3 -> 0 together, so a round leaves both edges undirected.
    pdag = _pdag(4, undirected=[(0, 2), (1, 3)], directed=[(2, 1), (3, 0)])
    out = meek(pdag)
    assert out[0, 2] == TAIL and out[2, 0] == TAIL and out[1, 3] == TAIL and out[3, 1] == TAIL
    # With the triple 2 - 1 - 3 ambiguous, 1 -> 3 is not implied, so 0 -> 2 closes no cycle.
    out = meek(pdag, ambiguous={(2, 1, 3)})
    assert out[0, 2] == TAIL and out[2, 0] == ARROW
    assert out[1, 3] == TAIL and out[3, 1] == TAIL


def test_admits_rejects_a_class_the_knowledge_contradicts():
    # The class a -> b <- c compels both edges; forbidding a -> b leaves no DAG in it.
    dag = np.zeros((3, 3), dtype=np.int8)
    for a, b in ((0, 1), (2, 1)):
        dag[a, b], dag[b, a] = TAIL, ARROW
    cpdag = dag2cpdag(dag)
    assert admits(cpdag, resolve(Knowledge(forbidden=[(2, 0)]), 3, None, method="ges"))
    assert not admits(cpdag, resolve(Knowledge(forbidden=[(0, 1)]), 3, None, method="ges"))


def test_admits_rejects_a_class_without_a_required_adjacency():
    constraints = resolve(Knowledge(required=[(0, 1)]), 2, None, method="ges")
    no_cut = np.zeros((2, 2), dtype=bool)
    line = _pdag(2, undirected=[(0, 1)])
    assert admits(line, constraints) and gies_admits(line, constraints, no_cut)
    empty = np.zeros((2, 2), dtype=np.int8)
    assert not admits(empty, constraints)
    assert not gies_admits(empty, constraints, no_cut)


def test_cli_reads_knowledge_from_a_file_and_inline(tmp_path, capsys, data):
    path = tmp_path / "data.csv"
    data.to_csv(path, index=False)
    spec = tmp_path / "knowledge.json"
    spec.write_text(json.dumps({"required": [["b", "f"]], "tiers": [EARLY, LATE]}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"knowledge": str(spec)}))
    for args in (
        [f"knowledge={spec}"],
        [f"knowledge='{spec}'"],
        [f"knowledge={spec.read_text()}"],
        ["--config", str(config)],
    ):
        code = main(["run", "pc", "--data", str(path), *args, "--json"])
        out, _ = capsys.readouterr()
        assert code == 0
        env = json.loads(out)
        names = env["structure"]["labels"]
        edges = {
            (names[e["source"]], names[e["target"]], e["type"]) for e in env["structure"]["edges"]
        }
        assert ("b", "f", "directed") in edges
        assert env["run"]["params"]["knowledge"]["required"] == [["b", "f"]]
    code = main(["run", "pc", "--data", str(path), 'knowledge={"must": []}', "--json"])
    _, err = capsys.readouterr()
    assert code == 2 and "keys" in err
    code = main(["run", "pc", "--data", str(path), 'knowledge={"tiers": [["g"]]}', "--json"])
    _, err = capsys.readouterr()
    assert code == 2 and "'g'" in err


def test_schema_lists_the_parts_each_method_honors():
    from andrey.spec.generate import to_json_schema

    for name in (*METHODS, *TEMPORAL):
        schema = to_json_schema(REGISTRY[name])
        assert schema["knowledge"] == list(REGISTRY[name].knowledge)
        assert schema["params"]["properties"]["knowledge"]["type"] == ["object", "null"]


def _collider(seed: int = 0, n: int = 1000) -> pd.DataFrame:
    # x -> y <- z; w is independent of the three.
    rng = np.random.default_rng(seed)
    x, z, w, noise = rng.uniform(-1.0, 1.0, (4, n))
    return pd.DataFrame({"x": x, "y": 0.8 * x + 0.8 * z + noise, "z": z, "w": w})


@pytest.mark.parametrize("name", ["pc", "cdnod", "fci"])
def test_a_pair_kept_apart_keeps_its_collider(name):
    knowledge = Knowledge(required=[("x", "y")], forbidden=[("x", "z"), ("z", "x")])
    marks, labels = _marks(_fit(name, _collider(), knowledge=knowledge))
    x, y, z = (labels.index(v) for v in "xyz")
    assert marks[y, x] == ARROW and marks[y, z] == ARROW
    if name not in PAGS:
        assert marks[x, y] == TAIL and marks[z, y] == TAIL


def _joined(seed: int = 0, n: int = 1000) -> pd.DataFrame:
    # x -> z, x -> y <- z: no set separates x and z.
    rng = np.random.default_rng(seed)
    x, y, z = rng.uniform(-1.0, 1.0, (3, n))
    z = 0.8 * x + z
    return pd.DataFrame({"x": x, "y": 0.8 * x + 0.8 * z + y, "z": z})


@pytest.mark.parametrize("name", ["pc", "cdnod"])
def test_no_rule_orients_through_a_kept_apart_pair_the_data_join(name):
    # x -> y and x, z kept apart: R1 would read x -> y -- z as a non-collider and direct y -> z.
    knowledge = Knowledge(required=[("x", "y")], forbidden=[("x", "z"), ("z", "x")])
    marks, labels = _marks(_fit(name, _joined(), knowledge=knowledge))
    x, y, z = (labels.index(v) for v in "xyz")
    assert marks[x, y] == TAIL and marks[y, x] == ARROW
    assert marks[y, z] == TAIL and marks[z, y] == TAIL


@pytest.mark.parametrize("name", [m for m in METHODS if "required" in REGISTRY[m].knowledge])
@pytest.mark.parametrize("edge", [("a", "b"), ("b", "a")])
def test_a_required_edge_keeps_its_direction_where_the_data_allow_both(name, edge):
    # Two dependent variables: either direction fits the data equally well.
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, 1000)
    data = pd.DataFrame({"a": a, "b": 0.8 * a + rng.uniform(-1.0, 1.0, 1000)})
    marks, labels = _marks(_fit(name, data, knowledge=Knowledge(required=[edge])))
    cause, effect = (labels.index(v) for v in edge)
    assert marks[cause, effect] == TAIL and marks[effect, cause] == ARROW


@pytest.mark.parametrize(
    "name", [m for m in METHODS if REGISTRY[m].family in ("lingam", "latent") and m != "gin"]
)
def test_lingam_knowledge_that_fits_the_truth_does_not_add_errors(name, data):
    def errors(out) -> int:
        found = {(a, b) for a, b, kind in out.structure.oriented_edges() if kind == "directed"}
        return len(found ^ TRUTH)

    knowledge = Knowledge(required=[("c", "d")], forbidden=[("e", "d")], tiers=[EARLY, LATE])
    knowledge = _honored(name, knowledge)
    assert errors(_fit(name, data, knowledge=knowledge)) <= errors(_fit(name, data))


def _confounded(seed: int = 0, n: int = 2000) -> np.ndarray:
    # A hidden h drives x0 and x1; x2 and x3 are their children.
    rng = np.random.default_rng(seed)
    h, e0, e1, e2, e3 = rng.uniform(-1.0, 1.0, (5, n))
    x0, x1 = 1.5 * h + e0, 1.5 * h + e1
    return np.column_stack([x0, x1, x1 + e2, x0 + e3])


def test_rcd_never_marks_a_pair_kept_apart_as_confounded():
    data = _confounded()
    assert andrey.rcd(data).metadata["confounded_pairs"] == [[0, 1]]
    out = andrey.rcd(data, knowledge=Knowledge(forbidden=[(0, 1), (1, 0)]))
    assert out.metadata["confounded_pairs"] == [] and from_structure(out.structure)[0, 1] == NULL


def test_rcd_raises_on_required_edges_that_close_a_cycle():
    data = _confounded()  # RCD finds 1 -> 2 and 0 -> 3
    with pytest.raises(KnowledgeError, match="cycle"):
        andrey.rcd(data, knowledge=Knowledge(required=[(2, 0), (3, 1), (0, 1)]))


@pytest.mark.parametrize("forbidden", [[(0, 1)], [(0, 1), (1, 0)]], ids=["one way", "both ways"])
def test_bottom_up_parce_lingam_reads_the_knowledge_in_the_unordered_block(forbidden):
    data = _confounded()
    plain = andrey.bottom_up_parce_lingam(data, alpha=0.01)
    assert plain.metadata["causal_order"] == [[0, 1], 2, 3]
    out = andrey.bottom_up_parce_lingam(data, alpha=0.01, knowledge=Knowledge(forbidden=forbidden))
    marks = from_structure(out.structure)
    if len(forbidden) == 2:
        assert marks[0, 1] == NULL and marks[1, 0] == NULL
    else:  # an arrowhead at the cause of the forbidden edge, a circle at its effect
        assert marks[0, 1] == ARROW and marks[1, 0] == CIRCLE


def test_every_group_honors_the_knowledge(data):
    half = len(data) // 2
    outs = andrey.multi_group_direct_lingam([data[:half], data[half:]], knowledge=KNOWLEDGE)
    for out in outs:
        _assert_required_forbidden_and_tiers_hold("multi_group_direct_lingam", *_marks(out))


def test_direct_lingam_keeps_a_required_edge_whose_weight_is_zero():
    # The two columns are uncorrelated, so least squares weighs the required 0 -> 1 at 0, up to the
    # rounding of the platform's BLAS.
    X = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    out = andrey.direct_lingam(X, knowledge=Knowledge(required=[(0, 1)]))
    assert out.weighted_adjacency[0, 1] == pytest.approx(0.0, abs=1e-12)
    marks = from_structure(out.structure)
    assert marks[0, 1] == TAIL and marks[1, 0] == ARROW


def test_a_skeleton_as_knowledge_keeps_the_colliders():
    data = _collider()
    plain = andrey.pc(data).structure.oriented_edges()
    assert ("z", "y", "directed") in plain
    screened = andrey.pc(data, knowledge=Knowledge.from_skeleton(andrey.pc(data)))
    assert screened.structure.oriented_edges() == plain


def _triangle() -> pd.DataFrame:
    # a -> b, a -> c, b -> c: every pair adjacent.
    rng = np.random.default_rng(1)
    a, b, c = rng.uniform(-1.0, 1.0, (3, 1000))
    b = 0.7 * a + b
    return pd.DataFrame({"a": a, "b": b, "c": a + b + c})


# The forbidden edges direct a -> b -> c -> a.
CYCLE = Knowledge(forbidden=[("b", "a"), ("c", "b"), ("a", "c")])
# c -> b -> a makes c an ancestor of a, and the tiers forbid c -> a.
ANCESTOR = Knowledge(required=[("c", "b"), ("b", "a")], tiers=[["a"], ["c"]])


@pytest.mark.parametrize(
    ("name", "knowledge"),
    [("pc", CYCLE), ("cdnod", CYCLE), ("pc", ANCESTOR), ("cdnod", ANCESTOR), ("fci", ANCESTOR)],
)
def test_knowledge_that_closes_a_cycle_raises(name, knowledge):
    with pytest.raises(KnowledgeError):
        _fit(name, _triangle(), knowledge=knowledge)


def test_fci_reads_a_cycle_of_arrowheads_as_a_pag():
    # An arrowhead at each end of a cycle fits three bidirected edges.
    marks, labels = _marks(_fit("fci", _triangle(), knowledge=CYCLE))
    a, b, c = (labels.index(v) for v in "abc")
    assert marks[b, a] == ARROW and marks[c, b] == ARROW and marks[a, c] == ARROW


# ---- time series --------------------------------------------------------------------------------

SERIES = ["x", "y", "z"]
# x -> x and y -> y at lag 1, x -> y at lags 1 and 2, y -> z at lag 0.
SERIES_TRUTH = {("x", "x", 1), ("y", "y", 1), ("x", "y", 1), ("x", "y", 2), ("y", "z", 0)}
SERIES_EFFECTS = {
    ("x", "x", 1): 0.5,
    ("y", "y", 1): 0.3,
    ("x", "y", 1): 0.5,
    ("x", "y", 2): 0.4,
    ("y", "z", 0): 0.7,
}


def _simulate(effects: dict, columns: list[str], *, panel: bool, seed: int = 0):
    """A linear series with the ``{(cause, effect, lag): weight}`` effects and uniform noise.

    One series of 1,000 steps, or with ``panel`` a list of 6 time points of 1,000 units each.
    The columns are listed in an order that puts the cause of every same-step effect first.
    """
    rng = np.random.default_rng(seed)
    steps, units = (6, 1000) if panel else (1000, 1)
    X = np.zeros((steps, units, len(columns)))
    for t in range(steps):
        X[t] = rng.uniform(-1.0, 1.0, (units, len(columns)))
        for j, effect in enumerate(columns):
            for (cause, target, lag), weight in effects.items():
                if target == effect and t >= lag:
                    X[t, :, j] += weight * X[t - lag, :, columns.index(cause)]
    if panel:
        return [pd.DataFrame(X[t], columns=columns) for t in range(steps)]
    return pd.DataFrame(X[:, 0], columns=columns)


def _panel_method(name: str) -> bool:
    return REGISTRY[name].data.inputs[0].kind == "panel"


def _input(name: str, effects: dict = SERIES_EFFECTS, columns: list[str] = SERIES):
    return _simulate(effects, columns, panel=_panel_method(name))


def _fit_series(name: str, data, lags: int = 2, **kwargs):
    """Fit ``name`` over lags ``0`` to ``lags``; VARMA-LiNGAM as a pruned VAR."""
    if name == "varma_lingam":
        return andrey.varma_lingam(data, order=(lags, 0), prune=True, **kwargs)
    if name == "longitudinal_lingam":
        return andrey.longitudinal_lingam(data, n_lags=lags, **kwargs)
    return getattr(andrey, name)(data, tau_max=lags, **kwargs)


def _reorder(data, columns: list[str]):
    return [frame[columns] for frame in data] if isinstance(data, list) else data[columns]


@pytest.fixture(scope="module")
def series() -> pd.DataFrame:
    return _simulate(SERIES_EFFECTS, SERIES, panel=False)


def _links(out) -> dict[tuple[str, str, int], str]:
    """Every link ``(a, b, lag)``, ``a`` at ``t - lag``, as its marks: ``"-->"``, ``"o->"``, ...

    A same-step link appears in both orders, ``"-->"`` and ``"<--"``; an undirected one is
    ``"---"``. A PAG lists every link with all its marks in ``metadata["links"]``.
    """
    names = list(out.structure.labels)
    if "links" in out.metadata:
        return {
            (names[i], names[j], lag): link
            for lag, rows in enumerate(out.metadata["links"])
            for i, row in enumerate(rows)
            for j, link in enumerate(row)
            if link
        }
    found = {}
    for lag in out.structure.lags:
        for a, b, kind in out.structure.lag(lag).oriented_edges():
            found[(a, b, lag)] = "-->" if kind == "directed" else "---"
            if lag == 0:
                found[(b, a, 0)] = "<--" if kind == "directed" else "---"
    return found


def _cause(links: dict, a: str, b: str, lag: int) -> bool:
    return links.get((a, b, lag)) == "-->"


def _may_cause(links: dict, a: str, b: str, lag: int) -> bool:
    """Whether the link leaves ``a`` open to be a cause of ``b``: a tail or a circle at ``a``."""
    return links.get((a, b, lag), "<")[0] in "-o"


@pytest.mark.parametrize("name", TEMPORAL)
def test_a_forbidden_lagged_link_is_absent_at_its_lag_only(name):
    data = _input(name)
    plain = _links(_fit_series(name, data))
    assert ("x", "y", 1) in plain and ("x", "y", 2) in plain
    links = _links(_fit_series(name, data, knowledge=Knowledge(forbidden=[("x", "y", 1)])))
    assert ("x", "y", 1) not in links
    assert ("x", "y", 2) in links


@pytest.mark.parametrize("name", TEMPORAL)
def test_a_forbidden_pair_is_no_cause_at_any_lag(name):
    links = _links(_fit_series(name, _input(name), knowledge=Knowledge(forbidden=[("y", "z")])))
    assert not any(_may_cause(links, "y", "z", lag) for lag in (0, 1, 2))


@pytest.mark.parametrize(("name", "missing"), SERIES_CASES)
def test_a_required_lagged_link_is_a_cause(name, missing):
    # z does not drive x; the required link holds anyway.
    knowledge = Knowledge(required=[("z", "x", 2)])
    data = _gaps(_input(name), missing)
    assert _cause(_links(_fit_series(name, data, knowledge=knowledge)), "z", "x", 2)


@pytest.mark.parametrize("name", TEMPORAL)
@pytest.mark.parametrize("edge", [("a", "b"), ("b", "a")])
def test_a_required_same_step_link_keeps_its_direction(name, edge):
    # Two variables that depend on each other within a step: the data favor a -> b.
    data = _input(name, {("a", "b", 0): 0.8}, ["a", "b"])
    knowledge = Knowledge(required=[(*edge, 0)])
    if 0 not in _fit_series(name, data, lags=1).structure.lags:
        with pytest.raises(KnowledgeError, match="lags 1 to"):
            _fit_series(name, data, lags=1, knowledge=knowledge)
        return
    assert _cause(_links(_fit_series(name, data, lags=1, knowledge=knowledge)), *edge, 0)


@pytest.mark.parametrize(("name", "missing"), SERIES_CASES)
def test_tiers_hold_at_every_lag(name, missing):
    # z first: no link runs into z from x or y, at any lag; the true y -> z at lag 0 included.
    data = _gaps(_input(name), missing)
    out = _fit_series(name, data, knowledge=Knowledge(tiers=[["z"], ["x", "y"]]))
    links = _links(out)
    for later in ("x", "y"):
        assert not any(_may_cause(links, later, "z", lag) for lag in (0, 1, 2)), links


@pytest.mark.parametrize("name", TEMPORAL)
def test_knowledge_that_fits_the_truth_does_not_add_errors(name):
    def errors(out) -> int:
        lags = out.structure.lags
        truth = {(a, b, lag, "-->") for a, b, lag in SERIES_TRUTH if lag in lags}
        truth |= {(b, a, 0, "<--") for a, b, lag in SERIES_TRUTH if lag == 0 and 0 in lags}
        found = {(a, b, lag, kind) for (a, b, lag), kind in _links(out).items()}
        return len(truth ^ found)

    knowledge = Knowledge(
        tiers=[["x"], ["y"], ["z"]], required=[("x", "y", 2)], forbidden=[("z", "z", 1)]
    )
    data = _input(name)
    plain = errors(_fit_series(name, data))
    assert errors(_fit_series(name, data, knowledge=knowledge)) <= plain


@pytest.mark.parametrize(("name", "missing"), SERIES_CASES)
def test_reordered_series_relabel_the_result_with_knowledge(name, missing):
    knowledge = Knowledge(forbidden=[("x", "y", 1)], required=[("z", "x", 2)], tiers=[["x"]])
    data = _gaps(_input(name), missing)

    def links(frame):
        return _links(_fit_series(name, frame, knowledge=knowledge))

    assert links(_reorder(data, ["z", "x", "y"])) == links(data)


@pytest.mark.parametrize("name", TEMPORAL)
def test_empty_knowledge_is_no_knowledge_for_a_series(name):
    data = _input(name)
    kept = _fit_series(name, data, knowledge=Knowledge())
    assert _links(kept) == _links(_fit_series(name, data))


def test_varma_lingam_without_pruning_rejects_a_forbidden_lagged_link(series):
    with pytest.raises(KnowledgeError, match="prune=True"):
        andrey.varma_lingam(series, order=(1, 0), knowledge=Knowledge(forbidden=[("x", "y", 1)]))
    # A required lagged link holds: every lagged weight stays.
    out = andrey.varma_lingam(series, order=(1, 0), knowledge=Knowledge(required=[("z", "x", 1)]))
    assert _cause(_links(out), "z", "x", 1)


@pytest.mark.parametrize(
    ("knowledge", "message"),
    [
        (Knowledge(required=[("x", "y")]), "without a lag"),
        (Knowledge(required=[("x", "y", 3)]), "lags"),
        (Knowledge(required=[("x", "y", 1)], forbidden=[("x", "y")]), "also forbids"),
        (Knowledge(required=[("x", "y", 0), ("y", "x", 0)]), "cycle"),
    ],
)
def test_resolve_lagged_rejects_what_no_series_graph_can_hold(knowledge, message):
    with pytest.raises(KnowledgeError, match=message):
        resolve_lagged(knowledge, 3, SERIES, 2, method="pcmciplus")


def test_resolve_lagged_holds_tiers_and_allowed_at_every_lag_but_not_the_own_past():
    c = resolve_lagged(
        Knowledge(tiers=[["x"], ["y"]], allowed=[("x", "y")]), 3, SERIES, 2, method="m"
    )
    for lag in range(3):
        assert c.forbidden[lag, 1, 0] and not c.forbidden[lag, 0, 1]
        assert c.forbidden[lag, 0, 2] and c.forbidden[lag, 2, 1]
    assert not c.forbidden[1:, [0, 1, 2], [0, 1, 2]].any()


def test_a_method_without_lags_rejects_a_lagged_link():
    with pytest.raises(NotImplementedError, match="learns no time lags"):
        andrey.pc(_sample(), knowledge=Knowledge(forbidden=[("a", "b", 1)]))


@pytest.mark.parametrize("collider_rule", ["majority", "none"])
def test_pcmciplus_orients_a_collider_over_a_pair_kept_apart_only_from_a_test(collider_rule):
    # x -> y <- z within one step, with x and z kept apart. The majority rule tests them for the
    # sets that separate them and finds the collider; "none" reads the separating set of the
    # skeleton phase, which never tested them, so the triple stays ambiguous.
    rng = np.random.default_rng(0)
    x, z, noise = rng.standard_normal((3, 1000))
    data = pd.DataFrame({"x": x, "y": 0.8 * x + 0.8 * z + noise, "z": z})
    knowledge = Knowledge(forbidden=[("x", "z"), ("z", "x")])
    out = andrey.pcmciplus(data, collider_rule=collider_rule, knowledge=knowledge)
    kinds = {kind for _, _, kind in out.structure.lag(0).oriented_edges()}
    if collider_rule == "majority":
        assert set(out.structure.lag(0).oriented_edges()) == {
            ("x", "y", "directed"),
            ("z", "y", "directed"),
        }
    else:
        assert [0, 0, 1, 2] in out.metadata["ambiguous_triples"] and kinds == {"undirected"}


def test_cli_reads_lagged_links(tmp_path, capsys, series):
    path = tmp_path / "series.csv"
    series.to_csv(path, index=False)
    knowledge = 'knowledge={"required": [["z", "x", 2]], "forbidden": [["x", "y", 1]]}'
    code = main(["run", "pcmci", "--data", str(path), "tau_max=2", knowledge, "--json"])
    out, _ = capsys.readouterr()
    assert code == 0
    env = json.loads(out)
    names = env["structure"]["labels"]
    links = {
        (names[e["source"]], names[e["target"]], lag["lag"])
        for lag in env["structure"]["lags"]
        for e in lag["edges"]
    }
    assert ("z", "x", 2) in links and ("x", "y", 1) not in links
    assert env["run"]["params"]["knowledge"]["required"] == [["z", "x", 2]]
    code = main(["run", "pc", "--data", str(path), knowledge, "--json"])
    _, err = capsys.readouterr()
    assert code == 2 and "time lags" in err
