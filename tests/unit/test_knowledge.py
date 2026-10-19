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
from andrey.constraint.fci import _fci_orient
from andrey.core import ARROW, TAIL
from andrey.core.knowledge import admits, resolve
from andrey.core.orient import dag2cpdag, from_structure, meek
from andrey.core.structure import CIRCLE
from andrey.spec import REGISTRY

METHODS = [name for name, spec in REGISTRY.items() if spec.knowledge]
PAGS = {"fci", "gfci"}
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


def _fit(name: str, data: pd.DataFrame, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "cdnod":
            return andrey.cdnod(data, np.repeat([0.0, 1.0], len(data) // 2), **kwargs)
        if name == "exact_search":
            return andrey.exact_search(data, **kwargs)
        return getattr(andrey, name)(data, **kwargs)


def _marks(out) -> tuple[np.ndarray, list[str]]:
    structure = out.structure
    return from_structure(structure), list(structure.labels)


def _assert_required_forbidden_and_tiers_hold(name: str, marks: np.ndarray, labels: list[str]):
    at = labels.index

    def directed(x, y):
        return marks[at(x), at(y)] == TAIL and marks[at(y), at(x)] == ARROW

    assert directed("b", "f"), marks
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


@pytest.mark.parametrize("name", METHODS)
def test_required_forbidden_and_tiers_hold(name, data):
    marks, labels = _marks(_fit(name, data, knowledge=KNOWLEDGE))
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


@pytest.mark.parametrize("name", METHODS)
def test_allowed_pairs_hold(name, data):
    allowed = {*TRUTH, ("b", "e")}
    out = _fit(name, data, knowledge=Knowledge(allowed=allowed))
    marks, labels = _marks(out)
    adjacent = {
        frozenset((labels[i], labels[j]))
        for i, j in zip(*np.nonzero(np.triu(marks != 0, 1)), strict=True)
    }
    assert adjacent <= {frozenset(pair) for pair in allowed}


@pytest.mark.parametrize("name", METHODS)
def test_reordered_columns_relabel_the_result(name, data):
    def edges(frame):
        return sorted(_fit(name, frame, knowledge=KNOWLEDGE).structure.oriented_edges())

    assert edges(data[LABELS[::-1]]) == edges(data)


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


@pytest.mark.parametrize("name", DEPTH)
def test_max_depth_caps_the_conditioning_sets(name, monkeypatch):
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
    _fit(name, frame)
    assert max(largest) >= 2
    largest.clear()
    _fit(name, frame, max_depth=1)
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

    for name in METHODS:
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


@pytest.mark.parametrize("name", METHODS)
@pytest.mark.parametrize("edge", [("a", "b"), ("b", "a")])
def test_a_required_edge_keeps_its_direction_where_the_data_allow_both(name, edge):
    # Two dependent variables: either direction fits the data equally well.
    rng = np.random.default_rng(0)
    a = rng.uniform(-1.0, 1.0, 1000)
    data = pd.DataFrame({"a": a, "b": 0.8 * a + rng.uniform(-1.0, 1.0, 1000)})
    marks, labels = _marks(_fit(name, data, knowledge=Knowledge(required=[edge])))
    cause, effect = (labels.index(v) for v in edge)
    assert marks[cause, effect] == TAIL and marks[effect, cause] == ARROW


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
    with pytest.raises(ValueError, match="into a cycle"):
        _fit(name, _triangle(), knowledge=knowledge)


def test_fci_reads_a_cycle_of_arrowheads_as_a_pag():
    # An arrowhead at each end of a cycle fits three bidirected edges.
    marks, labels = _marks(_fit("fci", _triangle(), knowledge=CYCLE))
    a, b, c = (labels.index(v) for v in "abc")
    assert marks[b, a] == ARROW and marks[c, b] == ARROW and marks[a, c] == ARROW
