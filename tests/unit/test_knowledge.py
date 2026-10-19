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
from andrey.core import ARROW, TAIL
from andrey.core.knowledge import admits, resolve
from andrey.core.orient import dag2cpdag, from_structure, meek
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


@pytest.mark.parametrize("name", METHODS)
def test_required_forbidden_and_tiers_hold(name, data):
    out = _fit(name, data, knowledge=KNOWLEDGE)
    marks, labels = _marks(out)
    at = labels.index

    def directed(x, y):
        return marks[at(x), at(y)] == TAIL and marks[at(y), at(x)] == ARROW

    assert directed("b", "f"), out.structure.oriented_edges()
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


def test_meek_rule4_orients_what_knowledge_implies():
    # i -- j, i -- k, i -- m, k -> m -> j, k and j non-adjacent  =>  i -> j.
    i, j, k, m = range(4)
    pdag = np.zeros((4, 4), dtype=np.int8)
    for a, b in ((i, j), (i, k), (i, m)):
        pdag[a, b] = pdag[b, a] = TAIL
    for a, b in ((k, m), (m, j)):
        pdag[a, b], pdag[b, a] = TAIL, ARROW
    assert meek(pdag)[i, j] == TAIL and meek(pdag)[j, i] == TAIL
    out = meek(pdag, rule4=True)
    assert out[i, j] == TAIL and out[j, i] == ARROW


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
    for value in (str(spec), spec.read_text()):
        code = main(["run", "pc", "--data", str(path), f"knowledge={value}", "--json"])
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


def test_schema_lists_the_parts_each_method_honors():
    from andrey.spec.generate import to_json_schema

    for name in METHODS:
        schema = to_json_schema(REGISTRY[name])
        assert schema["knowledge"] == list(REGISTRY[name].knowledge)
        assert schema["params"]["properties"]["knowledge"]["type"] == ["object", "null"]
