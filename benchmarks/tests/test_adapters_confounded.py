"""The hidden-confounder adapters and the scores of the pairs they attribute to a hidden cause.

Each lingam adapter must give the marks that Andrey's conversion gives its own result, and each
adapter must forward the settings that make both packages run the same method. The pair scores are
checked on small graphs whose confounded pairs are known.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import confounded_cpu
import numpy as np
import pytest

import andrey.metrics as M
from andrey import GraphStructure
from andrey.core import LATENT, OBSERVED
from andrey.data.latent import marginal
from andrey_bench import datasets
from andrey_bench.adapters.andrey_bottom_up_parce import AndreyBottomUpParceLiNGAM
from andrey_bench.adapters.andrey_rcd import AndreyRCD
from andrey_bench.adapters.lingam_bottom_up_parce import LingamBottomUpParceLiNGAM
from andrey_bench.adapters.lingam_rcd import LingamRCD
from andrey_bench.scoring import (
    CONFOUNDED_COLUMNS,
    CONFOUNDED_TARGETS,
    confounded_pair_scores,
    directed_edge_scores,
    projected_directed,
    true_confounded_pairs,
)

# Andrey endpoint marks.
_TAIL, _ARROW, _CIRCLE = 1, 2, 3
NAN = np.nan

#: Pairs of one method as the two packages name them: they must agree on everything the scorer reads.
PAIRS = [
    (AndreyRCD(), LingamRCD()),
    (AndreyBottomUpParceLiNGAM(), LingamBottomUpParceLiNGAM()),
]


@pytest.mark.parametrize("andrey_adapter, lingam_adapter", PAIRS, ids=lambda a: a.name)
def test_both_packages_report_one_method_the_same_way(andrey_adapter, lingam_adapter) -> None:
    for field in ("algorithm", "output_type", "confounded_mark", "mode"):
        assert getattr(andrey_adapter, field) == getattr(lingam_adapter, field), field


def _fake_lingam(monkeypatch, attribute: str, adjacency: np.ndarray, calls: list):
    """Replace ``lingam.<attribute>`` with a class that records its settings and returns `adjacency`.

    Where lingam is not installed, as in a bench environment whose lingam fits run in an
    environment of their own, a stand-in module takes its place.
    """
    try:
        import lingam
    except ImportError:
        lingam = ModuleType("lingam")
        monkeypatch.setitem(sys.modules, "lingam", lingam)

    class Fake:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def fit(self, X):
            return SimpleNamespace(adjacency_matrix_=adjacency)

    monkeypatch.setattr(lingam, attribute, Fake, raising=False)


def test_lingam_rcd_marks_a_nan_pair_bidirected_and_a_weight_directed(monkeypatch) -> None:
    # B[0, 2] != 0 is the edge 2 -> 0; NaN in both cells of (0, 1) is a hidden common cause.
    B = np.array([[0, NAN, 0.7], [NAN, 0, 0], [0, 0, 0]])
    calls: list = []
    _fake_lingam(monkeypatch, "RCD", B, calls)
    adapter = LingamRCD()
    marks = adapter.fit(np.zeros((10, 3)), adapter.params())
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[0, 1] = expected[1, 0] = _ARROW
    expected[2, 0], expected[0, 2] = _TAIL, _ARROW
    np.testing.assert_array_equal(marks, expected)
    assert calls == [
        {"max_explanatory_num": 2, "cor_alpha": 0.01, "ind_alpha": 0.01, "shapiro_alpha": 0.01}
    ]


def test_lingam_bottom_up_marks_a_nan_pair_as_circles(monkeypatch) -> None:
    B = np.array([[0, 0, 0], [0.5, 0, NAN], [0, NAN, 0]])
    calls: list = []
    _fake_lingam(monkeypatch, "BottomUpParceLiNGAM", B, calls)
    adapter = LingamBottomUpParceLiNGAM()
    marks = adapter.fit(np.zeros((10, 3)), adapter.params())
    expected = np.zeros((3, 3), dtype=np.int8)
    expected[1, 2] = expected[2, 1] = _CIRCLE
    expected[0, 1], expected[1, 0] = _TAIL, _ARROW
    np.testing.assert_array_equal(marks, expected)
    assert calls == [{"alpha": 0.1}]


@pytest.mark.parametrize(
    "adapter, function, settings",
    [
        (
            AndreyRCD(),
            "rcd",
            {
                "max_depth": 2,
                "cor_alpha": 0.01,
                "alpha": 0.01,
                "shapiro_alpha": 0.01,
            },
        ),
        (AndreyBottomUpParceLiNGAM(), "bottom_up_parce_lingam", {"alpha": 0.1}),
    ],
    ids=lambda a: getattr(a, "name", None),
)
def test_andrey_adapters_forward_their_settings_and_return_the_marks(
    adapter, function, settings, monkeypatch
) -> None:
    import andrey

    calls: list = []
    marks = np.array([[0, _CIRCLE], [_CIRCLE, 0]], dtype=np.int8)

    def fit(X, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(structure=GraphStructure.from_numpy(marks, kind="pag"))

    monkeypatch.setattr(andrey, function, fit)
    native = adapter.fit(np.zeros((10, 2)), adapter.params())
    np.testing.assert_array_equal(native, marks)
    assert calls == [settings]
    assert adapter.to_structure(native).kind == "pag"


# ==================================================================================================
# Pair scores.
# ==================================================================================================


def _dag(edges: list[tuple[int, int]], d: int, hidden: tuple[int, ...]) -> GraphStructure:
    """Return a DAG with the given ``i -> j`` edges whose `hidden` nodes are flagged hidden."""
    marks = np.zeros((d, d), dtype=np.uint8)
    for i, j in edges:
        marks[i, j], marks[j, i] = _TAIL, _ARROW
    types = np.full(d, OBSERVED, dtype=np.int8)
    types[list(hidden)] = LATENT
    return GraphStructure.from_numpy(marks, kind="dag", node_types=types)


def _pag(pairs: list[tuple[int, int]], d: int, mark: int) -> GraphStructure:
    """Return a PAG whose only edges join `pairs` with `mark` at both ends."""
    marks = np.zeros((d, d), dtype=np.int8)
    for i, j in pairs:
        marks[i, j] = marks[j, i] = mark
    return GraphStructure.from_numpy(marks, kind="pag")


def test_the_two_targets_differ_where_a_pair_has_no_identifiable_arrowheads() -> None:
    # Hidden 2 causes 0 and 1: the observed PAG joins them with o-o, since the data cannot say
    # which of 0 -> 1, 0 <- 1, or 0 <-> 1 holds.
    truth = _dag([(2, 0), (2, 1)], d=3, hidden=(2,))
    assert true_confounded_pairs(truth, "pag") == set()
    assert true_confounded_pairs(truth, "hidden_cause") == {(0, 1)}


def test_the_targets_agree_where_a_collider_fixes_the_arrowheads() -> None:
    # Hidden 4 causes 0 and 1, with 2 -> 0 and 3 -> 1: the observed PAG is 2 o-> 0 <-> 1 <-o 3.
    truth = _dag([(4, 0), (4, 1), (2, 0), (3, 1)], d=5, hidden=(4,))
    assert true_confounded_pairs(truth, "pag") == {(0, 1)}
    assert true_confounded_pairs(truth, "hidden_cause") == {(0, 1)}


def test_a_hidden_cause_reaches_through_hidden_nodes_only() -> None:
    # Hidden 3 -> hidden 4 -> 1, and 3 -> 0: 0 and 1 share the hidden cause 3. Observed 2 only
    # reaches 1 through the observed node 0 -> 2, so it shares no hidden cause with 1.
    truth = _dag([(3, 4), (4, 1), (3, 0), (0, 2)], d=5, hidden=(3, 4))
    assert true_confounded_pairs(truth, "hidden_cause") == {(0, 1)}


def test_an_edge_between_confounded_nodes_leaves_the_projection_target() -> None:
    # Hidden 3 causes 0, 1, and 2, and 0 -> 1: RCD marks (0, 1) as confounded and leaves it unoriented.
    truth = _dag([(3, 0), (3, 1), (3, 2), (0, 1)], d=4, hidden=(3,))
    assert true_confounded_pairs(truth, "hidden_cause") == {(0, 1), (0, 2), (1, 2)}
    assert projected_directed(truth) == {(0, 1)}
    assert directed_edge_scores(_pag([], 3, _ARROW), truth)["directed_projection_n_true"] == 0


def test_directed_edges_are_scored_against_the_latent_projection() -> None:
    # 0 -> hidden 3 -> 1 projects to 0 -> 1; 0 -> 2 stays. The estimate has 0 -> 1 and 1 -> 2.
    truth = _dag([(0, 3), (3, 1), (0, 2)], d=4, hidden=(3,))
    assert projected_directed(truth) == {(0, 1), (0, 2)}
    marks = np.zeros((3, 3), dtype=np.int8)
    for i, j in ((0, 1), (1, 2)):
        marks[i, j], marks[j, i] = _TAIL, _ARROW
    scores = directed_edge_scores(GraphStructure.from_numpy(marks, kind="pag"), truth)
    assert scores["directed_n_est"] == scores["directed_projection_n_true"] == 2
    assert [scores[f"directed_projection_{s}"] for s in ("precision", "recall", "f1")] == [0.5] * 3
    assert scores["directed_projection_errors"] == 2


def test_a_perfect_claim_scores_one_for_either_mark() -> None:
    truth = _dag([(4, 0), (4, 1), (2, 0), (3, 1)], d=5, hidden=(4,))
    for mark, marks in (("bidirected", _ARROW), ("circle", _CIRCLE)):
        scores = confounded_pair_scores(_pag([(0, 1)], 4, marks), truth, mark)
        assert tuple(scores) == CONFOUNDED_COLUMNS
        assert scores["confounded_n_est"] == 1
        for target in CONFOUNDED_TARGETS:
            assert scores[f"confounded_{target}_n_true"] == 1
            assert [scores[f"confounded_{target}_{s}"] for s in ("precision", "recall", "f1")] == [
                1.0,
                1.0,
                1.0,
            ]


def test_the_pag_scores_match_andrey_metrics_where_both_are_defined() -> None:
    # One right pair and one wrong pair: precision 1/2, recall 1.
    truth = _dag([(4, 0), (4, 1), (2, 0), (3, 1)], d=5, hidden=(4,))
    claim = _pag([(0, 1), (2, 3)], 4, _ARROW)
    ours = confounded_pair_scores(claim, truth, "bidirected")
    theirs = M.confounder_pair_scores(claim, marginal(truth, "pag"))
    for stat in ("precision", "recall", "f1"):
        assert ours[f"confounded_pag_{stat}"] == pytest.approx(theirs[stat])
    assert ours["confounded_pag_precision"] == 0.5


def test_a_claim_is_read_by_its_own_mark_only() -> None:
    truth = _dag([(4, 0), (4, 1), (2, 0), (3, 1)], d=5, hidden=(4,))
    scores = confounded_pair_scores(_pag([(0, 1)], 4, _CIRCLE), truth, "bidirected")
    assert scores["confounded_n_est"] == 0


def test_an_empty_claim_has_no_precision_and_an_undefined_ratio_is_not_one() -> None:
    truth = _dag([(2, 0), (2, 1)], d=3, hidden=(2,))
    none = _pag([], 2, _ARROW)
    scores = confounded_pair_scores(none, truth, "bidirected")
    assert np.isnan(scores["confounded_hidden_cause_precision"])
    assert scores["confounded_hidden_cause_recall"] == 0.0
    assert scores["confounded_hidden_cause_f1"] == 0.0
    # No true pair in the PAG and none claimed: nothing to score, not a perfect score.
    assert all(np.isnan(scores[f"confounded_pag_{s}"]) for s in ("precision", "recall", "f1"))
    # A claim against an empty target is all false positives.
    claim = confounded_pair_scores(_pag([(0, 1)], 2, _ARROW), truth, "bidirected")
    assert claim["confounded_pag_precision"] == 0.0
    assert np.isnan(claim["confounded_pag_recall"])
    assert claim["confounded_pag_f1"] == 0.0


# ==================================================================================================
# The default campaign's datasets.
# ==================================================================================================


@pytest.mark.parametrize("regime", ["latent_lingam_er"])
def test_the_default_ladder_hides_a_node_with_two_observed_children(regime) -> None:
    """Every default dataset generates, and one of its hidden nodes confounds two observed ones."""
    hidden_per_d = {5: 1, 10: 2, 15: 3}
    for key in confounded_cpu.ladder_keys(regime, list(hidden_per_d), seeds=5, n=20):
        assert key.latents == hidden_per_d[key.d]
        data, marks, _, _ = datasets.generate_one(key)
        observed = key.d - key.latents
        assert data.shape == (20, observed)
        children = ((marks == _TAIL) & (marks.T == _ARROW))[observed:, :observed].sum(axis=1)
        assert children.max() >= 2, key.label
