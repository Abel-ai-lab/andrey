"""The scoring bridge: which target each estimate is scored against.

CPDAG estimates are scored against DAG truth in equivalence-class space; PAG estimates against the
truth's PAG, and a truth with hidden nodes as its observed PAG. Tiny fixtures, no runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import andrey.metrics as M
from andrey import GraphStructure
from andrey.core import LATENT, OBSERVED
from andrey.data.latent import marginal
from andrey_bench.scoring import score_result

# Andrey endpoint marks: `TAIL=1`, `ARROW=2`, `CIRCLE=3`.
_TAIL, _ARROW, _CIRCLE = 1, 2, 3


def _dag_marks(edges: list[tuple[int, int]], d: int) -> np.ndarray:
    """Endpoint-mark matrix for a DAG with the given ``i -> j`` edges."""
    m = np.zeros((d, d), dtype=np.uint8)
    for i, j in edges:
        m[i, j] = _TAIL
        m[j, i] = _ARROW
    return m


# ==================================================================================================
# Scoring-bridge behavior: CPDAG-space projection and the PAG target.
# ==================================================================================================


def test_score_result_scores_cpdag_estimate_against_dag_truth_in_cpdag_space():
    # A CPDAG estimate that is the essential graph of the truth must score SHD 0, even though the
    # DAG truth is fully oriented. ``score_result`` hands both to ``andrey.metrics.score`` as they
    # are; the projection is the primitive's, and this is where it is held to it.
    truth = GraphStructure.from_numpy(_dag_marks([(0, 1), (1, 2)], d=3), kind="dag")  # 0->1->2
    estimate = M.to_cpdag(truth)  # all-undirected chain CPDAG
    assert estimate.kind == "cpdag"
    result = score_result(estimate, truth)
    assert result["family"] == "cpdag"
    assert result["shd"] == 0
    assert result["skeleton_f1"] == 1.0


def _collider_with_child() -> GraphStructure:
    """0 -> 2 <- 1 and 2 -> 3. Its PAG is 0 o-> 2 <-o 1, 2 -> 3: circles at 0 and 1, which no
    observational method can resolve, and a tail at 2 that rule R1 can."""
    return GraphStructure.from_numpy(_dag_marks([(0, 2), (1, 2), (2, 3)], d=4), kind="dag")


def test_the_true_pag_is_a_perfect_pag_estimate():
    """Score a PAG estimate against the truth's PAG.

    Scoring against the DAG would charge correct FCI circles as errors, penalizing a perfect PAG.
    """
    truth = _collider_with_child()
    pag = marginal(truth, "pag")
    # The fixture has marks the DAG fixes and the PAG does not.
    assert (pag.to_numpy() == _CIRCLE).any()
    result = score_result(pag, truth)
    assert result["family"] == "pag"
    assert (result["shd"], result["shd_endpoint"]) == (0, 0)
    assert result["arrowhead_f1"] == 1.0


def test_a_pag_that_misses_the_collider_is_still_charged():
    """Circles replacing the truth PAG's collider arrowheads count as errors."""
    truth = _collider_with_child()
    skeleton = (truth.to_numpy() != 0).astype(np.uint8) * _CIRCLE  # every edge o-o
    result = score_result(GraphStructure.from_numpy(skeleton, kind="pag"), truth)
    assert result["skeleton_f1"] == 1.0
    assert result["shd"] > 0 and result["arrowhead_recall"] < 1.0


def test_each_truth_gets_its_own_pag_target():
    """Sequential datasets each use their own PAG target, without reusing the preceding target."""
    first = _collider_with_child()
    second = GraphStructure.from_numpy(_dag_marks([(3, 1), (2, 1), (1, 0)], d=4), kind="dag")
    for truth in (first, second, first):
        assert score_result(marginal(truth, "pag"), truth)["shd"] == 0


def test_a_truth_is_projected_once_for_all_its_pag_estimates(monkeypatch):
    """A dataset's solutions and repeats share one PAG projection; it takes minutes at `d = 800`.

    DAG and CPDAG estimates require no PAG projection.
    """
    import andrey_bench.scoring as scoring

    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0])
        return marginal(*args, **kwargs)

    monkeypatch.setattr(scoring, "marginal", counting)
    truth = _collider_with_child()
    pag = marginal(truth, "pag")
    for _ in range(3):
        score_result(pag, truth)
    score_result(truth, truth)
    score_result(M.to_cpdag(truth), truth)
    assert calls == [truth]


@pytest.mark.parametrize("kind", ["dag", "cpdag"])
def test_a_dag_or_cpdag_estimate_is_scored_against_the_dag(kind):
    """Only PAG estimates change the target.

    DAGs and CPDAGs pass unchanged to `andrey.metrics.score`.
    """
    truth = _collider_with_child()
    estimate = GraphStructure.from_numpy(_dag_marks([(0, 2), (2, 1), (2, 3)], d=4), kind="dag")
    if kind == "cpdag":
        estimate = M.to_cpdag(estimate)
    assert score_result(estimate, truth) == dict(M.score(estimate, truth))


def _confounded() -> GraphStructure:
    """Return a DAG with 4 -> 0, 4 -> 1, 2 -> 0, 3 -> 1 and node 4 hidden.

    Its observed PAG is 2 o-> 0 <-> 1 <-o 3.
    """
    marks = _dag_marks([(4, 0), (4, 1), (2, 0), (3, 1)], d=5)
    types = np.array([OBSERVED] * 4 + [LATENT], dtype=np.int8)
    return GraphStructure.from_numpy(marks, kind="dag", node_types=types)


def test_a_truth_with_a_hidden_node_is_scored_as_its_observed_pag():
    truth = _confounded()
    pag = marginal(truth, "pag")
    assert pag.n_nodes == 4
    assert pag.endpoints(0, 1) == (_ARROW, _ARROW)  # the hidden common cause
    result = score_result(pag, truth)
    assert result["family"] == "pag"
    assert (result["shd"], result["shd_endpoint"]) == (0, 0)


def test_the_empty_graph_is_scored_against_the_observed_pag():
    """Check that `baseline.empty`'s CPDAG is scored against truth with hidden nodes."""
    truth = _confounded()
    empty = GraphStructure.from_numpy(np.zeros((4, 4), dtype=np.uint8), kind="cpdag")
    result = score_result(empty, truth)
    assert result["family"] == "pag"
    assert result["shd"] == 3  # 2 o-> 0, 0 <-> 1, 1 <-o 3
    assert result["skeleton_recall"] == 0.0
