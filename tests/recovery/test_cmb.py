"""Recovery checks for ``cmb``, Causal Markov Blanket discovery (Gao and Ji 2015).

On simulated linear-Gaussian and categorical data, pooled over every target, the answer is scored
against the edges each target has in the true CPDAG (``target_answer_scores``): its adjacencies
beat the empty answer's, and its marks are right more often than wrong, which the empty answer,
with no mark, is not. The paper's blanket learner, HITON-MB, and MMMB both pass.
"""

from __future__ import annotations

import warnings

import pytest

import andrey
from andrey import data
from andrey.metrics import pooled_answer_scores, target_answer_scores, target_oracle


def _sample_of(kind: str):
    """Linear-Gaussian data on 20 variables, or categorical data on 10, with their settings."""
    if kind == "gaussian":
        sample = data.sample_scm(
            graph="erdos_renyi", d=20, n=2000, seed=0, density=2.0, scale="standardize"
        )
        return sample, {}
    return data.sample_discrete(d=10, n=3000, seed=1, density=2.0), {"indep_test": "gsq"}


@pytest.mark.parametrize("blanket", ["hiton_mb", "mmmb"])
@pytest.mark.parametrize("kind", ["gaussian", "categorical"])
def test_cmb_beats_the_empty_answer(kind, blanket):
    sample, settings = _sample_of(kind)
    adjacency, marks = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        for t in range(sample.data.shape[1]):
            out = andrey.cmb(sample.data, target=t, blanket=blanket, **settings)
            scores = target_answer_scores(out.structure, target_oracle(sample.graph, t, "cpdag"), t)
            adjacency.append(scores["adjacency"])
            marks.append(scores["marks"])
    adjacency, marks = pooled_answer_scores(adjacency), pooled_answer_scores(marks)
    # The empty answer says False for every pair; it gets the true adjacencies wrong.
    assert adjacency["net"] > adjacency["policy_net"]["False"]
    assert adjacency["correct"] > 0.9 * adjacency["queries"]
    # The empty answer has no edge, so no mark: a net of zero.
    assert marks["net"] > 0
