"""Recovery checks for ``mmb_by_mmb``, MMB-by-MMB (Xie, Li, Wu, Zeng, Liu, and Geng 2024).

On simulated linear-Gaussian data with hidden common causes, pooled over every target, the answer
is scored against the edges each target has in the PAG of the observed variables
(``target_answer_scores``): its adjacencies beat the empty answer's, and its marks are right more
often than wrong, which the empty answer, with no mark, is not. The paper's blanket learner, total
conditioning, and IAMB both pass.
"""

from __future__ import annotations

import warnings

import pytest

import andrey
from andrey import data
from andrey.metrics import pooled_answer_scores, target_answer_scores, target_oracle


@pytest.mark.parametrize("blanket", ["total_conditioning", "iamb"])
def test_mmb_by_mmb_beats_the_empty_answer(blanket):
    # 20 nodes, 3 of them hidden common causes: 17 columns.
    sample = data.sample_scm(
        graph="erdos_renyi", d=20, n=5000, seed=0, density=2.0, latents=3, scale="standardize"
    )
    d = sample.data.shape[1]
    hidden = range(d, sample.graph.n_nodes)
    adjacency, marks = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        for t in range(d):
            out = andrey.mmb_by_mmb(sample.data, target=t, blanket=blanket)
            truth = target_oracle(sample.graph, t, "pag", latent=hidden)
            scores = target_answer_scores(out.structure, truth, t)
            adjacency.append(scores["adjacency"])
            marks.append(scores["marks"])
    adjacency, marks = pooled_answer_scores(adjacency), pooled_answer_scores(marks)
    # The empty answer says False for every pair; it gets the true adjacencies wrong.
    assert adjacency["net"] > adjacency["policy_net"]["False"]
    assert adjacency["correct"] > 0.9 * adjacency["queries"]
    # The empty answer has no edge, so no mark: a net of zero.
    assert marks["net"] > 0
