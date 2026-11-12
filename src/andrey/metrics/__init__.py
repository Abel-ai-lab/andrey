"""``andrey.metrics`` -- score a result against a known ground-truth structure, by output type.

Standalone and method-agnostic: every metric is a pure function of an estimated structure and a true
one, so what a metric computes depends on what the structure carries (its kind, whether it is
temporal, whether it holds an ordering / weights / latent flags), never on which algorithm produced
it. Depends only on :mod:`andrey.core` and NumPy.

Two layers:

- individual metric functions -- ``shd``, ``skeleton_scores``, ``orientation_scores``,
  ``confounder_pair_scores``, ``causal_order_accuracy``, ``coefficient_mae``,
  ``direction_accuracy``, ``temporal_scores``, ``latent_cluster_ari``, ``target_edge_scores``,
  ``claim_scores``, and friends;
- the oracle answers of local methods and the scores of answers that may abstain --
  ``target_oracle``, ``effect_oracle``, ``answer_scores``, ``target_answer_scores``,
  ``effect_scores``, and ``adjustment_scores`` for one adjustment set;
- :func:`score`, a capability resolver that reads which fields a result carries and returns the
  family-appropriate dict, silently skipping metrics whose inputs are absent.
"""

from .answers import answer_scores, pooled_answer_scores
from .effects import adjustment_scores, effect_oracle, effect_scores
from .graph import (
    confounder_pair_scores,
    markov_equivalent,
    orientation_scores,
    shd,
    skeleton_scores,
    to_cpdag,
)
from .latent import adjusted_rand_index, latent_cluster_ari, number_of_latents
from .local import claim_scores, target_answer_scores, target_edge_scores, target_oracle
from .order import (
    causal_order_accuracy,
    causal_order_kendall_tau,
    causal_order_scores,
)
from .pairwise import decision_rate, direction_accuracy
from .resolve import score
from .temporal import temporal_scores
from .weights import coefficient_mae

__all__ = [
    "score",
    "shd",
    "skeleton_scores",
    "orientation_scores",
    "confounder_pair_scores",
    "markov_equivalent",
    "to_cpdag",
    "causal_order_accuracy",
    "causal_order_kendall_tau",
    "causal_order_scores",
    "coefficient_mae",
    "direction_accuracy",
    "decision_rate",
    "temporal_scores",
    "latent_cluster_ari",
    "adjusted_rand_index",
    "number_of_latents",
    "target_edge_scores",
    "claim_scores",
    "target_oracle",
    "target_answer_scores",
    "effect_oracle",
    "effect_scores",
    "adjustment_scores",
    "answer_scores",
    "pooled_answer_scores",
]
