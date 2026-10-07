"""The scoring entry point, :func:`score`.

``score`` runs every metric whose inputs the two structures carry and skips the rest. It reads the
structures alone, never which method produced them, so a graph scores the same way whatever
emitted it. The keys it returns are listed on :func:`score`.
"""

from __future__ import annotations

from andrey.core.output import StructureOutput
from andrey.core.structure import LATENT, GraphStructure, Kind, Structure, TemporalStructure

from ._common import two_marks
from .graph import (
    _confounder,
    _orientation,
    _shd_endpoint,
    _shd_structural,
    _skeleton,
    canonicalize_marks,
    reject_digraph_pag,
)
from .latent import latent_cluster_ari, number_of_latents
from .order import causal_order_scores
from .temporal import temporal_scores
from .weights import coefficient_mae


def score(output: object, truth: object) -> dict[str, object]:
    """Score an estimated result against ground truth with the family-appropriate metrics.

    ``output`` and ``truth`` may each be a StructureOutput, a Structure, or a raw adjacency, and
    must index the same variables in the same order (a label or node-count mismatch raises). A
    latent structure is scored on its clustering alone, since latent node indices are not aligned
    across the two graphs; CDNOD's appended domain node and similar extra variables must be dropped
    by the caller first.

    Returns
    -------
    dict
        A flat, JSON-safe dict whose ``family`` names the structures' kind. Every metric whose
        inputs both structures carry is present; the rest are absent.

        - Every graph: ``skeleton_precision``, ``skeleton_recall``, ``skeleton_f1``,
          ``arrowhead_precision``, ``arrowhead_recall``, ``arrowhead_f1``, and ``shd``.
        - ``dag`` adds ``mec_shd`` and ``mec_arrowhead_f1``, scored through the CPDAGs for methods
          identified only up to their equivalence class, and ``order_*`` and
          ``coefficient_mae`` when an ordering or weights are present.
        - ``pag`` adds ``shd_endpoint`` and ``confounder_precision``, ``confounder_recall``, and
          ``confounder_f1``.
        - ``digraph`` keeps the base metrics only: it has no CPDAG, and a 2-cycle claims no latent
          confounder. Comparing a ``digraph`` with a ``pag`` raises.
        - ``latent``: ``latent_cluster_ari``, ``n_latents_estimated``, and ``n_latents_true``.
        - ``temporal``: ``per_lag``, ``contemporaneous``, ``lagged``, and ``summary``, each
          nested, plus ``coefficient_mae_per_lag``.

    Examples
    --------
    >>> import numpy as np
    >>> score(np.array([[0, 1], [0, 0]]), np.array([[0, 1], [0, 0]]))["family"]
    'dag'
    """
    est_temporal = isinstance(_structure(output), TemporalStructure)
    true_temporal = isinstance(_structure(truth), TemporalStructure)
    if est_temporal or true_temporal:
        if est_temporal != true_temporal:
            raise ValueError("cannot compare a temporal structure to a graph structure")
        result = dict(temporal_scores(output, truth))
        result["family"] = "temporal"
        return result

    if _has_latent(output) or _has_latent(truth):
        return {
            "family": "latent",
            "latent_cluster_ari": latent_cluster_ari(output, truth),
            "n_latents_estimated": number_of_latents(output),
            "n_latents_true": number_of_latents(truth),
        }

    m_est, k_est, m_true, k_true = two_marks(output, truth)  # validates shape + labels
    family = _graph_family(k_est, k_true)
    result: dict[str, object] = {"family": family}
    _fill_graph(result, m_est, k_est, m_true, k_true, family)
    if family == "dag":
        _fill_directed_extras(result, output, truth)
    return result


def _structure(x: object) -> Structure | None:
    """The underlying Structure of an input, or ``None`` for a raw adjacency."""
    if isinstance(x, StructureOutput):
        return x.structure
    return x if isinstance(x, Structure) else None


def _has_latent(x: object) -> bool:
    s = _structure(x)
    return (
        isinstance(s, GraphStructure)
        and s.node_types is not None
        and bool((s.node_types == LATENT).any())
    )


def _graph_family(k_est: Kind, k_true: Kind) -> str:
    """Family by kind precedence: any digraph -> digraph, any PAG -> pag, any CPDAG -> cpdag,
    else dag. A ``digraph`` (directed, cyclic) has no essential graph and makes no confounder
    claim, so it scores on the base skeleton / arrowhead / SHD trio alone. A digraph paired with a
    PAG is rejected -- their shared ARROW/ARROW mark means a 2-cycle versus a latent confounder."""
    reject_digraph_pag(k_est, k_true)
    if k_est == "digraph" or k_true == "digraph":
        return "digraph"
    if k_est == "pag" or k_true == "pag":
        return "pag"
    if k_est == "cpdag" or k_true == "cpdag":
        return "cpdag"
    return "dag"


def _flat(prefix: str, scores: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in scores.items()}


def _fill_graph(result: dict, m_est, k_est: Kind, m_true, k_true: Kind, family: str) -> None:
    """Skeleton / SHD / orientation; canonicalizes CPDAGs and dual-reports MEC scores for DAGs."""
    est, true_ = m_est, m_true
    if family == "cpdag":  # a DAG and any MEC-equivalent member score through their CPDAG
        est, true_ = canonicalize_marks(m_est, k_est), canonicalize_marks(m_true, k_true)
    diag = family == "digraph"  # a digraph's autoregressive self-loops are real structure
    result.update(_flat("skeleton", _skeleton(est, true_, include_diagonal=diag)))
    result.update(_flat("arrowhead", _orientation(est, true_, include_diagonal=diag)))
    result["shd"] = _shd_structural(est, true_, include_diagonal=diag)
    if family == "pag":
        result["shd_endpoint"] = _shd_endpoint(est, true_)
        result.update(_flat("confounder", _confounder(est, true_)))
    if family == "dag":
        # ExactSearch / CALM emit kind="dag" yet identify only up to the MEC; report both the
        # directed scores above and the canonicalized ones, so the consumer picks by method. A
        # non-acyclic input has no CPDAG, so skip the MEC pair and keep the directed scores valid.
        try:
            canon_est = canonicalize_marks(m_est, k_est)
            canon_true = canonicalize_marks(m_true, k_true)
        except ValueError:
            return
        result["mec_shd"] = _shd_structural(canon_est, canon_true)
        result["mec_arrowhead_f1"] = _orientation(canon_est, canon_true)["f1"]


def _fill_directed_extras(result: dict, output: object, truth: object) -> None:
    """Order accuracy and coefficient MAE when the run carries an ordering / weights."""
    if isinstance(output, StructureOutput) and output.ordering is not None:
        order = causal_order_scores(output.ordering, truth)
        result["order_accuracy"] = order["accuracy"]
        result["order_kendall_tau"] = order["kendall_tau"]
        result["n_comparable_pairs"] = order["n_comparable_pairs"]
    est_w = output.weighted_adjacency if isinstance(output, StructureOutput) else None
    true_w = truth.weighted_adjacency if isinstance(truth, StructureOutput) else None
    if est_w is not None and true_w is not None:
        result["coefficient_mae"] = coefficient_mae(output, truth)
