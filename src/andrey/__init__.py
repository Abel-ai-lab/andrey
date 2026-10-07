"""Andrey -- scalable causal discovery (familiar methods, accelerated on CPU & GPU).

Every algorithm returns one ``StructureOutput`` wrapping a first-class ``Structure`` (a
``GraphStructure`` or ``TemporalStructure``). The public graph types are re-exported here
for intuitive top-level access (``andrey.GraphStructure``) alongside ``andrey.core.*``; algorithm
callables (``andrey.ges``, ...) are exposed with their facade wrappers.
"""

from .api import (
    boss,
    calm,
    cdnod,
    direct_lingam,
    exact_search,
    fci,
    ges,
    gfci,
    gies,
    gin,
    grasp,
    hc,
    ica_lingam,
    longitudinal_lingam,
    multi_group_direct_lingam,
    pc,
    pnl,
    varma_lingam,
)
from .core import GraphStructure, Structure, StructureOutput, SummaryGraph, TemporalStructure
from .core.backend import config, describe
from .core.seeding import seed_all
from .core.warning_policy import (
    AndreyWarning,
    BackendFallbackWarning,
    ExperimentalWarning,
    PerformanceWarning,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Structure",
    "GraphStructure",
    "TemporalStructure",
    "SummaryGraph",
    "StructureOutput",
    # Seeds random, NumPy's global generator, and torch for the caller's own draws.
    "seed_all",
    # Backend grammar: pick the backend + num_workers, layered selection (see ANDREY_BACKEND).
    "config",
    "describe",
    # Warning categories: every warning Andrey raises is an AndreyWarning.
    "AndreyWarning",
    "BackendFallbackWarning",
    "ExperimentalWarning",
    "PerformanceWarning",
    # Algorithm facades - one uniform StructureOutput each.
    "pc",
    "fci",
    "gfci",
    "cdnod",
    "ges",
    "gies",
    "hc",
    "exact_search",
    "calm",
    "boss",
    "grasp",
    "direct_lingam",
    "ica_lingam",
    # Latent-variable structure - observed indicators plus discovered latents in causal order.
    "gin",
    # Multi-output: one StructureOutput per group (shared order, per-group weights).
    "multi_group_direct_lingam",
    # Temporal family - one StructureOutput over a per-lag stack.
    "varma_lingam",
    "longitudinal_lingam",
    # Pairwise direction test.
    "pnl",
]
