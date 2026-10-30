"""Andrey -- accelerated, accurate, agentic causal discovery.

Install ``andrey-core``; the import name is ``andrey``. Quick start, with ``data`` a pandas
DataFrame or a NumPy array of one row per sample and one column per variable::

    import andrey

    out = andrey.pc(data)  # a StructureOutput; the graph is out.structure
    print(out)

Every algorithm (``andrey.ges``, ``andrey.fci``, ...) returns one ``StructureOutput`` wrapping a
first-class ``Structure`` (a ``GraphStructure`` or ``TemporalStructure``). The public graph types
are re-exported here (``andrey.GraphStructure``) alongside ``andrey.core.*``.

For a coding agent, the command line carries the same methods: ``andrey --skill`` prints the guide
to save as its ``SKILL.md``, and ``andrey --help --json`` prints every method's contract as JSON.
"""

from .api import (
    anm,
    boss,
    bottom_up_parce_lingam,
    calm,
    cam,
    cdnod,
    dagma,
    direct_lingam,
    exact_search,
    fci,
    ges,
    gfci,
    gies,
    gin,
    granger,
    grasp,
    hc,
    iamb,
    ica_lingam,
    inter_iamb,
    longitudinal_lingam,
    lpcmci,
    mmhc,
    mmpc,
    multi_group_direct_lingam,
    notears,
    pc,
    pcmci,
    pcmciplus,
    pnl,
    rcd,
    resit,
    rsmax2,
    tabu,
    varma_lingam,
    xges,
)
from .core import GraphStructure, Structure, StructureOutput, SummaryGraph, TemporalStructure
from .core.backend import config, describe
from .core.knowledge import Knowledge
from .core.seeding import seed_all
from .core.warning_policy import (
    AndreyWarning,
    BackendFallbackWarning,
    BoundaryWarning,
    DataTypeWarning,
    DeprecatedArgumentWarning,
    ExperimentalWarning,
    PerformanceWarning,
    SampleSizeWarning,
    SearchLimitWarning,
)
from .resample import BootstrapOutput, bootstrap

__version__ = "0.11.0"

__all__ = [
    "__version__",
    "Structure",
    "GraphStructure",
    "TemporalStructure",
    "SummaryGraph",
    "StructureOutput",
    # Background knowledge every search takes as knowledge=.
    "Knowledge",
    # Refit a method on resamples of the rows: edge frequencies and an averaged graph.
    "bootstrap",
    "BootstrapOutput",
    # Seeds random, NumPy's global generator, and torch for the caller's own draws.
    "seed_all",
    # Backend grammar: pick the backend + n_jobs, layered selection (see ANDREY_BACKEND).
    "config",
    "describe",
    # Warning categories: every warning Andrey raises is an AndreyWarning.
    "AndreyWarning",
    "BackendFallbackWarning",
    "BoundaryWarning",
    "DataTypeWarning",
    "DeprecatedArgumentWarning",
    "ExperimentalWarning",
    "PerformanceWarning",
    "SearchLimitWarning",
    "SampleSizeWarning",
    # Algorithm facades - one uniform StructureOutput each.
    "pc",
    "fci",
    "gfci",
    "cdnod",
    "iamb",
    "inter_iamb",
    "mmpc",
    "ges",
    "gies",
    "xges",
    "hc",
    "tabu",
    "exact_search",
    "calm",
    "notears",
    "dagma",
    "mmhc",
    "rsmax2",
    "boss",
    "grasp",
    "direct_lingam",
    "ica_lingam",
    # Nonlinear additive noise.
    "cam",
    "resit",
    # Latent-variable structure - observed indicators plus discovered latents in causal order.
    "gin",
    # Multi-output: one StructureOutput per group (shared order, per-group weights).
    "multi_group_direct_lingam",
    # Temporal family - one StructureOutput over a per-lag stack.
    "varma_lingam",
    "longitudinal_lingam",
    "pcmci",
    "pcmciplus",
    "lpcmci",
    "granger",
    # Pairwise direction tests.
    "anm",
    "pnl",
    # Hidden common causes: a PAG over the observed variables.
    "rcd",
    "bottom_up_parce_lingam",
]
