"""Public wrappers over the Andrey engines.

Each algorithm is a callable returning one uniform ``StructureOutput`` that wraps a first-class
``Structure`` (``.structure``; concrete ``GraphStructure`` / ``TemporalStructure``). A
``GraphStructure`` exposes ``.adjacency`` (lazy dense ndarray) plus ``.to_numpy`` /
``.to_scipy_sparse`` / ``.to_networkx`` exports and the ``.oriented_edges`` list, and the envelope
adds ``.ordering`` / ``.weighted_adjacency`` / ``.metadata``. A DataFrame's column names become the
structure's ``labels``. Each callable is re-exported at the top level
(``andrey.pc`` ...).

Wrappers share :mod:`andrey.api._adapt` for adjacency conversion, JSON-safe metadata, and
``StructureOutput`` construction.
"""

from .blanket import iamb, inter_iamb, mmpc
from .constraint import cdnod, fci, gfci, pc
from .continuous import dagma, notears
from .fcm import cam
from .hybrid import mmhc, rsmax2
from .irregular import anm, bottom_up_parce_lingam, pnl, rcd
from .latent import gin
from .lingam import direct_lingam, ica_lingam, multi_group_direct_lingam
from .permutation import boss, grasp
from .score import calm, exact_search, ges, gies, hc, tabu, xges
from .temporal import granger, longitudinal_lingam, lpcmci, pcmci, pcmciplus, varma_lingam

__all__ = [
    # constraint-based
    "pc",
    "fci",
    "gfci",
    "cdnod",
    # local structure: one variable's Markov blanket or parents and children
    "iamb",
    "inter_iamb",
    "mmpc",
    # score-based
    "ges",
    "gies",
    "xges",
    "hc",
    "tabu",
    "exact_search",
    "calm",
    "notears",
    "dagma",
    # hybrid: a restrict phase of CI tests, then a score search
    "mmhc",
    "rsmax2",
    # permutation-based
    "boss",
    "grasp",
    # LiNGAM family
    "direct_lingam",
    "ica_lingam",
    "multi_group_direct_lingam",
    # nonlinear additive-noise family
    "cam",
    # temporal family
    "varma_lingam",
    "longitudinal_lingam",
    "pcmci",
    "pcmciplus",
    "lpcmci",
    "granger",
    # irregular / pairwise family
    "anm",
    "pnl",
    # latent-variable family
    "gin",
    # hidden-confounder family
    "rcd",
    "bottom_up_parce_lingam",
]
