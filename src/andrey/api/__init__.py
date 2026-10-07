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

from .constraint import cdnod, fci, gfci, pc
from .irregular import pnl
from .latent import gin
from .lingam import direct_lingam, ica_lingam, multi_group_direct_lingam
from .permutation import boss, grasp
from .score import calm, exact_search, ges, gies, hc
from .temporal import longitudinal_lingam, varma_lingam

__all__ = [
    # constraint-based
    "pc",
    "fci",
    "gfci",
    "cdnod",
    # score-based
    "ges",
    "gies",
    "hc",
    "exact_search",
    "calm",
    # permutation-based
    "boss",
    "grasp",
    # LiNGAM family
    "direct_lingam",
    "ica_lingam",
    "multi_group_direct_lingam",
    # temporal family
    "varma_lingam",
    "longitudinal_lingam",
    # irregular / pairwise family
    "pnl",
    # latent-variable family
    "gin",
]
