"""andrey.core -- the public, composable substrate (provisional until 1.0).

The method-agnostic building blocks shared across algorithm families: the first-class ``Structure``
(an ABC with concrete ``GraphStructure`` / ``TemporalStructure`` subclasses) backed by one canonical
compact CSR store, and the ``StructureOutput`` run envelope that every algorithm returns. The
CI-test and score protocols live in ``andrey.core.ci`` and ``andrey.core.score``.

The statistical primitives (``cov`` / ``corrcoef`` / ``entropy``) live in ``andrey.core.stats``; the
shared backend-selection layer they dispatch through is ``andrey.core.backend``. Both stay off
``__all__`` until a public consumer needs them, so torch is imported only behind the ``[torch]``
extra and never at package import. Method-specific fused kernels stay private -- not here.
"""

from .output import StructureOutput
from .structure import (
    ARROW,
    CIRCLE,
    EDGE_DTYPE,
    LATENT,
    NULL,
    OBSERVED,
    TAIL,
    GraphStructure,
    Kind,
    Structure,
    SummaryGraph,
    TemporalStructure,
)

__all__ = [
    "Structure",
    "GraphStructure",
    "TemporalStructure",
    "SummaryGraph",
    "StructureOutput",
    # Endpoint mark codes + graph-type aliases - adapter authors need these for the
    # signed->unsigned remap and edge-list construction.
    "NULL",
    "TAIL",
    "ARROW",
    "CIRCLE",
    # Node-type codes for latent-variable methods (GIN); see GraphStructure.node_types.
    "OBSERVED",
    "LATENT",
    "Kind",
    "EDGE_DTYPE",
]
