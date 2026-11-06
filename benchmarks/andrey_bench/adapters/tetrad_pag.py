"""Tetrad's FCI, FCI-JCI, BOSS-FCI, and GRaSP-FCI on continuous data, through causal-cmd's driver.

Each solution is a :class:`~andrey_bench.adapters.tetrad_cmd._TetradCmd`, so it times its search
as the GFCI solution does: ``fit`` writes the data to a CSV, starts ``java`` once on
``TetradDriver.java`` with causal-cmd's arguments, and reads the PAG back from the text file the
driver writes. The record's ``wall_s`` is Tetrad's search inside the JVM, after the warm-up searches
in the same JVM, and its ``process_s`` is the whole call. BOSS-FCI is causal-cmd's ``bfci``, which
it lists only under ``--experimental``. Tetrad's setting names map to Andrey's:

===================================  ============================================================
Tetrad (causal-cmd)                  Andrey
===================================  ============================================================
``fisher-z-test``                    ``indep_test="fisherz"``
``alpha``                            ``alpha``
``sem-bic-score``                    ``score_func="local_score_BIC"``
``penaltyDiscount``                  ``lambda_value``
``depth``                            ``depth``, Andrey's ``max_depth``
``--knowledge`` tier 1               ``context_vars``, the context variables' columns
===================================  ============================================================

Every search runs the complete rule set R1-R10 and the discriminating-path rule with no cap on
discriminating paths, and with no cap on conditioning sets unless ``params`` give ``depth``
(``completeRuleSetUsed``, ``doDiscriminatingPathRule``, ``maxPathLength -1``, ``depth -1``). FCI
runs the possible-d-separation step (``possibleMsepDone``). causal-cmd 1.12.0's ``bfci`` and
``grasp-fci`` (tetrad-lib 7.6.3's ``BFci`` and ``GraspFci``) take their FCI steps from GFCI's, one
pass of greedy separating sets over the score search's adjacencies, and have no
possible-d-separation step. The two score searches run at Tetrad's documented defaults: the BIC
with no structure prior and the standard penalty rule (``semBicStructurePrior 0``, ``semBicRule
1``), one start (``numStarts 1``), the covariance matrix read once (``precomputeCovariances``), and
no seed (``seed -1``). BOSS-FCI runs without BES; GRaSP-FCI runs from the data's column order
(``useDataOrder``), its stages in order (``graspOrderedAlg``), scored (``graspUseScore``, which
causal-cmd does not expose and leaves at its default), with singular and nonsingular tuck depths of
1, and ``depth`` as GRaSP's tuck depth too: causal-cmd sets ``graspDepth`` and then ``depth``.
causal-cmd sets a boolean it is not given to false, not to its documented default, so each one on
by default is named; a boolean that is off is left out, since causal-cmd turns a boolean on
whenever it is named.

FCI-JCI is FCI with the context variables in Tetrad's first knowledge tier, so no system variable
causes a context variable: JCI assumption 1 only. Tetrad's FCI in causal-cmd 1.12.0 has no setting
to leave out the rules for selection bias, which pcalg's FCI-JCI leaves out. The PAG covers every
column, context variables included.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.tetrad_cmd import _TetradCmd
from andrey_bench.adapters.tetrad_gfci import read_pag

_DEFAULT_ALPHA = 0.05
_DEFAULT_LAMBDA = 1.0
#: The FCI steps every search runs, by causal-cmd's names; ``params`` may cap ``depth``.
_FCI_STEPS = {
    "maxPathLength": -1,
    "completeRuleSetUsed": True,
    "doDiscriminatingPathRule": True,
}
#: The score settings BOSS-FCI and GRaSP-FCI share, by causal-cmd's names.
_SCORE = {
    "semBicStructurePrior": 0.0,
    "semBicRule": 1,
    "numStarts": 1,
    "precomputeCovariances": True,
    "seed": -1,
}


def flags(settings: Mapping[str, Any]) -> list[str]:
    """causal-cmd's arguments for ``settings``: a boolean on is named alone, one off is left out."""
    out: list[str] = []
    for name, value in settings.items():
        if value is True:
            out.append(f"--{name}")
        elif value is not False:
            out += [f"--{name}", repr(value)]
    return out


class _TetradPAG(_TetradCmd):
    """One causal-cmd PAG search with the Fisher-Z test, and the BIC score where it takes one."""

    output_type: str = "pag"
    data_type: str = "continuous"
    #: causal-cmd's name for the algorithm.
    command: str
    #: Whether the search takes ``--score sem-bic-score``.
    scored: bool = False
    #: The search's own settings by causal-cmd's names, beyond the test level, penalty, and steps.
    extra: dict[str, Any] = {}

    def params(self) -> dict[str, Any]:
        """Andrey's names for the shared settings; :meth:`settings` maps them to Tetrad's."""
        shared = {"alpha": _DEFAULT_ALPHA, "indep_test": "fisherz", "context_vars": []}
        if self.scored:
            shared |= {"score_func": "local_score_BIC", "lambda_value": _DEFAULT_LAMBDA}
        return shared

    def settings(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """Tetrad's settings, by causal-cmd's names, for Andrey's ``params``."""
        alpha = {"alpha": float(params.get("alpha", _DEFAULT_ALPHA))}
        penalty = {"penaltyDiscount": float(params.get("lambda_value", _DEFAULT_LAMBDA))}
        depth = {"depth": int(params.get("depth", -1))}
        steps = _FCI_STEPS | depth
        return {**alpha, **(penalty | _SCORE if self.scored else {}), **steps, **self.extra}

    def _arguments(self, params: Mapping[str, Any]) -> list[str]:
        command = ["--algorithm", self.command, "--test", "fisher-z-test"]
        command += ["--score", "sem-bic-score"] if self.scored else []
        return command + flags(self.settings(params))

    def read(self, text: str, d: int) -> np.ndarray:
        """The driver's PAG text as endpoint marks."""
        return read_pag(text, d)

    def to_structure(self, native: np.ndarray) -> Any:
        """PAG endpoint marks -> ``GraphStructure``; untimed."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


class TetradCmdFCI(_TetradPAG):
    """Tetrad's FCI with the Fisher-Z test: order-independent skeleton and possible-d-separation."""

    name = "tetrad.cmd.fci.fisherz"
    algorithm = "fci"
    command = "fci"
    extra = {"stableFAS": True, "possibleMsepDone": True}


class TetradCmdFCIJCI(TetradCmdFCI):
    """Tetrad's FCI with ``context_vars`` in the first knowledge tier, as FCI-JCI's context."""

    name = "tetrad.cmd.fci_jci.fisherz"
    algorithm = "fci_jci"

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run FCI with the context tier; raises ``ValueError`` without ``context_vars``."""
        if not params.get("context_vars"):
            raise ValueError(f"{self.name} needs context_vars, the context variables' columns")
        return super().fit(data, params)


class TetradBFCI(_TetradPAG):
    """Tetrad's BOSS-FCI (causal-cmd's experimental ``bfci``) with Fisher-Z and the BIC."""

    name = "tetrad.cmd.boss_fci.bic"
    algorithm = "boss_fci"
    command = "bfci"
    experimental = True
    scored = True
    extra = {"useBes": False}


class TetradCmdGraspFCI(_TetradPAG):
    """Tetrad's GRaSP-FCI with Fisher-Z and the BIC, one start from the column order.

    causal-cmd sets GRaSP-FCI's depth from ``graspDepth`` and then from ``depth``, so ``depth``
    (-1, no cap) is GRaSP's tuck depth and the FCI steps' cap alike.
    """

    name = "tetrad.cmd.grasp_fci.bic"
    algorithm = "grasp_fci"
    command = "grasp-fci"
    scored = True
    extra = {
        "graspSingularDepth": 1,
        "graspNonSingularDepth": 1,
        "graspOrderedAlg": True,
        "graspUseRaskuttiUhler": False,
        "useDataOrder": True,
    }
