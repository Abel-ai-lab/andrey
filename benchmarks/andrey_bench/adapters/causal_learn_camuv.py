"""causal-learn CAM-UV adapter for nonlinear additive data with hidden variables, in its own
environment.

Calls ``CAMUV.execute(X, 0.01, 3)`` (``alpha``, ``num_explanatory_vals``), the settings of
:mod:`~andrey_bench.adapters.andrey_camuv` in causal-learn's names, as
:mod:`~andrey_bench.adapters.lingam_camuv` maps them, and builds the marks of
:func:`~andrey_bench.adapters.lingam_camuv.camuv_marks` from the parents and pairs it returns, a
pair drawn ``<->``. causal-learn's CAM-UV fits its smooth functions with pygam, which the bench
environment does not hold, so its fits run in the environment
:mod:`~andrey_bench.adapters.lingam_env` describes, where causal-learn 0.1.4.8 is installed beside
lingam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench.adapters.lingam_camuv import DEFAULTS, camuv_marks
from andrey_bench.adapters.lingam_env import PYTHON_ENV_VAR, env_version


@dataclass(frozen=True)
class CausalLearnCAMUV:
    """Solution adapter for causal-learn's CAM-UV; build it with :func:`causal_learn_camuv_adapter`.

    Satisfies :class:`andrey_bench.contracts.SolutionAdapter`.
    """

    name: str = "causal-learn.camuv"
    package: str = "causal-learn"
    package_version: str = ""
    backend: str = "native"
    mode: str = "serial"
    algorithm: str = "camuv"
    output_type: str = "pag"
    #: The endpoint marks that carry the pairs this method attributes to a hidden variable.
    confounded_mark: str = "bidirected"
    python_env: str = PYTHON_ENV_VAR

    def setup(self) -> None:
        """Import causal-learn's CAM-UV before timing; see `contracts.SetupAdapter`."""
        from causallearn.search.FCMBased.lingam import CAMUV  # noqa: F401

    def params(self) -> dict[str, Any]:
        """Return the settings in Andrey's names."""
        return dict(DEFAULTS)

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run causal-learn CAM-UV; return the PAG's endpoint-mark array. Timed."""
        from causallearn.search.FCMBased.lingam import CAMUV

        X = np.asarray(data, dtype=np.float64)
        parents, pairs = CAMUV.execute(X, float(params["alpha"]), int(params["max_depth"]) + 1)
        return camuv_marks(parents, pairs, X.shape[1])

    def to_structure(self, native: np.ndarray) -> Any:
        """Rebuild the PAG from its endpoint marks outside the timed region."""
        from andrey import GraphStructure

        return GraphStructure.from_numpy(native, kind=self.output_type)


def causal_learn_camuv_adapter() -> CausalLearnCAMUV:
    """A :class:`CausalLearnCAMUV` carrying its environment's causal-learn version."""
    return CausalLearnCAMUV(package_version=env_version("causal-learn"))
