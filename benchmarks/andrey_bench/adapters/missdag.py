"""MissDAG adapters: the authors' linear-Gaussian EM with their NOTEARS, run unmodified.

MissDAG (Gao et al. 2022) fits a DAG on data with missing values by EM: the E-step takes the
expected first and second moments of each row's missing block, and the M-step runs a DAG search on
the expected covariance. ``miss_methods/miss_dag_gaussian.py`` is its linear-Gaussian EM, and
``dag_methods/notears/notears.py`` the NOTEARS it calls as the M-step. ``benchmarks/start_missdag.sh``
downloads the repository at a pinned commit, checks its SHA-256, and builds the environment its
fits run in (``$ANDREY_BENCH_MISSDAG_PYTHON``); ``$ANDREY_BENCH_MISSDAG`` names the extracted
repository. The repository has no license file, so the benchmark imports its files from there and
copies none of them.

:class:`MissDagGaussian` runs ``miss_dag_gaussian(X, mask, Notears(lambda_1_ev), em_iter,
equal_variances)`` and :class:`MissDagNotears` the same NOTEARS on complete data, the ``no_missing``
path of the repository's ``main.py``. Both then threshold the weights with the repository's
``postprocess(B, graph_thres)``, as ``main.py`` does, and return the 0/1 adjacency of the DAG.

The package ``__init__`` files of ``miss_methods`` and ``dag_methods`` import the nonlinear and
non-Gaussian paths, which need torch, and ``utils/utils.py`` imports torch and TensorFlow for its
``set_seed`` alone. The loader registers the two packages without running their ``__init__`` files
and gives ``torch`` and ``tensorflow`` empty modules, so the linear-Gaussian path imports with
NumPy, SciPy, pandas, and NetworkX alone; none of the code it runs touches either library.

References
----------
.. [1] Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H. (2022).
   MissDAG: Causal discovery in the presence of missing data with continuous additive noise models.
   Advances in Neural Information Processing Systems 35.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

#: The variable naming the extracted MissDAG repository.
CODE_ENV = "ANDREY_BENCH_MISSDAG"
#: The commit ``start_missdag.sh`` pins; the adapters report it as their version.
COMMIT = "3c1eeed8f6245670614d2f2938ee3c3b5a21f17b"

# Constants of MissDAG's own code: the defaults of `--em_iter`, `--lambda_1_ev`, and `--graph_thres`
# in utils/config.py, which main.py passes to miss_dag_gaussian, Notears, and postprocess. The
# function's own default is em_iter=20; the paper and config.py run 10 iterations.
EM_ITER = 10
LAMBDA_1_EV = 2e-2
GRAPH_THRES = 0.3


def _load(code: str | Path) -> tuple[Callable, type, Callable]:
    """Import ``miss_dag_gaussian``, ``Notears``, and ``postprocess`` from the repository at `code`."""
    root = str(Path(code).resolve())
    if not Path(root, "miss_methods", "miss_dag_gaussian.py").is_file():
        raise FileNotFoundError(
            f"{root} holds no MissDAG checkout; run benchmarks/start_missdag.sh"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    for package in ("miss_methods", "dag_methods", "utils"):
        module = types.ModuleType(package)
        module.__path__ = [os.path.join(root, package)]
        sys.modules[package] = module
    for absent in ("torch", "tensorflow"):
        sys.modules.setdefault(absent, types.ModuleType(absent))
    gaussian = importlib.import_module("miss_methods.miss_dag_gaussian")
    notears = importlib.import_module("dag_methods.notears.notears")
    utils = importlib.import_module("utils.utils")
    return gaussian.miss_dag_gaussian, notears.Notears, utils.postprocess


class MissDagGaussian:
    """MissDAG's linear-Gaussian EM with its NOTEARS as the M-step.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. Fits run in ``$ANDREY_BENCH_MISSDAG_PYTHON``.
    """

    name: str = "missdag.miss_dag_gaussian.notears"
    package: str = "missdag"
    package_version: str = COMMIT[:12]
    backend: str = "numpy"
    mode: str = "serial"
    algorithm: str = "notears"
    output_type: str = "dag"
    python_env: str = "ANDREY_BENCH_MISSDAG_PYTHON"

    def setup(self) -> None:
        """Import the repository's modules outside fit timing."""
        self._fns = _load(os.environ.get(CODE_ENV, ""))

    def params(self) -> dict[str, Any]:
        """MissDAG's own settings, with equal noise variances and seed 0."""
        return {
            "em_iter": EM_ITER,
            "lambda_1_ev": LAMBDA_1_EV,
            "graph_thres": GRAPH_THRES,
            "equal_variances": True,
            "seed": 0,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the EM on `data`, ``NaN`` where a value is missing; return the DAG's 0/1 adjacency.

        ``miss_dag_gaussian`` draws its initial noise variances from NumPy's global generator, which
        ``params["seed"]`` seeds. Its mask is True where a value is observed.
        """
        miss_dag_gaussian, notears, postprocess = self._fns
        X = np.asarray(data, dtype=np.float64)
        np.random.seed(int(params["seed"]))
        B, _, _ = miss_dag_gaussian(
            X,
            ~np.isnan(X),
            notears(float(params["lambda_1_ev"])),
            int(params["em_iter"]),
            bool(params["equal_variances"]),
        )
        _, B_bin = postprocess(B, float(params["graph_thres"]))
        return (np.asarray(B_bin) != 0).astype(np.int8)

    def to_structure(self, native: np.ndarray) -> Any:
        """0/1 adjacency, row -> column, to a DAG ``GraphStructure``; untimed."""
        from andrey_bench.contracts import structure_from_adjacency

        return structure_from_adjacency(native, kind=self.output_type)


class MissDagNotears(MissDagGaussian):
    """MissDAG's NOTEARS on complete data, the ``no_missing`` path of its ``main.py``.

    The control of the EM comparison: the same optimizer without the EM layer.
    """

    name: str = "missdag.notears"

    def params(self) -> dict[str, Any]:
        """NOTEARS's penalty and the threshold, as :class:`MissDagGaussian` runs them."""
        return {"lambda_1_ev": LAMBDA_1_EV, "graph_thres": GRAPH_THRES}

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run NOTEARS on complete `data`; return the DAG's 0/1 adjacency.

        Raises
        ------
        ValueError
            If `data` has a missing value.
        """
        _, notears, postprocess = self._fns
        X = np.asarray(data, dtype=np.float64)
        if np.isnan(X).any():
            raise ValueError("missdag.notears takes complete data")
        B = notears(float(params["lambda_1_ev"])).fit(X=X, cov_emp=None)
        _, B_bin = postprocess(B, float(params["graph_thres"]))
        return (np.asarray(B_bin) != 0).astype(np.int8)
