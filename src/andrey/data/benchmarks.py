"""Named benchmark SCM presets -- the reproducible substrate for competitive benchmarking.

Each benchmark is a family parametrized by node count ``d``, spanning the mechanism/topology/noise
combinations the benchmark grid needs (linear-Gaussian, linear-non-Gaussian, additive-nonlinear,
post-nonlinear, and a latent-confounded family). Build one with :func:`get` (or :func:`scm`) and
cross it with sizes and seeds via :class:`~andrey.data.generate.Ensemble`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import functional, graphs, noise
from .scm import SCM


@dataclass(frozen=True)
class Benchmark:
    """A benchmark SCM family: ``at(d)`` builds the scenario at node count ``d``."""

    name: str
    description: str
    build: Callable[[int], SCM]

    def at(self, d: int) -> SCM:
        """Build the benchmark SCM at ``d`` nodes."""
        return self.build(d)


_BENCHMARKS: dict[str, Benchmark] = {}


def _register(name: str, description: str, build: Callable[[int], SCM]) -> None:
    _BENCHMARKS[name] = Benchmark(name=name, description=description, build=build)


_register(
    "linear_gauss_er",
    "Linear-Gaussian Erdos-Renyi (PC / GES territory)",
    lambda d: SCM(
        graph=graphs.erdos_renyi(d, 3.0), functional=functional.linear(), noise=noise.gaussian()
    ),
)
_register(
    "lingam_sf",
    "Linear non-Gaussian scale-free (DirectLiNGAM)",
    lambda d: SCM(
        graph=graphs.scale_free(d, 2), functional=functional.linear(), noise=noise.uniform()
    ),
)
_register(
    "nonlinear_hub",
    "Additive-nonlinear hub (ANM)",
    lambda d: SCM(
        graph=graphs.hub(d), functional=functional.additive_noise(), noise=noise.gaussian()
    ),
)
_register(
    "pnl_small_world",
    "Post-nonlinear small-world (PNL)",
    lambda d: SCM(
        graph=graphs.small_world(d), functional=functional.post_nonlinear(), noise=noise.laplace()
    ),
)
_register(
    "latent_confounded_er",
    "Linear-Gaussian Erdos-Renyi with hidden confounders (FCI / GFCI)",
    lambda d: SCM(
        graph=graphs.erdos_renyi(d, 3.0),
        functional=functional.linear(),
        noise=noise.gaussian(),
        latents=max(1, d // 10),
    ),
)
_register(
    "var_gauss_er",
    "VAR(1) Gaussian temporal series on an Erdos-Renyi instantaneous DAG (modest d)",
    lambda d: SCM(
        graph=graphs.erdos_renyi(d, 2.0),
        functional=functional.var(n_lags=1),
        noise=noise.gaussian(),
    ),
)


def names() -> list[str]:
    """Return the registered benchmark names, sorted."""
    return sorted(_BENCHMARKS)


def get(name: str) -> Benchmark:
    """Return the :class:`Benchmark` registered under ``name``."""
    if name not in _BENCHMARKS:
        raise KeyError(f"unknown benchmark {name!r}; available: {names()}")
    return _BENCHMARKS[name]


def scm(name: str, d: int) -> SCM:
    """Build the named benchmark's SCM at ``d`` nodes (shorthand for ``get(name).at(d)``)."""
    return get(name).at(d)
