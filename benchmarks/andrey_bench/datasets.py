"""Generate and store benchmark datasets with ``andrey.data.sample_scm``.

Each ``(topology, functional, noise, density, d, n, seed, standardize, latents)`` dataset is stored
once as a ``.npz`` containing ``float64`` samples and the true graph's int8 endpoint marks. The
``categorical`` functional samples a discrete network with ``andrey.data.sample_discrete`` instead;
its samples are category codes. The ``network_tables`` functional samples a published network
from its own conditional probability tables with ``andrey.data.load_dataset``, the key's topology
naming the network. The ``conditional_gaussian`` functional samples a mixed network with
``andrey.data.sample_mixed``: category codes in its categorical columns, values in the rest.
:class:`DatasetKey` supplies the ``dataset_id``; :func:`load` casts to ``float32`` for those fits
outside timing, without storing another copy.

A temporal dataset (functional ``var``) is one series of ``n`` time steps. Its truth is stored as
the ``(L, d, d)`` int8 marks of lags ``1..L`` (:func:`~andrey_bench.contracts.lag_marks`), with
``graph_kind`` ``"temporal"`` in the registry.

The regimes of the PCMCI, PCMCI+, and LPCMCI papers are drawn by
:mod:`andrey_bench.temporal_models` instead (functional ``pcmci2019``, ``pcmciplus2020``, or
``lpcmci2020``; the key's density is the model's cross-links per variable). A model without
same-step links stores its truth as lag marks with the kind ``"temporal"``; a model with them
stores the ``(L + 1, d, d)`` window marks of lags ``0..L`` (:mod:`andrey_bench.window`) with the
kind ``"temporal_window"``. Unobserved variables are the key's ``latents``, last in the truth.

The regimes of the VAR-LiNGAM and longitudinal LiNGAM papers are drawn by
:mod:`andrey_bench.lingam_models` (functional ``varma2010`` or ``panel2013``) and store their truth
as a coefficient stack (:mod:`andrey_bench.coefficients`), with the kind ``"temporal_coefficients"``
or ``"panel_coefficients"``. A panel key's ``times`` is its number of time points, and its ``n``
the number of units observed at each; a panel dataset's data has shape ``(times, n, d)``.

The structural vector autoregressions of the DYNOTEARS paper are drawn by
:mod:`andrey_bench.svar_models` (functional ``dynotears2020`` and its variants; the key's density is
the same-step graph's mean degree) and store their truth as window marks over lags 0 and 1, with
the kind ``"temporal_window"``.

The data of CD-NOD, MultiGroupDirectLiNGAM, and GIN are drawn by :mod:`andrey_bench.regime_models`
instead (functional ``domain_shift``, ``multi_group``, or ``measurement``); that module gives each
layout, truth, and column order.

A key's ``shuffle`` lists the columns in an order drawn from the seed, with the truth permuted the
same way, so the column order does not follow the graph's construction (:class:`DatasetKey`).
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

import andrey.data as data
from andrey import TemporalStructure
from andrey.core import LATENT, OBSERVED
from andrey_bench import lingam_models, regime_models, svar_models, temporal_models
from andrey_bench.contracts import lag_marks

# Canonical on-disk dtype. An f32 fit down-casts at load.
CANONICAL_DTYPE = np.float64

REGISTRY_NAME = "registry.json"
_SCHEMA_VERSION = 3

# --- regime label -> andrey.data.sample_scm argument maps ------------------------------------------
# Kept explicit (not introspected): the mapping is part of the frozen contract, and a rename in
# Andrey should fail a test rather than silently regenerate every dataset under an old label.
TOPOLOGY_TO_GRAPH = {
    "er": "erdos_renyi",
    "scale_free": "scale_free",
    "small_world": "small_world",
    "hub": "hub",
}
FUNCTIONAL_TO_ANDREY = {
    "linear": "linear",
    "anm": "additive_noise",
    "lsnm": "location_scale",
    "pnl": "post_nonlinear",
    "nonmonotone": "nonmonotone",
    "var": "var",
}
STANDARDIZE_TO_SCALE = {
    "standardized": "standardize",
    "raw": "raw",
    "ancestral": "rescale",  # Andrey's "ancestral rescaling" scale (per SCM.sample docstring)
    "unitless": "unitless",  # weights and noise that give each node unit variance (SCM.sample)
}
#: Selectable noise families.
NOISES = {"gaussian", "uniform", "laplace", "exponential", "gumbel"}

#: Benchmark regimes as ``(topology, functional, noise, standardize)`` tuples, named for the
#: corresponding ``andrey.data`` families and translated into ``sample_scm`` arguments.
REGIMES: dict[str, tuple[str, str, str, str]] = {
    # sparse linear-Gaussian, where PC/GES are identifiable and every solution runs
    "linear_gauss_er": ("er", "linear", "gaussian", "standardized"),
    # non-Gaussian: DirectLiNGAM is only identifiable here
    "lingam_sf": ("scale_free", "linear", "uniform", "standardized"),
    # nonlinear additive noise
    "nonlinear_hub": ("hub", "anm", "gaussian", "standardized"),
    # post-nonlinear
    "pnl_small_world": ("small_world", "pnl", "laplace", "standardized"),
    # Linear-Gaussian with hidden confounders; the key sets `latents` using LATENT_REGIMES.
    "latent_gauss_er": ("er", "linear", "gaussian", "standardized"),
    # Linear with uniform noise and hidden confounders, where RCD and BottomUpParceLiNGAM are
    # identifiable.
    "latent_lingam_er": ("er", "linear", "uniform", "standardized"),
    # The four linear regimes with unitless weights and noise (Herman et al. 2025): every variable
    # has unit variance and R^2-sortability is close to chance, so neither the variances nor the
    # share of each variable the others explain gives the causal order away. The generator keeps a
    # dense correlation matrix and takes at most 1,000 variables.
    "linear_gauss_er_unitless": ("er", "linear", "gaussian", "unitless"),
    "lingam_sf_unitless": ("scale_free", "linear", "uniform", "unitless"),
    "latent_gauss_er_unitless": ("er", "linear", "gaussian", "unitless"),
    "latent_lingam_er_unitless": ("er", "linear", "uniform", "unitless"),
    # Cause-effect pairs: nonlinear additive noise on two nodes, where ANM is identifiable. At
    # ``d = 2`` and density 1.0 every pair has one edge, in a direction the seed draws.
    "anm_pair": ("er", "anm", "gaussian", "standardized"),
    # Cause-effect pairs: post-nonlinear ``y = g2(w * x + e)`` on two nodes with Laplace noise, where
    # PNL is identifiable. The inner transform is linear and the outer one nonlinear and invertible.
    # At ``d = 2`` and density 1.0 every pair has one edge, in a direction the seed draws.
    "pnl_pair": ("er", "pnl", "laplace", "standardized"),
    # Nonlinear additive noise without hidden nodes: PC with the nonlinear CI tests.
    "anm_er": ("er", "anm", "gaussian", "standardized"),
    # The regimes of the functional-causal-model lanes (CAM, RESIT): additive noise with Gaussian and
    # with Gumbel noise, the location-scale generator, and linear-Gaussian data as the control, each
    # z-scored (the default) and at raw scale. ``anm_er`` and ``linear_gauss_er`` are the z-scored
    # Gaussian pair.
    "anm_gumbel_er": ("er", "anm", "gumbel", "standardized"),
    "lsnm_er": ("er", "lsnm", "gaussian", "standardized"),
    "anm_er_raw": ("er", "anm", "gaussian", "raw"),
    "anm_gumbel_er_raw": ("er", "anm", "gumbel", "raw"),
    "lsnm_er_raw": ("er", "lsnm", "gaussian", "raw"),
    "linear_gauss_er_raw": ("er", "linear", "gaussian", "raw"),
    # Additive noise through cos, an even function: a child is uncorrelated with a symmetric
    # parent, so partial correlation misses edges the nonlinear CI tests find.
    "nonmonotone_er": ("er", "nonmonotone", "gaussian", "standardized"),
    # Categorical data from a discrete network, Dirichlet tables on an ER DAG: the chi-square and
    # G-square tests and the BDeu score. Codes are categories, so they are never rescaled.
    "discrete_er": ("er", "categorical", "dirichlet", "raw"),
    # Mixed data from a conditional linear Gaussian network on an ER DAG: the degenerate and
    # conditional Gaussian tests and scores. The key's `categorical_share` and `categories` set the
    # categorical columns; codes and values are never rescaled.
    "mixed_er": ("er", "conditional_gaussian", "gaussian", "raw"),
    # Linear-Gaussian rows pooled from several domains, where some means move with the domain:
    # CD-NOD's data (andrey_bench.regime_models; the key's density is the graph's mean degree).
    "domain_shift_er": ("er", "domain_shift", "gaussian", "ancestral"),
    # Several groups over one linear non-Gaussian DAG, each with its own weights:
    # MultiGroupDirectLiNGAM's data; the key's d is the variables in each group.
    "multi_group_er": ("er", "multi_group", "uniform", "ancestral"),
    # Latents that each cause every later one, each measured by observed indicators of its own:
    # GIN's data; the key's d counts latents and indicators, its density is indicators per latent.
    "latent_measurement": ("complete", "measurement", "uniform", "ancestral"),
    # A lagged linear-Gaussian series for PCMCI: andrey.data's VAR(1) on an empty instantaneous
    # graph (the key sets density LAG_ONLY_DENSITY), so every true link has lag 1, at raw scale.
    "var_lagged": ("er", "var", "gaussian", "raw"),
    # The simulations of the PCMCI, PCMCI+, and LPCMCI papers, which keep the expected number of
    # cross-links per variable fixed at every d; andrey_bench.temporal_models gives each setup and
    # its source: Runge et al. (2019), Supplementary Sect. S4.1; Runge (2020), Sect. 4; Gerhardus
    # and Runge (2020), Sect. 4.
    "pcmci_paper": ("random", "pcmci2019", "gaussian", "raw"),
    "pcmciplus_paper": ("random", "pcmciplus2020", "gaussian", "raw"),
    "lpcmci_paper": ("random", "lpcmci2020", "gaussian", "raw"),
    # The simulations of the VAR-LiNGAM and longitudinal LiNGAM papers, with non-Gaussian noise;
    # andrey_bench.lingam_models gives each setup and its source: Hyvarinen et al. (2010), Sect.
    # 5.4, with one moving-average matrix; Kadowaki et al. (2013), JSAI Sect. 6.1.
    "varma_lingam_paper": ("random", "varma2010", "nongaussian", "raw"),
    "longitudinal_lingam_paper": ("random", "panel2013", "nongaussian", "raw"),
    # The structural vector autoregressions of the DYNOTEARS paper, one lag, at raw scale, with
    # Gaussian or centered exponential noise, Erdos-Renyi graphs or the Barabasi-Albert and
    # block-model pair; andrey_bench.svar_models gives each setup and its source: Pamfil et al.
    # (2020), Appendix B.4. The z-scored, uncentered-noise, and nonlinear regimes lie outside the
    # paper's model or scale.
    "dynotears_paper": ("er", "dynotears2020", "gaussian", "raw"),
    "dynotears_paper_exponential": ("er", "dynotears2020", "exponential", "raw"),
    "dynotears_paper_ba_sbm": ("ba_sbm", "dynotears2020_ba_sbm", "gaussian", "raw"),
    "dynotears_paper_standardized": ("er", "dynotears2020", "gaussian", "standardized"),
    "dynotears_paper_uncentered": ("er", "dynotears2020", "exponential_uncentered", "raw"),
    "dynotears_sin_tanh": ("er", "dynotears2020_sin_tanh", "gaussian", "raw"),
}

#: The functional label of :data:`REGIMES` that ``andrey.data.sample_discrete`` generates, with the
#: arguments it takes beyond the key's coordinates: categories per node and the Dirichlet
#: concentration of each table row. They are recorded in the generator call, so the digest follows
#: them.
DISCRETE_FUNCTIONAL = "categorical"
DISCRETE_SETTINGS: dict[str, object] = {"categories": [2, 3], "concentration": 1.0}

#: The functional label ``andrey.data.sample_mixed`` generates, and the argument it takes beyond the
#: key's coordinates: the Dirichlet concentration of each categorical table row. The categorical
#: share and the categories per categorical column travel on the key.
MIXED_FUNCTIONAL = "conditional_gaussian"
MIXED_SETTINGS: dict[str, object] = {"concentration": 1.0}

#: The functional label of a published network's rows, sampled from its own conditional probability
#: tables by ``andrey.data.load_dataset(..., discrete=True)``. A network key's topology is the
#: dataset name, its noise :data:`NETWORK_NOISE`, its density 0, and its ``d`` the network's
#: variables (:func:`network_key`): the network fixes its graph and tables.
NETWORK_FUNCTIONAL = "network_tables"
NETWORK_NOISE = "tables"

#: Temporal regimes and the density their keys set. At density 0 the instantaneous graph has no
#: edges, so the truth is lag-only: lag 1 holds every link.
LAG_ONLY_DENSITY = 0.0
TEMPORAL_REGIMES: dict[str, float] = {
    "var_lagged": LAG_ONLY_DENSITY,
    # The paper models' cross-links per variable.
    **{
        regime: temporal_models.MODELS[REGIMES[regime][1]].cross_degree
        for regime in ("pcmci_paper", "pcmciplus_paper", "lpcmci_paper")
    },
    # The LiNGAM paper models record no density: each fixes its own link counts.
    "varma_lingam_paper": 0.0,
    "longitudinal_lingam_paper": 0.0,
    # The DYNOTEARS models' same-step mean degree.
    **{
        regime: svar_models.MODELS[functional].same_step_degree
        for regime, (_, functional, _, _) in REGIMES.items()
        if functional in svar_models.MODELS
    },
}

#: The registry's ``graph_kind`` for a truth stored as window marks.
WINDOW_KIND = "temporal_window"

#: The registry's ``graph_kind`` for a truth stored as a coefficient stack, by model.
COEFFICIENT_KINDS = {"varma2010": "temporal_coefficients", "panel2013": "panel_coefficients"}

#: Functionals whose column order carries no information about the graph, so their keys take no
#: shuffle step: the PCMCI-family models draw each variable independently and link uniformly drawn
#: pairs, the DYNOTEARS models relabel their nodes by a random permutation, the longitudinal LiNGAM
#: panel lists its variables in a random order, and the GIN measurement model lists its indicators
#: in a random order. A model that draws its graph with
#: ``andrey.data.graphs``, as the other :mod:`~andrey_bench.regime_models` models do, takes the step.
ORDER_FREE = frozenset({*temporal_models.MODELS, *svar_models.MODELS, "panel2013", "measurement"})

#: Regimes with hidden confounders, mapped to the number of generated nodes per hidden node. At
#: least one node is hidden. Andrey's ``latent_confounded_er`` family uses the same rule as
#: ``latent_gauss_er``, at ER density 3.0 instead of 2.0.
#:
#: The two non-Gaussian regimes hide one node in five, so ``d = 5, 10, 15`` hide one, two, and three.
#: Each hidden node has two or more children. The hidden node latest in causal order has no hidden
#: child, so at least one hidden node has two observed children at every ``d``. A seed fails to
#: generate when fewer nodes have two or more children than nodes to hide. That is about 3% of seeds
#: at ``d = 10`` and ``d = 15``; seeds 0 to 28 all generate on that ladder.
LATENT_REGIMES: dict[str, int] = {
    "latent_gauss_er": 10,
    "latent_lingam_er": 5,
    "latent_gauss_er_unitless": 10,
    "latent_lingam_er_unitless": 5,
}

DEFAULT_REGIME = "linear_gauss_er"

# The canonical generated edge density: the degree-2 slice, the sparse regime the GES adapters'
# limit of 4 parents is sized for. Density travels on the key, so a denser run is a different
# dataset, a different file, and a different digest.
CANONICAL_DENSITY = 2.0

#: Shared dataset directory; defaults to ``<out>/data`` per run. A shared store avoids duplicate
#: files and gives every run identical bytes.
DATA_STORE_ENV = "ANDREY_BENCH_DATA"


#: The process umask, read once. `mkstemp` ignores it, and a store shared through `ANDREY_BENCH_DATA`
#: needs the mode an ordinary `open` gives.
_UMASK = os.umask(0)
os.umask(_UMASK)
#: Held while a writer reads, merges, and replaces the registry.
_LOCK_NAME = ".registry.lock"


def _scratch(out: Path, stem: str) -> Path:
    """Create a temporary path in `out` unique to its writer.

    Renames are atomic regardless of the temporary filename. Concurrent writers also need unique
    names. Campaign jobs share a store across nodes, where PIDs can repeat; `mkstemp` reserves a
    unique name through the shared filesystem. It creates the file 0600, so the file is given the
    mode the umask allows instead.
    """
    fd, name = tempfile.mkstemp(prefix=f".{stem}.", suffix=".tmp", dir=out)
    os.fchmod(fd, 0o666 & ~_UMASK)
    os.close(fd)
    return Path(name)


def latents_for(regime: str, d: int) -> int:
    """Return how many of `d` nodes `regime` hides, or 0 for a regime without hidden nodes."""
    if regime not in REGIMES:
        raise KeyError(f"unknown regime {regime!r}; known: {sorted(REGIMES)}")
    functional = REGIMES[regime][1]
    if functional in regime_models.MODELS:
        return regime_models.latents_for(functional, d)
    if functional in temporal_models.MODELS:
        return temporal_models.hidden_count(functional, d)
    per = LATENT_REGIMES.get(regime)
    return 0 if per is None else max(1, d // per)


def latent_node_types(d: int, latents: int) -> np.ndarray | None:
    """Return node types for a truth with `d` nodes and the last `latents` nodes hidden.

    Return `None` when no nodes are hidden. `andrey.data.latent.latent_order` puts hidden nodes
    last, as `marginal` requires.
    """
    if not latents:
        return None
    types = np.full(d, OBSERVED, dtype=np.int8)
    types[d - latents :] = LATENT
    return types


def store_dir(out: str | Path) -> Path:
    """The dataset store for a run: the shared one if set, else this run's own ``data/``."""
    shared = os.environ.get(DATA_STORE_ENV, "").strip()
    return Path(shared) if shared else Path(out) / "data"


@dataclass(frozen=True)
class DatasetKey:
    """Generator coordinates identifying one materialized dataset.

    :attr:`digest` is the ``dataset_id`` component of ``run_id``. Machine, core count, and dtype
    belong to :data:`~andrey_bench.contracts.RUN_ENV_FIELDS`; dtype changes are load-time casts of
    the canonical file. ``density`` affects every topology through ``sample_scm``'s builder mapping.

    ``latents`` nodes are generated and then dropped from the data. The data has ``d - latents``
    columns; the stored truth keeps all ``d`` nodes. At ``0``, ``latents`` affects neither the label
    nor the generator call. Datasets without hidden nodes therefore keep their identity.

    ``times`` is a panel's number of time points, with ``n`` units at each. At ``0``, its default, it
    affects neither the label nor the generator call either.

    ``categorical_share`` and ``categories`` reach the generator only for the mixed functional,
    :data:`MIXED_FUNCTIONAL`, where they join the label too. Every other key keeps its identity.

    ``shuffle`` lists the observed columns in an order drawn from the seed and permutes the truth
    the same way; hidden nodes stay last. It enters the generator call, and so the digest, only
    when on, and a shuffled key's label ends in ``-shuffled``: an unshuffled key keeps the label and
    digest its dataset has in records measured on the generated order. A functional in
    :data:`ORDER_FREE` takes no shuffle step and needs ``shuffle`` on.
    """

    topology: str
    functional: str
    noise: str
    density: float
    d: int
    n: int
    seed: int
    standardize: str
    latents: int = 0
    categorical_share: float = 0.5
    categories: tuple[int, int] = (2, 3)
    times: int = 0
    shuffle: bool = True

    @property
    def label(self) -> str:
        """Human-readable, filesystem-safe key string (also the registry map key).

        Carries ``density``: :func:`materialize` dedupes on the label, so a label blind to
        density would merge a dense key into the sparse key's file.
        """
        hidden = f"-L{self.latents}" if self.latents else ""
        mixed = (
            f"-cat{self.categorical_share:g}-k{self.categories[0]}to{self.categories[1]}"
            if self.functional == MIXED_FUNCTIONAL
            else ""
        )
        panel = f"-T{self.times}" if self.times else ""
        shuffled = "-shuffled" if self.shuffle and self.functional not in ORDER_FREE else ""
        return (
            f"{self.topology}-{self.functional}-{self.noise}-deg{self.density:g}"
            f"-d{self.d}-n{self.n}-s{self.seed}-{self.standardize}{hidden}{mixed}{panel}{shuffled}"
        )

    @property
    def digest(self) -> str:
        """Hash the generator call returned by :func:`generation_kwargs`.

        Only coordinates in the generator call affect the hash. Sorted JSON keys make the identity
        independent of dictionary order.
        """
        payload = json.dumps(generation_kwargs(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha1(payload.encode()).hexdigest()[:12]

    @property
    def filename(self) -> str:
        return f"ds_{self.digest}.npz"


def generation_kwargs(key: DatasetKey) -> dict[str, object]:
    """Return the ``andrey.data.sample_scm`` arguments for ``key``, including ``key.density``.

    The registry stores this call for reproduction. Unmapped labels raise ``KeyError`` or
    ``ValueError``. A categorical key returns the ``andrey.data.sample_discrete`` arguments, with
    ``"generator"`` naming it, and a mixed key the ``andrey.data.sample_mixed`` arguments. The call
    of a paper model (:mod:`~andrey_bench.temporal_models`, :mod:`~andrey_bench.lingam_models`) or of
    a :mod:`~andrey_bench.regime_models` model is its sampler's, with the model's whole setup, so a
    change to the setup is a new dataset.

    A shuffled key's call holds ``shuffle=True``. An unshuffled key's call leaves ``shuffle`` out,
    and :func:`generate_one` passes ``shuffle=False`` to the generator. A key of an
    :data:`ORDER_FREE` functional raises ``ValueError`` with ``shuffle`` off.
    """
    if key.functional in ORDER_FREE and not key.shuffle:
        raise ValueError(f"{key.label}: {key.functional} has no shuffle step to turn off")
    shuffle = {"shuffle": True} if key.shuffle else {}
    if key.functional in temporal_models.MODELS:
        call: dict[str, object] = {
            "generator": "andrey_bench.temporal_models.sample",
            "model": key.functional,
            "setup": temporal_models.model_params(key.functional),
            "d": key.d,
            "n": key.n,
            "seed": key.seed,
        }
        if key.density != temporal_models.MODELS[key.functional].cross_degree:
            raise ValueError(f"{key.label}: density must be the model's cross-links per variable")
        if key.latents != temporal_models.hidden_count(key.functional, key.d):
            raise ValueError(f"{key.label}: latents must be the model's unobserved variables")
        return call
    if key.functional in svar_models.MODELS:
        model = svar_models.MODELS[key.functional]
        if (
            key.topology != svar_models.TOPOLOGIES[key.functional]
            or key.density != model.same_step_degree
            or key.latents
            or key.times
        ):
            raise ValueError(
                f"{key.label}: topology and density are the model's, latents and times are 0"
            )
        return {
            "generator": "andrey_bench.svar_models.sample",
            "model": key.functional,
            "setup": svar_models.model_params(key.functional),
            "d": key.d,
            "n": key.n,
            "seed": key.seed,
            "noise": key.noise,
            "standardize": key.standardize,
        }
    if key.functional in lingam_models.MODELS:
        panel = key.functional == "panel2013"
        if panel != bool(key.times) or key.latents:
            raise ValueError(f"{key.label}: times is set for panel2013 alone, and latents is 0")
        call = {
            "generator": f"andrey_bench.lingam_models.{'sample_panel' if panel else 'sample_varma'}",
            "model": key.functional,
            "setup": lingam_models.model_params(key.functional),
            "d": key.d,
            "n": key.n,
            "seed": key.seed,
        }
        if panel:
            call["times"] = key.times
        else:
            call.update(shuffle)
        return call
    if key.functional == MIXED_FUNCTIONAL:
        if key.standardize != "raw" or key.latents:
            raise ValueError(
                f"{key.label}: mixed data are raw codes and values with no hidden nodes"
            )
        return {
            "generator": "sample_mixed",
            "graph": TOPOLOGY_TO_GRAPH[key.topology],
            "d": key.d,
            "n": key.n,
            "seed": key.seed,
            "density": key.density,
            "categorical_share": key.categorical_share,
            "categories": list(key.categories),
            **MIXED_SETTINGS,
            **shuffle,
        }
    if key.functional == NETWORK_FUNCTIONAL:
        if key.standardize != "raw" or key.latents or key.density:
            raise ValueError(f"{key.label}: network rows are raw codes, no hidden nodes, density 0")
        return {
            "generator": "andrey.data.load_dataset",
            "name": key.topology,
            "n": key.n,
            "seed": key.seed,
            "discrete": True,
            **shuffle,
        }
    if key.functional == DISCRETE_FUNCTIONAL:
        if key.standardize != "raw" or key.latents:
            raise ValueError(f"{key.label}: categorical data are raw codes with no hidden nodes")
        return {
            "generator": "sample_discrete",
            "graph": TOPOLOGY_TO_GRAPH[key.topology],
            "d": key.d,
            "n": key.n,
            "seed": key.seed,
            "density": key.density,
            **DISCRETE_SETTINGS,
            **shuffle,
        }
    if key.functional in regime_models.MODELS:
        if key.latents != regime_models.latents_for(key.functional, key.d):
            raise ValueError(f"{key.label}: latents must be the model's latent count")
        spec = regime_models.MODELS[key.functional]
        if isinstance(spec, regime_models.Measurement) and key.density != spec.indicators:
            raise ValueError(f"{key.label}: density must be the model's indicators per latent")
        return {
            "generator": "andrey_bench.regime_models.sample",
            "model": key.functional,
            "setup": regime_models.model_params(key.functional),
            "d": key.d,
            "n": key.n,
            "seed": key.seed,
            "density": key.density,
            "noise": key.noise,
            **({} if key.functional in ORDER_FREE else shuffle),
        }
    try:
        graph = TOPOLOGY_TO_GRAPH[key.topology]
        functional = FUNCTIONAL_TO_ANDREY[key.functional]
        scale = STANDARDIZE_TO_SCALE[key.standardize]
    except KeyError as exc:  # pragma: no cover - defensive; the labels above are the whole map
        raise KeyError(f"no andrey.data mapping for regime label {exc!s} in {key.label}") from exc
    if key.noise not in NOISES:
        raise ValueError(f"unknown noise {key.noise!r} (andrey knows {sorted(NOISES)})")
    kwargs: dict[str, object] = {
        "graph": graph,
        "functional": functional,
        "noise": key.noise,
        "d": key.d,
        "n": key.n,
        "seed": key.seed,
        "scale": scale,
        "density": key.density,
        **shuffle,
    }
    if key.latents:
        kwargs["latents"] = key.latents
    return kwargs


def graph_record(key: DatasetKey) -> dict[str, object]:
    """Return ``{model, num_nodes, density, params}`` in the ``andrey.data`` provenance format.

    ``params_for_density`` maps the key's density to builder parameters. Keys match the sample's
    ``config.graph`` block. A :mod:`~andrey_bench.regime_models` model records its setup, and a
    paper model its cross-links per variable.
    """
    if key.functional in regime_models.MODELS:
        return {
            "model": key.functional,
            "num_nodes": key.d,
            "density": key.density,
            "params": regime_models.model_params(key.functional),
        }
    if key.functional in temporal_models.MODELS:
        return {
            "model": key.functional,
            "num_nodes": key.d,
            "density": key.density,
            "params": {"cross_links_per_variable": key.density},
        }
    if key.functional in svar_models.MODELS:
        return {
            "model": key.functional,
            "num_nodes": key.d,
            "density": key.density,
            "params": svar_models.model_params(key.functional),
        }
    if key.functional in lingam_models.MODELS:
        return {
            "model": key.functional,
            "num_nodes": key.d,
            "density": key.density,
            "params": lingam_models.model_params(key.functional),
        }
    if key.functional == NETWORK_FUNCTIONAL:
        return {"model": key.topology, "num_nodes": key.d, "density": key.density, "params": {}}
    model = TOPOLOGY_TO_GRAPH[key.topology]
    return {
        "model": model,
        "num_nodes": key.d,
        "density": key.density,
        "params": data.graphs.params_for_density(model, key.d, key.density),
    }


def generate_one(key: DatasetKey) -> tuple[np.ndarray, np.ndarray, str, list[int] | None]:
    """Return ``(data_f64, graph_marks_int8, graph_kind, column_order)`` for ``key``.

    Samples are contiguous canonical ``float64``. Truth uses ``ds.graph.to_numpy()`` endpoint marks,
    as consumed by the scorer and Andrey adapter. With hidden nodes, the truth covers all ``d``
    nodes. :func:`latent_node_types` rebuilds their types from the key. A temporal truth is
    returned as :func:`lag_only_marks` with the kind ``"temporal"``. A paper model's series and
    truth come from :mod:`andrey_bench.temporal_models` or :mod:`andrey_bench.lingam_models`, as
    the module docstring says, and a :mod:`~andrey_bench.regime_models` dataset's data and truth come
    from that module. A network key's rows and truth come from ``andrey.data.load_dataset``.

    ``column_order[j]`` is the observed column that column ``j`` holds with ``shuffle`` off: the
    identity for an unshuffled key, and ``None`` for an :data:`ORDER_FREE` functional.
    """
    kwargs = generation_kwargs(key)
    if key.functional in regime_models.MODELS:
        data_arr, marks, kind, columns = regime_models.sample(
            key.functional,
            d=key.d,
            n=key.n,
            seed=key.seed,
            density=key.density,
            noise=key.noise,
            shuffle=key.shuffle,  # an unshuffled key's call leaves it out
        )
        data_arr = np.ascontiguousarray(data_arr, dtype=CANONICAL_DTYPE)
        return data_arr, marks, kind, None if columns is None else columns.tolist()
    if key.functional in temporal_models.MODELS:
        series, window = temporal_models.sample(key.functional, key.d, key.n, key.seed)
        series = np.ascontiguousarray(series, dtype=CANONICAL_DTYPE)
        if temporal_models.MODELS[key.functional].same_step:
            return series, window, WINDOW_KIND, None
        return series, lag_marks(window[1:] != 0), "temporal", None
    if key.functional in svar_models.MODELS:
        series, window = svar_models.sample(
            key.functional, key.d, key.n, key.seed, key.noise, key.standardize
        )
        return series, window, WINDOW_KIND, None
    if key.functional == "varma2010":
        series, stack, columns = lingam_models.sample_varma(
            key.functional, key.d, key.n, key.seed, shuffle=key.shuffle
        )
        return series, stack, COEFFICIENT_KINDS[key.functional], columns.tolist()
    if key.functional == "panel2013":
        panel, stack = lingam_models.sample_panel(key.functional, key.d, key.times, key.n, key.seed)
        return panel, stack, COEFFICIENT_KINDS[key.functional], None
    if key.functional == NETWORK_FUNCTIONAL:
        ds = data.load_dataset(
            key.topology,
            n=key.n,
            seed=key.seed,
            discrete=True,
            shuffle=key.shuffle,
            as_frame=False,
        )
        if len(ds.feature_names) != key.d:
            raise ValueError(f"{key.label}: {key.topology} has {len(ds.feature_names)} variables")
        file_order = _network_names(key.topology)
        columns = [file_order.index(name) for name in ds.feature_names]
        marks = np.ascontiguousarray(ds.graph.to_numpy())
        return (
            np.ascontiguousarray(ds.data, dtype=CANONICAL_DTYPE),
            marks,
            str(ds.graph.kind),
            columns,
        )
    generator = kwargs.pop("generator", None)
    kwargs["shuffle"] = key.shuffle  # an unshuffled key's call leaves it out
    if generator == "sample_discrete":
        ds = data.sample_discrete(**kwargs)  # type: ignore[arg-type]
    elif generator == "sample_mixed":
        kwargs["categories"] = tuple(kwargs["categories"])  # type: ignore[arg-type]
        ds = data.sample_mixed(**kwargs)  # type: ignore[arg-type]
    else:
        ds = data.sample_scm(**kwargs)  # type: ignore[arg-type]
    data_arr = np.ascontiguousarray(ds.data, dtype=CANONICAL_DTYPE)
    columns = [int(j) for j in ds.provenance["column_order"]]
    graph = ds.graph
    if isinstance(graph, TemporalStructure):
        return data_arr, lag_only_marks(key, graph), "temporal", columns
    if not hasattr(graph, "to_numpy"):
        raise TypeError(
            f"graph for {key.label} has no to_numpy() (kind={type(graph).__name__}); "
            "the store holds a GraphStructure or a TemporalStructure truth"
        )
    types = getattr(graph, "node_types", None)
    hidden = np.flatnonzero(types == LATENT) if types is not None else np.arange(0)
    if not np.array_equal(hidden, np.arange(key.d - key.latents, key.d)):
        raise ValueError(f"{key.label}: hidden nodes are not the last {key.latents} nodes")
    marks = np.ascontiguousarray(graph.to_numpy())  # int8 endpoint marks
    return data_arr, marks, str(graph.kind), columns


def _network_names(name: str) -> list[str]:
    """The published network `name`'s variables in its file's order."""
    return list(
        data.load_dataset(name, n=2, discrete=True, shuffle=False, as_frame=False).feature_names
    )


def network_key(name: str, n: int, seed: int, *, shuffle: bool = True) -> DatasetKey:
    """Return the key of `n` rows sampled from the published network `name`'s own tables.

    The first call downloads the network's file into ``andrey.data``'s cache.
    """
    return DatasetKey(
        topology=name,
        functional=NETWORK_FUNCTIONAL,
        noise=NETWORK_NOISE,
        density=0.0,
        d=len(_network_names(name)),
        n=n,
        seed=seed,
        standardize="raw",
        shuffle=shuffle,
    )


def lag_only_marks(key: DatasetKey, truth: TemporalStructure) -> np.ndarray:
    """Return the ``(L, d, d)`` int8 marks of lags ``1..L`` of a lag-only temporal truth.

    Raises ``ValueError`` when the instantaneous (lag-0) graph has an edge or the lags are not
    ``1..L``: the stored stack holds lagged links only, and a lag-0 edge would be dropped unseen.
    """
    if 0 in truth.lags and np.asarray(truth.lag(0).to_numpy()).any():
        raise ValueError(
            f"{key.label}: the instantaneous graph has edges; a temporal key needs density "
            f"{LAG_ONLY_DENSITY:g} so that its truth is lag-only"
        )
    lags = [lag for lag in truth.lags if lag > 0]
    if lags != list(range(1, len(lags) + 1)):
        raise ValueError(f"{key.label}: truth lags {lags} are not 1..L")
    return np.stack([np.asarray(truth.lag(lag).to_numpy(), dtype=np.int8) for lag in lags])


def _same_call(recorded: Any, wanted: Any) -> bool:
    """Compare generator calls after JSON normalization, treating tuples and lists alike."""
    return recorded is not None and json.dumps(recorded, sort_keys=True) == json.dumps(
        wanted, sort_keys=True
    )


def _recorded_keys(registry_path: Path) -> dict[str, dict]:
    """The entries the registry at `registry_path` holds; none when there is no registry yet."""
    if not registry_path.exists():
        return {}
    recorded = json.loads(registry_path.read_text())
    if recorded["schema"] != _SCHEMA_VERSION:
        raise ValueError(
            f"{registry_path} is schema {recorded['schema']}, this writes {_SCHEMA_VERSION}; "
            "materialize into a fresh directory"
        )
    return recorded["keys"]


@contextmanager
def _registry_lock(out: Path) -> Iterator[None]:
    """Hold an exclusive lock on the store in `out`; closing the descriptor releases it.

    Opened read-only, so a writer that did not create the lock file can still take the lock.
    """
    fd = os.open(out / _LOCK_NAME, os.O_RDONLY | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def materialize(keys: Iterable[DatasetKey], out_dir: str | Path) -> dict:
    """Write unique datasets to ``out_dir`` and return the saved ``registry.json`` contents.

    Deduplicates by :attr:`DatasetKey.label`, including when multiple fit dtypes share a key.
    Each :func:`generate_one` result is saved as ``data`` and ``graph`` arrays in a ``.npz`` for all
    readers to reuse, with its ``column_order`` in the registry entry. Existing files are reused only
    when the recorded ``generation_call`` matches.

    Dataset and registry writes use temporary files and atomic renames to prevent partial files
    after interruption. The registry is re-read and merged under a store lock, preserving concurrent
    writers' entries and earlier rungs' generator calls.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Dedupe, preserving first-seen order for a stable registry.
    unique: dict[str, DatasetKey] = {}
    for key in keys:
        unique.setdefault(key.label, key)

    registry_path = out / REGISTRY_NAME
    entries = _recorded_keys(registry_path)
    # This call's entries, reused or new: the only ones merged into the registry at the end.
    mine: dict[str, dict] = {}

    for label, key in unique.items():
        path = out / key.filename
        recorded = entries.get(label) or {}
        # Compare the recorded call as well as file existence before reusing a dataset on resume.
        if path.exists() and _same_call(recorded.get("generation_call"), generation_kwargs(key)):
            mine[label] = recorded
            continue
        data_arr, marks, graph_kind, columns = generate_one(key)
        # Write uncompressed arrays for fast reloads. The temporary file prevents partial writes;
        # pass a file object so savez does not append another ``.npz`` suffix.
        tmp = _scratch(out, key.filename)
        with open(tmp, "wb") as fh:
            np.savez(fh, data=data_arr, graph=marks)
        os.replace(tmp, path)
        mine[label] = {
            "file": key.filename,
            "graph": graph_record(key),
            "functional": key.functional,
            "noise": key.noise,
            "n": key.n,
            "seed": key.seed,
            "standardize": key.standardize,
            "latents": key.latents,
            "graph_kind": graph_kind,
            "data_shape": list(data_arr.shape),
            "data_dtype": str(data_arr.dtype),
            "graph_shape": list(marks.shape),
            "graph_dtype": str(marks.dtype),
            "column_order": columns,
            "generation_call": generation_kwargs(key),
        }

    # Another job may have added datasets since `entries` was read, so the registry is read again and
    # merged under the store's lock. It goes through a temp for the same reason the ``.npz`` does: a
    # truncated registry cannot be re-read, and it holds the reuse check for every dataset on disk.
    with _registry_lock(out):
        entries = {**_recorded_keys(registry_path), **mine}
        registry = {
            "schema": _SCHEMA_VERSION,
            "generator": "andrey.data.sample_scm",
            "canonical_dtype": str(np.dtype(CANONICAL_DTYPE)),
            "n_datasets": len(entries),
            "keys": entries,
        }
        tmp = _scratch(out, REGISTRY_NAME)
        tmp.write_text(json.dumps(registry, indent=2, sort_keys=True))
        os.replace(tmp, registry_path)
    return registry


def data_path(out_dir: str | Path, key: DatasetKey) -> Path:
    """The materialized ``.npz`` path for ``key`` - the file :func:`materialize` wrote.

    A pure path derivation (does not read, cast, or generate): the runner hands this path straight to
    the child worker so the canonical bytes are serialized **once** at materialize time and never
    re-serialized per repeat.
    """
    return Path(out_dir) / key.filename


def load_registry(out_dir: str | Path) -> dict:
    """Read back the registry written by :func:`materialize`."""
    return json.loads((Path(out_dir) / REGISTRY_NAME).read_text())


def load(
    out_dir: str | Path,
    key: DatasetKey,
    *,
    dtype: str | np.dtype = "float64",
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(data, graph_marks)`` for ``key``, casting ``data`` to ``dtype`` **at load** (untimed).

    This is the single point where a ``float32`` run diverges from a ``float64`` one - both read the
    same canonical ``float64`` file and cast here, so there is never a second materialized copy.
    ``graph_marks`` is returned as stored (int8 endpoint marks).
    """
    with np.load(Path(out_dir) / key.filename) as z:
        arr = np.ascontiguousarray(z["data"], dtype=np.dtype(dtype))
        marks = z["graph"].copy()
    return arr, marks


__all__ = [
    "CANONICAL_DTYPE",
    "CANONICAL_DENSITY",
    "REGIMES",
    "LATENT_REGIMES",
    "TEMPORAL_REGIMES",
    "LAG_ONLY_DENSITY",
    "WINDOW_KIND",
    "COEFFICIENT_KINDS",
    "ORDER_FREE",
    "DEFAULT_REGIME",
    "latents_for",
    "latent_node_types",
    "REGISTRY_NAME",
    "DatasetKey",
    "generation_kwargs",
    "generate_one",
    "lag_only_marks",
    "graph_record",
    "materialize",
    "data_path",
    "load_registry",
    "load",
    "TOPOLOGY_TO_GRAPH",
    "FUNCTIONAL_TO_ANDREY",
    "DISCRETE_FUNCTIONAL",
    "DISCRETE_SETTINGS",
    "MIXED_FUNCTIONAL",
    "MIXED_SETTINGS",
    "NETWORK_FUNCTIONAL",
    "NETWORK_NOISE",
    "network_key",
    "NOISES",
    "STANDARDIZE_TO_SCALE",
]
