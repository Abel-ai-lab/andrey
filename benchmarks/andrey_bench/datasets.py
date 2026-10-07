"""Generate and store benchmark datasets with ``andrey.data.sample_scm``.

Each ``(topology, functional, noise, density, d, n, seed, standardize, latents)`` dataset is stored
once as a ``.npz`` containing ``float64`` samples and the true graph's int8 endpoint marks.
:class:`DatasetKey` supplies the ``dataset_id``; :func:`load` casts to ``float32`` for those fits
outside timing, without storing another copy.
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
from andrey.core import LATENT, OBSERVED

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
    "pnl": "post_nonlinear",
}
STANDARDIZE_TO_SCALE = {
    "standardized": "standardize",
    "raw": "raw",
    "ancestral": "rescale",  # Andrey's "ancestral rescaling" scale (per SCM.sample docstring)
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
}

#: Regimes with hidden confounders, mapped to the number of generated nodes per hidden node. At
#: least one node is hidden. Andrey's ``latent_confounded_er`` family uses the same rule, at ER
#: density 3.0 instead of 2.0.
LATENT_REGIMES: dict[str, int] = {"latent_gauss_er": 10}

DEFAULT_REGIME = "linear_gauss_er"

# The canonical generated edge density: the degree-2 slice, the sparse regime the GES adapters'
# maxP=4 parent cap is sized for. Density travels on the key, so a denser run is a different
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
    per = LATENT_REGIMES.get(regime)
    return 0 if per is None else max(1, d // per)


def latent_node_types(d: int, latents: int) -> np.ndarray | None:
    """Return node types for a truth with `d` nodes and the last `latents` nodes hidden.

    Return `None` when no nodes are hidden. `andrey.data.latent.with_latents` puts hidden nodes
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

    @property
    def label(self) -> str:
        """Human-readable, filesystem-safe key string (also the registry map key).

        Carries ``density``: :func:`materialize` dedupes on the label, so a label blind to
        density would merge a dense key into the sparse key's file.
        """
        hidden = f"-L{self.latents}" if self.latents else ""
        return (
            f"{self.topology}-{self.functional}-{self.noise}-deg{self.density:g}"
            f"-d{self.d}-n{self.n}-s{self.seed}-{self.standardize}{hidden}"
        )

    @property
    def digest(self) -> str:
        """Hash the generator call returned by :func:`generation_kwargs`.

        Only coordinates passed to ``sample_scm`` affect the hash. Sorted JSON keys make the identity
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
    ``ValueError``.
    """
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
    }
    if key.latents:
        kwargs["latents"] = key.latents
    return kwargs


def graph_record(key: DatasetKey) -> dict[str, object]:
    """Return ``{model, num_nodes, density, params}`` in the ``andrey.data`` provenance format.

    ``params_for_density`` maps the key's density to builder parameters. Keys match the sample's
    ``config.graph`` block.
    """
    model = TOPOLOGY_TO_GRAPH[key.topology]
    return {
        "model": model,
        "num_nodes": key.d,
        "density": key.density,
        "params": data.graphs.params_for_density(model, key.d, key.density),
    }


def generate_one(key: DatasetKey) -> tuple[np.ndarray, np.ndarray, str]:
    """Return ``(data_f64, graph_marks_int8, graph_kind)`` generated by ``andrey.data``.

    Samples are contiguous canonical ``float64``. Truth uses ``ds.graph.to_numpy()`` endpoint marks,
    as consumed by the scorer and Andrey adapter. With hidden nodes, the truth covers all ``d``
    nodes. :func:`latent_node_types` rebuilds their types from the key.
    """
    kwargs = generation_kwargs(key)
    ds = data.sample_scm(**kwargs)  # type: ignore[arg-type]
    data_arr = np.ascontiguousarray(ds.data, dtype=CANONICAL_DTYPE)
    graph = ds.graph
    if not hasattr(graph, "to_numpy"):  # Temporal families are unsupported here.
        raise TypeError(
            f"graph for {key.label} has no to_numpy() (kind={type(graph).__name__}); "
            "temporal families are not materialized by this node"
        )
    types = getattr(graph, "node_types", None)
    hidden = np.flatnonzero(types == LATENT) if types is not None else np.arange(0)
    if not np.array_equal(hidden, np.arange(key.d - key.latents, key.d)):
        raise ValueError(f"{key.label}: hidden nodes are not the last {key.latents} nodes")
    marks = np.ascontiguousarray(graph.to_numpy())  # int8 endpoint marks
    return data_arr, marks, str(graph.kind)


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
    readers to reuse. Existing files are reused only when the recorded ``generation_call`` matches.

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
        data_arr, marks, graph_kind = generate_one(key)
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
    "DEFAULT_REGIME",
    "latents_for",
    "latent_node_types",
    "REGISTRY_NAME",
    "DatasetKey",
    "generation_kwargs",
    "generate_one",
    "graph_record",
    "materialize",
    "data_path",
    "load_registry",
    "load",
    "TOPOLOGY_TO_GRAPH",
    "FUNCTIONAL_TO_ANDREY",
    "NOISES",
    "STANDARDIZE_TO_SCALE",
]
