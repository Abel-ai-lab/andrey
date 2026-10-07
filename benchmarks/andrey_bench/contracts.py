"""Define benchmark adapters, records, and measurement identities."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from andrey import GraphStructure


class Status(str, Enum):
    """Outcome of a single run."""

    OK = "ok"
    TIMEOUT = "timeout"
    OOM = "oom"
    # The step was never scheduled: nothing about the solution was measured.
    BLOCKED = "blocked"
    ERROR = "error"


#: Environment fields of a ``run_id``, in hash order. Include every ``extra_env`` value that can
#: change a measurement so distinct environments cannot share an identity.
#:
#: ``dtype`` belongs here because reading one dataset as f32 or f64 produces distinct measurements.
RUN_ENV_FIELDS: tuple[str, ...] = (
    "backend",
    "mode",
    "dtype",
    "device",
    "cap_mem_mb",
    "cap_wall_s",
    "warmup",
    "machine_id",
    "threads",
    "num_workers",
)

#: Environment fields appended as ``name=value`` only when set. Unset fields preserve existing
#: digests; ``None`` remains a value for the unconditional fields above.
RUN_ENV_FIELDS_IF_SET: tuple[str, ...] = ("ges_min_work", "hc_min_work")


def _identity_key(value: Any) -> str:
    """Canonical key for one environment field.

    Numbers compare by value and everything else by ``repr``, so ``1800`` and ``1800.0`` are one
    environment while ``None``, ``""`` and ``"None"`` are three.
    """
    if isinstance(value, (int, float)):
        return repr(float(value))
    return repr(value)


def environment_id(env: Mapping[str, Any]) -> str:
    """12-hex digest of the environment half of a ``run_id``.

    :data:`RUN_ENV_FIELDS` always contributes; :data:`RUN_ENV_FIELDS_IF_SET` only where set.
    """
    parts = [_identity_key(env[name]) for name in RUN_ENV_FIELDS]
    parts += [
        f"{name}={_identity_key(env[name])}"
        for name in RUN_ENV_FIELDS_IF_SET
        if env.get(name) is not None
    ]
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:12]


def run_id(*, dataset_id: str, name: str, repeat: int, env: Mapping[str, Any]) -> str:
    """The identity of one measurement: ``<dataset_id>:<solution>:<repeat>:<environment_id>``.

    ``dataset_id`` is :attr:`~andrey_bench.datasets.DatasetKey.digest`.
    """
    return f"{dataset_id}:{name}:{repeat}:{environment_id(env)}"


@runtime_checkable
class SolutionAdapter(Protocol):
    """Wrap one package, method, backend, and mode combination.

    The runner times only ``fit``. Data is already materialized, and ``to_structure`` converts the
    result after timing. Every package uses this protocol.

    One member is optional and so is not declared below: ``python_env``, the *name* of the
    environment variable holding the interpreter this solution's fits run in. Declaring it moves
    only ``fit`` into that environment — ``to_structure`` and scoring stay in the parent — which is
    how a package whose pins conflict with bench-env's still enters the field. Without it, fits run
    in bench-env with everything else.
    """

    name: str  # "andrey.ges.numpy.serial" | "causal-learn.ges" | "gcastle.pc"
    algorithm: str  # ges | pc | fci | lingam | ica_lingam | baseline; explicitly declared
    package: str  # andrey | causal-learn | gcastle | lingam
    package_version: str
    backend: str  # numpy | numba | torch-cpu | torch-cuda | native
    mode: str  # serial | parallel
    output_type: str  # cpdag | dag | pag | order

    def params(self) -> Mapping[str, Any]:
        """Return the search parameters used by this solution.

        Values must be JSON-serializable scalars. The harness passes them to :meth:`fit` and records
        them with the result.
        """
        ...

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the algorithm and return a picklable numpy structural array.

        Competitors return 0/1 adjacency matrices. Andrey adapters return endpoint-mark matrices
        tagged by ``output_type``. This method is the only timed region.
        """
        ...

    def to_structure(self, native: np.ndarray) -> GraphStructure:
        """Convert a native array to an Andrey ``GraphStructure`` outside the timed region."""
        ...


@runtime_checkable
class SetupAdapter(Protocol):
    """Optional per-child initialization outside fit timing.

    Separate from :class:`SolutionAdapter`; adapters without initialization need no setup method.
    The worker looks up `setup` by name and calls it once before warm-up fits, including at
    `--warmup 0`. Its duration is reported as `setup_s` and included in `warmup_s`.
    """

    def setup(self) -> None:
        """Prepare interpreters, libraries, and JITs that `fit` cannot carry through a pickle."""
        ...


@dataclass
class RunRecord:
    """One atomic benchmark run.

    Each repeat carries its dataset coordinates, resource limits, timings, and memory peaks.
    Parent-declared fields such as ``cores`` and ``device`` survive censored runs. Child-observed
    fields such as ``threads``, ``num_workers``, and ``device_id`` remain ``None`` when the child
    does not report them.
    """

    # Record fields
    run_id: str
    dataset_id: str  # DatasetKey.digest: the generator call that made the data
    name: str
    algorithm: str  # ges | pc | fci | lingam | ica_lingam | baseline; independent of the package
    package: str
    package_version: str
    backend: str
    mode: str
    output_type: str  # cpdag | dag | pag | order
    # --- environment, as the parent declared it ---
    machine_id: str
    cores: int  # per-task core budget; split_cores(cores, mode) recovers threads/num_workers
    device: str  # cpu | cuda; the ANDREY_DEVICE request passed to the child
    dtype: str  # how the canonical f64 file was read for this fit
    # --- dataset coordinate, flat: the DatasetKey behind dataset_id ---
    topology: str
    functional: str
    noise: str
    standardize: str
    density: float  # generated edge density, on every topology
    d: int
    n: int
    data_seed: int
    repeat: int
    # --- measures (None when censored / failed) ---
    status: str
    # Dataset coordinate with a default: hidden nodes included in `d` but absent from the data.
    latents: int = 0
    wall_s: float | None = None
    cpu_s: float | None = None  # fit-scoped self+children CPU
    child_cpu_s: float | None = None  # CPU attributed to the (parallel) worker pool alone
    warmup_s: float | None = None  # Setup and warm-up fits, outside timed fit.
    setup_s: float | None = None  # `setup()` duration in `warmup_s`; `None` without setup.
    peak_rss_mb: float | None = None  # child ru_maxrss
    child_peak_rss_mb: float | None = None  # peak RSS of the worker pool alone
    # Peak MiB held in torch's allocator during the timed fit. Excludes the reserved cache,
    # the CUDA context, and library workspaces, so it understates the card's footprint.
    gpu_peak_alloc_mb: float | None = None
    gpu_energy_j: float | None = None  # NVML total-energy counter (~free)
    cap_wall_s: float | None = None
    cap_mem_mb: float | None = None
    # How the memory cap was enforced, and what the parent watched the tree reach under it. The cap
    # is the comparable number and a run_id field. The route and peak are supporting evidence.
    # Under polling, the peak is a lower bound because samples can miss spikes.
    mem_route: str | None = None  # poll | slurm | none
    tree_peak_mem_mb: float | None = None  # parent-observed process-tree peak
    oom_source: str | None = None  # poller | cgroup | external; source of OOM attribution
    # Why a non-ok record is non-ok.
    # Scheduler context as JSON: job, node, partition, CPUs. `job` tells two sittings apart inside
    # one directory after a resume; `run_id` is stable across runs and `machine_id` is hardware.
    scheduler: str = ""
    error_text: str | None = None  # child stderr tail, or the exception that ended the unit
    exit_code: int | None = None  # the child's exit status, when it had one
    warmup: int | None = None  # untimed fits per repeat; parent-declared, and a run_id field
    elapsed_at_kill_s: float | None = None
    ges_min_work: int | None = None  # ANDREY_GES_PARALLEL_MIN_WORK as the child saw it
    hc_min_work: int | None = None  # ANDREY_HC_PARALLEL_MIN_WORK as the child saw it
    output_hash: str | None = None  # of the native encoding; comparable only within a package
    structure_hash: str | None = None  # of the normalized graph; comparable across packages
    # The solution's search parameters as JSON with sorted keys. They are determined by ``name`` and
    # excluded from ``run_id`` until the benchmark allows callers to override them.
    params: str = ""
    # --- per-dataset context (varsortability defense) ---
    varsortability: float | None = None
    r2_sortability: float | None = None
    # --- environment, as the child observed it ---
    device_id: str | None = None  # "cpu" | "cuda:<index>:<name>"; resolved backend device
    threads: int | None = None  # BLAS/numeric pin the child ran under
    num_workers: int | None = None  # Andrey worker-pool size the child ran under
    # --- metrics vs the DGP truth (filled by ``scoring.score_result``; kept open) ---
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------------------
# Shared helpers (the components import these rather than re-implementing them).
# --------------------------------------------------------------------------------------------------

# Andrey endpoint-mark convention (core/structure): mark AT node i on edge i-j lives at marks[i,j].
_NULL, _TAIL, _ARROW, _CIRCLE = 0, 1, 2, 3


def structure_from_adjacency(adj: np.ndarray, *, kind: str) -> GraphStructure:
    """Convert a competitor's directed or undirected adjacency to ``GraphStructure``.

    Andrey graphs are endpoint-mark matrices: a directed edge ``i -> j`` is
    ``marks[i,j]=TAIL, marks[j,i]=ARROW``; a symmetric edge is ``TAIL-TAIL``.
    This helper does that conversion for the DAG/CPDAG case, which covers GES/PC outputs. Adapters
    whose native output carries circle endpoints (FCI PAGs, LiNGAM's own encoding) build the mark
    matrix themselves and call ``GraphStructure.from_numpy`` directly.
    """
    a = np.asarray(adj)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be square 2-D, got {a.shape}")
    # A mark matrix read as an adjacency yields a plausible wrong graph, silently: ARROW and CIRCLE
    # are both nonzero, so every endpoint would come back TAIL.
    if np.issubdtype(a.dtype, np.integer) and np.isin(a, (_ARROW, _CIRCLE)).any():
        raise ValueError(
            "adjacency carries endpoint-mark codes (2=ARROW, 3=CIRCLE); build the mark matrix and "
            "pass it to GraphStructure.from_numpy(..., kind=...) instead"
        )
    a = a != 0
    np.fill_diagonal(a, False)  # no self-loops
    undirected = a & a.T  # symmetric i - j -> TAIL-TAIL
    directed = a & ~a.T  # i -> j -> TAIL at i, ARROW at j
    marks = np.zeros(a.shape, dtype=np.uint8)
    marks[undirected] = _TAIL
    marks[directed] = _TAIL  # tail at the source
    marks[directed.T] = _ARROW  # arrow at the target (marks[j,i] for each i -> j)
    return GraphStructure.from_numpy(marks, kind=kind)  # type: ignore[arg-type]


# BLAS / numeric thread env vars
_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def thread_env(threads: int) -> dict[str, str]:
    """Env dict pinning every BLAS/numeric pool to ``threads`` (1 = the single-core lane).

    Backend/device selection (for example, Andrey's device var) is adapter-specific and layered on
    top by the Andrey adapter. This helper owns only the universally-correct thread pinning.
    """
    return {v: str(threads) for v in _THREAD_VARS}
