"""Define the machine-readable list of ``ANDREY_*`` environment variables.

Each variable has one :class:`EnvVarSpec`, grouped by backend selection, reproducibility,
determinism, or data. CLI help, configuration reports, ``llms.txt``, and ``SKILL.md`` render these
definitions. Each variable's parser is in :mod:`andrey.core.env`. The human-readable list is
``docs/docs/configuration.md``.
"""

from __future__ import annotations

from andrey.core.env import SEED_MAX
from andrey.search._parallel_ges import _DEFAULT_MIN_WORK as _GES_MIN_WORK
from andrey.search._parallel_hc import _DEFAULT_MIN_WORK as _HC_MIN_WORK

from .models import EnvVarSpec

ENV_VARS: tuple[EnvVarSpec, ...] = (
    EnvVarSpec(
        name="ANDREY_BACKEND",
        values="auto, numpy, numba, cpu/none, cuda, mps/metal",
        default="auto",
        effect=(
            "Compute backend preference. `auto` uses `cuda` above an operation's size threshold "
            "and `numba` when installed, else numpy; it never selects `cpu` or `mps`. `numpy` "
            "forces plain NumPy in float64 (the reference results); `numba` the JIT kernels "
            "(`[numba]` extra); `cpu` runs torch on the CPU; `mps` runs unvalidated float32 "
            "kernels on Apple Metal. An unavailable backend warns once and falls back. "
            "`ANDREY_DEVICE` is an alias."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_DEVICE",
        values="same as ANDREY_BACKEND",
        default="ANDREY_BACKEND",
        effect="Alias for `ANDREY_BACKEND`; set either one.",
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_NUM_WORKERS",
        values="integer >= -1 (-1 = all usable cores)",
        default="1",
        effect=(
            "Worker processes for GES, GIES, GFCI (its GES phase), and HC; `0` or `1` runs "
            "serially. Separate from `ANDREY_BACKEND`, which picks the numeric code."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_GES_PARALLEL_MIN_WORK",
        values="integer (estimated candidate evaluations)",
        default=str(_GES_MIN_WORK),
        effect=(
            "Per-pass crossover gate for parallel GES: a greedy pass fans out to the worker pool "
            "only when its estimated candidate-evaluation count (density-aware, ~2^degree) exceeds "
            "this, else in-process. `0` sends every pass to the pool. No effect "
            "unless the worker count (`num_workers` / `ANDREY_NUM_WORKERS`) is above 1. A fit with "
            "more than one worker warns once on the default when a pass reaches a tenth of it; "
            "`andrey calibrate` measures a value for a process that runs many fits."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_HC_PARALLEL_MIN_WORK",
        values="integer (estimated moves scanned per pass)",
        default=str(_HC_MIN_WORK),
        effect=(
            "Crossover gate for parallel Hill-Climbing: a run fans out to the worker pool only "
            "when its per-pass move-scan estimate (~d^2) exceeds this, else in-process. `0` always "
            "uses the workers. No effect unless the worker count "
            "(`num_workers` / `ANDREY_NUM_WORKERS`) is above 1. A fit with more than one worker "
            "warns once on the default when a pass reaches a tenth of it; `andrey calibrate` "
            "measures a value for a process that runs many fits."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_COV_GPU_THRESHOLD",
        values="integer (elements: rows times columns)",
        default="cuda 90000, mps never",
        effect=(
            "Smallest data size (rows times columns) at which `auto` computes covariance and "
            "correlation matrices, used by the CI tests and BIC scores, on a CUDA GPU. A set value "
            "replaces the `cuda` default; it never enables `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_ENTROPY_GPU_THRESHOLD",
        values="integer (elements: rows times the columns scored at once)",
        default="cuda 6250, mps never",
        effect=(
            "Smallest input size at which `auto` computes the entropy terms of DirectLiNGAM-based "
            "methods on a CUDA GPU. A set value replaces the `cuda` default; it never enables "
            "`mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_GPU_CALIBRATE",
        values="1/true/yes/on or 0/false/no/off",
        default="off",
        effect=(
            "Opt-in auto-tuning for `cuda`. The first `auto` use of each operation measures the "
            "local numpy-to-`cuda` crossover, caches it by operation and machine fingerprint, and "
            "logs the cutoff at INFO. A manual `*_GPU_THRESHOLD` takes precedence. `auto` never "
            "selects `mps`, so calibration never measures it."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_SEED",
        values=f"integer from 0 to {SEED_MAX} (2**32 - 1)",
        default="0",
        effect=(
            "Global default seed for every stochastic method (BOSS, GRaSP, ICA-LiNGAM, CALM). An "
            "explicit `seed=` / `random_state=` overrides it. Deterministic methods (PC, GES, FCI, "
            "DirectLiNGAM, ...) ignore it."
        ),
        group="reproducibility",
    ),
    EnvVarSpec(
        name="ANDREY_REQUIRE_GPU",
        values="1/true/yes/on or 0/false/no/off",
        default="off",
        effect=(
            "For the package's own test suite: its GPU tests fail, instead of skipping, when no "
            "CUDA or MPS device is present. No effect on a fit."
        ),
        group="determinism",
    ),
    EnvVarSpec(
        name="ANDREY_DATA_DIR",
        values="a directory path (`~` expands)",
        default="$XDG_CACHE_HOME/andrey, else ~/.cache/andrey",
        effect=(
            "Where `andrey.data.load_dataset` keeps downloaded datasets. A `data_home=` argument "
            "overrides it."
        ),
        group="data",
    ),
)

# Which setting wins where several apply, most specific first.
PRECEDENCE: tuple[str, ...] = (
    "Backend: `with andrey.config(backend=...)` > `andrey.config.backend = ...` > "
    "`ANDREY_BACKEND`/`ANDREY_DEVICE` > default (`auto`).",
    "Seed: explicit `seed=`/`random_state=` argument > `ANDREY_SEED` > `0`.",
    "GPU threshold: manual `ANDREY_*_GPU_THRESHOLD` > `ANDREY_GPU_CALIBRATE` measurement > "
    "per-device default.",
)
