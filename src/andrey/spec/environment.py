"""Define the machine-readable list of ``ANDREY_*`` environment variables.

Each variable has one :class:`EnvVarSpec`, grouped by backend selection, reproducibility,
determinism, or data. CLI help, configuration reports, and ``SKILL.md`` render these
definitions. Each variable's parser is in :mod:`andrey.core.env`. The human-readable list is
``docs/docs/configuration.md``.
"""

from __future__ import annotations

from andrey.constraint._parallel_skeleton import MIN_POOL_QUERIES as _SKELETON_MIN_WORK
from andrey.constraint._parallel_skeleton import MIN_POOL_RCOT_WORK as _RCOT_MIN_WORK
from andrey.core.env import SEED_MAX
from andrey.search._parallel_ges import _DEFAULT_MIN_WORK as _GES_MIN_WORK

from .models import EnvVarSpec

ENV_VARS: tuple[EnvVarSpec, ...] = (
    EnvVarSpec(
        name="ANDREY_BACKEND",
        values="auto, numpy, cpu/none, cuda, mps/metal",
        default="auto",
        effect=(
            "Compute backend preference. `auto` uses `cuda` above an operation's size threshold, "
            "else numpy; it never selects `cpu` or `mps`. `numpy` forces plain NumPy in float64 "
            "(the reference results); `cpu` runs torch on the CPU; `mps` runs unvalidated float32 "
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
        name="ANDREY_N_JOBS",
        values="integer >= -1 (-1 = all usable cores)",
        default="1",
        effect=(
            "Worker processes for GES, GFCI (its GES phase), and the skeleton searches of PC, "
            "FCI, GFCI, and CD-NOD with `fisherz`, `student_t`, or `rcot`, and worker threads for "
            "the regressions of `granger` and the two directions of `pnl`; `0` or `1` runs "
            "serially. "
            "Separate from `ANDREY_BACKEND`, which picks the numeric code."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_NUM_WORKERS",
        values="same as ANDREY_N_JOBS",
        default="ANDREY_N_JOBS",
        effect=(
            "Former name of `ANDREY_N_JOBS`, read with a `DeprecatedArgumentWarning` when "
            "`ANDREY_N_JOBS` is unset; it stops being read in the next release."
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
            "unless the worker count (`n_jobs` / `ANDREY_N_JOBS`) is above 1. A fit with "
            "more than one worker warns once on the default when a pass reaches a tenth of it; "
            "`andrey calibrate` measures a value for a process that runs many fits."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_SKELETON_PARALLEL_MIN_WORK",
        values="integer (CI tests per conditioning-size pass)",
        default=str(_SKELETON_MIN_WORK),
        effect=(
            "Per-pass crossover gate for the parallel skeleton search of PC, FCI, GFCI, and "
            "CD-NOD with `fisherz` or `student_t`: a conditioning-size pass fans out to the "
            "worker pool only when it holds at least this many CI tests, else in-process. `0` "
            "sends every pass to the pool. The default was measured on an AMD EPYC 9534 server "
            "running Linux. No effect unless the worker count (`n_jobs` / "
            "`ANDREY_N_JOBS`) is above 1. A fit with more than one worker warns once on the "
            "default when a pass reaches a tenth of it; `andrey calibrate --method skeleton` "
            "measures a value for your machine."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_RCOT_PARALLEL_MIN_WORK",
        values="integer (RCoT tests times rows per conditioning-size pass)",
        default=str(_RCOT_MIN_WORK),
        effect=(
            "Per-pass crossover gate for the parallel skeleton search of PC, FCI, GFCI, and "
            "CD-NOD with `rcot`: a conditioning-size pass fans out to the worker pool only when "
            "its RCoT tests times the data's rows reach this value, else in-process. `0` sends "
            "every pass to the pool. The default was measured on an Intel Core i7-7820X running "
            "Linux with 4 workers. No effect unless the worker count (`n_jobs` / "
            "`ANDREY_N_JOBS`) is above 1. A fit with more than one worker warns once on the "
            "default when a pass reaches a tenth of it; `andrey calibrate --method rcot` measures "
            "a value for your machine."
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
        name="ANDREY_KCI_GPU_THRESHOLD",
        values="integer (elements: rows squared)",
        default="cuda 31250, mps never",
        effect=(
            "Smallest kernel matrix (rows squared) at which `auto` runs a KCI query with a "
            "conditioning set on a CUDA GPU, in float64. A set value replaces the `cuda` default; "
            "KCI never runs on `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_CMIKNN_GPU_THRESHOLD",
        values="integer (elements: rows squared)",
        default="cuda 1024, mps never",
        effect=(
            "Smallest distance matrix (rows squared) at which `auto` runs a CMIknn query on a CUDA "
            "GPU. A set value replaces the `cuda` default; CMIknn never runs on `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_PNL_GPU_THRESHOLD",
        values="integer (elements: rows squared)",
        default="cuda 26278, mps never",
        effect=(
            "Smallest kernel matrix (rows squared) at which `auto` runs the HSIC tests of PNL's "
            "pairs on a CUDA GPU, in float64. A set value replaces the `cuda` default; PNL never "
            "runs on `mps`."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_GPU_CALIBRATE",
        values="1/true/yes/on or 0/false/no/off",
        default="off",
        effect=(
            "Opt-in auto-tuning for `cuda`. The first `auto` use of covariance, correlation, or "
            "entropy measures the local numpy-to-`cuda` crossover, caches it by operation and "
            "machine fingerprint, and logs the cutoff at INFO. A manual `*_GPU_THRESHOLD` takes "
            "precedence. `auto` never selects `mps`, so calibration never measures it. KCI and "
            "CMIknn are not calibrated."
        ),
        group="backend",
    ),
    EnvVarSpec(
        name="ANDREY_SEED",
        values=f"integer from 0 to {SEED_MAX} (2**32 - 1)",
        default="0",
        effect=(
            "Global default seed for every method that draws random numbers (BOSS, GRaSP, "
            "ICA-LiNGAM, CALM, and the searches that test independence, with the rcot, cmiknn, and "
            "gpdc tests). An explicit `seed=` overrides it. Deterministic methods (GES, "
            "DirectLiNGAM, ...) and the other tests ignore it."
        ),
        group="reproducibility",
    ),
    EnvVarSpec(
        name="ANDREY_TIE_BREAK",
        values="rank, position",
        default="rank",
        effect=(
            "How the searches order tied candidates. `rank` orders them by a rank computed from "
            "each column's own data, so reordering the columns relabels the result. `position` "
            "orders them by column position, which skips computing the ranks, fires the "
            "orientation rules of `pc`, `cdnod`, `fci`, and `gfci` one at a time in column order, "
            "and visits the possible-d-separation step of `fci` and `gfci` in column order; "
            "the result can then depend on the column order. Also `andrey.config(tie_break=...)`."
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
    "Seed: explicit `seed=` argument > `ANDREY_SEED` > `0`.",
    "Tie break: `with andrey.config(tie_break=...)` > `andrey.config.tie_break = ...` > "
    "`ANDREY_TIE_BREAK` > default (`rank`).",
    "GPU threshold: manual `ANDREY_*_GPU_THRESHOLD` > `ANDREY_GPU_CALIBRATE` measurement > "
    "per-device default.",
)
