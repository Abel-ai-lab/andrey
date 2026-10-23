---
name: configuration
description: The environment variables Andrey reads, by group, with their values and defaults.
---

# Configuration

Andrey reads a few environment variables, all named `ANDREY_*`. Each has a default, so none needs
to be set. `andrey config` prints the value of each one in your shell and the backends your machine
can use; `andrey config --json` prints the same as JSON.

A blank value counts as unset. A switch takes `1`, `true`, `yes`, or `on`, and `0`, `false`, `no`,
or `off`. Any other value Andrey cannot read raises `ValueError` naming the variable, and
`andrey config` and `andrey run` exit with `invalid_configuration`.

The backend, the worker count, and the tie-break rule can also be set in code. The most specific
setting wins:

1. `with andrey.config(backend=..., num_workers=..., tie_break=...):`, for the calls inside the
   block;
2. `andrey.config.backend = ...`, `andrey.config.num_workers = ...`, or
   `andrey.config.tie_break = ...`, for every later call;
3. the environment variables below, read at each call;
4. the defaults.

## Backend

- `ANDREY_BACKEND`: your preferred implementation for numeric operations. `auto` (the default)
  considers `cuda` when it is available and the input meets the operation's size threshold, subject
  to the startup rule under [GPU dispatch](#gpu-dispatch); otherwise it uses `numpy`. It never
  picks `cpu` or `mps`. `numpy` uses NumPy on the CPU in float64, which we use as the reference.
  `cpu` uses PyTorch on the CPU, `cuda` uses PyTorch on a CUDA-compatible GPU, and `mps` uses
  PyTorch on an Apple GPU in float32, which we have not validated. `none` and `metal` are aliases
  for `cpu` and `mps`. An explicit choice bypasses the size threshold and startup rule. If an
  operation supports your choice but the machine cannot run it, Andrey warns once with
  `andrey.BackendFallbackWarning` and uses `auto` for that operation.
- `ANDREY_DEVICE`: another name for `ANDREY_BACKEND`; if you set both, `ANDREY_BACKEND` takes
  precedence, but both values must be valid.

A GPU speeds up covariance, correlation, entropy, the KCI and CMIknn tests, and the independence
tests of `pnl`. GES, hill climbing, and the other searches run on the CPU. An operation without the
requested backend uses `auto`, without a warning.

## Parallel search

- `ANDREY_NUM_WORKERS`: the worker processes for GES and GFCI, and the worker threads for the
  per-variable regressions of `granger` and the two directions of `pnl`. PC, FCI, GFCI, and CD-NOD
  also use the worker processes for their skeleton search with the `fisherz` and `student_t` tests,
  and send a conditioning-set size to them only when it has enough tests to repay them. `1` (the
  default) and `0` run serially; `-1` uses every usable core. `pnl` with one worker leaves the BLAS
  threads unlimited and with several limits each thread to one, so on a machine with several cores
  its p-values and statistics can differ in the last digits between one worker and several. Each
  `pnl` thread holds three n x n float64 matrices; when the threads you ask for would hold more than
  4 GiB, fewer run, with a `PerformanceWarning`.
- `ANDREY_GES_PARALLEL_MIN_WORK`: a GES pass uses the workers only when it expects to evaluate more
  candidates than this. The default is `10000`; `0` always uses them.
- `ANDREY_SKELETON_PARALLEL_MIN_WORK`: a conditioning-size pass of the skeleton search of PC, FCI,
  GFCI, and CD-NOD uses the workers only when it holds at least this many conditional-independence
  tests. The default is `500000`, measured on an AMD EPYC 9534 server running Linux; `0` always uses
  them.

The cutoffs matter only with more than one worker. `andrey calibrate` measures both for your
machine and prints the values to set; `andrey calibrate --method skeleton` measures the skeleton's
alone. With more than one worker and a default cutoff, a fit warns once when a pass reaches a tenth
of it; setting the variable, to any value, silences the warning.

The workers start from a forkserver, whose socket goes in your temporary directory (`TMPDIR`). If
that directory is on a file system that rejects the socket, as some cluster file systems do, we
start the forkserver with its socket in `/dev/shm`, or `/tmp`, and warn once; if neither works,
the workers start with `fork` or `spawn`. The results do not change. Set `TMPDIR` to a directory on
a local file system to silence the warning.

## GPU dispatch

- `ANDREY_COV_GPU_THRESHOLD`: under `auto`, covariance and correlation run on `cuda` from this many
  elements (rows times columns). The default is `90000`.
- `ANDREY_ENTROPY_GPU_THRESHOLD`: the same for the entropy terms of DirectLiNGAM and the methods
  built on it. The default is `6250`.
- `ANDREY_KCI_GPU_THRESHOLD`: under `auto`, a KCI query with a conditioning set runs on `cuda`, in
  float64, from this many kernel-matrix elements (rows squared). The default is `31250`, about 177
  rows.
- `ANDREY_CMIKNN_GPU_THRESHOLD`: the same for a CMIknn query, from this many distance-matrix
  elements (rows squared). The default is `1024`, 32 rows. Its p-values on `cuda` equal those on
  the CPU.
- `ANDREY_PNL_GPU_THRESHOLD`: the same for the independence tests of `pnl`, in float64, from this
  many kernel-matrix elements (rows squared). The default is `26278`, about 163 rows. A single pair
  uses the GPU only when it is already running, by the rule below.
- `ANDREY_GPU_CALIBRATE`: when on, the first `auto` use of covariance, correlation, or entropy
  measures where `cuda` starts to beat NumPy on this machine, and keeps the result in
  `$XDG_CACHE_HOME/andrey/gpu_calibration.json`, else `~/.cache/andrey/gpu_calibration.json`. Off
  by default. A threshold you set wins.

Under `auto`, an operation that runs once per fit uses the GPU only when PyTorch and the device are
already running in the process, so a CPU fit never pays to start a GPU.
`andrey.core.backend.dispatch_stats()` counts which backend each operation ran on.

## Reproducibility

- `ANDREY_SEED`: the seed of every stochastic method (BOSS, GRaSP, ICA-LiNGAM, CALM) when the call
  gives no `seed=` or `random_state=`. The default is `0`; it takes any integer from `0` to
  `2**32 - 1`. Deterministic methods ignore it.
- `ANDREY_TIE_BREAK`: how the searches choose between tied candidates. `rank` (the default) orders
  them by a rank computed from each column's own data, so reordering the columns relabels the
  result. `position` orders them by column position, which skips computing the ranks, fires the
  orientation rules of `pc`, `cdnod`, `fci`, and `gfci` one at a time in column order, and visits
  the possible-d-separation step of `fci` and `gfci` in column order; the result can then depend on
  the column order.

Results repeat on the same machine; across platforms they can differ, CALM's most of all.
[Column order and machines](guides/column-order.md) lists what each method keeps under a new column
order and on another machine.
`andrey.seed_all(seed)` seeds Python, NumPy, and PyTorch from one number, for data you simulate.

## Datasets

- `ANDREY_DATA_DIR`: where `andrey.data.load_dataset` keeps the files it downloads. The default is
  `$XDG_CACHE_HOME/andrey`, else `~/.cache/andrey`. A `data_home=` argument wins.

## Testing

- `ANDREY_REQUIRE_GPU`: when on, Andrey's own GPU tests fail, instead of skipping, without a CUDA
  or Apple GPU. It does not change a fit.
