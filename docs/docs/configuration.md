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

The backend and the worker count can also be set in code. The most specific setting wins:

1. `with andrey.config(backend=..., num_workers=...):`, for the calls inside the block;
2. `andrey.config.backend = ...` or `andrey.config.num_workers = ...`, for every later call;
3. the environment variables below, read at each call;
4. the defaults.

## Backend

- `ANDREY_BACKEND`: the code a numeric operation runs on. `auto` (the default) uses `cuda` for
  large inputs and `numba` when it is installed, else NumPy; it never picks `cpu` or `mps`. `numpy`
  runs plain NumPy in float64, the reference results. `numba` runs the compiled kernels of the
  `numba` extra. `cpu` runs PyTorch on the CPU, `cuda` on an NVIDIA GPU, and `mps` on an Apple GPU
  in float32, which is not yet validated; `none` and `metal` are other names for `cpu` and `mps`. A
  backend the machine lacks warns once with `andrey.BackendFallbackWarning` and falls back.
- `ANDREY_DEVICE`: another name for `ANDREY_BACKEND`.

A GPU speeds up covariance, correlation, and entropy. GES, hill climbing, and the other searches run
on the CPU. An operation without the requested backend uses its own default, without a warning.

## Parallel search

- `ANDREY_NUM_WORKERS`: the worker processes for GES, GIES, GFCI, and hill climbing. `1` (the
  default) and `0` run serially; `-1` uses every usable core.
- `ANDREY_GES_PARALLEL_MIN_WORK`: a GES pass uses the workers only when it expects to evaluate more
  candidates than this. The default is `10000`; `0` always uses them.
- `ANDREY_HC_PARALLEL_MIN_WORK`: the same for hill climbing, in moves scanned per pass. The default
  is `30000`.

Both cutoffs matter only with more than one worker. `andrey calibrate` measures them for your
machine.

## GPU dispatch

- `ANDREY_COV_GPU_THRESHOLD`: under `auto`, covariance and correlation run on `cuda` from this many
  elements (rows times columns). The default is `90000`.
- `ANDREY_ENTROPY_GPU_THRESHOLD`: the same for the entropy terms of DirectLiNGAM and the methods
  built on it. The default is `6250`.
- `ANDREY_GPU_CALIBRATE`: when on, the first `auto` use of each operation measures where `cuda`
  starts to beat NumPy on this machine, and keeps the result in
  `$XDG_CACHE_HOME/andrey/gpu_calibration.json`, else `~/.cache/andrey/gpu_calibration.json`. Off
  by default. A threshold you set wins.

Under `auto`, an operation that runs once per fit uses the GPU only when PyTorch and the device are
already running in the process, so a CPU fit never pays to start a GPU.
`andrey.core.backend.dispatch_stats()` counts which backend each operation ran on.

## Reproducibility

- `ANDREY_SEED`: the seed of every stochastic method (BOSS, GRaSP, ICA-LiNGAM, CALM) when the call
  gives no `seed=` or `random_state=`. The default is `0`; it takes any integer from `0` to
  `2**32 - 1`. Deterministic methods ignore it.

Results repeat on the same machine; across platforms they can differ, CALM's most of all.
`andrey.seed_all(seed)` seeds Python, NumPy, and PyTorch from one number, for data you simulate.

## Datasets

- `ANDREY_DATA_DIR`: where `andrey.data.load_dataset` keeps the files it downloads. The default is
  `$XDG_CACHE_HOME/andrey`, else `~/.cache/andrey`. A `data_home=` argument wins.

## Testing

- `ANDREY_REQUIRE_GPU`: when on, Andrey's own GPU tests fail, instead of skipping, without a CUDA
  or Apple GPU. It does not change a fit.
