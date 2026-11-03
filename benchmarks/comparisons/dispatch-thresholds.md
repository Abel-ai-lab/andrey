# Measure the dispatch thresholds

Andrey uses hardcoded thresholds for accelerated paths:

- `ANDREY_GES_PARALLEL_MIN_WORK` selects the worker pool.
- `ANDREY_COV_GPU_THRESHOLD`, `ANDREY_ENTROPY_GPU_THRESHOLD`, `ANDREY_KCI_GPU_THRESHOLD`,
  `ANDREY_CMIKNN_GPU_THRESHOLD`, and `ANDREY_PNL_GPU_THRESHOLD` select the accelerator.

Three thresholds were calibrated once on one machine. Later measurements found errors of 7x for GES,
25x for `cov`, and 40x for entropy. Each error kept the faster path disabled.

**The worker-pool default suits a single fit, not a campaign.** A campaign runs many fits in one
process, and after the first pool start the pool can pay off at smaller sizes. `andrey calibrate`
measures the cutoff in the gate's unit, the largest estimated work of a pass. Repeated runs produce
a range because other node activity affects pool round-trip time.

Measure on the campaign node with the pool size used for fits, and **apply the measured
thresholds**. Recording an unused calibration is worse than no calibration: it records a threshold
that did not control the run.

**Check the unit before setting a threshold.** GES uses candidate evaluations, `cov` uses `n * d`,
`entropy` uses batch width `B`, and KCI, CMIknn, and PNL use `n * n`. `andrey.spec.environment`
lists these units, and `andrey --help --json` prints them. A value in the wrong unit silently
selects the wrong path.

**Backend pins affect the gates differently.** Worker-pool gates run on each search pass and ignore
the backend. Below the GES threshold, `ges.numpy.parallel` uses the serial path. Device thresholds
apply only to the `auto` path. `backend.select` gives priority to a pinned backend when it is
supported and available. `AndreyDirectLiNGAM` and `AndreyGES` pin inside `fit`, so
`torch-cuda` DirectLiNGAM uses the GPU at every size. This includes sizes where numpy is faster.
`AndreyPC` and `AndreyFCI` use `ANDREY_DEVICE` and the `auto` path, which may fall back to CPU.
Check the adapter before treating a lane name as the measured device.

Before a campaign:

1. Measure the crossover on the campaign machine. Run `andrey calibrate --workers <pool size>` for
   GES. For a device threshold, compare the primitive with numpy across the sizes used by the
   fits.
2. Set the environment variable and record it in `run_meta.json`. Declare every hyperparameter.
3. Record which side of the threshold contains each rung. Keep one rung where the accelerated path
   is slower.

`ANDREY_GPU_CALIBRATE=1` measures device thresholds at runtime for the `auto` path.
`_resolve_thresholds` calibrates when the active backend is `auto`. A pinned run keeps the static
threshold while recording that calibration was enabled. For pinned campaigns, use the primitive
sweep in step 1. The calibration grid is coarse and starts above some crossovers, so its result is
a bound.
