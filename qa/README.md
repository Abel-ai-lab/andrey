# QA tools

On-demand tools check Andrey against itself or known ground truth. They run by hand on any machine
with the required package extras. Correctness tests and recorded outputs live in `tests/`;
comparisons with other packages and older Andrey builds live in `benchmarks/`.

| Tool | Purpose |
|---|---|
| `quality_gate.py` | Pre-publish gate for the supported methods. |
| `capture_baselines.py` | Record Andrey outputs in `tests/recovery/baselines/`. |
| `column_order.py` | Count how often a column relabeling changes FCI's and GFCI's marks. |
| `gpu_calibration_validate.py` | Check accelerator crossovers, calibration cache reuse, and dispatch. |

## FCI column order

FCI and GFCI depend on column order. `column_order.py` measures how much, for the column-order
sentence on the benchmarks page. It fits both methods on 237 latent-confounded models
(Erdos-Renyi and scale-free graphs with one hidden confounder, 6-12 nodes, 500-5000 samples, seeds
0-9), refits each on three seeded column orders, maps the answers back, and counts the relabelings
that change any endpoint mark. It is a measurement with no passing threshold: permutation equality
is not a requirement for these methods.

```shell
uv run python -m qa.column_order
```

## Quality gate

One command checks the current commit before a release, and exits 1 if any check fails:

```shell
uv run python -m qa.quality_gate
```

The `quality-gate` workflow runs the same command. Every job that publishes to PyPI must need a job
that calls it; `tests/unit/test_release_gates.py` fails otherwise. Pull requests do not run the
gate. It checks PC, FCI, GES, BOSS, GRaSP, DirectLiNGAM, and ICA-LiNGAM on the datasets whose
assumptions each method needs:

| Dataset | Draws | Methods |
|---|---|---|
| `linear_gauss_er` at `d = 20, 50` | 10 seeds each | PC, GES, BOSS, GRaSP |
| `latent_gauss_er` at `d = 20, 50` | 10 seeds each | FCI |
| `lingam_sf` at `d = 20, 50` | 10 seeds each | DirectLiNGAM, ICA-LiNGAM |
| ASIA, linear-Gaussian | 10 seeds | PC, GES, BOSS, GRaSP, FCI |
| ASIA, linear with uniform noise | 10 seeds | DirectLiNGAM, ICA-LiNGAM |
| Sachs, measured | 1 | all seven |

- **Regimes.** `andrey.data` generates the published benchmark regimes with the benchmarks'
  arguments, at `n = 50d`; a test keeps the two in step. At the benchmarks' `n = 10d`, ICA-LiNGAM
  scores worse than the empty graph, so the check could not tell a broken build from a correct one.
- **ASIA.** The 8-arc network of Lauritzen and Spiegelhalter (1988), sampled as a linear SCM with
  1,000 rows per seed. Its arcs are bnlearn's (`andrey.data.load_dataset("asia")`); the gate lists
  them in its own order, which fixes each seed's weights, and a test checks the two agree.
- **Sachs.** The 853 observational rows of Sachs et al. (2005), scored against the 17-arc consensus
  network, from `andrey.data.load_dataset("sachs")`. It downloads
  [`sachs.2005.continuous.txt`](https://github.com/cmu-phil/example-causal-datasets/blob/4ba0565b8164b46063daf0f6fba6b94e5a99baf6/real/sachs/data/sachs.2005.continuous.txt)
  from `cmu-phil/example-causal-datasets` (CC0-1.0) at a pinned commit once, checks its SHA-256,
  and caches it, so the first run needs network access.

SHD follows the published summary: PC, GES, BOSS, and GRaSP against the truth's CPDAG, the LiNGAM
methods against the DAG, and FCI by endpoint SHD against the PAG of the observed variables. On each
dataset, a method passes when its median SHD over the draws is:

- below the empty graph's median SHD;
- within the method's tolerance of its baseline, in either direction.

`quality_baselines.json` holds the baselines, and in its `tolerances` block every tolerance with the
reason for its size. A median better than the baseline by more than the tolerance also fails, so an
intended change is recorded by a recapture. Every fit runs on the numpy backend with one worker and
one BLAS thread, so the host's accelerators and core count do not enter the result; the tolerances
absorb rounding differences between CPUs. On every pull request,
`tests/recovery/test_quality_gate.py` feeds each method the empty graph and a column-shifted true
graph, and the gate rejects both.

```shell
uv run python -m qa.quality_gate PC FCI
uv run python -m qa.quality_gate --capture  # re-measure; review the diff
```

## Other tools

```shell
uv run python -m qa.capture_baselines --out-dir outputs/baseline-review
uv run --no-sync python qa/gpu_calibration_validate.py
```

The GPU check needs a driver-matched torch installation; see
[PyTorch for a GPU](../docs/docs/guides/getting-started.md#pytorch-for-a-gpu). `uv run --no-sync`
keeps that installation; plain `uv run` re-syncs the locked CPU build. Calibration results depend
on the machine and worker count.
