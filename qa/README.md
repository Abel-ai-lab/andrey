# QA tools

On-demand tools check Andrey against itself or known ground truth. They run by hand on any machine
with the required package extras. Correctness tests and recorded outputs live in `tests/`;
comparisons with other packages and older Andrey builds live in `benchmarks/`.

| Tool | Purpose |
|---|---|
| `quality_gate.py` | Pre-publish gate for the supported methods. |
| `reference_parity.py` | Fail a published row where the reference package has fewer errors. |
| `capture_baselines.py` | Record Andrey outputs in `tests/recovery/baselines/`. |
| `column_order.py` | Count how often a column relabeling changes FCI's and GFCI's marks. |
| `gpu_calibration_validate.py` | Check accelerator crossovers, calibration cache reuse, and dispatch. |

## FCI column order

FCI and GFCI depend on column order. `column_order.py` measures how much, for the column-order
sentence in the 0.1.0 benchmark report. It fits both methods on 237 latent-confounded models
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
gate. It checks PC, FCI, GES, XGES, HC, Tabu, BOSS, GRaSP, DirectLiNGAM, ICA-LiNGAM, RCD,
BottomUpParceLiNGAM, ANM, PNL, PCMCI, PCMCI+, LPCMCI, VARMA-LiNGAM, LongitudinalLiNGAM, and Granger
on the datasets whose assumptions each method needs:

| Dataset | Draws | Methods |
|---|---|---|
| `linear_gauss_er` at `d = 20, 50` | 10 seeds each | PC, GES, XGES, HC, Tabu, BOSS, GRaSP |
| `latent_gauss_er` at `d = 20, 50` | 10 seeds each | FCI |
| `lingam_sf` at `d = 20, 50` | 10 seeds each | DirectLiNGAM, ICA-LiNGAM |
| `latent_lingam_er` at `d = 10, 15` | 10 seeds each | RCD, BottomUpParceLiNGAM |
| `anm_pair` at `n = 1000` | 10 seeds | ANM |
| `pnl_pair` at `n = 1000` | 10 seeds | PNL |
| `pcmci_paper` at `d = 10`, `n = 500, 1000` | 10 seeds each | PCMCI |
| `pcmciplus_paper` at `(d, n) = (10, 500), (20, 1000)` | 10 seeds each | PCMCI+ |
| `lpcmci_paper` at `(d, n) = (5, 500), (10, 1000)` | 10 seeds each | LPCMCI |
| `varma_lingam_paper` at `d = 3, 5`, `n = 300` | 10 seeds each | VARMA-LiNGAM |
| `longitudinal_lingam_paper` at `(d, n, m) = (3, 20, 200), (5, 20, 500)` | 10 seeds each | LongitudinalLiNGAM |
| `var2_lagged` at `d = 5`, `n = 1000, 2000` | 10 seeds each | Granger |
| ASIA, linear-Gaussian | 10 seeds | PC, GES, XGES, HC, Tabu, BOSS, GRaSP, FCI |
| ASIA, linear with uniform noise | 10 seeds | DirectLiNGAM, ICA-LiNGAM |
| Sachs, measured | 1 | PC, FCI, GES, XGES, HC, Tabu, BOSS, GRaSP, DirectLiNGAM, ICA-LiNGAM |

- **Regimes.** `andrey.data` generates the published benchmark regimes with the benchmarks'
  arguments, at `n = 50d`; a test keeps the two in step. At the benchmarks' `n = 10d`, ICA-LiNGAM
  scores worse than the empty graph, so the check could not tell a broken build from a correct one.
  `latent_lingam_er` keeps its benchmark's 2,000 rows, with one node in five hidden; `d` counts
  the hidden nodes. From `d = 20`, BottomUpParceLiNGAM's median SHD is no lower than the empty
  graph's, and on ASIA's 1,000 rows RCD's is higher, so RCD and BottomUpParceLiNGAM run on this
  regime only.
- **Pairs.** `anm_pair` and `pnl_pair` are the published cause-effect pairs: two variables and one
  edge, in a direction the seed draws, at the benchmarks' 1,000-row size; a test keeps them in step
  with the benchmark drivers. The empty graph's SHD on a pair is 1, so ANM and PNL pass only when
  they name the true direction on most draws.
- **Time series.** The published time-series regimes, drawn by the benchmark harness
  (`benchmarks/andrey_bench`), which the gate imports; each method runs through its benchmark
  adapter with the published settings. The sizes are published sizes at which the method's median
  error is below the empty graph's: PCMCI's only at `d = 10`, and VARMA-LiNGAM's the two smallest,
  as its fit takes seconds. `longitudinal_lingam_paper` draws `m` units at `n` time points.
  Granger's published median error is above the empty graph's at every published size, so its
  gate draws 5 variables, where it is below. A test keeps the Granger draws in step with the
  benchmark driver and the regime names with the harness.
- **ASIA.** The 8-arc network of Lauritzen and Spiegelhalter (1988), sampled as a linear SCM with
  1,000 rows per seed. Its arcs are bnlearn's (`andrey.data.load_dataset("asia")`); the gate lists
  them in its own order, which fixes each seed's weights, and a test checks the two agree.
- **Sachs.** The 853 observational rows of Sachs et al. (2005), scored against the 17-arc consensus
  network, from `andrey.data.load_dataset("sachs")`. It downloads
  [`sachs.2005.continuous.txt`](https://github.com/cmu-phil/example-causal-datasets/blob/4ba0565b8164b46063daf0f6fba6b94e5a99baf6/real/sachs/data/sachs.2005.continuous.txt)
  from `cmu-phil/example-causal-datasets` (CC0-1.0) at a pinned commit once, checks its SHA-256,
  and caches it, so the first run needs network access.

SHD follows the published summaries: PC, GES, XGES, HC, Tabu, BOSS, and GRaSP against the truth's
CPDAG, the LiNGAM methods, ANM, and PNL against the DAG, and FCI by endpoint SHD against the PAG of
the observed variables. RCD and BottomUpParceLiNGAM are scored by endpoint SHD against the latent
projection, the edges and pairs their benchmark scores: `i -> j` for a directed path from `i` to `j`
through hidden nodes only, and `i <-> j` for a pair that one hidden node reaches through hidden
nodes only. The time-series methods are scored in their summaries' units: PCMCI by the lag-resolved
SHD, PCMCI+ by endpoint SHD over the window of lags, LPCMCI by endpoint SHD against the true PAG of
the observed series, VARMA-LiNGAM and LongitudinalLiNGAM by the missing and extra cross-links, and
Granger by the missing and extra lagged edges. On each dataset, a method passes when its median SHD
over the draws is:

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

## Reference parity

Before any fit, the gate checks the published benchmark summaries
(`benchmarks/published/*/summary.json`) with `reference_parity.py`, which reads only those files,
so the reference packages need not be installed. Each row pairs Andrey with one reference package at one size, and it
fails at any margin:

- where the row counts the series on which each side has fewer errors, when the reference has fewer
  errors on more series than Andrey;
- otherwise, when Andrey's errors (the row's mean or median SHD, or directed errors) are higher.

A row where either side has no result is skipped. A released summary never changes, so a method
re-measured against the same reference in a later summary is judged by the later rows only.

`benchmarks/published/parity-allowlist.json` exempts a failing row only when it names the cause:
`correct-by-paper` (Andrey follows the method's paper and the reference does not) or
`reference-defect`, with a reason and a citation. Each entry pins the row's two error values (its
two series counts, where the row has them), so a re-run that changes the row voids it, and an
entry that matches no failing row fails the check.

```shell
uv run python -m qa.reference_parity  # exit 1 if any row fails
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
