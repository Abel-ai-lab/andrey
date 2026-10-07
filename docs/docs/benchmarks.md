---
name: benchmarks
description: Andrey benchmark timings, accuracy, and practical limits by method family.
meta:
  type: guide
  tags: [benchmarks, performance, accuracy]
---

# Benchmarks

**Over 100x faster on PC, and faster on most other supported methods.**
The [summary](#summary) gives Andrey's speedup and, separately, its accuracy for each comparison
with causal-learn, gCastle, and lingam, including where Andrey is not faster. These results
describe the measured workloads; speed and accuracy depend on the data, settings, and hardware. The
[recorded campaigns](#recorded-campaigns) list the measurement dates and package versions and link
the summary data. [Public datasets](#public-datasets) compares the packages on Sachs's measured data
and on four published network structures, separately from the simulated campaigns.

## Summary

causal-learn is the baseline. The tables also report gCastle and lingam. Speed and accuracy are
reported separately: every paired speed ratio is given, whatever the accuracy.

**Speed.**

| Family | Method | Compared with | Variables (`d`) | Andrey speedup |
|---|---|---|---|---|
| Constraint-based | PC | causal-learn | 20-400 | 15-941x |
| Constraint-based | PC | gCastle | 20-400 | 12-254x |
| Constraint-based | FCI | causal-learn | 20-800 | 3.4-16.5x at `d = 50` to 800; 0.62x at `d = 20` |
| Score-based | GES | causal-learn | 20-150 | 2.9-4.3x |
| Permutation-based | BOSS | causal-learn | 20-800 | 3.2-5.7x |
| Permutation-based | GRaSP | causal-learn | 20-400 | 1.4-2.1x |
| Linear non-Gaussian | DirectLiNGAM | causal-learn | 20-200 | 3.3-6.9x |
| Linear non-Gaussian | DirectLiNGAM | lingam | 20-100 | 3.6-5.7x on CPU |
| Linear non-Gaussian | ICA-LiNGAM | causal-learn | 20-100 | 0.98-1.01x |

**Accuracy.**

| Method | Compared with | Accuracy |
|---|---|---|
| PC | causal-learn | Identical SHD at every measured size |
| PC | gCastle | Andrey's SHD is lower from `d = 50` |
| FCI | causal-learn | Same PAG on 13 of 18 datasets; median endpoint SHD within two |
| GES | causal-learn | Identical CPDAG on 12 of 12 datasets |
| BOSS | causal-learn | Equal or lower SHD at every size |
| GRaSP | causal-learn | Same graph on 14 of 15 datasets |
| DirectLiNGAM | causal-learn | Same graph on 12 of 12 datasets |
| DirectLiNGAM | lingam | Mean SHD within one edge |
| ICA-LiNGAM | causal-learn | Same graph on 9 of 9 datasets |

The runs whose graphs are no better than the empty graph are listed under
[Large graphs](#large-graphs).

## How to read the results

**Data.** PC, GES, BOSS, and GRaSP use `linear_gauss_er`: Erdos-Renyi DAGs with mean degree 2,
linear-Gaussian mechanisms, standardized data, and `n = 10d` samples. Here `d` is the number of
variables. FCI uses `latent_gauss_er`, the same graphs with one variable in ten hidden; `d` includes
the hidden variables. DirectLiNGAM and ICA-LiNGAM use `lingam_sf`: scale-free DAGs with linear
mechanisms and uniform noise. There are three seeds per size unless noted otherwise.

**Accuracy.** `andrey.metrics.score` compares each estimate against generated ground truth.
Structural Hamming distance (SHD) counts graph errors; lower is better. PC, GES, BOSS, and GRaSP
are scored against the truth's CPDAG, and DirectLiNGAM and ICA-LiNGAM against the truth's DAG. FCI
is scored against the PAG of the observed variables by endpoint SHD, which counts each mismatched
tail, arrowhead, or circle. External outputs are converted to Andrey endpoint marks before scoring.
"Same graph" means equal `structure_hash` values. The DirectLiNGAM comparison with lingam uses
`output_hash`; its adapters share the same binary adjacency encoding. PC compares SHD; its records
have no cross-package graph hash.

The empty graph's SHD is the number of true edges, and its endpoint SHD is twice the number of true
adjacencies. An estimate with higher SHD has more graph errors than an estimate returning no edges.
Accuracy does not gate the speed ratios: each comparison reports its paired ratio and, beside it,
each package's SHD.

**Timing.** Times are seconds per fit. Each fit runs in its own process after imports. PC, GES,
BOSS, GRaSP, and the DirectLiNGAM comparison with lingam run a warm-up fit first; FCI and the
LiNGAM comparisons with causal-learn do not. Wall caps include setup, any warm-up, and the timed
fit; memory caps cover the process tree. Each speed ratio pairs solutions on the same dataset in the
same run, then takes the median over datasets. Ratios therefore need not equal ratios of the
displayed median times. For an even number of datasets, the median averages the two middle ratios.

**Hardware.** PC, GES, BOSS, GRaSP, FCI, and the LiNGAM comparisons with causal-learn used
exclusive CPU nodes. FCI and the LiNGAM comparisons with causal-learn ran each size on a single
node, using two CPU models across sizes, so their times are comparable only within a size. The
DirectLiNGAM comparison with lingam reserved 16 of 48 host cores and one GPU on shared nodes, so
its timings include possible contention. Ratios apply within a campaign and do not transfer between
machines.

| Comparison | Hardware | Cores per fit |
|---|---|---:|
| PC, BOSS, GRaSP | AMD EPYC 7713 | 16 |
| GES | AMD EPYC 7713 | 128 |
| FCI | AMD EPYC 9534; AMD EPYC 7713 at `d = 200` | 16 |
| DirectLiNGAM and ICA-LiNGAM compared with causal-learn | AMD EPYC 7713; AMD EPYC 9534 at `d = 100` | 16 |
| DirectLiNGAM compared with lingam | NVIDIA A100-SXM4-80GB; AMD EPYC 7413 host | 16 |

## Recorded campaigns

The measurements were captured from 21 August to 15 September 2026 with pre-release Andrey builds
installed from source.

| Comparison | d | Date | Measured package versions |
|---|---|---|---|
| PC | 20-400 | 21 August | causal-learn 0.1.4.8; gCastle 1.0.4 |
| PC, additional Andrey timings | 400 | 21 August | Andrey only |
| FCI | 20-800 | 15 September | causal-learn 0.1.4.8 |
| GES | 20, 50 | 31 August | causal-learn 0.1.4.8 |
| GES and penalty comparison | 100-200 | 31 August | causal-learn 0.1.4.8 through 150 |
| BOSS, GRaSP | 20-200 | 31 August | causal-learn 0.1.4.8 |
| BOSS, GRaSP | 400 | 1 September | causal-learn 0.1.4.8 |
| BOSS | 800 | 2 September | causal-learn 0.1.4.8 |
| DirectLiNGAM compared with lingam, and quality check | 20-400 | 21 August | lingam 1.12.2 |
| DirectLiNGAM and ICA-LiNGAM compared with causal-learn | 20-200; ICA-LiNGAM 20-100 | 15 September | causal-learn 0.1.4.8 |

The {download}`summary <../../benchmarks/published/alpha-2026-09/summary.json>` gives one row per
comparison and size: median seconds, the paired ratio, and each package's SHD alongside the empty
graph's. Its seconds are each package's median over every run at that size, so they can pool runs,
while the ratio pairs datasets within a run; the ratio therefore need not equal the ratio of the
two times. `seeds` counts the paired datasets or, where there are none, the datasets measured. SHD
is the median over datasets, except the mean for the DirectLiNGAM comparison with lingam; FCI's is
endpoint SHD.

The headline's PC figure, 941x at `d = 400`, was measured on 21 August. The three
dataset ratios are 801.20x, 941.05x, and 955.73x. Each divides `causal-learn.pc`'s median fit
time by `andrey.pc.numpy`'s median fit time on that dataset. Their median is 941.05x.

## Constraint-based methods

### PC

PC runs conditional-independence tests to find a skeleton, then orients edges and returns a
CPDAG. Andrey batches the tests and screens first-order removals before the general loop.

| d | Andrey | causal-learn | gCastle | vs causal-learn | vs gCastle |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.0076 | 0.098 | 0.093 | 15x | 12x |
| 50 | 0.024 | 0.75 | 0.68 | 31x | 29x |
| 100 | 0.061 | 5.01 | 2.80 | 79x | 45x |
| 200 | 0.22 | 50.8 | 16.2 | 219x | 70x |
| 400 | 0.82 | 617 | 185 | 941x | 254x |

Andrey's `d = 400` time combines two runs on two machines; speedups use only within-run pairs. The
gCastle comparison at `d = 400` pairs two datasets.

| d | Andrey SHD | causal-learn SHD | gCastle SHD | Empty-graph SHD |
|---:|---:|---:|---:|---:|
| 20 | 12 | 12 | 12 | 20 |
| 50 | 20 | 20 | 31 | 50 |
| 100 | 54 | 54 | 88 | 100 |
| 200 | 133 | 133 | 256 | 200 |
| 400 | 357 | 357 | 792.5 | 400 |

Andrey and causal-learn have identical SHD at every size.

### FCI

FCI allows hidden confounders and returns a PAG. It uses a skeleton search, a possible-d-separation
pass, and ten orientation rules. Both packages use Fisher-z tests at `alpha = 0.05`. Accuracy is
endpoint SHD.

| d | Andrey | causal-learn | Speedup | Andrey SHD | causal-learn SHD | Empty-graph SHD |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.061 | 0.038 | 0.62x | 29 | 29 | 48 |
| 50 | 0.089 | 0.29 | 3.4x | 42 | 40 | 96 |
| 100 | 0.17 | 1.42 | 7.5x | 135 | 134 | 216 |
| 200 | 0.86 | 12.3 | 16.5x | 391 | 391 | 492 |
| 400 | 3.56 | 31.5 | 9.6x | 836 | 836 | 884 |
| 800 | 72.2 | 620 | 8.7x | 2,579 | 2,579 | 1,772 |

The two packages return the same PAG on 13 of 18 datasets, and their median SHD differs by at most
two endpoints (42 compared with 40 at `d = 50`).
Andrey is slower at `d = 20`, where both packages' median times are under 0.1 s. Its time varies
because one dataset's first repeat took 0.20 s against 0.06 s for its other two. The campaign has
no warm-up fit. The ratio pairs per-dataset medians, so that repeat does not change it.

## Score-based methods

GES searches graphs one edge at a time using linear-Gaussian BIC.

### GES

The GES comparison uses Andrey's NumPy serial search with `lambda_value = 1.0`, the textbook BIC.
The recorded causal-learn penalty is `0.5` on its log-likelihood scale; both adapters use
`maxP = 4`. Andrey's library default is looser: in the forward phase, a variable with more than
`n_variables / 2` parents gets no new edges. Both packages return the identical CPDAG on all 12
datasets they both finished.

| d | Andrey speedup compared with causal-learn |
|---:|---:|
| 20 | 2.94x |
| 50 | 3.25x |
| 100 | 3.97x |
| 150 | 4.28x |

causal-learn was not run above `d = 150` in this comparison.

**Penalty choice matters at large `d`.** In the sparse `n = 10d` regime, the default
`lambda_value = 1.0` can retain many false edges. At `d = 200`, capped Andrey GES has median
SHD 206, compared with 200 for the empty graph. With `lambda_value = 2.0`, its median SHD is 43.
The stronger penalty suits large `d` in this regime. This is a tuning choice, not a universal
accuracy guarantee.

## Permutation-based methods

BOSS and GRaSP search variable orders using linear-Gaussian BIC.

### BOSS and GRaSP

The matched BIC penalty is `lambda_value = 2` in causal-learn and `lambda_value = 4.0` in
Andrey: these methods use different penalty scales in the two packages.

| d | Andrey BOSS | causal-learn BOSS | Speedup | Andrey GRaSP | causal-learn GRaSP | Speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 0.031 | 0.097 | 3.2x | 0.037 | 0.071 | 2.1x |
| 50 | 0.17 | 1.02 | 5.7x | 0.42 | 0.77 | 1.8x |
| 100 | 1.31 | 6.74 | 4.9x | 2.83 | 4.38 | 1.5x |
| 200 | 6.73 | 37.1 | 5.6x | 40.1 | 55.9 | 1.4x |
| 400 | 39.0 | 163 | 4.3x | 1,040 | 1,170 | 1.4x |
| 800 | 292 | 1,020 | 3.4x | - | - | - |

BOSS median SHD is equal through `d = 100`; then Andrey's is 12 and causal-learn's 13 at 200, 4
and 12 at 400, and 24 and 58 at 800. GRaSP returns the same graph on 14 of 15 datasets. BOSS uses
more memory: 2.4x the baseline at `d = 200`, 2.9x at 400, and 1.7x at 800 (638 MB compared with
372 MB).

## Linear non-Gaussian methods

### DirectLiNGAM

DirectLiNGAM assumes linear mechanisms with non-Gaussian noise and returns a DAG. Both comparisons
below use its pairwise-likelihood measure. Timings are comparable within each table; the tables use
different builds and machines.

Compared with causal-learn, on exclusive CPU nodes:

| d | Andrey | causal-learn | Speedup | SHD, both packages | Empty-graph SHD |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.19 | 0.82 | 4.3x | 0 | 19 |
| 50 | 2.69 | 14.2 | 5.3x | 5 | 49 |
| 100 | 8.42 | 57.8 | 6.9x | 7 | 99 |
| 200 | 545 | 1,800 | 3.3x | 13 | 199 |

The two packages return the same graph on all 12 datasets.

Compared with lingam, on CPU and CUDA:

| d | Seeds | Andrey CUDA | Andrey CPU | lingam | CUDA vs CPU |
|---:|---:|---:|---:|---:|---:|
| 20 | 3 | 0.18 | 0.13 | 0.70 | 0.74x |
| 30 | 3 | 0.40 | 0.41 | 2.38 | 1.02x |
| 50 | 3 | 1.21 | 2.30 | 12.3 | 1.86x |
| 100 | 3 | 15.4 | 37.7 | 136 | 2.44x |
| 200 | 2 | 112 | 482 | Past 1,200 s | 4.34x |
| 300 | 1 | 528 | Past 3,600 s | Past 3,600 s | - |
| 400 | 1 | 1,410 | Past 5,400 s | Past 5,400 s | - |

Andrey CPU is 3.6-5.7x faster than lingam on shared completed sizes. CUDA overtakes Andrey CPU
at `d = 30`. The CPU timeout at `d = 300` includes warm-up; it is not a settled ceiling for a
single fit. CUDA is the only arm to finish at `d = 300` and 400 under the campaign caps.

Both Andrey arms return identical graphs on all 14 datasets they both finish. Compared with
lingam, mean SHD agrees within one edge at every shared size (6.67 compared with 6.33 at
`d = 100`). A
20-seed check at `d = 50` returns the same graph on 18 of 20 datasets.

### ICA-LiNGAM

ICA-LiNGAM estimates the mixing matrix with FastICA, derives a causal order from it, and returns a
DAG. Both packages run scikit-learn's FastICA with seed 0 and 1,000 iterations. They take the same
time, with paired ratios of 0.98-1.01x, and return the same graph on all 9 datasets.

| d | Andrey | causal-learn | Speedup | SHD, both packages | Empty-graph SHD |
|---:|---:|---:|---:|---:|---:|
| 20 | 0.44 | 0.44 | 0.99x | 27 | 19 |
| 50 | 1.08 | 1.08 | 1.01x | 62 | 49 |
| 100 | 15.8 | 15.5 | 0.98x | 181 | 99 |

## Large graphs

At the largest measured sizes, some runs return graphs with no fewer errors than the empty graph:

| Method | Package | d | SHD | Empty-graph SHD |
|---|---|---:|---:|---:|
| PC | gCastle | 200 | 256 | 200 |
| PC | gCastle | 400 | 792.5 | 400 |
| FCI | Andrey and causal-learn | 800 | 2,579 | 1,772 |
| GES, `lambda_value = 1.0` and `maxP = 4` | Andrey | 200 | 206 | 200 |

FCI's SHD is endpoint SHD. FCI at `d = 800` is worse than the empty graph in both packages, so there
the limit is the algorithm's. PC is worse than the empty graph only in gCastle: in Andrey and
causal-learn it stays below the empty graph at every measured size, up to `d = 400` (SHD 357
against 400). GES at `d = 200` is Andrey's run from the penalty comparison; causal-learn was not
run at that size, and with `lambda_value = 2.0` Andrey's median SHD there is 43.

Separately, on the `lingam_sf` data, ICA-LiNGAM in both packages scores worse than the empty graph
at every measured size, from `d = 20` to 100; DirectLiNGAM's SHD on the same data is 0-13.

On the published ANDES network (223 variables) with 1,000 simulated samples, every method in every
package scores worse than the empty graph by median SHD; see [Public datasets](#public-datasets).

## Known limits

- **Data coverage.** PC, GES, BOSS, and GRaSP results cover sparse linear-Gaussian Erdos-Renyi
  graphs at `n = 10d`; FCI hides variables in the same graphs; the LiNGAM methods use scale-free
  graphs with non-Gaussian noise. These results do not establish performance across all data
  regimes.
- **Column order.** FCI and GFCI depend on column order. A random relabeling of the columns changes
  FCI's marks in about 3% of cases and GFCI's in about 22% (20 and 155 of 711 relabelings of 237
  latent-confounded models, counted by `qa/column_order.py`). Keep a fixed column order for
  reproducibility.
- **PC and FCI scale.** Both are practical to roughly 1,000 variables on sparse data at
  `n = 10d`. The number of CI tests grows with neighborhood width and limits runtime. PC warns
  before a conditioning-size pass requiring at least 10 million tests and continues the search.
- **GES sparsity.** The default `lambda_value = 1.0` is textbook BIC; `lambda_value = 2.0` suits
  large `d` in the sparse-data regime above.
- **Hardware.** Ratios are specific to the measured hardware, thread budget, and settings.

## Reproduce

The [benchmark harness guide][harness] documents environment setup, campaign drivers, and
comparison rules. To rerun a comparison, install the Andrey alpha release (`andrey-core` 0.1.0)
and the package versions under [Recorded campaigns](#recorded-campaigns). Run its driver with the
data regimes under [How to read the results](#how-to-read-the-results), the penalties and
parameters in each method's section, and the settings below. Times will differ with the hardware
and the build; graphs and SHD should match closely.

| Comparison | Driver | d | Seeds x repeats | Warm-up fits | Wall cap (s) | Cores |
|---|---|---|---|---:|---:|---:|
| PC | `benchmark.py` | 20-400 | 3 x 3 | 1 | 1,800 | 16 |
| FCI | `fci_latent.py` | 20-800 | 3 x 3 | 0 | 3,600 | 16 |
| GES | `ges_cpu.py` | 20-200 | 3 x 1 | 1 | 3,600 | 128 |
| BOSS, GRaSP | `boss_grasp_cpu.py`; `boss_grasp_reach.py` for BOSS at 800 | 20-800 | 3 x 3 | 1 | 3,600 | 16 |
| DirectLiNGAM compared with lingam | `lingam_gpu.py` | 20-400 | 3 (2 at `d = 200`, 1 above) x 1-3 | 1 | 120-5,400, by size | 16 |
| DirectLiNGAM and ICA-LiNGAM compared with causal-learn | `lingam_cpu.py` | 20-200 | 3 x 3 | 0 | 3,600 | 16 |
| [Public datasets](#public-datasets) | `public_datasets.py` | 8-223 | Sachs 1 x 20; ASIA, ALARM 3 x 3; HEPAR2, ANDES 3 x 1 | 1 | 1,800 per fit, plus 300 | 1 |

The wall cap covers setup, warm-up, and the timed fit. Memory was capped at 64 GB per fit, except
for GES and the DirectLiNGAM comparison with lingam.

## Public datasets

This section compares the packages on data loaded with `andrey.data.load_dataset`: Sachs's
measured protein-signalling data, and four published Bayesian-network structures from the bnlearn
repository with linear-Gaussian data simulated on each. These datasets are small, so the bottom
line is parity: the same graphs in most cells, Andrey faster in every comparison where both
packages finish, and the exceptions named below.

**Settings.** The same methods and matched settings throughout:

- PC and FCI use Fisher-z at `alpha = 0.05`.
- GES uses Andrey's `lambda_value = 1.0` and causal-learn's 0.5, the same penalty on its scale,
  with `maxP = d/2`: a variable with more than `d/2` parents gets no new ones. gCastle's GES has no
  penalty setting.
- BOSS and GRaSP use `lambda_value = 1.0` (0.5 in causal-learn) and search seed 0; GRaSP uses
  depth 3.
- DirectLiNGAM uses the pairwise-likelihood measure, and ICA-LiNGAM uses FastICA with seed 0 and
  1,000 iterations.

Scoring is as in the rest of this report: CPDAGs against the truth's CPDAG, FCI by endpoint SHD
against the truth's PAG, and the LiNGAM methods against the DAG. "Same graph" means equal endpoint
marks after conversion to the method's graph class.

**Timing.** Seconds per fit, with one BLAS thread per fit, after one warm-up fit on ASIA that
covers imports and first-call costs: the median of 20 fits on Sachs, 3 on ASIA and ALARM, and 1 on
HEPAR2 and ANDES, then the median over seeds. Fits ran in parallel, one per core, on 16 cores of a
shared AMD EPYC 9555 machine, so single large fits vary under that load. Each fit's process was
stopped after 1,800 s of wall time per timed fit plus 300 s for imports and the warm-up fit. Every
stopped fit was a single timed fit on HEPAR2 or ANDES, stopped at 2,100 s, so "did not finish in
1,800 s" understates how long it ran. A fit that did not finish has no speed ratio. Speedups pair
the two packages' fits on each seed and take the median over seeds. Andrey 0.1.0 ran on Python 3.14
with NumPy 2.4.6; causal-learn 0.1.4.8, gCastle 1.0.4, and lingam 1.12.2 ran on Python 3.13 with
NumPy 2.5.2. The
{download}`public-datasets summary <../../benchmarks/published/alpha-2026-09/public-datasets.json>`
has every fit.

To rerun it, use `public_datasets.py` from the [benchmark harness][harness], once per repeat
count. It writes one JSON line per timed repeat to `records.jsonl`, with status `timeout` for a
fit past the cap. Its FCI rows compare on `shd_endpoint`, the endpoint SHD reported here; their
`shd` is the plain SHD. It runs one fit at a time, in one environment, with the warm-up fit on the
same data, so times will differ from the run above. The HEPAR2 and ANDES command takes about 7
hours.

```shell
$ANDREY_BENCH_PYTHON benchmarks/public_datasets.py --out outputs/public-datasets \
    --datasets sachs --repeats 20
$ANDREY_BENCH_PYTHON benchmarks/public_datasets.py --out outputs/public-datasets \
    --datasets asia alarm --repeats 3
$ANDREY_BENCH_PYTHON benchmarks/public_datasets.py --out outputs/public-datasets \
    --datasets hepar2 andes --repeats 1
```

### Sachs

853 observational cells and 11 variables, scored against the 17-edge consensus network. Every
package returns the same graph as Andrey for all seven methods. The first table is SHD (endpoint
SHD for FCI), where lower is better; the second is the time of one fit.

| Method | Andrey | causal-learn | gCastle | lingam | Empty graph |
|---|---:|---:|---:|---:|---:|
| PC | 11 | 11 | 11 | - | 17 |
| FCI | 20 | 20 | - | - | 34 |
| GES | 11 | 11 | 11 | - | 17 |
| BOSS | 11 | 11 | - | - | 17 |
| GRaSP | 11 | 11 | - | - | 17 |
| DirectLiNGAM | 14 | 14 | - | 14 | 17 |
| ICA-LiNGAM | 14 | 14 | - | 14 | 17 |

| Method | Andrey | causal-learn | gCastle | lingam | Speedup, causal-learn | Speedup, gCastle | Speedup, lingam |
|---|---:|---:|---:|---:|---:|---:|---:|
| PC | 0.7 ms | 4.2 ms | 7.1 ms | - | 6.3x | 11x | - |
| FCI | 0.6 ms | 4.5 ms | - | - | 7.1x | - | - |
| GES | 8.7 ms | 20.4 ms | 258 ms | - | 2.4x | 30x | - |
| BOSS | 2.2 ms | 5.1 ms | - | - | 2.3x | - | - |
| GRaSP | 2.9 ms | 6.0 ms | - | - | 2.1x | - | - |
| DirectLiNGAM | 20.6 ms | 60.8 ms | - | 66.4 ms | 3.0x | - | 3.2x |
| ICA-LiNGAM | 13.1 ms | 14.4 ms | - | 21.1 ms | 1.1x | - | 1.6x |

### Published network structures

ASIA (8 variables, 8 edges), ALARM (37, 46), HEPAR2 (70, 123), and ANDES (223, 338), each with
1,000 rows simulated at seeds 0, 1, and 2. The simulated data are Gaussian, so the LiNGAM methods,
which need non-Gaussian noise, are not run. The first table is accuracy: the median SHD over the
three seeds (endpoint SHD for FCI), where lower is better; "same graph" counts the seeds on which
causal-learn returns Andrey's graph. The second is the time of one fit.

| Network | Method | Andrey | causal-learn | gCastle | Empty graph | Same graph |
|---|---|---:|---:|---:|---:|---:|
| ASIA | PC | 3 | 3 | 1 | 8 | 3 of 3 |
| ASIA | FCI | 4 | 4 | - | 16 | 3 of 3 |
| ASIA | GES | 0 | 0 | 0 | 8 | 3 of 3 |
| ASIA | BOSS | 0 | 0 | - | 8 | 3 of 3 |
| ASIA | GRaSP | 0 | 0 | - | 8 | 3 of 3 |
| ALARM | PC | 24 | 24 | 30 | 46 | 3 of 3 |
| ALARM | FCI | 39 | 39 | - | 92 | 3 of 3 |
| ALARM | GES | 20 | 20 | 28 | 46 | 3 of 3 |
| ALARM | BOSS | 42 | 45 | - | 46 | 0 of 3 |
| ALARM | GRaSP | 31 | 31 | - | 46 | 3 of 3 |
| HEPAR2 | PC | 108 | 108 | 122 | 123 | 3 of 3 |
| HEPAR2 | FCI | 206 | 206 | - | 246 | 3 of 3 |
| HEPAR2 | GES | 109 | did not finish | did not finish | 123 | - |
| HEPAR2 | BOSS | 26 | 27 | - | 123 | 2 of 3 |
| HEPAR2 | GRaSP | 84 | 84 | - | 123 | 3 of 3 |
| ANDES | PC | 372 | 372 | 449 | 338 | 2 of 3 |
| ANDES | FCI | 696 | 696 | - | 676 | 3 of 3 |
| ANDES | GES | 612 | did not finish | did not finish | 338 | - |
| ANDES | BOSS | 506 | 457 | - | 338 | 0 of 3 |
| ANDES | GRaSP | 471 | 666 | - | 338 | 0 of 3 |

| Network | Method | Andrey | causal-learn | gCastle | Speedup, causal-learn | Speedup, gCastle |
|---|---|---:|---:|---:|---:|---:|
| ASIA | PC | 1.6 ms | 7.2 ms | 9.7 ms | 4.5x | 6.0x |
| ASIA | FCI | 1.5 ms | 7.8 ms | - | 5.0x | - |
| ASIA | GES | 5.7 ms | 12.7 ms | 86.0 ms | 2.2x | 15x |
| ASIA | BOSS | 3.7 ms | 5.4 ms | - | 1.4x | - |
| ASIA | GRaSP | 2.2 ms | 3.8 ms | - | 1.6x | - |
| ALARM | PC | 11.6 ms | 280 ms | 284 ms | 24x | 26x |
| ALARM | FCI | 18.1 ms | 274 ms | - | 15x | - |
| ALARM | GES | 0.50 s | 2.26 s | 51 s | 4.5x | 108x |
| ALARM | BOSS | 0.15 s | 0.65 s | - | 4.3x | - |
| ALARM | GRaSP | 0.12 s | 0.46 s | - | 3.9x | - |
| HEPAR2 | PC | 0.17 s | 4.24 s | 4.85 s | 29x | 29x |
| HEPAR2 | FCI | 0.26 s | 4.19 s | - | 14x | - |
| HEPAR2 | GES | 6.48 s | did not finish in 1,800 s | see below | - | - |
| HEPAR2 | BOSS | 0.77 s | 4.08 s | - | 5.3x | - |
| HEPAR2 | GRaSP | 1.29 s | 6.53 s | - | 5.1x | - |
| ANDES | PC | 0.41 s | 274 s | 14 s | 593x | 34x |
| ANDES | FCI | 0.68 s | 40 s | - | 56x | - |
| ANDES | GES | 375 s | did not finish in 1,800 s | did not finish in 1,800 s | - | - |
| ANDES | BOSS | 52 s | 274 s | - | 5.3x | - |
| ANDES | GRaSP | 41 s | 142 s | - | 3.0x | - |

- **GES.** causal-learn's GES did not finish in 1,800 s on any HEPAR2 or ANDES seed. gCastle's
  did not finish on ANDES, nor on HEPAR2 at seeds 0 and 2; at seed 1 it stopped with an error from
  gCastle ("The PDAG does not admit any extension").
- **BOSS on ANDES.** Andrey's BOSS returns graphs with more errors than causal-learn's on all
  three seeds: SHD 187, 762, and 506, compared with 186, 601, and 457. Scored with the same BIC, Andrey's
  graphs score better (lower) on every seed, by 1, 28,242, and 64,300. Both packages return more
  edges than the 338 true edges: Andrey 515, 997, and 714; causal-learn 514, 735, and 502.
- **BOSS on ALARM.** Andrey's SHD is 24, 42, and 52, compared with 45, 43, and 53; the BICs
  differ by at most 58.
- **GRaSP on ANDES.** Andrey's graphs score better BIC on every seed, by 56, 54,084, and 88,911,
  with SHD 415, 471, and 484, compared with 381, 805, and 666.
- **PC on ANDES.** At seed 2 the two packages return the same skeleton and SHD, but orient two
  edges at one node differently: one reversed, one undirected in causal-learn.
- **ANDES and the empty graph.** At 1,000 samples, by median SHD, every method in every package
  scores worse than the empty graph on ANDES. Per seed, BOSS at seed 0 and FCI at seed 2 beat it in
  both packages. See [Large graphs](#large-graphs).

[harness]: https://github.com/Abel-ai-lab/andrey/blob/main/benchmarks/README.md
