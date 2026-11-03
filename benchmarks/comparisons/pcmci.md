# PCMCI on lagged series

`pcmci_cpu.py` compares Andrey's PCMCI with tigramite's, with the empty lag stack as the floor.
Time-series sizes are written N x T: N variables (nodes) and a series of T time steps (the window
length), and where a cell holds several trajectories or samples, M of them per N x T cell. A rung
is one N x T size; the driver runs every N in its ladder at every T in its lengths, and a
dataset is one series, stored with `d = N` and `n = T`.

Each method runs on the simulation of its own paper, drawn by `andrey_bench/temporal_models.py`,
which keeps the expected number of cross-links per variable fixed as N grows. Every variable
depends on its own previous step, and a model whose process is not stationary is drawn again. If
none of 1000 draws is stationary, as for PCMCI+ at N = 240 and seed 1, the first draw is kept and
its lagged coefficients are scaled together until the spectral radius is 0.95:

| regime | method | source | cross-links per variable | same-step share | lags |
|---|---|---|---|---|---|
| `pcmci_paper` | PCMCI | Runge et al. (2019), Supplementary Sect. S4.1 | 1 | 0 | 1-2 |
| `pcmciplus_paper` | PCMCI+ | Runge (2020), Sect. 4 | 1.5 | 30% | 1-5 |
| `lpcmci_paper` | LPCMCI | Gerhardus and Runge (2020), Sect. 4 | 1, 30% of series unobserved | 30% | 1-3 |

The coefficients, autocorrelations, and noise follow each paper as well; the module docstring lists
them. A truth without same-step links is stored as the marks of lags `1..L`. `var_lagged`, the
earlier regime, draws each lag-1 cross-link with probability 0.3, so its links per variable grow
with N.

Every solution tests lags 1 to `tau_max`, with a PC1 level of 0.2 and a link level of 0.05 (Andrey's
`pc1_alpha` and `alpha`, tigramite's `pc_alpha` and `alpha_level`):

| solution | test |
|---|---|
| `andrey.pcmci.student_t.tau<k>` | partial correlation, Student-t p-value (Andrey's default) |
| `andrey.pcmci.fisherz.tau<k>` | partial correlation, Fisher-z p-value |
| `tigramite.pcmci.parcorr.tau<k>` | `ParCorr(significance="analytic")`, Student-t p-value |
| `baseline.empty.temporal.tau<k>` | none: no link at any lag |

tigramite runs `run_pcmci` with `tau_min=1` and `fdr_method="fdr_bh"`, Andrey's default, and its
other defaults; its own `tau_min=0` would also test lag 0, which is PCMCI+'s scope. `tau_max` is in
every name because `params` are not part of a `run_id`, and the record carries it as `tau_max`.
Andrey's PCMCI runs its tests with one BLAS thread whatever the core budget, while tigramite's fits
get the budget's BLAS threads; set the core budget to 1 to compare one core with one core. Andrey's
Student-t test and tigramite's analytic `ParCorr` share their test, so their p-values match. Both
run the same `fdr_method="fdr_bh"`, so the comparison is like for like. The driver ends by listing,
per dataset, the solutions that returned identical graphs (`structure_hash`).

A temporal estimate is scored lag by lag with `andrey.metrics.temporal_scores`; its CPDAG and PAG
columns (`shd`, `skeleton_*`, `arrowhead_*`) stay empty. Each lag is a directed graph, so a link in
the wrong direction counts 2:

| columns | what they score |
|---|---|
| `lag<k>_shd`, `lag<k>_precision`, `lag<k>_recall`, `lag<k>_f1` | the links at lag `k`, for every lag in the estimate or the truth |
| `temporal_shd` | the lag-resolved SHD: the per-lag SHDs summed |
| `lagged_*` | the lags merged into one graph, self-loops included |
| `summary_*` | the lags merged into one graph, self-loops left out |

A lag with no true link has recall 1, and a lag with no estimated link precision 1, so read them
beside the SHD: with every true link at lag 1, the columns of lag 2 and above count false links
only. A smoke run covers every solution on one series of 200 steps over five variables:

```shell
ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python \
    $ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py --out outputs/pcmci-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py --out outputs/pcmci \
    --ladder 5 10 20 --lengths 500 1000 --tau-max 2 --cores 1
```

The driver refuses a series shorter than `2 * tau_max * (N + 1) + 3` steps, the shortest Andrey's
PCMCI accepts.
