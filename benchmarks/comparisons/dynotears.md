# DYNOTEARS

`dynotears_ts.py` compares Andrey's DYNOTEARS with CausalNex's `from_pandas_dynamic`, the
implementation the paper's authors released, on the structural vector autoregressions of the
DYNOTEARS paper (Pamfil et al., 2020), with the empty window as the floor. The claim: on the paper's
design, at the paper's settings for 500 and for 50 time steps, Andrey's graphs make no more errors
than CausalNex's, and far fewer than the empty window's.

## The model

`andrey_bench/svar_models.py` draws one series per seed from
`x_t = x_t W + x_{t-1} A + z_t` (row vectors; `W[i, j]` weights `x_i(t) -> x_j(t)`, `A[i, j]`
weights `x_i(t-1) -> x_j(t)`), following Appendix B.4 with one lag, the order of the paper's
Figure 3:

| regime | same-step graph | lagged graph | noise | scale | claim |
|---|---|---|---|---|---|
| `dynotears_paper` | Erdos-Renyi, mean degree 2 | Erdos-Renyi, mean in-degree 2 | Gaussian | raw | in |
| `dynotears_paper_exponential` | Erdos-Renyi, mean degree 2 | Erdos-Renyi, mean in-degree 2 | exponential, centered | raw | in |
| `dynotears_paper_ba_sbm` | Barabasi-Albert, 2 edges per new node | two-block model, `p_out / p_in = 0.3`, mean in-degree 2 | Gaussian | raw | in |
| `dynotears_paper_standardized` | as `dynotears_paper` | | Gaussian | z-scored | out |
| `dynotears_paper_uncentered` | as `dynotears_paper` | | exponential of mean 1 | raw | out |
| `dynotears_sin_tanh` | as `dynotears_paper`, with `sin` of each same-step parent | with `tanh` of each lagged parent | Gaussian | raw | out |

Same-step weights are uniform in magnitude on [0.5, 2] and lagged ones on [0.3, 0.5], each with a
random sign; the lagged graph includes self-lags. The nodes are relabeled by a permutation drawn
from the seed. Section 2.1 takes noise of mean zero, so the exponential noise of the paper's design
has its mean 1 subtracted. A draw whose reduced-form lag matrix `A (I - W)^-1` has spectral radius 1
or more is drawn again, as the paper assumes a stationary series; the Barabasi-Albert graphs need
more than 1,000 draws for some seeds at 100 variables, so that regime runs to 50. A series has
`n + 1` steps, so every fit regresses `n` steps on the step before each, the paper's `n`. The truth
is the window over lags 0 and 1: the same-step DAG and the lagged links.

## Settings

Both packages run the paper's settings (Appendix B.3) through the same adapter fields, in Andrey's
names. `n500` is the paper's for 500 steps and Andrey's defaults; `n50` is the paper's for 50 steps.
A run at 500 steps uses `n500` and a run at 50 steps `n50`; the defaults at 50 steps are out of
claim.

| setting | `n500` | `n50` | CausalNex 0.12.2 (`from_pandas_dynamic`) |
|---|---|---|---|
| `tau_max` | 1 | 1 | `p` |
| `lambda1` | 0.05 | 0.2 | `lambda_w` and `lambda_a`, both set to it |
| `w_threshold` | 0.3 | 0.3 | `w_threshold=0.0` in the fit, then same-step weights kept where `abs(w) >= w_threshold` |
| `lag_threshold` | 0.1 | 0.2 | lagged weights kept where `abs(a) >= lag_threshold` |
| `max_iter`, `h_tol` | 100, `1e-8` | 100, `1e-8` | 100, `1e-8` (CausalNex's defaults) |

CausalNex's own defaults are `lambda_w = lambda_a = 0.1` and one threshold, 0, for both blocks, so
the adapter fits without a threshold and applies the two. Two differences remain, and each adapter's
docstring names them: Andrey centers every column of `X` and of the lagged `Y` before the fit, the
same as fitting an intercept, where CausalNex fits the data as given; and Andrey stops when the
penalty weight on the acyclicity function reaches `1e16` and then removes the weakest same-step
edges until no cycle is left, where CausalNex raises the weight up to `1e20` and returns its
thresholded weights. `--centered` adds `causalnex.dynotears.centered.<settings>`: CausalNex on `X`
and `Y` centered on their column means, built by its `DynamicDataTransformer` and fitted by
`from_numpy_dynamic`, so the two fits solve the same problem.

## Scores

Every estimate is a window over lags 0 and 1, scored link by link (`scoring.window_scores`). Its
lag 0 is a DAG, so it also has the paper's units (Appendix B.7):

| columns | what they score |
|---|---|
| `lag0_shd`, `lagged_shd`, `shd` | the edge SHD: same-step pairs whose state differs (a missing, an extra, or a reversed edge costs 1), lagged links in one graph and not the other, and their sum |
| `lag0_f1`, `lagged_f1`, `f1` | the F1 of the directed edges, a reversed edge one false and one missed: per block, and over both |

Two solutions return the same graph when their window marks are equal (`structure_hash`). A pair is
the same series fitted by both solutions in one run. D is Andrey's SHD minus the other solution's,
and its interval the percentile bootstrap 95% interval of the mean D over the pairs (10,000
resamples). A cell in claim is worse when the interval lies above 0, better when it lies below, and
a tie otherwise; a worse cell is run again on 20 fresh seeds and counts as worse only if that
interval also lies above 0. A fit stopped at the wall cap is a timeout: it is counted in the
coverage and never scored.

## Results

The numbers come from [`published/0.13.0/dynotears.json`](../published/0.13.0/dynotears.json): an
Andrey pre-release build of the 0.13.0 branch (31 October 2026) against CausalNex 0.12.2, on one
Intel Core i7-7820X, one core and one BLAS thread per fit, four fits at a time. Every setting in
claim has 20 series (seeds 0 to 19), every one out of claim 10 (seeds 0 to 9), each fitted once with
a 3,600 s cap; no fit reached it. SHD and F1 are means over the series, CPU s the median CPU time of
a fit. D is Andrey's SHD minus CausalNex's, split into its same-step and lagged parts.

At 500 steps, the paper's settings for 500 steps and Andrey's defaults, every setting is a tie, with
an F1 of at least 0.968 for both packages. Where the graphs differ, CausalNex on centered columns,
fitted afterward on those series alone, returns Andrey's graph on 29 of 34; the last column gives
both counts. Of the other 5, a rounding-level change, the data multiplied by
`1 + 1e-12` noise or the columns fitted in their given order, changes Andrey's graph on 4. On the
fifth, the one edge that differs is a true lagged link whose weight lies within 0.001 of the
threshold in both fits, kept by Andrey and dropped by CausalNex.

| noise | d | same graph | SHD Andrey | SHD CausalNex | SHD empty | D same-step | D lagged | D [95% CI] | F1 Andrey | F1 CausalNex | CPU s Andrey | CPU s CausalNex | graphs differ, centering explains |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian | 5 | 20 | 0.15 | 0.15 | 13.5 | +0.00 | +0.00 | +0.00 [+0.00, +0.00] | 0.995 | 0.995 | 0.30 | 0.36 | 0 / 0 |
| Gaussian | 10 | 19 | 1.75 | 1.80 | 29.3 | +0.00 | -0.05 | -0.05 [-0.15, +0.00] | 0.969 | 0.968 | 0.74 | 1.21 | 1 / 1 |
| Gaussian | 20 | 19 | 1.35 | 1.30 | 57.6 | +0.00 | +0.05 | +0.05 [+0.00, +0.15] | 0.988 | 0.988 | 2.72 | 4.51 | 1 / 1 |
| Gaussian | 50 | 16 | 1.20 | 1.75 | 144.6 | -0.35 | -0.20 | -0.55 [-1.60, +0.05] | 0.996 | 0.994 | 29 | 50 | 4 / 4 |
| Gaussian | 100 | 16 | 3.15 | 3.35 | 283.2 | +0.00 | -0.20 | -0.20 [-0.75, +0.20] | 0.994 | 0.994 | 299 | 393 | 4 / 2 |
| exponential | 5 | 20 | 0.15 | 0.15 | 13.5 | +0.00 | +0.00 | +0.00 [+0.00, +0.00] | 0.995 | 0.995 | 0.30 | 0.37 | 0 / 0 |
| exponential | 10 | 19 | 1.35 | 1.30 | 29.3 | +0.00 | +0.05 | +0.05 [+0.00, +0.15] | 0.977 | 0.977 | 0.73 | 1.27 | 1 / 1 |
| exponential | 20 | 16 | 1.35 | 1.35 | 57.6 | +0.00 | +0.00 | +0.00 [-0.20, +0.20] | 0.987 | 0.987 | 2.79 | 4.53 | 4 / 3 |
| exponential | 50 | 15 | 2.10 | 1.95 | 144.6 | +0.00 | +0.15 | +0.15 [-0.05, +0.35] | 0.993 | 0.993 | 30 | 50 | 5 / 5 |
| exponential | 100 | 12 | 6.25 | 6.30 | 283.2 | +0.00 | -0.05 | -0.05 [-0.85, +0.55] | 0.988 | 0.988 | 385 | 547 | 8 / 6 |

At 50 steps, the paper's settings for 50 steps, every run includes CausalNex on centered columns.
Against it Andrey returns the same graph in most series up to 20 variables, and no setting differs
once replicated. Against CausalNex as given, three settings have an interval above 0 on seeds 0 to
19; each was run again on seeds 20 to 39 (the replication below).

| noise | d | same graph | SHD Andrey | SHD CausalNex | SHD empty | D same-step | D lagged | D [95% CI] | F1 Andrey | F1 CausalNex | CPU s Andrey | CPU s CausalNex | SHD CausalNex centered | D vs centered [95% CI] | same graph vs centered |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian | 5 | 14 | 5.00 | 5.35 | 13.5 | -0.15 | -0.20 | -0.35 [-0.80, +0.00] | 0.780 | 0.760 | 0.30 | 0.33 | 5.00 | +0.00 [+0.00, +0.00] | 20 |
| Gaussian | 10 | 7 | 12.35 | 12.20 | 29.3 | +0.10 | +0.05 | +0.15 [-0.30, +0.55] | 0.748 | 0.752 | 0.86 | 1.15 | 12.35 | +0.00 [+0.00, +0.00] | 20 |
| Gaussian | 20 | 4 | 21.65 | 20.80 | 57.6 | +0.05 | +0.80 | +0.85 [+0.05, +1.65] | 0.772 | 0.784 | 4.48 | 6.71 | 21.65 | +0.00 [-0.15, +0.15] | 18 |
| Gaussian | 50 | 0 | 59.60 | 59.00 | 144.6 | -0.05 | +0.65 | +0.60 [-0.15, +1.35] | 0.756 | 0.757 | 63 | 76 | 59.65 | -0.05 [-0.40, +0.25] | 12 |
| Gaussian | 100 | 0 | 143.95 | 137.05 | 283.2 | +2.65 | +4.25 | +6.90 [+4.30, +9.55] | 0.692 | 0.708 | 700 | 850 | 143.30 | +0.65 [-0.40, +2.45] | 6 |
| exponential | 5 | 14 | 5.40 | 5.25 | 13.5 | +0.05 | +0.10 | +0.15 [-0.05, +0.35] | 0.759 | 0.766 | 0.28 | 0.34 | 5.40 | +0.00 [+0.00, +0.00] | 20 |
| exponential | 10 | 7 | 12.95 | 12.75 | 29.3 | +0.20 | +0.00 | +0.20 [-0.25, +0.70] | 0.739 | 0.744 | 1.00 | 1.33 | 12.90 | +0.05 [-0.10, +0.20] | 17 |
| exponential | 20 | 3 | 23.55 | 23.40 | 57.6 | +0.05 | +0.10 | +0.15 [-0.50, +0.75] | 0.755 | 0.758 | 4.03 | 5.25 | 23.55 | +0.00 [+0.00, +0.00] | 20 |
| exponential | 50 | 0 | 59.45 | 58.80 | 144.6 | -0.05 | +0.70 | +0.65 [-0.75, +2.00] | 0.762 | 0.764 | 62 | 76 | 59.35 | +0.10 [-0.15, +0.35] | 14 |
| exponential | 100 | 0 | 148.95 | 145.75 | 283.2 | +1.00 | +2.20 | +3.20 [+0.10, +6.00] | 0.685 | 0.693 | 785 | 839 | 148.20 | +0.75 [+0.20, +1.40] | 8 |

The Barabasi-Albert and block-model graphs, at 500 steps and up to 50 variables, are a tie at every
size.

| noise | d | same graph | SHD Andrey | SHD CausalNex | SHD empty | D same-step | D lagged | D [95% CI] | F1 Andrey | F1 CausalNex | CPU s Andrey | CPU s CausalNex | graphs differ, centering explains |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian | 5 | 20 | 0.75 | 0.75 | 14.7 | +0.00 | +0.00 | +0.00 [+0.00, +0.00] | 0.972 | 0.972 | 0.34 | 0.38 | 0 / 0 |
| Gaussian | 10 | 19 | 0.20 | 0.25 | 31.4 | +0.00 | -0.05 | -0.05 [-0.15, +0.00] | 0.997 | 0.996 | 0.74 | 1.06 | 1 / 1 |
| Gaussian | 20 | 20 | 0.55 | 0.55 | 67.8 | +0.00 | +0.00 | +0.00 [+0.00, +0.00] | 0.995 | 0.995 | 2.96 | 4.52 | 0 / 0 |
| Gaussian | 50 | 15 | 1.85 | 1.95 | 176.6 | +0.00 | -0.10 | -0.10 [-0.30, +0.10] | 0.995 | 0.994 | 35 | 61 | 5 / 5 |

## Replication

In claim, 23 of the 24 settings count as a tie and one as worse:

| setting | seeds 0 to 19 | seeds 20 to 39 | counts as |
|---|---|---|---|
| Gaussian, 20 variables, 50 steps | worse | tie | tie |
| Gaussian, 100 variables, 50 steps | worse | tie | tie |
| exponential, 100 variables, 50 steps | worse | worse | worse |

| noise | d | seeds | same graph | SHD Andrey | SHD CausalNex | D [95% CI] | SHD CausalNex centered | D vs centered [95% CI] | same graph vs centered |
|---|---|---|---|---|---|---|---|---|---|
| Gaussian | 20 | 0-19 | 4 | 21.65 | 20.80 | +0.85 [+0.05, +1.65] | 21.65 | +0.00 [-0.15, +0.15] | 18 |
| Gaussian | 20 | 20-39 | 3 | 21.55 | 20.90 | +0.65 [-0.45, +1.70] | 21.55 | +0.00 [+0.00, +0.00] | 20 |
| Gaussian | 100 | 0-19 | 0 | 143.95 | 137.05 | +6.90 [+4.30, +9.55] | 143.30 | +0.65 [-0.40, +2.45] | 6 |
| Gaussian | 100 | 20-39 | 0 | 144.70 | 142.90 | +1.80 [-1.05, +4.70] | 144.65 | +0.05 [-0.30, +0.35] | 10 |
| exponential | 100 | 0-19 | 0 | 148.95 | 145.75 | +3.20 [+0.10, +6.00] | 148.20 | +0.75 [+0.20, +1.40] | 8 |
| exponential | 100 | 20-39 | 0 | 154.45 | 150.30 | +4.15 [+1.30, +7.10] | 153.90 | +0.55 [-0.20, +1.45] | 4 |

The worse setting is explained by centering, Andrey's fit of every column on its own mean. With 50
steps for 100 variables the least-squares problem has more weights than steps, and whether the
sample means are fitted moves the solution: on CausalNex's side, centering the columns brings the
mean gap from 3.20 errors per series down to 0.75 on seeds 0 to 19, and from 4.15 to 0.55 on seeds
20 to 39. The gap left lies above 0 on seeds 0 to 19 alone, and it is the optimizer's path. On four
of those series Andrey stops on `h_tol` with no edge removed by the cycle repair, and refitting it
with CausalNex's cap on the penalty weight, `1e20`, returns CausalNex's graph on none of them; the
data multiplied by `1 + 1e-12` noise changes Andrey's graph on three of the four, one of them to
CausalNex's graph exactly. The same centering keeps Andrey's fit from reading a series' mean: with
exponential noise of mean 1 (out of claim) CausalNex as given makes several times Andrey's errors,
and CausalNex on centered columns exactly as many as Andrey.

## Out of claim

Out of claim, at ten series per setting: the defaults at 50 steps, z-scored columns, exponential
noise of mean 1, and a nonlinear vector autoregression. Both packages read the data's units, so
z-scored columns raise both packages' errors alike, and the defaults at 50 steps pass the empty
window's errors from 20 variables on.

| regime | T | d | same graph | SHD Andrey | SHD CausalNex | SHD empty | D [95% CI] | F1 Andrey | F1 CausalNex | SHD CausalNex centered |
|---|---|---|---|---|---|---|---|---|---|---|
| the defaults at 50 steps | 50 | 5 | 6 | 4.70 | 4.70 | 14.6 | +0.00 [-0.40, +0.40] | 0.850 | 0.851 | - |
| the defaults at 50 steps | 50 | 10 | 0 | 22.70 | 22.50 | 29.9 | +0.20 [-1.60, +2.00] | 0.698 | 0.698 | - |
| the defaults at 50 steps | 50 | 20 | 0 | 88.40 | 84.80 | 59.4 | +3.60 [+0.00, +7.20] | 0.534 | 0.549 | - |
| the defaults at 50 steps | 50 | 50 | 0 | 330.70 | 337.10 | 147.2 | -6.40 [-14.60, +2.60] | 0.409 | 0.407 | - |
| z-scored columns | 500 | 5 | 10 | 8.30 | 8.30 | 14.6 | +0.00 [+0.00, +0.00] | 0.648 | 0.648 | - |
| z-scored columns | 500 | 10 | 10 | 19.70 | 19.70 | 29.9 | +0.00 [+0.00, +0.00] | 0.602 | 0.602 | - |
| z-scored columns | 500 | 20 | 9 | 39.20 | 39.10 | 59.4 | +0.10 [+0.00, +0.30] | 0.604 | 0.605 | - |
| z-scored columns | 500 | 50 | 8 | 82.60 | 82.40 | 147.2 | +0.20 [+0.00, +0.50] | 0.666 | 0.667 | - |
| exponential noise of mean 1 | 500 | 5 | 2 | 0.20 | 4.60 | 14.6 | -4.40 [-6.00, -2.70] | 0.994 | 0.859 | 0.20 |
| exponential noise of mean 1 | 500 | 10 | 4 | 1.60 | 7.80 | 29.9 | -6.20 [-12.30, -1.60] | 0.972 | 0.885 | 1.60 |
| exponential noise of mean 1 | 500 | 20 | 1 | 1.70 | 11.20 | 59.4 | -9.50 [-13.80, -5.60] | 0.985 | 0.910 | 1.70 |
| nonlinear (sin, tanh) | 500 | 5 | 9 | 4.10 | 4.20 | 15.8 | -0.10 [-0.30, +0.00] | 0.861 | 0.858 | - |
| nonlinear (sin, tanh) | 500 | 10 | 10 | 9.40 | 9.40 | 30.0 | +0.00 [+0.00, +0.00] | 0.820 | 0.820 | - |
| nonlinear (sin, tanh) | 500 | 20 | 8 | 13.60 | 13.60 | 54.4 | +0.00 [-0.30, +0.30] | 0.860 | 0.859 | - |

## Reproduce

Build CausalNex's environment, fill the shared dataset store once, then cut the campaign into runs
with their own `--out`; pairs form inside one run, so keep a run's whole solution list:

```shell
benchmarks/start_causalnex.sh
export ANDREY_BENCH_CAUSALNEX_PYTHON=$PWD/benchmarks/.venv-causalnex/bin/python
export ANDREY_BENCH_DATA=$PWD/outputs/dynotears-data
$ANDREY_BENCH_PYTHON benchmarks/dynotears_ts.py --out outputs/dynotears-500 \
    --ladder 5 10 20 50 100 --length 500 --seeds 20 --cores 1 --cap-wall-s 3600
$ANDREY_BENCH_PYTHON benchmarks/dynotears_ts.py --out outputs/dynotears-50 \
    --ladder 5 10 20 50 100 --length 50 --seeds 20 --centered --cores 1 --cap-wall-s 3600
```

`--regimes dynotears_paper_ba_sbm` with `--ladder 5 10 20 50` runs the Barabasi-Albert graphs;
`--seed-start 20` runs a replication; `--regimes` with an out-of-claim regime, or `--length 50
--settings n500`, runs the settings out of claim, at `--seeds 10`.
