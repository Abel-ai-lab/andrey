# UT-IGSP

`ut_igsp_cpu.py` compares Andrey's `ut_igsp` with `unknown_target_igsp`, the authors' library
(graphical-model-learning 0.1a8 with conditional-independence 0.1a6 and graphical-models 0.1a21),
on data that mixes an observational environment with interventional settings whose targets are
known in part or not at all. The library's IGSP and JCI-GSP, Andrey's `gies` with the designed
targets, and Andrey's `cdnod` run beside them as context: reported, never judged.

## Data

`andrey.data` draws each dataset in the setting of Squires, Wang, and Uhler (2020, Section 5.1): an
Erdos-Renyi DAG on 20 variables with mean degree 1.5 (exactly 15 edges, where the authors'
generator draws each pair with probability 1.5/19), weights uniform on `[0.25, 1]` in absolute
value with a random sign, standard normal noise in the generator's own units (`scale="raw"`), and
its columns in an order drawn from the seed. An observational environment and 5 settings have `n`
rows each. Each setting intervenes on one designed target, which the fit is told, and on `l`
off-targets, which it is not. The DAG depends on the seed alone, so the cells of a seed range
share it, and their designed targets too except in G6, which draws its designed targets among the
variables with parents; the cells at one seed and `n` share their observational rows.

| Kind | Intervention on each target | Source |
|---|---|---|
| `shift` | noise mean moved by 1, `soft(shift=1.0)` | the paper's text |
| `perfect` | parents cut, `N(1, 0.1)`, `hard(1.0, sd=0.1**0.5)` | the paper's text (variance 0.1) |
| `shift2` | noise mean moved by 2, `soft(shift=2.0)` | the authors' code |
| `perfect_code` | parents cut, `N(1, 0.01)`, `hard(1.0, sd=0.1)` | the authors' code (variance 0.01) |
| `variance` | noise standard deviation doubled, `soft(noise_scale=2.0)` | claimed, not reported |
| `weight` | parent weights halved, `soft(weight_scale=0.5)`, targets among variables with parents | claimed, not reported |

## Cells

In claim, where the paper claims the method works (perfect and imperfect interventions, targets
known in part or not at all) at the model class and regime it reports (linear Gaussian, 20
variables, 5 settings, at least 1,000 rows per setting), 20 datasets per cell, 5 draws per dataset:

| Arm | Kind | `l` | `n` per setting |
|---|---|---|---|
| G1 | `shift` | 0-3 | 1,000-5,000 |
| G2 | `perfect` | 0-3 | 1,000-5,000 |
| G3 | `shift2`, `perfect_code` | 0, 3 | 1,000 |
| G4 | `variance` | 0, 3 | 1,000, 5,000 |
| G5 | `shift`, G1's datasets with no targets given | 0, 3 | 1,000, 5,000 |
| G6 | `weight` | 0, 3 | 1,000, 5,000 |

Out of claim, measured for the method's limitations, one draw per dataset: small `n` (O1), weak
shifts of 0.25 and 0.5 (O2), a dense graph on 50 variables with 3 settings (O3), 50 and 100
variables (O4), additive-noise data with the Gaussian tests (O5), additive-noise data on 10
variables with kernel tests (O6), categorical data with the G-test, Andrey alone (O7), 1 and 2
settings (O8), and 2,000 observational rows with settings of 300 (O9).

The comparison runs on seeds 0 to 19 (0 to 9 at 50 and 100 variables and in O3 and O7). A cell
whose interval lies above 0 reruns on seeds 2000 to 2019. Andrey's defaults were chosen beforehand
on seeds of their own: 1000 to 1019, then 4000 to 4019 with a rerun on 5000 to 5019 (see
[Choosing the defaults](#choosing-the-defaults)).

## Settings

The library runs at its defaults: Gaussian tests at alpha `1e-5`, the paper's level, depth 4, 5
runs, and the lexicographic objective. Andrey runs twice against those same fits: at its defaults,
alpha one over the number of observational rows for both tests (`1e-3` at 1,000 rows, `2e-4` at
5,000) with 10 starts, and at the paper's level, alpha `1e-5` with 5 starts. A draw seeds Andrey's
`seed` or the library's Python `random` and NumPy global generator with the draw number, `0` to
`4`. As context, the library also runs at Andrey's defaults: alpha one over the observational rows
and 10 runs.

JCI-GSP runs as a copy of the 0.1a8 `jci_gsp` with two call fixes: the shipped function passes
tuple pairs as fixed adjacencies, which the library's minimal-I-MAP builder rejects, and unpacks two
values from a search that returns one. The copy hands the builder frozensets and asks the search
for its summaries. Its tester is the authors' combined Gaussian test. In O6, the library's kernel
tests at their defaults regress on the conditioning set with a GAM and test the residuals with
HSIC: GAM-residual HSIC, not a conditional KCI. Its conditional KCI (`regress=False`) raised an
`IndexError` on a first fit and is not run.

Where Andrey's method differs from the library's:

| | Andrey | Library |
|---|---|---|
| Objective | edges plus targets (the sum), the paper's text | edges, then targets that test variant (lexicographic, `tup_score=True`), the authors' experiments |
| CI test | two-sided Fisher z at alpha | removals: two-sided at alpha; the start graph and start I-MAPs: a precision-matrix threshold with a one-sided p-value, so an edge with a two-sided p-value up to 2 alpha stays |
| Coefficients and intercept | Chow F, upper tail at alpha / 2 | a Wald-type statistic with separate residual variances on `F(q, n1 + n2 - q)`, two-sided at alpha / 4 per tail, so close agreement also rejects |
| Residual variance | F on `(n_k - q_k, n_0 - q_0)`, two-sided at alpha / 2 | F on `(n1 - 1, n2 - 1)`, two-sided at alpha / 4 per tail |
| Known targets | always counted as targets | counted only when they test variant; one that tests invariant leaves the returned list |
| Returned targets | the known ones and the learned ones | the learned ones; scored with the known ones added, as the authors score them |
| Search | a depth-first search over reversals, depth 4, that backtracks | its own loop over the same reversals, depth 4; the library's IGSP does not backtrack at a dead end |

## Scores

The parent scores every package with one function. Each estimate is the interventional essential
graph of the returned graph under the targets the method returned or was given; the truth is that
of the generating DAG under the true targets.

| Metric | Meaning |
|---|---|
| `shd` | variable pairs whose endpoint marks differ from the truth's (judged) |
| `target_error` | false plus missed targets over the settings (judged) |
| `target_floor` | `target_error` of the given targets alone |
| `shd_true_targets`, `in_true_imec` | the estimate's DAG re-read under the true targets, and whether it is in the true class |
| `skeleton_shd` | adjacency differences |
| `empty_shd` | the truth's adjacent pairs: the empty graph's `shd` |
| `wall_s` | the timed fit: the tests' construction and the search |

A dataset's error per package is the mean over its draws, and D is Andrey's error minus the
library's, per dataset. A comparison is worse when the 95% percentile bootstrap interval of the mean
of D over the 20 datasets lies above 0 and its rerun on 20 fresh datasets lies above 0 too, and
better when it lies below 0 and so does its rerun on seeds 2000 to 2019. At Andrey's defaults a
comparison also has seeds 4000 to 4019, the seeds that chose the default (see
[Replication](#replication)); they are shown and decide no verdict. An interval that contains 0 is a
tie. Its ends are rounded to 10 decimals before its side is read, so an end that is floating-point
residue of an exact 0, such as 5e-18, counts as 0. The run's own scripts read 3 such intervals as
above or below 0; here they tie, which changes no decision, and at `1e-5` with the lexicographic
objective the confirmation has 2 comparisons above 0 where the run recorded 3. In claim, a cell also
fails when Andrey timed out on a dataset the library finished, or when Andrey is not below the empty
estimate where the library is: its `shd` against the empty graph's, and where the given targets miss
some, its target error against theirs alone. The floor applies only where the library's own interval
against the empty estimate lies below 0. `qa/reference_parity.py` applies the rule to a summary in
this format.

## Run

```shell
benchmarks/start_utigsp.sh
export ANDREY_BENCH_UTIGSP_PYTHON=$PWD/benchmarks/.venv-utigsp/bin/python
$ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py check-truth --out outputs/utigsp-truth
$ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py run --arm G1 --out outputs/utigsp-G1 --workers 4
$ANDREY_BENCH_PYTHON benchmarks/ut_igsp_cpu.py summarize --runs outputs/utigsp-G1 \
    --out outputs/utigsp-summary.json
```

`check-truth` compares the scorer's essential graph with graphical-models' on 100 random DAGs and
target families. `run` takes one arm, optionally some of its cells and seeds, and fits each
dataset's packages back to back, one child process per package, each pinned to one core with one
BLAS thread; `--resume` continues a run. A draw over its cap, 1,800 s (3,600 s at 100 variables
and in O6), is a timeout, never scored. `run --block C` reruns a cell on seeds 2000 to 2019, and
`summarize --replications` adds those reruns to their rows. `select --runs ...` applies the rule
of the first step of [Choosing the defaults](#choosing-the-defaults).

## Results

Andrey's pre-release build of 3 November 2026 against graphical-model-learning 0.1a8, one fit at a
time on one core of an AMD EPYC 7713 processor with one BLAS thread. The library's fits come from
an earlier run on the same datasets: the build regenerated all 1,800 of them identically, and its
fits at the paper's level returned the same graph and targets as that run's in all 7,160 draws.
Every number below is in [`published/0.14.0/ut-igsp.json`](../published/0.14.0/ut-igsp.json), with
the context arms, the coverage, and the floors of every cell.

### In the claim

At its defaults, Andrey is worse than the library in none of the 112 comparisons (56 cells, SHD and
target error) and better in 1: its errors were lower on seeds 0 to 19 and again on the rerun on
seeds 2000 to 2019. 17 more were lower on seeds 0 to 19 and on seeds 4000 to 4019, the seeds that
chose the default, as the setting with the most comparisons below 0 there; those seeds are not an
independent rerun, so these count as ties (see [Replication](#replication)). Another 15 were lower
on seeds 0 to 19 and tied on the rerun, 1 was higher on seeds 0 to 19 and tied on the rerun, and the
other 78 tie. At the paper's level, Andrey is worse in none and better in none: 5 comparisons were
higher on seeds 0 to 19 and tied on the rerun, 3 were lower and had no rerun, and the other 104 tie.

The 18 lower on seeds 0 to 19 and on a second seed set come mostly from the level. Run at Andrey's
defaults, alpha one over the observational rows and 10 runs, the library ties Andrey in 107 of the
112 comparisons; in the other 5, all with weight changes (G6), Andrey's errors were lower on seeds 0
to 19, with no rerun.

At Andrey's defaults, means over the 20 datasets of each cell, seeds 0 to 19. D is Andrey's error
minus the library's, with its 95% interval. The empty graph's SHD is 15 in every cell.

| Cell | SHD, Andrey | SHD, reference | SHD D [95%] | SHD | Target error, Andrey | Target error, reference | Target error D [95%] | Target error |
|---|---|---|---|---|---|---|---|---|
| G1 `shift`, l 0, n 1,000 | 0.06 | 1.08 | -1.02 [-1.63, -0.47] | lower on seeds 0-19 and 4000-4019 | 0.14 | 0.20 | -0.06 [-0.20, +0.08] | tie |
| G1 `shift`, l 0, n 2,000 | 0.09 | 0.14 | -0.05 [-0.22, +0.11] | tie | 0.07 | 0.05 | +0.02 [-0.09, +0.15] | tie |
| G1 `shift`, l 0, n 3,000 | 0.10 | 0.06 | +0.04 [-0.09, +0.18] | tie | 0.10 | 0.00 | +0.10 [0.00, +0.25] | tie |
| G1 `shift`, l 0, n 4,000 | 0.02 | 0.04 | -0.02 [-0.12, +0.06] | tie | 0.05 | 0.00 | +0.05 [0.00, +0.15] | tie |
| G1 `shift`, l 0, n 5,000 | 0.02 | 0.04 | -0.02 [-0.12, +0.06] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G1 `shift`, l 1, n 1,000 | 0.28 | 1.18 | -0.90 [-1.42, -0.42] | lower on seeds 0-19 and 4000-4019 | 0.25 | 0.57 | -0.32 [-0.57, -0.07] | lower on seeds 0-19 only |
| G1 `shift`, l 1, n 2,000 | 0.30 | 0.44 | -0.14 [-0.40, +0.07] | tie | 0.17 | 0.18 | -0.01 [-0.08, +0.07] | tie |
| G1 `shift`, l 1, n 3,000 | 0.18 | 0.25 | -0.07 [-0.39, +0.17] | tie | 0.14 | 0.11 | +0.03 [-0.15, +0.20] | tie |
| G1 `shift`, l 1, n 4,000 | 0.01 | 0.08 | -0.07 [-0.20, +0.02] | tie | 0.05 | 0.03 | +0.02 [-0.07, +0.15] | tie |
| G1 `shift`, l 1, n 5,000 | 0.00 | 0.16 | -0.16 [-0.44, 0.00] | tie | 0.00 | 0.07 | -0.07 [-0.20, 0.00] | tie |
| G1 `shift`, l 2, n 1,000 | 0.47 | 1.27 | -0.80 [-1.35, -0.26] | lower on seeds 0-19 only | 0.49 | 1.36 | -0.87 [-1.80, -0.15] | lower on seeds 0-19 only |
| G1 `shift`, l 2, n 2,000 | 0.32 | 0.42 | -0.10 [-0.36, +0.15] | tie | 0.26 | 0.33 | -0.07 [-0.25, +0.12] | tie |
| G1 `shift`, l 2, n 3,000 | 0.35 | 0.35 | 0.00 [-0.29, +0.30] | tie | 0.30 | 0.28 | +0.02 [-0.19, +0.24] | tie |
| G1 `shift`, l 2, n 4,000 | 0.15 | 0.25 | -0.10 [-0.28, 0.00] | tie | 0.20 | 0.21 | -0.01 [-0.14, +0.13] | tie |
| G1 `shift`, l 2, n 5,000 | 0.20 | 0.23 | -0.03 [-0.24, +0.15] | tie | 0.25 | 0.19 | +0.06 [-0.12, +0.30] | tie |
| G1 `shift`, l 3, n 1,000 | 0.18 | 1.22 | -1.04 [-1.79, -0.40] | lower on seeds 0-19 only | 0.35 | 1.29 | -0.94 [-1.65, -0.35] | lower on seeds 0-19 only |
| G1 `shift`, l 3, n 2,000 | 0.46 | 0.20 | +0.26 [-0.09, +0.69] | tie | 0.39 | 0.29 | +0.10 [-0.30, +0.48] | tie |
| G1 `shift`, l 3, n 3,000 | 0.35 | 0.15 | +0.20 [0.00, +0.43] | tie | 0.41 | 0.20 | +0.21 [+0.01, +0.45] | higher on seeds 0-19 only |
| G1 `shift`, l 3, n 4,000 | 0.24 | 0.11 | +0.13 [-0.04, +0.36] | tie | 0.35 | 0.16 | +0.19 [0.00, +0.43] | tie |
| G1 `shift`, l 3, n 5,000 | 0.25 | 0.10 | +0.15 [-0.06, +0.48] | tie | 0.26 | 0.16 | +0.10 [-0.04, +0.32] | tie |
| G2 `perfect`, l 0, n 1,000 | 0.10 | 1.04 | -0.94 [-1.57, -0.37] | lower on seeds 0-19 only | 0.10 | 0.20 | -0.10 [-0.29, +0.06] | tie |
| G2 `perfect`, l 0, n 2,000 | 0.14 | 0.14 | 0.00 [-0.12, +0.14] | tie | 0.09 | 0.05 | +0.04 [-0.03, +0.15] | tie |
| G2 `perfect`, l 0, n 3,000 | 0.10 | 0.06 | +0.04 [-0.09, +0.18] | tie | 0.10 | 0.00 | +0.10 [0.00, +0.25] | tie |
| G2 `perfect`, l 0, n 4,000 | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie | 0.05 | 0.00 | +0.05 [0.00, +0.15] | tie |
| G2 `perfect`, l 0, n 5,000 | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G2 `perfect`, l 1, n 1,000 | 0.10 | 0.89 | -0.79 [-1.35, -0.30] | better | 0.10 | 0.38 | -0.28 [-0.61, 0.00] | tie |
| G2 `perfect`, l 1, n 2,000 | 0.12 | 0.14 | -0.02 [-0.15, +0.12] | tie | 0.08 | 0.06 | +0.02 [-0.07, +0.15] | tie |
| G2 `perfect`, l 1, n 3,000 | 0.10 | 0.05 | +0.05 [-0.08, +0.20] | tie | 0.10 | 0.01 | +0.09 [-0.02, +0.25] | tie |
| G2 `perfect`, l 1, n 4,000 | 0.01 | 0.04 | -0.03 [-0.12, +0.03] | tie | 0.05 | 0.01 | +0.04 [-0.03, +0.15] | tie |
| G2 `perfect`, l 1, n 5,000 | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie | 0.00 | 0.01 | -0.01 [-0.03, 0.00] | tie |
| G2 `perfect`, l 2, n 1,000 | 0.10 | 1.08 | -0.98 [-1.71, -0.39] | lower on seeds 0-19 and 4000-4019 | 0.22 | 1.04 | -0.82 [-1.65, -0.16] | lower on seeds 0-19 only |
| G2 `perfect`, l 2, n 2,000 | 0.05 | 0.25 | -0.20 [-0.49, +0.05] | tie | 0.05 | 0.13 | -0.08 [-0.23, +0.05] | tie |
| G2 `perfect`, l 2, n 3,000 | 0.10 | 0.13 | -0.03 [-0.24, +0.14] | tie | 0.10 | 0.06 | +0.04 [-0.11, +0.21] | tie |
| G2 `perfect`, l 2, n 4,000 | 0.00 | 0.08 | -0.08 [-0.24, 0.00] | tie | 0.05 | 0.04 | +0.01 [-0.12, +0.15] | tie |
| G2 `perfect`, l 2, n 5,000 | 0.00 | 0.08 | -0.08 [-0.24, 0.00] | tie | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie |
| G2 `perfect`, l 3, n 1,000 | 0.08 | 0.78 | -0.70 [-1.16, -0.33] | lower on seeds 0-19 and 4000-4019 | 0.16 | 0.98 | -0.82 [-1.53, -0.27] | lower on seeds 0-19 only |
| G2 `perfect`, l 3, n 2,000 | 0.05 | 0.19 | -0.14 [-0.41, +0.07] | tie | 0.00 | 0.23 | -0.23 [-0.59, 0.00] | tie |
| G2 `perfect`, l 3, n 3,000 | 0.10 | 0.07 | +0.03 [-0.14, +0.19] | tie | 0.10 | 0.06 | +0.04 [-0.13, +0.20] | tie |
| G2 `perfect`, l 3, n 4,000 | 0.00 | 0.06 | -0.06 [-0.18, 0.00] | tie | 0.05 | 0.06 | -0.01 [-0.18, +0.15] | tie |
| G2 `perfect`, l 3, n 5,000 | 0.00 | 0.03 | -0.03 [-0.09, 0.00] | tie | 0.00 | 0.03 | -0.03 [-0.09, 0.00] | tie |
| G3 `shift2`, l 0, n 1,000 | 0.00 | 0.84 | -0.84 [-1.41, -0.34] | lower on seeds 0-19 only | 0.05 | 0.32 | -0.27 [-0.56, -0.01] | lower on seeds 0-19 and 4000-4019 |
| G3 `shift2`, l 3, n 1,000 | 0.20 | 0.98 | -0.78 [-1.38, -0.25] | lower on seeds 0-19 only | 0.28 | 1.56 | -1.28 [-2.34, -0.37] | lower on seeds 0-19 only |
| G3 `perfect_code`, l 0, n 1,000 | 0.10 | 1.04 | -0.94 [-1.55, -0.39] | lower on seeds 0-19 and 4000-4019 | 0.19 | 0.20 | -0.01 [-0.11, +0.11] | tie |
| G3 `perfect_code`, l 3, n 1,000 | 0.08 | 0.78 | -0.70 [-1.16, -0.33] | lower on seeds 0-19 and 4000-4019 | 0.14 | 1.00 | -0.86 [-1.58, -0.29] | lower on seeds 0-19 only |
| G4 `variance`, l 0, n 1,000 | 0.20 | 1.01 | -0.81 [-1.43, -0.27] | lower on seeds 0-19 and 4000-4019 | 0.10 | 0.12 | -0.02 [-0.17, +0.13] | tie |
| G4 `variance`, l 0, n 5,000 | 0.06 | 0.04 | +0.02 [0.00, +0.05] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G4 `variance`, l 3, n 1,000 | 0.20 | 0.87 | -0.67 [-1.08, -0.32] | lower on seeds 0-19 and 4000-4019 | 0.15 | 0.74 | -0.59 [-1.14, -0.14] | lower on seeds 0-19 only |
| G4 `variance`, l 3, n 5,000 | 0.01 | 0.03 | -0.02 [-0.09, +0.03] | tie | 0.00 | 0.02 | -0.02 [-0.06, 0.00] | tie |
| G5 `shift`, l 0, n 1,000 | 0.10 | 0.95 | -0.85 [-1.37, -0.37] | lower on seeds 0-19 and 4000-4019 | 0.10 | 0.13 | -0.03 [-0.17, +0.10] | tie |
| G5 `shift`, l 0, n 5,000 | 0.02 | 0.04 | -0.02 [-0.12, +0.06] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G5 `shift`, l 3, n 1,000 | 0.18 | 1.05 | -0.87 [-1.55, -0.34] | lower on seeds 0-19 only | 0.35 | 1.22 | -0.87 [-1.58, -0.29] | lower on seeds 0-19 only |
| G5 `shift`, l 3, n 5,000 | 0.25 | 0.17 | +0.08 [-0.20, +0.45] | tie | 0.26 | 0.19 | +0.07 [-0.10, +0.31] | tie |
| G6 `weight`, l 0, n 1,000 | 1.00 | 3.06 | -2.06 [-3.12, -1.18] | lower on seeds 0-19 and 4000-4019 | 0.11 | 0.11 | 0.00 [-0.09, +0.13] | tie |
| G6 `weight`, l 0, n 5,000 | 0.08 | 0.39 | -0.31 [-0.62, -0.03] | lower on seeds 0-19 and 4000-4019 | 0.01 | 0.19 | -0.18 [-0.37, -0.04] | lower on seeds 0-19 and 4000-4019 |
| G6 `weight`, l 3, n 1,000 | 1.58 | 3.56 | -1.98 [-2.77, -1.26] | lower on seeds 0-19 and 4000-4019 | 3.15 | 5.67 | -2.52 [-3.81, -1.39] | lower on seeds 0-19 and 4000-4019 |
| G6 `weight`, l 3, n 5,000 | 0.10 | 0.51 | -0.41 [-0.74, -0.11] | lower on seeds 0-19 and 4000-4019 | 0.12 | 1.50 | -1.38 [-2.55, -0.38] | lower on seeds 0-19 and 4000-4019 |

At the paper's level, alpha `1e-5` and 5 starts, against the same library fits:

| Cell | SHD, Andrey | SHD, reference | SHD D [95%] | SHD | Target error, Andrey | Target error, reference | Target error D [95%] | Target error |
|---|---|---|---|---|---|---|---|---|
| G1 `shift`, l 0, n 1,000 | 1.02 | 1.08 | -0.06 [-0.50, +0.34] | tie | 0.29 | 0.20 | +0.09 [0.00, +0.21] | tie |
| G1 `shift`, l 0, n 2,000 | 0.14 | 0.14 | 0.00 [0.00, 0.00] | tie | 0.05 | 0.05 | 0.00 [0.00, 0.00] | tie |
| G1 `shift`, l 0, n 3,000 | 0.31 | 0.06 | +0.25 [0.00, +0.74] | tie | 0.04 | 0.00 | +0.04 [0.00, +0.12] | tie |
| G1 `shift`, l 0, n 4,000 | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G1 `shift`, l 0, n 5,000 | 0.09 | 0.04 | +0.05 [0.00, +0.15] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G1 `shift`, l 1, n 1,000 | 1.41 | 1.18 | +0.23 [-0.07, +0.67] | tie | 0.50 | 0.57 | -0.07 [-0.20, +0.04] | tie |
| G1 `shift`, l 1, n 2,000 | 0.41 | 0.44 | -0.03 [-0.33, +0.26] | tie | 0.16 | 0.18 | -0.02 [-0.08, +0.04] | tie |
| G1 `shift`, l 1, n 3,000 | 0.33 | 0.25 | +0.08 [-0.14, +0.38] | tie | 0.16 | 0.11 | +0.05 [-0.05, +0.20] | tie |
| G1 `shift`, l 1, n 4,000 | 0.00 | 0.08 | -0.08 [-0.20, 0.00] | tie | 0.00 | 0.03 | -0.03 [-0.08, 0.00] | tie |
| G1 `shift`, l 1, n 5,000 | 0.08 | 0.16 | -0.08 [-0.36, +0.12] | tie | 0.02 | 0.07 | -0.05 [-0.18, +0.03] | tie |
| G1 `shift`, l 2, n 1,000 | 1.41 | 1.27 | +0.14 [-0.22, +0.50] | tie | 1.26 | 1.36 | -0.10 [-0.44, +0.18] | tie |
| G1 `shift`, l 2, n 2,000 | 0.49 | 0.42 | +0.07 [-0.06, +0.27] | tie | 0.40 | 0.33 | +0.07 [-0.06, +0.27] | tie |
| G1 `shift`, l 2, n 3,000 | 0.44 | 0.35 | +0.09 [-0.08, +0.34] | tie | 0.31 | 0.28 | +0.03 [-0.08, +0.16] | tie |
| G1 `shift`, l 2, n 4,000 | 0.15 | 0.25 | -0.10 [-0.28, 0.00] | tie | 0.15 | 0.21 | -0.06 [-0.16, 0.00] | tie |
| G1 `shift`, l 2, n 5,000 | 0.20 | 0.23 | -0.03 [-0.24, +0.15] | tie | 0.25 | 0.19 | +0.06 [-0.12, +0.30] | tie |
| G1 `shift`, l 3, n 1,000 | 1.12 | 1.22 | -0.10 [-0.50, +0.29] | tie | 1.29 | 1.29 | 0.00 [-0.27, +0.28] | tie |
| G1 `shift`, l 3, n 2,000 | 0.83 | 0.20 | +0.63 [+0.11, +1.32] | higher on seeds 0-19 only | 0.64 | 0.29 | +0.35 [+0.08, +0.66] | higher on seeds 0-19 only |
| G1 `shift`, l 3, n 3,000 | 0.42 | 0.15 | +0.27 [+0.04, +0.55] | higher on seeds 0-19 only | 0.40 | 0.20 | +0.20 [+0.02, +0.42] | higher on seeds 0-19 only |
| G1 `shift`, l 3, n 4,000 | 0.37 | 0.11 | +0.26 [-0.03, +0.67] | tie | 0.38 | 0.16 | +0.22 [-0.02, +0.48] | tie |
| G1 `shift`, l 3, n 5,000 | 0.39 | 0.10 | +0.29 [-0.03, +0.73] | tie | 0.39 | 0.16 | +0.23 [-0.02, +0.53] | tie |
| G2 `perfect`, l 0, n 1,000 | 0.98 | 1.04 | -0.06 [-0.63, +0.38] | tie | 0.25 | 0.20 | +0.05 [-0.04, +0.15] | tie |
| G2 `perfect`, l 0, n 2,000 | 0.16 | 0.14 | +0.02 [0.00, +0.06] | tie | 0.06 | 0.05 | +0.01 [0.00, +0.03] | tie |
| G2 `perfect`, l 0, n 3,000 | 0.00 | 0.06 | -0.06 [-0.16, 0.00] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G2 `perfect`, l 0, n 4,000 | 0.04 | 0.04 | 0.00 [0.00, 0.00] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G2 `perfect`, l 0, n 5,000 | 0.19 | 0.04 | +0.15 [0.00, +0.41] | tie | 0.02 | 0.00 | +0.02 [0.00, +0.06] | tie |
| G2 `perfect`, l 1, n 1,000 | 1.28 | 0.89 | +0.39 [+0.01, +0.93] | higher on seeds 0-19 only | 0.30 | 0.38 | -0.08 [-0.28, +0.07] | tie |
| G2 `perfect`, l 1, n 2,000 | 0.20 | 0.14 | +0.06 [0.00, +0.17] | tie | 0.07 | 0.06 | +0.01 [0.00, +0.03] | tie |
| G2 `perfect`, l 1, n 3,000 | 0.10 | 0.05 | +0.05 [-0.12, +0.29] | tie | 0.04 | 0.01 | +0.03 [-0.03, +0.12] | tie |
| G2 `perfect`, l 1, n 4,000 | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie | 0.00 | 0.01 | -0.01 [-0.03, 0.00] | tie |
| G2 `perfect`, l 1, n 5,000 | 0.20 | 0.04 | +0.16 [0.00, +0.40] | tie | 0.06 | 0.01 | +0.05 [0.00, +0.14] | tie |
| G2 `perfect`, l 2, n 1,000 | 0.86 | 1.08 | -0.22 [-0.71, +0.17] | tie | 0.96 | 1.04 | -0.08 [-0.58, +0.27] | tie |
| G2 `perfect`, l 2, n 2,000 | 0.10 | 0.25 | -0.15 [-0.35, 0.00] | tie | 0.05 | 0.13 | -0.08 [-0.18, 0.00] | tie |
| G2 `perfect`, l 2, n 3,000 | 0.00 | 0.13 | -0.13 [-0.32, 0.00] | tie | 0.00 | 0.06 | -0.06 [-0.16, 0.00] | tie |
| G2 `perfect`, l 2, n 4,000 | 0.00 | 0.08 | -0.08 [-0.24, 0.00] | tie | 0.00 | 0.04 | -0.04 [-0.12, 0.00] | tie |
| G2 `perfect`, l 2, n 5,000 | 0.08 | 0.08 | 0.00 [0.00, 0.00] | tie | 0.04 | 0.04 | 0.00 [0.00, 0.00] | tie |
| G2 `perfect`, l 3, n 1,000 | 0.88 | 0.78 | +0.10 [-0.15, +0.42] | tie | 0.95 | 0.98 | -0.03 [-0.12, +0.06] | tie |
| G2 `perfect`, l 3, n 2,000 | 0.16 | 0.19 | -0.03 [-0.21, +0.15] | tie | 0.18 | 0.23 | -0.05 [-0.20, +0.06] | tie |
| G2 `perfect`, l 3, n 3,000 | 0.02 | 0.07 | -0.05 [-0.18, +0.03] | tie | 0.00 | 0.06 | -0.06 [-0.18, 0.00] | tie |
| G2 `perfect`, l 3, n 4,000 | 0.03 | 0.06 | -0.03 [-0.09, 0.00] | tie | 0.03 | 0.06 | -0.03 [-0.09, 0.00] | tie |
| G2 `perfect`, l 3, n 5,000 | 0.00 | 0.03 | -0.03 [-0.09, 0.00] | tie | 0.00 | 0.03 | -0.03 [-0.09, 0.00] | tie |
| G3 `shift2`, l 0, n 1,000 | 0.76 | 0.84 | -0.08 [-0.52, +0.29] | tie | 0.34 | 0.32 | +0.02 [-0.07, +0.13] | tie |
| G3 `shift2`, l 3, n 1,000 | 1.06 | 0.98 | +0.08 [-0.18, +0.34] | tie | 1.68 | 1.56 | +0.12 [-0.18, +0.45] | tie |
| G3 `perfect_code`, l 0, n 1,000 | 1.17 | 1.04 | +0.13 [-0.32, +0.60] | tie | 0.19 | 0.20 | -0.01 [-0.06, +0.05] | tie |
| G3 `perfect_code`, l 3, n 1,000 | 0.60 | 0.78 | -0.18 [-0.36, -0.01] | lower, no rerun | 0.95 | 1.00 | -0.05 [-0.20, +0.12] | tie |
| G4 `variance`, l 0, n 1,000 | 0.90 | 1.01 | -0.11 [-0.67, +0.32] | tie | 0.11 | 0.12 | -0.01 [-0.06, +0.03] | tie |
| G4 `variance`, l 0, n 5,000 | 0.05 | 0.04 | +0.01 [0.00, +0.03] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G4 `variance`, l 3, n 1,000 | 0.84 | 0.87 | -0.03 [-0.25, +0.20] | tie | 0.73 | 0.74 | -0.01 [-0.12, +0.09] | tie |
| G4 `variance`, l 3, n 5,000 | 0.03 | 0.03 | 0.00 [0.00, 0.00] | tie | 0.02 | 0.02 | 0.00 [0.00, 0.00] | tie |
| G5 `shift`, l 0, n 1,000 | 0.76 | 0.95 | -0.19 [-0.59, +0.13] | tie | 0.21 | 0.13 | +0.08 [-0.02, +0.22] | tie |
| G5 `shift`, l 0, n 5,000 | 0.09 | 0.04 | +0.05 [0.00, +0.15] | tie | 0.00 | 0.00 | 0.00 [0.00, 0.00] | tie |
| G5 `shift`, l 3, n 1,000 | 1.10 | 1.05 | +0.05 [-0.34, +0.44] | tie | 1.29 | 1.22 | +0.07 [-0.24, +0.41] | tie |
| G5 `shift`, l 3, n 5,000 | 0.34 | 0.17 | +0.17 [-0.05, +0.51] | tie | 0.32 | 0.19 | +0.13 [-0.05, +0.38] | tie |
| G6 `weight`, l 0, n 1,000 | 3.13 | 3.06 | +0.07 [-0.47, +0.60] | tie | 0.03 | 0.11 | -0.08 [-0.20, 0.00] | tie |
| G6 `weight`, l 0, n 5,000 | 0.23 | 0.39 | -0.16 [-0.48, +0.12] | tie | 0.06 | 0.19 | -0.13 [-0.33, +0.01] | tie |
| G6 `weight`, l 3, n 1,000 | 3.63 | 3.56 | +0.07 [-0.49, +0.71] | tie | 5.11 | 5.67 | -0.56 [-1.07, -0.08] | lower, no rerun |
| G6 `weight`, l 3, n 5,000 | 0.49 | 0.51 | -0.02 [-0.19, +0.14] | tie | 0.94 | 1.50 | -0.56 [-1.28, -0.02] | lower, no rerun |

### Replication

Every comparison whose interval lay above or below 0 on seeds 0 to 19, with its interval on fresh
seeds. The 11 cells with a comparison above 0 in the first comparison (see [Choosing the
defaults](#choosing-the-defaults)) reran on seeds 2000 to 2019; they hold every comparison above 0
here. At Andrey's defaults every comparison also has seeds 4000 to 4019, where the same settings ran
on an earlier build of the same search in the run that chose the default; a comparison with both
shows both. That run chose the setting with the most comparisons below 0 on seeds 4000 to 4019, so
those seeds favor below 0 for the default and decide no verdict: 17 comparisons were lower on seeds
0 to 19 and on seeds 4000 to 4019, and the one better comparison, G2 `perfect` l 1 SHD, was lower on
seeds 2000 to 2019 and tied on 4000 to 4019. The 3 comparisons at the paper's level that were lower
on seeds 0 to 19 (G3 `perfect_code` l 3 SHD, and G6 `weight` l 3 target error at 1,000 and 5,000
rows) are in cells that did not rerun.

| Settings | Cell | Error | D, seeds 0-19 | D, fresh seeds | Fresh seeds | Result |
|---|---|---|---|---|---|---|
| defaults | G1 `shift`, l 0, n 1,000 | SHD | -1.02 [-1.63, -0.47] | -0.49 [-0.94, -0.08] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G1 `shift`, l 1, n 1,000 | SHD | -0.90 [-1.42, -0.42] | -0.57 [-1.06, -0.14] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G1 `shift`, l 1, n 1,000 | target error | -0.32 [-0.57, -0.07] | -0.09 [-0.41, +0.23] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G1 `shift`, l 2, n 1,000 | SHD | -0.80 [-1.35, -0.26] | -0.53 [-1.15, 0.00] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G1 `shift`, l 2, n 1,000 | target error | -0.87 [-1.80, -0.15] | -0.13 [-0.53, +0.24] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G1 `shift`, l 3, n 1,000 | SHD | -1.04 [-1.79, -0.40] | -0.27 [-0.68, +0.13] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G1 `shift`, l 3, n 1,000 | target error | -0.94 [-1.65, -0.35] | +0.04 [-0.56, +0.66] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G1 `shift`, l 3, n 3,000 | target error | +0.21 [+0.01, +0.45] | -0.15 [-0.53, +0.15]; +0.16 [0.00, +0.40] | 2000-2019; 4000-4019 | higher on seeds 0-19 only |
| defaults | G2 `perfect`, l 0, n 1,000 | SHD | -0.94 [-1.57, -0.37] | -0.27 [-0.78, +0.23] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G2 `perfect`, l 1, n 1,000 | SHD | -0.79 [-1.35, -0.30] | -0.90 [-1.71, -0.10]; -0.40 [-0.90, +0.07] | 2000-2019; 4000-4019 | better |
| defaults | G2 `perfect`, l 2, n 1,000 | SHD | -0.98 [-1.71, -0.39] | -0.66 [-1.21, -0.18] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G2 `perfect`, l 2, n 1,000 | target error | -0.82 [-1.65, -0.16] | +0.04 [-0.26, +0.36] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G2 `perfect`, l 3, n 1,000 | SHD | -0.70 [-1.16, -0.33] | -0.40 [-0.76, -0.08] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G2 `perfect`, l 3, n 1,000 | target error | -0.82 [-1.53, -0.27] | -0.15 [-0.66, +0.39] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G3 `shift2`, l 0, n 1,000 | SHD | -0.84 [-1.41, -0.34] | -0.34 [-0.86, +0.17] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G3 `shift2`, l 0, n 1,000 | target error | -0.27 [-0.56, -0.01] | -0.13 [-0.26, -0.02] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G3 `shift2`, l 3, n 1,000 | SHD | -0.78 [-1.38, -0.25] | -0.31 [-0.65, +0.03] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G3 `shift2`, l 3, n 1,000 | target error | -1.28 [-2.34, -0.37] | -0.37 [-0.86, +0.09] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G3 `perfect_code`, l 0, n 1,000 | SHD | -0.94 [-1.55, -0.39] | -0.47 [-0.95, -0.05] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G3 `perfect_code`, l 3, n 1,000 | SHD | -0.70 [-1.16, -0.33] | -0.48 [-0.87, -0.11] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G3 `perfect_code`, l 3, n 1,000 | target error | -0.86 [-1.58, -0.29] | -0.15 [-0.68, +0.43] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G4 `variance`, l 0, n 1,000 | SHD | -0.81 [-1.43, -0.27] | -0.57 [-1.02, -0.17] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G4 `variance`, l 3, n 1,000 | SHD | -0.67 [-1.08, -0.32] | -0.43 [-0.84, -0.04] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G4 `variance`, l 3, n 1,000 | target error | -0.59 [-1.14, -0.14] | -0.02 [-0.41, +0.43] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G5 `shift`, l 0, n 1,000 | SHD | -0.85 [-1.37, -0.37] | -0.47 [-0.93, -0.07] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G5 `shift`, l 3, n 1,000 | SHD | -0.87 [-1.55, -0.34] | -0.27 [-0.82, +0.25] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G5 `shift`, l 3, n 1,000 | target error | -0.87 [-1.58, -0.29] | +0.08 [-0.55, +0.72] | 4000-4019 | lower on seeds 0-19 only |
| defaults | G6 `weight`, l 0, n 1,000 | SHD | -2.06 [-3.12, -1.18] | -1.42 [-2.26, -0.61] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G6 `weight`, l 0, n 5,000 | SHD | -0.31 [-0.62, -0.03] | -0.50 [-0.80, -0.25] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G6 `weight`, l 0, n 5,000 | target error | -0.18 [-0.37, -0.04] | -0.23 [-0.38, -0.09] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G6 `weight`, l 3, n 1,000 | SHD | -1.98 [-2.77, -1.26] | -1.76 [-2.55, -1.05] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G6 `weight`, l 3, n 1,000 | target error | -2.52 [-3.81, -1.39] | -2.63 [-3.95, -1.39] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G6 `weight`, l 3, n 5,000 | SHD | -0.41 [-0.74, -0.11] | -0.51 [-0.91, -0.20] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| defaults | G6 `weight`, l 3, n 5,000 | target error | -1.38 [-2.55, -0.38] | -1.15 [-2.16, -0.33] | 4000-4019 | lower on seeds 0-19 and 4000-4019 |
| paper's level | G1 `shift`, l 3, n 2,000 | SHD | +0.63 [+0.11, +1.32] | -0.06 [-0.35, +0.22] | 2000-2019 | higher on seeds 0-19 only |
| paper's level | G1 `shift`, l 3, n 2,000 | target error | +0.35 [+0.08, +0.66] | -0.13 [-0.47, +0.24] | 2000-2019 | higher on seeds 0-19 only |
| paper's level | G1 `shift`, l 3, n 3,000 | SHD | +0.27 [+0.04, +0.55] | -0.21 [-0.55, +0.09] | 2000-2019 | higher on seeds 0-19 only |
| paper's level | G1 `shift`, l 3, n 3,000 | target error | +0.20 [+0.02, +0.42] | -0.25 [-0.84, +0.18] | 2000-2019 | higher on seeds 0-19 only |
| paper's level | G2 `perfect`, l 1, n 1,000 | SHD | +0.39 [+0.01, +0.93] | -0.20 [-0.60, +0.29] | 2000-2019 | higher on seeds 0-19 only |

### Floors and coverage

In all 56 cells the library's SHD is below the empty graph's (the interval of its SHD minus the
empty graph's has its upper end at most -10.62), and in the 40 cells where the given targets miss
some (`l` of 1 or more, and G5), its target error is below the given targets' alone (upper end at
most -4.01), so every floor applies. Andrey passes each: at both settings its SHD is below the empty
graph's, with the interval's upper end at most -12.75 at the defaults and -10.39 at the paper's
level, and its target error is below the given targets' alone, with the upper end at most -4.45 and
-4.11. Every draw of both packages finished within its cap, on all 1,120 datasets of seeds 0 to 19
and on every rerun.

### Choosing the defaults

Andrey's defaults were chosen before the comparison above, in four steps, each on seeds of its
own.

1. **Alpha and objective, then starts (seeds 1000 to 1019).** Andrey alone, against itself at the
   paper's alpha `1e-5` with the sum and 5 starts, on 15 cells: `shift` and `perfect` at `l` 0
   and 3 with 1,000, 3,000, and 5,000 rows per setting, and three out-of-claim cells (300 rows; 50
   variables). Alpha `1e-3` with the sum had lower SHD in 7 cells and higher in 2, lower target
   error in 3 and higher in none; with the lexicographic objective, lower SHD in 6 and higher in 2.
   Alpha `1e-2` had higher SHD in 9 or 10 cells and higher target error in 11; alpha `1e-5` with
   the lexicographic objective had higher SHD in 1 cell and higher target error in 2. At alpha
   `1e-3`, 10 starts had lower SHD than 5 in 4 cells (by 0.17 to 0.42) and higher in none, lower
   target error in 2 and higher in none, at a median of 0.24 s per fit against 0.21 s; 1 start had
   higher SHD in 10 cells.
2. **First comparison (seeds 0 to 19 and 2000 to 2019).** With alpha `1e-3`, the sum, and 10
   starts, 15 of the 224 comparisons lay above 0 on seeds 0 to 19, 5 at the paper's level and 10
   at those defaults; none of the 15 lay above 0 on the rerun.
3. **Trace.** The 15 had two causes, below.
4. **Confirmation (seeds 4000 to 4019, rerun on 5000 to 5019).** Six settings, all with 10 starts,
   against the library at the paper's level over the 112 comparisons, by a rule written before the
   run: a setting is eligible when none of its comparisons lies above 0 on both seed ranges, and
   the eligible setting with the most comparisons below 0 wins.

| Alpha | Objective | Above 0, seeds 4000-4019 | Above 0 again, seeds 5000-5019 | Below 0, seeds 4000-4019 |
|---|---|---|---|---|
| `1e-5` | sum | 0 | 0 | 7 |
| `1e-5` | lexicographic | 2 | 0 | 5 |
| `1e-3` | sum | 3 | 0 | 15 |
| `1e-3` | lexicographic | 3 | 0 | 15 |
| one over the observational rows | sum | 0 | 0 | 20 |
| one over the observational rows | lexicographic | 0 | 0 | 20 |

Every setting was eligible; the two at one over the observational rows tied on 20, and the rule's
tie order, the paper's sum first, chose alpha one over the observational rows with the sum and 10
starts. Its 20 comparisons below 0 lay 0.13 to 2.63 errors per dataset below, 11 of them SHD at
1,000 rows per setting. Against alpha `1e-3` with the sum, the chosen setting's SHD lay below 0 in
4 comparisons, all at 5,000 rows, by 0.2 to 0.65, and above 0 in none.

**The objective.** The 5 comparisons at the paper's level (G1 `shift` l 3 at 2,000 and 3,000 rows,
SHD and target error, and G2 `perfect` l 1 at 1,000 rows, SHD) come from the objective. Andrey
minimizes the number of edges plus the number of targets, as the paper's text defines the score
(Section 4, Algorithm 1). The library by default compares the number of edges first and the
targets that test variant second, and the authors' experiments ran that default. Under the sum, one
extra edge can replace two targets whose invariance test missed, so at a finite sample the sum's
minimum can be a wrong DAG; in the population both objectives are minimized by the true class.
With the lexicographic objective, Andrey at the paper's level lay above 0 in none of the 112
comparisons (below 0 in 2), and Andrey's two objectives tied each other in all 112. Andrey keeps
the paper's sum: the difference is small, and none of the 5 held on the rerun.

**The level.** The 10 comparisons at alpha `1e-3` all have 3,000 to 5,000 rows per setting, where
the paper's `1e-5` removes edges that `1e-3` keeps. Run at the same `1e-3`, the library was below
Andrey in none of the 10. The library itself at `1e-3` against `1e-5` lay above 0 in 30 of the 112
comparisons and below 0 in 28. A fixed level suits one sample size; one over the observational
rows falls as they grow, and it is the library's own default level for its partial-correlation
test.

### The library at Andrey's defaults

Andrey at its defaults against the library at the same alpha, one over the observational rows, and
10 runs: of the 112 comparisons in claim, 107 tie and 5 lie below 0, all in G6 (`weight`): SHD at
`l` 0 with 1,000 rows (1.00 against 1.76) and 5,000 rows (0.08 against 0.37), and target error at
`l` 0 with 1,000 and 5,000 rows and at `l` 3 with 5,000 rows. None lies above 0.

### Other methods

Andrey at its defaults against the context methods, on SHD over the cells each ran in (and target
error for JCI-GSP):

| Method | Cells | Below 0 | Tie | Above 0 |
|---|---|---|---|---|
| The library's IGSP, designed targets | G1, G2: 40 | 32 | 8 | 0 |
| The library's JCI-GSP (call fixes above) | G1, G2, G5: 44 | 44; target error 43 | 0; target error 1 | 0 |
| Andrey's `gies`, designed targets | G1, G2, G4, G6: 48 | 41 | 7 | 0 |
| Andrey's `cdnod` | G1, G2: 40 | 40 | 0 | 0 |

In the 30 cells of G1 and G2 with off-targets, the paper reports UT-IGSP ahead of GIES and IGSP
given the designed targets. At the paper's level, Andrey's SHD lay below both in all 30 cells; on
the SHD of its DAG read under the true targets, below `gies` in 30 and below IGSP in 26; on the
skeleton, below `gies` in 30 and below IGSP in 6.

### Out of the claim

Means at Andrey's defaults against the library at the paper's level, one draw per dataset, 20
datasets per cell (10 in O3, O4, and O7). Every fit finished within its cap.

| Cell | Setting | SHD, Andrey | SHD, reference | SHD, empty | SHD D [95%] | Target error, Andrey | Target error, reference | Target floor | Target error D [95%] |
|---|---|---|---|---|---|---|---|---|---|
| O1, l 0, n 100 | `shift`, fewer rows | 8.35 | 12.95 | 15.00 | -4.60 [-5.90, -3.35] | 1.50 | 0.25 | 0.00 | +1.25 [+0.70, +1.85] |
| O1, l 0, n 300 | `shift`, fewer rows | 2.85 | 6.30 | 15.00 | -3.45 [-4.90, -2.15] | 1.20 | 0.55 | 0.00 | +0.65 [0.00, +1.30] |
| O1, l 3, n 100 | `shift`, fewer rows | 7.40 | 12.55 | 15.00 | -5.15 [-6.40, -3.80] | 3.60 | 4.75 | 15.00 | -1.15 [-3.00, +0.65] |
| O1, l 3, n 300 | `shift`, fewer rows | 2.55 | 5.05 | 15.00 | -2.50 [-3.25, -1.80] | 2.55 | 2.15 | 15.00 | +0.40 [-0.50, +1.30] |
| O2 shift 0.25, l 0, n 1,000 | weaker shift, `soft(shift=...)` | 1.35 | 3.50 | 15.00 | -2.15 [-3.25, -1.15] | 0.05 | 0.05 | 0.00 | 0.00 [-0.15, +0.15] |
| O2 shift 0.25, l 3, n 1,000 | weaker shift, `soft(shift=...)` | 4.45 | 4.40 | 15.00 | +0.05 [-1.30, +1.45] | 3.30 | 5.40 | 15.00 | -2.10 [-3.25, -0.85] |
| O2 shift 0.5, l 0, n 1,000 | weaker shift, `soft(shift=...)` | 0.75 | 2.50 | 15.00 | -1.75 [-3.10, -0.65] | 0.20 | 0.20 | 0.00 | 0.00 [-0.20, +0.20] |
| O2 shift 0.5, l 3, n 1,000 | weaker shift, `soft(shift=...)` | 1.10 | 2.10 | 15.00 | -1.00 [-2.05, -0.05] | 0.65 | 1.35 | 15.00 | -0.70 [-1.60, +0.05] |
| O3, 50 variables, 3 settings, l 0, n 500 | `perfect`, mean degree 5, 3 settings | 130.20 | 130.90 | 125.00 | -0.70 [-10.60, +11.20] | 6.30 | 5.30 | 0.00 | +1.00 [-0.60, +2.70] |
| O3, 50 variables, 3 settings, l 0, n 2,000 | `perfect`, mean degree 5, 3 settings | 123.90 | 126.20 | 125.00 | -2.30 [-9.00, +3.90] | 6.80 | 7.70 | 0.00 | -0.90 [-2.10, +0.20] |
| O4, 50 variables, l 3, n 2,000 | `shift`, more variables | 1.50 | 2.50 | 38.00 | -1.00 [-3.20, +0.60] | 0.50 | 0.90 | 15.00 | -0.40 [-1.20, +0.20] |
| O4, 100 variables, l 3, n 2,000 | `shift`, more variables | 4.80 | 5.30 | 75.00 | -0.50 [-2.10, +1.20] | 0.70 | 1.20 | 15.00 | -0.50 [-1.00, 0.00] |
| O5, l 0, n 1,000 | additive nonlinear, Gaussian tests | 1.55 | 1.30 | 15.00 | +0.25 [-0.55, +1.05] | 1.85 | 0.75 | 0.00 | +1.10 [+0.70, +1.55] |
| O5, l 3, n 1,000 | additive nonlinear, Gaussian tests | 1.40 | 1.50 | 15.00 | -0.10 [-1.20, +0.95] | 4.85 | 2.90 | 15.00 | +1.95 [+0.75, +3.05] |
| O6, 10 variables, l 0, n 200 | additive nonlinear, kernel tests | 5.30 | 3.80 | 8.00 | +1.50 [0.00, +3.05] | 1.95 | 1.30 | 0.00 | +0.65 [-0.50, +1.75] |
| O6, 10 variables, l 0, n 400 | additive nonlinear, kernel tests | 2.40 | 1.75 | 8.00 | +0.65 [-0.30, +1.50] | 0.70 | 0.70 | 0.00 | 0.00 [-0.45, +0.45] |
| O6, 10 variables, l 3, n 200 | additive nonlinear, kernel tests | 5.50 | 4.65 | 8.00 | +0.85 [-0.20, +1.85] | 3.95 | 5.60 | 15.00 | -1.65 [-3.40, 0.00] |
| O6, 10 variables, l 3, n 400 | additive nonlinear, kernel tests | 2.35 | 2.70 | 8.00 | -0.35 [-1.35, +0.60] | 2.85 | 4.05 | 15.00 | -1.20 [-2.65, +0.10] |
| O7, 10 variables, l 0, n 1,000 | categorical, G-test; Andrey alone | 5.40 |  | 10.00 |  | 0.50 |  | 0.00 |  |
| O7, 10 variables, l 0, n 5,000 | categorical, G-test; Andrey alone | 1.60 |  | 10.00 |  | 0.20 |  | 0.00 |  |
| O8, 1 setting, l 0, n 1,000 | `shift`, fewer settings | 0.05 | 0.70 | 15.00 | -0.65 [-1.25, -0.15] | 0.00 | 0.05 | 0.00 | -0.05 [-0.15, 0.00] |
| O8, 1 setting, l 3, n 1,000 | `shift`, fewer settings | 0.30 | 0.95 | 15.00 | -0.65 [-1.20, -0.20] | 0.15 | 0.20 | 3.00 | -0.05 [-0.15, 0.00] |
| O8, 2 settings, l 0, n 1,000 | `shift`, fewer settings | 0.05 | 1.20 | 15.00 | -1.15 [-2.00, -0.40] | 0.05 | 0.10 | 0.00 | -0.05 [-0.30, +0.15] |
| O8, 2 settings, l 3, n 1,000 | `shift`, fewer settings | 0.20 | 1.00 | 15.00 | -0.80 [-1.45, -0.25] | 0.20 | 0.25 | 6.00 | -0.05 [-0.30, +0.15] |
| O9, l 0, 2,000 observational, 300 per setting | `shift`, unequal sizes | 0.05 | 0.25 | 15.00 | -0.20 [-0.55, +0.05] | 0.20 | 0.10 | 0.00 | +0.10 [-0.20, +0.35] |
| O9, l 3, 2,000 observational, 300 per setting | `shift`, unequal sizes | 0.60 | 0.55 | 15.00 | +0.05 [-0.50, +0.65] | 0.55 | 0.65 | 15.00 | -0.10 [-0.65, +0.40] |

- **Fewer rows (O1).** With 100 and 300 rows per setting Andrey's SHD lay 2.5 to 5.15 below the
  library's, all below the empty graph's 15. With no off-targets its target error was above the
  library's at 100 rows (1.50 against 0.25): it names false targets there.
- **Weaker shifts (O2).** SHD below or tied with the library's, target error tied or below.
- **Dense graph (O3).** At 50 variables, mean degree 5, and 3 settings, neither package beats the
  empty graph's 125: Andrey 130.2 and 123.9, the library 130.9 and 126.2, at 500 and 2,000 rows per
  setting, both intervals against the empty graph containing 0.
- **More variables (O4).** At 50 and 100 variables with 2,000 rows per setting and 3 off-targets,
  SHD 1.5 and 4.8 against the empty graph's 38 and 75, tied with the library.
- **Nonlinear data, Gaussian tests (O5).** SHD tied with the library's and far below the empty
  graph's; target error above the library's, 1.85 against 0.75 and 4.85 against 2.90.
- **Kernel tests (O6).** On 10 variables with additive nonlinear effects, Andrey's KCI test and
  the library's GAM-residual HSIC tie on both errors at 200 and 400 rows per setting, and both
  beat the empty graph's 8. Against the library's kernel tests at Andrey's alpha, Andrey's SHD lay
  above 0 in all 4 cells, by 1.2 to 2.3.
- **Categorical data (O7).** Andrey alone, with the G-test: SHD 5.4 at 1,000 rows per setting and
  1.6 at 5,000, against the empty graph's 10.
- **Fewer settings (O8) and unequal sizes (O9).** With 1 or 2 settings Andrey's SHD lay below the
  library's, by 0.65 to 1.15; with 2,000 observational rows and settings of 300, both tie.

The paper's Sachs data, the authors' own data generator, and hidden common causes were not
measured.

### Speed

Andrey's median fit on 20 variables took 0.24 s at its defaults and 0.20 s at the paper's level,
against the library's 0.14 s. Per dataset, Andrey's time over the library's had a median of 1.68 at
the defaults (quartiles 1.47 and 1.93) and 1.44 at the paper's level. The two sets of fits ran in
separate jobs on the same processor model; in the earlier run, where they ran back to back, the
ratio at the paper's level was 1.50. At 100 variables Andrey took a median 15.2 s against the
library's 16.8 s, at 50 variables 1.7 s against 0.9 s, and with kernel tests at 400 rows per
setting 28 to 30 s against 33 to 34 s.

### Where the differences do not matter

On the 16 datasets behind the 5 comparisons at the paper's level, the trace replayed each
difference of the Settings table but the objective. The CI tests agreed on all 68,518 of Andrey's
and 1,968 of the library's queries; the library's one-sided start threshold changed the start
I-MAP by one edge in 34 of 800 start DAGs; the invariance tests disagreed on 30 of 20,821 queries,
by the library's split of alpha, and putting the library's test in Andrey's search changed none of
300 fits over the three cells' 20 seeds; and no known target tested invariant on a final DAG. From
the same start order, Andrey's depth-first search and the library's loop reached the same
objective.

## References

- Squires, C., Wang, Y., Uhler, C. (2020). Permutation-based causal structure learning with
  unknown intervention targets. UAI, PMLR 124, 1039-1048.
- Wang, Y., Solus, L., Yang, K., Uhler, C. (2017). Permutation-based causal inference algorithms
  with interventions. NeurIPS 30.
- Mooij, J. M., Magliacane, S., Claassen, T. (2020). Joint causal inference from multiple
  contexts. JMLR 21(99), 1-108.
