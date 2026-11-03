# ICP

`icp_cpu.py` compares Andrey's `icp` with `ICP()` of R's InvariantCausalPrediction 0.8, the
software of Peters, Buhlmann and Meinshausen (2016), on linear Gaussian data from several
environments. Every fit gets the same data, target, and level on both sides: `alpha=0.05`, set
explicitly, since R's default is 0.01. Load R as [The R lane](../README.md#the-r-lane) describes,
with InvariantCausalPrediction in the library; R runs in `Rscript` through
`andrey_bench/icp_worker.R`, and Andrey in the driver's processes, one thread each.

## Settings

Each R fit runs `ICP()` without the package's shortcuts, so that both packages test the same sets:

- `selection = "all"`: every subset of the other variables, with no screening. Under a cap on the
  set size, Andrey's `max_parents`, `selection = "lasso"` with `maxNoVariables` equal to the number
  of variables keeps every variable and tests the sets of at most `maxNoVariablesSimult`; R then
  leaves the empty set out of its accepted sets, so the worker reports its p-value apart.
- `maxNoObs` above every environment's rows, so the exact test fits on every row.
- `stopIfEmpty = FALSE`, the full search.

## Where the two differ by design

Andrey follows the paper where R departs from it. Each difference changes p-values by a known
factor, and the comparison checks Andrey against R's p-values with that factor applied:

- **The approximate test's correction.** R multiplies the smallest environment's p-value by
  `K - 1` (`getpval.R`, line 42); the paper's Method II corrects over the `K` environments
  (Section 3.1). Andrey multiplies by `K`, and by 1 with two environments, where the two
  comparisons are one test. With three or more environments, Andrey's p-values are R's times
  `K / (K - 1)`.
- **The exact test's residual variance.** R's is `var(r) n_out / (n_out - p)` (`getpval.R`, line
  65), larger than the residual variance of the regression, `RSS / (n_out - p)`, by
  `n_out / (n_out - 1)`. With the latter, Chow's statistic follows the F distribution of eq. 17,
  so Andrey uses it.
- **The intervals.** R widens each accepted set's estimates by `qnorm(1 - alpha / 4)` standard
  errors (`ICP.R`, lines 165-170); Andrey by `t_{1 - alpha / (2 |S|), n - |S| - 1}`, the paper's
  rectangle (Section 3.1, step 3).

`andrey_bench/icp_probes.R` shows each on a small seeded dataset, with two defects of R that the
settings above avoid: `stopIfEmpty = TRUE` does not stop when the accepted sets stop overlapping,
since `ICP.R` line 158 tests `length(intersect)`, the base function; and with more rows in an
environment than `maxNoObs`, the exact test's subsample pairs each prediction error with another
row's covariance (`getpval.R`, lines 60-63), so its p-value differs from the same draws' with the
rows matched.

## Fixed cases

`--fixed` runs F1-F9, small seeded datasets that reach each branch: two, three, and four
environments, both tests, few rows, a target without parents, two near-copies of a parent, a
noise intervention on the target, a cap of 2 on the set size, and `alpha=0.01`. R reports every
tested set's p-value from its own `getpval`, and Chow's p-value with each residual variance. For
each case and test, the comparison checks:

1. that the accepted sets of R's `ICP()` are the sets whose `getpval` p-value exceeds the level,
   and that Chow's p-value with R's residual variance is R's exact one, within `1e-10`;
2. that Andrey's accepted sets, parents, per-variable p-values, and largest set p-value are those
   R's per-set p-values give under Andrey's rules, the p-values within `1e-9` relative and `1e-14`
   absolute: Andrey's regressions read Gram matrices, R's a QR decomposition, and R rounds
   p-values below about `1e-15` (`1 - pf`);
3. that R's intervals, rebuilt from its per-set coefficients and standard errors with its
   quantile, are its `ConfInt`, and Andrey's, rebuilt with the paper's quantile where the two
   accept the same sets, are Andrey's, within `1e-9`;
4. whether Andrey and R accept the same sets and return the same parents, and, where they do not,
   the sets whose two p-values lie on either side of the level.

```shell
$ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --fixed --out outputs/icp-fixed
```

`fixed.json` holds one row per case and test.

## Campaign

The cells follow the simulation of Peters, Buhlmann and Meinshausen (2016, Section 7.1 and
Appendix G): two environments, observational and interventional, with 100 to 500 rows each; a
random DAG with mean degree 1 to 4; random weights and noise variances; and noise interventions
that multiply the noise of one variable, or of a share of them, by a uniform draw per row, with new
weights into those variables a third of the time. A scenario draws the parameters and the SEM, and
its datasets draw the noise. Where Appendix G is silent, the driver draws one SEM per scenario,
the multipliers per row, the set size `max(1, round(theta m))` of the `m` variables it draws from,
and new weights with a random sign, and never subsamples.

Cell C's datasets come from `andrey.data`: `SCM.sample` with `envs=`, each environment's
do-intervention as `hard(2.0, sd=0.2)`, on the scenario's DAG (`FixedDAG`), weights, and noise
scales, in the data's own units (`scale="raw"`), and `icp` takes the sample's `env` and `targets`.
The weights and noise scales are pinned by small subclasses of the linear functional and the
Gaussian noise, since a seed of `SCM.sample` redraws both. On four scenarios of 300 datasets each,
the two generators' per-environment means and covariances, and Andrey's false selections and
successes, agree within sampling error. The other cells keep the driver's generator, since
`envs=` cannot express their settings exactly: Appendix G multiplies the noise of an intervened
variable by a new uniform draw on every row, where `soft(noise_scale=...)` multiplies it by one
number per environment; it draws each new weight into an intervened variable on its own, where
`soft(weight_scale=...)` multiplies them all by one number; and a scenario fixes its SEM across
datasets. E-hidden places one hidden parent of the target and of one other variable, where
`latents=` places the hidden variables itself, and E-nonlinear makes one term of the target's
equation nonlinear, where a functional applies to every equation. The speed cells use cell A's
generator.

| Cell | Claim | Data | Scenarios x datasets, Andrey / R |
|---|---|---|---|
| A | in | Appendix G with 5 to 12 nodes, no intervention on the target | 100 x 1000 / 100 x 50 |
| B | out | A's scenarios with Appendix G's interventions, which can hit the target | 100 x 200 / none |
| C | in | observation and 2 or 3 single-variable do-interventions, 6 or 10 nodes, 200 rows each | 20 x 200 / 20 x 50 |
| D | in | Appendix G with 16 or 21 nodes, `max_parents=3`, targets with at most 3 parents | 20 x 200 / 10 x 20 |
| D40 | in | as D with 40 nodes | 20 x 50 / none |
| E-hidden | out | A's data with a hidden parent of the target and of one other variable | 20 x 200 / none |
| E-nonlinear | out | A's data with `tanh` of one parent in the target's equation | 20 x 200 / none |
| speed | in | 9, 12, and 15 nodes with 500 + 500 rows; 12 nodes with 5000 + 5000 (R's approximate test) | 5 / 5 |

Both tests run on every dataset. Each fit is scored against the target's parents: a false
selection is an estimate that is not a subset of them, and its share is the FWER; a success is an
estimate equal to them, the paper's measure; the share of the parents found averages over the fits
whose target has parents. A fit with no accepted set is an empty estimate, counted apart as
rejected. With 1,000 datasets, a scenario's FWER near `alpha` has a standard error of about 0.007.

```shell
$ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --out outputs/icp-cpu --workers 4
$ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --out outputs/icp-cpu --workers 4 --resume
$ANDREY_BENCH_PYTHON benchmarks/icp_cpu.py --out outputs/icp-cpu --summary
```

Each scenario writes `tasks/<cell>/<scenario>.json` when it finishes, so `--resume` continues a
run where it stopped. `--scenarios`, `--datasets`, and `--r-datasets` rerun chosen scenarios of a
cell with R fitting the first datasets of each, as for cell B below, and `--r-defaults` runs R at
its package defaults but for the level and the test. `--trace` checks each paired fit whose
accepted sets differ. `summary.json` gives each cell's FWER with its 95% Clopper-Pearson interval,
the scenarios whose FWER is above `alpha` with a one-sided binomial p-value below `0.05` over the
number of scenarios, the success and found shares, the rejected share, the median fit time, and,
for the paired fits, how many accepted the same sets and returned the same parents, and the median
ratio of R's fit time to Andrey's.

## Results

One run of every cell, with Andrey's development build for 0.14.0 and InvariantCausalPrediction
0.8 on R 4.6.1, at `alpha=0.05`, on an 8-core Intel Core i7-7820X with 4 worker processes, each
fit on one thread. The numbers come from the run's `summary.json` and, for the traced differences,
`trace.json`.

### Agreement with R

With two environments, the approximate test accepted the same sets as R on every paired fit, 5,220
of 5,220: 5,000 in cell A, 200 in cell D, and 20 in the speed cells. `--trace` traced every other
difference, 2,112 fits, to one of the two causes in
[Where the two differ by design](#where-the-two-differ-by-design), and found none it could not
explain: rebuilt from the rows, R's rules give R's accepted sets and Andrey's give Andrey's on
every one. The settings keep `stopIfEmpty` and the exact test's
subsample out of every fit.

| Cell | Test | Environments | Paired fits | Same accepted sets | Same parents | Cause of the differences | Median time ratio, R / Andrey |
|---|---|---|---|---|---|---|---|
| A | approximate | 2 | 5,000 | 5,000 | 5,000 | | 50 |
| A | exact | 2 | 5,000 | 3,568 | 4,964 | residual variance | 169 |
| C | approximate | 3 or 4 | 1,000 | 584 | 975 | correction | 41 |
| C | exact | 3 or 4 | 1,000 | 813 | 991 | residual variance | 96 |
| D | approximate | 2 | 200 | 200 | 200 | | 464 |
| D | exact | 2 | 200 | 131 | 200 | residual variance | 760 |

R's exact p-values are the larger ones, so on every fit where the exact test differs, R accepts
every set Andrey accepts and more. With three or four environments, R's approximate p-values are
the smaller ones, so there Andrey accepts every set R accepts and more.

### The claim

In every cell within the method's claim, the pooled share of fits whose estimate holds a variable
outside the true parents, the FWER, stays well below `alpha=0.05`, and no scenario's FWER is above
`alpha` beyond sampling error: none has a one-sided binomial p-value below `0.05` over the number
of scenarios.

| Cell | Test | Fits | FWER | 95% interval | Largest scenario FWER | Success | Parents found |
|---|---|---|---|---|---|---|---|
| A | approximate | 100,000 | 0.0070 | 0.0065-0.0075 | 0.049 | 0.477 | 0.118 |
| A | exact | 100,000 | 0.0063 | 0.0058-0.0068 | 0.053 | 0.494 | 0.172 |
| C | approximate | 4,000 | 0.0095 | 0.0067-0.0130 | 0.025 | 0.498 | 0.294 |
| C | exact | 4,000 | 0.0123 | 0.0091-0.0162 | 0.055 | 0.422 | 0.133 |
| D | approximate | 4,000 | 0.0075 | 0.0051-0.0107 | 0.030 | 0.494 | 0.150 |
| D | exact | 4,000 | 0.0065 | 0.0043-0.0095 | 0.020 | 0.544 | 0.261 |
| D40 | approximate | 1,000 | 0.0030 | 0.0006-0.0087 | 0.040 | 0.497 | 0.000 |
| D40 | exact | 1,000 | 0.0010 | 0.0000-0.0056 | 0.020 | 0.500 | 0.002 |

A scenario's FWER is over its 1,000 datasets in cell A, 200 in C and D, and 50 in D40. Success
counts an empty estimate for a target without parents, which most of D40's successes are: with
`max_parents=3` on 40 variables, it found almost none of the parents.

### Out of the claim

When Appendix G's interventions could hit the target (cell B), the FWER rose to 0.1843 under the
approximate test and 0.1146 under the exact one over the 55 scenarios that hit it, and 0.0079 and
0.0077 over the 45 that did not. The method causes it, not the implementation: on the first 20
datasets of each of the 55 scenarios, R at the like-for-like settings gives the same FWER as Andrey,
0.1936 under the approximate test and 0.1227 under the exact one, with the same parents on 1,100 and
1,094 of the 1,100 fits; `--trace` traces the 104 fits whose accepted sets differ to the exact
test's residual variance. An intervention that changes the target's noise makes the regression on
its parents differ across the environments, so their set is rejected, while a set that holds a child
of the target, whose own equation does not change, stays close to invariant and is accepted; 88% and
87% of the false selections hold a child. The paper reports control of the FWER in its simulation,
which allows the same interventions. On the same 1,100 datasets, R at its package defaults, with the
exact test's 200-row subsamples and the boosting screen above 8 variables, gives 0.0573 under the
exact test, the paper's Method I, near the paper's report, and 0.1836 under the approximate one; the
like-for-like settings test every set on every row. Where the targets are known, `targets` leaves
the environments that hit the target out. A hidden common cause of the target and one other variable
(E-hidden) gave 0.0118 and 0.0182, and 0.145 in one scenario of 20 under the exact test; a `tanh`
effect of one parent on the target (E-nonlinear) gave 0.0030 and 0.0100, with success in 0.4% and
0.2% of the fits.

### Speed

| Variables | Rows | Test | Andrey, median s | R, median s | Median ratio |
|---|---|---|---|---|---|
| 9 | 1,000 | approximate | 0.010 | 0.72 | 72 |
| 9 | 1,000 | exact | 0.009 | 4.2 | 458 |
| 12 | 1,000 | approximate | 0.035 | 6.1 | 170 |
| 12 | 1,000 | exact | 0.030 | 34 | 1,159 |
| 15 | 1,000 | approximate | 0.24 | 58 | 241 |
| 15 | 1,000 | exact | 0.17 | 273 | 1,554 |
| 12 | 10,000 | approximate | 0.049 | 32 | 656 |

Each row is 5 scenarios, one fit each, Andrey's the median of 3 timed fits after one untimed one.
Andrey's exact test reads the same Gram matrices as its approximate one, so it costs no more; R's
solves an `n_e x n_e` system per environment and set.
