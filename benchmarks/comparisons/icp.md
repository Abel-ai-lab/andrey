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
  `K / (K - 1)`; [The correction](#the-correction) measures the guarantee under each factor.
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
The weights and noise scales are the scenario's, set by small subclasses of the linear functional
and the Gaussian noise. On four scenarios of 300 datasets each, the two generators'
per-environment means and covariances, and Andrey's false selections and successes, agree within
sampling error. The other cells keep the driver's generator, since `envs=` cannot express all of
their settings: Appendix G draws each new weight into an intervened variable from a range of its
own, where `soft(weights="redraw")` draws from the functional's `weight_range`. E-hidden places one
hidden parent of the target and of one other variable, where `latents=` places the hidden variables
itself, and E-nonlinear makes one term of the target's equation nonlinear, where a functional
applies to every equation. The speed cells use cell A's
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
`trace.json`; [`published/0.14.0/icp.json`](../published/0.14.0/icp.json) holds them with the
paired errors of each cell.

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
of scenarios. The claim bounds the variables ICP names as parents.

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

The paired fits also give a floor in each of their 47 rows, in and out of the claim, one test at one
setting and number of variables: Andrey's and R's mean errors minus those of the empty estimate,
which names no parents, each with its 95% interval. The floor applies where R beats the empty
estimate, its interval wholly below 0, which it does in 20 rows, 19 of them in the claim, and Andrey
beats the empty estimate in all 20; in the other 27 the floor is reported, not judged.
[`icp.json`](../published/0.14.0/icp.json) lists each row's floor under `floors`. We refitted every
paired fit on a second machine, an AMD EPYC 7713 with R 4.6.1 and InvariantCausalPrediction 0.8, and
the refit reproduces every published row and interval.

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

## The correction

`icp_correction.py` tests the approximate test's correction directly, Andrey's factor `K` against
R's `K - 1`. The guarantee holds when the test rejects the true parents' set `S*` with probability
at most `alpha`, so the driver measures that rate and the FWER of the full fit, the share of fits
whose parents are not a subset of `S*`, under each factor. The test with the factor `K - 1` is
Andrey's approximate test with the factor changed, registered beside it in the invariance-test
registry, so that `icp`'s own set search runs with it. At the factor `K` it returns Andrey's
p-values bit for bit on every set of 20 datasets, and SciPy's Welch and F tests on the pooled
residuals give the same p-values within `1e-12` relative.

Each dataset comes from `andrey.data`, `SCM.sample` with `envs=`, on a fixed linear Gaussian SEM
of 6 variables with unit noise: `0 -> 1`, `2 -> 1`, `0 -> 3`, `1 -> 3`, `3 -> 4`, `4 -> 5`, and
`2 -> 5`, with the target 3, so `S* = {0, 1}`, 4 is its child, and 2 is neither a parent nor a
descendant of it. The first environment is observational; each other one intervenes on one of the
five other variables, drawn per dataset, by a do-intervention `hard(2.0, sd=1.0)` or a mean shift
`soft(shift=2.0)`, each with probability 1/2. The grid crosses `K` = 3, 4, 5, 8, and 10
environments with `n` = 30, 100, and 500 rows per environment, every environment at `n` rows or
the observational one at `5 n`, with 2,000 datasets per cell, so a rate of 0.05 has a standard
error of about 0.005. One fit at `alpha=0.05` gives both levels, since a fit's per-variable
p-values and its largest set p-value do not depend on `alpha`; the driver checks this against a
fit at `alpha=0.10`, and that every decision under `K - 1` at `alpha` is the decision under `K` at
`alpha K / (K - 1)`.

In the grid, most rejections of `S*` yield no false parent, since a set with a variable outside
`S*` is either rejected too or meets another accepted set that leaves the variable out. The hard
cell removes both exits, so that its FWER follows the rejection rate of `S*`. Its SEM has 5
variables, `0 -> 1`, `1 -> 3`, `2 -> 3`, and `3 -> 4`, with the target 3, `S* = {1, 2}`, and the
child 4 a near copy of it, `x_4 = 2 x_3 + N(0, 0.5^2)`. With `K = 3` environments of 100 rows,
the second sets parent 1 by `hard(3.0, sd=1.0)` and the third shifts parent 2 by
`soft(shift=3.0)`, so every set without both parents is far from invariant unless it holds the
child; both shift the child by 0.1, a fifth of its noise, so the sets with both parents and the
child are nearly invariant without being invariant. The only other invariant set, `{0, 1, 2}`, has
nearly the residuals of `S*`, so it is rejected with it. When `S*` is rejected, the accepted sets
are then the near-invariant ones with the child, and their intersection holds it: a false parent.

```shell
$ANDREY_BENCH_PYTHON benchmarks/icp_correction.py --seed 0 --out outputs/icp-correction
```

`correction.json` holds each cell's counts and rates, with their 95% Clopper-Pearson intervals,
and the checks.

### Rates

One run with `--seed 0`, on the machine of [Results](#results) with 4 worker processes, took 3
minutes. Each rate is over 2,000 datasets, with its 95% Clopper-Pearson interval.

| Cell | K | Rows per environment | alpha | `S*` rejected, K | `S*` rejected, K - 1 | K / (K - 1) x alpha | FWER, K | FWER, K - 1 |
|---|---|---|---|---|---|---|---|---|
| grid | 3 | 30 | 0.05 | 0.0300 (0.0230-0.0384) | 0.0485 (0.0395-0.0588) | 0.0750 | 0.0020 (0.0005-0.0051) | 0.0030 (0.0011-0.0065) |
| grid | 3 | 30 | 0.10 | 0.0645 (0.0541-0.0762) | 0.0955 (0.0830-0.1092) | 0.1500 | 0.0055 (0.0027-0.0098) | 0.0115 (0.0073-0.0172) |
| grid | 3 | 100 | 0.05 | 0.0350 (0.0274-0.0440) | 0.0505 (0.0413-0.0610) | 0.0750 | 0.0045 (0.0021-0.0085) | 0.0090 (0.0053-0.0142) |
| grid | 3 | 100 | 0.10 | 0.0655 (0.0551-0.0772) | 0.1095 (0.0961-0.1240) | 0.1500 | 0.0125 (0.0081-0.0184) | 0.0225 (0.0165-0.0300) |
| grid | 3 | 500 | 0.05 | 0.0285 (0.0217-0.0368) | 0.0415 (0.0332-0.0512) | 0.0750 | 0.0070 (0.0038-0.0117) | 0.0115 (0.0073-0.0172) |
| grid | 3 | 500 | 0.10 | 0.0650 (0.0546-0.0767) | 0.0950 (0.0825-0.1087) | 0.1500 | 0.0190 (0.0135-0.0260) | 0.0240 (0.0177-0.0317) |
| grid | 4 | 30 | 0.05 | 0.0385 (0.0305-0.0479) | 0.0520 (0.0427-0.0627) | 0.0667 | 0.0065 (0.0035-0.0111) | 0.0070 (0.0038-0.0117) |
| grid | 4 | 30 | 0.10 | 0.0750 (0.0638-0.0874) | 0.1050 (0.0919-0.1193) | 0.1333 | 0.0135 (0.0089-0.0196) | 0.0215 (0.0156-0.0289) |
| grid | 4 | 100 | 0.05 | 0.0405 (0.0323-0.0501) | 0.0530 (0.0436-0.0637) | 0.0667 | 0.0090 (0.0053-0.0142) | 0.0115 (0.0073-0.0172) |
| grid | 4 | 100 | 0.10 | 0.0715 (0.0606-0.0837) | 0.0930 (0.0806-0.1066) | 0.1333 | 0.0140 (0.0093-0.0202) | 0.0185 (0.0131-0.0254) |
| grid | 4 | 500 | 0.05 | 0.0420 (0.0336-0.0517) | 0.0550 (0.0454-0.0659) | 0.0667 | 0.0115 (0.0073-0.0172) | 0.0170 (0.0118-0.0237) |
| grid | 4 | 500 | 0.10 | 0.0780 (0.0666-0.0906) | 0.1035 (0.0905-0.1177) | 0.1333 | 0.0265 (0.0199-0.0345) | 0.0380 (0.0301-0.0473) |
| grid | 5 | 30 | 0.05 | 0.0400 (0.0318-0.0495) | 0.0500 (0.0409-0.0605) | 0.0625 | 0.0080 (0.0046-0.0130) | 0.0100 (0.0061-0.0154) |
| grid | 5 | 30 | 0.10 | 0.0785 (0.0671-0.0912) | 0.0950 (0.0825-0.1087) | 0.1250 | 0.0165 (0.0114-0.0231) | 0.0220 (0.0160-0.0294) |
| grid | 5 | 100 | 0.05 | 0.0355 (0.0278-0.0446) | 0.0425 (0.0341-0.0523) | 0.0625 | 0.0130 (0.0085-0.0190) | 0.0120 (0.0077-0.0178) |
| grid | 5 | 100 | 0.10 | 0.0725 (0.0615-0.0848) | 0.0930 (0.0806-0.1066) | 0.1250 | 0.0200 (0.0143-0.0271) | 0.0300 (0.0230-0.0384) |
| grid | 5 | 500 | 0.05 | 0.0365 (0.0287-0.0457) | 0.0415 (0.0332-0.0512) | 0.0625 | 0.0135 (0.0089-0.0196) | 0.0135 (0.0089-0.0196) |
| grid | 5 | 500 | 0.10 | 0.0730 (0.0620-0.0853) | 0.0915 (0.0792-0.1050) | 0.1250 | 0.0265 (0.0199-0.0345) | 0.0335 (0.0261-0.0424) |
| grid | 8 | 30 | 0.05 | 0.0415 (0.0332-0.0512) | 0.0505 (0.0413-0.0610) | 0.0571 | 0.0160 (0.0110-0.0225) | 0.0185 (0.0131-0.0254) |
| grid | 8 | 30 | 0.10 | 0.0875 (0.0755-0.1007) | 0.0980 (0.0853-0.1119) | 0.1143 | 0.0285 (0.0217-0.0368) | 0.0340 (0.0265-0.0429) |
| grid | 8 | 100 | 0.05 | 0.0430 (0.0345-0.0528) | 0.0480 (0.0391-0.0583) | 0.0571 | 0.0120 (0.0077-0.0178) | 0.0115 (0.0073-0.0172) |
| grid | 8 | 100 | 0.10 | 0.0780 (0.0666-0.0906) | 0.0890 (0.0769-0.1023) | 0.1143 | 0.0255 (0.0190-0.0334) | 0.0275 (0.0208-0.0356) |
| grid | 8 | 500 | 0.05 | 0.0390 (0.0309-0.0484) | 0.0455 (0.0368-0.0556) | 0.0571 | 0.0130 (0.0085-0.0190) | 0.0160 (0.0110-0.0225) |
| grid | 8 | 500 | 0.10 | 0.0805 (0.0689-0.0933) | 0.0900 (0.0778-0.1034) | 0.1143 | 0.0290 (0.0221-0.0373) | 0.0305 (0.0234-0.0390) |
| grid | 10 | 30 | 0.05 | 0.0465 (0.0377-0.0567) | 0.0510 (0.0418-0.0616) | 0.0556 | 0.0195 (0.0139-0.0266) | 0.0240 (0.0177-0.0317) |
| grid | 10 | 30 | 0.10 | 0.0910 (0.0788-0.1045) | 0.1030 (0.0900-0.1172) | 0.1111 | 0.0295 (0.0225-0.0379) | 0.0345 (0.0269-0.0435) |
| grid | 10 | 100 | 0.05 | 0.0420 (0.0336-0.0517) | 0.0485 (0.0395-0.0588) | 0.0556 | 0.0135 (0.0089-0.0196) | 0.0155 (0.0106-0.0219) |
| grid | 10 | 100 | 0.10 | 0.0860 (0.0741-0.0992) | 0.0980 (0.0853-0.1119) | 0.1111 | 0.0265 (0.0199-0.0345) | 0.0300 (0.0230-0.0384) |
| grid | 10 | 500 | 0.05 | 0.0440 (0.0354-0.0539) | 0.0520 (0.0427-0.0627) | 0.0556 | 0.0145 (0.0097-0.0208) | 0.0160 (0.0110-0.0225) |
| grid | 10 | 500 | 0.10 | 0.0775 (0.0662-0.0901) | 0.0835 (0.0717-0.0965) | 0.1111 | 0.0220 (0.0160-0.0294) | 0.0245 (0.0182-0.0323) |
| grid | 3 | 150 + 2 x 30 | 0.05 | 0.0320 (0.0247-0.0407) | 0.0505 (0.0413-0.0610) | 0.0750 | 0.0040 (0.0017-0.0079) | 0.0100 (0.0061-0.0154) |
| grid | 3 | 150 + 2 x 30 | 0.10 | 0.0680 (0.0574-0.0799) | 0.1010 (0.0881-0.1150) | 0.1500 | 0.0105 (0.0065-0.0160) | 0.0195 (0.0139-0.0266) |
| grid | 3 | 500 + 2 x 100 | 0.05 | 0.0345 (0.0269-0.0435) | 0.0560 (0.0463-0.0670) | 0.0750 | 0.0080 (0.0046-0.0130) | 0.0115 (0.0073-0.0172) |
| grid | 3 | 500 + 2 x 100 | 0.10 | 0.0800 (0.0685-0.0928) | 0.1075 (0.0943-0.1219) | 0.1500 | 0.0175 (0.0122-0.0243) | 0.0250 (0.0186-0.0328) |
| grid | 3 | 2,500 + 2 x 500 | 0.05 | 0.0350 (0.0274-0.0440) | 0.0560 (0.0463-0.0670) | 0.0750 | 0.0065 (0.0035-0.0111) | 0.0120 (0.0077-0.0178) |
| grid | 3 | 2,500 + 2 x 500 | 0.10 | 0.0750 (0.0638-0.0874) | 0.1035 (0.0905-0.1177) | 0.1500 | 0.0205 (0.0148-0.0277) | 0.0265 (0.0199-0.0345) |
| grid | 4 | 150 + 3 x 30 | 0.05 | 0.0420 (0.0336-0.0517) | 0.0535 (0.0440-0.0643) | 0.0667 | 0.0055 (0.0027-0.0098) | 0.0090 (0.0053-0.0142) |
| grid | 4 | 150 + 3 x 30 | 0.10 | 0.0850 (0.0731-0.0981) | 0.1155 (0.1018-0.1303) | 0.1333 | 0.0145 (0.0097-0.0208) | 0.0200 (0.0143-0.0271) |
| grid | 4 | 500 + 3 x 100 | 0.05 | 0.0400 (0.0318-0.0495) | 0.0515 (0.0422-0.0621) | 0.0667 | 0.0120 (0.0077-0.0178) | 0.0160 (0.0110-0.0225) |
| grid | 4 | 500 + 3 x 100 | 0.10 | 0.0715 (0.0606-0.0837) | 0.0945 (0.0820-0.1082) | 0.1333 | 0.0230 (0.0169-0.0306) | 0.0280 (0.0212-0.0362) |
| grid | 4 | 2,500 + 3 x 500 | 0.05 | 0.0375 (0.0296-0.0468) | 0.0490 (0.0400-0.0594) | 0.0667 | 0.0155 (0.0106-0.0219) | 0.0190 (0.0135-0.0260) |
| grid | 4 | 2,500 + 3 x 500 | 0.10 | 0.0780 (0.0666-0.0906) | 0.1060 (0.0928-0.1203) | 0.1333 | 0.0295 (0.0225-0.0379) | 0.0395 (0.0314-0.0490) |
| grid | 5 | 150 + 4 x 30 | 0.05 | 0.0370 (0.0292-0.0462) | 0.0450 (0.0363-0.0550) | 0.0625 | 0.0080 (0.0046-0.0130) | 0.0120 (0.0077-0.0178) |
| grid | 5 | 150 + 4 x 30 | 0.10 | 0.0710 (0.0601-0.0831) | 0.0940 (0.0816-0.1076) | 0.1250 | 0.0210 (0.0152-0.0283) | 0.0265 (0.0199-0.0345) |
| grid | 5 | 500 + 4 x 100 | 0.05 | 0.0380 (0.0301-0.0473) | 0.0485 (0.0395-0.0588) | 0.0625 | 0.0130 (0.0085-0.0190) | 0.0165 (0.0114-0.0231) |
| grid | 5 | 500 + 4 x 100 | 0.10 | 0.0805 (0.0689-0.0933) | 0.0975 (0.0848-0.1113) | 0.1250 | 0.0285 (0.0217-0.0368) | 0.0355 (0.0278-0.0446) |
| grid | 5 | 2,500 + 4 x 500 | 0.05 | 0.0410 (0.0327-0.0506) | 0.0510 (0.0418-0.0616) | 0.0625 | 0.0190 (0.0135-0.0260) | 0.0220 (0.0160-0.0294) |
| grid | 5 | 2,500 + 4 x 500 | 0.10 | 0.0735 (0.0624-0.0858) | 0.0935 (0.0811-0.1071) | 0.1250 | 0.0350 (0.0274-0.0440) | 0.0410 (0.0327-0.0506) |
| grid | 8 | 150 + 7 x 30 | 0.05 | 0.0385 (0.0305-0.0479) | 0.0440 (0.0354-0.0539) | 0.0571 | 0.0135 (0.0089-0.0196) | 0.0145 (0.0097-0.0208) |
| grid | 8 | 150 + 7 x 30 | 0.10 | 0.0805 (0.0689-0.0933) | 0.0945 (0.0820-0.1082) | 0.1143 | 0.0255 (0.0190-0.0334) | 0.0275 (0.0208-0.0356) |
| grid | 8 | 500 + 7 x 100 | 0.05 | 0.0400 (0.0318-0.0495) | 0.0450 (0.0363-0.0550) | 0.0571 | 0.0115 (0.0073-0.0172) | 0.0135 (0.0089-0.0196) |
| grid | 8 | 500 + 7 x 100 | 0.10 | 0.0835 (0.0717-0.0965) | 0.0925 (0.0802-0.1061) | 0.1143 | 0.0285 (0.0217-0.0368) | 0.0285 (0.0217-0.0368) |
| grid | 8 | 2,500 + 7 x 500 | 0.05 | 0.0375 (0.0296-0.0468) | 0.0450 (0.0363-0.0550) | 0.0571 | 0.0120 (0.0077-0.0178) | 0.0140 (0.0093-0.0202) |
| grid | 8 | 2,500 + 7 x 500 | 0.10 | 0.0740 (0.0629-0.0864) | 0.0820 (0.0703-0.0949) | 0.1143 | 0.0240 (0.0177-0.0317) | 0.0300 (0.0230-0.0384) |
| grid | 10 | 150 + 9 x 30 | 0.05 | 0.0460 (0.0372-0.0561) | 0.0480 (0.0391-0.0583) | 0.0556 | 0.0165 (0.0114-0.0231) | 0.0145 (0.0097-0.0208) |
| grid | 10 | 150 + 9 x 30 | 0.10 | 0.0780 (0.0666-0.0906) | 0.0890 (0.0769-0.1023) | 0.1111 | 0.0230 (0.0169-0.0306) | 0.0275 (0.0208-0.0356) |
| grid | 10 | 500 + 9 x 100 | 0.05 | 0.0445 (0.0359-0.0545) | 0.0520 (0.0427-0.0627) | 0.0556 | 0.0110 (0.0069-0.0166) | 0.0135 (0.0089-0.0196) |
| grid | 10 | 500 + 9 x 100 | 0.10 | 0.0870 (0.0750-0.1002) | 0.0945 (0.0820-0.1082) | 0.1111 | 0.0220 (0.0160-0.0294) | 0.0250 (0.0186-0.0328) |
| grid | 10 | 2,500 + 9 x 500 | 0.05 | 0.0520 (0.0427-0.0627) | 0.0580 (0.0482-0.0692) | 0.0556 | 0.0110 (0.0069-0.0166) | 0.0140 (0.0093-0.0202) |
| grid | 10 | 2,500 + 9 x 500 | 0.10 | 0.0870 (0.0750-0.1002) | 0.0960 (0.0834-0.1098) | 0.1111 | 0.0185 (0.0131-0.0254) | 0.0220 (0.0160-0.0294) |
| hard | 3 | 100 | 0.05 | 0.0175 (0.0122-0.0243) | 0.0315 (0.0243-0.0401) | 0.0750 | 0.0135 (0.0089-0.0196) | 0.0265 (0.0199-0.0345) |
| hard | 3 | 100 | 0.10 | 0.0455 (0.0368-0.0556) | 0.0645 (0.0541-0.0762) | 0.1500 | 0.0400 (0.0318-0.0495) | 0.0575 (0.0477-0.0686) |

Under `K`, the rejection rate of `S*` stayed at or below `alpha`, within sampling error, in every
cell, and `K - 1` raised it by about `K / (K - 1)`, to near `alpha`, pooled 0.0494 at `alpha=0.05`
over the grid: well below
the worst case at `K = 3` and 4, where it is 1.5 and 1.33 times `alpha`, and at it for `K` = 8 and
10, where the worst case is within a tenth of `alpha`. One of the 62 intervals under `K - 1` lies
above `alpha`, as chance gives over 62, and a rerun of that cell with `--seed 1` and 10,000
datasets put its rate at `alpha`. The FWER of the full fit stayed below `alpha` under both factors
in every cell, and in the hard cell it followed the rejection rate of `S*`.
