# GnIES

`gnies_cases.py` compares Andrey's `gnies` with gnies 0.3.3, the package of the method's authors
(Gamella et al. 2022) and its only implementation, on six small fixed cases. Each case draws
linear-Gaussian data with `andrey.data`: an Erdos-Renyi DAG on 5 or 6 variables at mean degree 2,
weights of either sign with magnitude in `[0.5, 1]`, unit noise variance, an observational
environment, and two or three environments that each multiply one variable's noise variance by
`U[5, 10] / U[1, 2]`, with 1,000 rows per environment, standardized on the pooled rows. Both
packages fit the same rows with the targets unknown.

The reference runs at every default, `gnies.fit(data)`: the standard BIC (`lmbda = 0.5 * log(N)`),
the greedy outer search with its forward and backward phases, and its inner search with forward,
backward, and turning phases, repeated while the score improves. Andrey runs at the same objective,
`penalty_discount=1.0, ebic_gamma=0.0`, and at its defaults, which add the extended-BIC term
(`ebic_gamma=1.0`). Totals are in Andrey's deviance units, lower is better: a reference score `s`
is `-2 * s - N * p - p * log(N)`, with `N` the rows and `p` the variables.

Build the reference's environment once, which pins gnies 0.3.3, ges 1.1.1, and NumPy 2.5.4 on
Python 3.12, then print the table:

```shell
benchmarks/start_gnies.sh
export ANDREY_BENCH_GNIES_PYTHON=$PWD/benchmarks/.venv-gnies/bin/python
$ANDREY_BENCH_PYTHON benchmarks/gnies_cases.py
```

| Case | Variables | Environments | True targets | Reference targets | Andrey targets (BIC) | Same graph | Same targets | Reference total | Andrey total (BIC) | Andrey defaults: same graph and targets |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 5 | obs + 2 | [1, 3] | [1, 3] | [1, 3] | yes | yes | -13883.935 | -13883.935 | yes |
| 2 | 5 | obs + 2 | [2, 3] | [2, 3] | [2, 3] | yes | yes | -6817.389 | -6817.389 | yes |
| 3 | 5 | obs + 3 | [0, 1, 2] | [0, 1, 2] | [0, 1, 2] | yes | yes | -16681.221 | -16681.221 | yes |
| 4 | 6 | obs + 2 | [0, 4] | [0, 4] | [0, 4] | yes | yes | -8529.596 | -8529.596 | yes |
| 5 | 6 | obs + 3 | [0, 1, 4] | [0, 1, 4] | [0, 1, 4] | yes | yes | -19237.917 | -19237.917 | yes |
| 6 | 6 | obs + 3 | [2, 3, 5] | [2, 3, 5] | [2, 3, 5] | yes | yes | -13381.082 | -13381.082 | yes |

On all six cases the two packages return the same interventional essential graph, the same
targets, and the same total to the printed digits, and Andrey's defaults return the same result as
its BIC run.

`tests/test_gnies_lane.py` checks the score both packages compute, and how the comparison reads the
reference's result. On 20 draws of six variables, an observational and two other environments of
60 to 150 rows each, a mean shift per environment, a random DAG, and up to three targets, Andrey's
total at the standard BIC equals the reference's `GnIESScore` run to convergence (`max_iter=1000`),
converted to deviance, within `1e-6`. On a fixed chain with a known target, the reference's 0/1
matrix, where `A[i, j]` alone is `i -> j` and both directions an undirected edge, reads as the true
class. The tests that run the reference skip without `ANDREY_BENCH_GNIES_PYTHON`.

The two packages differ in ways a larger comparison can show:

- The search over graphs for a fixed set of targets. Andrey moves between neighboring
  interventional essential graphs with the operators of Hauser and Buhlmann (2012); the reference
  runs GES's operators on CPDAGs, then completes each graph by orienting every edge at a target
  (the paper's Algorithm 3). Neither search is proven consistent, so on a dataset where either
  stops short of the true graph, the two can stop at different graphs.
- The fit of a target's coefficients. Andrey runs the alternating fit to its fixed point, within
  1,000 steps; the reference runs 10 iterations.
- Small environments. Andrey makes a target's parent set unavailable when it has more than the
  smallest environment's row count minus 2 members, where the likelihood has no maximum, and when
  the fit leaves an environment a residual variance of zero up to rounding; the reference scores
  these families.
- Ties. Andrey breaks equal rounded score changes by each column's rank, computed from the data,
  so a column relabeling relabels its result; the reference's code takes the first candidate in
  column order. On the order check below, no column order changed either package's result.
- The default penalty. Andrey's default adds the extended-BIC term; the reference's is the standard
  BIC, which Andrey runs with `ebic_gamma=0.0`.

## Campaign

`gnies_cpu.py` compares the two packages on the regimes of the paper's simulations (Gamella et al.
2022, Sec. 5.1 and App. H) and on the regimes the paper claims without showing numbers for them.
Each cell draws 20 datasets, seeds 0 to 19, one SCM each, with `andrey.data`: an Erdos-Renyi DAG
on 10 variables, linear equations with weights `U[0.5, 1]`, positive as the paper's code draws
them, Gaussian noise with variances `U[1, 2]`, and environments that each set the noise variance of
their targets to `v`. The targets of every environment are distinct, and every environment draws
its own noise. The rows are standardized on the pooled sample, as the paper does.

| Cell | Mean degree | Environments (rows each) | Targets per environment | `v` | Source |
|---|---|---|---|---|---|
| A10, A100, A1000 | 2.7 | observational + 3 (10, 100, 1,000) | 1 | `U[5, 10]` | Fig. 3 |
| B10, B100, B1000 | 4 | observational + 3 (10, 100, 1,000) | 1 | `U[5, 10]` | Fig. 8 |
| C10, C100, C1000 | 2.7 | observational + 3 (10, 100, 1,000) | 3 | `U[5, 10]` | Fig. 9 |
| D10, D100, D1000 | 2.7 | observational + 3 (10, 100, 1,000) | 1 | `U[3, 4]` | Sec. 5.1 text |
| E100, E1000 | 2.7 | 4, none observational (100, 1,000) | 1 | `U[5, 10]` | Sec. 5.1, claim |
| F10, F100 | 2.7 | observational + 9 (10, 100) | 1 | `U[5, 10]` | Sec. 1 and 6, claim |
| G | 2.7 | observational 1,000 + 3 x 100 | 1 | `U[5, 10]` | claim |
| H1, H2 | 2.7 | observational + 1, observational + 2 (1,000) | 1 | `U[5, 10]` | Theorem 1, claim |
| S100, S1000 | 2.7 | observational + 3 (100, 1,000) | 1 | ratio `U[5, 10] / U[1, 2]` | claim |

The S cells draw weights of either sign and unit noise variances, and multiply a target's noise
variance by the ratio. The paper's code sets `v` from `U[5, 10]`, its text from `U[3, 4]`: the D
cells measure the text's range. No released command produced Figs. 8 and 9; we take `U[5, 10]`
for B and C from the Fig. 3 results those figures overlay. Five more differences from the paper's
simulations: the DAG has exactly `round(k p / 2)` edges instead of a binomial count; environments
draw independent noise instead of sharing one random state; each cell has 20 datasets instead of
100 SCMs with 10 datasets each; the reference is gnies 0.3.3 instead of 0.3.2; and each package
runs at its default penalty only.

Cells outside the method's claim are measured with the same arms and metrics for its limitations,
and never judged: hard interventions (O1), a mean shift alone (O2), weights halved (O3), Laplace
noise (O4), additive nonlinear equations (O5), and 20 to 100 variables (O6).

Every arm fits the same rows, one after another on one CPU with one BLAS thread:

- Andrey's `gnies` at its defaults, the judged arm, and four more settings: the standard BIC
  (`ebic_gamma=0.0`), the true targets given at the standard BIC and at the defaults, and ties
  broken by column position (`andrey.config(tie_break="position")`);
- gnies at every default, the judged reference, and with the true targets fixed
  (`known_targets=I*, phases=[]`), which runs one search over graphs at the true targets;
- Andrey's `ges` on the pooled rows, Andrey's `gies` given the true targets (its score drops a
  target's rows in its environments, so it is not a ceiling here), and the empty graph.

A fit gets 30 minutes, the reference's known-target call 5; a fit past its cap is recorded as a
timeout, never as an empty graph. One code path scores every result:

- `shd` against the true interventional essential graph, the true DAG with every edge at a true
  target oriented;
- the target error, the number of targets learned wrongly or missed;
- target precision and recall, whether the class and the targets are both exact, skeleton and
  orientation errors, and the paper's TDP and FDP over the members of both classes;
- the score gap to the truth in deviance units, lower is better, under Andrey's score at the
  standard BIC for every package and under each fit's own objective. Under its own objective, a
  gap above 2 means that fit's search stopped short of the truth's score.

Before scoring, each SCM is checked against the paper's Assumption 2 at its drawn parameters, for
every member of the true DAG's Markov equivalence class and every environment, and in C against
Assumption 1. A dataset that fails leaves its cell's judged pairs for both packages.

Each cell is judged on `shd` and on the target error. Per dataset, D is Andrey's error minus the
reference's; the cell is worse when the 95% percentile bootstrap interval of the mean of D lies
entirely above 0, better when it lies entirely below 0, and a tie otherwise. A worse cell counts
only if a rerun on seeds 20 to 39 is worse too, and a better cell is cited only if its rerun is
better too. The interval comes from 10,000 resamples of the datasets, with the generator seeded by
the CRC-32 of the cell, metric, and comparison names. In every cell with 100 rows per environment
or more, Andrey must also beat the empty graph on both errors: the interval of its error minus the
empty graph's must lie entirely below 0. With 10 rows per environment it must beat the empty
graph's SHD only where the reference does, and the target-error floor does not apply.

Run it with the reference's environment built as above:

```shell
$ANDREY_BENCH_PYTHON benchmarks/gnies_cpu.py --out outputs/gnies --cells A100 --seeds 0 1 2
```

`--cells` and `--seeds` default to every in-claim cell and seeds 0 to 19, `--arms` to every arm.
Each fit writes one line to `records.jsonl` in the output directory, with its marks, so
`--rescore` recomputes the scores without fitting.

## Results

Andrey's pre-release build of 3 November 2026 against gnies 0.3.3, one fit at a time on one core
of an AMD EPYC 7713 processor with one BLAS thread; the processor model of the order check was not
recorded. Every number below is in
[`published/0.14.0/gnies.json`](../published/0.14.0/gnies.json), with the context arms, the
coverage, and the score gaps of every cell. Every Andrey fit finished within its cap, and every
in-claim SCM passed the model check.

### In the claim

Means over the 20 datasets of each cell, seeds 0 to 19. D is Andrey's error minus the reference's,
with its 95% interval. The empty graph's target error is the number of true targets: 3 in A, B, D,
G, and S; 9 in C and F; 4 in E; 1 in H1; 2 in H2.

| Cell | SHD, Andrey | SHD, gnies | SHD, empty | SHD D [95%] | SHD | Target error, Andrey | Target error, gnies | Target error D [95%] | Target error |
|---|---|---|---|---|---|---|---|---|---|
| A10 | 11.35 | 14.4 | 14 | -3.05 [-4.85, -1.2] | better | 1.85 | 2.45 | -0.6 [-1.0, -0.3] | better |
| A100 | 4.95 | 6.0 | 14 | -1.05 [-3.3, 0.8] | tie | 0.5 | 0.45 | 0.05 [-0.2, 0.35] | tie |
| A1000 | 2.05 | 2.15 | 14 | -0.1 [-0.6, 0.3] | tie | 0.25 | 0.3 | -0.05 [-0.15, 0.0] | tie |
| B10 | 20.0 | 21.95 | 20 | -1.95 [-3.55, -0.45] | better | 2.7 | 2.9 | -0.2 [-0.7, 0.25] | tie |
| B100 | 16.45 | 15.85 | 20 | 0.6 [-0.5, 1.8] | tie | 1.0 | 1.2 | -0.2 [-0.5, 0.05] | tie |
| B1000 | 12.45 | 11.9 | 20 | 0.55 [-0.4, 1.95] | tie | 1.55 | 1.45 | 0.1 [0.0, 0.3] | tie |
| C10 | 11.3 | 12.75 | 14 | -1.45 [-2.9, -0.2] | better | 4.35 | 4.7 | -0.35 [-0.9, 0.15] | tie |
| C100 | 3.0 | 4.05 | 14 | -1.05 [-2.55, -0.05] | better | 0.65 | 0.45 | 0.2 [-0.1, 0.65] | tie |
| C1000 | 1.8 | 1.7 | 14 | 0.1 [0.0, 0.3] | tie | 0.3 | 0.35 | -0.05 [-0.15, 0.0] | tie |
| D10 | 11.9 | 14.3 | 14 | -2.4 [-3.55, -1.35] | better | 3.0 | 3.25 | -0.25 [-0.65, 0.15] | tie |
| D100 | 7.45 | 8.0 | 14 | -0.55 [-1.05, -0.15] | better | 1.05 | 1.1 | -0.05 [-0.15, 0.0] | tie |
| D1000 | 2.35 | 1.75 | 14 | 0.6 [-0.3, 2.1] | tie | 0.3 | 0.4 | -0.1 [-0.3, 0.0] | tie |
| E100 | 3.5 | 4.0 | 14 | -0.5 [-1.2, 0.15] | tie | 0.4 | 0.55 | -0.15 [-0.4, 0.0] | tie |
| E1000 | 1.75 | 1.95 | 14 | -0.2 [-0.5, 0.0] | tie | 0.4 | 0.4 | 0.0 [0.0, 0.0] | tie |
| F10 | 10.25 | 11.05 | 14 | -0.8 [-2.8, 1.35] | tie | 7.7 | 7.55 | 0.15 [-0.05, 0.35] | tie |
| F100 | 1.65 | 2.0 | 14 | -0.35 [-0.65, -0.1] | better | 0.6 | 0.7 | -0.1 [-0.3, 0.0] | tie |
| G | 2.65 | 3.25 | 14 | -0.6 [-1.65, 0.3] | tie | 0.45 | 0.35 | 0.1 [0.0, 0.25] | tie |
| H1 | 2.45 | 2.55 | 14 | -0.1 [-0.25, 0.0] | tie | 0.3 | 0.3 | 0.0 [0.0, 0.0] | tie |
| H2 | 3.7 | 2.7 | 14 | 1.0 [-0.3, 3.25] | tie | 0.55 | 0.3 | 0.25 [-0.05, 0.65] | tie |
| S100 | 3.5 | 3.7 | 14 | -0.2 [-1.0, 0.75] | tie | 0.25 | 0.35 | -0.1 [-0.25, 0.0] | tie |
| S1000 | 1.4 | 1.4 | 14 | 0.0 [0.0, 0.0] | tie | 0.3 | 0.3 | 0.0 [0.0, 0.0] | tie |

No cell is worse on either error. On SHD, 7 of the 21 cells are better and 14 tie; on the target
error, A10 is better and the other 20 tie.

At the reference's own objective, the standard BIC (`ebic_gamma=0.0`), Andrey ties the reference in
all 21 cells on both errors: it returned the same graph and targets on 412 of the 420 datasets, and
the other 8 have 10 rows per environment. Given the true targets at the standard BIC, it returned
the same graph as the reference's known-target call on 418 of 420. The better cells come from
Andrey's default penalty: on SHD, Andrey at its defaults beats Andrey at the standard BIC in the
same 7 cells.

### Replication

The 7 better cells ran again on seeds 20 to 39:

| Cell | SHD D, seeds 0-19 | SHD D, seeds 20-39 | Target error D, seeds 0-19 | Target error D, seeds 20-39 | Better on both seed sets |
|---|---|---|---|---|---|
| A10 | -3.05 [-4.85, -1.2] | -1.47 [-2.32, -0.63] | -0.6 [-1.0, -0.3] | -0.42 [-0.79, -0.11] | SHD and target error |
| B10 | -1.95 [-3.55, -0.45] | -2.2 [-4.4, -0.45] | -0.2 [-0.7, 0.25] | -0.15 [-0.65, 0.3] | SHD |
| C10 | -1.45 [-2.9, -0.2] | -1.21 [-2.74, 0.37] | -0.35 [-0.9, 0.15] | 0.16 [-0.37, 0.68] | neither |
| C100 | -1.05 [-2.55, -0.05] | -0.85 [-2.95, 0.45] | 0.2 [-0.1, 0.65] | 0.25 [0.0, 0.65] | neither |
| D10 | -2.4 [-3.55, -1.35] | -0.42 [-2.05, 1.05] | -0.25 [-0.65, 0.15] | -0.68 [-1.05, -0.32] | neither |
| D100 | -0.55 [-1.05, -0.15] | 0.2 [-0.8, 1.6] | -0.05 [-0.15, 0.0] | 0.05 [-0.1, 0.2] | neither |
| F100 | -0.35 [-0.65, -0.1] | 0.1 [-0.45, 0.95] | -0.1 [-0.3, 0.0] | 0.15 [0.0, 0.45] | neither |

Two cells replicate, both in settings the paper plots, both with 10 rows per environment. In A10
(Fig. 3), Andrey's mean SHD was 11.35 against the reference's 14.4 on seeds 0 to 19 and 10.47
against 11.95 on seeds 20 to 39, and its target error 1.85 against 2.45 and lower again on the
rerun. In B10 (Fig. 8, mean degree 4), its SHD was 20.0 against 21.95 and 17.9 against 20.1; the
empty graph's is 20, and neither package beats it there on either seed set. D10's target error was
better on seeds 20 to 39 only, and its SHD on seeds 0 to 19 only, so neither replicates. On the
rerun, three of the reference's fits returned a local score of NaN, one each in A10, C10, and D10,
and leave their cells' pairs; every Andrey fit finished.

### Floors

In all 16 cells with 100 or more rows per environment, Andrey beats the empty graph on both errors,
on seeds 0 to 19 and on the reruns of C100, D100, and F100. With 10 rows per environment, the SHD
floor applied in F10 on seeds 0 to 19 and in A10, C10, and D10 on seeds 20 to 39, the cells where
the reference beat the empty graph. Andrey passed it in F10 (its SHD minus the empty graph's
-3.75 [-5.85, -1.75]), A10 (-3.05 [-4.75, -1.3]), and C10 (-4.5 [-6.45, -2.6]).

D10 failed it on seeds 20 to 39 only. On seeds 0 to 19 the reference's SHD minus the empty graph's
was 0.3 [-1.65, 2.35], so the floor did not apply there; Andrey's was -2.1 [-4.35, 0.2]. On seeds
20 to 39 the reference's was -2.16 [-3.95, -0.42] over its 19 finished fits, so the floor applied,
and Andrey's was -2.05 [-4.15, 0.1], whose interval reaches above 0: a fail. On those seeds
Andrey's SHD ties the reference's (D -0.42 [-2.05, 1.05]), and the reference's target error was
above the empty set's, 0.84 [0.42, 1.26] more errors than the true targets' count, where Andrey's
was 0.15 [-0.2, 0.5].

With 10 rows per environment, a target with 9 parents can fit one environment's rows exactly once
their mean is taken out, and its likelihood then has no maximum. Andrey makes a target with more
than 8 parents unavailable there; the reference scores it. On seeds 0 to 19 the reference returned
a target with all 9 other variables as parents on 2 datasets of A10 and 1 of D10.

### Out of the claim

Same harness and metrics, seeds 0 to 19, 1,000 rows per environment unless named, never judged.
The empty graph's SHD is 14 and its target error 3 in every row.

| Cell | Change from A | SHD, Andrey | SHD, gnies | SHD D [95%] | Target error, Andrey | Target error, gnies | SHD floor | Target-error floor |
|---|---|---|---|---|---|---|---|---|
| O1-100 | hard interventions, 100 rows | 3.2 | 3.95 | -0.75 [-2.55, 0.85] | 0.35 | 0.2 | pass | pass |
| O1-1000 | hard interventions | 3.75 | 3.75 | 0.0 [0.0, 0.0] | 0.65 | 0.65 | pass | pass |
| O2 | mean shift alone | 4.0 | 3.6 | 0.4 [-1.2, 2.4] | 3.1 | 3.2 | pass | fail |
| O3 | each target's incoming weights halved | 11.6 | 11.8 | -0.2 [-0.55, 0.0] | 3.05 | 3.0 | fail | fail |
| O4 | Laplace noise | 1.65 | 2.0 | -0.35 [-1.0, 0.0] | 0.35 | 0.35 | pass | pass |
| O5 | additive nonlinear equations | 7.55 | 8.0 | -0.45 [-1.2, 0.25] | 0.7 | 0.7 | pass | pass |

Every out-of-claim cell ties. With hard interventions, Andrey's `gies` given the true targets made
2.55 and 1.75 SHD errors at 100 and 1,000 rows. A mean shift alone is invisible to a score that
centers each environment: Andrey's target error minus the true count was 0.1 [-0.1, 0.3], and its
SHD against the observational CPDAG was 2.55 (the reference's 2.35). Halving each target's
incoming weights breaks the shared weights the model assumes: Andrey's SHD minus the empty graph's
was -2.4 [-4.75, 0.0] and its target error minus the true count 0.05 [-0.35, 0.45]; the
reference's floors fail there too.

On 20 to 100 variables (O6), the A design with 1,000 rows per environment and three
single-target environments, seeds 0 to 9, a 3,600 s cap per fit; medians:

| Variables | SHD, Andrey | Target error, Andrey | Seconds, Andrey | SHD, Andrey at the standard BIC | SHD, gnies | Seconds, gnies | SHD, empty |
|---|---|---|---|---|---|---|---|
| 20 | 0 | 0 | 3.5 | 2 | 2 | 388 | 27 |
| 30 | 0 | 0 | 9.0 | 5 | 6 and 17 (seeds 0 and 2) | 3,327 and 2,735 | 40 |
| 50 | 1 | 0 | 30 | 16.5 | not run | | 68 |
| 100 | 2.5 | 0 | 176 | 54.5 | not run | | 135 |

At 30 variables the reference ran seeds 0 to 2, and its fit on seed 1 passed the cap. Every Andrey
fit finished; the slowest took 478 s, at 100 variables and the standard BIC, and 415 s at its
defaults.

### Order check

`qa/gnies_order.py` fits 80 models, the A design at 6, 8, 10, and 12 variables with 100 and 1,000
rows per environment, seeds 0 to 9, in the original order, 3 column orders, and 3 row shuffles each.
Andrey's result changed under 0 of 240 column orders and 0 of 240 row shuffles, and none of its 80
fits in the original order logged a step that a tie decided. The reference's result changed under
0 of 240 column orders.

### Speed

Over the 420 in-claim datasets, Andrey's fit took a median 0.030 times the reference's (quartiles
0.027 and 0.033), about 34 times faster; per cell, the median ratio ran from 0.024 to 0.039. At 10
variables Andrey's median fit took 0.29 to 1.33 s per cell, the reference's 9.1 to 48.6 s.

### The default penalty

Andrey at its default `ebic_gamma=1.0` against `ebic_gamma=0.0`, paired per cell by the same
interval: on seeds 0 to 19 the default made fewer errors in 7 of the 21 in-claim cells on SHD and in
1 on the target error, and more in none. On seeds 20 to 39, in the 7 rerun cells, it made fewer in
3 on SHD and 1 on the target error, and more in none. Out of the claim the two tie in all 6 cells at
10 variables; on the O6 ladder the default's median SHD was 1 against 16.5 at 50 variables and 2.5
against 54.5 at 100.

### Known targets

Given every environment's true targets, both packages run one search over graphs from the empty
graph, and at the standard BIC they returned the same graph on 418 of 420 datasets. Andrey's mean
SHD with the true targets given was higher than with the targets learned in 17 of the 21 in-claim
cells at its defaults (A1000: 5.6 against 2.05), lower in 3, and equal in 1. In A1000 at the
standard BIC the two fits differ on 5 datasets, each one where the learned targets add or miss one
or two, and each time the learned fit has the lower score: a greedy search over graphs can stop
short of the lowest score, and a different set of targets changes where it stops.

### gnies 0.3.3 and the paper's 0.3.2

The paper's results come from gnies 0.3.2; this comparison runs 0.3.3, the current release. 0.3.3
adds the `center` parameter, whose default `True` is the 0.3.2 behavior, and its outer search takes
a step only when the score strictly improves, where 0.3.2 also took a step that tied, which matters
only on exact ties. In 0.3.3 `ges_iterate` is documented as `False` by default and is `True` in the
code; the comparison runs the code's `True`, as the paper's runs did.

## References

- Gamella, J. L., Taeb, A., Heinze-Deml, C., and Buhlmann, P. (2022). Characterization and greedy
  learning of Gaussian structural causal models under unknown interventions.
  [arXiv:2211.14897](https://arxiv.org/abs/2211.14897).
- Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of interventional
  Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
