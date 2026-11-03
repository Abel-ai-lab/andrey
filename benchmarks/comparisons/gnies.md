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
- The fit of a target's coefficients. Andrey runs the alternating fit to its floating-point fixed
  point; the reference runs 10 iterations.
- Small environments. Andrey makes a target's parent set unavailable when it has more than the
  smallest environment's row count minus 2 members, where the likelihood has no maximum, and when
  the fit leaves an environment a residual variance of zero up to rounding; the reference scores
  these families.
- Ties. Andrey breaks equal rounded score changes by each column's rank, computed from the data,
  so a column relabeling relabels its result; the reference takes the first candidate in column
  order.
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
entirely above 0, and counts as worse only if a rerun on seeds 20 to 39 is worse too. In every cell
with 100 rows per environment or more, Andrey must also beat the empty graph on both errors.

Run it with the reference's environment built as above:

```shell
$ANDREY_BENCH_PYTHON benchmarks/gnies_cpu.py --out outputs/gnies --cells A100 --seeds 0 1 2
```

`--cells` and `--seeds` default to every in-claim cell and seeds 0 to 19, `--arms` to every arm.
Each fit writes one line to `records.jsonl` in the output directory, with its marks, so
`--rescore` recomputes the scores without fitting.

## References

- Gamella, J. L., Taeb, A., Heinze-Deml, C., and Buhlmann, P. (2022). Characterization and greedy
  learning of Gaussian structural causal models under unknown interventions.
  [arXiv:2211.14897](https://arxiv.org/abs/2211.14897).
- Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of interventional
  Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
