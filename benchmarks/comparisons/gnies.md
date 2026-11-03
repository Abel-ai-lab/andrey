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

## References

- Gamella, J. L., Taeb, A., Heinze-Deml, C., and Buhlmann, P. (2022). Characterization and greedy
  learning of Gaussian structural causal models under unknown interventions.
  [arXiv:2211.14897](https://arxiv.org/abs/2211.14897).
- Hauser, A., and Buhlmann, P. (2012). Characterization and greedy learning of interventional
  Markov equivalence classes of directed acyclic graphs. JMLR 13, 2409-2464.
