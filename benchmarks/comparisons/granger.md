# Granger causality

`granger_var.py` compares Andrey's `granger` with causal-learn's `Granger.granger_lasso` on
vector-autoregressive series with known lagged edges, drawn by `andrey.data`: two lags, Gaussian
noise, and no effects within a time step, so the true lagged edges are the nonzero entries of the
lag matrices. Every variable keeps its own past, and each other variable enters a lag with
probability `2 / (N - 1)`; the lag matrices are scaled so the series is stationary. Each variable
is z-scored before the fits. The data do not fit the dataset store, which holds one graph per
dataset, so the driver draws each series itself and calls the runner directly, as
`public_datasets.py` does.

The two packages fit different models. Andrey fits each variable on the past `tau_max` steps of
every variable by least squares and keeps a lagged link when the F-test of its coefficient,
adjusted by Benjamini-Hochberg over all the links, has a p-value of at most `alpha`; the driver
leaves `alpha` and `fdr_method` at their defaults, 0.05 and `"fdr_bh"`. causal-learn fits a
cross-validated lasso per variable and keeps its nonzero coefficients. `--maxlag` is the lag count
both fit, Andrey's `tau_max` and causal-learn's `maxlag`; `--cv`, causal-learn's folds, goes to
causal-learn alone. Each package gets only its own settings.

The defaults are N of 5, 10, 20, 40, and 80 variables, T of 1000 time steps, five seeds, `maxlag` 2,
`cv` 5, one repeat without warm-up, and `--cap-wall-s` of 1800. A fit past the cap is a `timeout`
record with no scores. A smoke run fits one series of 5 variables:

```shell
$ANDREY_BENCH_PYTHON benchmarks/granger_var.py --out outputs/granger-smoke --smoke
```

A lagged edge is a link the solution keeps at lags 1 to `maxlag`. `precision`, `recall`, and `f1`
score every lagged edge; `cross_precision`, `cross_recall`, and `cross_f1` leave out each
variable's edges from its own past, which every true series holds. The summary prints their means
per N and solution, the median wall time, and `agree`, the share of series on which the solutions
return the same lagged edges: here, the F-test's links against the lasso's support. With the
defaults at 80 variables, Andrey's precision and recall are 0.95 and 0.48, and causal-learn's 0.28
and 0.69: the F-test keeps fewer links, and more of them true.

Each fit is one JSON line in `records.jsonl`, with its coefficients. A resumed run skips the fits
already recorded, and a rescore recomputes the scores without fitting anything:

```shell
$ANDREY_BENCH_PYTHON benchmarks/granger_var.py --out outputs/granger-var --rescore
```
