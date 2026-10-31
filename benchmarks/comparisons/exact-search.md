# Exact search

`exact_cpu.py` compares Andrey's `exact_search` with causal-learn's `bic_exact_search`, both with
A* and no parent cap, and the empty graph. Both find a DAG with the lowest BIC, so they should
reach the same score on the same data, possibly with different DAGs of the optimal class. Both
score a node by `n * log(RSS / n) + n_parents * log(n)`; Andrey centers each column first and
causal-learn does not, which agree on the standardized data of the R lane's slice:
linear-Gaussian on Erdos-Renyi graphs at average degree 2 with `n = 10 * d` rows. The cost at least
doubles with each added variable, so the ladder stays small.

Each fit also gets `bic`, the BIC of its DAG in Andrey's deviance units. The summary prints, per
solution and `d`, the largest absolute difference between its `bic` and causal-learn's on the same
dataset, zero when both reach the optimum, and the median `mec_shd`:

```shell
$ANDREY_BENCH_PYTHON benchmarks/exact_cpu.py --out outputs/exact-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/exact_cpu.py --out outputs/exact-cpu --cores 1 --cap-mem-mb 32000
```

The defaults are `d` of 5, 10, and 15, five seeds, one untimed warm-up fit, one timed fit, and
`--cap-wall-s` of 1800.
