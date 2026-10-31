# XGES

`xges_cpu.py` compares Andrey's `xges` with the xges package, each with and without the extended
search (the arms `andrey.xges.numpy`, `andrey.xges.numpy.no_extended_search`, `xges.xges`, and
`xges.xges.no_extended_search`), with Andrey's `ges` and the empty-graph baseline. The data are
linear-Gaussian and standardized, on Erdos-Renyi graphs of average degree 2 and 4, with `10 * d`
rows. XGES gains most over GES on the denser graphs. Every arm searches under one BIC weight: the
package scores in log-likelihood units, -0.5 times Andrey's deviance plus terms that cancel in every
move, so its `alpha` is Andrey's `penalty_discount`. No arm caps the parents of a node: XGES has
no cap, and the GES arm lifts the adapter's usual cap of 4, as `andrey.ges` does. `params` records
the settings.

Each record's `shd` is against the true CPDAG. After each rung the driver adds two columns from the
graphs in `structures/`: `bic`, the BIC of the returned graph under Andrey's `BICScore` at the
campaign's weight (lower is better), and `bic_gap`, `bic` minus the true graph's BIC on the same
data. One scorer scores every package's graph, so `bic` compares the searches on one scale; a
negative gap is a graph that scores better than the truth on its data. A fit with no graph has
`NaN` in both.

Andrey's XGES runs on one BLAS thread whatever its core budget; Andrey GES uses the budget as BLAS
threads and the package as numba threads. The record's `threads` is the budget.

The package arms need their own environment, as in [Separate solution
environments](../README.md#separate-solution-environments). A smoke run fits every solution once at
`d = 10` on both densities:

```shell
export ANDREY_BENCH_XGES_PYTHON=$PWD/benchmarks/.venv-xges/bin/python
$ANDREY_BENCH_PYTHON benchmarks/xges_cpu.py --out outputs/xges-smoke --smoke
```

The defaults are `d` of 25, 50, 100, 200, 500, and 1000, five seeds per density, one untimed
warm-up fit (the package's numba scorer compiles on its first call), and one timed fit. The wall
cap covers the warm-up too. The package's extended search copies its search state, score cache
included, for each edge it tries, so the cap bounds that arm first at large `d`:

```shell
$ANDREY_BENCH_PYTHON benchmarks/xges_cpu.py --out outputs/xges-cpu \
    --cap-wall-s 3600 --cap-mem-mb 32000
```

Continue an interrupted run with the first run's flags and `--resume`. A rescore recomputes the BIC
columns of every part without fitting anything; pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/xges_cpu.py --out outputs/xges-cpu --rescore
```
