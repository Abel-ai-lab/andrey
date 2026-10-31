# GIES

`gies_cpu.py` compares Andrey's `gies` with pcalg's on data that mixes observation with
interventions. Its data follows the simulation of Hauser and Buhlmann (2012, Section 5.2.1): a
random DAG with mean vertex degree 2; edge weights from `U([-1, -0.1] U [0.1, 1])` and error
variances from `U(0.5, 1)`, rescaled to unit variance without interventions; single-variable
intervention targets, `0.4 * d` of them by default, each setting its variable from `N(2, 0.2^2)`;
and `10 * d` rows, split evenly between no intervention and each target. A fit takes the samples
and the 0/1 intervention matrix side by side; pcalg's adapter turns the matrix into its `targets`
and `target.index` inside the timed fit.

Both packages score BIC with every row counted in the penalty, pcalg's `lambda = 0.5 * log(n)` in
log-likelihood units and Andrey's `penalty_discount=1.0` in deviance units, and both fit an
intercept: pcalg's `GaussL0penIntScore` defaults to none, so the adapter sets `intercept = TRUE`.
Both run the forward, backward, and turning phases until none improves, with no degree cap.
`params` records the settings.

Each fit's graph is scored against the interventional essential graph of the generating DAG: `shd`
counts the variable pairs whose edge differs, and `empty_shd`, the true edge count, is the empty
graph's. `bic` is the graph's BIC on its data in Andrey's deviance units and `truth_bic` the
generating DAG's. The records go to `records.jsonl`, one line per fit, with each graph's endpoint
marks, and the build and machine to `provenance.json`; the summary counts the datasets where Andrey
returned pcalg's graph. Load R as [The R lane](../README.md#the-r-lane) describes, with pcalg in the
library, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/gies_cpu.py --out outputs/gies-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/gies_cpu.py --out outputs/gies-cpu --cores 1 --cap-mem-mb 32000
```

The defaults are `d` of 10, 20, 50, and 100, five seeds, one untimed warm-up fit, one timed fit,
and `--cap-wall-s` of 1800. Both searches are serial.
