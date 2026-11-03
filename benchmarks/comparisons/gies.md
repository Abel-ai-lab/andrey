# GIES

`gies_cpu.py` compares Andrey's `gies` with pcalg's on data that mixes observation with
interventions. Andrey's `andrey.data` draws it in the setting of Hauser and Buhlmann (2012, Section
5.2.1): an Erdos-Renyi DAG with mean vertex degree 2; edge weights from `U([-1, -0.1] U [0.1, 1])`
and standard Gaussian errors, each variable divided by its standard deviation as it is generated
(`scale="rescale"`), so every variable has variance 1 without interventions; single-variable
intervention targets, `0.4 * d` of them by default, each environment cutting its target from its
parents and setting it from `N(2, 0.2^2)`; and `10 * d` rows, split evenly between the environment
without intervention and each target's. A fit takes the samples and the 0/1 intervention matrix,
built from the sample's `env` and `targets`, side by side; pcalg's adapter turns the matrix into its
`targets` and `target.index` inside the timed fit. The 0.7.0 and 0.9.0 reports drew their data
with an earlier sampler of the harness, with error variances from `U(0.5, 1)` and the unit variance
exact rather than of the sample; `--rescore` refuses their runs.

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
