# PNL direction test

`pnl_pairs.py` compares Andrey's `pnl` with causal-learn's `PNL` and the empty baseline on
cause-effect pairs from the `pnl_pair` regime: two nodes at density 1.0, so every pair has one
edge, in a direction the seed draws, with a post-nonlinear mechanism `y = g2(w * x + e)` and Laplace
noise. The inner transform is linear and the outer one, `g2(z) = z + 0.5 tanh(z)`, nonlinear and
invertible. The driver takes the ANM driver's flags and writes the same direction columns. Its
`agree` column compares the two PNL solutions only, since the baseline never draws an edge.

```shell
$ANDREY_BENCH_PYTHON benchmarks/pnl_pairs.py --out outputs/pnl-smoke --smoke
```

The two packages fit the same model by different means and test it differently. causal-learn
trains two small neural networks per direction for 3000 epochs in batches of 128 rows and drops the
last partial batch, so the smoke pair of 100 rows trains nothing. It then compares the mean of the
assumed cause with the mean of the estimated noise by a two-sample t-test, rounded to three
decimals. Andrey fits smooth basis expansions by Newton's method and tests independence with HSIC.
causal-learn's fit takes over a minute at 1000 rows and grows with the rows.
