# Tabu search

`tabu_cpu.py` compares Andrey's `tabu` with bnlearn's, with each package's `hc` beside them and the
empty-graph baseline. Tabu search is hill climbing that goes on past a local optimum, so the pair
shows what that costs and buys. Every solution is serial. The data is the R lane's slice:
linear-Gaussian on an Erdos-Renyi graph at average degree 2, with `n = 10 * d` rows. Both packages
use a tabu list of 10 DAGs and stop after 10 non-improving moves without a new best, bnlearn's
defaults. bnlearn 5.2.1 stops after `tabu` such moves whatever `max.tabu` is, so the two settings
stay equal.

The defaults are `d` of 10, 20, 50, and 100, five seeds, three repeats each after an untimed fit,
and `--cap-wall-s` of 1800. Load R as [The R lane](../README.md#the-r-lane) describes, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/tabu_cpu.py --out outputs/tabu-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/tabu_cpu.py --out outputs/tabu-cpu --cap-mem-mb 32000
```

To run only some of the solutions, name them, comma-separated. A run without bnlearn's solutions
does not need R:

```shell
$ANDREY_BENCH_PYTHON benchmarks/tabu_cpu.py --out outputs/tabu-andrey --ladder 1600 3200 \
    --repeats 1 --solutions andrey.tabu.numpy.serial,andrey.hc.numpy.serial,baseline.empty
```

Each fit also gets `bic`, the BIC of the graph it returned on its dataset, in Andrey's deviance
units (lower is better); a CPDAG is scored through any of its DAGs, which all have the same BIC. The
summary prints, per solution and `d`, the median `bic` minus `andrey.hc`'s on the same dataset:
below zero is a better score than hill climbing. Its graph error is a CPDAG row's `shd` or a DAG
row's `mec_shd`. A rescore recomputes `bic` without fitting anything; pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/tabu_cpu.py --out outputs/tabu-cpu --rescore
```

The two tabu searches can end on different graphs from the same data. At hill climbing's optimum
the best non-improving moves are often several reversals of covered edges, which leave the score
unchanged, and each package takes the first in its own scan order. bnlearn scans additions, then
removals, then reversals; Andrey ranks all moves by score change, then by the edge's endpoints.

The two packages also score a graph slightly differently. Both charge `log(n)` per parameter in
deviance units, but bnlearn's `bic-g` divides each node's residual sum of squares by `n - p - 1`,
with `p` its parent count. Andrey's BIC divides by a count that does not depend on the parents, so
its score differences are those of the maximum-likelihood BIC. bnlearn's score therefore adds a
small cost per parent: it can rank two nearly equal moves the other way, and it rejects a last move
whose BIC gain is below that cost. The two hill climbs can then end a few edges apart, with
Andrey's graph at the lower BIC. Read a difference in `shd` together with the `bic` column.
