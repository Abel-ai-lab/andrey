# Mixed data

`mixed_cpu.py` compares Andrey's PC with the degenerate and the conditional Gaussian test, and GES
and hill climbing with the two matching scores, against the same algorithm in another package:
Tetrad's PC and FGES with the same four, bnlearn's `pc.stable` with its conditional Gaussian test,
and bnlearn's hill climbing with its conditional linear Gaussian BIC, the comparator for both of
Andrey's hill-climbing scores. The empty graph is the baseline. It runs on `mixed_er`: a conditional
linear Gaussian network on an Erdos-Renyi graph from `andrey.data.sample_mixed`. Two key fields set
the data: the categorical share of the variables (0.5 by default) and the least and most categories
of a categorical variable (2 and 3); each changes the dataset's digest and label, and other regimes'
keys stay as they were. The Dirichlet concentration of the categorical tables is
`datasets.MIXED_SETTINGS`. Codes and values are never rescaled.

Every solution reads a column as categorical when it holds at most ten distinct whole numbers.
The Andrey adapters name those columns in `categorical`; Tetrad's reader marks a column of at most
ten distinct values categorical, and the bnlearn adapters turn such columns into factors. On
`mixed_er` the rule reads exactly the generated categorical columns, which
`tests/test_adapters_mixed.py` checks, and every Tetrad fit stops unless Tetrad read exactly the
rule's columns. Each
package's names for the settings map to Andrey's:

| Andrey | Tetrad (causal-cmd) | bnlearn |
|---|---|---|
| `categorical=[...]`, the columns of at most 10 whole numbers | `--data-type mixed --numCategories 10` | integer columns of at most 10 values made `factor` |
| `indep_test="degen_gauss"` | `--test dg-lr-test` | - |
| `indep_test="cond_gauss"` | `--test cg-lr-test` | `pc.stable(test = "mi-cg")` |
| `score_func="degen_gauss"` | `--score dg-bic-score` | - |
| `score_func="cond_gauss"` | `--score cg-bic-score` | `hc(score = "bic-cg")` |
| `max_iter=None`, from the empty graph | - | the defaults `max.iter = Inf`, `start = NULL` |
| `alpha=0.05` | `--alpha 0.05` | `alpha = 0.05` |
| `penalty_discount=1.0` | `--penaltyDiscount 1` | the default `k = log(n) / 2` |

Tetrad's PC runs order-independent (`--stableFAS`) and has no depth cap, as Andrey's PC does. It
keeps Tetrad's default collider conflict rule (`--conflictRule 1`), under which a later collider
replaces an earlier one; Andrey's PC keeps the collider whose pair was separated with the larger
p-value. FGES runs with no structure prior (`--structurePrior 0`), since Andrey's GES has none. Both
hill climbings take the best add, remove, or reverse move from the empty graph until none improves
the score, with no restarts, and neither gives a categorical node a continuous parent. bnlearn's
hill climbing returns a DAG, so its comparable error is `mec_shd`; every other row is a CPDAG,
scored by `shd`. The two packages define some of these tests and scores differently from Andrey, so
their graphs can differ on the same data: Tetrad's conditional Gaussian parameter count leaves out
each category's means, its degenerate Gaussian test statistic is twice the likelihood ratio, and its
degenerate Gaussian score sums one regression per category of a categorical child.

causal-cmd 1.12.0 parses `--numCategories` but never stores it, so its own command line reads every
column as continuous. Tetrad's fits therefore run through `andrey_bench/adapters/TetradDriver.java`,
which takes causal-cmd's argument parser, parameters, and algorithm factory from the jar and
replaces only the reader with one that applies `--numCategories`. Each fit writes the data to a CSV
file, starts `java` once on the driver's source with causal-cmd's arguments, and reads its graph
back:

```shell
java -XX:ActiveProcessorCount=4 --class-path causal-cmd-1.12.0-jar-with-dependencies.jar \
    andrey_bench/adapters/TetradDriver.java search --warmup 0 --data-type mixed ...
```

Tetrad's time is its search inside the JVM, excluding JVM start, source compile and data loading,
as Andrey's time excludes interpreter start and import. The driver times the search call alone and
the record keeps it as `wall_s`; the whole call is `process_s`. With `--warmup` above 0, the
warm-up searches run in the same JVM before the timed one. The JVM sees only the fit's cores. The
[GFCI](gfci.md) lane's Tetrad solution runs the same driver and timing.

Running a source file needs a `java` with the `jdk.compiler` module, as a JDK has; the adapter's
setup checks for it. Each record's `package_version` names the causal-cmd, tetrad-lib, and Java
releases. The adapters were written against causal-cmd 1.12.0, which bundles tetrad-lib 7.6.3;
download it from Maven Central, check its SHA-1 against the published one, and point
`ANDREY_BENCH_TETRAD_JAR` at it (`ANDREY_BENCH_JAVA` names `java` when it is not on `PATH`):

```shell
curl -fLO https://repo1.maven.org/maven2/io/github/cmu-phil/causal-cmd/1.12.0/causal-cmd-1.12.0-jar-with-dependencies.jar
curl -fsL https://repo1.maven.org/maven2/io/github/cmu-phil/causal-cmd/1.12.0/causal-cmd-1.12.0-jar-with-dependencies.jar.sha1
sha1sum causal-cmd-1.12.0-jar-with-dependencies.jar  # b0e3c3fb3b0277e0a44c54dbb82f0023c4fd9e54
export ANDREY_BENCH_TETRAD_JAR=$PWD/causal-cmd-1.12.0-jar-with-dependencies.jar
```

The bnlearn rows need the R lane: see [The R lane](../README.md#the-r-lane). The defaults are 2000
rows, `d` of 10, 20, and 40, five seeds, one repeat after one warm-up fit, and a wall cap of 1800 s.
The summary prints each solution's median wall time and error per `d`, then each pair's speed-up:
the comparator's median wall time over Andrey's on the datasets where both finished. A smoke run
covers every solution on one small dataset, a run can keep to some solutions, and a rescore
recomputes the score columns from the stored estimates; pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-cpu --cap-mem-mb 32000 \
    --categorical-share 0.5 --categories 2 3
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-andrey --solutions \
    andrey.pc.degen_gauss andrey.ges.cond_gauss baseline.empty
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-cpu --rescore
```
