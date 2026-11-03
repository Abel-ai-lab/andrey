# Tetrad on continuous data

`tetrad_cpu.py` compares Andrey's PC, GES, BOSS, and GRaSP with Tetrad's PC, FGES, BOSS, and GRaSP,
and `tetrad_fci_latent.py` compares Andrey's FCI with Tetrad's, each with the empty-graph baseline.
Tetrad runs on tetrad-lib 7.6.10 through the one-file driver
`andrey_bench/adapters/TetradContinuousDriver.java`, which calls Tetrad's own search classes
(`Pc`, `Fges`, `Boss` with `PermutationSearch`, `Grasp`, `Fci`) with every setting taken from its
command line, none from a default. Each fit writes the data to a CSV file, starts `java` once, and
reads the graph back from the text file the driver writes. The driver runs the warm-up searches in
the same JVM, then times one search that builds the test or the score from the data and runs the
search. The record's `wall_s` is that search, as a Python package's time excludes interpreter start
and import; `process_s` is the whole call, with the JVM start, the compile of the driver, and the
CSV round trip, so the JVM's time is reported apart. `--cap-wall-s` covers the whole call, the JVM
start and the warm-up searches included. The JVM sees the fit's cores, and FGES's and BOSS's thread
pools take every one, and the stable skeleton search of PC and FCI runs on the JVM's common pool,
which has as many threads as the JVM sees processors; with `--cores 1`, every Tetrad search is
single-threaded. Tetrad 7.6.10 wraps FCI's test in a cache that is a plain `HashMap` and runs the
skeleton search in a parallel stream over it, which on several threads throws `ClassCastException`
or never returns, so the driver gives that search the test uncached.

`tetrad_cpu.py` runs on standardized linear-Gaussian Erdos-Renyi graphs with mean degree 2 and `n =
10 * d` rows, and scores every estimate against the true graph's CPDAG; `tetrad_fci_latent.py` runs
on `latent_gauss_er` (see [Hidden confounders](../README.md#hidden-confounders)) and scores every
estimate against the observed PAG, so read `shd_endpoint`. Each solution's `params` list every
setting it runs, by Andrey's name where both packages have the setting and by Tetrad's setter name
for the rest. The settings map as follows:

| Andrey | Tetrad (driver setting) |
|---|---|
| `indep_test="fisherz"`, `alpha=0.05` | `IndTestFisherZ`, `alpha 0.05` |
| `score_func="bic"` | `SemBicScore` with `ruleType chickering`, `structurePrior 0`, `precomputeCovariances true` |
| `penalty_discount` | `penaltyDiscount`, the same number: both score a node `n log s2 + lambda k log n` |
| `max_parents=None`, no cap on a node's parents (GES) | `maxDegree -1`, no cap on a node's degree (FGES) |
| `seed=0` (BOSS, GRaSP) | `seed 0` of `PermutationSearch` and of `Grasp` |
| GRaSP `depth=3` | `depth 3`, `uncoveredDepth 1`, `nonSingularDepth 1` |
| no depth cap in PC and FCI | `depth -1` |
| `collider_rule="sepsets"` (FCI) | `r0ColliderRule sepsets` |

PC runs order-independent (`fasStable true`), orients colliders from the separating sets
(`colliderOrientationStyle sepsets`), and allows no bidirected edge. GES and FGES start from the
empty graph with no structure prior; FGES keeps `faithfulnessAssumed false` and
`symmetricFirstStep false`, and Andrey's GES runs with `max_parents=None`, which leaves a node's
parent count at its default of `d / 2`, a bound no node of a mean-degree-2 graph reaches. BOSS and
GRaSP run one start from the data's column order (`useDataOrder true`, `numStarts 1`), which a
dataset draws from its seed. BOSS runs without the BES step (`useBes false`) and with Tetrad's two
reset settings (`resetAfterBM false`, `resetAfterRS true`), which Andrey's BOSS has no counterpart
for. Both run at the penalty the BOSS and GRaSP lane runs, `penalty_discount=4`, so their Andrey
rows are that lane's. FCI runs order-independent with possible-d-separation, the complete rule set R1-R10
(`completeRuleSetUsed`), and no cap on conditioning sets or discriminating paths; selection bias is
not excluded (`excludeSelectionBias false`), so R5-R7 run as in Andrey. `guaranteePag` is off.

Tetrad's Fisher z p-values and BIC scores equal Andrey's, which
`tests/test_adapters_tetrad_search.py` checks query by query, and the two skeleton searches of PC
find the same adjacencies. The graphs can still differ, for reasons in each algorithm:

- **PC and FCI colliders.** Tetrad keeps the first conditioning set that separates a pair; Andrey
  keeps the union of every set of that size that separates it. A middle node that is outside the
  first set and inside a later one is a collider for Tetrad and a non-collider for Andrey. When two
  colliders disagree on an edge, Tetrad keeps the one it meets first, in the order of the nodes'
  names as text (`x10` before `x2`), and skips a collider that would close a directed cycle; Andrey
  keeps the one whose pair was separated with the larger p-value. Tetrad 7.6.10 has no setting for
  either rule.
- **FCI possible-d-separation.** Tetrad tests each remaining edge once per endpoint, against the
  whole possible-d-separation set of paths of at most three edges, and keeps no separating set for
  a pair it removes there. Andrey searches each endpoint's possible-d-separation set for a
  separating subset. The two can remove different edges, and Tetrad runs far fewer tests in the
  step.
- **GES and FGES.** FGES runs its forward and backward phases three times when
  `faithfulnessAssumed` is false: over the pairs whose single edge improves the BIC from the empty
  graph, then over the pairs joined by a path of two edges whose middle node is not a collider,
  then over every pair. Andrey's GES runs one forward and one backward phase over every pair. Both
  reach a local optimum of the BIC, and which one depends on the path, so neither graph has the
  lower BIC on every dataset.
- **BOSS and GRaSP.** Both packages search from a start permutation: Tetrad's is the data's column
  order, Andrey's BOSS starts from the columns ranked by their score and its GRaSP from a seeded
  shuffle of them, and each package's seeded random choices are its own, so the same seed is not
  the same search. Both reach local optima of the BIC, and a package's optimum varies with its
  start or seed. On a dataset the two packages often reach the same BIC from different starts, and
  where they differ, neither has the lower BIC on every dataset.

The summary prints each pair's speed-up, Tetrad's median search time over Andrey's, and `same`, the
datasets on which both packages returned the same graph, of those both finished. Differences are
reported with their cause; a reference package's graph is a measurement, not a target.

Running a source file needs a `java` with the `jdk.compiler` module, as a JDK has; the adapters'
setup checks for it, and `ANDREY_BENCH_JAVA` names `java` when it is not on `PATH`. The jar is the
one the [GIN](hidden-confounder-methods.md) lane runs on, named by `ANDREY_BENCH_TETRAD_GIN_JAR`:

```shell
curl -fLO https://repo1.maven.org/maven2/io/github/cmu-phil/tetrad-lib/7.6.10/tetrad-lib-7.6.10-shaded.jar
curl -fsL https://repo1.maven.org/maven2/io/github/cmu-phil/tetrad-lib/7.6.10/tetrad-lib-7.6.10-shaded.jar.sha1
sha1sum tetrad-lib-7.6.10-shaded.jar  # 3a160a01a8cb648c03ceb8c7d5f47e5a8a1e2283
export ANDREY_BENCH_TETRAD_GIN_JAR=$PWD/tetrad-lib-7.6.10-shaded.jar
```

Each record's `package_version` names the tetrad-lib and Java releases. The defaults of both drivers
are `d` of 20, 50, 100, and 200, five seeds, one repeat after one warm-up fit, and a wall cap of
1800 s. A smoke run fits every solution on two small datasets, and a run can keep to some
solutions:

```shell
$ANDREY_BENCH_PYTHON benchmarks/tetrad_cpu.py --out outputs/tetrad-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/tetrad_fci_latent.py --out outputs/tetrad-fci-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/tetrad_cpu.py --out outputs/tetrad-cpu --cap-mem-mb 30000 \
    --solutions andrey.pc.numpy tetrad.pc.fisherz baseline.empty
```
