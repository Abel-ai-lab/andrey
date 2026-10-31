# GFCI

`gfci_latent.py` compares Andrey's GFCI with Tetrad's, with Andrey's FCI and the empty-graph
baseline beside them. Andrey's GFCI runs twice: `andrey.gfci.numpy` in one process, and
`andrey.gfci.numpy.parallel` with the GES step in the fit's cores as worker processes. Both return
the same PAG. It runs on `latent_gauss_er`, as `fci_latent.py` does (see [Hidden
confounders](../README.md#hidden-confounders)), and every estimate is scored against the observed
PAG; read `shd_endpoint`. Each package's names for the settings map to Andrey's:

| Andrey | Tetrad (causal-cmd) |
|---|---|
| `indep_test="fisherz"`, `alpha=0.05` | `--test fisher-z-test --alpha 0.05` |
| `score_func="bic"`, `penalty_discount=1.0` | `--score sem-bic-score --penaltyDiscount 1`, with `--semBicStructurePrior 0 --semBicRule 1` |
| no cap on conditioning sets or discriminating paths | `--depth -1 --maxPathLength -1` |
| possible-d-separation, the discriminating-path rule, and rules R1-R10 | `--possibleMsepDone --doDiscriminatingPathRule --completeRuleSetUsed` |

Tetrad runs through the driver of the [Mixed data](mixed.md) lane, with one timing mechanism
for every Tetrad solution. Each fit writes the data to a CSV file, starts `java` once on
`andrey_bench/adapters/TetradDriver.java` with causal-cmd's arguments for GFCI on continuous data,
and reads the PAG back from the text file the driver writes. The driver runs the warm-up search in
the same JVM and times the search call alone, so `wall_s` is Tetrad's search inside the JVM,
excluding JVM start, source compile, and data loading, and `process_s` is the whole call. Its PAG
equals the one `java -jar` returns, which the lane's tests check. The JVM sees only the fit's
cores, and `precomputeCovariances` lets it compute the covariance matrix once. Running a source
file needs a `java` with the `jdk.compiler` module, as a JDK has. The adapter was written against
causal-cmd 1.12.0, which bundles tetrad-lib 7.6.3. Download the jar from Maven Central, check its
SHA-1 against the published one, and point `ANDREY_BENCH_TETRAD_JAR` at it (`ANDREY_BENCH_JAVA`
names `java` when it is not on `PATH`):

```shell
curl -fLO https://repo1.maven.org/maven2/io/github/cmu-phil/causal-cmd/1.12.0/causal-cmd-1.12.0-jar-with-dependencies.jar
curl -fsL https://repo1.maven.org/maven2/io/github/cmu-phil/causal-cmd/1.12.0/causal-cmd-1.12.0-jar-with-dependencies.jar.sha1
sha1sum causal-cmd-1.12.0-jar-with-dependencies.jar  # b0e3c3fb3b0277e0a44c54dbb82f0023c4fd9e54
export ANDREY_BENCH_TETRAD_JAR=$PWD/causal-cmd-1.12.0-jar-with-dependencies.jar
```

The defaults are `d` of 20, 50, 100, and 200, five seeds, one repeat after one warm-up fit, and a
wall cap of 1800 s. A smoke run fits every solution once at `d = 10`:

```shell
$ANDREY_BENCH_PYTHON benchmarks/gfci_latent.py --out outputs/gfci-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/gfci_latent.py --out outputs/gfci-latent --cap-mem-mb 30000
```
