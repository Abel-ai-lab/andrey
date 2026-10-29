# andrey-bench

Compare Andrey with other causal discovery packages and older Andrey builds.

## Published measurements

The site has a [release card](../docs/docs/benchmarks.md) per release, and each card has its own
folder under `published/` with the summary its numbers come from:

| Release | Release card | Summary |
|---|---|---|
| 0.4.0 | [Pairwise direction](../docs/docs/benchmarks/0.4.0.md) | [`published/0.4.0/summary.json`](published/0.4.0/summary.json) |
| 0.3.0 | [Score-based search](../docs/docs/benchmarks/0.3.0.md) | [`published/0.3.0/summary.json`](published/0.3.0/summary.json) |
| 0.2.0 | [Hidden common causes](../docs/docs/benchmarks/0.2.0.md) | [`published/0.2.0/summary.json`](published/0.2.0/summary.json) |
| 0.1.0 | [The alpha benchmarks](../docs/docs/benchmarks/0.1.0.md) | [`published/0.1.0/summary.json`](published/0.1.0/summary.json) |

A summary holds one row per comparison and size. It is generated from the recorded fits; those
records and the script that writes it are kept outside this repository. A released report and its
folder do not change: new measurements go into the next release's report and folder. To rerun a
comparison, use its driver with the builds, versions, and settings its report lists under
Reproduce ([0.4.0](../docs/docs/benchmarks/0.4.0.md#reproduce),
[0.3.0](../docs/docs/benchmarks/0.3.0.md#reproduce),
[0.2.0](../docs/docs/benchmarks/0.2.0.md#reproduce),
[0.1.0](../docs/docs/benchmarks/0.1.0.md#reproduce)).

[`published/parity-allowlist.json`](published/parity-allowlist.json) lists the rows where a
reference package makes fewer errors than Andrey by the parity rule, each with its cited cause; any
other such row fails the quality gate before a release
([reference parity](../qa/README.md#reference-parity)).

## Comparison packages

Packages:

- [causal-learn](https://github.com/py-why/causal-learn)
- [lingam](https://github.com/cdt15/lingam)
- [pcalg](https://cran.r-project.org/package=pcalg) (R)
- [bnlearn](https://www.bnlearn.com/) (R)
- [pgmpy](https://github.com/pgmpy/pgmpy) - see [Categorical data](#categorical-data)
- [Tetrad](https://github.com/cmu-phil/tetrad), through its causal-cmd jar and tetrad-lib's - see
  [Mixed data](#mixed-data), [GFCI](#gfci), and
  [Tetrad on continuous data](#tetrad-on-continuous-data)
- [xges](https://github.com/ANazaret/XGES) - uses a separate environment; see below
- [tigramite](https://github.com/jakobrunge/tigramite) - GPL-licensed; uses a separate environment
  and is compared by its outputs only; see below

## Setup

Requires [`uv`](https://docs.astral.sh/uv/).

```shell
benchmarks/start.sh
export ANDREY_BENCH_PYTHON=$PWD/benchmarks/.venv-bench/bin/python
```

The runner starts each fit in `.venv-bench`. Set `$ANDREY_BENCH_PYTHON` to benchmark another
checkout.

### The R lane

R solutions require an R module and an additional setup step:

```shell
benchmarks/start_r.sh          # R package library beside .venv-bench, then rpy2
eval "$(benchmarks/start_r.sh --env)"
```

`start_r.sh --env` prints the module load and the `R_LIBS_USER` export that every process fitting an
R solution needs, including batch jobs. `$ANDREY_BENCH_R_MODULE` names the modules (an R built
`--enable-R-shlib`, plus the compiler it was built with) and `$ANDREY_BENCH_R_LIBS` the library
directory.

`rpy2` embeds R in the benchmark worker. Memory and CPU accounting and the wall cap cover R. Startup
and `library()` run in the adapter's `setup()`, before warm-up fits and outside fit timing. See
[Where the clock starts](#where-the-clock-starts).

### Separate solution environments

xges pins `numpy<2` and `numba<0.60`, which bench-env cannot satisfy. Its fits use a second
environment:

```shell
benchmarks/start_xges.sh
export ANDREY_BENCH_XGES_PYTHON=$PWD/benchmarks/.venv-xges/bin/python
```

tigramite (GPL-3.0), the reference for the CMIknn and GPDC conditional-independence tests, also
fits in an environment of its own, so it and its dependencies stay out of bench-env:

```shell
benchmarks/start_tigramite.sh
export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
```

Its adapter calls the package's public API only.

lingam 1.13.0 pins scipy 1.13.1 or older, and pygam, which lingam imports, pins scipy below 1.17.
lingam's MultiGroupDirectLiNGAM runs in a third environment, on Python 3.12:

```shell
benchmarks/start_lingam.sh
export ANDREY_BENCH_LINGAM_PYTHON=$PWD/benchmarks/.venv-lingam/bin/python
```

An adapter selects another environment by setting `python_env` to the name of the variable that
holds its interpreter. The runner starts that solution's `fit` calls there. `to_structure` and
scoring remain in `.venv-bench`, so every solution uses the same graph model and scorer. An unset
declared variable stops the run at launch.

tigramite is GPL-licensed, so it is installed only in an environment of its own, and the harness
calls its public API there and compares the graphs it returns:

```shell
benchmarks/start_tigramite.sh
export ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python
```

Drivers pass each additional environment to `rundir.provenance_once(..., extra_venvs={...})`.
`run_id` excludes interpreter identity; `provenance.json` records every environment used, and
`aggregate.py` exposes additional environments in `extra_environments`.

### Older Andrey builds

An older Andrey build is a benchmark contestant. Give its adapter a distinct solution name and a
`python_env` pointing to the environment containing that build. Keep the adapter in an importable
module so the worker can reconstruct it, and record that environment in `extra_venvs`. Keep a
checkout's `src/` off `PYTHONPATH` so it cannot override the selected installation. Use the same
dataset, parameters, and resource limits for both builds.

For separate campaigns, `ANDREY_BENCH_PYTHON` selects the shared fit environment. Give each build
its own output directory and retain `provenance.json`: `run_id` excludes the commit, so a new build
must not resume into an older build's results. Compare paired campaign measurements; PR checks
do not measure speed beyond one direction-only test.

## Run

```shell
$ANDREY_BENCH_PYTHON benchmarks/benchmark.py --out outputs/smoke --ladder 20 --seeds 1 --repeats 1
cd benchmarks && .venv-bench/bin/python -m pytest tests/ -q
```

The first command should end with `table: 4 rows` and `errors: 0`. The second checks the harness
against known answers.

### FCI collider configurations

`fci_latent.py --include-majority` measures both Andrey collider rules alongside causal-learn FCI
and the empty-graph baseline. The default rule keeps the name `andrey.fci.numpy`; the optional rule
uses `andrey.fci.majority.numpy`. Each record includes `collider_rule` in its parameters. The same
datasets and resource limits apply to both configurations, with wall-clock timing and recovery
metrics recorded separately. Agreement with reference implementations is reported as evidence;
exact agreement is not a correctness requirement.

```shell
$ANDREY_BENCH_PYTHON benchmarks/fci_latent.py --out outputs/fci-colliders \
    --ladder 6 8 --seeds 3 --repeats 3 --include-majority
```

PAG recovery scores use the projection of the known generating DAG. That projection shares
orientation helpers with Andrey FCI, so it does not independently validate those helpers. The column-order
count is documented in [`qa/README.md`](../qa/README.md#fci-column-order).

## Writing a benchmark

Copy `benchmark.py`, then edit its `SOLUTIONS` and `SLICE`. Each adapter in
`andrey_bench/adapters/` represents one package, method, backend, and mode.

| method | solutions |
|---|---|
| GES | Andrey (numpy/serial, numpy/parallel), causal-learn, pcalg, Tetrad (FGES) |
| XGES (GES family) | Andrey and xges, each with and without the extended search |
| HC | Andrey, bnlearn (`hc`), pgmpy (categorical data) |
| Tabu | Andrey, bnlearn (`tabu`) |
| GIES | Andrey, pcalg (`gies`) |
| Exact search | Andrey, causal-learn (`bic_exact_search`) |
| PC | Andrey, causal-learn, pcalg, bnlearn (`pc.stable`), Tetrad |
| PC with nonlinear CI tests | Andrey (KCI, RCoT, CMIknn, GPDC), causal-learn (KCI, RCoT), tigramite (CMIknn, GPDC), pcalg with RCIT (RCoT) |
| FCI | Andrey, causal-learn, pcalg (`fci`, `rfci`), Tetrad |
| FCI with nonlinear CI tests | Andrey (KCI, RCoT, CMIknn, GPDC), causal-learn (KCI, RCoT) |
| Nonlinear CI tests, query by query | Andrey (KCI, RCoT, CMIknn, GPDC), causal-learn (KCI, RCoT, FastKCI), tigramite (CMIknn, GPDC), RCIT (RCoT, KCIT) |
| GFCI | Andrey, Tetrad |
| CALM | Andrey (two optimizer schedules), causal-learn |
| BOSS | Andrey, causal-learn, Tetrad |
| GRaSP | Andrey, causal-learn, Tetrad |
| DirectLiNGAM | Andrey, causal-learn, lingam |
| ICA-LiNGAM | Andrey, causal-learn |
| RCD, BottomUpParceLiNGAM | Andrey, lingam |
| ANM | Andrey, causal-learn |
| PNL | Andrey, causal-learn |
| CD-NOD | Andrey, causal-learn |
| GIN | Andrey, Tetrad, causal-learn |
| MultiGroupDirectLiNGAM | Andrey, lingam |
| PCMCI | Andrey (Student-t, Fisher-z), tigramite |
| PCMCI+, LPCMCI | Andrey (Student-t), tigramite |
| VARMA-LiNGAM, LongitudinalLiNGAM | Andrey, lingam |
| Granger (lasso) | Andrey (serial, parallel), causal-learn |

Include `baseline.empty` to bound the solutions run on the same datasets. Keep one method per file.
An adapter's `params` name the BIC penalty `lambda_value` and the scores `local_score_*` for every
package, the harness's recorded vocabulary; the Andrey adapters pass Andrey's `penalty_discount` and
score names (`bic`, `bdeu`, `discrete_bic`, `cond_gauss`, `degen_gauss`) to it.
DirectLiNGAM and ICA-LiNGAM need the `lingam_sf` values in `SLICE` because Gaussian data cannot
identify the model. Use `andrey.direct_lingam.torch-cuda` for GPU runs. It selects CUDA inside
`fit`, so `ANDREY_DEVICE` has no effect. Both ICA-LiNGAM solutions pin FastICA's seed to 0.

`nonlinear_ci_cpu.py` runs PC, or with `--method fci` FCI, with each nonlinear CI test on additive
nonlinear noise, with rows as an axis of their own. `--regime nonmonotone_er` makes each effect
`w cos(parent)`, an even function partial correlation does not see. Every reference runs at
Andrey's defaults under its own names:

| Andrey | causal-learn | tigramite | RCIT (R) |
|---|---|---|---|
| `indep_test="kci"` | `indep_test="kci"` | - | `KCIT(Bootstrap = FALSE)` |
| `indep_test="rcot"` | `indep_test="rcit"`, `rcit=False`, `approx="hbe"` | - | `RCoT(approx = "hbe")` |
| RCoT `n_features=5`, `n_features_cond=100` | `num_f2=5`, `num_f=100` | - | `num_f2 = 5`, `num_f = 100` |
| `indep_test="cmiknn"` | - | `CMIknn`, `significance="shuffle_test"`, `sig_blocklength=1` | - |
| CMIknn `k_cmi=0.1`, `k_perm=5`, `n_permutations=500` | - | `knn=0.1`, `shuffle_neighbors=5`, `sig_samples=500` | - |
| `indep_test="gpdc"` | - | `GPDC`, `significance="analytic"` | - |
| `alpha=0.05` | `alpha=0.05` | `pc_alpha=0.05`, `contemp_collider_rule="none"` | pcalg `alpha = 0.05` |

pcalg's PC with RCIT's RCoT (`pcalg.pc.rcot`) needs the R library of `start_r.sh`, which installs
RCIT from its GitHub repository at a fixed commit. `nonlinear_ci_queries.py` compares the tests
themselves: every contestant answers the same queries on the same seeded draws, independent and
dependent, and the summary gives each one's level, power, distance of its p-values to the uniform
under independence, agreement with Andrey's test, and seconds per query.

The harness does not submit jobs. One file is one job.

### Column order

A generated dataset lists its columns in an order drawn from its seed, with its truth permuted the
same way, so a method's accuracy cannot depend on the column order the simulation used. The shuffle
is part of the generator call: a shuffled dataset has its own `dataset_id` and a label ending in
`-shuffled`, the registry records its `column_order`, and `DatasetKey(shuffle=False)` keeps the
generated order under the `dataset_id` its records carry. The PCMCI, PCMCI+, LPCMCI, and
longitudinal LiNGAM simulations, GIN's `latent_measurement`, and the GIES lane's own simulation draw
an order that carries no information about the graph and take no shuffle step. The summaries
published up to 0.6.0 were measured on the generated order and stay as recorded.

`public_datasets.py` loads the bnlearn networks with `load_dataset`, which lists their columns in an
order drawn from the seed rather than the file's order, often a causal one; each record carries
`shuffle`. Sachs keeps its file's order, which does not follow its network. The public-datasets
rows published with 0.1.0 were measured on the file's order and stay as recorded.

### Scale

A key's `standardize` coordinate sets how `andrey.data` scales the values, and it enters the
generator call, so each scale is a dataset of its own. The linear and nonlinear regimes are
`standardized`: every column is z-scored, so no solution can order the variables by their variances
(var-sortability, Reisach et al. 2021). Z-scoring leaves each variable's R², the share of its
variance the other variables explain, as it was, and in linear data that share still grows down the
causal order (R²-sortability, Reisach et al. 2023).

The `_unitless` regimes, `linear_gauss_er_unitless`, `lingam_sf_unitless`,
`latent_gauss_er_unitless`, and `latent_lingam_er_unitless`, are the four linear regimes with the
weights and noise of Herman et al. (2025): every variable has variance 1 in the population, and a
variable's weights depend only on its number of parents. Their R²-sortability is close to chance as
well. Over 20 seeds at `d = 10, 20, 50` and `n = 10 d`, its mean is 0.46 to 0.62 against 0.65 to
0.76 for the standardized regimes; the hidden-confounder ones sit at the top of that range, since
hiding a node changes what the others explain. The generator keeps the variables' dense correlation
matrix, so a unitless key takes at most 1,000 variables and raises above that; run these regimes
at small sizes. A unitless key's data are not z-scored, so its sample variances differ from 1 by
sampling error.

Every record carries its dataset's `varsortability` and `r2_sortability` under the truth, over all
directed paths between observed variables, computed once per dataset in the parent; a time series
has neither. A solution's fit receives the data matrix alone.

### Hidden confounders

`latent_gauss_er` hides one node in ten, at least one, from an Erdos-Renyi linear-Gaussian graph.
Hidden nodes are chosen among nodes with two or more children, so hiding one confounds its children.
Set `latents` on the `DatasetKey` with `datasets.latents_for`, as `fci_latent.py` does. `d` includes
hidden nodes: the data has `d - latents` columns; the stored truth keeps all `d` nodes.

Every estimate is scored against the observed PAG, including the empty graph's CPDAG. The PAG is
built untimed in the parent, once per dataset. Its cost grows about 8x per doubling of `d`. A DAG or
CPDAG estimate is scored as a PAG without canonicalization. Its tails and undirected edges count
against the PAG's circles. Use PAG methods and the empty graph on these datasets.

### Hidden-confounder methods

`confounded_cpu.py` compares Andrey's `rcd` and `bottom_up_parce_lingam` with lingam's `RCD` and
`BottomUpParceLiNGAM`, with the empty-graph baseline. It holds both methods in one file because they
answer the same question: which variables share a hidden common cause. Both run on
`latent_lingam_er`, linear with uniform noise on an Erdos-Renyi graph that hides one node in five,
at least one, as in [Hidden confounders](#hidden-confounders); `d` includes the hidden nodes. Both
packages run each method with the same settings, listed in `params`.

The defaults are 2000 rows, `d` of 5, 10, and 15, five seeds, one repeat without warm-up, and
`--cap-wall-s` of 1800. A fit that reaches the cap is a `timeout` record with no scores. A smoke run
covers every solution on one small dataset:

```shell
$ANDREY_BENCH_PYTHON benchmarks/confounded_cpu.py --out outputs/confounded-smoke --smoke
```

Every estimate is a PAG and gets the observed-PAG scores of the FCI runs, but those do not rank
these methods: RCD uses its functional assumptions to mark `<->` where the observed PAG has `o-o`,
so the PAG scores count a correct `<->` as an error and BottomUpParceLiNGAM's `o-o` as a match. Read
two other scores instead.

The directed edges are scored against the latent projection, observed `i -> j` when a directed path
runs from `i` to `j` through hidden nodes only, less the edges whose endpoints share a hidden cause:
RCD and BottomUpParceLiNGAM estimate these, and mark a pair with a hidden cause instead of orienting
it. `directed_projection_*` holds `n_true`, `precision`, `recall`, `f1`, and `errors`, the edges in
one set and not the other (a reversed edge counts twice); `directed_n_est` counts the estimate's
edges.

A fit also names the pairs it attributes to a hidden common cause: the `<->` edges for RCD,
which match `metadata["confounded_pairs"]` of Andrey's result, and the `o-o` edges for
BottomUpParceLiNGAM. BottomUpParceLiNGAM's `o-o` edges join every pair of the one block of variables
it leaves unordered: the confounded variables, their observed ancestors, and any other confounded
pairs. Pairs in the block that share no cause count against its precision, and edges among the
block's variables stay unoriented, which caps its directed recall. Read its pair recall, the share
of true pairs it leaves unordered. `confounded_n_est` counts the pairs, and two sets of true pairs
score them:

| columns | true pairs |
|---|---|
| `confounded_hidden_cause_*` | the observed pairs that one hidden node reaches through hidden nodes only |
| `confounded_pag_*` | the observed pairs joined by `<->` in the observed PAG |

Each set has `n_true`, `precision`, `recall`, and `f1`. Score RCD and BottomUpParceLiNGAM on
`directed_projection_*` and `confounded_hidden_cause_*`: both mark a pair that shares a hidden
cause, even when one node is the other's parent. Compare each method across the two packages, not
RCD with BottomUpParceLiNGAM, whose block caps its scores as described above. The driver's summary
prints these. A PAG keeps `<->` only where conditional independences fix both
arrowheads, so `confounded_pag_*` has nothing to find on nearly every dataset at `d = 5`. A ratio
with no denominator is `NaN` rather than 1: report how many datasets define it, and read the
`n_true` columns beside the scores. The `confounder_*` columns from `andrey.metrics` score empty
against empty as 1 and do not see the `o-o` pairs of BottomUpParceLiNGAM.

The driver adds these columns to the record parts after each rung, from the estimates stored in
`structures/`. A resumed run scores only the units it measures, so a part scored by an older scorer
keeps its columns. A rescore recomputes every part's columns without fitting anything; pass the
first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/confounded_cpu.py --out outputs/confounded-cpu --rescore
```

### Domains, groups, and latent indicators

Three methods take data that is more than one plain sample. Their regimes come from
`andrey_bench/regime_models.py`, which gives each layout and truth; every dataset is still one
`(n, columns)` matrix, so the harness stores, fits, and scores it as any other. Each driver runs
Andrey and the reference with the same settings, listed in `params`, and the empty-graph baseline.
Its summary counts, in `same`, the datasets where the reference returned Andrey's graph.

| driver | method | reference | regime | data | scored against |
|---|---|---|---|---|---|
| `cdnod_cpu.py` | CD-NOD | causal-learn | `domain_shift_er` | `n` rows from four domains in equal blocks | the CPDAG CD-NOD can identify |
| `multi_group_cpu.py` | MultiGroupDirectLiNGAM | lingam | `multi_group_er` | three groups of `n` rows, side by side | the groups' DAGs, side by side |
| `gin_cpu.py` | GIN | Tetrad, causal-learn | `latent_measurement` | three indicators per latent | the latents' clusters and order |

`domain_shift_er` is a linear-Gaussian Erdos-Renyi DAG at mean degree 2 in which one variable in
three, at least one, shifts its mean over the domains, linearly in the domain's number, so the
Fisher-Z test of both packages can detect the shift. Both adapters rebuild the domain index,
Andrey's `domain_index` and causal-learn's `c_indx`, from the rows; the shuffle permutes the
variables' columns and leaves the rows, and so the domains, in place. The truth adds the index as a
source pointing into the shifted variables, takes that DAG's CPDAG, directs the index's edges out
of it, applies Meek's rules, and drops the index: an edge only the index can orient is directed.
That projection uses the orientation helpers of Andrey's CD-NOD, so it does not check them
independently. Andrey runs with its default `dependence_threshold=None`, which skips the
independent-change step, so the projection matches what it can orient. Both packages run Fisher-Z
at `alpha = 0.05` with a stable skeleton search and a collider rule that keeps an edge already
directed (causal-learn's `uc_rule=0, uc_priority=2`); the order in which colliders are visited can
still differ when two conflict.

`multi_group_er` gives every group the same DAG with weights drawn for each, and uniform noise. A
dataset holds the groups side by side as one sample of their block-diagonal DAG, and each solution
returns the groups' DAGs the same way, so `shd` is the sum over the groups. The shuffle draws one
order of the variables and applies it in every group, so each group keeps its block of columns, and
the recorded `column_order` covers all of them.

`latent_measurement` has `m` latents, each a cause of every later one, each with three indicators
in a random column order; `--ladder` gives `m`, and `d` counts latents and indicators. Andrey and
causal-learn run the gamma-approximation HSIC test at `alpha = 0.05` (causal-learn's
`indep_test_method="hsic"`; its default is KCI). Tetrad's GIN takes a raw marginal test, and runs
Tetrad's KCI under its defaults: a Gaussian kernel with a median-distance bandwidth and the gamma
approximation, the same family with its own bandwidth rule, at `alpha = 0.05`. causal-learn
0.1.4.8's GIN tests each candidate cluster against the wrong columns
([py-why/causal-learn#271](https://github.com/py-why/causal-learn/issues/271)), so that reference
is causal-learn with that one fix, which `start.sh` applies from `benchmarks/patches/`; Tetrad's GIN
needs no fix. Every package can leave a cluster unordered, and every adapter keeps it, with no
order. After each rung the driver
adds `measurement_cluster_ari`, the adjusted Rand index of the clustering, `measurement_n_latents`,
`measurement_clusters_found`, the share of true clusters found exactly, and
`measurement_order_accuracy`, the share of pairs of found latents put in the true order. A rescore
recomputes them without fitting anything; pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/gin_cpu.py --out outputs/gin-cpu --rescore
```

Tetrad's GIN runs as GFCI does (below): the adapter writes a CSV, runs the one-file driver
`TetradGinDriver.java` in one `java` call, and reads the clusters back, and `wall_s` is the search
inside the JVM. GIN needs tetrad-lib 7.6.10, which no causal-cmd release bundles, so it runs on
tetrad-lib's shaded jar, named by `ANDREY_BENCH_TETRAD_GIN_JAR`:

```shell
curl -fLO https://repo1.maven.org/maven2/io/github/cmu-phil/tetrad-lib/7.6.10/tetrad-lib-7.6.10-shaded.jar
curl -fsL https://repo1.maven.org/maven2/io/github/cmu-phil/tetrad-lib/7.6.10/tetrad-lib-7.6.10-shaded.jar.sha1
sha1sum tetrad-lib-7.6.10-shaded.jar  # 3a160a01a8cb648c03ceb8c7d5f47e5a8a1e2283
export ANDREY_BENCH_TETRAD_GIN_JAR=$PWD/tetrad-lib-7.6.10-shaded.jar
```

The three regimes rescale every variable to unit variance in causal order. A smoke run covers every
solution on one small dataset:

```shell
$ANDREY_BENCH_PYTHON benchmarks/cdnod_cpu.py --out outputs/cdnod-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/multi_group_cpu.py --out outputs/multi-group-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/gin_cpu.py --out outputs/gin-smoke --smoke
```

### GFCI

`gfci_latent.py` compares Andrey's GFCI with Tetrad's, with Andrey's FCI and the empty-graph
baseline beside them. Andrey's GFCI runs twice: `andrey.gfci.numpy` in one process, and
`andrey.gfci.numpy.parallel` with the GES step in the fit's cores as worker processes. Both
return the same PAG. It runs on `latent_gauss_er`, as `fci_latent.py` does (see
[Hidden confounders](#hidden-confounders)), and every estimate is scored against the observed PAG;
read `shd_endpoint`. Each package's names for the settings map to Andrey's:

| Andrey | Tetrad (causal-cmd) |
|---|---|
| `indep_test="fisherz"`, `alpha=0.05` | `--test fisher-z-test --alpha 0.05` |
| `score_func="bic"`, `penalty_discount=1.0` | `--score sem-bic-score --penaltyDiscount 1`, with `--semBicStructurePrior 0 --semBicRule 1` |
| no cap on conditioning sets or discriminating paths | `--depth -1 --maxPathLength -1` |
| possible-d-separation, the discriminating-path rule, and rules R1-R10 | `--possibleMsepDone --doDiscriminatingPathRule --completeRuleSetUsed` |

Tetrad runs through the driver of the [Mixed data](#mixed-data) lane, with one timing mechanism
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

### Tetrad on continuous data

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

`tetrad_cpu.py` runs on standardized linear-Gaussian Erdos-Renyi graphs with mean degree 2 and
`n = 10 * d` rows, and scores every estimate against the true graph's CPDAG; `tetrad_fci_latent.py`
runs on `latent_gauss_er` (see [Hidden confounders](#hidden-confounders)) and scores every estimate
against the observed PAG, so read `shd_endpoint`. Each solution's `params` list every setting it
runs, by Andrey's name where both packages have the setting and by Tetrad's setter name for the
rest. The settings map as follows:

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
one the [GIN](#hidden-confounder-methods) lane runs on, named by `ANDREY_BENCH_TETRAD_GIN_JAR`:

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

### CALM

`calm_cpu.py` compares Andrey's CALM with causal-learn's, which the CALM paper's first author
contributed, with the empty-graph baseline. The paper's code repository runs the same optimizer
but has no license, so the lane runs causal-learn's copy. It runs on the canonical sparse slice of
`ges_cpu.py`: linear-Gaussian data on an Erdos-Renyi graph at average degree 2, standardized, with
`n = 10 * d` rows. Each estimate is a DAG, scored against the true DAG (`shd`) and through its
equivalence class (`mec_shd`, the score the paper reports).

Andrey runs twice. `andrey.calm.torch-cpu` uses Andrey's defaults: 3,000 Adam steps per round at
learning rate 0.01, `rho` from 1e-3, at most 25 rounds, and an edge kept only when its weight is
at least 0.3. `andrey.calm.torch-cpu.paper_schedule` uses the paper's schedule, which is
causal-learn's default: 40,000 steps per round at learning rate 0.001, `rho` from 1e-5, at most
100 rounds, and no weight threshold. Compare causal-learn with the paper-schedule solution, which
runs its settings. Every CALM solution uses `lambda1 = 0.005`, a gate temperature of 0.5,
Fisher-Z moral-graph tests at `alpha = 0.01`, `rho` three times larger each round until the
acyclicity residual is below 1e-8, and seed 0; `params` records every setting. causal-learn's
names map to Andrey's:

| Andrey | causal-learn |
|---|---|
| `lambda1`, `alpha`, `tau`, `rho_init`, `rho_mult`, `subproblem_iter`, `standardize` | the same names |
| `h_tol` | `htol` |
| `lr=0.001`, `max_outer=100`, `rho_max=1e16`, `gate_threshold=0.5`, `w_threshold=0` | fixed inside `calm` |
| `seed=0` | `torch.manual_seed(0)` before the fit |

causal-learn's `calm` stops after 100 rounds whether or not its graph is acyclic; a cyclic graph
fails the conversion to a DAG and ends the fit in an error record. Andrey removes the weakest edge
of any cycle left after its last round. causal-learn also runs every step of every round, while
Andrey ends a round early once, after 50 steps, its loss changes by less than a 1e-7 fraction from
one step to the next, so the paper-schedule solution can run fewer steps than causal-learn.

The defaults are `d` of 10, 20, and 50, five seeds, one repeat without a warm-up fit, and a wall
cap of 3600 s. A smoke run fits every solution once at `d = 5`, and a run can keep to some
solutions:

```shell
$ANDREY_BENCH_PYTHON benchmarks/calm_cpu.py --out outputs/calm-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/calm_cpu.py --out outputs/calm-cpu --cap-mem-mb 30000
$ANDREY_BENCH_PYTHON benchmarks/calm_cpu.py --out outputs/calm-andrey --ladder 100 \
    --solutions andrey.calm.torch-cpu,baseline.empty
```

### GES parent limit

`andrey.ges(max_parents=k)` caps the parents a node has in any DAG of the returned CPDAG's class.
The limit of the reference packages is a gate: causal-learn's `maxP` stops a node that already
holds more than `maxP` directed parents from gaining new ones, and one move can still leave a node
above it. The Andrey GES adapter therefore reads `max_parents` through a private keyword of
`andrey.search.ges.ges`, `_parent_rule`, whose `"gate"` applies the limit as that gate and whose
default, `"cap"`, is the public behavior. A lane sets `parent_rule="gate"` on its Andrey arms only
when it limits the reference's parents too, so both packages apply one rule: `ges_cpu.py`,
`xges_ges.py`, `public_datasets.py`, and the BDeu pair of `discrete_cpu.py` run causal-learn with
the same limit. A lane that limits only Andrey, such as the R lane (pcalg has no parent limit),
keeps the cap, and an arm with `max_parents=None` has no limit under either rule. `params` records
the rule next to the limit.

### XGES

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

The package arms need their own environment, as in
[Separate solution environments](#separate-solution-environments). A smoke run fits every solution
once at `d = 10` on both densities:

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

### Tabu search

`tabu_cpu.py` compares Andrey's `tabu` with bnlearn's, with each package's `hc` beside them and the
empty-graph baseline. Tabu search is hill climbing that goes on past a local optimum, so the pair
shows what that costs and buys. Every solution is serial. The data is the R lane's slice:
linear-Gaussian on an Erdos-Renyi graph at average degree 2, with `n = 10 * d` rows. Both packages
use a tabu list of 10 DAGs and stop after 10 non-improving moves without a new best, bnlearn's
defaults. bnlearn 5.2.1 stops after `tabu` such moves whatever `max.tabu` is, so the two settings
stay equal.

The defaults are `d` of 10, 20, 50, and 100, five seeds, three repeats each after an untimed fit,
and `--cap-wall-s` of 1800. Load R as [The R lane](#the-r-lane) describes, then:

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

### ANM direction test

`anm_pairs.py` compares Andrey's `anm` with causal-learn's `ANM` on cause-effect pairs from the
`anm_pair` regime: two nodes at density 1.0, so every pair has one edge, in a direction the seed
draws, with a nonlinear additive-noise mechanism and Gaussian noise. The ladder runs over the
number of rows, not over `d`, since a pair always has two columns. causal-learn returns the two
p-values only, so its adapter decides the edge with Andrey's function: both solutions draw the
direction with the larger p-value. On equal p-values, Andrey takes the direction with the smaller
HSIC statistic relative to its mean under independence, and causal-learn draws no edge.

The defaults are sizes of 200, 500, 1000, and 2000 rows, 20 seeds, one repeat without warm-up, and
`--cap-wall-s` of 1800. The Gaussian-process regression's time grows with the cube of the rows. A
smoke run fits one pair of 100 rows:

```shell
$ANDREY_BENCH_PYTHON benchmarks/anm_pairs.py --out outputs/anm-smoke --smoke
```

Each record gets `direction_correct`, `direction_reversed`, and `direction_none`, one of them 1.0,
from the estimate stored in `structures/`. The summary prints, per size and solution, the median
wall time, the share of pairs with the true direction, and `agree`, the share of pairs on which the
two solutions return the same graph. A rescore recomputes the direction columns without fitting
anything, as in [Hidden-confounder methods](#hidden-confounder-methods); pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/anm_pairs.py --out outputs/anm-pairs --rescore
```

Both packages fit the same Gaussian process and the same HSIC test. Andrey standardizes each
column first, so its answer does not depend on the units; causal-learn fits the data as given. The
regime's data are standardized, so the two compute the same p-values here up to rounding, and the
comparison measures time.

### PNL direction test

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

### Categorical data

`discrete_cpu.py` compares Andrey's PC with the chi-square and the G-square test, GES, BOSS, and
GRaSP with the BDeu score, and hill climbing with the BDeu score and the multinomial BIC against the
same method in causal-learn, bnlearn, and pgmpy, with the empty-graph baseline. Andrey's BOSS and
GRaSP with the multinomial BIC run alone, since no package here takes that score for them. It runs
on `discrete_er`: category codes from a discrete network on an Erdos-Renyi graph, each node with 2
or 3 categories and each row of its conditional probability table drawn from a flat Dirichlet
(`datasets.DISCRETE_SETTINGS`, recorded in each dataset's generator call). The codes are never
rescaled. Every solution is serial. Andrey's GES runs on the numpy backend; BDeu moves are scored
in one process.

| method | Andrey | causal-learn | bnlearn | pgmpy |
|---|---|---|---|---|
| PC, chi-square | `andrey.pc.chisq` | `causal-learn.pc.chisq` | `bnlearn.pc-stable.chisq` | |
| PC, G-square | `andrey.pc.gsq` | `causal-learn.pc.gsq` | `bnlearn.pc-stable.gsq` | |
| GES, BDeu | `andrey.ges.bdeu` | `causal-learn.ges.bdeu` | | |
| BOSS, BDeu | `andrey.boss.bdeu` | `causal-learn.boss.bdeu` | | |
| GRaSP, BDeu | `andrey.grasp.bdeu` | `causal-learn.grasp.bdeu` | | |
| BOSS, BIC | `andrey.boss.bic_categorical` | | | |
| GRaSP, BIC | `andrey.grasp.bic_categorical` | | | |
| HC, BDeu | `andrey.hc.BDeu` | | `bnlearn.hc.BDeu` | `pgmpy.hc.BDeu` |
| HC, BIC | `andrey.hc.BIC_categorical` | | `bnlearn.hc.BIC_categorical` | `pgmpy.hc.BIC_categorical` |

A run can keep to some of these solutions, by name, with `baseline.empty`. Each pair runs one
method with one test or score and the same settings, each set explicitly and listed in `params`.
Where a package's default differs, the adapter overrides it:

| setting | Andrey | bnlearn | pgmpy 1.1.2 |
|---|---|---|---|
| PC test | `chisq`, `gsq` | `x2-adf`, `mi-adf` | |
| PC level, conditioning-set size | `alpha=0.05`, no limit | `alpha=0.05`, no limit (`max.sx`) | |
| BDeu equivalent sample size | `equivalent_sample_size=1` | `iss=1` | `equivalent_sample_size=1` (default 10) |
| BDeu structure prior | each other variable a parent with probability `1 / (d - 1)` | `prior="vsp"`, `beta = 1 / (d - 1)` | the same, added through `structure_prior_ratio` (pgmpy has none) |
| BIC penalty | `penalty_discount=1`: `log(n)` per parameter, in deviance units | `k = log(n) / 2`, in log-likelihood units | `log(n) / 2`, fixed |
| parent cap | none | `maxp=Inf` | `max_indegree=None` |
| move budget | `max_iter=100000` | `max.iter=100000` | `max_iter=100000` |
| restarts, tabu list | none | `restart=0` | `tabu_length=0` (default 100) |

Andrey's scores are `-2` times the log-likelihood scale of bnlearn and pgmpy. Hill climbing stops
at the first pass with no improving move; pgmpy's threshold is its default `epsilon=1e-4`, and
Andrey's is `1e-9` in deviance units, both far below the gain of a real move. pgmpy is pinned to
1.1.2, which installs on Python 3.13 with NumPy 2.5 and pandas 3 (`start.sh` installs it; the
driver stops when another version is installed). bnlearn is the R
library of [The R lane](#the-r-lane); its version is in `run_meta.json`.

bnlearn names Pearson's chi-square test `x2` and the G-square test `mi`, the mutual information
times `2 n`, which is the G statistic. Each has an `-adf` form that takes the empty rows and
columns of a table out of the degrees of freedom, as Andrey's tests do, so the `-adf` forms are the
ones paired with Andrey's: the plain `x2` and `mi` differ from Andrey's p-values on tables with an
empty row or column. The `-adf` p-values are Andrey's to rounding, with one exception: when a table
has fewer than 5 observations per cell on average (`n < 5 * r * c * q`, for `r` and `c` categories
in the two variables and `q` configurations of the conditioning set), bnlearn returns `p = 1` and
Andrey's tests have no such rule. At 2000 rows and three-category variables that starts at four
conditioning variables, where bnlearn's PC removes the edge and Andrey's PC tests it.

The defaults are 2000 rows, `d` of 10, 20, and 40, five seeds, one repeat after one warm-up fit, and
`--cap-wall-s` of 1800. Each estimate is scored against the true DAG's CPDAG: `shd` and the
skeleton and arrowhead columns. Andrey's solutions and causal-learn's return CPDAGs. bnlearn's and
pgmpy's hill climbing return DAGs, whose comparable columns are `mec_shd` and `mec_arrowhead_f1`,
scored through the DAG's essential graph. The summary prints each solution's median wall time and
error per `d`, with the F1 columns in the same comparable form, then each pair's speed-up: the other
package's median wall time over Andrey's on the datasets where both finished. A smoke run covers
every solution on one small dataset, and a rescore recomputes the score columns from the stored
estimates; pass the first run's flags. The bnlearn rows need the R lane and the pgmpy rows need
pgmpy; a run that names neither needs neither:

```shell
$ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py --out outputs/discrete-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py --out outputs/discrete-cpu --rescore
```

To run only some solutions, name them. `--resume` requires the first run's solution list, so a run
of other solutions takes a fresh `--out`; pointing `ANDREY_BENCH_DATA` at the first run's `data/`
gives it the same datasets, and `andrey_bench.aggregate.pool` joins the two runs' records:

```shell
ANDREY_BENCH_DATA=outputs/discrete-cpu/data $ANDREY_BENCH_PYTHON benchmarks/discrete_cpu.py \
    --out outputs/discrete-comparators --solutions andrey.hc.BDeu andrey.hc.BIC_categorical \
    bnlearn.pc-stable.chisq bnlearn.pc-stable.gsq bnlearn.hc.BDeu bnlearn.hc.BIC_categorical \
    pgmpy.hc.BDeu pgmpy.hc.BIC_categorical
```

The p-values and BDeu scores of Andrey and causal-learn follow the same definitions (Andrey's BDeu
is `-2` times causal-learn's), and on the datasets of `benchmarks/tests/test_adapters_discrete.py`
the two packages' PC and GES return the same graph, with Andrey's PC on the column-order collider
rule that causal-learn's PC runs; PC's default p-value order can orient conflicting colliders
differently. BOSS and GRaSP run on search seed 0 in both packages, but each package draws its orders
from its own random streams, so their graphs can differ at equal settings.
Each comparator's objective is checked against Andrey's in
`benchmarks/tests/test_adapters_discrete_comparators.py`, where R or pgmpy is installed: the scores
agree on a fixed DAG, bnlearn's hill climbing reaches the score Andrey's does on those datasets, and
each declared setting takes effect. pgmpy's search ends in a different local optimum on some
datasets: each package takes the first of equal moves in its own scan order, and the graphs can
differ at equal settings. pgmpy breaks ties by the iteration order of a set of column-name pairs, so
its columns are named by integer position, which keeps the graph the same under any
`PYTHONHASHSEED`. Read the error columns beside the speed-up: where the graphs differ, the
difference is a result to report.

### Mixed data

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
[GFCI](#gfci) lane's Tetrad solution runs the same driver and timing.

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

The bnlearn rows need the R lane: see [The R lane](#the-r-lane). The defaults are 2000 rows, `d` of
10, 20, and 40, five seeds, one repeat after one warm-up fit, and a wall cap of 1800 s. The summary
prints each solution's median wall time and error per `d`, then each pair's speed-up: the
comparator's median wall time over Andrey's on the datasets where both finished. A smoke run covers
every solution on one small dataset, a run can keep to some solutions, and a rescore recomputes the
score columns from the stored estimates; pass the first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-cpu --cap-mem-mb 32000 \
    --categorical-share 0.5 --categories 2 3
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-andrey --solutions \
    andrey.pc.degen_gauss andrey.ges.cond_gauss baseline.empty
$ANDREY_BENCH_PYTHON benchmarks/mixed_cpu.py --out outputs/mixed-cpu --rescore
```

### GIES

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

Each fit's graph is scored against the interventional essential graph of the generating DAG:
`shd` counts the variable pairs whose edge differs, and `empty_shd`, the true edge count, is the
empty graph's. `bic` is the graph's BIC on its data in Andrey's deviance units and `truth_bic` the
generating DAG's. The records go to `records.jsonl`, one line per fit, with each graph's endpoint
marks, and the build and machine to `provenance.json`; the summary counts the datasets where Andrey
returned pcalg's graph. Load R as [The R lane](#the-r-lane) describes, with pcalg in the library,
then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/gies_cpu.py --out outputs/gies-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/gies_cpu.py --out outputs/gies-cpu --cores 1 --cap-mem-mb 32000
```

The defaults are `d` of 10, 20, 50, and 100, five seeds, one untimed warm-up fit, one timed fit,
and `--cap-wall-s` of 1800. Both searches are serial.

### Exact search

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

### PCMCI on lagged series

`pcmci_cpu.py` compares Andrey's PCMCI with tigramite's, with the empty lag stack as the floor.
Time-series sizes are written N x T: N variables (nodes) and a series of T time steps (the window
length), and where a cell holds several trajectories or samples, M of them per N x T cell. A rung
is one N x T size; the driver runs every N in its ladder at every T in its lengths, and a
dataset is one series, stored with `d = N` and `n = T`.

Each method runs on the simulation of its own paper, drawn by `andrey_bench/temporal_models.py`,
which keeps the expected number of cross-links per variable fixed as N grows. Every variable
depends on its own previous step, and a model whose process is not stationary is drawn again. If
none of 1000 draws is stationary, as for PCMCI+ at N = 240 and seed 1, the first draw is kept and
its lagged coefficients are scaled together until the spectral radius is 0.95:

| regime | method | source | cross-links per variable | same-step share | lags |
|---|---|---|---|---|---|
| `pcmci_paper` | PCMCI | Runge et al. (2019), Supplementary Sect. S4.1 | 1 | 0 | 1-2 |
| `pcmciplus_paper` | PCMCI+ | Runge (2020), Sect. 4 | 1.5 | 30% | 1-5 |
| `lpcmci_paper` | LPCMCI | Gerhardus and Runge (2020), Sect. 4 | 1, 30% of series unobserved | 30% | 1-3 |

The coefficients, autocorrelations, and noise follow each paper as well; the module docstring lists
them. A truth without same-step links is stored as the marks of lags `1..L`. `var_lagged`, the
earlier regime, draws each lag-1 cross-link with probability 0.3, so its links per variable grow
with N.

Every solution tests lags 1 to `tau_max`, at `pc_alpha=0.2` and `alpha_level=0.05`:

| solution | test |
|---|---|
| `andrey.pcmci.student_t.tau<k>` | partial correlation, Student-t p-value (Andrey's default) |
| `andrey.pcmci.fisherz.tau<k>` | partial correlation, Fisher-z p-value |
| `tigramite.pcmci.parcorr.tau<k>` | `ParCorr(significance="analytic")`, Student-t p-value |
| `baseline.empty.temporal.tau<k>` | none: no link at any lag |

tigramite runs `run_pcmci` with `tau_min=1` and `fdr_method="none"` and its other defaults; its own
`tau_min=0` would also test lag 0, which is PCMCI+'s scope. `tau_max` is in every name because
`params` are not part of a `run_id`, and the record carries it as `tau_max`. Andrey's PCMCI runs
its tests with one BLAS thread whatever the core budget, while tigramite's fits get the budget's
BLAS threads; set the core budget to 1 to compare one core with one core. Andrey's Student-t test
and tigramite's analytic `ParCorr` share their test, so their p-values match. Andrey's PCMCI runs at
its default `fdr_method="fdr_bh"` and tigramite's at `"none"`, so their graphs differ in the links
the correction removes. The driver ends by listing, per dataset, the solutions that returned
identical graphs (`structure_hash`).

A temporal estimate is scored lag by lag with `andrey.metrics.temporal_scores`; its CPDAG and PAG
columns (`shd`, `skeleton_*`, `arrowhead_*`) stay empty. Each lag is a directed graph, so a link in
the wrong direction counts 2:

| columns | what they score |
|---|---|
| `lag<k>_shd`, `lag<k>_precision`, `lag<k>_recall`, `lag<k>_f1` | the links at lag `k`, for every lag in the estimate or the truth |
| `temporal_shd` | the lag-resolved SHD: the per-lag SHDs summed |
| `lagged_*` | the lags merged into one graph, self-loops included |
| `summary_*` | the lags merged into one graph, self-loops left out |

A lag with no true link has recall 1, and a lag with no estimated link precision 1, so read them
beside the SHD: with every true link at lag 1, the columns of lag 2 and above count false links
only. A smoke run covers every solution on one series of 200 steps over five variables:

```shell
ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python \
    $ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py --out outputs/pcmci-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py --out outputs/pcmci \
    --ladder 5 10 20 --lengths 500 1000 --tau-max 2 --cores 1
```

The driver refuses a series shorter than `2 * tau_max * (N + 1) + 3` steps, the shortest Andrey's
PCMCI accepts.

### PCMCI+ and LPCMCI on series with same-step links

`pcmci_cpu.py --method pcmciplus` and `--method lpcmci` compare Andrey's PCMCI+ and LPCMCI with
tigramite's on `pcmciplus_paper` and `lpcmci_paper`, where some links act within a time step. The
truth is a window over lags 0 to the largest true lag: the same-step DAG and the lagged links.
`lpcmci_paper` generates N series and drops 30% of them, so its rows are scored against the true
PAG over the observed series instead (`andrey_bench/window_pag.py`), the graph LPCMCI estimates
(Gerhardus and Runge, 2020, Sect. 2.3). It is built from the generating graph over lags 0 to
`tau_max`: the unobserved series and the observed ones before the window are marginalized, a link
is kept only when all its copies in the window are adjacent, and time order, the FCI rules, and
repeating orientations mark its ends. A truth with hidden series is rescored without fitting by
`rescore_window_pag.py`, which writes `window_pag.parquet`, one row per `run_id`.

Every solution tests lags 0 to `tau_max` (default 5 for PCMCI+ and 3 for LPCMCI, each paper's
largest true lag) with the Student-t partial-correlation test and the method's defaults, which are
both packages' defaults:

| solution | settings |
|---|---|
| `andrey.pcmciplus.student_t.tau<k>` | `pc_alpha=0.01`, `collider_rule="majority"` |
| `tigramite.pcmciplus.parcorr.tau<k>` | `run_pcmciplus(tau_min=0, pc_alpha=0.01, contemp_collider_rule="majority")` |
| `andrey.lpcmci.student_t.tau<k>` | `pc_alpha=0.05`, `n_preliminary_iterations=1` |
| `tigramite.lpcmci.parcorr.tau<k>` | `run_lpcmci(tau_min=0, pc_alpha=0.05, n_preliminary_iterations=1)` |
| `baseline.empty.window.tau<k>` | none: no link at any lag |

A link carries a mark at each end, so these solutions return window marks
(`andrey_bench/window.py`), an `(L + 1, n, n)` int8 stack: lag 0 as endpoint marks, and at each
lag `k >= 1` the mark at the earlier end of each link, whose later end is always an arrowhead. Both
packages' marks come from the same three-character link strings (`-->`, `o->`, `<->`, `o-o`,
`x-x`, ...): tigramite's `results["graph"]`, Andrey's LPCMCI `metadata["links"]`, and the same
strings built from Andrey's PCMCI+ structure and its `metadata["conflicts"]`. Two solutions return
the same graph when their marks are equal, conflicts included (`structure_hash`, `output_hash`).

A window estimate is scored link by link against the truth (`scoring.window_scores`); a link is a
same-step pair or a lagged pair, self-links included:

| columns | what they score |
|---|---|
| `endpoint_shd`, `lag0_endpoint_shd`, `lagged_endpoint_shd` | marks that differ from the truth's: two for a missing or extra link, one per wrong end |
| `adjacency_precision`, `adjacency_recall`, `adjacency_f1` | the links, whatever their marks |
| `wrong_marks` | tails and arrowheads on true links that contradict the truth |
| `lagged_tail`, `lagged_circle`, `lagged_arrow` | the true lagged links found, by the mark at their earlier end |

Without hidden series the truth's lag 0 is the DAG, not the class of graphs the data can tell
apart, so an unoriented PCMCI+ link or a PAG circle costs its open ends in `endpoint_shd`;
`wrong_marks` counts only the marks a solution claims. Against a true PAG a circle in both is a
match, and a tail or an arrowhead where the true PAG has a circle costs one in `endpoint_shd` but
is not in `wrong_marks`. A CPDAG's conflicting link is read as undirected and a PAG's conflict mark
as a circle, as Andrey returns them.

```shell
ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python \
    $ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py --method lpcmci --out outputs/lpcmci-smoke --smoke
```

The driver refuses a series shorter than the shortest Andrey's method accepts:
`2 * tau_max * (N + 1) + N + 1` steps for PCMCI+, `N * (tau_max + 1) + 2 * tau_max + 1` for LPCMCI.

### Lasso Granger causality

`granger_var.py` compares Andrey's `granger` with causal-learn's `Granger.granger_lasso` on
vector-autoregressive series with known lagged edges, drawn by `andrey.data`: two lags, Gaussian
noise, and no effects within a time step, so the true lagged edges are the nonzero entries of the
lag matrices. Every variable keeps its own past, and each other variable enters a lag with
probability `2 / (N - 1)`; the lag matrices are scaled so the series is stationary. Each variable
is z-scored before the fits, so both packages get the same standardized series, and both fit with
the same `maxlag` and `cv`. The data do not fit the dataset store, which holds one graph
per dataset, so the driver draws each series itself and calls the runner directly, as
`public_datasets.py` does.

The defaults are N of 5, 10, 20, 40, and 80 variables, T of 1000 time steps, five seeds, `maxlag` 2,
`cv` 5, one repeat without warm-up, and `--cap-wall-s` of 1800. A fit past the cap is a `timeout`
record with no scores. A smoke run fits one series of 5 variables:

```shell
$ANDREY_BENCH_PYTHON benchmarks/granger_var.py --out outputs/granger-smoke --smoke
```

A lagged edge is a nonzero coefficient at lags 1 to `maxlag`. `precision`, `recall`, and `f1`
score every lagged edge; `cross_precision`, `cross_recall`, and `cross_f1` leave out each
variable's edges from its own past, which every true series holds. The summary prints their means
per N and solution, the median wall time, and `agree`, the share of series on which every
solution returns the same lagged edges. Andrey runs twice: serially, and with its per-variable fits
on `--cores` threads (`andrey.granger.numpy.parallel`); both return the same coefficients.

Each fit is one JSON line in `records.jsonl`, with its coefficients. A resumed run skips the fits
already recorded, and a rescore recomputes the scores without fitting anything:

```shell
$ANDREY_BENCH_PYTHON benchmarks/granger_var.py --out outputs/granger-var --rescore
```

Both packages fit the same cross-validated lasso per variable. Andrey standardizes each variable
first and returns the coefficients in the data's units, so its edges do not depend on the units;
causal-learn fits the series as given, so on raw series with unequal variances the two keep
different small coefficients. The driver gives both the z-scored series, so `agree` compares the
same fit.

### Cross-link F1 of a time-series run

Every time-series row is scored by one measure: precision, recall, and F1 of the window causal
graph over its cross-links (`scoring.cross_link_scores`), the F1 of Assaad et al. (2022), Sect. 9.1,
with the autodependencies left out, as Runge et al. (2019) leave them out. A cross-link joins two
different variables: at a lag, oriented by time, or within a time step, counted when the two
variables are adjacent (`cross_*`) and, in `cross_oriented_*`, only when it also points the same
way. A panel pools its time points from the second. `rescore_cross_links.py` adds these scores to a
finished `pcmci_cpu.py` or `lingam_ts.py` run without fitting anything: it reads the stored graphs
and the dataset truths and writes `cross_links.parquet`, one row per `run_id`. A truth with
unobserved series is left out of these scores; LPCMCI on `lpcmci_paper` is scored against its true
PAG instead (`rescore_window_pag.py`). `granger_var.py` records the same scores
as `cross_precision`, `cross_recall`, and `cross_f1`.

```shell
$ANDREY_BENCH_PYTHON benchmarks/rescore_cross_links.py outputs/pcmci outputs/pcmciplus
```

### VARMA-LiNGAM and longitudinal LiNGAM

`lingam_ts.py --method varma` and `--method longitudinal` compare Andrey's VARMA-LiNGAM and
longitudinal LiNGAM with lingam's (`VARMALiNGAM`, `LongitudinalLiNGAM`), each on the simulation of
its paper, drawn by `andrey_bench/lingam_models.py`:

| regime | method | source | size |
|---|---|---|---|
| `varma_lingam_paper` | VARMA-LiNGAM | Hyvarinen et al. (2010), Sect. 5.4, plus one moving-average matrix | N x T, one series |
| `longitudinal_lingam_paper` | LongitudinalLiNGAM | Kadowaki et al. (2013), JSAI Sect. 6.1 | N x T, M units per time point |

`varma_lingam_paper` draws a same-step DAG with 0.8 N links (strictly lower-triangular before the
[column shuffle](#column-order)) and a lag-1 matrix with 2 N links, the paper's 40% of entries at
N = 5 held per variable as N grows, weights 0.05 to 0.5 in magnitude with a random sign, and
super-Gaussian noise. The paper's model has no moving-average part, so the series adds one
moving-average matrix drawn like the lag-1 one. `longitudinal_lingam_paper` draws, at every time
point, a dense same-step DAG with weights 0.5 to 1.5 in magnitude and a dense lag-1 matrix with
weights up to 0.5, a hidden common cause at the first time point, and non-Gaussian noise; Bach and
Jordan's (2002) densities that the paper names are specified in full for 5 of the 18, which are the
ones used. A time point whose lag matrix through the same-step part, `(I - B(t,t))^-1 B(t,t-1)`, has
spectral norm 1 or more (about 60% of the time points at N = 3 and every one from N = 5 on) has its
lag-1 matrix scaled down until that norm is 0.95, so every step contracts and the panel's scale
stays bounded as T grows; a time point below 1 keeps its draw. A panel's data has shape `(T, M, N)`,
stored with `d = N`, `n = M`, and `times = T`.

Both packages run the same settings: VARMA-LiNGAM with `order=(1, 1)`, no order search, and the lag
blocks pruned by adaptive Lasso (lingam's defaults are an order search by BIC and pruning);
LongitudinalLiNGAM with one lag and the pairwise likelihood-ratio measure. Every estimate is a
coefficient stack in the `[effect, cause]` layout both packages report
(`andrey_bench/coefficients.py`), scored by `scoring.coefficient_scores`:

| columns | what they score |
|---|---|
| `precision`, `recall`, `f1` | the directed cross-links of the same-step and lag-1 blocks, a link counted when its lag, cause, and effect match; `instantaneous_*` and `lagged_*` split them |
| `instantaneous_rmse`, `lagged_rmse`, `ma_rmse` | the root mean square error of the blocks' weights, the papers' accuracy measure |

Two solutions return the same graph when the same weights are nonzero (`structure_hash`). lingam
1.13.0, the release with both estimators, installs on Python 3.12, so its fits run in an environment
of their own:

```shell
benchmarks/start_lingam_ts.sh
export ANDREY_BENCH_LINGAM_TS_PYTHON=$PWD/benchmarks/.venv-lingam-ts/bin/python
$ANDREY_BENCH_PYTHON benchmarks/lingam_ts.py --method longitudinal --out outputs/longitudinal \
    --ladder 3 5 10 --lengths 20 50 --samples 200 1000 --seeds 10
```

Where the two VARMA-LiNGAM graphs differ, two causes remain. lingam's pruning builds the lagged
columns with `np.roll`, so its first regression row pairs `x(0)` with `x(T-1)` and `e(T-1)`; Andrey
starts the regressions at step `max(p, q)`. And the likelihood fit stops at 100 iterations, the
default of both, usually before it converges, so the same statsmodels call returns different
estimates under the different NumPy and SciPy versions of the two environments.

### Where the clock starts

Only `fit` is timed. Package imports, interpreter startup, and library loading belong in an optional
`setup()`, called once before warm-up fits.

At `--warmup 0`, the reach setting, the first fit includes any lazy imports inside `fit`.
Every adapter that imports a package declares a `setup()` except the xges package's, which imports
numba and XGES inside `fit`. Account for those imports when comparing fit times.

`setup_s` records setup duration and is included in `warmup_s`. At `--warmup 0`, they are equal and
provide no fit-time bound.

A package whose search runs in a process `fit` starts, such as Tetrad's JVM in every lane that
runs it, times that search itself: its adapter declares `search_s` and sets it in `fit` to the
seconds the process measured around the search call alone. The record keeps that time as `wall_s`
and the whole `fit` call as `process_s`, which is null for every other adapter. Its warm-up
searches run in the timed fit's process before its clock, so `warmup_s` holds only `setup_s`. See
`contracts.SolutionAdapter`.

### Measure the dispatch thresholds

Andrey uses hardcoded thresholds for accelerated paths:

- `ANDREY_GES_PARALLEL_MIN_WORK` selects the worker pool.
- `ANDREY_COV_GPU_THRESHOLD`, `ANDREY_ENTROPY_GPU_THRESHOLD`, `ANDREY_KCI_GPU_THRESHOLD`,
  `ANDREY_CMIKNN_GPU_THRESHOLD`, and `ANDREY_PNL_GPU_THRESHOLD` select the accelerator.

Three thresholds were calibrated once on one machine. Later measurements found errors of 7x for GES,
25x for `cov`, and 40x for entropy. Each error kept the faster path disabled.

**The worker-pool default suits a single fit, not a campaign.** A campaign runs many fits in one
process, and after the first pool start the pool can pay off at smaller sizes. `andrey calibrate`
measures the cutoff in the gate's unit, the largest estimated work of a pass. Repeated runs produce
a range because other node activity affects pool round-trip time.

Measure on the campaign node with the pool size used for fits, and **apply the measured
thresholds**. Recording an unused calibration is worse than no calibration: it records a threshold
that did not control the run.

**Check the unit before setting a threshold.** GES uses candidate evaluations, `cov` uses `n * d`,
`entropy` uses batch width `B`, and KCI, CMIknn, and PNL use `n * n`. `andrey.spec.environment`
lists these units, and `andrey --help --json` prints them. A value in the wrong unit silently
selects the wrong path.

**Backend pins affect the gates differently.** Worker-pool gates run on each search pass and ignore
the backend. Below the GES threshold, `ges.numpy.parallel` uses the serial path. Device thresholds
apply only to the `auto` path. `backend.select` gives priority to a pinned backend when it is
supported and available. `AndreyDirectLiNGAM` and `AndreyGES` pin inside `fit`, so
`torch-cuda` DirectLiNGAM uses the GPU at every size. This includes sizes where numpy is faster.
`AndreyPC` and `AndreyFCI` use `ANDREY_DEVICE` and the `auto` path, which may fall back to CPU.
Check the adapter before treating a lane name as the measured device.

Before a campaign:

1. Measure the crossover on the campaign machine. Run `andrey calibrate --workers <pool size>` for
   GES. For a device threshold, compare the primitive with numpy across the sizes used by the
   fits.
2. Set the environment variable and record it in `run_meta.json`. Declare every hyperparameter.
3. Record which side of the threshold contains each rung. Keep one rung where the accelerated path
   is slower.

`ANDREY_GPU_CALIBRATE=1` measures device thresholds at runtime for the `auto` path.
`_resolve_thresholds` calibrates when the active backend is `auto`. A pinned run keeps the static
threshold while recording that calibration was enabled. For pinned campaigns, use the primitive
sweep in step 1. The calibration grid is coarse and starts above some crossovers, so its result is
a bound.

## Flags

| flag | |
|---|---|
| `--out` | output directory, one per run. Required |
| `--ladder` | the `d` sizes, in the order given. Required |
| `--seeds` | datasets per rung. Default 3 |
| `--repeats` | timed repeats per (dataset, solution). Default 5 |
| `--warmup` | untimed fits per repeat. Default 1; use 0 for reach runs |
| `--cap-wall-s` | wall seconds one repeat may spend, imports and warm-up included. Default 1800 |
| `--cap-mem-mb` | memory one repeat may use. Default uncapped |
| `--mem-route` | cap enforcement: `poll` (default) or `slurm`. `none` with a cap is an error |
| `--cores` | core budget per fit. Defaults to the scheduler affinity mask |
| `--machine-id` | the machine name stamped on every record |
| `--resume` | continue the run in `--out`, skipping units already measured |

Every rung runs every solution. A rung that no solution finishes costs `cap_wall_s` per fit. Prefer
seeds over repeats: each seed creates a dataset, while repeats mostly measure startup again.

The cap covers imports, warmups, and the timed fit. `wall_s` covers only the timed fit, so a
solution can be killed having never been timed. Set `--cap-wall-s` for the total child runtime.
With `WARMUP = 1`, this is about twice the fit time plus import time. A child, and any worker pool
it opened, dies with the driver that started it, so a stopped run leaves nothing running. The
supervisor also kills the pool when the child exits, preserving the child's exit status in the
record.

Cost is `spawns * startup + fit time`, where
`spawns = solutions * rungs * seeds * repeats`. Each repeat starts one child, warms up, and fits.
Startup dominates below `d` ~ 200 and varies with cache state and machine load. The smoke run
uses four children, so measure `smoke_wall / 4`.

## Where output lives

Output has three levels. The harness manages only runs.

| | what it is |
|---|---|
| **results root** | directory holding many batches, named by `ANDREY_BENCH_OUTPUTS` |
| **batch** | one issue's worth of work, with its scripts and conditions. See [Batches](#batches) |
| **run** | one `--out`, driver, ladder, and `runs.parquet`; the unit managed by the harness |

```text
<results root>/
    <batch>/                        one batch
        campaign/<run>/             publishable seeds and repeats from one held node
        probes/<run>/               one seed, one repeat: ceilings and sizing only
        data/                       the shared dataset store (ANDREY_BENCH_DATA)
        scratch/                    ad-hoc scripts, submission files, and job logs
```

See [Output](#output) for the files within a run.

No code reads `ANDREY_BENCH_OUTPUTS`; it names the results root in the commands below. Drivers
write where `--out` points, and git ignores `outputs/` for local runs. Store real campaigns outside
the repository because a batch may use gigabytes.

## Batches

Batch scripts accompany the results:

```text
$ANDREY_BENCH_OUTPUTS/<batch>/
    run.sh            what was submitted
    CONDITIONS.md     what was measured, and the conditions the numbers depend on
    report.py         this batch's tables
    <driver>.py       any batch-specific driver
    campaign/  probes/  data/  scratch/
```

To start a batch:

1. Create `$ANDREY_BENCH_OUTPUTS/<batch>/`, named `<issue>-<slug>`.
2. Write `run.sh`. It points `ANDREY_BENCH_DATA` at the absolute path of `<batch>/data`, so every
   run reads the same datasets wherever it starts, and gives each driver an `--out` under
   `campaign/` or `probes/`. Job logs go to `scratch/`.
3. Record the core count, caps, isolation, ladder, and solution list in `CONDITIONS.md`.
4. Write `report.py`. It pools the batch with `andrey_bench.aggregate.pool` and keeps the rows
   whose `run_dir` starts with `campaign/` or `probes/`: `pool` reads every `records/` below the
   path it is given, so a run left in `scratch/` would join the probe rows. It checks that each
   dimension a table holds fixed - machine, build, cap, cores - takes one value, and prints the
   tables, with ratios from `andrey_bench.pairing`. It exits non-zero when a check fails.

From the repository root, run and report a batch with:

```shell
export ANDREY_BENCH_OUTPUTS=/path/to/results   # the root, above every batch
$ANDREY_BENCH_OUTPUTS/<batch>/run.sh
PYTHONPATH=benchmarks $ANDREY_BENCH_PYTHON $ANDREY_BENCH_OUTPUTS/<batch>/report.py \
    $ANDREY_BENCH_OUTPUTS/<batch>
```

Commit reusable changes such as harness fixes, comparison rules in `andrey_bench/pairing.py`, tests,
and guidance in [Reading a result](#reading-a-result).

## Output

| file | |
|---|---|
| `runs.parquet` | one row per timed fit, assembled from `records/` when the ladder finishes |
| `STATUS.json` | job, node, and units done against expected. `complete` stays false until the |
| | table assembles, so an interrupted run still shows how far it got |
| `records/` | one parquet per (dataset, solution), written as each finishes |
| `provenance.json` | package versions and BLAS of the bench venv (`$ANDREY_BENCH_PYTHON`'s when |
| | set), CPU, GPU, git sha. Captured once, so a resume keeps the recorded build |
| `run_meta.json` | flags and literals, written before the first fit |
| `data/registry.json` | each dataset's generator call and column order; under |
| | `$ANDREY_BENCH_DATA` when set |

Each fit uses a temporary compute-node directory for its pickled job, result, and start marker. The
runner removes the directory after the fit.

A record stores its data (`d`, `n`, `latents`, regime coordinates, `data_seed`, `repeat`), run
settings (`cap_wall_s`, `cap_mem_mb`, `cores`, `device`, `dtype`, `backend`, `mode`, `machine_id`, `params`),
and results (`status`, `wall_s`, `process_s`, `cpu_s`, `warmup_s`, `setup_s`, `peak_rss_mb`,
`output_hash`, `shd`, `skeleton_f1`, `arrowhead_f1`). A temporal solution's record also carries
`tau_max`, and its scores are the lag columns of [PCMCI on lagged series](#pcmci-on-lagged-series),
or the link columns of [PCMCI+ and LPCMCI](#pcmci-and-lpcmci-on-series-with-same-step-links) for a
window.

`device_id` records `cpu` or the CUDA device index and name. The backend determines the device, so a
pin inside `fit` outranks `ANDREY_DEVICE`; the record keeps the request separately as `device`.
`gpu_peak_alloc_mb` is null for CPU backends. For CUDA, it records peak tensor memory in PyTorch's
allocator during the timed fit. It excludes the reserved cache, CUDA context, and library
workspaces, so it understates the free card memory required.

Without `ANDREY_BENCH_DATA`, each run creates its own `data/` and copies every dataset it uses.
The shared store configured in [Batches](#batches) avoids those copies and gives every run identical
bytes.

`dataset_id` identifies a dataset. It is a content hash of the generator call and the `.npz`
filename. `run_id` identifies one measurement from the dataset, solution, repeat, caps, dtype,
device, backend, mode, and thread split.

`run_id` excludes the commit, so two builds in one environment have the same ID. An editable install
can also change under an ID. `provenance.json` identifies the build for each output directory. Keep
that file with records combined from several directories.

## Resume

Use `--resume` to continue an interrupted run from its completed dataset/solution units:

```shell
$ANDREY_BENCH_PYTHON benchmarks/benchmark.py --out outputs/smoke --ladder 20 \
    --seeds 1 --repeats 1 --resume
```

A part filename includes its dataset, solution, and environment. Resume skips existing part files.
A different cap, core count, or machine creates new measurements. Set `--machine-id` when the
scheduler may choose another node; the default hostname would rerun every measurement.

`--resume` compares `run_meta.json` and accepts only the same run settings. It prints any
difference. Starting without `--resume` fails when `records/` contains files.

Resume cannot detect solution edits because part filenames exclude `params` and no digest includes
the body of `fit`. Freeze the code during a run.

The run declares `cores` and `device`, so timeout records include them. The child reports `threads`,
`n_jobs`, and `device_id`; these fields are empty when the fit does not return.

Each failure produces a record. A stopped fit includes `status=timeout` or `oom` and its cap.
`error_text` describes other failures.

`status=blocked` means the step was queued and dropped. With `--mem-route slurm`, a step that asks
for more memory than the job owns waits until the wall cap. Treat that size as a coverage gap. Give
the job more memory than the cap.

Read `params` before comparing solutions because their defaults may optimize different objectives.

## Reading a result

### Before the run

- **Write down the expected result before running.**
- **Run a pilot at one seed and one repeat**, and keep its rows.
- **Include the sizes where the favorite might lose.**
- **Put `baseline.empty` in every `SOLUTIONS` list.**
- **Choose every hyperparameter, and record it in `run_id`.**
- **Use one `--cap-wall-s` everywhere.**
- **Set `--warmup 0` when measuring the largest size a method completes.**

### Conditions

- **Time on an uncontended node.**
- **Pin the node with `-w`** across whatever is compared.
- **Repeat anything surprising.**
- **Don't run a method past the size where it times out.** Take larger sizes in a separate run.

### Comparing fairly

- **Match the quantity measured by each parameter.** A penalty in deviance units is twice the same
  penalty in log-likelihood units. `adapters/bic_penalty.py` derives both BIC coefficients from one
  value. A cap on parents is not a cap on degree.
- **Feed every method the same data through the same scorer.**
- **Read `output_type` before comparing metrics.** DAG and CPDAG scores use different units. A DAG
  search returns one member of the equivalence class and is charged for unidentifiable orientations.
  A `dag` row's `mec_shd` and `mec_arrowhead_f1` score it through its essential graph, in the units
  of a `cpdag` row's `shd` and `arrowhead_f1`; a `cpdag` row has no `mec_*` columns. `bnlearn.hc`,
  `bnlearn.tabu`, and the DirectLiNGAM and ICA-LiNGAM solutions return DAGs; the other
  structure-learning solutions return a CPDAG or PAG.
- **Score a PAG against the truth's PAG.** `scoring.score_result` projects the DAG truth with
  `andrey.data.latent.marginal` before scoring a `pag` solution, so correct FCI circles incur no
  penalty. Scoring against the DAG penalizes even the true PAG. `shd` counts one per pair with
  differing marks; `shd_endpoint` counts each differing endpoint.
- **Compare graphs with `scoring.structure_hash`.** Equal metrics can come from different graphs.
  Between families, hash the canonicalized graph instead: a DAG's hash never equals its CPDAG's.
- **Save the estimated graph with the metrics.**
- **Report the size of a difference**: how large, against what spread, over how many measurements.

### Reading the table

- **Pair a ratio within one run** with `pairing.paired_ratios`.
- **Group on `dataset_id`.**
- **Name the method a speedup is against.**
- **Read `status` before `wall_s`.** `timeout` means the fit ran and hit the cap; `blocked` means
  the step was dropped before it ran.
- **Do not infer a ceiling from which method stopped first.** Each method has its own per-fit clock.
- **Do not call a timeout a cap artifact from an extrapolation.** These curves steepen, so a fitted
  exponent gives a lower bound.
- **Read `1/speedup` as a serial fraction only when the gate sits upstream of the pinned
  config.**

### Guarding yourself

- **Assert the grouping key in a test.**
- **Check the harness, the estimator and the grouping key.**
- **Support a retraction with evidence.**

## Layout

- `benchmark.py` - defines solutions, slice, and ladder, then writes one `runs.parquet`. Copy it.
- `xges_ges.py` - compares serial and parallel Andrey GES, causal-learn GES, and XGES at two BIC
  weights on CPU, with the baseline.
- `andrey_bench/aggregate.py` - pools a results root and joins each run with its provenance.
- `andrey_bench/pairing.py` - defines valid comparisons, including run pairing, dataset grouping,
  duration skew, and timeout meaning.
- `ges_cpu.py` - compares serial and parallel Andrey GES, causal-learn, and the baseline on CPU.
- `boss_grasp_cpu.py` - compares BOSS and GRaSP with causal-learn at matched BIC penalties.
- `boss_grasp_reach.py` - extends the paired BOSS comparison beyond GRaSP's measured range.
- `lingam_gpu.py` - compares DirectLiNGAM on CPU and GPU with non-Gaussian data.
- `lingam_cpu.py` - compares DirectLiNGAM and ICA-LiNGAM with causal-learn on CPU.
- `fci_latent.py` - compares FCI with causal-learn on data with hidden confounders.
- `pcmci_cpu.py` - compares PCMCI, PCMCI+, or LPCMCI with tigramite on series, over variables and
  series lengths.
- `lingam_ts.py` - compares VARMA-LiNGAM and longitudinal LiNGAM with lingam on their papers'
  simulations.
- `rescore_cross_links.py` - adds cross-link F1 to a finished time-series run, from its stored
  graphs.
- `confounded_cpu.py` - compares RCD and BottomUpParceLiNGAM with lingam on data with
  hidden confounders.
- `gfci_latent.py` - compares GFCI with Tetrad on data with hidden confounders.
- `tetrad_cpu.py` - compares PC, GES, BOSS, and GRaSP with Tetrad's PC, FGES, BOSS, and GRaSP on
  continuous data. `tetrad_fci_latent.py` compares FCI with Tetrad's on data with hidden
  confounders.
- `calm_cpu.py` - compares CALM with causal-learn on linear-Gaussian data.
- `cdnod_cpu.py` - compares CD-NOD with causal-learn on data pooled from several domains.
- `multi_group_cpu.py` - compares MultiGroupDirectLiNGAM with lingam on several groups of data.
- `gin_cpu.py` - compares GIN with Tetrad and causal-learn on latent variables and their indicators.
- `start_lingam.sh` - builds the environment of lingam's fits.
- `xges_cpu.py` - compares Andrey's XGES, with and without the extended search, with the xges
  package's and with Andrey GES, by time, SHD, and BIC under one scorer.
- `tabu_cpu.py` - compares tabu search and hill climbing with bnlearn, and scores each graph's
  BIC.
- `anm_pairs.py` - compares the ANM direction test with causal-learn on cause-effect pairs.
- `pnl_pairs.py` - compares the PNL direction test with causal-learn on post-nonlinear pairs.
- `nonlinear_ci_cpu.py` - compares PC and FCI with the nonlinear CI tests with causal-learn,
  tigramite, and pcalg with RCIT on additive nonlinear noise.
- `nonlinear_ci_queries.py` - compares the nonlinear CI tests query by query with causal-learn,
  tigramite, and RCIT. `start_tigramite.sh` builds the environment of tigramite's fits.
- `discrete_cpu.py` - compares PC with the chi-square and G-square tests, GES, BOSS, and GRaSP with
  BDeu, and hill climbing with BDeu and the multinomial BIC with causal-learn, bnlearn, and pgmpy on
  categorical data, and runs BOSS and GRaSP with the multinomial BIC. `bnlearn_discrete.py` and
  `pgmpy_hc.py` adapt bnlearn's and pgmpy's categorical tests and scores.
- `mixed_cpu.py` - compares PC, GES, and hill climbing with the degenerate and conditional Gaussian
  tests and scores with Tetrad and bnlearn on mixed data.
- `gies_cpu.py` - compares GIES with pcalg on simulated interventional data.
- `exact_cpu.py` - compares exact search with causal-learn on small graphs, and scores each DAG's
  BIC.
- `granger_var.py` - compares lasso Granger causality with causal-learn on vector-autoregressive
  series.
- `public_datasets.py` - compares every package on Sachs and the bnlearn networks from
  `andrey.data.load_dataset`.
- `r_packages.py` - compares Andrey with pcalg and bnlearn, one lane per job.
- `start_r.sh` - builds the R package library and installs `rpy2`.
- `start_xges.sh`, `start_tigramite.sh`, `start_lingam_ts.sh` - build the separate xges, tigramite,
  and lingam time-series environments.
- `andrey_bench/contracts.py` - adapter protocol, record schema, dataset and run identity.
- `andrey_bench/adapters/` - one adapter per package, method, backend, and mode.
- `andrey_bench/rsession.py` - embedded R interpreter, data transfer, and result retrieval within
  the R adapters' timed fits.
- `andrey_bench/runner.py` - one subprocess per fit, fit-only timing, caps. Each fit runs under
  a supervisor that kills its process group when the driver dies.
- `andrey_bench/rundir.py` - manages record parts, resume, and assembly in the output directory.
- `andrey_bench/memcap.py` - caps process-tree memory or uses `srun --mem=`.
- `andrey_bench/datasets.py` - dataset store and regime vocabulary; every solution reads identical
  bytes.
- `andrey_bench/regime_models.py` - the data of the domain, group, and latent-indicator regimes.
- `andrey_bench/integration.py` - runs dataset key -> data -> fit -> structure -> score -> record.
- `andrey_bench/scoring.py` - computes metrics and the cross-package structure hash.

- `andrey_bench/oracle.py` - a solution returning the true graph; a correct pipeline scores SHD 0.
- `andrey_bench/provenance.py` - reads versions, BLAS, CPU, and GPU from the benchmark environment.
- `tests/test_validation.py`, `tests/test_memory_cap.py` - the validation suite.
- `tests/test_contract.py` - dataset digest, record schema, environment fields, one unstubbed run.
