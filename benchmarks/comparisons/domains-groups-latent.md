# Domains, groups, and latent indicators

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
