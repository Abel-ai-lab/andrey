# Path and confounder claims

`knowledge_paths.py` measures the three `Knowledge` parts that state paths and hidden causes:
`required_paths` (`a` is an ancestor of `b`), `forbidden_paths` (`a` is not an ancestor of `b`),
and `confounded` (a hidden common cause of `a` and `b`). The driver checks them in three ways:
against the exact answer on every small graph, against the same fit without the claims on
simulated data, and against reference packages given the same knowledge. No reference takes a path
claim between variables that are not adjacent for PC, GES, BOSS, or FCI, and Tetrad's `Knowledge`
has no claim of a hidden common cause, so for those claims the evidence is the first two checks.

## Exactness

`oracle` answers every independence test by d-separation in the true DAG, so a search returns its
large-sample answer, and compares the claimed result with the answer that listing graphs gives.

| lane | graphs | methods | what must hold |
|---|---|---|---|
| `cpdag` | every DAG on 4 and 5 nodes; random DAGs on 6 to 8 nodes whose CPDAG has at most 11 undirected edges | `pc`, `ges`, `boss`, through the step they share | the MPDAG directs an edge exactly when every DAG of the class that meets the claims directs it alike; the MPDAG with its clauses keeps exactly those DAGs; when no DAG of the class meets every claim, some are dropped |
| `pag` | every DAG on 4 observed nodes with one hidden parent of two or more; random DAGs on 5 to 8 observed and 1 to 3 hidden nodes | `fci`; `gfci` and `boss_fci` with the score phase at its large-sample limit | every tail and arrowhead is the true MAG's, the adjacencies are the MAG's, and every confounded pair is adjacent |

The claims are drawn from the truth: in `cpdag`, 20 %, 50 %, and all of the ordered pairs, with
ancestor and non-ancestor facts in equal parts, and a fourth set with one false fact; in `pag`,
30 % and all of the ordered observed pairs, with every confounded pair. `pc` with d-separation
must return the true CPDAG, which the CPDAG step then restricts; `ges` and `boss` apply the same
step to their result. The score phase of `gfci` and `boss_fci` is replaced by its large-sample
limit, the CPDAG of the minimal I-map in a causal order of the DAG.

Where a PAG has at most 12 circles (`--max-circles`), `pag` also lists the MAGs of its class,
keeps those that meet the claims, and compares each mark they share with the fit's. The PAG rule
sets marks on adjacent pairs only, so a claim about a non-adjacent pair can leave a circle that
every kept MAG fills the same way; the count of such fits is reported and not judged.

```shell
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py oracle --lane cpdag --out outputs/kp-oracle-cpdag
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py oracle --lane pag --out outputs/kp-oracle-pag \
    --exhaustive 4 --graphs 1500
```

Each writes `oracle-<lane>.json` with the counts and up to 20 failing graphs, `provenance.json`,
and `done.json`.

## No worse than without the claims

`run` fits each dataset without claims (`bare`), with true claims at each fraction (`true`), and
with claims stated wrongly on 10 % of the ordered pairs (`false`, context only). The claims are
sampled from the true graph and passed as input; nothing else about the generator reaches a fit.

| lane | methods | data | sizes | claims |
|---|---|---|---|---|
| `cpdag` | `pc`, `ges`, `boss` | linear-Gaussian, Erdos-Renyi, standardized | `d` 10, 20, 50; `n` 500, 2000; expected degree 2 and 4 | 5, 10, 30 % of ordered pairs |
| `pag` | `fci`, `gfci`, `boss_fci` | linear-Gaussian, Erdos-Renyi, standardized, with hidden nodes | 10 observed and 2 hidden, 20 and 3; `n` 1000, 5000 | 10, 30 % of ordered observed pairs and of the confounded pairs |
| `lpcmci` | `lpcmci`, `tau_max = 3` | the simulation of Gerhardus and Runge (2020) | 5 and 10 series; `T` 500, 1000 | 10, 30 % of the window's lagged pairs and confounded links |
| `lingam` | `direct_lingam`; lingam's DirectLiNGAM with the claims as `prior_knowledge`, as context | linear, uniform noise, Erdos-Renyi, standardized | `d` 10, 20, 50; `n` 500, 2000 | 10, 30 % of ordered pairs |

Path claims take ancestor and non-ancestor facts in equal parts. In `pag`, `lpcmci`, and `lingam`,
`errors` counts the pairs adjacent in the fit and not in the truth, or the reverse, plus every tail
or arrowhead, on a pair adjacent in both, that differs from the true MAG of the observed variables
(`pag`), the true window MAG (`lpcmci`), or the true DAG (`lingam`). A circle states nothing and is
never wrong.

A CPDAG method cannot direct every edge, and true claims let it direct more, so `cpdag` scores a
fit against the class that the truth and the fit's claims define, with two of Andrey's metrics:

- `errors` is the structural Hamming distance (`andrey.metrics.shd`) to the true CPDAG for a fit
  without claims and, for a fit with true claims, to the MPDAG of the DAGs of the true class that
  meet them: the most a fit told those claims can direct. An edge the claims direct as the true DAG
  does is no error, an edge left undirected where the claims imply its direction is one, and so is
  a missing, extra, or reversed edge. The MPDAG comes from the step the exactness lane checks.
- `dag_errors` is the endpoint-aware distance (`shd(..., endpoint_aware=True)`) to the true DAG,
  the same target for every fit: an edge directed as the DAG directs it costs 0, an undirected
  edge 1, and a reversed, missing, or extra edge 2. A claim that directs an edge the right way
  lowers it by 1, and one that directs it the wrong way raises it by 1.

`wrong_marks` counts two for each edge directed against the true DAG, and `definite_marks` two for
each directed edge on a pair adjacent in the truth.

`empty_errors`, the empty graph's count, is the floor. A fit drops the claims the graph it found
cannot meet and records how many as `unmet`; a fit that raises `KnowledgeError`, for knowledge no
graph meets, is recorded as `knowledge_error`, with its message, and never scored as a graph.

```shell
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane cpdag --cell d20-n500-deg2 \
    --seeds 0 20 --out outputs/kp-cpdag-d20-n500-deg2
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane cpdag --cell d20-n500-deg2 \
    --seeds 100 120 --out outputs/kp-rep-cpdag-d20-n500-deg2
```

`--part required_paths`, `--part forbidden_paths`, or `--part confounded` keeps only that part of
each true claim set and leaves out the false arm, so each part is measured on its own.

Each run writes `records.jsonl`, one line per fit with its status, claim counts, and error columns;
`run_meta.json`; `provenance.json`; and `done.json`. A run resumes from its records. `--workers`
fits that many datasets at once, each in its own process; set one BLAS thread per process.

`summarize` pairs each `true` and `false` fit with the `bare` fit of its dataset and reports, per
method, cell, and fraction, the mean difference in `errors` and its 95 % interval over 10,000
bootstrap resamples, how many datasets each side wins, the `knowledge_error` count, and both arms'
distance from the floor. In `cpdag` it reports the same for `dag_errors`, in the `dag_` fields:

```shell
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py summarize --runs outputs/kp-cpdag-* \
    --replicate outputs/kp-rep-* --oracle outputs/kp-oracle-* --out outputs/kp-summary.json
```

A cell is `worse` when the interval lies above 0. A worse cell counts only when the same cell on
fresh seeds, the runs `--replicate` names, is worse too in the same measure. A cell outside a
method's stated setting has `in_setting` false: it is reported and left out of the method's
verdict. These are the `cpdag` cells at expected degree 4 for `ges`, which suits sparse graphs: its
release checks run at expected degree 2, and its measured limits at degree 4 hold for the standard
BIC with `max_parents=4`, not for the default settings the lane runs.

## Against reference packages

`run --methods` also fits reference packages on the same datasets, each unmodified in its own
environment, without knowledge and with the same true claims:

| method | reference | settings | knowledge |
|---|---|---|---|
| `pc` | Tetrad 7.6.10 PC | Fisher z, `alpha` 0.05 | required and forbidden edges as `Knowledge` |
| `pc` | pcalg `pc` | Gaussian test, `alpha` 0.05 | required pairs kept adjacent with `fixedEdges`, then each edge applied with `addBgKnowledge`, one at a time |
| `ges` | Tetrad 7.6.10 FGES | the plain BIC (SEM BIC, penalty discount 1), three passes of forward and backward search | `Knowledge` |
| `boss` | Tetrad 7.6.10 BOSS | SEM BIC, penalty discount 1 | `Knowledge` |
| `fci` | Tetrad 7.6.10 FCI | Fisher z, `alpha` 0.05, `excludeSelectionBias=true` | `Knowledge` |
| `boss_fci` | Tetrad 7.6.10 BOSS-FCI | `excludeSelectionBias=true` | `Knowledge` |
| `gfci` | GFCI of causal-cmd 1.12.0, which bundles tetrad-lib 7.6.3 | its defaults | `Knowledge` |
| `lpcmci` | tigramite 5.2.10.1 LPCMCI | ParCorr, `pc_alpha` 0.05 | the claims as `link_assumptions` |
| `direct_lingam` | lingam DirectLiNGAM | its defaults | the claims as `prior_knowledge` |

Tetrad's FCI steps orient by knowledge only without the rules for selection bias, so FCI and
BOSS-FCI leave those rules out with and without knowledge. pcalg's `addBgKnowledge` returns no
graph when the edges given together conflict with its CPDAG, so the driver applies them one at a
time and counts the edges it refuses; pcalg's `ges` has no knowledge argument, so GES has one
reference. FGES scores with the plain BIC and `ges` by default with the extended BIC
(`ebic_gamma=1`), so the lane also runs `ges` with `ebic_gamma=0` (`ges_bic`), the same score.

tigramite and lingam take the claims themselves. Tetrad and pcalg take edges, so each set of true
claims is translated to the edges it implies (`edge_claims`): a forbidden path from `a` to `b`
forbids the edge `a -> b`; a required path on a pair adjacent in the true graph is the required
edge `a -> b`; a required path on any other pair has no edge form, and neither has a confounded
pair. Both are left out and counted (`n_no_edge_form`, `n_confounded_no_edge_form`). A required
edge says more than the path it comes from: it also states that the pair is adjacent, which the
translation reads from the true graph.

Each method of the `cpdag` and `pag` lanes therefore runs in two arms beside the reference's:
`edges`, with the translated edges as `required` and `forbidden`, the same knowledge the reference
has; and `true`, with the path claims.

```shell
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane cpdag --seeds 0 20 \
    --methods pc ges ges_bic boss pcalg.pc tetrad.pc tetrad.ges tetrad.boss \
    --arms bare true edges --out outputs/kp-ref-cpdag
```

The references need R with pcalg (`benchmarks/start_r.sh`), tigramite's environment
(`benchmarks/start_tigramite.sh`, `ANDREY_BENCH_TIGRAMITE_PYTHON`), and Java with the two Tetrad
jars (`ANDREY_BENCH_JAVA`, `ANDREY_BENCH_TETRAD_GIN_JAR` for tetrad-lib 7.6.10, and
`ANDREY_BENCH_TETRAD_JAR` for causal-cmd). `summarize` adds `reference_rows`: per method, reference,
cell, part, and fraction, the mean over the datasets both fit of the method's count minus the
reference's (`dag_errors` in `cpdag`, `errors` elsewhere), its 95 % interval over 10,000 bootstrap
resamples, the repeat on fresh seeds for a worse row, and the method's distance from the floor. A
reference's own change with knowledge is a row of `rows` with the reference as its method.

## Results

Andrey's pre-release build of 9 November 2026 on the 0.16.0 release branch, on AMD EPYC 7713
processors with one BLAS thread per fit. Every number below is in
[`published/0.16.0/knowledge-paths.json`](../published/0.16.0/knowledge-paths.json), with every
row of every cell and the false-claims arm.

### Exactness

| Lane | Graphs | Checked | Failures |
|---|---|---|---|
| `cpdag` | 32,824: all 543 DAGs on 4 nodes, all 29,281 on 5, and 3,000 random on 6 to 8 | 131,296 claim sets: 103,315 kept every claim and gave the exact MPDAG with its clauses, and 27,981 dropped a claim where no DAG of the class meets them all | 0 |
| `pag` | 7,473: all 5,973 with 4 observed nodes and one hidden parent, and 1,500 random with 5 to 8 observed | 14,946 fits each of `fci`, `gfci`, and `boss_fci`, every mark sound | 0 |

`pc` with d-separation returned the true CPDAG on all 32,824 graphs, and 493 claim sets left
clauses. In `pag`, the class was small enough to list for 14,042 fits of each method: 12,685 set
every mark the kept MAGs share, and the others left 2,417 circles that every kept MAG fills alike.
With `--max-circles 8` the counts are 7,956, 7,030, and 1,631, and the fits are as sound.

### One part at a time

True claims of one part alone, seeds 0 to 19. A row is a cell at one fraction; every fit returned
a graph, so each row pairs all its datasets (in `pag` at 10 observed, 19: on one seed's graph the
generator could not hide two variables). D is the mean over a row's datasets of the errors
with the claims minus the errors without. Worse counts the rows whose 95 % interval of D lies above
0, Worse again those that were worse on seeds 100 to 119 too, and Above floor the rows whose
interval of errors minus the empty graph's errors reaches 0. Claims dropped is the share of the
true claims the fits dropped because the graph they found does not meet them. A part passes with
no row worse again and no row above the floor.

The CPDAG methods, with 20 datasets in every row. The first Worse, Worse again, Better, and D
columns are for `errors`, the distance to the class the claims leave, and the second four for
`dag_errors`, the distance to the true DAG; a row worse again in either would fail the part. Edges
directed counts, on pairs adjacent in the truth, the edges the fits directed with the claims
beyond those they directed without, and Against the true DAG those of them that point the wrong
way. The `ges` rows are its 18 at expected degree 2.

| Method | Part | Rows | Worse | Worse again | Better | D | Worse | Worse again | Better | D | Edges directed | Against the true DAG | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pc` | `required_paths` | 36 | 1 | 0 | 12 | -4.55 to +0.40 | 0 | 0 | 33 | -12.60 to -0.10 | 3,441 | 198 (5.8 %) | 0 | 22,615 of 37,726 (59.9 %) | yes |
| `pc` | `forbidden_paths` | 36 | 1 | 0 | 4 | -2.15 to +0.60 | 0 | 0 | 32 | -6.85 to -0.15 | 1,538 | 168 (10.9 %) | 0 | 12,434 of 52,320 (23.8 %) | yes |
| `ges` | `required_paths` | 18 | 0 | 0 | 9 | -1.25 to -0.05 | 0 | 0 | 18 | -12.20 to -0.80 | 2,121 | 24 (1.1 %) | 0 | 953 of 13,536 (7.0 %) | yes |
| `ges` | `forbidden_paths` | 18 | 0 | 0 | 2 | -0.50 to +0.50 | 0 | 0 | 18 | -7.15 to -0.60 | 828 | 14 (1.7 %) | 0 | 344 of 26,160 (1.3 %) | yes |
| `boss` | `required_paths` | 36 | 0 | 0 | 32 | -2.10 to -0.20 | 0 | 0 | 36 | -15.05 to -0.60 | 4,154 | 71 (1.7 %) | 0 | 1,100 of 37,726 (2.9 %) | yes |
| `boss` | `forbidden_paths` | 36 | 0 | 0 | 17 | -2.05 to +0.25 | 0 | 0 | 36 | -9.00 to -0.55 | 2,145 | 87 (4.1 %) | 0 | 418 of 52,320 (0.8 %) | yes |

With true claims no method was farther from the true DAG in any row, and each was closer in 32
to 36 of its 36 rows (`ges`: 18 of 18), by up to 15 per graph. Against the class the claims leave,
`pc` was worse in one row per part, both at expected degree 2 with 5 % of the pairs claimed (`d`
10 with 2,000 samples, and `d` 50 with 500), and neither repeated on fresh seeds. Both parts pass
this check in `pc`, `ges`, and `boss`.

The CPDAG step restricts the graph a search found, and its rules take that graph's class as given
(Fang et al. 2022, 2025). On finite data the class found can differ from the true one, and a true
claim then can direct an edge the wrong way: a required path from `a` to `b` whose first edge the
search removed forces `a` into another neighbor, and a collider the search missed lets a direction
pass through it. Of the edges the claims directed, 1.1 to 10.9 % pointed against the true DAG. Over
both seed groups, all 1,218 such edges of the single-part fits were in fits whose graph did not
stand for the true DAG, 1,209 with an edge missing or extra and 9 with the true adjacencies and a
wrong collider; in the 2,172 fits whose graph did, the claims directed 7,720 edges and none the
wrong way. `pc` also drops the most claims: its tests remove an edge on the only path a required
path needs.

In its 18 rows per part at expected degree 4, outside its setting, the `ges` fit with the default
settings does not stay below the empty graph in 4 of the 6 cells without claims, or in 19 of the
36 rows with them; there too no row is worse in either measure, and 31 of the 36 are closer to the
true DAG.

The other methods, whose `errors` count is the same for every fit:

| Method | Part | Rows | Datasets | Worse | Worse again | Better | D, least to most | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| `fci` | `required_paths` | 8 | 156 of 156 | 0 | 0 | 6 | -4.05 to -0.16 | 0 | 238 of 2,454 (9.7 %) | yes |
| `fci` | `forbidden_paths` | 8 | 156 of 156 | 0 | 0 | 0 | -0.15 to +0.42 | 0 | 13 of 3,686 (0.4 %) | yes |
| `fci` | `confounded` | 8 | 156 of 156 | 0 | 0 | 0 | -0.50 to +0.15 | 0 | 0 of 206 (0.0 %) | yes |
| `gfci` | `required_paths` | 8 | 156 of 156 | 0 | 0 | 5 | -1.85 to -0.11 | 0 | 195 of 2,454 (7.9 %) | yes |
| `gfci` | `forbidden_paths` | 8 | 156 of 156 | 0 | 0 | 0 | -0.10 to +0.42 | 0 | 16 of 3,686 (0.4 %) | yes |
| `gfci` | `confounded` | 8 | 156 of 156 | 0 | 0 | 1 | -0.55 to 0 | 0 | 0 of 206 (0.0 %) | yes |
| `boss_fci` | `required_paths` | 8 | 156 of 156 | 0 | 0 | 4 | -1.55 to 0 | 0 | 171 of 2,454 (7.0 %) | yes |
| `boss_fci` | `forbidden_paths` | 8 | 156 of 156 | 0 | 0 | 0 | -0.15 to +0.42 | 0 | 11 of 3,686 (0.3 %) | yes |
| `boss_fci` | `confounded` | 8 | 156 of 156 | 0 | 0 | 2 | -0.55 to -0.05 | 0 | 0 of 206 (0.0 %) | yes |
| `lpcmci` | `required_paths` | 8 | 160 of 160 | 0 | 0 | 3 | -1.90 to -0.20 | 0 | 59 of 2,027 (2.9 %) | yes |
| `lpcmci` | `forbidden_paths` | 8 | 160 of 160 | 0 | 0 | 1 | -1.50 to +0.45 | 0 | 3 of 1,960 (0.2 %) | yes |
| `lpcmci` | `confounded` | 8 | 160 of 160 | 0 | 0 | 6 | -1.80 to -0.10 | 0 | 0 of 345 (0.0 %) | yes |
| `direct_lingam` | `required_paths` | 12 | 240 of 240 | 0 | 0 | 0 | -0.25 to 0 | 0 | 0 of 10,616 (0.0 %) | yes |
| `direct_lingam` | `forbidden_paths` | 12 | 240 of 240 | 0 | 0 | 1 | -0.55 to +0.10 | 0 | 131 of 23,280 (0.6 %) | yes |

In the `pag` and `lpcmci` lanes a claim can set a mark that was a circle, right or wrong, and
`direct_lingam` returns a DAG. No row is worse, with one part or with all, and required paths
lower the errors of `fci`, `gfci`, `boss_fci`, and `lpcmci` in 18 of their 32 rows. No fit dropped
a `confounded` pair.

### All parts together

Every part of each true claim set at once, the same seeds and columns.

| Method | Part | Rows | Worse | Worse again | Better | D | Worse | Worse again | Better | D | Edges directed | Against the true DAG | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pc` | all parts | 36 | 0 | 0 | 13 | -4.60 to +0.45 | 0 | 0 | 35 | -12.60 to -0.20 | 3,566 | 201 (5.6 %) | 0 | 35,228 of 90,046 (39.1 %) | yes |
| `ges` | all parts | 18 | 0 | 0 | 9 | -1.25 to -0.05 | 0 | 0 | 18 | -12.20 to -1.20 | 2,168 | 24 (1.1 %) | 0 | 1,313 of 39,696 (3.3 %) | yes |
| `boss` | all parts | 36 | 0 | 0 | 34 | -2.20 to -0.30 | 0 | 0 | 36 | -15.05 to -0.95 | 4,434 | 89 (2.0 %) | 0 | 1,583 of 90,046 (1.8 %) | yes |

| Method | Part | Rows | Datasets | Worse | Worse again | Better | D, least to most | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| `fci` | all parts | 8 | 156 of 156 | 0 | 0 | 6 | -4.70 to -0.58 | 0 | 242 of 6,346 (3.8 %) | yes |
| `gfci` | all parts | 8 | 156 of 156 | 0 | 0 | 7 | -2.10 to -0.53 | 0 | 202 of 6,346 (3.2 %) | yes |
| `boss_fci` | all parts | 8 | 156 of 156 | 0 | 0 | 6 | -1.90 to -0.53 | 0 | 175 of 6,346 (2.8 %) | yes |
| `lpcmci` | all parts | 8 | 160 of 160 | 0 | 0 | 7 | -5.25 to -0.45 | 0 | 59 of 4,332 (1.4 %) | yes |
| `direct_lingam` | all parts | 12 | 240 of 240 | 0 | 0 | 1 | -0.75 to +0.05 | 0 | 130 of 33,896 (0.4 %) | yes |

### DirectLiNGAM and lingam

lingam's DirectLiNGAM takes the same claims as `prior_knowledge`. With `required_paths` alone it
fit every dataset, and Andrey tied it in all 12 rows. With `forbidden_paths` alone it raised
`ValueError` on 140 of 240 datasets; on the 100 it fit, Andrey made fewer errors in 5 of the 6
rows with pairs and tied the sixth. With all parts it fit 68 of 240, and Andrey tied 5 of those 6
rows and made fewer errors in the sixth. Without claims the two tied in all 6 cells.
`direct_lingam` returned a graph on every dataset.

### Against reference packages

Seeds 0 to 19, the cells and fractions above; a row pairs the datasets both sides fit. Each cell
gives the rows where the method made fewer errors than the reference, tied, made more, and made
more again on seeds 100 to 119, by the 95 % interval of the mean difference (`dag_errors` in
`cpdag`, `errors` elsewhere). Same knowledge compares the method and the reference given the same
edges, or the same claims for tigramite and lingam. The last column compares the method given the
path claims with the reference given the edges they imply. Every judged row of every method is
below the empty graph. The `ges` rows are its 18 at expected degree 2.

| Method | Reference | Part | Rows | Same knowledge | Path claims against the reference's edges |
|---|---|---|---|---|---|
| `pc` | pcalg `pc` | `required_paths` | 36 | 32 / 4 / 0 / 0 | 4 / 23 / 9 / 6 |
| `pc` | Tetrad PC | `required_paths` | 36 | 17 / 18 / 1 / 0 | 0 / 7 / 29 / 23 |
| `pc` | pcalg `pc` | `forbidden_paths` | 36 | 19 / 17 / 0 / 0 | 17 / 19 / 0 / 0 |
| `pc` | Tetrad PC | `forbidden_paths` | 36 | 9 / 27 / 0 / 0 | 12 / 23 / 1 / 0 |
| `pc` | pcalg `pc` | all parts | 36 | 33 / 3 / 0 / 0 | 4 / 23 / 9 / 6 |
| `pc` | Tetrad PC | all parts | 36 | 19 / 16 / 1 / 0 | 0 / 7 / 29 / 22 |
| `ges` | Tetrad FGES | `required_paths` | 18 | 16 / 2 / 0 / 0 | 7 / 11 / 0 / 0 |
| `ges` | Tetrad FGES | `forbidden_paths` | 18 | 11 / 7 / 0 / 0 | 13 / 5 / 0 / 0 |
| `ges` | Tetrad FGES | all parts | 18 | 16 / 2 / 0 / 0 | 7 / 9 / 2 / 0 |
| `ges`, `ebic_gamma=0` | Tetrad FGES | `required_paths` | 18 | 0 / 17 / 1 / 0 | 0 / 6 / 12 / 11 |
| `ges`, `ebic_gamma=0` | Tetrad FGES | `forbidden_paths` | 18 | 0 / 17 / 1 / 0 | 1 / 16 / 1 / 0 |
| `ges`, `ebic_gamma=0` | Tetrad FGES | all parts | 18 | 1 / 16 / 1 / 0 | 0 / 7 / 11 / 10 |
| `boss` | Tetrad BOSS | `required_paths` | 36 | 16 / 19 / 1 / 0 | 10 / 25 / 1 / 0 |
| `boss` | Tetrad BOSS | `forbidden_paths` | 36 | 1 / 32 / 3 / 0 | 4 / 32 / 0 / 0 |
| `boss` | Tetrad BOSS | all parts | 36 | 18 / 17 / 1 / 0 | 11 / 24 / 1 / 1 |
| `fci` | Tetrad FCI | `required_paths` | 8 | 5 / 3 / 0 / 0 | 0 / 3 / 5 / 4 |
| `fci` | Tetrad FCI | `forbidden_paths` | 8 | 2 / 6 / 0 / 0 | 2 / 6 / 0 / 0 |
| `fci` | Tetrad FCI | all parts | 8 | 4 / 4 / 0 / 0 | 0 / 6 / 2 / 2 |
| `gfci` | Tetrad GFCI | `required_paths` | 8 | 3 / 5 / 0 / 0 | 0 / 3 / 5 / 4 |
| `gfci` | Tetrad GFCI | `forbidden_paths` | 8 | 0 / 7 / 1 / 0 | 1 / 7 / 0 / 0 |
| `gfci` | Tetrad GFCI | all parts | 8 | 2 / 6 / 0 / 0 | 0 / 6 / 2 / 0 |
| `boss_fci` | Tetrad BOSS-FCI | `required_paths` | 8 | 5 / 3 / 0 / 0 | 1 / 6 / 1 / 0 |
| `boss_fci` | Tetrad BOSS-FCI | `forbidden_paths` | 8 | 1 / 7 / 0 / 0 | 1 / 7 / 0 / 0 |
| `boss_fci` | Tetrad BOSS-FCI | all parts | 8 | 6 / 2 / 0 / 0 | 5 / 3 / 0 / 0 |
| `lpcmci` | tigramite LPCMCI | `required_paths` | 8 | 0 / 8 / 0 / 0 |  |
| `lpcmci` | tigramite LPCMCI | `forbidden_paths` | 8 | 0 / 8 / 0 / 0 |  |
| `lpcmci` | tigramite LPCMCI | `confounded` | 8 | 0 / 8 / 0 / 0 |  |
| `lpcmci` | tigramite LPCMCI | all parts | 8 | 1 / 7 / 0 / 0 |  |
| `direct_lingam` | lingam DirectLiNGAM | `required_paths` | 12 | 0 / 12 / 0 / 0 |  |
| `direct_lingam` | lingam DirectLiNGAM | `forbidden_paths` | 6 | 5 / 1 / 0 / 0 |  |
| `direct_lingam` | lingam DirectLiNGAM | all parts | 6 | 1 / 5 / 0 / 0 |  |

Without knowledge, no method was worse than a reference in a row that repeated. Given the same
knowledge, none was either: with the same edges, `pc` made fewer errors than pcalg in 33 of 36
rows and than Tetrad's PC in 19, `boss` fewer than Tetrad's BOSS in 18, and `fci`, `gfci`, and
`boss_fci` fewer than Tetrad's in 4, 2, and 6 of 8; `lpcmci` tied tigramite, and `direct_lingam`
tied or beat lingam. `ges` with the plain BIC tied FGES in 16 of 18 rows with the edges and 5 of 6
without, and with its default extended BIC it made fewer errors in 16 of 18 rows.

`forbidden_paths` passes in every method: given the forbidden paths, no method was worse than its
reference given the forbidden edges in a row that repeated.

`required_paths` does not pass in `pc`, `ges`, `boss`, `fci`, or `gfci`. Given the required paths,
each made more errors than its reference given the required edges, in rows that repeated: `pc` in
23 of 36 rows against Tetrad and 6 against pcalg, `fci` and `gfci` in 4 of 8, `ges` with the plain
BIC in 11 of 18, and `boss` in one row of the all-parts runs (10 variables, expected degree 4,
2,000 samples, 30 % of the pairs). The cause is what the knowledge says, not how it is applied. A
required edge states that the pair is adjacent, and the reference puts the edge in; a required
path says only that one variable is an ancestor of the other, so these methods read it on the
graph the search found and cannot restore an edge the search left out. In the rows that repeated,
the fits with path claims kept the adjacency errors of the fits without claims (in `fci` with 20
observed variables, 1,000 samples, and 30 % of the pairs: 5.55 per graph with the paths and
without, 2.65 with the edges, and 3.55 for Tetrad with the edges). `ges` with its default score
was not worse than FGES in any row that repeated, because the extended BIC starts from fewer
errors than the plain BIC of FGES; with the same score it is, so the part stays experimental in
`ges` too. `boss_fci` was worse in one row that did not repeat, and passes.

Of the true required paths, 21,818 of 37,726 in `cpdag` (57.8 %) and 784 of 2,454 in `pag` (31.9 %)
were between variables that are not adjacent and had no edge form, and none of the 206 confounded
pairs had one. For those claims no reference exists, and the evidence is the exactness lanes and the
check against the same fit without the claims.

Each reference's own change with the knowledge, in `published/0.16.0/knowledge-paths.json` under
`reference_change_with_knowledge`: Tetrad's PC, FGES, FCI, and GFCI and pcalg's `pc` made fewer
errors with the edges in most rows and more in none that repeated. Tetrad's BOSS made more errors
with the required edges than without in 3 of 36 rows that repeated, by up to 62.55 per graph at 50
variables and expected degree 4. Tetrad's BOSS-FCI made more with the forbidden edges in one row
that repeated. lingam with forbidden paths made more errors than without in 4 of the 6 rows it fit.
Some reference fits returned no graph and are left out of their rows: FGES raised on a cycle in 11
of 2,160 fits with knowledge, tigramite refused the link assumptions of 15 of 640 fits and ran past
the 30-minute cap on 2, and lingam raised on 140 of 240 datasets with forbidden paths. pcalg's
`addBgKnowledge` refused 5,004 of 15,908 required edges and 539 of 52,320 forbidden edges.

### Status

A part is supported in a method when it passes all three checks: the exactness lane, no worse
than the same fit without the claims, and no worse than its references in any row that repeats.

| Part | Supported in |
|---|---|
| `forbidden_paths` | `pc`, `ges`, `boss`, `fci`, `gfci`, `boss_fci`, `lpcmci`, `direct_lingam` |
| `required_paths` | `boss_fci`, `lpcmci`, `direct_lingam` |
| `confounded` | `fci`, `gfci`, `boss_fci`, `lpcmci` |

`required_paths` is experimental in `pc`, `ges`, `boss`, `fci`, and `gfci`: it lowers their errors
against the same fit without it, and a required edge, where the pair is known to be adjacent,
lowers them more. Every part is experimental in the methods the driver does not run.
