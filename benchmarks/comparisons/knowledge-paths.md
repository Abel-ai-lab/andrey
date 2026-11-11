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
has; and `true`, with the path claims. The two answer different questions. `edges` against the
reference is like for like, and decides a part's status. `true` against the reference's edges
compares a path claim with the edge it implies, which says more; it is reported and does not decide
the status.

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
edges, or the same claims for tigramite and lingam: the like-for-like comparison, which decides a
part's status. The last column compares the method given the path claims with the reference given
the edges they imply; it is not like for like, and is shown for what it says about paths and edges.
Every row of every method is below the empty graph. `ges` is judged in its setting: its rows are
its 18 at expected degree 2.

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
tied or beat lingam. FGES scores with the plain BIC, so `ges` with `ebic_gamma=0` is its
like-for-like arm: it tied FGES in 16 of 18 rows with the edges and 5 of 6 without, and was worse
in no row that repeated. With its default extended BIC, `ges` made fewer errors than FGES in 16 of
18 rows. Every part passes this comparison in every method.

The last column is not like for like. A required edge says more than the required path it comes
from: it also states that the pair is adjacent, which the translation reads from the true graph,
and the reference puts the edge in. A required path says only that one variable is an ancestor of
the other, so a method reads it on the graph the search found and cannot restore an edge the search
left out. With forbidden paths, no method was worse than its reference given the forbidden edges in
a row that repeated. With required paths, `pc`, `ges` with the plain BIC, `fci`, and `gfci` made
more errors than their references given the required edges, in rows that repeated: `pc` in 23 of 36
rows against Tetrad and 6 against pcalg, `fci` and `gfci` in 4 of 8, and `ges` with the plain BIC in
11 of 18. `boss` did in one row of the all-parts runs (10 variables, expected degree 4, 2,000
samples, 30 % of the pairs), and `boss_fci` was worse in one row that did not repeat. In the rows
that repeated, the fits with path claims kept the adjacency errors of the fits without claims (in
`fci` with 20 observed variables, 1,000 samples, and 30 % of the pairs: 5.55 per graph with the
paths and without, 2.65 with the edges, and 3.55 for Tetrad with the edges). With its default
score, `ges` was not worse than FGES in any row that repeated here, because the extended BIC starts
from fewer errors than the plain BIC of FGES.

So when you know the edge itself, state it as `required`. A path claim says less, and so directs
less.

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

## The other methods

The 14 other methods that honor these parts ran in the same lanes, or in a lane with data of their
kind, on Andrey's pre-release build of 10 November 2026 on the 0.17.0 release branch, on AMD EPYC
7713 processors with one BLAS thread per fit. The fits with path claims of `cdnod`, `iamb`,
`inter_iamb`, and `mmpc` are from a later build of that day, which reads claims on
[a graph with no consistent extension](#a-graph-with-no-consistent-extension) as that section
describes. Every number below is in
[`published/0.17.0/knowledge-paths.json`](../published/0.17.0/knowledge-paths.json). The exactness
lanes above cover these methods through the step they share: the CPDAG step for the 11 that return a
CPDAG, the PAG rule for `fci_jci` and `grasp_fci`, and the order rule of `direct_lingam` for
`multi_group_direct_lingam`.

| lane | methods | data | sizes | claims |
|---|---|---|---|---|
| `cpdag` | `xges`, `hc`, `tabu`, `mmhc`, `rsmax2`, `grasp`, `exact_search`, `iamb`, `inter_iamb`, `mmpc` | as above | as above; `exact_search` at `d` 10 and, with expected degree 2, `d` 20 | 5, 10, 30 % of ordered pairs |
| `cdnod` | `cdnod` | 4 domains whose means shift, linear-Gaussian, Erdos-Renyi | `d` 10, 20, 50; `n` 500, 2000; expected degree 2 and 4 | 5, 10, 30 % of ordered pairs |
| `jci` | `fci_jci` | two domains and one context variable, hidden confounders, linear-Gaussian, the simulation of Mooij et al. (2020) | 10 and 20 system variables; 500 and 2000 rows per domain | 10, 30 % of ordered pairs and of the confounded pairs |
| `pag` | `grasp_fci` | as above | as above | as above |
| `multigroup` | `multi_group_direct_lingam` | 3 groups that share one DAG, linear, uniform noise | `d` 10, 20, 50; `n` 500, 2000 per group | 10, 30 % of ordered pairs |

`iamb`, `inter_iamb`, and `mmpc` run with `structure="cpdag"` and no `target`, the only output in
which a path claim has a meaning: with a `target` or an undirected `structure` the path parts raise
`NotImplementedError`, and those outputs are not measured. `cdnod` is judged by `dag_errors`, its
distance to the true DAG; its `errors`, the distance to the class the domain index identifies, are
reported. In `multigroup` a fit returns one graph per group, and `errors` and the floor are sums
over the 3 groups. `hc`, `tabu`, and `xges` suit sparse graphs as `ges` does, so their rows at
expected degree 4 are reported and left out of their verdicts.

| method | reference | settings | knowledge |
|---|---|---|---|
| `cdnod` | causal-learn 0.1.4.8 CD-NOD | Fisher z, `alpha` 0.05 | required and forbidden edges as `background_knowledge` |
| `hc`, `tabu` | bnlearn 5.2.1 `hc`, `tabu` | score `bic-g`, the plain BIC | required edges as `whitelist`, forbidden edges as `blacklist` |
| `mmhc`, `rsmax2` | bnlearn 5.2.1 `mmhc`, `rsmax2` (`mmpc`, then `hc`) | Fisher z, `alpha` 0.05; `bic-g` | `whitelist`, `blacklist` |
| `iamb`, `inter_iamb` | bnlearn 5.2.1 `iamb`, `inter.iamb` | Fisher z, `alpha` 0.05 | `whitelist`, `blacklist` |
| `grasp_fci` | Tetrad 7.6.10 GRaSP-FCI | `excludeSelectionBias=true` | `Knowledge` |
| `multi_group_direct_lingam` | lingam 1.12.2 MultiGroupDirectLiNGAM | its defaults | the claims as `prior_knowledge` |

bnlearn scores with the plain BIC and `hc` and `tabu` by default with the extended BIC
(`ebic_gamma=1`), so the lane also runs them with `ebic_gamma=0` (`hc_bic`, `tabu_bic`), the same
score; that arm is like for like. bnlearn's DAG is read as its CPDAG with the listed arcs kept
directed. Five methods have no reference that takes the knowledge: the XGES package takes none, the
Tetrad driver passes none to GRaSP, no package with an exact search takes any, bnlearn's `mmpc`
returns only a skeleton, and how Tetrad's FCI with a context tier combines the tier with edge claims
is unverified. For `xges`, `grasp`, `exact_search`, `mmpc`, and `fci_jci` the comparison is with the
same fit without the claims only, as it is for `confounded` everywhere. With its default phases
`rsmax2` is the same search as `mmhc`, and their rows are equal.

```shell
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane cpdag --seeds 0 20 \
    --methods hc hc_bic tabu tabu_bic mmhc rsmax2 iamb inter_iamb bnlearn.hc bnlearn.tabu \
    bnlearn.mmhc bnlearn.rsmax2 bnlearn.iamb bnlearn.inter_iamb --arms bare true edges \
    --out outputs/kp-rest-cpdag
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane cdnod --seeds 0 20 \
    --methods cdnod causal_learn.cdnod --arms bare true edges --out outputs/kp-rest-cdnod
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py run --lane multigroup --seeds 0 20 \
    --methods multi_group_direct_lingam lingam.multi_group_direct_lingam --arms bare true \
    --out outputs/kp-rest-multigroup
```

`--lane jci --methods fci_jci`, `--lane pag --methods grasp_fci tetrad.grasp_fci`, and `--lane cpdag
--methods grasp xges mmpc exact_search` run the rest; `--part` and `--cell` select as above, and
`--seeds 100 120` gives the fresh seeds. bnlearn needs R with the package on `R_LIBS_USER`, and
causal-learn and lingam the benchmark environment.

### One part at a time, the other methods

The columns are those of the tables above. Every fit of every method returned a graph. Only the rows
that were worse on seeds 0 to 19 were fit again on seeds 100 to 119. Of the fresh datasets, one at
50 variables, 2,000 samples, and expected degree 4 is missing from the all-parts rows of `iamb` and
`inter_iamb`, which pair 19: the reference's fits on it ran past the run's budget, and the run was
stopped.

| Method | Part | Rows | Worse | Worse again | Better | D | Worse | Worse again | Better | D | Edges directed | Against the true DAG | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cdnod` | `required_paths` | 36 | 0 | 0 | 31 | -4.90 to +0.10 | 0 | 0 | 31 | -7.50 to -0.10 | 2,022 | 156 (7.7 %) | 0 | 20,370 of 37,750 (54.0 %) | yes |
| `cdnod` | `forbidden_paths` | 36 | 0 | 0 | 17 | -2.65 to 0 | 0 | 0 | 22 | -2.75 to 0 | 768 | 111 (14.5 %) | 0 | 20,095 of 52,320 (38.4 %) | yes |
| `xges` | `required_paths` | 18 | 0 | 0 | 1 | -0.50 to +0.05 | 0 | 0 | 18 | -12.75 to -0.85 | 2,191 | 3 (0.1 %) | 0 | 223 of 13,536 (1.6 %) | yes |
| `xges` | `forbidden_paths` | 18 | 0 | 0 | 0 | -0.30 to +0.40 | 0 | 0 | 18 | -7.45 to -0.60 | 836 | 3 (0.4 %) | 0 | 83 of 26,160 (0.3 %) | yes |
| `hc` | `required_paths` | 18 | 0 | 0 | 18 | -8.35 to -0.50 | 0 | 0 | 18 | -17.05 to -0.90 | 3,066 | 82 (2.7 %) | 0 | 2,377 of 13,536 (17.6 %) | yes |
| `hc` | `forbidden_paths` | 18 | 0 | 0 | 8 | -3.85 to -0.10 | 0 | 0 | 18 | -9.05 to -0.45 | 1,366 | 149 (10.9 %) | 0 | 944 of 26,160 (3.6 %) | yes |
| `tabu` | `required_paths` | 18 | 0 | 0 | 15 | -4.50 to -0.15 | 0 | 0 | 18 | -14.65 to -0.80 | 2,387 | 43 (1.8 %) | 0 | 2,079 of 13,536 (15.4 %) | yes |
| `tabu` | `forbidden_paths` | 18 | 0 | 0 | 6 | -2.35 to +0.05 | 0 | 0 | 18 | -8.10 to -0.45 | 972 | 70 (7.2 %) | 0 | 697 of 26,160 (2.7 %) | yes |
| `mmhc` | `required_paths` | 36 | 0 | 0 | 32 | -8.40 to +0.05 | 0 | 0 | 33 | -13.05 to +0.05 | 5,840 | 1,063 (18.2 %) | 0 | 19,691 of 37,726 (52.2 %) | yes |
| `mmhc` | `forbidden_paths` | 36 | 0 | 0 | 23 | -5.45 to +0.05 | 0 | 0 | 31 | -7.15 to +0.05 | 2,966 | 732 (24.7 %) | 0 | 1,453 of 52,320 (2.8 %) | yes |
| `rsmax2` | `required_paths` | 36 | 0 | 0 | 32 | -8.40 to +0.05 | 0 | 0 | 33 | -13.05 to +0.05 | 5,840 | 1,063 (18.2 %) | 0 | 19,691 of 37,726 (52.2 %) | yes |
| `rsmax2` | `forbidden_paths` | 36 | 0 | 0 | 23 | -5.45 to +0.05 | 0 | 0 | 31 | -7.15 to +0.05 | 2,966 | 732 (24.7 %) | 0 | 1,453 of 52,320 (2.8 %) | yes |
| `grasp` | `required_paths` | 36 | 0 | 0 | 12 | -1.20 to +0.05 | 0 | 0 | 35 | -13.45 to -0.15 | 3,429 | 57 (1.7 %) | 0 | 2,169 of 37,726 (5.7 %) | yes |
| `grasp` | `forbidden_paths` | 36 | 1 | 0 | 7 | -1.30 to +0.45 | 0 | 0 | 36 | -7.85 to -0.35 | 1,734 | 56 (3.2 %) | 0 | 1,072 of 52,320 (2.0 %) | yes |
| `exact_search` | `required_paths` | 18 | 0 | 0 | 2 | -0.50 to +0.10 | 0 | 0 | 16 | -5.30 to -0.20 | 846 | 6 (0.7 %) | 0 | 107 of 4,176 (2.6 %) | yes |
| `exact_search` | `forbidden_paths` | 18 | 0 | 0 | 1 | -0.55 to +0.10 | 0 | 0 | 18 | -2.70 to -0.40 | 426 | 6 (1.4 %) | 0 | 50 of 4,920 (1.0 %) | yes |
| `iamb` | `required_paths` | 36 | 0 | 0 | 25 | -6.80 to +0.10 | 0 | 0 | 35 | -13.75 to -0.05 | 3,785 | 244 (6.4 %) | 0 | 21,326 of 37,726 (56.5 %) | yes |
| `iamb` | `forbidden_paths` | 36 | 1 | 0 | 9 | -1.60 to +1.20 | 0 | 0 | 32 | -7.20 to 0 | 1,435 | 189 (13.2 %) | 0 | 26,689 of 52,320 (51.0 %) | yes |
| `inter_iamb` | `required_paths` | 36 | 0 | 0 | 24 | -6.60 to +0.10 | 0 | 0 | 35 | -13.75 to -0.05 | 3,802 | 247 (6.5 %) | 0 | 21,187 of 37,726 (56.2 %) | yes |
| `inter_iamb` | `forbidden_paths` | 36 | 1 | 0 | 8 | -1.35 to +1.20 | 0 | 0 | 31 | -7.20 to 0 | 1,459 | 198 (13.6 %) | 0 | 26,657 of 52,320 (50.9 %) | yes |
| `mmpc` | `required_paths` | 36 | 0 | 0 | 26 | -8.85 to +0.40 | 0 | 0 | 35 | -14.05 to -0.10 | 3,852 | 239 (6.2 %) | 0 | 23,297 of 37,726 (61.8 %) | yes |
| `mmpc` | `forbidden_paths` | 36 | 1 | 0 | 8 | -1.70 to +2.35 | 0 | 0 | 32 | -7.30 to -0.05 | 1,518 | 228 (15.0 %) | 0 | 31,287 of 52,320 (59.8 %) | yes |

| Method | Part | Rows | Datasets | Worse | Worse again | Better | D, least to most | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| `fci_jci` | `required_paths` | 8 | 160 of 160 | 0 | 0 | 8 | -13.65 to -2.25 | 0 | 294 of 2,990 (9.8 %) | yes |
| `fci_jci` | `forbidden_paths` | 8 | 160 of 160 | 0 | 0 | 0 | -0.25 to +0.10 | 0 | 14 of 3,720 (0.4 %) | yes |
| `fci_jci` | `confounded` | 8 | 160 of 160 | 0 | 0 | 4 | -1.55 to -0.15 | 0 | 0 of 544 (0.0 %) | yes |
| `grasp_fci` | `required_paths` | 8 | 156 of 156 | 0 | 0 | 6 | -1.90 to +0.05 | 0 | 171 of 2,454 (7.0 %) | yes |
| `grasp_fci` | `forbidden_paths` | 8 | 156 of 156 | 0 | 0 | 0 | -0.11 to +0.42 | 0 | 15 of 3,686 (0.4 %) | yes |
| `grasp_fci` | `confounded` | 8 | 156 of 156 | 0 | 0 | 2 | -0.55 to -0.05 | 0 | 0 of 206 (0.0 %) | yes |
| `multi_group_direct_lingam` | `required_paths` | 12 | 240 of 240 | 0 | 0 | 0 | 0 to 0 | 0 | 0 of 11,088 (0.0 %) | yes |
| `multi_group_direct_lingam` | `forbidden_paths` | 12 | 240 of 240 | 1 | 0 | 0 | -1.20 to +0.90 | 0 | 72 of 23,280 (0.3 %) | yes |

No method was farther from the true DAG with true claims than without in any row, and none was above
the empty graph. None was farther from the class the claims leave either in a row that was worse
again on fresh seeds. `grasp` was in one row with forbidden paths (50 variables, 500 samples,
expected degree 2, 10 % of the pairs), which did not repeat, and so were `iamb`, `inter_iamb`, and
`mmpc` in one row each with forbidden paths, at 50 variables and expected degree 2. The rows of
`cdnod`, `iamb`, `inter_iamb`, and `mmpc` are fits under the rule for
[a graph with no consistent extension](#a-graph-with-no-consistent-extension), where a claim directs
the undirected edge it names. Before the rule such a fit dropped every claim, and `iamb`,
`inter_iamb`, and `mmpc` were farther from the class at 50 variables in rows that repeated. Their
fits still drop 51 to 62 % of the claims, and with forbidden paths alone they do not pass a stricter
reading of the fresh seeds: that section has both. In `mmhc` and `rsmax2`, 18.2 % of the
edges that required paths directed and 24.7 % of those that forbidden paths directed point against
the true DAG, the most of any method, and the fits were still closer to the true DAG in 33 and 31 of
their 36 rows. `fci_jci`, `grasp_fci`, and `multi_group_direct_lingam` pass every part:
`multi_group_direct_lingam` was worse in one row with forbidden paths (50 variables, 500 samples per
group, 10 % of the pairs), by 0.9 errors summed over the groups, and on fresh seeds the difference
was 0.75 with an interval that reaches 0. Required paths changed no error count of
`multi_group_direct_lingam`: its order search already placed those pairs.

### All parts together, the other methods

| Method | Part | Rows | Worse | Worse again | Better | D | Worse | Worse again | Better | D | Edges directed | Against the true DAG | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `cdnod` | all parts | 36 | 0 | 0 | 33 | -4.90 to +0.10 | 0 | 0 | 32 | -7.50 to -0.20 | 2,099 | 170 (8.1 %) | 0 | 40,595 of 90,070 (45.1 %) | yes |
| `xges` | all parts | 18 | 0 | 0 | 1 | -0.50 to +0.05 | 0 | 0 | 18 | -12.75 to -1.25 | 2,238 | 3 (0.1 %) | 0 | 312 of 39,696 (0.8 %) | yes |
| `hc` | all parts | 18 | 0 | 0 | 18 | -8.35 to -0.50 | 0 | 0 | 18 | -17.05 to -1.15 | 3,140 | 83 (2.6 %) | 0 | 3,469 of 39,696 (8.7 %) | yes |
| `tabu` | all parts | 18 | 0 | 0 | 15 | -4.50 to -0.10 | 0 | 0 | 18 | -14.65 to -1.05 | 2,441 | 44 (1.8 %) | 0 | 2,843 of 39,696 (7.2 %) | yes |
| `mmhc` | all parts | 36 | 0 | 0 | 32 | -8.40 to -0.25 | 0 | 0 | 35 | -13.05 to -0.20 | 6,039 | 1,096 (18.1 %) | 0 | 21,609 of 90,046 (24.0 %) | yes |
| `rsmax2` | all parts | 36 | 0 | 0 | 32 | -8.40 to -0.25 | 0 | 0 | 35 | -13.05 to -0.20 | 6,039 | 1,096 (18.1 %) | 0 | 21,609 of 90,046 (24.0 %) | yes |
| `grasp` | all parts | 36 | 0 | 0 | 18 | -1.60 to -0.10 | 0 | 0 | 36 | -13.45 to -0.50 | 3,665 | 65 (1.8 %) | 0 | 3,291 of 90,046 (3.7 %) | yes |
| `exact_search` | all parts | 18 | 0 | 0 | 2 | -0.65 to +0.05 | 0 | 0 | 18 | -5.30 to -0.50 | 947 | 6 (0.6 %) | 0 | 160 of 9,096 (1.8 %) | yes |
| `iamb` | all parts | 36 | 0 | 0 | 25 | -6.85 to 0 | 0 | 0 | 35 | -13.75 to -0.05 | 3,931 | 259 (6.6 %) | 0 | 48,115 of 90,046 (53.4 %) | yes |
| `inter_iamb` | all parts | 36 | 0 | 0 | 25 | -6.65 to 0 | 0 | 0 | 35 | -13.75 to -0.05 | 3,966 | 267 (6.7 %) | 0 | 47,926 of 90,046 (53.2 %) | yes |
| `mmpc` | all parts | 36 | 0 | 0 | 28 | -8.90 to +0.15 | 0 | 0 | 35 | -14.05 to -0.25 | 4,017 | 252 (6.3 %) | 0 | 54,712 of 90,046 (60.8 %) | yes |

| Method | Part | Rows | Datasets | Worse | Worse again | Better | D, least to most | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| `fci_jci` | all parts | 8 | 160 of 160 | 0 | 0 | 8 | -16.90 to -2.65 | 0 | 317 of 7,254 (4.4 %) | yes |
| `grasp_fci` | all parts | 8 | 156 of 156 | 0 | 0 | 7 | -2.25 to -0.47 | 0 | 176 of 6,346 (2.8 %) | yes |
| `multi_group_direct_lingam` | all parts | 12 | 240 of 240 | 1 | 0 | 0 | -1.20 to +0.90 | 0 | 72 of 34,368 (0.2 %) | yes |

### Against reference packages, the other methods

The columns are those of the table above, with one difference: the last column gives fewer, tied,
and more on seeds 0 to 19 only. Its rows were not fit again on fresh seeds, as it does not decide a
status. `hc` and `tabu` are judged in their setting, their 18 rows at expected degree 2. lingam
raised `ValueError` on 152 of 240 datasets with forbidden paths and on 180 of 240 with all parts, on
every dataset at 50 variables, and those datasets are left out of its rows.

| Method | Reference | Part | Rows | Same knowledge | Path claims against the reference's edges |
|---|---|---|---|---|---|
| `cdnod` | causal-learn CD-NOD | `required_paths` | 36 | 32 / 4 / 0 / 0 | 0 / 16 / 20 |
| `cdnod` | causal-learn CD-NOD | `forbidden_paths` | 36 | 18 / 18 / 0 / 0 | 11 / 23 / 2 |
| `cdnod` | causal-learn CD-NOD | all parts | 36 | 33 / 3 / 0 / 0 | 0 / 14 / 22 |
| `hc` | bnlearn `hc` | `required_paths` | 18 | 12 / 6 / 0 / 0 | 0 / 6 / 12 |
| `hc` | bnlearn `hc` | `forbidden_paths` | 18 | 7 / 11 / 0 / 0 | 7 / 11 / 0 |
| `hc` | bnlearn `hc` | all parts | 18 | 12 / 6 / 0 / 0 | 0 / 5 / 13 |
| `hc`, `ebic_gamma=0` | bnlearn `hc` | `required_paths` | 18 | 0 / 17 / 1 / 0 | 0 / 3 / 15 |
| `hc`, `ebic_gamma=0` | bnlearn `hc` | `forbidden_paths` | 18 | 0 / 17 / 1 / 0 | 1 / 15 / 2 |
| `hc`, `ebic_gamma=0` | bnlearn `hc` | all parts | 18 | 0 / 16 / 2 / 0 | 0 / 3 / 15 |
| `tabu` | bnlearn `tabu` | `required_paths` | 18 | 14 / 4 / 0 / 0 | 0 / 15 / 3 |
| `tabu` | bnlearn `tabu` | `forbidden_paths` | 18 | 8 / 10 / 0 / 0 | 6 / 12 / 0 |
| `tabu` | bnlearn `tabu` | all parts | 18 | 14 / 4 / 0 / 0 | 0 / 14 / 4 |
| `tabu`, `ebic_gamma=0` | bnlearn `tabu` | `required_paths` | 18 | 0 / 17 / 1 / 1 | 0 / 4 / 14 |
| `tabu`, `ebic_gamma=0` | bnlearn `tabu` | `forbidden_paths` | 18 | 1 / 16 / 1 / 0 | 0 / 15 / 3 |
| `tabu`, `ebic_gamma=0` | bnlearn `tabu` | all parts | 18 | 0 / 16 / 2 / 0 | 0 / 3 / 15 |
| `mmhc` | bnlearn `mmhc` | `required_paths` | 36 | 10 / 23 / 3 / 2 | 1 / 4 / 31 |
| `mmhc` | bnlearn `mmhc` | `forbidden_paths` | 36 | 6 / 30 / 0 / 0 | 7 / 28 / 1 |
| `mmhc` | bnlearn `mmhc` | all parts | 36 | 7 / 23 / 6 / 4 | 0 / 4 / 32 |
| `rsmax2` | bnlearn `rsmax2` | `required_paths` | 36 | 10 / 23 / 3 / 2 | 1 / 4 / 31 |
| `rsmax2` | bnlearn `rsmax2` | `forbidden_paths` | 36 | 6 / 30 / 0 / 0 | 7 / 28 / 1 |
| `rsmax2` | bnlearn `rsmax2` | all parts | 36 | 7 / 23 / 6 / 4 | 0 / 4 / 32 |
| `iamb` | bnlearn `iamb` | `required_paths` | 36 | 0 / 35 / 1 / 0 | 0 / 6 / 30 |
| `iamb` | bnlearn `iamb` | `forbidden_paths` | 36 | 0 / 36 / 0 / 0 | 9 / 24 / 3 |
| `iamb` | bnlearn `iamb` | all parts | 36 | 0 / 34 / 2 / 0 | 0 / 7 / 29 |
| `inter_iamb` | bnlearn `inter.iamb` | `required_paths` | 36 | 0 / 34 / 2 / 0 | 0 / 8 / 28 |
| `inter_iamb` | bnlearn `inter.iamb` | `forbidden_paths` | 36 | 0 / 36 / 0 / 0 | 10 / 19 / 7 |
| `inter_iamb` | bnlearn `inter.iamb` | all parts | 36 | 0 / 33 / 3 / 0 | 0 / 7 / 29 |
| `grasp_fci` | Tetrad GRaSP-FCI | `required_paths` | 8 | 8 / 0 / 0 / 0 | 2 / 6 / 0 |
| `grasp_fci` | Tetrad GRaSP-FCI | `forbidden_paths` | 8 | 1 / 6 / 1 / 0 | 2 / 6 / 0 |
| `grasp_fci` | Tetrad GRaSP-FCI | all parts | 8 | 7 / 1 / 0 / 0 | 6 / 2 / 0 |
| `multi_group_direct_lingam` | lingam MultiGroupDirectLiNGAM | `required_paths` | 12 | 0 / 12 / 0 / 0 |  |
| `multi_group_direct_lingam` | lingam MultiGroupDirectLiNGAM | `forbidden_paths` | 6 | 4 / 2 / 0 / 0 |  |
| `multi_group_direct_lingam` | lingam MultiGroupDirectLiNGAM | all parts | 6 | 0 / 6 / 0 / 0 |  |

Without knowledge, no method was worse than its reference in any row. Given the same knowledge,
`cdnod` made fewer errors than causal-learn's CD-NOD in 33 of 36 rows with all parts and more in
none, `grasp_fci` fewer than Tetrad's GRaSP-FCI in 7 of 8, and `multi_group_direct_lingam` tied or
beat lingam in every row lingam fit. `iamb` and `inter_iamb` tied bnlearn in 33 or more of 36 rows
and were worse in no row that repeated. With its default score `hc` made fewer errors than bnlearn's
`hc` in 12 of 18 rows and `tabu` than bnlearn's `tabu` in 14, and neither more in any. With the
plain BIC, the like-for-like arm, `hc` was worse in no row that repeated. `tabu` with the plain BIC
was, in one row with the required edges alone (20 variables, 500 samples, 5 % of the pairs; 3.55
more errors per graph on seeds 0 to 19 and 3.15 on fresh seeds), so `required_paths` fails this
comparison in `tabu`. `mmhc` and `rsmax2` made more errors than bnlearn's given the same required
edges at expected degree 4, in 2 of 36 rows with required paths alone and 4 of 36 with all parts
that repeated, by 1.75 to 4.9 per graph, so `required_paths` fails it in both; with forbidden paths
they were worse in no row.

With the plain BIC, `hc` and `tabu` are not below the empty graph at 50 variables, 500 samples, and
expected degree 2, with the claims or without, and neither is bnlearn there: in 2 rows of `hc` and 1
of `tabu` with the forbidden edges. The floor is judged with a method's default settings, where
every row is below it, so these rows are reported (`plain_bic_floor` in the published file) and do
not decide a status. Outside their setting, at expected degree 4, `hc` and `tabu` with the same
edges were worse than bnlearn in up to 3 rows per part, and in none on fresh seeds.

Each reference's own change with the knowledge is under `reference_change_with_knowledge`. bnlearn's
searches and causal-learn's CD-NOD made fewer errors with the edges in most rows and more in none.
Tetrad's GRaSP-FCI made more errors with the forbidden edges than without in one row that repeated,
and so did `grasp_fci` given the same forbidden edges, in another: both are about forbidden edges,
not paths. lingam with forbidden paths made more errors than without in 4 of the 6 rows it fit.

<!-- end of the other methods -->

## A graph with no consistent extension

`pc`, `cdnod`, `iamb`, `inter_iamb`, and `mmpc` direct edges by tests, and on finite data the
tests can contradict each other. The graph found then has no consistent extension: it stands for no
DAG, and the CPDAG step has no class to restrict. Such a fit used to drop every path claim and
return the graph it found. From 0.17.0 a claim acts on such a graph one edge at a time:

- a claim on a pair joined by an undirected edge directs it, `a -> b` for a required path from
  `a` to `b` and `b -> a` for a forbidden one, and Meek's rules follow (Meek 1995);
- a claim that the directed edge of its pair already meets changes nothing;
- a claim on a pair that is not adjacent, a claim against an edge the data directed, and claims
  that conflict, by asking for both directions of one edge or by closing a directed cycle, are
  dropped and listed in `metadata["unmet_knowledge"]`.

`metadata["edge_claims"]` lists the claims applied to an edge. The reading of one claim is
published: in any DAG that joins `a` and `b` by an edge, `a` is an ancestor of `b` exactly when
the edge is `a -> b` (Fang and He 2020; Fang et al. 2025). Those results take the class of a DAG as
given, and no published result covers a graph that stands for none, so applying the reading there
is Andrey's own choice. The result is not guaranteed to stand for a DAG, as the graph found did not.
A fit without path claims, and a fit whose graph has a consistent extension, return what they
returned before.

The five methods' fits with path claims ran again under the rule, on every cell and fraction above
and seeds 0 to 19, on a pre-release build of 10 November 2026 on the 0.17.0 release branch, each
paired with the recorded fit of its dataset without claims. The fits without claims and the fits
with edges hold no path claim and were not refit: the build returned the recorded scores on each of
the 1,200 fits without claims and on each of 1,728 fits with edges (seeds 0 to 3). The rows of
`cdnod`, `iamb`, `inter_iamb`, and `mmpc` in [the tables above](#the-other-methods) are these
fits. `pc`'s rows in [Results](#results) are 0.16.0's, measured before the rule, and its rows under
the rule are below. Every number is in
[`published/0.17.0/knowledge-paths.json`](../published/0.17.0/knowledge-paths.json) under
`no_consistent_extension`.

### Against the fits before the rule

The same true claims, with the rule and before it, paired by dataset. D is the mean over a row's
20 datasets of the errors with the rule minus the errors before it. Fits changed counts the fits
whose graph or dropped claims differ. The first Better, Tied, Worse, and D columns are for
`errors`, the distance to the class the claims leave, and the second four for `dag_errors`, the
distance to the true DAG, by the 95 % interval of D.

| Method | Part | Rows | Fits changed | Better | Tied | Worse | D | Better | Tied | Worse | D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `pc` | `required_paths` | 36 | 119 of 720 | 14 | 22 | 0 | -5.35 to 0 | 14 | 22 | 0 | -5.35 to 0 |
| `pc` | `forbidden_paths` | 36 | 61 of 720 | 6 | 30 | 0 | -1.85 to 0 | 6 | 30 | 0 | -1.85 to 0 |
| `pc` | all parts | 36 | 123 of 720 | 16 | 20 | 0 | -5.40 to 0 | 16 | 20 | 0 | -5.40 to 0 |
| `cdnod` | `required_paths` | 36 | 139 of 720 | 19 | 17 | 0 | -4.70 to 0 | 18 | 18 | 0 | -4.70 to 0 |
| `cdnod` | `forbidden_paths` | 36 | 52 of 720 | 5 | 31 | 0 | -1.45 to 0 | 5 | 31 | 0 | -1.40 to 0 |
| `cdnod` | all parts | 36 | 145 of 720 | 19 | 17 | 0 | -4.70 to 0 | 18 | 18 | 0 | -4.70 to 0 |
| `iamb` | `required_paths` | 36 | 248 of 720 | 24 | 12 | 0 | -11.00 to 0 | 24 | 12 | 0 | -11.00 to 0 |
| `iamb` | `forbidden_paths` | 36 | 150 of 720 | 17 | 19 | 0 | -3.05 to 0 | 17 | 19 | 0 | -2.85 to +0.05 |
| `iamb` | all parts | 36 | 263 of 720 | 25 | 11 | 0 | -11.05 to -0.10 | 25 | 11 | 0 | -11.05 to 0 |
| `inter_iamb` | `required_paths` | 36 | 244 of 720 | 24 | 12 | 0 | -10.95 to 0 | 24 | 12 | 0 | -10.95 to 0 |
| `inter_iamb` | `forbidden_paths` | 36 | 148 of 720 | 17 | 19 | 0 | -3.00 to 0 | 17 | 19 | 0 | -2.80 to +0.05 |
| `inter_iamb` | all parts | 36 | 257 of 720 | 25 | 11 | 0 | -11.00 to -0.10 | 25 | 11 | 0 | -11.00 to 0 |
| `mmpc` | `required_paths` | 36 | 296 of 720 | 28 | 8 | 0 | -9.30 to 0 | 28 | 8 | 0 | -9.30 to 0 |
| `mmpc` | `forbidden_paths` | 36 | 167 of 720 | 16 | 20 | 0 | -2.40 to 0 | 15 | 21 | 0 | -2.10 to 0 |
| `mmpc` | all parts | 36 | 307 of 720 | 29 | 7 | 0 | -9.35 to 0 | 29 | 7 | 0 | -9.35 to 0 |

No row of any method is worse than before in either measure, so none was fit again on fresh seeds
for this comparison. Every one of the 6,696 fits whose graph has a consistent extension scored what
it scored before. In the 4,104 fits whose graph has none, `errors` was lower in 2,697, equal in
1,398, and higher in 9, and `dag_errors` lower in 2,652 and higher in 12; such a fit with required
paths was 2.4 (`cdnod`) to 4.4 (`inter_iamb`) edges closer to the class than before on average, and
with forbidden paths 0.5 to 1.1. The rule changes no adjacency.

### Against the same fit without the claims

The columns of [One part at a time](#one-part-at-a-time). `pc` under the rule:

| Method | Part | Rows | Worse | Worse again | Better | D | Worse | Worse again | Better | D | Edges directed | Against the true DAG | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `pc` | `required_paths` | 36 | 1 | 0 | 25 | -8.50 to +0.40 | 0 | 0 | 35 | -13.40 to -0.20 | 4,051 | 209 (5.2 %) | 0 | 20,668 of 37,726 (54.8 %) | yes |
| `pc` | `forbidden_paths` | 36 | 1 | 0 | 9 | -3.35 to +0.60 | 0 | 0 | 35 | -7.10 to -0.15 | 1,698 | 174 (10.2 %) | 0 | 12,027 of 52,320 (23.0 %) | yes |
| `pc` | all parts | 36 | 0 | 0 | 26 | -8.60 to +0.15 | 0 | 0 | 36 | -13.40 to -0.30 | 4,208 | 210 (5.0 %) | 0 | 32,874 of 90,046 (36.5 %) | yes |

The rows of the other four methods are in the tables above. Before and under the rule, as better /
tied / worse and, in parentheses, worse again on fresh seeds, for both measures:

| Method | Part | Class, before | Class, under the rule | True DAG, before | True DAG, under the rule |
|---|---|---|---|---|---|
| `pc` | `required_paths` | 12 / 23 / 1 (0) | 25 / 10 / 1 (0) | 33 / 3 / 0 (0) | 35 / 1 / 0 (0) |
| `pc` | `forbidden_paths` | 4 / 31 / 1 (0) | 9 / 26 / 1 (0) | 32 / 4 / 0 (0) | 35 / 1 / 0 (0) |
| `pc` | all parts | 13 / 23 / 0 (0) | 26 / 10 / 0 (0) | 35 / 1 / 0 (0) | 36 / 0 / 0 (0) |
| `cdnod` | `required_paths` | 16 / 20 / 0 (0) | 31 / 5 / 0 (0) | 28 / 8 / 0 (0) | 31 / 5 / 0 (0) |
| `cdnod` | `forbidden_paths` | 14 / 22 / 0 (0) | 17 / 19 / 0 (0) | 21 / 15 / 0 (0) | 22 / 14 / 0 (0) |
| `cdnod` | all parts | 21 / 15 / 0 (0) | 33 / 3 / 0 (0) | 26 / 10 / 0 (0) | 32 / 4 / 0 (0) |
| `iamb` | `required_paths` | 2 / 31 / 3 (2) | 25 / 11 / 0 (0) | 31 / 5 / 0 (0) | 35 / 1 / 0 (0) |
| `iamb` | `forbidden_paths` | 2 / 29 / 5 (3) | 9 / 26 / 1 (0) | 21 / 15 / 0 (0) | 32 / 4 / 0 (0) |
| `iamb` | all parts | 3 / 30 / 3 (3) | 25 / 11 / 0 (0) | 30 / 6 / 0 (0) | 35 / 1 / 0 (0) |
| `inter_iamb` | `required_paths` | 2 / 31 / 3 (2) | 24 / 12 / 0 (0) | 29 / 7 / 0 (0) | 35 / 1 / 0 (0) |
| `inter_iamb` | `forbidden_paths` | 2 / 28 / 6 (2) | 8 / 27 / 1 (0) | 24 / 12 / 0 (0) | 31 / 5 / 0 (0) |
| `inter_iamb` | all parts | 3 / 30 / 3 (2) | 25 / 11 / 0 (0) | 29 / 7 / 0 (0) | 35 / 1 / 0 (0) |
| `mmpc` | `required_paths` | 1 / 32 / 3 (3) | 26 / 10 / 0 (0) | 18 / 18 / 0 (0) | 35 / 1 / 0 (0) |
| `mmpc` | `forbidden_paths` | 0 / 32 / 4 (3) | 8 / 27 / 1 (0) | 19 / 17 / 0 (0) | 32 / 4 / 0 (0) |
| `mmpc` | all parts | 0 / 33 / 3 (3) | 28 / 8 / 0 (0) | 18 / 18 / 0 (0) | 35 / 1 / 0 (0) |

Under the rule five rows were worse than without claims on seeds 0 to 19, all in `errors` and all
worse before the rule too. They were fit again on seeds 100 to 119, with the fits without claims of
the same build:

| Method | Part | Row | D, seeds 0 to 19 | D, seeds 100 to 119 | Worse again |
|---|---|---|---|---|---|
| `pc` | `required_paths` | 10 variables, 2,000 samples, expected degree 2, 5 % | +0.40 [+0.05, +0.90] | -0.20 [-0.55, +0.05] | no |
| `pc` | `forbidden_paths` | 50 variables, 500 samples, expected degree 2, 5 % | +0.60 [+0.20, +1.05] | +0.30 [-0.50, +1.10] | no |
| `iamb` | `forbidden_paths` | 50 variables, 500 samples, expected degree 2, 5 % | +0.90 [+0.25, +1.65] | +0.20 [-0.70, +1.10] | no |
| `inter_iamb` | `forbidden_paths` | 50 variables, 500 samples, expected degree 2, 5 % | +0.90 [+0.25, +1.65] | +0.20 [-0.70, +1.10] | no |
| `mmpc` | `forbidden_paths` | 50 variables, 2,000 samples, expected degree 2, 30 % | +2.35 [+0.85, +3.95] | +1.55 [-0.10, +3.10] | no |

None was worse again, and each was closer to the true DAG on both sets of seeds. Before the rule,
`iamb`, `inter_iamb`, and `mmpc` were worse again in 2 or 3 rows of every part, at 50 variables.
Under it no row with required paths or with all parts is worse on either set of seeds, and of the
23 rows that failed before, 12 are closer to the class than without claims, 10 tie, and one is the
row of `mmpc` above.

Forbidden paths alone in `iamb`, `inter_iamb`, and `mmpc` pass the rule above and not a stricter
one. The fresh-seed fits cover every row of three cells (10 variables with 2,000 samples, and 50
variables with 500 and 2,000, all at expected degree 2). On seeds 100 to 119, 3 of each method's 6
rows with forbidden paths at 50 variables were worse than without claims, by 0.7 to 1.7 edges per
graph, none of them a row that was worse on seeds 0 to 19. With both sets of seeds pooled, 40
datasets a row, 3 of those 6 rows are worse in `iamb` and in `inter_iamb` (by 0.65 to 1.35) and 4
in `mmpc` (by 0.72 to 1.95); each of them is closer to the true DAG on seeds 100 to 119. No row of
`pc` is worse pooled, and none of any method with required paths or with all parts. The class the
truth and the claims define is more directed than the class without them, and these fits drop about
96 % of the forbidden paths on a graph with no consistent extension, so they leave undirected what
the class directs. `forbidden_paths` therefore stays experimental in the three methods
(`no_consistent_extension`, `pooled` in the published file).

### Against reference packages under the rule

The like-for-like comparison, each method and its reference given the same edges, holds no path
claim, so its rows are those above: `pc` was worse than Tetrad's PC in one row and `iamb` and
`inter_iamb` worse than bnlearn's in 1 to 3 rows on seeds 0 to 19, none of them again on fresh
seeds, and `mmpc` has no reference. The column that is not like for like, the method given the
path claims against the reference given the edges they imply, changes little: the rule directs an
undirected edge and cannot restore an edge the search left out. `pc` under the rule, with fewer /
tied / more on seeds 0 to 19 in the last column:

| Method | Reference | Part | Rows | Same knowledge | Path claims against the reference's edges |
|---|---|---|---|---|---|
| `pc` | pcalg `pc` | `required_paths` | 36 | 32 / 4 / 0 / 0 | 4 / 24 / 8 |
| `pc` | Tetrad PC | `required_paths` | 36 | 17 / 18 / 1 / 0 | 0 / 7 / 29 |
| `pc` | pcalg `pc` | `forbidden_paths` | 36 | 19 / 17 / 0 / 0 | 17 / 19 / 0 |
| `pc` | Tetrad PC | `forbidden_paths` | 36 | 9 / 27 / 0 / 0 | 12 / 23 / 1 |
| `pc` | pcalg `pc` | all parts | 36 | 33 / 3 / 0 / 0 | 4 / 26 / 6 |
| `pc` | Tetrad PC | all parts | 36 | 19 / 16 / 1 / 0 | 0 / 7 / 29 |

These rows of the last column were not fit again on fresh seeds: the column decides no status, and
[Against reference packages](#against-reference-packages-1) states what it shows.

### Claims applied, met, and dropped

A fit whose graph has a consistent extension keeps a claim when the class it restricts meets it. In
a fit whose graph has none, a claim is applied to an undirected edge, met by a directed edge, or
dropped. Per such fit, on seeds 0 to 19:

| Method | Part | Fits whose graph has no consistent extension | Claims per such fit | Applied to an edge | Met by a directed edge | Dropped | Dropped over all fits, before | Now |
|---|---|---|---|---|---|---|---|---|
| `pc` | `required_paths` | 147 of 720 | 69.5 | 3.8 | 9.4 | 56.3 (80.9 %) | 59.9 % | 54.8 % |
| `pc` | `forbidden_paths` | 147 of 720 | 78.5 | 0.8 | 2.0 | 75.8 (96.5 %) | 23.8 % | 23.0 % |
| `pc` | all parts | 147 of 720 | 148.1 | 4.6 | 11.4 | 132.0 (89.2 %) | 39.1 % | 36.5 % |
| `cdnod` | `required_paths` | 213 of 720 | 74.9 | 2.3 | 15.5 | 57.2 (76.3 %) | 64.0 % | 54.0 % |
| `cdnod` | `forbidden_paths` | 213 of 720 | 91.7 | 0.4 | 2.6 | 88.7 (96.7 %) | 39.6 % | 38.4 % |
| `cdnod` | all parts | 213 of 720 | 166.6 | 2.6 | 18.1 | 145.8 (87.5 %) | 50.0 % | 45.1 % |
| `iamb` | `required_paths` | 318 of 720 | 70.5 | 4.0 | 13.1 | 53.3 (75.7 %) | 71.0 % | 56.5 % |
| `iamb` | `forbidden_paths` | 318 of 720 | 85.8 | 0.9 | 2.3 | 82.6 (96.3 %) | 53.0 % | 51.0 % |
| `iamb` | all parts | 318 of 720 | 156.3 | 4.9 | 15.4 | 136.0 (87.0 %) | 60.6 % | 53.4 % |
| `inter_iamb` | `required_paths` | 315 of 720 | 70.9 | 4.1 | 13.2 | 53.5 (75.5 %) | 70.6 % | 56.2 % |
| `inter_iamb` | `forbidden_paths` | 315 of 720 | 86.4 | 0.9 | 2.3 | 83.1 (96.3 %) | 52.9 % | 50.9 % |
| `inter_iamb` | all parts | 315 of 720 | 157.2 | 5.0 | 15.6 | 136.7 (86.9 %) | 60.4 % | 53.2 % |
| `mmpc` | `required_paths` | 375 of 720 | 69.2 | 4.2 | 11.2 | 53.8 (77.7 %) | 77.1 % | 61.8 % |
| `mmpc` | `forbidden_paths` | 375 of 720 | 85.1 | 0.9 | 1.9 | 82.3 (96.7 %) | 61.8 % | 59.8 % |
| `mmpc` | all parts | 375 of 720 | 154.3 | 5.1 | 13.1 | 136.1 (88.2 %) | 68.3 % | 60.8 % |

Most claims are still dropped in these fits: 76 to 81 % of the required paths and 96 to 97 % of the
forbidden paths. A claim on a pair that is not adjacent has no edge to act on, and most claims name
such a pair: 57.8 % of the true required paths do even in the true graph, and a forbidden path
names an adjacent pair only where the edge runs the other way. Of 69 to 75 required paths per fit,
2 to 4 direct an edge and 9 to 16 are met by an edge the data directed; of 79 to 92 forbidden
paths, fewer than 1 and 2 to 3. The few
that are applied account for the change above. Over all fits the share of claims dropped falls by
5 to 15 points for required paths and by 1 to 2 for forbidden paths. The graph found had no
consistent extension in 49 of the 240 datasets of `pc`, 71 of `cdnod`, 106 of `iamb`, 105 of
`inter_iamb`, and 125 of `mmpc`, and at 50 variables, 2,000 samples, and expected degree 4 in 10,
15, 20, 20, and 19 of 20.

## Status

A part is supported in a method when it passes all three checks: the exactness lane, no worse than
the same fit without the claims, and no worse than its references given the same knowledge in any
row that repeats. Where no reference takes the knowledge, the first two decide.

| Part | Supported in | Experimental in |
|---|---|---|
| `required_paths` | `pc`, `cdnod`, `ges`, `xges`, `hc`, `boss`, `grasp`, `exact_search`, `fci`, `fci_jci`, `gfci`, `boss_fci`, `grasp_fci`, `lpcmci`, `direct_lingam`, `multi_group_direct_lingam`, `iamb`, `inter_iamb`, `mmpc` | `tabu`, `mmhc`, `rsmax2` |
| `forbidden_paths` | the same without `iamb`, `inter_iamb`, `mmpc`, and `tabu`, `mmhc`, `rsmax2` | `iamb`, `inter_iamb`, `mmpc` |
| `confounded` | `fci`, `fci_jci`, `gfci`, `boss_fci`, `grasp_fci`, `lpcmci` | |

Why a part is experimental: in `iamb`, `inter_iamb`, and `mmpc`, forbidden paths alone left the
CPDAG 0.65 to 1.95 edges farther from the class the claims define at 50 variables and expected
degree 2, over two sets of seeds pooled, because on a graph with no consistent extension about 96 %
of them are dropped (`metadata["unmet_knowledge"]` lists them); in `tabu` with the
plain BIC, and in `mmhc` and `rsmax2` at expected degree 4, the reference given the required edges
the paths imply made fewer errors in rows that repeated.

The limits the measurements show:

- A claim is read on the graph the search found. Where that graph has an edge missing or extra or
  a collider missed, a true claim can force a wrong direction: 5.8 and 10.9 % of the edges that
  required and forbidden paths directed in `pc` (5.2 and 10.2 % from 0.17.0), 1.7 and 4.1 % in
  `boss`, 1.1 and 1.7 % in `ges`, 0.1 to 3.2 % in `xges`, `grasp`, and `exact_search`, 2.7 and
  10.9 % in `hc`, 7.7 and 14.5 % in `cdnod`, 6.2 to 6.5 and 13.2 to 15.0 % in `iamb`, `inter_iamb`,
  and `mmpc`, and 18.2 and 24.7 % in `mmhc`.
- On a graph with no consistent extension a claim acts only on an edge, and most claims name a pair
  that is not adjacent: `pc`, `cdnod`, `iamb`, `inter_iamb`, and `mmpc` drop 76 to 81 % of the
  required and 96 to 97 % of the forbidden paths there, and the result is not guaranteed to stand
  for a DAG.
- A required path says less than a required edge and cannot restore an edge the search left out. In
  `pc`, `ges` with the plain BIC, `fci`, and `gfci`, a reference given the required edges made
  fewer errors than the method given the paths, and so it did on seeds 0 to 19 in `cdnod`, `hc`,
  `mmhc`, `rsmax2`, `iamb`, and `inter_iamb`. When you know the edge itself, state it as
  `required`.
- `ges`, `xges`, `hc`, and `tabu` are judged in their setting, expected degree 2, and `exact_search`
  at 10 and 20 variables.
- `iamb`, `inter_iamb`, and `mmpc` take the path parts only in their CPDAG output.

`pc`'s status is unchanged under the rule (`no_consistent_extension`, `pc_from_0_17_0` in the 0.17.0
file). `published/0.16.0/knowledge-paths.json` and `published/0.17.0/knowledge-paths.json` hold each
part's status under `references`, `status`, the first with the rows of the path-against-edge comparison
that repeated in `path_claims_against_reference_edges`.
