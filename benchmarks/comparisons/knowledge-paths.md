# Path and confounder claims

`knowledge_paths.py` measures the three `Knowledge` parts that state paths and hidden causes:
`required_paths` (`a` is an ancestor of `b`), `forbidden_paths` (`a` is not an ancestor of `b`),
and `confounded` (a hidden common cause of `a` and `b`). No other package implements these rules
for the CPDAG and PAG searches, so the driver checks them in two ways instead of against a
reference: against the exact answer on every small graph, and against the same fit without the
claims on simulated data.

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

Path claims take ancestor and non-ancestor facts in equal parts. `errors` counts the pairs
adjacent in the fit and not in the truth, or the reverse, plus every tail or arrowhead, on a pair
adjacent in both, that differs from the true DAG (`cpdag`, `lingam`), the true MAG of the observed
variables (`pag`), or the true window MAG (`lpcmci`). An undirected edge or a circle states nothing
and is never wrong, so a claim that orients an edge the right way lowers `errors`, and one that
orients it the wrong way raises it. `empty_errors`, the empty graph's count, is the floor. A fit
drops the claims the graph it found cannot meet and records how many as `unmet`; a fit that raises
`KnowledgeError`, for knowledge no graph meets, is recorded as `knowledge_error`, with its message,
and never scored as a graph.

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
distance from the floor:

```shell
$ANDREY_BENCH_PYTHON benchmarks/knowledge_paths.py summarize --runs outputs/kp-cpdag-* \
    --replicate outputs/kp-rep-* --oracle outputs/kp-oracle-* --out outputs/kp-summary.json
```

A cell is `worse` when the interval lies above 0. A worse cell counts only when the same cell on
fresh seeds, the runs `--replicate` names, is worse too.

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

| Method | Part | Rows | Datasets | Worse | Worse again | Better | D, least to most | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| `pc` | `required_paths` | 36 | 720 of 720 | 13 | 9 | 0 | 0 to +1.90 | 0 | 22,615 of 37,726 (59.9 %) | no |
| `pc` | `forbidden_paths` | 36 | 720 of 720 | 8 | 7 | 0 | 0 to +2.50 | 0 | 12,434 of 52,320 (23.8 %) | no |
| `ges` | `required_paths` | 36 | 720 of 720 | 2 | 0 | 0 | 0 to +0.50 | 18 | 6,747 of 37,726 (17.9 %) | no |
| `ges` | `forbidden_paths` | 36 | 720 of 720 | 2 | 0 | 0 | 0 to +0.80 | 18 | 4,935 of 52,320 (9.4 %) | no |
| `boss` | `required_paths` | 36 | 720 of 720 | 4 | 2 | 0 | 0 to +0.90 | 0 | 1,100 of 37,726 (2.9 %) | no |
| `boss` | `forbidden_paths` | 36 | 720 of 720 | 4 | 2 | 0 | 0 to +0.90 | 0 | 418 of 52,320 (0.8 %) | no |
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

In the `cpdag` lane an undirected edge is never an error, so a claim can only add errors there, by
directing an edge against the truth; what it gains shows as directed edges, on average 2 to 14 more
per graph. The class a search finds on finite data can differ from the true one, and restricting it
by true claims then directs some edges the wrong way. With `pc` this repeated in six cells, five of
them at expected degree 4, by 0.6 to 2.5 errors per graph where the fit without claims makes 16 to
70. With `boss` it repeated in two cells at degree 4, by 0.6 to 0.9 errors where the fit without
claims makes 5 and 9. `ges` has no row worse again, but in the six cells at degree 4 its fit does
not beat the empty graph with or without the claims, so those cells say nothing about them. `pc`
also drops the most claims: the tests remove an edge on the only path a required path needs.

In the `pag` and `lpcmci` lanes a claim can set a mark that was a circle, right or wrong, and
`direct_lingam` returns a DAG. No row is worse, with one part or with all, and required paths
lower the errors of `fci`, `gfci`, `boss_fci`, and `lpcmci` in 18 of their 32 rows. No fit dropped
a `confounded` pair.

### All parts together

Every part of each true claim set at once, the same seeds and columns.

| Method | Part | Rows | Datasets | Worse | Worse again | Better | D, least to most | Above floor | Claims dropped | Passes |
|---|---|---|---|---|---|---|---|---|---|---|
| `pc` | all parts | 36 | 720 of 720 | 14 | 11 | 0 | 0 to +1.90 | 0 | 35,228 of 90,046 (39.1 %) | no |
| `ges` | all parts | 36 | 720 of 720 | 2 | 0 | 0 | 0 to +0.50 | 18 | 11,757 of 90,046 (13.1 %) | no |
| `boss` | all parts | 36 | 720 of 720 | 5 | 2 | 0 | 0 to +1.00 | 0 | 1,583 of 90,046 (1.8 %) | no |
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

### Status

`required_paths`, `forbidden_paths`, and `confounded` are supported in `fci`, `gfci`, `boss_fci`,
and `lpcmci`, and the two path parts in `direct_lingam`. They are experimental in `pc`, `ges`, and
`boss`, and in the methods the driver does not run.
