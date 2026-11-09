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
