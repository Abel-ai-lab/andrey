# Markov blankets, and parents and children

`blanket_cpu.py` compares Andrey's learners of Markov blankets (`iamb`, `inter_iamb`) and of parents
and children (`mmpc`) with bnlearn's, and HITON-PC, HITON-MB, and MMMB with the authors' Causal
Explorer, on the set each learns at a target and on the graph the sets give. A learner joins the
lane by being named in `LEARNERS` of `andrey_bench/adapters/andrey_blanket.py`, and its reference by
being named in `COUNTERPART` of `andrey_bench/adapters/bnlearn_blanket.py` or
`andrey_bench/adapters/causal_explorer_blanket.py`. A learner that no reference implementation
runs says so in its row (`reference=False`) and is judged on its paper's published results and the
true sets. The driver runs every learner the installed Andrey has.

## Data

A cell is one data setting, written as the driver's `--cells` take it:

- `net:<name>:<n>`: `n` rows sampled from a published network's own tables, discrete
  (`andrey.data`'s networks, such as `child`, `insurance`, and `alarm`, and the harness's
  `mildew`, `win95pts`, `hailfinder`, and `pigs`);
- `gauss:<d>:<n>[:<degree>]`: `n` rows of a linear-Gaussian graph on `d` variables with `degree`
  expected edges per variable (2 by default), standardized.

The columns come in an order drawn from the seed, so no learner reads the generating order. The
truth is the generating DAG: its parents and children are the parents-and-children sets, those and
its children's other parents the Markov blankets, and the moral graph and skeleton follow from them.
The default cells are three networks and two Gaussian sizes, each at two sample sizes, on 20 seeds.

## Solutions

| Andrey | bnlearn 5.2.1, at a target | bnlearn, for the whole graph |
|---|---|---|
| `iamb` | `learn.mb(method = "iamb")` | `iamb` |
| `inter_iamb` | `learn.mb(method = "inter.iamb")` | `inter.iamb` |
| `mmpc` | `learn.nbr(method = "mmpc")`, kept to members whose own set holds the target | `mmpc` |
| `gs` | `learn.mb(method = "gs")` | `gs` |
| `fast_iamb` | `learn.mb(method = "fast.iamb")` | `fast.iamb` |
| `hiton_pc` | `learn.nbr(method = "si.hiton.pc")`, kept as for `mmpc` | `si.hiton.pc` |
| `hiton_mb` | none | none |
| `mmmb` | none | none |
| `pcmb` | none | none |

A learner runs beside the function in its row once it is named in `LEARNERS` and the installed
Andrey has it. `hiton_mb`, `mmmb`, and `pcmb` have no bnlearn function; the lane's `mmpc` row
compares MMMB's first step, the parents and children, with bnlearn's.

The authors of HITON-PC, HITON-MB, and MMMB distribute their code as Causal Explorer 1.5, a MATLAB
toolbox under a non-commercial license. The lane runs it as a reference only:
`start_causal_explorer.sh` fetches the public repository at a pinned commit into
`benchmarks/.causal-explorer`, which git ignores, and the adapter calls it through MATLAB in batch,
one MATLAB process per fit, for every target. It answers for targets only and reports no test
count, and its wall time includes MATLAB's start. Without MATLAB or the toolbox the lane leaves the
arm out, so nothing about it runs in CI.

| Andrey | Causal Explorer 1.5, at a target |
|---|---|
| `hiton_pc` | `Causal_Explorer('HITON_PC', ...)`, kept as for `mmpc` |
| `hiton_mb` | `Causal_Explorer('HITON_MB', ...)` |
| `mmmb` | `Causal_Explorer('MMMB', ..., 'MMPC', options)` |

bnlearn's `pc.stable` is read at a target as well, once per fit: `mb()` of its CPDAG for a
blanket, `nbr()` for parents and children, the answer a graph learner gives to the same question.
`baseline.empty` answers the empty set and the empty graph, the floor each score is read against.

bnlearn is pinned to 5.2.1, the release the lane was written against (`fast.iamb` is deprecated in
it and removed in 2027). The driver stops when the R library holds another version; every row names
the version it ran.

## The same input

Both sides get the same data, the same test, the same `alpha`, and the same largest conditioning
set (`--max-depth`, Andrey's `max_depth` and bnlearn's `max.sx`), and every row records them. Every
setting is passed, never defaulted: a default of one side that the other does not share, such as
bnlearn's t-test for continuous data or the limit a learner of Andrey may carry, would otherwise
change what is compared. `adapters/bnlearn_blanket.py` lists each place bnlearn's defaults differ
from Andrey's and how the lane lines them up. The ones that decide a comparison:

- **The test.** Continuous data runs Fisher's z (`fisherz` and bnlearn's `zf`). Discrete data runs
  the G-square or chi-square test (`--indep-test`, `gsq` or `chisq`) against bnlearn's `mi` or
  `x2` form. `--discrete-test adf` picks `mi-adf` and `x2-adf`, which assume independence when a
  table averages fewer than 5 rows per cell and take empty rows and columns out of the degrees of
  freedom; `--discrete-test plain` picks `mi` and `x2`, which do neither. Andrey's tests have no
  such rule, so with `adf` a sparse table can end a search earlier in bnlearn. The test used, in
  both names, and whether the rule is on, are in every row (`indep_test`, `bnlearn_test`,
  `sparse_rule`).
- **The limit.** With `--max-depth`, bnlearn's `mmpc` and `si.hiton.pc` also stop growing the
  candidate set once it holds more members than the limit, and Andrey's `mmpc` caps only the size of
  the conditioning sets, so the parents-and-children learners agree only with no limit. A blanket
  learner stops growing at the same size on both sides.
- **Parents and children are symmetric.** bnlearn's `learn.nbr` is a one-sided search, and
  Andrey's `mmpc(target=)` keeps a member only when the member's own search holds the target. The
  lane applies the same rule to bnlearn, and counts the tests of those searches. A blanket has no
  such rule on either side.
- **Causal Explorer.** Its HITON functions take the test (`'g2'` for `gsq`), `alpha` as the
  threshold, and the largest conditioning set as their last argument, the paper's max-k, which
  Andrey's `max_depth` is; no limit passes the number of variables less one. Its G-square test takes
  empty cells out of the degrees of freedom and runs only with at least five rows per cell of its
  table, the rule Andrey's HITON learners follow. MMMB takes `alpha` as `options.threshold`, the
  limit as `options.maxK`, and five rows per cell as `options.epc`, a cap on the conditioning sets
  with no off switch. Its MMPC, inside MMMB, runs without the symmetry check that `mmmb` runs, so
  its blankets keep members that check removes; like `mmmb`, it tests a candidate spouse only with a
  parent or child of the target whose own set holds the candidate. Categorical columns are coded
  from 0 over their observed levels. There is no counterpart of `chisq`, and none on continuous
  data: the toolbox's manual takes Fisher's z (`'z'`) in HITON only with a categorical target, and
  MMMB reads categorical data only. On linear-Gaussian data with a continuous target, its HITON-PC
  and HITON-MB with `'z'` returned every other variable for every target, so the Gaussian cells run
  without the arm.
- **Ties.** bnlearn orders candidates of equal strength by column, Andrey by a rank computed from
  the data, so where several p-values agree, for example when they underflow to zero, the two can
  add variables in a different order and end with different sets. This is a known cause of
  differences, and a difference is a finding to trace, not a setting to tune.

## Measures

`--mode target` (the default) fits each solution once per dataset and answers for `--targets` of
them: every column (`all`), or a number drawn from the seed. The time of a fit is the time for all
its targets. A set is scored against the true set of its kind, per target and pooled:

- `precision`, `recall`, and `f1` of the set; `distance`, the Euclidean distance of (recall,
  precision) to (1, 1); and `exact`, the share of targets whose set is the true one. A ratio with
  a zero denominator is undefined, so an empty set has no precision and an isolated target has no
  recall; the pooled ratios add the counts of every target (`tp`, `est`, `true`) before dividing,
  and `mean_` takes each ratio over the targets where it is defined.
- the number of tests of each search where the package reports it. bnlearn reports them per
  search; a package that reports none has `NO_COUNT` (-1) in the row.
- `same`, in the summary: the targets where a learner's set is the set of its bnlearn counterpart,
  and `authors` the targets where it is the set of its Causal Explorer counterpart.

`--mode graph` fits once per dataset and `--structures` (`moral`, `skeleton`, and `cpdag`, each
where a learner offers it) and scores the graph's adjacencies, `precision`, `recall`, `f1`, and
`shd` against the empty graph's. The CPDAG's `shd` is `andrey.metrics`', against the true CPDAG.

## PCMB against its paper

PCMB has no reference implementation (Pena et al., 2007), so `pcmb_summary.py` sets its sets beside
Table 6 of its paper: the mean and standard deviation, over 10 databases, of the precision, the
recall, and the distance from perfect precision and recall of the set learned at every node of the
Alarm and Pigs networks, at significance level 0.01. The cells are `net:alarm:<n>` at 100, 200, 500,
1,000, 2,000, and 5,000 rows and `net:pigs:<n>` at 100, 200, and 500, with every node a target, on
datasets the lane draws: 20 per Alarm cell (seeds 0-19) and 10 per Pigs cell (seeds 0-9). The fits
run G-square at `--alpha 0.01` with no limit on the conditioning set (`--max-depth none`). On
categorical data `pcmb` counts degrees of freedom as Spirtes, Glymour, and Scheines (1993) do, its
default `degrees_of_freedom="nonempty"`, where the other learners default to `"mean"`.
`--drop-constant` fits the columns that hold more than one value, because a test needs two: an
Alarm sample at 100 rows has a few columns of one value, whose targets answer the empty set and
count against the full network.

A dataset's value is the mean over its targets of the lane's per-target scores, in two readings of
an empty answer:

- `skip`, the reading of the comparison with the paper: an empty answer has no precision or
  distance and counts in the recall as 0, so the means of the precision and of the distance leave
  it out;
- `one`: an empty answer has the precision 1, so its distance is 1.

The floor is the errors of the sets, the members missed plus the members added, against the empty
set's, the size of the true set, with the paired bootstrap interval of the difference over the
datasets. The table beside the paper's is on the
[0.18.0 release card](../../docs/docs/benchmarks/0.18.0.md#pcmb). Here are the other reading, the
floor, and the time of a fit for every node on one core:

| Network | Rows | Datasets | Empty answers per dataset | Distance, empty counted 1 | Errors | Empty set's errors | Errors minus the empty set's [95% interval] | Median fit |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Alarm | 100 | 20 | 8.6 | 0.662 | 2.89 | 3.51 | -0.63 [-0.69, -0.56] | 2.6 s |
| Alarm | 200 | 20 | 6.3 | 0.516 | 2.16 | 3.51 | -1.35 [-1.40, -1.30] | 2.8 s |
| Alarm | 500 | 20 | 3.5 | 0.358 | 1.63 | 3.51 | -1.89 [-1.96, -1.82] | 3.9 s |
| Alarm | 1,000 | 20 | 2.5 | 0.229 | 1.02 | 3.51 | -2.49 [-2.56, -2.43] | 4.3 s |
| Alarm | 2,000 | 20 | 1.3 | 0.146 | 0.72 | 3.51 | -2.79 [-2.83, -2.74] | 5.8 s |
| Alarm | 5,000 | 20 | 1.0 | 0.111 | 0.58 | 3.51 | -2.94 [-2.96, -2.91] | 7.6 s |
| Pigs | 100 | 10 | 3.5 | 0.305 | 1.39 | 3.66 | -2.27 [-2.30, -2.24] | 3 min |
| Pigs | 200 | 10 | 0.4 | 0.074 | 0.35 | 3.66 | -3.31 [-3.32, -3.29] | 9 min |
| Pigs | 500 | 4 | 0.0 | 0.022 | 0.09 | 3.66 | -3.57 [-3.59, -3.56] | 16 min |

On Alarm, the datasets of seeds 0-9 and of seeds 10-19 give mean distances that differ by at most
0.03 in each cell.

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with bnlearn 5.2.1 in the library, and
for the Causal Explorer arm put `matlab` on the `PATH`, run `benchmarks/start_causal_explorer.sh`,
and export what it prints; then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-cpu --max-depth none
$ANDREY_BENCH_PYTHON benchmarks/blanket_cpu.py --out outputs/blanket-graph --mode graph
```

The defaults are ten cells, 20 seeds from `--first-seed 0`, every target, `--indep-test gsq`,
`--discrete-test adf`, `--alpha 0.05`, no limit on the conditioning set, one untimed warm-up fit,
one timed fit, and `--cap-wall-s` of 1800. `--cells net:alarm:5000 gauss:50:1000:4` selects cells,
`--solutions andrey.iamb,bnlearn.iamb` solutions, and `--part` with `--parts` splits the cells over
jobs. The records go to `records.jsonl`, one line per fit, with each answer, so `--rescore`
recomputes the scores without fitting, and the build and machine to `provenance.json`. A fit starts
a process, so on a small cell most of its wall time is that start, several seconds for R.
