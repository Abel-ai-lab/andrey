# MMB-by-MMB

`mmb_by_mmb_cpu.py` measures what a method learns about one target when some variables are hidden.
It runs Andrey's `mmb_by_mmb`, the authors' MMB-by-MMB code, pcalg's RFCI, and Andrey's FCI on the
simulation of the method's paper (Xie, Li, Wu, Zeng, Liu, and Geng, 2024, Section 5.1), and it
scores the edges at the target, with both endpoint marks, against the PAG of the observed variables.
The lane is built for the methods that answer for one target as well: its data, its scores, and the
contract of an adapter (below) serve `mb_by_mb`, `cmb`, and `mmb_by_mmb`.

## Data

The model is a published network, read as a linear-Gaussian structural model
(`andrey_bench/latent_networks.py`):

- **Graph.** The network's DAG. The paper's four are `mildew` (35 nodes), `alarm` (37),
  `win95pts` (76), and `andes` (223); the lane also reads `hailfinder` (56), `pigs` (441), and
  every other network of `andrey.data`. `mildew`, `win95pts`, `hailfinder`, and `pigs` come from the
  bnlearn repository: the harness fetches each file on first use and checks its SHA-256, and Andrey
  itself does not ship them.
- **Weights and noise.** Each arc has a weight with a random sign and a magnitude uniform on
  `[0.5, 1]`, the paper's uniform draw on `[-1, -0.5]` together with `[0.5, 1]`, and every noise
  term is standard Gaussian. The data are not rescaled; no column's variance reaches 100 in 20
  seeds of any of the networks.
- **Hidden nodes.** Random nodes, 4, 4, 6, and 10 in the paper's four networks, drawn per seed
  among the nodes that are not targets. The paper hides random nodes, so a hidden leaf simply
  leaves the data; `andrey.data`'s own latent datasets hide nodes with two or more children
  instead. `--hidden 0` hides none, and the truth is then the DAG's CPDAG.
- **Targets.** The paper's own in its four networks, those of its Tables 7-10 (4, 4, 5, and 7
  nodes; `PAPER_TARGETS`), and the two nodes with the most neighbors in any other network, ties in
  file order; `--targets k` takes the `k` nodes with the most neighbors in any network. The paper
  says it picks "nodes with more adjacent nodes", yet two of its WIN95PTS targets have one
  neighbor. Targets are fixed per network, and a target is never hidden.
- **Rows and seeds.** 1,000 and 5,000 rows, 20 datasets per network (the paper repeats 100 times).
  The model (weights, hidden nodes, column order) is a function of the network and the seed, and the
  rows are a function of the model and the sample size, so the two sample sizes of a seed are
  samples of one model. The observed columns come in an order drawn from the seed.

## Solutions

Every solution runs at `alpha = 0.01`, the value of the authors' example; the paper states none.
`andrey.mmb_by_mmb` with total conditioning and the authors' code both test each blanket at
`alpha / (10 * floor(d / 10))` over `d` columns (`alpha` itself below 10 columns) and the local
structures at `alpha`.

- `andrey.mmb_by_mmb` and `andrey.mmb_by_mmb.iamb` run `andrey.mmb_by_mmb` with Fisher's z test
  (`indep_test="fisherz"`) and the blanket learner `"total_conditioning"`, the paper's, or
  `"iamb"`. They fit once per target, and the answer holds the marks at the target and the fit's
  `metadata["n_tests"]`.
- `mmb-by-mmb.authors` is the authors' code, `fengxie009/MMB-by-MMB` at a pinned commit, run
  unmodified in an environment of its own with `maxK = 10`. It has no license, so
  `benchmarks/start_mmb.sh` downloads its five files, checks each SHA-256, and keeps them out of the
  repository. It fits once per target. Its answer is the target's row and column of the PAG its
  object holds when the call returns, and its test count is the `ci_test` it returns.
- `pcalg.rfci` is pcalg's `rfci` with Fisher's z test, its tests counted by a wrapper around
  `gaussCItest`: the count is every test the fit evaluated, the skeleton's and the orientation
  phase's. It fits once per dataset and is read at every target.
- `andrey.fci.numpy` is Andrey's `fci` with Fisher's z test, read the same way. It reports no
  test count yet.
- `baseline.empty` answers no edge anywhere; its SHD at a target is the target's degree in the
  truth.

A fit that passes `--cap-wall-s` is a `timeout` with no scores, never an empty graph.

## Measures

A graph is scored on the edges at the target (`andrey_bench/local_lane.py`). An edge of the answer
is a true edge when the truth has it with both marks the same; a circle is a mark, so `o->` is not
`->`, which follows the paper's rule that a true edge needs both ends right.

- `precision`, `recall`, and `f1` over true edges, and `distance`, the paper's Euclidean distance
  of `(recall, precision)` from `(1, 1)`. An answer with no edge has no precision, so the summary
  pools the counts (`correct_edges`, `est_edges`, `true_edges`) over datasets and targets.
- `adj_precision` and `adj_recall` over adjacencies, whatever the marks.
- `shd`, the local endpoint SHD: the pairs `(target, j)` with an extra edge, a missing edge, or a
  mismarked one (`extra`, `missing`, `mismarked`). `empty_shd` is the empty answer's SHD.
- `n_tests`, the conditional-independence tests a solution evaluated, when it reports them.

The paper averages these measures over its repetitions and does not say how it treats an answer
with no edge or an edge with a circle; the records keep the counts, so either reading can be
computed from them.

## Results

Release 0.18.0 judged `andrey.mmb_by_mmb` against the authors' code on the paper's four networks at
1,000 and 5,000 rows, 20 datasets (seeds 0-19) per setting, at the paper's named targets; FCI, RFCI,
and the IAMB blanket arm are context. `published/0.18.0/mmb-by-mmb.json` holds every setting, and
the release summary's MMB-by-MMB rows are built from it by `mmb_by_mmb_summary.py`. The
[release card](https://andrey.abel.ai/docs/benchmarks/0.18.0.html#mmb-by-mmb) reads it.

Per dataset and solution, each measure is the mean over the targets with a true edge. The judged
difference is Andrey's minus the authors' code per dataset, with its percentile bootstrap 95%
interval over the datasets (10,000 resamples, `default_rng(0)`). A setting is worse when the
interval lies wholly on the worse side of 0 (F1 below, distance above) on seeds 0-19 and again on
the 20 fresh datasets (seeds 20-39), better when it lies wholly on the better side on seeds 0-19,
and a tie otherwise. A setting worse on seeds 0-19 on either measure was rerun on seeds 20-39.

| Network | Rows | F1, Andrey | F1, authors | Distance, Andrey | Distance, authors | Result | Errors at the target, Andrey | Errors, empty answer |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| MILDEW | 1,000 | 0.968 | 0.939 | 0.048 | 0.088 | better | 0.19 | 3.14 |
| MILDEW | 5,000 | 0.982 | 0.949 | 0.027 | 0.073 | better | 0.11 | 3.14 |
| ALARM | 1,000 | 0.867 | 0.774 | 0.195 | 0.323 | better | 0.76 | 4.29 |
| ALARM | 5,000 | 0.894 | 0.844 | 0.153 | 0.222 | better | 0.59 | 4.29 |
| WIN95PTS | 1,000 | 0.889 | 0.843 | 0.161 | 0.219 | better | 0.72 | 3.90 |
| WIN95PTS | 5,000 | 0.953 | 0.941 | 0.068 | 0.084 | tie | 0.34 | 3.90 |
| ANDES | 1,000 | 0.819 | 0.844 | 0.256 | 0.221 | tie | 0.56 | 3.04 |
| ANDES | 5,000 | 0.888 | 0.890 | 0.158 | 0.156 | tie | 0.35 | 3.04 |

The F1 differences are Andrey's minus the authors' (positive is better), the distance differences
likewise (negative is better), each with its interval; the last column is Andrey's errors at the
target minus the empty answer's, which a solution passes when its interval lies wholly below 0.

| Network | Rows | F1, first 20 | Distance, first 20 | F1, fresh 20 | Distance, fresh 20 | Errors minus the empty answer's |
|---|---:|---|---|---|---|---|
| MILDEW | 1,000 | +0.028 [+0.005, +0.053] | -0.040 [-0.077, -0.007] | - | - | -2.95 [-3.08, -2.84] |
| MILDEW | 5,000 | +0.033 [+0.012, +0.058] | -0.046 [-0.081, -0.018] | - | - | -3.02 [-3.15, -2.91] |
| ALARM | 1,000 | +0.092 [+0.058, +0.133] | -0.129 [-0.186, -0.079] | - | - | -3.52 [-3.80, -3.23] |
| ALARM | 5,000 | +0.050 [+0.019, +0.085] | -0.070 [-0.119, -0.025] | - | - | -3.70 [-4.00, -3.38] |
| WIN95PTS | 1,000 | +0.046 [+0.027, +0.065] | -0.059 [-0.086, -0.033] | +0.036 [+0.017, +0.057] | -0.049 [-0.077, -0.022] | -3.18 [-3.33, -3.04] |
| WIN95PTS | 5,000 | +0.012 [-0.009, +0.037] | -0.016 [-0.051, +0.014] | - | - | -3.56 [-3.68, -3.47] |
| ANDES | 1,000 | -0.025 [-0.048, -0.005] | +0.035 [+0.007, +0.068] | +0.003 [-0.013, +0.020] | -0.004 [-0.028, +0.019] | -2.48 [-2.59, -2.34] |
| ANDES | 5,000 | -0.001 [-0.015, +0.014] | +0.002 [-0.019, +0.022] | - | - | -2.69 [-2.76, -2.59] |

ANDES at 1,000 rows was worse on seeds 0-19 and a tie on the fresh datasets, and so it is a tie.
WIN95PTS at 1,000 rows was better on both seed sets. Every fit of every judged solution finished
under its cap. The floor of `andrey.metrics` for the adjacencies and for the marks, pooled over a
setting's targets, passes for Andrey's answers in every setting.

### Accuracy against cost

Each cell holds the F1 and, in parentheses, the conditional-independence tests and the seconds of
one fit, as the means over the 20 datasets. MMB-by-MMB and the authors' code fit once per target,
FCI and RFCI once per dataset and are read at every target. Andrey's fits come from a run of their
own on other shared nodes, so the seconds are indicative; the tests are counts. Andrey's FCI
reports no test count yet. Neither FCI nor RFCI is more accurate than MMB-by-MMB in any setting,
so neither is better on accuracy and cost together.

| Network | Rows | MMB-by-MMB | Authors' code | Andrey's FCI | pcalg's RFCI |
|---|---:|---|---|---|---|
| MILDEW | 1,000 | 0.968 (2,014; 0.43) | 0.939 (561; 0.07) | 0.778 (-; 0.29) | 0.603 (4,923; 0.13) |
| MILDEW | 5,000 | 0.982 (5,572; 0.48) | 0.949 (911; 0.27) | 0.875 (-; 0.30) | 0.793 (8,752; 0.24) |
| ALARM | 1,000 | 0.867 (1,664; 0.33) | 0.774 (522; 0.10) | 0.734 (-; 0.23) | 0.634 (3,770; 0.14) |
| ALARM | 5,000 | 0.894 (2,629; 0.35) | 0.844 (661; 0.52) | 0.833 (-; 0.25) | 0.816 (5,135; 0.20) |
| WIN95PTS | 1,000 | 0.889 (19,146; 0.51) | 0.843 (3,051; 1.07) | 0.723 (-; 0.29) | 0.682 (12,691; 0.53) |
| WIN95PTS | 5,000 | 0.953 (125,086; 0.64) | 0.941 (6,079; 4.98) | 0.812 (-; 0.55) | 0.778 (21,723; 1.07) |
| ANDES | 1,000 | 0.819 (7,341; 0.52) | 0.844 (3,679; 17.91) | 0.663 (-; 0.77) | 0.635 (162,615; 6.10) |
| ANDES | 5,000 | 0.888 (18,112; 0.44) | 0.890 (4,740; 73.99) | 0.724 (-; 1.28) | 0.707 (328,354; 13.68) |

The ladder holds the cost against the number of variables: a planted core with hidden common
causes, in which the target has a blanket of 6 variables, among independent variables, with 5
datasets per row (the medians are shown). An answer is right when the target's edges and marks
equal the true PAG's. MMB-by-MMB ran on the 0.18.0 release branch of 14 November 2026 and FCI on
that of 12 November 2026; an FCI fit has a cap of 1,800 s, and a fit that reaches it ends the
ladder for that number of rows.

| Rows | Variables | MMB-by-MMB tests | seconds | answers right | FCI tests | seconds | answers right |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 30 | 632 | 0.03 | 5 of 5 | 7,080 | 0.03 | 5 of 5 |
| 2,000 | 100 | 1,402 | 0.06 | 5 of 5 | 443,673 | 0.82 | 5 of 5 |
| 2,000 | 300 | 3,589 | 0.21 | 5 of 5 | 7.6 million | 7.20 | 4 of 5 |
| 2,000 | 1,000 | 11,220 | 1.91 | 2 of 5 | 115.6 million | 64 | 5 of 5 |
| 2,000 | 3,000 | needs rows > variables |  |  | 1,302 million | 1,345 | 0 of 2 |
| 10,000 | 30 | 632 | 0.05 | 5 of 5 | 13,096 | 0.08 | 5 of 5 |
| 10,000 | 100 | 1,402 | 0.11 | 5 of 5 | 1.1 million | 1.63 | 5 of 5 |
| 10,000 | 300 | 3,602 | 0.42 | 5 of 5 | 28.2 million | 35 | 4 of 5 |
| 10,000 | 1,000 | 11,319 | 3.13 | 4 of 5 | timed out (1,800 s) |  |  |
| 10,000 | 3,000 | 33,302 | 29 | 5 of 5 | not run |  |  |

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with pcalg in the library, build the
authors' environment with `start_mmb.sh` and export what it prints, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/mmb_by_mmb_cpu.py --out outputs/mmb-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/mmb_by_mmb_cpu.py --out outputs/mmb --cores 1
```

The defaults are the paper's four networks and two sample sizes, 20 seeds from `--first-seed 0`, the
paper's targets, one timed fit with no warm-up fit, and `--cap-wall-s` of 1800. `--part` and
`--parts` split the datasets between jobs, `--solutions` selects solutions, `--resume` continues a
run, and `--rescore` recomputes the scores from the stored marks without fitting. One JSON line per
fit goes to `records.jsonl`: the solution, the dataset, the target, the time, the test count, the
marks, and the scores at each target the fit answers for.

## Adding a solution

A solution that answers at a target fills the contract of `local_lane.LocalAdapter` and goes into
`ANDREY_ARMS` in the driver; nothing else changes.

- `fit(data, params)` receives the data and `params["target"]`, the target's column, and returns
  `pack_answer(marks, n_tests)`: a `(d, d)` array of endpoint marks (0 none, 1 tail, 2 arrowhead, 3
  circle; `marks[i, j]` is the mark at `i` of the edge between `i` and `j`), of which only the
  target's row and column are read, and the number of tests, or `None`.
- `local = True` makes the driver fit it once per target. A solution that learns a whole graph sets
  `local = False` and is fitted once per dataset.
- `output_type` is `"cpdag"` for the marks of a CPDAG and `"pag"` for a PAG's; with `--hidden 0`
  the truth is the CPDAG, and with hidden nodes the PAG.
