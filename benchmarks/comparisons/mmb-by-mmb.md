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
- **Targets.** The two nodes with the most neighbors in the network, ties in file order: the
  paper's "nodes with more adjacent nodes", fixed per network. A target is never hidden.
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

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with pcalg in the library, build the
authors' environment with `start_mmb.sh` and export what it prints, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/mmb_by_mmb_cpu.py --out outputs/mmb-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/mmb_by_mmb_cpu.py --out outputs/mmb --cores 1
```

The defaults are the paper's four networks and two sample sizes, 20 seeds from `--first-seed 0`, two
targets per network, one timed fit with no warm-up fit, and `--cap-wall-s` of 1800. `--part` and
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
