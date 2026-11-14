# Edges and directions at a target

`direct_cpu.py` runs Andrey's methods that learn the edges at one variable and their directions
(`mb_by_mb`, `cmb`) as their papers did, beside global PC read at the target: Andrey's `pc` with the
same test and bnlearn's `pc.stable`. `direct_summary.py` judges a method per cell against both PCs
and the empty answer, and sets its means beside its paper's. Neither method has a reference
implementation, so global PC, which the papers compare with, is the comparison, and the papers'
numbers are a soft check.

## Data

- `alarm` (37 variables), forward samples of the network's own tables from `andrey.data`, the
  columns in an order drawn from the seed; every variable is a target.
- `hailfinder` (56), forward samples of the published network, its columns permuted by the seed;
  every variable is a target.
- `chess45`, the 45 x 45 chessboard of the MB-by-MB paper: 2,025 binary variables, each square
  pointing to the squares to its right and below, with 10 targets drawn once from a fixed seed, as
  the paper draws 10 (`andrey_bench/chessboard.py` gives the tables).

## Solutions

| Solution | What it runs |
|---|---|
| `andrey.mb_by_mb`, `andrey.cmb` | the method at one target per fit, with its own defaults but `alpha` |
| `andrey.pc.gsq` | Andrey's `pc` with `gsq` on the whole dataset, read at every target |
| `bnlearn.pc-stable.graph` | bnlearn 5.2.1 `pc.stable` with `mi-adf` on the columns that vary, read at every target |
| `baseline.empty` | the empty answer, the floor |

All run at the same `--alpha`; `--max-depth` caps the conditioning sets of the two PCs only.

## Measures

Each answer is scored on the edges at its target in both papers' readings
(`andrey_bench/direct_lane.py`). Against the DAG (`dag_counts`, the MB-by-MB paper's): undirected,
missing, extra, and reversed edges, whose sum is the SHD. Against the truth's CPDAG (`cpdag_counts`,
the CMB paper's): extra, missing, and reversed edges, an edge the CPDAG directs and the answer
leaves undirected (`under`) among the reversed, whose sum is `total`. `direct_summary.py` takes a
dataset's mean over its targets, pairs the datasets of two solutions, and gives the mean
difference with its 95% percentile-bootstrap interval: better or worse when the interval lies
wholly on that side of 0 over at least 10 datasets. A cell worse on seeds 0-19 counts only when it
is worse again on seeds 20-39.

## CMB against global PC

On the CMB paper's setting, Alarm at 1,000 rows with G-square tests at alpha 0.02 and no limit on
the conditioning set, CMB's mean total per target was 0.751 on seeds 0-19, against 0.997 for
Andrey's PC (-0.246 [-0.368, -0.118]), 1.800 for bnlearn's `pc.stable` (-1.049 [-1.137, -0.962]),
and 2.486 for the empty answer; on seeds 20-39 it was 0.689 against 0.954 and 1.822. Without the
undirected edges, 0.430 against 0.638 and 1.335. The paper reports 0.69 extra, 0.61 missing, and
0.51 reversed edges per target, 1.81 in all; CMB's extra edges here are 0.02. Without the
symmetry check of HITON-MB, CMB's default blanket learner, they were 0.21; counting the degrees of
freedom from the cells that occur, or capping the conditioning sets at 3, left them at 0.01 to 0.02.
With exact tests, by d-separation in the true graph, CMB's only errors were 0.09 edges per target
left undirected that the CPDAG directs. The [0.18.0 release card][card] has the table.

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with bnlearn 5.2.1 in the library, for
the bnlearn arm; the hailfinder network comes from the published networks the harness downloads.

```shell
$ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/direct-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/direct_cpu.py --out outputs/cmb --paper cmb --seeds 40
$ANDREY_BENCH_PYTHON benchmarks/direct_summary.py outputs/cmb --method cmb --json cmb-rows.json
```

`--paper` sets the networks, rows, level, conditioning-set cap, and solutions of a paper (`cmb`:
alarm at 1,000 rows, alpha 0.02, no cap; `mb_by_mb`: alarm and hailfinder at 500, 1,000, 2,000,
and 5,000 rows, alpha 0.05, a cap of 5); `--networks`, `--rows`, `--alpha`, `--max-depth`, and
`--solutions` override them, and `--part` with `--parts` splits the datasets over jobs. A fit
starts a process and has a wall cap (`--cap-wall-s`, 1,800 s); a fit past it is a timeout, never
an empty answer. The records go to `records.jsonl`, one line per fit and target with the marks at
the target, so `--rescore` recomputes the scores without fitting, and the build and machine to
`provenance.json`.

[card]: https://andrey.abel.ai/docs/benchmarks/0.18.0.html#cmb
