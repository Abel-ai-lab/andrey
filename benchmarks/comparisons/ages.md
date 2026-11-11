# AGES

`ages_cpu.py` compares Andrey's `ages` with pcalg's `ages`, the implementation of the method's
authors, on the simulation of the method's paper (Eigenmann, Nandy, and Maathuis, 2017, Section
4.1), and `ages_summary.py` compares the solutions per setting. The published rows are in
[`published/0.17.0/ages.json`](../published/0.17.0/ages.json).

## Data

Every dataset has 10 variables. Each pair of variables, in a causal order, holds a strong edge with
probability `qs`, a weak edge with probability `qw`, and no edge otherwise. A strong edge has a
weight of magnitude `U(0.8, 1.2)` and a weak one `U(0.1, 0.3)`, each with a random sign, and the
errors are Gaussian with variances from `U(0.5, 1.5)`. The columns come in an order drawn from the
seed. The paper's 23 settings are the 13 pairs `(qs, qw)` from 0.1, 0.3, 0.5, and 0.7 with
`qs + qw <= 1`, and `qs` from 0.1 to 1.0 without weak edges; its sample sizes are 10,000 rows in the
main text and 1,000 and 100 in the supplement. The lane runs each setting at each sample size, 69
cells, on 20 datasets per cell.

The paper also reports 100 variables. There it replaces GES above the lowest penalty with
ARGES-skeleton, a variant Andrey does not implement, so the lane stops at 10 variables.

## Solutions

- `andrey.ages.numpy` is `ages(data)` with `penalty_discount=1.0`, the BIC.
- `pcalg.ages` is `pcalg::ages(data)` with its defaults: `lambda_min = 0.5 * log(n)`, the same BIC
  in log-likelihood units, the plain search (`adaptive = "none"`), and no degree cap.
- `andrey.ges.numpy.bic` is Andrey's `ges` with the BIC (`ebic_gamma=0.0`) and `max_parents=9`, so
  no limit binds: the baseline the paper compares with, and the first CPDAG of Andrey's path.

Each fit runs in a fresh process on one core with one BLAS thread, after one untimed warm-up fit,
under `--cap-wall-s`.

## Measures

A graph is scored against the generating DAG.

- **Precision and recall**, the paper's measures. Precision is the share of the graph's directed
  edges that the true DAG directs the same way, and recall the share of the true DAG's edges that
  the graph directs correctly. Both pool the counts of a cell's datasets (`correct`, `directed`,
  `true_edges`), since a graph without a directed edge has no precision of its own.
- **`shd`** counts the variable pairs where the graph differs from every graph between the true
  CPDAG and the true DAG. The graph AGES estimates has the true skeleton, directs every edge the
  true CPDAG directs, and directs no edge against the true DAG. So a missing or extra edge, an edge
  directed against the true DAG, and an edge left undirected that the true CPDAG directs each count
  one, and an edge directed as the true DAG directs it counts nothing, whether or not the CPDAG
  directs it. SHD against the true CPDAG, `shd_cpdag`, would count each correct direction beyond
  the CPDAG as an error, and SHD against the true DAG, `shd_dag`, every edge of a correct CPDAG
  that no data can direct; the records hold both.
- **`empty_shd`** is the empty graph's count on each of the three, the number of true edges.

`ages_summary.py` pairs a cell's datasets by seed and compares `shd`, `correct`, and `wrong` (the
directed edges that are not correct) between two solutions: the mean of the per-dataset difference
with its percentile-bootstrap 95% interval over 10,000 resamples of the datasets. Against pcalg, a
cell where every dataset has the same graph is `identical`. Otherwise a measure is worse when its
interval lies wholly on the worse side of 0, and the cell is `worse` if any measure is, `better` if
one is better and none worse, and a `tie` otherwise. A worse cell counts only if it is worse again
on fresh seeds.

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with pcalg in the library, then:

```shell
$ANDREY_BENCH_PYTHON benchmarks/ages_cpu.py --out outputs/ages-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/ages_cpu.py --out outputs/ages-cpu --cores 1
$ANDREY_BENCH_PYTHON benchmarks/ages_summary.py outputs/ages-cpu --json outputs/ages-cells.json
```

The defaults are the paper's 23 settings and three sample sizes, 20 seeds from `--first-seed 0`, one
untimed warm-up fit, one timed fit, and `--cap-wall-s` of 1800. `--cells 0.3:0.7 0.5:0.0` selects
settings, and `--part` with `--parts` splits them over jobs. The records go to `records.jsonl`, one
line per fit, with each graph's endpoint marks, so `--rescore` recomputes the scores without
fitting, and the build and machine to `provenance.json`. Nearly all of a fit's wall time is the
start of its process, about a second for Andrey and several for R.

## Results (0.17.0)

The rows are in [`published/0.17.0/ages.json`](../published/0.17.0/ages.json): 69 cells, 20
datasets each, seeds 0 to 19, 4,140 fits, none failed or timed out. Andrey is the 0.17.0 release
branch, pcalg is 2.7-12 on R 4.6.1, and every fit ran on one core of an AMD EPYC 7713.

**Against pcalg.** Andrey returned pcalg's graph on 1,378 of the 1,380 datasets. 67 cells are
identical and 2 are ties: with 10,000 rows, one dataset of `(0.1, 0.7)` and one of `(0.3, 0.7)`
differ, and no measure's interval excludes 0. No cell is worse, so none was rerun on fresh seeds.
Andrey's median fit took 0.016 to 0.22 s and pcalg's 0.18 to 1.9 s, 6.7 to 11.9 times as long.
Andrey's peak memory was 110 MB.

**Against the empty graph.** In 62 cells `shd` is below the empty graph's with an interval that
excludes 0, for both packages. In the other 7 neither package's interval excludes 0: `(0.4, 0)`
and `(0.6, 0)` with 10,000 rows, the same two and `(0.5, 0.1)` with 1,000, and `(0.4, 0)` and
`(0.5, 0.1)` with 100. These are graphs with about 18 to 27 edges on 10 variables, nearly all
strong, where GES itself adds and misdirects many edges.

**Against GES at the same penalty.** The skeleton is that of GES on every dataset.

| Rows | Cells with more correct directions | More wrong directions | Recall: AGES | GES | Precision: AGES | GES | Graphs equal to the GES CPDAG |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 20 of 23 | 20 of 23 | 0.49 | 0.46 | 0.49 | 0.49 | 195 of 460 |
| 1,000 | 13 of 23 | 15 of 23 | 0.44 | 0.43 | 0.49 | 0.49 | 310 of 460 |
| 100 | 1 of 23 | 3 of 23 | 0.32 | 0.32 | 0.46 | 0.46 | 398 of 460 |

A cell counts when the interval of the paired difference excludes 0; recall and precision pool
every cell of the sample size. With 10,000 rows AGES directs about two more edges per graph than
GES (28.4 against 26.5), and about half of them are correct, the same share as among the edges GES
directs: more correct directions at the same precision, as the paper reports, and more wrong ones
in the same cells. The three cells without a gain at 10,000 rows are `(0.1, 0.1)`, `(0.1, 0)`, and
`(0.2, 0)`, the sparsest. On `shd` AGES is better than GES in 4 cells with 10,000 rows, worse in 5,
and tied in 14; with 1,000 rows better in 3, worse in 3; with 100 rows tied in all 23. With 100
rows there is almost nothing to gain, as the paper finds.

The AGES graph held an unshielded collider that the GES CPDAG lacks on 8 of 460 datasets with
10,000 rows (12 colliders), on 3 with 1,000 rows, and on none with 100.

A fit of AGES took 1.6 to 3.9 times one of GES at 10 variables.

In the tables, Strong and Weak are `qs` and `qw`, Edges and Empty the mean number of true edges,
SHD the mean `shd`, and the fit time the median.

### 10,000 rows

| Strong | Weak | Edges | Same graph | SHD: Andrey | pcalg | GES | Empty | Precision: AGES | GES | Recall: AGES | GES | Fit, s: Andrey | pcalg | GES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 8.1 | 20 of 20 | 0.5 | 0.5 | 0.5 | 8.1 | 0.93 | 0.92 | 0.62 | 0.60 | 0.02 | 0.28 | 0.014 |
| 0.1 | 0.3 | 16.9 | 20 of 20 | 6.2 | 6.2 | 6.1 | 16.9 | 0.68 | 0.69 | 0.68 | 0.65 | 0.07 | 0.72 | 0.025 |
| 0.1 | 0.5 | 27.5 | 20 of 20 | 18.6 | 18.6 | 18.9 | 27.5 | 0.48 | 0.49 | 0.55 | 0.53 | 0.13 | 1.09 | 0.036 |
| 0.1 | 0.7 | 37.1 | 19 of 20 | 21.3 | 21.3 | 22.2 | 37.1 | 0.51 | 0.51 | 0.49 | 0.45 | 0.17 | 1.28 | 0.044 |
| 0.3 | 0.1 | 18.6 | 20 of 20 | 12.1 | 12.1 | 12.6 | 18.6 | 0.51 | 0.50 | 0.62 | 0.57 | 0.08 | 0.81 | 0.030 |
| 0.3 | 0.3 | 27.9 | 20 of 20 | 18.4 | 18.4 | 18.4 | 27.9 | 0.52 | 0.52 | 0.59 | 0.57 | 0.15 | 1.27 | 0.042 |
| 0.3 | 0.5 | 35.4 | 20 of 20 | 22.2 | 22.2 | 22.6 | 35.4 | 0.51 | 0.51 | 0.52 | 0.49 | 0.20 | 1.55 | 0.052 |
| 0.3 | 0.7 | 45.0 | 19 of 20 | 26.1 | 26.1 | 24.8 | 45.0 | 0.46 | 0.46 | 0.38 | 0.36 | 0.22 | 1.86 | 0.060 |
| 0.5 | 0.1 | 27.9 | 20 of 20 | 20.8 | 20.8 | 20.8 | 27.9 | 0.47 | 0.48 | 0.55 | 0.53 | 0.22 | 1.43 | 0.063 |
| 0.5 | 0.3 | 36.5 | 20 of 20 | 26.4 | 26.4 | 27.3 | 36.5 | 0.42 | 0.41 | 0.43 | 0.39 | 0.22 | 1.52 | 0.056 |
| 0.5 | 0.5 | 45.0 | 20 of 20 | 25.6 | 25.6 | 23.1 | 45.0 | 0.47 | 0.47 | 0.41 | 0.36 | 0.21 | 1.64 | 0.054 |
| 0.7 | 0.1 | 35.9 | 20 of 20 | 25.8 | 25.8 | 26.6 | 35.9 | 0.42 | 0.44 | 0.44 | 0.41 | 0.20 | 1.60 | 0.053 |
| 0.7 | 0.3 | 45.0 | 20 of 20 | 25.6 | 25.6 | 23.9 | 45.0 | 0.46 | 0.46 | 0.39 | 0.36 | 0.19 | 1.78 | 0.051 |
| 0.1 | 0.0 | 5.0 | 20 of 20 | 0.1 | 0.1 | 0.1 | 5.0 | 0.98 | 0.98 | 0.48 | 0.46 | 0.02 | 0.20 | 0.011 |
| 0.2 | 0.0 | 8.9 | 20 of 20 | 1.8 | 1.8 | 1.6 | 8.9 | 0.78 | 0.82 | 0.63 | 0.62 | 0.03 | 0.29 | 0.015 |
| 0.3 | 0.0 | 13.7 | 20 of 20 | 4.0 | 4.0 | 3.5 | 13.7 | 0.73 | 0.75 | 0.67 | 0.62 | 0.06 | 0.59 | 0.025 |
| 0.4 | 0.0 | 18.3 | 20 of 20 | 15.2 | 15.2 | 15.1 | 18.3 | 0.43 | 0.45 | 0.56 | 0.54 | 0.12 | 1.02 | 0.040 |
| 0.5 | 0.0 | 21.7 | 20 of 20 | 12.2 | 12.2 | 12.2 | 21.7 | 0.58 | 0.59 | 0.66 | 0.63 | 0.14 | 1.09 | 0.048 |
| 0.6 | 0.0 | 26.6 | 20 of 20 | 23.1 | 23.1 | 23.0 | 26.6 | 0.39 | 0.40 | 0.47 | 0.46 | 0.18 | 1.34 | 0.047 |
| 0.7 | 0.0 | 33.0 | 20 of 20 | 24.5 | 24.5 | 24.7 | 33.0 | 0.44 | 0.44 | 0.48 | 0.46 | 0.19 | 1.34 | 0.049 |
| 0.8 | 0.0 | 36.4 | 20 of 20 | 25.9 | 25.9 | 26.1 | 36.4 | 0.45 | 0.46 | 0.46 | 0.44 | 0.19 | 1.37 | 0.061 |
| 0.9 | 0.0 | 40.2 | 20 of 20 | 25.4 | 25.4 | 25.8 | 40.2 | 0.46 | 0.47 | 0.44 | 0.41 | 0.20 | 1.40 | 0.057 |
| 1.0 | 0.0 | 45.0 | 20 of 20 | 24.6 | 24.6 | 22.6 | 45.0 | 0.50 | 0.51 | 0.42 | 0.39 | 0.21 | 1.48 | 0.054 |

### 1,000 rows

| Strong | Weak | Edges | Same graph | SHD: Andrey | pcalg | GES | Empty | Precision: AGES | GES | Recall: AGES | GES | Fit, s: Andrey | pcalg | GES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 8.7 | 20 of 20 | 1.2 | 1.2 | 1.2 | 8.7 | 0.86 | 0.86 | 0.58 | 0.58 | 0.03 | 0.27 | 0.012 |
| 0.1 | 0.3 | 19.1 | 20 of 20 | 7.9 | 7.9 | 8.0 | 19.1 | 0.72 | 0.72 | 0.62 | 0.62 | 0.06 | 0.48 | 0.019 |
| 0.1 | 0.5 | 27.4 | 20 of 20 | 15.9 | 15.9 | 16.4 | 27.4 | 0.57 | 0.57 | 0.50 | 0.48 | 0.08 | 0.74 | 0.025 |
| 0.1 | 0.7 | 34.8 | 20 of 20 | 22.1 | 22.1 | 22.2 | 34.8 | 0.52 | 0.52 | 0.42 | 0.41 | 0.10 | 0.85 | 0.027 |
| 0.3 | 0.1 | 18.0 | 20 of 20 | 10.2 | 10.2 | 10.3 | 18.0 | 0.57 | 0.57 | 0.61 | 0.59 | 0.07 | 0.56 | 0.025 |
| 0.3 | 0.3 | 26.6 | 20 of 20 | 17.6 | 17.6 | 18.0 | 26.6 | 0.54 | 0.54 | 0.53 | 0.51 | 0.10 | 0.97 | 0.032 |
| 0.3 | 0.5 | 36.5 | 20 of 20 | 24.1 | 24.1 | 24.4 | 36.5 | 0.52 | 0.52 | 0.44 | 0.42 | 0.14 | 1.06 | 0.037 |
| 0.3 | 0.7 | 45.0 | 20 of 20 | 26.9 | 26.9 | 26.1 | 45.0 | 0.53 | 0.55 | 0.39 | 0.38 | 0.15 | 1.43 | 0.043 |
| 0.5 | 0.1 | 27.3 | 20 of 20 | 24.4 | 24.4 | 24.6 | 27.3 | 0.37 | 0.37 | 0.42 | 0.41 | 0.15 | 1.02 | 0.037 |
| 0.5 | 0.3 | 36.6 | 20 of 20 | 24.9 | 24.9 | 25.1 | 36.6 | 0.48 | 0.48 | 0.44 | 0.43 | 0.15 | 1.10 | 0.038 |
| 0.5 | 0.5 | 45.0 | 20 of 20 | 27.3 | 27.3 | 27.1 | 45.0 | 0.46 | 0.46 | 0.36 | 0.35 | 0.14 | 1.18 | 0.037 |
| 0.7 | 0.1 | 35.5 | 20 of 20 | 27.0 | 27.0 | 27.4 | 35.5 | 0.42 | 0.42 | 0.41 | 0.39 | 0.15 | 1.19 | 0.041 |
| 0.7 | 0.3 | 45.0 | 20 of 20 | 26.1 | 26.1 | 25.4 | 45.0 | 0.51 | 0.51 | 0.39 | 0.38 | 0.16 | 1.28 | 0.040 |
| 0.1 | 0.0 | 4.8 | 20 of 20 | 0.9 | 0.9 | 0.8 | 4.8 | 0.74 | 0.74 | 0.44 | 0.42 | 0.02 | 0.18 | 0.008 |
| 0.2 | 0.0 | 9.1 | 20 of 20 | 1.1 | 1.1 | 0.9 | 9.1 | 0.89 | 0.90 | 0.69 | 0.66 | 0.03 | 0.25 | 0.013 |
| 0.3 | 0.0 | 13.4 | 20 of 20 | 7.2 | 7.2 | 7.1 | 13.4 | 0.57 | 0.58 | 0.57 | 0.56 | 0.04 | 0.36 | 0.018 |
| 0.4 | 0.0 | 18.3 | 20 of 20 | 15.6 | 15.6 | 15.7 | 18.3 | 0.42 | 0.41 | 0.49 | 0.48 | 0.09 | 0.64 | 0.029 |
| 0.5 | 0.0 | 22.9 | 20 of 20 | 17.9 | 17.9 | 18.0 | 22.9 | 0.48 | 0.48 | 0.57 | 0.55 | 0.11 | 0.87 | 0.034 |
| 0.6 | 0.0 | 27.2 | 20 of 20 | 24.8 | 24.8 | 24.8 | 27.2 | 0.37 | 0.37 | 0.43 | 0.42 | 0.12 | 0.94 | 0.034 |
| 0.7 | 0.0 | 32.3 | 20 of 20 | 27.4 | 27.4 | 27.6 | 32.3 | 0.36 | 0.36 | 0.38 | 0.37 | 0.14 | 0.96 | 0.037 |
| 0.8 | 0.0 | 35.8 | 20 of 20 | 27.1 | 27.1 | 27.6 | 35.8 | 0.42 | 0.41 | 0.42 | 0.39 | 0.14 | 1.01 | 0.037 |
| 0.9 | 0.0 | 40.6 | 20 of 20 | 28.6 | 28.6 | 28.6 | 40.6 | 0.42 | 0.42 | 0.36 | 0.35 | 0.15 | 1.07 | 0.039 |
| 1.0 | 0.0 | 45.0 | 20 of 20 | 25.5 | 25.5 | 24.8 | 45.0 | 0.53 | 0.53 | 0.41 | 0.40 | 0.15 | 1.05 | 0.036 |

### 100 rows

| Strong | Weak | Edges | Same graph | SHD: Andrey | pcalg | GES | Empty | Precision: AGES | GES | Recall: AGES | GES | Fit, s: Andrey | pcalg | GES |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 7.4 | 20 of 20 | 4.5 | 4.5 | 4.5 | 7.4 | 0.58 | 0.58 | 0.41 | 0.41 | 0.02 | 0.22 | 0.010 |
| 0.1 | 0.3 | 17.1 | 20 of 20 | 11.5 | 11.5 | 11.5 | 17.1 | 0.63 | 0.63 | 0.39 | 0.39 | 0.03 | 0.29 | 0.012 |
| 0.1 | 0.5 | 25.6 | 20 of 20 | 20.1 | 20.1 | 20.1 | 25.6 | 0.57 | 0.57 | 0.28 | 0.28 | 0.04 | 0.37 | 0.014 |
| 0.1 | 0.7 | 36.6 | 20 of 20 | 28.6 | 28.6 | 28.6 | 36.6 | 0.54 | 0.54 | 0.24 | 0.24 | 0.05 | 0.45 | 0.018 |
| 0.3 | 0.1 | 18.4 | 20 of 20 | 13.9 | 13.9 | 14.0 | 18.4 | 0.52 | 0.52 | 0.48 | 0.48 | 0.05 | 0.41 | 0.018 |
| 0.3 | 0.3 | 26.9 | 20 of 20 | 19.1 | 19.1 | 19.1 | 26.9 | 0.54 | 0.55 | 0.37 | 0.37 | 0.06 | 0.56 | 0.020 |
| 0.3 | 0.5 | 35.7 | 20 of 20 | 26.5 | 26.5 | 26.6 | 35.7 | 0.53 | 0.54 | 0.30 | 0.30 | 0.08 | 0.60 | 0.025 |
| 0.3 | 0.7 | 45.0 | 20 of 20 | 33.4 | 33.4 | 33.4 | 45.0 | 0.49 | 0.48 | 0.24 | 0.24 | 0.08 | 0.83 | 0.028 |
| 0.5 | 0.1 | 27.9 | 20 of 20 | 25.9 | 25.9 | 25.9 | 27.9 | 0.35 | 0.35 | 0.31 | 0.31 | 0.10 | 0.72 | 0.030 |
| 0.5 | 0.3 | 37.1 | 20 of 20 | 29.2 | 29.2 | 29.3 | 37.1 | 0.45 | 0.45 | 0.30 | 0.30 | 0.10 | 0.69 | 0.032 |
| 0.5 | 0.5 | 45.0 | 20 of 20 | 28.7 | 28.7 | 28.7 | 45.0 | 0.57 | 0.57 | 0.33 | 0.33 | 0.10 | 0.89 | 0.028 |
| 0.7 | 0.1 | 36.7 | 20 of 20 | 29.7 | 29.7 | 29.7 | 36.7 | 0.40 | 0.40 | 0.30 | 0.30 | 0.10 | 0.91 | 0.030 |
| 0.7 | 0.3 | 45.0 | 20 of 20 | 32.1 | 32.1 | 31.9 | 45.0 | 0.44 | 0.44 | 0.27 | 0.27 | 0.10 | 0.97 | 0.030 |
| 0.1 | 0.0 | 4.3 | 20 of 20 | 3.0 | 3.0 | 3.0 | 4.3 | 0.49 | 0.48 | 0.50 | 0.48 | 0.02 | 0.18 | 0.008 |
| 0.2 | 0.0 | 8.7 | 20 of 20 | 4.5 | 4.5 | 4.7 | 8.7 | 0.56 | 0.55 | 0.52 | 0.50 | 0.03 | 0.27 | 0.011 |
| 0.3 | 0.0 | 13.4 | 20 of 20 | 8.2 | 8.2 | 8.2 | 13.4 | 0.55 | 0.55 | 0.57 | 0.57 | 0.04 | 0.38 | 0.017 |
| 0.4 | 0.0 | 18.1 | 20 of 20 | 15.8 | 15.8 | 15.9 | 18.1 | 0.41 | 0.41 | 0.43 | 0.42 | 0.06 | 0.53 | 0.023 |
| 0.5 | 0.0 | 22.9 | 20 of 20 | 20.9 | 20.9 | 20.9 | 22.9 | 0.39 | 0.39 | 0.39 | 0.39 | 0.08 | 0.69 | 0.026 |
| 0.6 | 0.0 | 27.0 | 20 of 20 | 24.7 | 24.7 | 24.6 | 27.0 | 0.38 | 0.38 | 0.36 | 0.36 | 0.08 | 0.70 | 0.028 |
| 0.7 | 0.0 | 31.2 | 20 of 20 | 25.2 | 25.2 | 25.1 | 31.2 | 0.41 | 0.41 | 0.34 | 0.34 | 0.09 | 0.69 | 0.028 |
| 0.8 | 0.0 | 36.3 | 20 of 20 | 29.7 | 29.7 | 29.8 | 36.3 | 0.40 | 0.40 | 0.31 | 0.30 | 0.10 | 0.82 | 0.031 |
| 0.9 | 0.0 | 40.5 | 20 of 20 | 31.3 | 31.3 | 31.4 | 40.5 | 0.41 | 0.41 | 0.29 | 0.29 | 0.10 | 0.81 | 0.030 |
| 1.0 | 0.0 | 45.0 | 20 of 20 | 30.6 | 30.6 | 30.4 | 45.0 | 0.49 | 0.49 | 0.28 | 0.28 | 0.10 | 0.77 | 0.030 |
