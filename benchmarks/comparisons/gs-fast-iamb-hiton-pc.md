# GS, Fast-IAMB, and HITON-PC

This campaign compares Andrey's `gs`, `fast_iamb`, and `hiton_pc` with bnlearn 5.2.1's `gs`,
`fast.iamb`, and `si.hiton.pc` on the set each learns at every target, on four published networks
and on linear-Gaussian graphs. `blanket_cpu.py` fits the solutions, as
[Markov blankets](blankets.md) describes, and `blanket_summary.py` judges each cell. The rows are in
[`published/0.18.0/gs-fast-iamb-hiton-pc.json`](../published/0.18.0/gs-fast-iamb-hiton-pc.json),
and the [0.18.0 release card](../../docs/docs/benchmarks/0.18.0.md) summarizes them.

## Data

- **Networks.** alarm (37 variables), child (20), insurance (27), and hepar2 (70), sampled forward
  from each network's own tables. GS and Fast-IAMB run at 1,000, 5,000, and 20,000 rows, up to the
  20,000 of their papers' experiments; HITON-PC at 200, 500, and 5,000 rows, as in its paper's
  Section 6.
  A small sample of a skewed network can hold a column with one value, which Andrey refuses, so the
  driver drops such columns and their nodes (`--drop-constant`) before every fit of every solution.
- **Gaussian graphs.** Linear-Gaussian Erdos-Renyi graphs, standardized: 20, 50, and 100 variables
  at 1,000 and 5,000 rows with mean degree 2, and 50 variables with mean degree 4.

Every cell has 20 datasets, seeds 0 to 19, with the columns in an order drawn from the seed. Every
variable of a dataset is a target, answered in one fit.

## Solutions

| Andrey | bnlearn 5.2.1 | Role |
|---|---|---|
| `gs` | `learn.mb(method = "gs")` | judged |
| `fast_iamb` | `learn.mb(method = "fast.iamb")` | judged |
| `hiton_pc`, `max_depth=None` | `learn.nbr(method = "si.hiton.pc")`, kept to members whose own set holds the target | judged |
| `iamb`, `inter_iamb` | | context for GS and Fast-IAMB |
| `mmpc`; `hiton_pc` at its default `max_depth=3` | | context for HITON-PC |
| `hiton_pc` without its rows-per-cell rule | | context: an arm that does not ship |

The judged pairs run the same data, the same `alpha` of 0.05, and no limit on the conditioning set
on both sides: with a limit, bnlearn's `si.hiton.pc` also stops growing the candidate set, so the
parents-and-children learners agree only without one. On continuous data both sides run Fisher's z
(`fisherz` and `zf`). On discrete data Andrey runs its G-square test (`gsq`) and bnlearn its
`mi-adf`, which computes the same statistic and, like Andrey's test, counts the degrees of freedom
from the configurations that occur in the data: Andrey's are the statistic's mean when one variable
is shuffled within each configuration, and `mi-adf`'s add up `(r_x - 1)(r_y - 1)` over the
configurations, where `r` counts the values present. `mi-adf` also treats a table with fewer than 5
rows per cell as independent. bnlearn's plain `mi`, which counts every possible configuration and
has no such rule, runs as context and is never judged.

The arm without the rows-per-cell rule runs `hiton_pc` with the rule's threshold set to 0, so every
test runs; it is not an option of `hiton_pc`.

Each fit runs in a fresh process on one core with one BLAS thread, without a warm-up fit, under a
cap of 1,800 s. All 228 comparisons have 20 paired datasets; no fit failed or reached the cap.

## Measures

Per dataset, over the targets whose true set is not empty:

- **distance**: the mean over the targets of the Euclidean distance of the set's (precision, recall)
  from (1, 1). The papers define precision, recall, and distance per target but not for an empty
  answer; here an empty answer has no wrong member, so it counts precision 1 and distance 1.
- **F1**: the mean over the targets of 2 tp / (found + true), 0 for an empty answer.
- **errors**: the mean over the targets of the missing plus the extra members, the wrong answers of
  `andrey.metrics.target_answer_scores`. The empty set's errors are the size of the true set: this
  is the floor.

D is Andrey's measure minus the other's per dataset, with the 95% percentile-bootstrap interval of
its mean over the datasets (10,000 resamples). A measure is worse when the interval lies wholly on
its worse side of 0 and better when it lies wholly on the better side. A cell is worse when either
measure is worse, better when one is better and neither is worse, and a tie otherwise. A cell worse
on seeds 0 to 19 was rerun on seeds 20 to 39 and stays worse only when the rerun is worse too. The
reruns ran the cells where some comparison was worse; the tables show them where they exist. A
method passes the floor when the interval of its errors minus the empty set's lies wholly below 0.
The tests per target are the mean number of tests of a search for one target, and a fit's time is
the median over the datasets of the wall time of the search for every target.

## Results (0.18.0)

Andrey is the 0.18.0 release branch with `gs`, `fast_iamb`, and `hiton_pc` merged; bnlearn is 5.2.1
on R 4.6.1, and every fit ran on one core of an AMD EPYC 7713.

| Method | Better | Tie | Worse | Below the empty set's errors |
|---|---:|---:|---:|---|
| GS | 13 | 6 | 1 | every cell but hepar2 |
| Fast-IAMB | 10 | 7 | 3 | every cell but hepar2 and Gaussian graphs with 100 variables |
| HITON-PC | 13 | 7 | 0 | every cell |

On the networks, bnlearn's searches with `mi-adf` often make more errors than the empty set: `gs`
in all 12 cells, `fast.iamb` on hepar2 and on child at 20,000 rows, and `si.hiton.pc` on hepar2 at
200 rows. Andrey's leads on the networks are measured against those answers, so the context tables
add bnlearn's plain `mi`.

On the discrete networks, Andrey's fits take longer than bnlearn's compiled searches: the median
GS fit took 1.3 to 26 s against 0.12 to 5.2 s, 4.5 to 14 times as long, Fast-IAMB's 0.78 to 16 s
against 0.12 to 3.6 s, and HITON-PC's 1.7 to 18 s against 0.14 to 4.5 s. On the Gaussian graphs
GS took 0.35 to 4.1 s and bnlearn 0.17 to 162 s; at 100 variables Andrey was 13 to 40 times faster.
Fast-IAMB took 0.35 to 3.5 s against 0.08 to 4.1 s, and HITON-PC 0.42 to 4.3 s against 0.17 to
97 s.

### GS

| Data | Rows | Distance: Andrey | bnlearn | D [95%] | F1: Andrey | bnlearn | D [95%] | Seeds 20-39: D distance | D F1 | Verdict |
|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| alarm | 1,000 | 0.395 | 0.943 | -0.548 [-0.587, -0.510] | 0.724 | 0.324 | +0.400 [+0.372, +0.428] | -0.509 [-0.546, -0.473] | +0.373 [+0.346, +0.400] | better |
| alarm | 5,000 | 0.338 | 0.889 | -0.551 [-0.586, -0.514] | 0.766 | 0.364 | +0.402 [+0.375, +0.427] |  |  | better |
| alarm | 20,000 | 0.260 | 0.884 | -0.624 [-0.662, -0.585] | 0.821 | 0.366 | +0.455 [+0.428, +0.482] |  |  | better |
| child | 1,000 | 0.263 | 0.825 | -0.562 [-0.605, -0.517] | 0.825 | 0.408 | +0.417 [+0.384, +0.448] |  |  | better |
| child | 5,000 | 0.224 | 0.905 | -0.681 [-0.738, -0.622] | 0.849 | 0.346 | +0.503 [+0.461, +0.543] |  |  | better |
| child | 20,000 | 0.164 | 0.928 | -0.764 [-0.808, -0.719] | 0.887 | 0.323 | +0.564 [+0.531, +0.596] |  |  | better |
| insurance | 1,000 | 0.457 | 1.032 | -0.575 [-0.616, -0.533] | 0.684 | 0.249 | +0.435 [+0.404, +0.465] |  |  | better |
| insurance | 5,000 | 0.368 | 1.038 | -0.670 [-0.712, -0.627] | 0.748 | 0.254 | +0.494 [+0.464, +0.523] |  |  | better |
| insurance | 20,000 | 0.285 | 1.035 | -0.750 [-0.791, -0.706] | 0.807 | 0.259 | +0.548 [+0.518, +0.576] |  |  | better |
| hepar2 | 1,000 | 0.827 | 1.104 | -0.277 [-0.306, -0.247] | 0.384 | 0.194 | +0.190 [+0.170, +0.211] |  |  | better |
| hepar2 | 5,000 | 0.664 | 1.090 | -0.426 [-0.452, -0.400] | 0.497 | 0.199 | +0.298 [+0.280, +0.317] |  |  | better |
| hepar2 | 20,000 | 0.594 | 1.096 | -0.502 [-0.539, -0.465] | 0.546 | 0.188 | +0.358 [+0.334, +0.382] |  |  | better |
| Gaussian, d 20, degree 2 | 1,000 | 0.144 | 0.141 | +0.003 [-0.007, +0.014] | 0.907 | 0.909 | -0.002 [-0.009, +0.005] |  |  | tie |
| Gaussian, d 20, degree 2 | 5,000 | 0.151 | 0.155 | -0.005 [-0.015, +0.005] | 0.903 | 0.898 | +0.005 [-0.002, +0.012] |  |  | tie |
| Gaussian, d 50, degree 2 | 1,000 | 0.317 | 0.318 | -0.001 [-0.009, +0.007] | 0.788 | 0.786 | +0.002 [-0.004, +0.008] | -0.007 [-0.017, +0.003] | +0.006 [-0.001, +0.013] | tie |
| Gaussian, d 50, degree 2 | 5,000 | 0.311 | 0.313 | -0.002 [-0.012, +0.007] | 0.792 | 0.787 | +0.005 [-0.002, +0.012] |  |  | tie |
| Gaussian, d 100, degree 2 | 1,000 | 0.496 | 0.496 | +0.000 [-0.006, +0.008] | 0.641 | 0.639 | +0.002 [-0.004, +0.007] | -0.001 [-0.007, +0.006] | +0.002 [-0.004, +0.008] | tie |
| Gaussian, d 100, degree 2 | 5,000 | 0.489 | 0.498 | -0.009 [-0.014, -0.005] | 0.647 | 0.638 | +0.009 [+0.005, +0.014] | -0.003 [-0.010, +0.003] | +0.005 [-0.000, +0.010] | better |
| Gaussian, d 50, degree 4 | 1,000 | 0.195 | 0.173 | +0.022 [+0.011, +0.035] | 0.875 | 0.890 | -0.015 [-0.025, -0.006] | +0.007 [-0.009, +0.023] | -0.005 [-0.017, +0.007] | tie |
| Gaussian, d 50, degree 4 | 5,000 | 0.169 | 0.156 | +0.013 [+0.002, +0.023] | 0.895 | 0.902 | -0.008 [-0.015, -0.001] | +0.011 [+0.001, +0.022] | -0.006 [-0.014, +0.002] | worse |

| Data | Rows | Errors: Andrey | bnlearn | Empty set | Andrey minus empty [95%] | Tests per target: Andrey | bnlearn | Fit, s: Andrey | bnlearn |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| alarm | 1,000 | 2.08 | 4.46 | 3.51 | -1.43 [-1.54, -1.32] | 94 | 60 | 2.35 | 0.19 |
| alarm | 5,000 | 1.81 | 4.94 | 3.51 | -1.70 [-1.80, -1.60] | 96 | 68 | 3.85 | 0.49 |
| alarm | 20,000 | 1.38 | 5.49 | 3.51 | -2.14 [-2.21, -2.06] | 98 | 72 | 7.85 | 1.60 |
| child | 1,000 | 1.17 | 3.69 | 3.00 | -1.83 [-1.93, -1.74] | 45 | 31 | 1.29 | 0.12 |
| child | 5,000 | 0.93 | 4.93 | 3.00 | -2.07 [-2.17, -1.97] | 47 | 32 | 1.89 | 0.24 |
| child | 20,000 | 0.63 | 6.12 | 3.00 | -2.37 [-2.46, -2.27] | 49 | 32 | 3.22 | 0.71 |
| insurance | 1,000 | 2.78 | 5.89 | 5.12 | -2.34 [-2.40, -2.27] | 66 | 39 | 1.46 | 0.13 |
| insurance | 5,000 | 2.30 | 6.72 | 5.19 | -2.89 [-2.98, -2.78] | 70 | 41 | 2.87 | 0.34 |
| insurance | 20,000 | 1.84 | 7.55 | 5.19 | -3.35 [-3.43, -3.27] | 72 | 42 | 5.10 | 0.94 |
| hepar2 | 1,000 | 5.61 | 7.15 | 4.52 | +1.09 [+1.01, +1.18] | 206 | 144 | 7.81 | 0.57 |
| hepar2 | 5,000 | 5.07 | 8.22 | 4.51 | +0.56 [+0.46, +0.67] | 221 | 145 | 12.7 | 1.51 |
| hepar2 | 20,000 | 4.75 | 9.47 | 4.51 | +0.23 [+0.13, +0.35] | 234 | 143 | 26.1 | 5.20 |
| Gaussian, d 20, degree 2 | 1,000 | 0.57 | 0.58 | 3.35 | -2.77 [-2.95, -2.61] | 49 | 54 | 0.35 | 0.17 |
| Gaussian, d 20, degree 2 | 5,000 | 0.60 | 0.65 | 3.35 | -2.75 [-2.90, -2.61] | 49 | 55 | 0.45 | 0.57 |
| Gaussian, d 50, degree 2 | 1,000 | 1.71 | 1.75 | 3.72 | -2.01 [-2.17, -1.86] | 134 | 209 | 0.68 | 2.52 |
| Gaussian, d 50, degree 2 | 5,000 | 1.63 | 1.75 | 3.72 | -2.09 [-2.27, -1.89] | 133 | 218 | 1.21 | 12.7 |
| Gaussian, d 100, degree 2 | 1,000 | 3.57 | 3.68 | 3.88 | -0.31 [-0.46, -0.17] | 295 | 596 | 2.04 | 27.5 |
| Gaussian, d 100, degree 2 | 5,000 | 3.49 | 3.74 | 3.88 | -0.39 [-0.55, -0.21] | 287 | 618 | 4.08 | 161.9 |
| Gaussian, d 50, degree 4 | 1,000 | 1.92 | 1.77 | 8.52 | -6.59 [-6.88, -6.32] | 162 | 268 | 0.81 | 9.81 |
| Gaussian, d 50, degree 4 | 5,000 | 1.59 | 1.48 | 8.52 | -6.92 [-7.20, -6.67] | 156 | 262 | 1.33 | 46.3 |

GS is better than bnlearn's `gs` in all 12 network cells, where bnlearn's answers make more errors
than the empty set. Against bnlearn's plain `mi` it is better in 9 of 12 (see
[bnlearn's plain test](#bnlearns-plain-test)). On the Gaussian graphs it ties in 6 cells and is
better in one; the worse cell, 50 variables with mean degree 4 at 5,000 rows, has a distance D of
+0.011 [+0.001, +0.022] on the rerun and a tie on F1, and at 1,000 rows the same graphs tie on the
rerun. The cause of the worse Gaussian rows is being traced.

### Fast-IAMB

| Data | Rows | Distance: Andrey | bnlearn | D [95%] | F1: Andrey | bnlearn | D [95%] | Seeds 20-39: D distance | D F1 | Verdict |
|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| alarm | 1,000 | 0.460 | 0.434 | +0.026 [+0.016, +0.035] | 0.675 | 0.694 | -0.019 [-0.026, -0.013] | +0.023 [+0.009, +0.038] | -0.019 [-0.029, -0.009] | worse |
| alarm | 5,000 | 0.412 | 0.408 | +0.004 [-0.012, +0.021] | 0.715 | 0.719 | -0.004 [-0.017, +0.008] |  |  | tie |
| alarm | 20,000 | 0.399 | 0.410 | -0.011 [-0.035, +0.010] | 0.722 | 0.715 | +0.007 [-0.009, +0.025] |  |  | tie |
| child | 1,000 | 0.344 | 0.412 | -0.068 [-0.085, -0.051] | 0.764 | 0.712 | +0.052 [+0.040, +0.065] |  |  | better |
| child | 5,000 | 0.302 | 0.405 | -0.103 [-0.116, -0.090] | 0.791 | 0.711 | +0.080 [+0.071, +0.090] |  |  | better |
| child | 20,000 | 0.262 | 0.478 | -0.217 [-0.249, -0.182] | 0.822 | 0.649 | +0.173 [+0.144, +0.199] |  |  | better |
| insurance | 1,000 | 0.571 | 0.590 | -0.018 [-0.034, -0.004] | 0.587 | 0.573 | +0.014 [+0.003, +0.026] |  |  | better |
| insurance | 5,000 | 0.502 | 0.522 | -0.021 [-0.032, -0.010] | 0.648 | 0.631 | +0.017 [+0.009, +0.025] |  |  | better |
| insurance | 20,000 | 0.404 | 0.471 | -0.066 [-0.103, -0.032] | 0.722 | 0.670 | +0.052 [+0.026, +0.080] |  |  | better |
| hepar2 | 1,000 | 0.838 | 0.873 | -0.034 [-0.043, -0.026] | 0.380 | 0.347 | +0.033 [+0.027, +0.040] |  |  | better |
| hepar2 | 5,000 | 0.670 | 0.745 | -0.075 [-0.086, -0.065] | 0.495 | 0.421 | +0.074 [+0.066, +0.083] |  |  | better |
| hepar2 | 20,000 | 0.594 | 0.750 | -0.156 [-0.167, -0.144] | 0.549 | 0.405 | +0.144 [+0.135, +0.152] |  |  | better |
| Gaussian, d 20, degree 2 | 1,000 | 0.165 | 0.164 | +0.001 [-0.002, +0.003] | 0.893 | 0.894 | -0.000 [-0.002, +0.001] |  |  | tie |
| Gaussian, d 20, degree 2 | 5,000 | 0.171 | 0.172 | -0.001 [-0.006, +0.003] | 0.890 | 0.890 | +0.001 [-0.002, +0.004] |  |  | tie |
| Gaussian, d 50, degree 2 | 1,000 | 0.339 | 0.338 | +0.001 [+0.000, +0.001] | 0.772 | 0.772 | -0.000 [-0.001, -0.000] | +0.001 [-0.001, +0.003] | -0.001 [-0.002, +0.001] | tie |
| Gaussian, d 50, degree 2 | 5,000 | 0.343 | 0.343 | +0.001 [-0.001, +0.003] | 0.767 | 0.768 | -0.000 [-0.002, +0.001] |  |  | tie |
| Gaussian, d 100, degree 2 | 1,000 | 0.515 | 0.513 | +0.002 [+0.001, +0.002] | 0.625 | 0.627 | -0.002 [-0.002, -0.001] | +0.003 [+0.002, +0.003] | -0.002 [-0.003, -0.002] | worse |
| Gaussian, d 100, degree 2 | 5,000 | 0.509 | 0.507 | +0.002 [+0.001, +0.003] | 0.631 | 0.632 | -0.001 [-0.002, -0.001] | +0.002 [+0.001, +0.003] | -0.002 [-0.003, -0.001] | worse |
| Gaussian, d 50, degree 4 | 1,000 | 0.234 | 0.239 | -0.004 [-0.008, -0.001] | 0.850 | 0.846 | +0.004 [+0.002, +0.006] | -0.003 [-0.007, +0.001] | +0.003 [-0.000, +0.006] | better |
| Gaussian, d 50, degree 4 | 5,000 | 0.235 | 0.231 | +0.004 [-0.001, +0.009] | 0.850 | 0.853 | -0.003 [-0.006, +0.001] | +0.004 [+0.001, +0.007] | -0.002 [-0.004, +0.000] | tie |

| Data | Rows | Errors: Andrey | bnlearn | Empty set | Andrey minus empty [95%] | Tests per target: Andrey | bnlearn | Fit, s: Andrey | bnlearn |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| alarm | 1,000 | 2.20 | 2.10 | 3.51 | -1.32 [-1.39, -1.24] | 95 | 96 | 1.13 | 0.19 |
| alarm | 5,000 | 2.18 | 2.22 | 3.51 | -1.33 [-1.44, -1.22] | 116 | 119 | 2.22 | 0.41 |
| alarm | 20,000 | 2.24 | 2.35 | 3.51 | -1.28 [-1.40, -1.16] | 125 | 130 | 5.48 | 1.22 |
| child | 1,000 | 1.52 | 1.84 | 3.00 | -1.48 [-1.55, -1.40] | 44 | 44 | 0.78 | 0.12 |
| child | 5,000 | 1.42 | 2.09 | 3.00 | -1.58 [-1.67, -1.50] | 54 | 58 | 1.33 | 0.24 |
| child | 20,000 | 1.26 | 2.94 | 3.00 | -1.74 [-1.89, -1.59] | 60 | 58 | 2.63 | 0.64 |
| insurance | 1,000 | 3.37 | 3.51 | 5.12 | -1.75 [-1.82, -1.68] | 56 | 56 | 0.82 | 0.13 |
| insurance | 5,000 | 3.26 | 3.52 | 5.19 | -1.93 [-2.02, -1.83] | 71 | 72 | 1.73 | 0.30 |
| insurance | 20,000 | 2.86 | 3.43 | 5.19 | -2.33 [-2.49, -2.17] | 86 | 82 | 3.81 | 0.79 |
| hepar2 | 1,000 | 5.38 | 6.00 | 4.52 | +0.86 [+0.78, +0.93] | 256 | 221 | 3.04 | 0.48 |
| hepar2 | 5,000 | 4.93 | 6.18 | 4.51 | +0.42 [+0.33, +0.51] | 308 | 239 | 6.03 | 1.19 |
| hepar2 | 20,000 | 4.70 | 7.20 | 4.51 | +0.19 [+0.10, +0.28] | 359 | 215 | 16.2 | 3.55 |
| Gaussian, d 20, degree 2 | 1,000 | 0.68 | 0.68 | 3.35 | -2.67 [-2.84, -2.50] | 58 | 57 | 0.35 | 0.08 |
| Gaussian, d 20, degree 2 | 5,000 | 0.70 | 0.70 | 3.35 | -2.65 [-2.81, -2.49] | 56 | 56 | 0.44 | 0.15 |
| Gaussian, d 50, degree 2 | 1,000 | 1.86 | 1.85 | 3.72 | -1.86 [-2.02, -1.70] | 172 | 170 | 0.56 | 0.33 |
| Gaussian, d 50, degree 2 | 5,000 | 1.93 | 1.91 | 3.72 | -1.79 [-2.01, -1.57] | 165 | 164 | 1.09 | 0.85 |
| Gaussian, d 100, degree 2 | 1,000 | 3.82 | 3.79 | 3.88 | -0.06 [-0.21, +0.07] | 402 | 397 | 1.37 | 1.55 |
| Gaussian, d 100, degree 2 | 5,000 | 3.76 | 3.74 | 3.88 | -0.12 [-0.28, +0.04] | 379 | 375 | 3.47 | 4.13 |
| Gaussian, d 50, degree 4 | 1,000 | 2.28 | 2.40 | 8.52 | -6.24 [-6.52, -5.97] | 223 | 207 | 0.73 | 0.80 |
| Gaussian, d 50, degree 4 | 5,000 | 2.33 | 2.29 | 8.52 | -6.19 [-6.54, -5.86] | 204 | 188 | 1.28 | 1.93 |

Fast-IAMB is better than bnlearn's `fast.iamb` in 9 network cells and one Gaussian cell, and ties
in 7. The worse rows:

- **alarm, 1,000 rows** (distance +0.026, F1 -0.019; on the rerun +0.023 [+0.009, +0.038] and
  -0.019 [-0.029, -0.009]). The two packages compute the same G-square statistic and differ in its
  degrees of freedom on sparse tables. With `mi-adf`'s degrees of freedom in Andrey's search,
  Andrey returns bnlearn's sets on 97% of the targets and ties it on both seed halves (distance
  -0.0009 [-0.0043, +0.0025] and +0.0007 [-0.0021, +0.0040]). bnlearn's sparse-table rule and each
  of its departures from the paper's search (the shrink, the candidate pool, the stop) move the
  difference by at most 0.002. The paper orders the candidates by the significance of a G-square
  test and asks for 5 rows per cell, but gives no degrees of freedom. On alarm at this size,
  `mi-adf`'s degrees of freedom reject at most 5.1% of the true independences and have a little
  more power from 3 conditioning variables; on child they reject up to 13.5%, and the same swap
  turns Andrey's lead at 1,000 rows (distance -0.068 [-0.085, -0.051]) into a tie. On hepar2 it cuts
  Andrey's lead from -0.034 to -0.004, and on insurance it turns -0.018 into +0.005. Both choices
  are valid readings of the paper, and each wins on some networks.
- **Gaussian graphs with 100 variables, 1,000 and 5,000 rows** (distance +0.002, F1 -0.002, on both
  seed halves). Here neither package's errors are below the empty set's. The cause of the worse
  Gaussian rows is being traced.

### HITON-PC

| Data | Rows | Distance: Andrey | bnlearn | D [95%] | F1: Andrey | bnlearn | D [95%] | Seeds 20-39: D distance | D F1 | Verdict |
|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| alarm | 200 | 0.459 | 0.646 | -0.187 [-0.213, -0.162] | 0.644 | 0.457 | +0.187 [+0.165, +0.210] |  |  | better |
| alarm | 500 | 0.285 | 0.436 | -0.151 [-0.163, -0.137] | 0.784 | 0.637 | +0.147 [+0.134, +0.159] |  |  | better |
| alarm | 5,000 | 0.091 | 0.313 | -0.223 [-0.240, -0.205] | 0.935 | 0.739 | +0.195 [+0.180, +0.212] |  |  | better |
| child | 200 | 0.444 | 0.561 | -0.118 [-0.149, -0.085] | 0.656 | 0.521 | +0.135 [+0.106, +0.162] |  |  | better |
| child | 500 | 0.204 | 0.403 | -0.199 [-0.218, -0.181] | 0.853 | 0.686 | +0.168 [+0.153, +0.182] |  |  | better |
| child | 5,000 | 0.006 | 0.236 | -0.230 [-0.247, -0.213] | 0.997 | 0.821 | +0.176 [+0.162, +0.189] |  |  | better |
| insurance | 200 | 0.591 | 0.778 | -0.187 [-0.202, -0.173] | 0.570 | 0.340 | +0.230 [+0.213, +0.247] |  |  | better |
| insurance | 500 | 0.474 | 0.670 | -0.196 [-0.214, -0.179] | 0.657 | 0.459 | +0.198 [+0.184, +0.212] |  |  | better |
| insurance | 5,000 | 0.273 | 0.545 | -0.272 [-0.292, -0.251] | 0.810 | 0.569 | +0.241 [+0.223, +0.259] |  |  | better |
| hepar2 | 200 | 0.928 | 0.955 | -0.026 [-0.042, -0.010] | 0.255 | 0.242 | +0.013 [+0.001, +0.024] |  |  | better |
| hepar2 | 500 | 0.798 | 0.817 | -0.019 [-0.030, -0.008] | 0.361 | 0.349 | +0.012 [+0.005, +0.019] |  |  | better |
| hepar2 | 5,000 | 0.489 | 0.510 | -0.021 [-0.028, -0.014] | 0.622 | 0.602 | +0.020 [+0.014, +0.026] |  |  | better |
| Gaussian, d 20, degree 2 | 1,000 | 0.094 | 0.099 | -0.005 [-0.010, -0.000] | 0.937 | 0.934 | +0.003 [+0.000, +0.006] |  |  | better |
| Gaussian, d 20, degree 2 | 5,000 | 0.067 | 0.068 | -0.001 [-0.008, +0.006] | 0.955 | 0.955 | +0.000 [-0.004, +0.005] |  |  | tie |
| Gaussian, d 50, degree 2 | 1,000 | 0.110 | 0.111 | -0.001 [-0.004, +0.001] | 0.926 | 0.925 | +0.001 [-0.001, +0.002] |  |  | tie |
| Gaussian, d 50, degree 2 | 5,000 | 0.084 | 0.085 | -0.001 [-0.007, +0.005] | 0.945 | 0.945 | +0.000 [-0.004, +0.004] |  |  | tie |
| Gaussian, d 100, degree 2 | 1,000 | 0.156 | 0.154 | +0.001 [-0.001, +0.004] | 0.895 | 0.896 | -0.001 [-0.002, +0.000] |  |  | tie |
| Gaussian, d 100, degree 2 | 5,000 | 0.127 | 0.125 | +0.002 [-0.001, +0.004] | 0.914 | 0.915 | -0.001 [-0.003, +0.001] |  |  | tie |
| Gaussian, d 50, degree 4 | 1,000 | 0.415 | 0.419 | -0.004 [-0.012, +0.004] | 0.708 | 0.704 | +0.004 [-0.002, +0.010] |  |  | tie |
| Gaussian, d 50, degree 4 | 5,000 | 0.381 | 0.392 | -0.011 [-0.026, +0.002] | 0.735 | 0.725 | +0.011 [-0.001, +0.023] |  |  | tie |

| Data | Rows | Errors: Andrey | bnlearn | Empty set | Andrey minus empty [95%] | Tests per target: Andrey | bnlearn | Fit, s: Andrey | bnlearn |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| alarm | 200 | 1.97 | 1.75 | 2.49 | -0.52 [-0.61, -0.43] | 292 | 144 | 2.67 | 0.25 |
| alarm | 500 | 0.99 | 1.26 | 2.49 | -1.49 [-1.54, -1.45] | 322 | 203 | 3.09 | 0.32 |
| alarm | 5,000 | 0.25 | 1.02 | 2.49 | -2.24 [-2.25, -2.22] | 451 | 440 | 5.81 | 1.05 |
| child | 200 | 2.12 | 1.66 | 2.50 | -0.38 [-0.47, -0.28] | 179 | 81 | 1.66 | 0.14 |
| child | 500 | 0.69 | 1.24 | 2.50 | -1.81 [-1.86, -1.76] | 243 | 127 | 2.18 | 0.18 |
| child | 5,000 | 0.03 | 0.78 | 2.50 | -2.48 [-2.49, -2.46] | 754 | 355 | 5.31 | 0.59 |
| insurance | 200 | 3.20 | 2.95 | 3.72 | -0.53 [-0.65, -0.41] | 281 | 125 | 2.01 | 0.18 |
| insurance | 500 | 2.08 | 2.60 | 3.80 | -1.71 [-1.78, -1.64] | 345 | 202 | 3.11 | 0.26 |
| insurance | 5,000 | 1.06 | 2.13 | 3.85 | -2.80 [-2.83, -2.76] | 1188 | 674 | 8.89 | 1.15 |
| hepar2 | 200 | 3.35 | 3.49 | 3.51 | -0.16 [-0.22, -0.09] | 236 | 254 | 2.50 | 0.70 |
| hepar2 | 500 | 2.97 | 3.08 | 3.51 | -0.54 [-0.59, -0.49] | 320 | 324 | 3.89 | 0.84 |
| hepar2 | 5,000 | 1.94 | 2.05 | 3.51 | -1.57 [-1.60, -1.54] | 1726 | 1223 | 18.1 | 4.51 |
| Gaussian, d 20, degree 2 | 1,000 | 0.30 | 0.32 | 2.23 | -1.93 [-2.03, -1.82] | 183 | 229 | 0.42 | 0.17 |
| Gaussian, d 20, degree 2 | 5,000 | 0.20 | 0.20 | 2.23 | -2.03 [-2.11, -1.93] | 214 | 367 | 0.51 | 0.53 |
| Gaussian, d 50, degree 2 | 1,000 | 0.34 | 0.34 | 2.30 | -1.96 [-2.02, -1.91] | 355 | 410 | 0.81 | 0.62 |
| Gaussian, d 50, degree 2 | 5,000 | 0.25 | 0.25 | 2.30 | -2.04 [-2.09, -1.99] | 477 | 1055 | 1.40 | 4.49 |
| Gaussian, d 100, degree 2 | 1,000 | 0.47 | 0.46 | 2.31 | -1.84 [-1.89, -1.79] | 668 | 703 | 2.06 | 1.86 |
| Gaussian, d 100, degree 2 | 5,000 | 0.38 | 0.37 | 2.31 | -1.93 [-1.98, -1.89] | 764 | 1675 | 4.34 | 13.9 |
| Gaussian, d 50, degree 4 | 1,000 | 2.03 | 2.01 | 4.06 | -2.04 [-2.18, -1.90] | 2020 | 2170 | 2.02 | 3.83 |
| Gaussian, d 50, degree 4 | 5,000 | 1.91 | 1.91 | 4.06 | -2.16 [-2.32, -2.00] | 5038 | 9778 | 3.46 | 97.4 |

HITON-PC is better than bnlearn's `si.hiton.pc` in all 12 network cells, with `mi-adf` and with
plain `mi`, and ties in 7 of the 8 Gaussian cells, better in the eighth. No cell is worse, and its
errors are below the empty set's in every cell.

### Fast-IAMB beside IAMB and Inter-IAMB

Fast-IAMB's paper claims fewer tests than IAMB at the same accuracy. It ran 42% to 73% of IAMB's
tests per target, and it was less accurate than IAMB and Inter-IAMB in every alarm, child,
insurance, and Gaussian cell: its mean F1 was lower than IAMB's by 0.054 to 0.106 on the networks
and by 0.010 to 0.052 on the Gaussian graphs. On hepar2 it tied IAMB at every size. These
comparisons are context: only the rows marked "worse" were rerun.

| Data | Rows | Tests per target: Fast-IAMB | IAMB | Inter-IAMB | F1: Fast-IAMB | IAMB | Inter-IAMB | D F1 against IAMB [95%] | Against IAMB | Against Inter-IAMB |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| alarm | 1,000 | 95 | 164 | 168 | 0.675 | 0.740 | 0.740 | -0.065 [-0.072, -0.057] | worse | worse |
| alarm | 5,000 | 116 | 187 | 195 | 0.715 | 0.779 | 0.776 | -0.064 [-0.079, -0.049] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| alarm | 20,000 | 125 | 212 | 230 | 0.722 | 0.825 | 0.793 | -0.104 [-0.119, -0.088] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| child | 1,000 | 44 | 70 | 73 | 0.764 | 0.831 | 0.830 | -0.066 [-0.076, -0.058] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| child | 5,000 | 54 | 81 | 88 | 0.791 | 0.853 | 0.852 | -0.062 [-0.070, -0.053] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| child | 20,000 | 60 | 91 | 102 | 0.822 | 0.876 | 0.868 | -0.054 [-0.070, -0.039] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| insurance | 1,000 | 56 | 119 | 125 | 0.587 | 0.689 | 0.688 | -0.102 [-0.118, -0.086] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| insurance | 5,000 | 71 | 143 | 154 | 0.648 | 0.753 | 0.754 | -0.106 [-0.116, -0.096] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| insurance | 20,000 | 86 | 164 | 188 | 0.722 | 0.816 | 0.816 | -0.095 [-0.110, -0.080] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| hepar2 | 1,000 | 256 | 426 | 441 | 0.380 | 0.383 | 0.387 | -0.003 [-0.013, +0.006] | tie | worse, seeds 0-19 only |
| hepar2 | 5,000 | 308 | 494 | 518 | 0.495 | 0.495 | 0.499 | +0.000 [-0.007, +0.008] | tie | tie |
| hepar2 | 20,000 | 359 | 552 | 588 | 0.549 | 0.541 | 0.542 | +0.007 [-0.001, +0.015] | tie | tie |
| Gaussian, d 20, degree 2 | 1,000 | 58 | 80 | 88 | 0.893 | 0.903 | 0.903 | -0.010 [-0.017, -0.004] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| Gaussian, d 20, degree 2 | 5,000 | 56 | 81 | 89 | 0.890 | 0.902 | 0.902 | -0.012 [-0.018, -0.007] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| Gaussian, d 50, degree 2 | 1,000 | 172 | 270 | 283 | 0.772 | 0.788 | 0.788 | -0.016 [-0.022, -0.011] | worse | worse |
| Gaussian, d 50, degree 2 | 5,000 | 165 | 274 | 287 | 0.767 | 0.793 | 0.793 | -0.026 [-0.032, -0.021] | worse, seeds 0-19 only | worse, seeds 0-19 only |
| Gaussian, d 100, degree 2 | 1,000 | 402 | 753 | 780 | 0.625 | 0.640 | 0.640 | -0.015 [-0.019, -0.012] | worse | worse |
| Gaussian, d 100, degree 2 | 5,000 | 379 | 748 | 774 | 0.631 | 0.648 | 0.648 | -0.018 [-0.021, -0.015] | worse | worse |
| Gaussian, d 50, degree 4 | 1,000 | 223 | 462 | 521 | 0.850 | 0.860 | 0.861 | -0.010 [-0.022, +0.004] | worse | worse |
| Gaussian, d 50, degree 4 | 5,000 | 204 | 484 | 550 | 0.850 | 0.902 | 0.903 | -0.052 [-0.062, -0.043] | worse | worse |

### GS beside IAMB

GS ran 32% to 64% of IAMB's tests per target. It tied IAMB in 9 network cells and 6 Gaussian cells,
was better in 2 Gaussian cells, and was worse on alarm at 1,000 rows (also on the rerun) and, on the
first seeds only, alarm at 5,000 rows and child at 1,000 rows.

| Data | Rows | Tests per target: GS | IAMB | F1: GS | IAMB | D distance [95%] | D F1 [95%] | Result |
|---|---:|---:|---:|---:|---:|---|---|---|
| alarm | 1,000 | 94 | 164 | 0.724 | 0.740 | +0.018 [+0.006, +0.030] | -0.015 [-0.025, -0.006] | worse |
| alarm | 5,000 | 96 | 187 | 0.766 | 0.779 | +0.014 [+0.003, +0.025] | -0.013 [-0.021, -0.004] | worse, seeds 0-19 only |
| alarm | 20,000 | 98 | 212 | 0.821 | 0.825 | +0.004 [-0.014, +0.022] | -0.005 [-0.018, +0.008] | tie |
| child | 1,000 | 45 | 70 | 0.825 | 0.831 | +0.009 [+0.000, +0.018] | -0.006 [-0.013, +0.001] | worse, seeds 0-19 only |
| child | 5,000 | 47 | 81 | 0.849 | 0.853 | +0.005 [-0.008, +0.020] | -0.004 [-0.015, +0.005] | tie |
| child | 20,000 | 49 | 91 | 0.887 | 0.876 | -0.013 [-0.030, +0.002] | +0.011 [-0.001, +0.023] | tie |
| insurance | 1,000 | 66 | 119 | 0.684 | 0.689 | +0.007 [-0.005, +0.018] | -0.005 [-0.013, +0.004] | tie |
| insurance | 5,000 | 70 | 143 | 0.748 | 0.753 | +0.006 [-0.007, +0.020] | -0.005 [-0.016, +0.004] | tie |
| insurance | 20,000 | 72 | 164 | 0.807 | 0.816 | +0.012 [-0.002, +0.026] | -0.009 [-0.020, +0.001] | tie |
| hepar2 | 1,000 | 206 | 426 | 0.384 | 0.383 | -0.001 [-0.009, +0.007] | +0.000 [-0.006, +0.006] | tie |
| hepar2 | 5,000 | 221 | 494 | 0.497 | 0.495 | -0.003 [-0.013, +0.007] | +0.003 [-0.005, +0.009] | tie |
| hepar2 | 20,000 | 234 | 552 | 0.546 | 0.541 | -0.008 [-0.018, +0.004] | +0.004 [-0.005, +0.013] | tie |
| Gaussian, d 20, degree 2 | 1,000 | 49 | 80 | 0.907 | 0.903 | -0.007 [-0.012, -0.003] | +0.004 [+0.001, +0.007] | better |
| Gaussian, d 20, degree 2 | 5,000 | 49 | 81 | 0.903 | 0.902 | -0.002 [-0.007, +0.002] | +0.001 [-0.002, +0.004] | tie |
| Gaussian, d 50, degree 2 | 1,000 | 134 | 270 | 0.788 | 0.788 | -0.001 [-0.007, +0.005] | +0.000 [-0.004, +0.005] | tie |
| Gaussian, d 50, degree 2 | 5,000 | 133 | 274 | 0.792 | 0.793 | +0.002 [-0.001, +0.005] | -0.002 [-0.004, +0.000] | tie |
| Gaussian, d 100, degree 2 | 1,000 | 295 | 753 | 0.641 | 0.640 | -0.001 [-0.007, +0.005] | +0.001 [-0.004, +0.005] | tie |
| Gaussian, d 100, degree 2 | 5,000 | 287 | 748 | 0.647 | 0.648 | +0.000 [-0.003, +0.004] | -0.001 [-0.004, +0.002] | tie |
| Gaussian, d 50, degree 4 | 1,000 | 162 | 462 | 0.875 | 0.860 | -0.022 [-0.037, -0.007] | +0.015 [+0.004, +0.026] | better |
| Gaussian, d 50, degree 4 | 5,000 | 156 | 484 | 0.895 | 0.902 | +0.010 [+0.000, +0.019] | -0.008 [-0.014, -0.001] | tie |

### HITON-PC at its default `max_depth=3`

The judged rows run `hiton_pc` without a limit on the conditioning set; its default is
`max_depth=3`. With the default it tied the unlimited search in 17 of 20 cells and was better in 3,
alarm, insurance, and hepar2 at 5,000 rows. Against `mmpc` it was better in 9 cells, tied in 9, and
was worse, on the first seeds only, on alarm at 500 rows and child at 200 rows. Its errors were
below the empty set's in every cell.

| Data | Rows | F1: HITON-PC, `max_depth=3` | `max_depth=None` | MMPC | Errors: HITON-PC, `max_depth=3` | Empty set | Against MMPC | Against `max_depth=None` | Fit, s: HITON-PC, `max_depth=3` | MMPC |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| alarm | 200 | 0.644 | 0.644 | 0.645 | 1.97 | 2.49 | tie | tie | 2.58 | 1.26 |
| alarm | 500 | 0.784 | 0.784 | 0.795 | 0.99 | 2.49 | worse, seeds 0-19 only | tie | 3.15 | 1.73 |
| alarm | 5,000 | 0.936 | 0.935 | 0.933 | 0.24 | 2.49 | tie | better | 5.73 | 3.97 |
| child | 200 | 0.656 | 0.656 | 0.704 | 2.12 | 2.50 | worse, seeds 0-19 only | tie | 1.38 | 0.93 |
| child | 500 | 0.853 | 0.853 | 0.839 | 0.69 | 2.50 | better | tie | 2.03 | 1.52 |
| child | 5,000 | 0.997 | 0.997 | 0.953 | 0.03 | 2.50 | better | tie | 4.54 | 4.37 |
| insurance | 200 | 0.570 | 0.570 | 0.537 | 3.20 | 3.72 | better | tie | 1.95 | 1.00 |
| insurance | 500 | 0.657 | 0.657 | 0.626 | 2.08 | 3.80 | better | tie | 3.00 | 1.84 |
| insurance | 5,000 | 0.814 | 0.810 | 0.803 | 1.03 | 3.85 | better | better | 7.70 | 7.32 |
| hepar2 | 200 | 0.255 | 0.255 | 0.238 | 3.35 | 3.51 | better | tie | 2.57 | 2.04 |
| hepar2 | 500 | 0.362 | 0.361 | 0.356 | 2.97 | 3.51 | better | tie | 3.97 | 3.02 |
| hepar2 | 5,000 | 0.681 | 0.622 | 0.620 | 1.75 | 3.51 | better | better | 16.4 | 18.3 |
| Gaussian, d 20, degree 2 | 1,000 | 0.937 | 0.937 | 0.936 | 0.30 | 2.23 | tie | tie | 0.43 | 0.37 |
| Gaussian, d 20, degree 2 | 5,000 | 0.955 | 0.955 | 0.959 | 0.20 | 2.23 | tie | tie | 0.52 | 0.48 |
| Gaussian, d 50, degree 2 | 1,000 | 0.926 | 0.926 | 0.925 | 0.34 | 2.30 | tie | tie | 0.78 | 0.65 |
| Gaussian, d 50, degree 2 | 5,000 | 0.946 | 0.945 | 0.946 | 0.25 | 2.30 | tie | tie | 1.39 | 1.21 |
| Gaussian, d 100, degree 2 | 1,000 | 0.895 | 0.895 | 0.894 | 0.47 | 2.31 | tie | tie | 2.00 | 1.58 |
| Gaussian, d 100, degree 2 | 5,000 | 0.914 | 0.914 | 0.913 | 0.38 | 2.31 | tie | tie | 4.29 | 3.78 |
| Gaussian, d 50, degree 4 | 1,000 | 0.709 | 0.708 | 0.699 | 2.04 | 4.06 | better | tie | 1.85 | 1.14 |
| Gaussian, d 50, degree 4 | 5,000 | 0.735 | 0.735 | 0.736 | 1.94 | 4.06 | tie | tie | 2.98 | 2.09 |

### HITON-PC's rows-per-cell rule

`hiton_pc` runs a test on categorical data only when the data hold at least five rows per cell of
its table, the rule of its paper. Without the rule, every test runs: that arm was worse in 9 of the
12 network cells, all four networks at 5,000 rows, child, insurance, and hepar2 at 500 rows, and
insurance and hepar2 at 200 rows. It tied on alarm at 200 and 500 rows and was better on child at
200 rows. These rows are from the first seeds only, and their D is the arm without the rule minus
`hiton_pc`.

| Data | Rows | Distance: with the rule | without | D [95%] | F1: with the rule | without | D [95%] | Tests per target: with | without | Without the rule |
|---|---:|---:|---:|---|---:|---:|---|---:|---:|---|
| alarm | 200 | 0.459 | 0.449 | -0.010 [-0.022, +0.002] | 0.644 | 0.641 | -0.003 [-0.014, +0.008] | 292 | 186 | tie |
| alarm | 500 | 0.285 | 0.278 | -0.007 [-0.016, +0.001] | 0.784 | 0.790 | +0.006 [-0.001, +0.012] | 322 | 271 | tie |
| alarm | 5,000 | 0.091 | 0.095 | +0.004 [+0.002, +0.007] | 0.935 | 0.932 | -0.003 [-0.005, -0.001] | 451 | 462 | worse, seeds 0-19 only |
| child | 200 | 0.444 | 0.378 | -0.066 [-0.090, -0.041] | 0.656 | 0.692 | +0.035 [+0.013, +0.058] | 179 | 146 | better |
| child | 500 | 0.204 | 0.226 | +0.021 [+0.008, +0.034] | 0.853 | 0.833 | -0.021 [-0.033, -0.009] | 243 | 256 | worse, seeds 0-19 only |
| child | 5,000 | 0.006 | 0.064 | +0.059 [+0.051, +0.065] | 0.997 | 0.944 | -0.052 [-0.058, -0.045] | 754 | 814 | worse, seeds 0-19 only |
| insurance | 200 | 0.591 | 0.586 | -0.005 [-0.025, +0.015] | 0.570 | 0.537 | -0.033 [-0.052, -0.016] | 281 | 178 | worse, seeds 0-19 only |
| insurance | 500 | 0.474 | 0.495 | +0.021 [+0.004, +0.035] | 0.657 | 0.623 | -0.034 [-0.046, -0.021] | 345 | 295 | worse, seeds 0-19 only |
| insurance | 5,000 | 0.273 | 0.291 | +0.018 [+0.010, +0.028] | 0.810 | 0.796 | -0.014 [-0.022, -0.008] | 1188 | 1372 | worse, seeds 0-19 only |
| hepar2 | 200 | 0.928 | 0.939 | +0.010 [+0.007, +0.013] | 0.255 | 0.243 | -0.011 [-0.014, -0.008] | 236 | 233 | worse, seeds 0-19 only |
| hepar2 | 500 | 0.798 | 0.805 | +0.007 [+0.004, +0.009] | 0.361 | 0.354 | -0.007 [-0.010, -0.005] | 320 | 316 | worse, seeds 0-19 only |
| hepar2 | 5,000 | 0.489 | 0.496 | +0.007 [+0.004, +0.010] | 0.622 | 0.616 | -0.006 [-0.009, -0.003] | 1726 | 1663 | worse, seeds 0-19 only |

### bnlearn's plain test

With plain `mi`, bnlearn's searches answer closer to Andrey's than with `mi-adf`. GS is better than
bnlearn's `gs` with `mi` in 9 of the 12 network cells, ties on hepar2 at 1,000 and 5,000 rows, and
is worse on hepar2 at 20,000 rows on the first seeds. Fast-IAMB is better in 5, ties in 6, and is
worse on alarm at 20,000 rows on the first seeds. HITON-PC is better in all 12. These rows are
context and were not rerun.

| Data | Rows | GS: F1, Andrey | bnlearn `mi-adf` | bnlearn `mi` | Errors: Andrey | bnlearn `mi` | Against `mi` | Fast-IAMB: F1, Andrey | bnlearn `mi-adf` | bnlearn `mi` | Errors: Andrey | bnlearn `mi` | Against `mi` | Errors: empty set |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| alarm | 1,000 | 0.724 | 0.324 | 0.509 | 2.08 | 3.00 | better | 0.675 | 0.694 | 0.650 | 2.20 | 2.27 | better | 3.51 |
| alarm | 5,000 | 0.766 | 0.364 | 0.590 | 1.81 | 2.84 | better | 0.715 | 0.719 | 0.722 | 2.18 | 2.06 | tie | 3.51 |
| alarm | 20,000 | 0.821 | 0.366 | 0.615 | 1.38 | 2.86 | better | 0.722 | 0.715 | 0.766 | 2.24 | 1.66 | worse, seeds 0-19 only | 3.51 |
| child | 1,000 | 0.825 | 0.408 | 0.641 | 1.17 | 2.26 | better | 0.764 | 0.712 | 0.736 | 1.52 | 1.61 | better | 3.00 |
| child | 5,000 | 0.849 | 0.346 | 0.690 | 0.93 | 2.18 | better | 0.791 | 0.711 | 0.789 | 1.42 | 1.40 | tie | 3.00 |
| child | 20,000 | 0.887 | 0.323 | 0.723 | 0.63 | 2.12 | better | 0.822 | 0.649 | 0.808 | 1.26 | 1.33 | tie | 3.00 |
| insurance | 1,000 | 0.684 | 0.249 | 0.425 | 2.78 | 4.16 | better | 0.587 | 0.573 | 0.527 | 3.37 | 3.57 | better | 5.12 |
| insurance | 5,000 | 0.748 | 0.254 | 0.479 | 2.30 | 4.11 | better | 0.648 | 0.631 | 0.643 | 3.26 | 3.07 | tie | 5.19 |
| insurance | 20,000 | 0.807 | 0.259 | 0.520 | 1.84 | 4.04 | better | 0.722 | 0.670 | 0.722 | 2.86 | 2.63 | tie | 5.19 |
| hepar2 | 1,000 | 0.384 | 0.194 | 0.378 | 5.61 | 4.78 | tie | 0.380 | 0.347 | 0.382 | 5.38 | 4.79 | tie | 4.52 |
| hepar2 | 5,000 | 0.497 | 0.199 | 0.502 | 5.07 | 4.28 | tie | 0.495 | 0.421 | 0.492 | 4.93 | 4.33 | better | 4.51 |
| hepar2 | 20,000 | 0.546 | 0.188 | 0.597 | 4.75 | 3.80 | worse, seeds 0-19 only | 0.549 | 0.405 | 0.532 | 4.70 | 4.17 | better | 4.51 |

| Data | Rows | HITON-PC: F1, Andrey | bnlearn `mi-adf` | bnlearn `mi` | Errors: Andrey | bnlearn `mi` | Against `mi` | Errors: empty set |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| alarm | 200 | 0.644 | 0.457 | 0.484 | 1.97 | 1.70 | better | 2.49 |
| alarm | 500 | 0.784 | 0.637 | 0.567 | 0.99 | 1.50 | better | 2.49 |
| alarm | 5,000 | 0.935 | 0.739 | 0.732 | 0.25 | 1.04 | better | 2.49 |
| child | 200 | 0.656 | 0.521 | 0.590 | 2.12 | 1.45 | better | 2.50 |
| child | 500 | 0.853 | 0.686 | 0.754 | 0.69 | 1.01 | better | 2.50 |
| child | 5,000 | 0.997 | 0.821 | 0.815 | 0.03 | 0.79 | better | 2.50 |
| insurance | 200 | 0.570 | 0.340 | 0.373 | 3.20 | 2.73 | better | 3.72 |
| insurance | 500 | 0.657 | 0.459 | 0.484 | 2.08 | 2.46 | better | 3.80 |
| insurance | 5,000 | 0.810 | 0.569 | 0.516 | 1.06 | 2.33 | better | 3.85 |
| hepar2 | 200 | 0.255 | 0.242 | 0.223 | 3.35 | 3.40 | better | 3.51 |
| hepar2 | 500 | 0.361 | 0.349 | 0.321 | 2.97 | 3.04 | better | 3.51 |
| hepar2 | 5,000 | 0.622 | 0.602 | 0.550 | 1.94 | 2.24 | better | 3.51 |

### The empty set on hepar2

On hepar2, GS and Fast-IAMB make more set errors than the empty set at every size, and so do
Andrey's supported `iamb` and `inter_iamb` and bnlearn's `gs` and `fast.iamb` with `mi-adf`. With
plain `mi`, bnlearn's searches are below the empty set at 5,000 and 20,000 rows. The errors fall
with the rows but stay above the empty set's at 20,000. The blanket learners test each candidate
given the whole current blanket, and a member added in error enlarges every later conditioning set.
HITON-PC, which conditions on subsets of its members, has fewer errors than the empty set at every
size, with or without a limit on the conditioning set, and so does `mmpc`.

Mean errors per target on hepar2 at the GS and Fast-IAMB sizes:

| Rows | Empty set | GS | Fast-IAMB | IAMB | Inter-IAMB | bnlearn `gs`, `mi-adf` | `mi` | bnlearn `fast.iamb`, `mi-adf` | `mi` |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 4.52 | 5.61 | 5.38 | 5.72 | 5.72 | 7.15 | 4.78 | 6.00 | 4.79 |
| 5,000 | 4.51 | 5.07 | 4.93 | 5.16 | 5.11 | 8.22 | 4.28 | 6.18 | 4.33 |
| 20,000 | 4.51 | 4.75 | 4.70 | 4.85 | 4.87 | 9.47 | 3.80 | 7.20 | 4.17 |

and at the HITON-PC sizes:

| Rows | Empty set | HITON-PC, `max_depth=None` | `max_depth=3` | MMPC | bnlearn `si.hiton.pc`, `mi-adf` | `mi` |
|---:|---:|---:|---:|---:|---:|---:|
| 200 | 3.51 | 3.35 | 3.35 | 3.41 | 3.49 | 3.40 |
| 500 | 3.51 | 2.97 | 2.97 | 3.01 | 3.08 | 3.04 |
| 5,000 | 3.51 | 1.94 | 1.75 | 1.96 | 2.05 | 2.24 |

## Run

Load R as [The R lane](../README.md#the-r-lane) describes, with bnlearn 5.2.1 in the library. Each
arm is one run of the driver:

```shell
B="benchmarks/blanket_cpu.py --drop-constant --warmup 0 --cores 1 --first-seed 0 --seeds 20"
MB=andrey.gs,andrey.fast_iamb,andrey.iamb,andrey.inter_iamb,bnlearn.gs,bnlearn.fast-iamb
PC=andrey.hiton_pc,andrey.mmpc,bnlearn.si-hiton-pc
NETS="alarm child insurance hepar2"
MB_NET=$(for x in $NETS; do for n in 1000 5000 20000; do echo -n "net:$x:$n "; done; done)
PC_NET=$(for x in $NETS; do for n in 200 500 5000; do echo -n "net:$x:$n "; done; done)
GAUSS="gauss:20:1000:2 gauss:20:5000:2 gauss:50:1000:2 gauss:50:5000:2 gauss:100:1000:2
  gauss:100:5000:2 gauss:50:1000:4 gauss:50:5000:4"
$ANDREY_BENCH_PYTHON $B --out out/mb-adf --cells $MB_NET $GAUSS --solutions $MB
$ANDREY_BENCH_PYTHON $B --out out/mb-plain --cells $MB_NET --discrete-test plain \
  --solutions bnlearn.gs,bnlearn.fast-iamb
$ANDREY_BENCH_PYTHON $B --out out/pc-adf --cells $PC_NET $GAUSS --solutions $PC
$ANDREY_BENCH_PYTHON $B --out out/pc-plain --cells $PC_NET --discrete-test plain \
  --solutions bnlearn.si-hiton-pc
$ANDREY_BENCH_PYTHON $B --out out/hiton3 --cells $PC_NET $GAUSS --max-depth 3 \
  --solutions andrey.hiton_pc
$ANDREY_BENCH_PYTHON benchmarks/blanket_summary.py 'mb-adf=out/mb-adf*' 'mb-plain=out/mb-plain*' \
  'pc-adf=out/pc-adf*' 'pc-plain=out/pc-plain*' 'hiton3=out/hiton3*' --json cells.json
```

The driver's defaults supply the rest: every target, `--indep-test gsq`, `--discrete-test adf`,
`--alpha 0.05`, no limit on the conditioning set, and `--cap-wall-s` of 1800. A continuous test has
no plain form, so the plain arms run the networks only. A rerun is the same command with
`--first-seed 20` on the cells to rerun, into a folder its arm's pattern matches, and
`blanket_summary.py` reads seeds 0 to 19 first and 20 to 39 as the rerun (`--first`, `--fresh`);
`--summary-rows` also writes the judged cells as release-summary rows. The arm without the
rows-per-cell rule runs `hiton_pc` like `pc-adf` from a copy of the build with `HITON_CELL_ROWS` set
to 0 in `andrey.constraint.blanket`, and the summarizer reads it as `'norule=out/norule*'`; it skips
an arm it has no records for. Times will differ with the hardware; the sets should match
closely.
