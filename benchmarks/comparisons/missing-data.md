# Missing data

`missing_cpu.py` measures the two corrections Andrey has for missing values: Missing-Value PC,
`pc(missing="mvpc")` (Tu et al., 2019), and structural EM, `missing="em"` on the searches that read
the data through a covariance (Friedman, 1997; Gao et al., 2022). `missing_summary.py` pairs the
records into the rows of
[`published/0.16.0/missing-data.json`](../published/0.16.0/missing-data.json) and, for the lane
`m2-tetrad`, of
[`published/0.17.0/missing-data-em.json`](../published/0.17.0/missing-data-em.json). From 0.17.0
`missing="em"` runs the alternating scheme of Friedman (1997). The fits of Andrey's EM arms in the
lanes with a reference ran again with it:
[`published/0.17.0/missing-data-em-bnlearn-missdag.json`](../published/0.17.0/missing-data-em-bnlearn-missdag.json)
holds the rows of `hc`, `tabu`, and `notears`, and the `m2-tetrad` record holds both schemes. The
tables of the lanes without a reference were measured with the earlier scheme, one refit per
round, and say so.

## Arms

Every cell masks a simulated sample and fits these arms on the same file:

- the method with `missing` set;
- test-wise PC, `pc` at its default, which deletes rows per test;
- the method after listwise deletion, which drops every incomplete row, and after mean imputation;
- the method on the complete data, the values before the mask;
- the empty graph, the floor;
- the reference, where the lane has one.

An arm that raises or runs past the cap of 1,800 s per fit is recorded as an error or a timeout and
is never scored as a graph. Listwise deletion that keeps fewer than 4 rows is an error.

`missing="em"` runs 10 rounds: an E-step under the linear-Gaussian model of the current graph, the
search on the expected covariance, and a least-squares fit of the graph the search returns. Before
each search after the first, 5 parameter-only EM steps fit the current graph's coefficients and
noise variances again, the alternating scheme of Friedman (1997), who leaves the number of steps
open. The earlier scheme, which 0.16.0 ran, fit them once per round. [The alternating scheme against
the earlier one](#the-alternating-scheme-against-the-earlier-one) compares the two.

## Lanes

| lane | methods | data | cells | seeds |
|---|---|---|---|---|
| `m1-paper` | `pc(missing="mvpc")`; the MVPC authors' R code | the authors' generator: linear-Gaussian random DAGs, 20 variables with 10 masked and 50 with 16; 100, 1,000, 5,000, 10,000 rows; MAR and MNAR masks, and for each the authors' MCAR control with the same number of missing values | 32 | 50 |
| `m1-generic` | `pc(missing="mvpc")` | Erdos-Renyi, mean degree 2 and 4, standardized; 10 to 100 variables; 500 to 10,000 rows; MCAR, logistic MAR, quantile MAR, quantile MNAR, logistic self-masking at 10, 20, 30% | 360 | 20 |
| `m2-a` | `notears(missing="em")`; MissDAG with its NOTEARS | ER1, ER2, SF2, weights from U(+-[0.5, 2]), unit noise variances, not standardized; 20 variables, 100 rows; MCAR at 10 to 50% | 15 | 20 |
| `m2-b` | `ges`, `boss`, `grasp` with `missing="em"`, at `ebic_gamma` 1.0 and 0.0 | as `m2-a`, 15 variables | 15 | 20 |
| `m2-c` | the `m2-a` and `m2-b` methods | the `m2-a` and `m2-b` cells with 30% of the columns always observed: logistic MAR and logistic self-masking at 10, 30, 50% | 18 each | 20 |
| `m2-generic` | `ges`, `boss`, `grasp` with `missing="em"` | the `m1-generic` graphs and masks at 100, 500, 2,000 rows | 360 | 20 |
| `m2-reduced` | `hc`, `tabu`, `xges`, `exact_search`, `calm`, `notears`, `dagma` with `missing="em"` | ER2, standardized, 20 variables, 500 rows; MCAR and logistic MAR at 20% | 2 | 20 |
| `m2-tetrad` | `ges`, `boss`, `grasp` with `missing="em"` at `ebic_gamma=0.0`; Tetrad's FGES, BOSS, GRaSP under its EM covariance policy, with the effective sample size at the number of rows and at the mean pairwise count, and under test-wise deletion | the `m2-b` cells and the 15-variable logistic MAR cells of `m2-c`; the MCAR and logistic MAR cells of `m2-generic` at 10, 20, 50 variables | 132 | 20 |
| `m2-bnlearn` | `hc` and `tabu` with `missing="em"` at `ebic_gamma=0.0`; bnlearn's `structural.em` | the `m2-b` cells and the 15-variable `m2-c` cells; ER2, standardized, 20 variables, 500 and 2,000 rows, MCAR and logistic MAR at 10, 20, 30% | 45 | 20 |

Every PC arm of `m1-generic` and `m1-paper` runs at `alpha` 0.05, Andrey's default, and 0.01, the
level of the MVPC authors' `demo.R`; the comparison with their code is at 0.01. `exact_search` runs
up to 20 variables.

## Reference implementations

| method | reference | version | setting |
|---|---|---|---|
| `pc(missing="mvpc")` | the MVPC authors' R code, `mvpc` with `gaussCItest.td` and `gaussCItest.permc`, the indicators' parents detected by its `detection_prt_m`; pcalg's `pc` with `gaussCItest.td` as the test-wise control | commit `e95160c0`, R 4.6.1, pcalg 2.7.12 | `alpha = 0.01`, R seed 0 |
| `notears(missing="em")` | MissDAG, `miss_dag_gaussian` with its NOTEARS; its NOTEARS on the complete data as the control | commit `3c1eeed8`, Python 3.10, NumPy 1.21.5 | `em_iter = 10`, `lambda_1_ev = 0.02`, `graph_thres = 0.3`, `equal_variances=True`, NumPy seed 0 |
| `hc(missing="em")`, `tabu(missing="em")` | bnlearn's `structural.em` with `maximize = "hc"` or `"tabu"`; `hc` and `tabu` on the complete data as the control | bnlearn 5.2.1 | `maximize.args = list(score = "bic-g")`, `fit = "mle-g"`, `impute = "exact"`, `max.iter = 10`, the rounds Andrey's EM runs |
| `ges(missing="em")`, `boss(missing="em")`, `grasp(missing="em")` | Tetrad's FGES, BOSS, and GRaSP with `SemBicScore` under `MissingDataPolicy.EM_COVARIANCE`; the same searches under `TESTWISE`, Tetrad's default, as a baseline, and on the complete data as the control | tetrad-lib 8.0.0, Java 21 | `MissingDataSpec.of(policy)`: ridge 0, tolerance 1e-6, at most 1,000 iterations, `FULL_N`; `penaltyDiscount = 1.0`; search settings as the `tetrad-continuous` lane; a second EM arm with `MEAN_PAIRWISE` |

The MVPC and MissDAG repositories carry no license, so `start_mvpc_r.sh` and `start_missdag.sh`
download each at its pinned commit, check the archive's SHA-256, and the benchmark runs the code
unmodified from a git-ignored folder. bnlearn is GPL (>= 2) and comes with the R lane
(`start_r.sh`). Tetrad is GPL (>= 2); `start_tetrad_missing.sh` downloads the 8.0.0 jar from Maven
Central and checks its SHA-256. The other Tetrad lanes run 7.6.10, which has no EM policy.

No package implements structural EM around GES, BOSS, GRaSP, XGES, or an exact search. MissDAG's
release holds NOTEARS M-steps only, although its paper reports GES and A*; bnlearn's
`structural.em` takes `hc` and `tabu`; causal-learn, pcalg, XGES, pgmpy, and pyAgrum search
structures on complete data only, and Tetrad's structural EM is for discrete Bayesian networks.

For `ges`, `boss`, and `grasp` the reference is the closest thing a standard package has: Tetrad
8.0.0's EM covariance policy. It is a one-pass EM estimate of a saturated covariance followed by the
search, not structural EM: EM fits the mean and covariance of an unrestricted Gaussian once, without
a graph, and the search then scores from that matrix. Tetrad's documentation calls it "the cheap
approximation to structural EM, in which imputation under the current best graph alternates with
re-search". It handles missing values by EM in the same searches with the same score, so the three
searches are judged against it, at two settings of the sample size its score counts: every row, its
default, and the mean pairwise count, which makes it a stronger arm. Both count for the status. For
`xges`, `exact_search`, `calm`, and `dagma` no reference exists, and the page reports the baselines
and the floor.

## Scores

A comparison is the per-dataset difference `D = SHD(method) - SHD(other)` on the datasets both
finished, with the 95% percentile-bootstrap interval of its mean over 10,000 resamples: worse when
the interval lies wholly above 0, better when wholly below, a tie otherwise. Fewer than 10 paired
datasets are reported and not judged. A worse cell counts only if it is worse again on 20 fresh
seeds. The floor holds in a cell when the interval of `SHD(method) - SHD(empty graph)` lies below 0.
It decides a judged cell at the papers' sizes, `m1-paper`, `m2-a`, `m2-b`, `m2-c`, the 100-row cells
of `m2-bnlearn`, and the cells of 15 variables and 100 rows of `m2-tetrad`, where the reference is
below the empty graph too; elsewhere it is reported. A mask outside a method's claim, self-masking
for both corrections, is reported and not judged. `m2-bnlearn` pairs DAGs with CPDAGs, so every arm
there is scored on the SHD between the equivalence classes of the estimate and of the truth.

The bar for a supported method: never less accurate than its reference in a cell that repeats, and
below the empty graph in its claimed setting.

## Results

Every number is on simulated data, one core per fit, on AMD EPYC 7713 and AMD EPYC 9555 processors.
The 0.16.0 record holds 192,700 fits; with the reruns on fresh seeds, the reruns of the fits that
raised, and the reference check on fresh seeds, 200,860 were run, in 232 CPU-hours of fit time. The
lane `m2-tetrad` holds 55,740 more. With its control, its reruns on fresh seeds, the fits that
raised and were fit again, and the 7,920 fits of Andrey's EM arms under each of the two schemes,
64,677 were run, on AMD EPYC 7713 processors, in 78.5 CPU-hours of fit time. The 2,460 EM fits of
`hc`, `tabu`, and `notears` ran again with the alternating scheme in 7.0 CPU-hours. In the table,
"with the earlier scheme" is the count the same comparison gave with one refit per round.

| method | lane | reference | better / tie / worse | with the earlier scheme | worse again | floor | median fit time, Andrey / reference | status |
|---|---|---|---|---|---|---|---|---|
| `pc(missing="mvpc")` | `m1-paper` | the MVPC authors' R code | 28 / 4 / 0 of 32 | | 0 | 32 of 32 | 0.02 to 1.24 | supported |
| `notears(missing="em")` | `m2-a`, `m2-c` | MissDAG | 23 / 1 / 0 of 24 | 19 / 5 / 0 | 0 | 15 of the 15 cells where MissDAG meets it; 24 of 24 | 0.16 to 0.43 | supported |
| `hc(missing="em")` | `m2-bnlearn` | bnlearn `structural.em` | 28 / 8 / 0 of 36 | 29 / 7 / 0 | 0 | 12 of 12 at 500 and 2,000 rows; 10 of 24 at 100 rows, where bnlearn meets it in none | 0.16 to 1.35 | supported |
| `tabu(missing="em")` | `m2-bnlearn` | bnlearn `structural.em` | 31 / 5 / 0 of 36 | 32 / 4 / 0 | 0 | 12 of 12 at 500 and 2,000 rows; 17 of 24 at 100 rows, where bnlearn meets it in one | 0.17 to 2.47 | supported |
| `ges(missing="em")` | `m2-tetrad` | Tetrad FGES, EM covariance | 111 / 21 / 0 of 132 | 107 / 25 / 0 | 0 | 2 of the 2 cells where Tetrad meets it; 73 of 132 | 0.15 to 5.58 | supported |
| `ges(missing="em")` | `m2-tetrad` | Tetrad FGES, EM covariance at `MEAN_PAIRWISE` | 90 / 42 / 0 of 132 | 81 / 50 / 1 | 0 of 0 | 4 of the 4 cells where Tetrad meets it; 73 of 132 | | supported |
| `boss(missing="em")` | `m2-tetrad` | Tetrad BOSS, EM covariance | 100 / 32 / 0 of 132 | 95 / 37 / 0 | 0 | 1 of the 1 cell where Tetrad meets it; 122 of 132 | 0.11 to 8.01 | supported |
| `boss(missing="em")` | `m2-tetrad` | Tetrad BOSS, EM covariance at `MEAN_PAIRWISE` | 72 / 59 / 1 of 132 | 70 / 58 / 4 | 0 of 1 | 3 of the 3 cells where Tetrad meets it; 122 of 132 | | supported |
| `grasp(missing="em")` | `m2-tetrad` | Tetrad GRaSP, EM covariance | 101 / 31 / 0 of 132 | 89 / 43 / 0 | 0 | 2 of the 2 cells where Tetrad meets it; 124 of 132 | 0.05 to 7.76 | supported |
| `grasp(missing="em")` | `m2-tetrad` | Tetrad GRaSP, EM covariance at `MEAN_PAIRWISE` | 68 / 61 / 3 of 132 | 64 / 62 / 6 | 0 of 3 | 4 of the 4 cells where Tetrad meets it; 124 of 132 | | supported |
| `xges`, `exact_search`, `calm`, `dagma` with `missing="em"` | `m2-reduced` | none exists | | | | see below | | experimental |

### MVPC against the authors' R code

`m1-paper`, `alpha` 0.01, 50 datasets per cell. SHD is the mean over the datasets both arms
finished; the difference is Andrey minus the R code, with its 95% interval; the control is the
same difference between the two packages' test-wise PC.

| variables | rows | mask | Andrey | R code | difference | result | control | test-wise PC | empty graph | Andrey, s | R code, s | Andrey fits failed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 100 | MAR | 18.12 | 18.02 | +0.10 [-0.14, +0.36] | tie | +0.12 [-0.12, +0.38] | 18.12 | 20.76 | 0.37 | 0.53 | 0 |
| 20 | 100 | MCAR control of MAR | 17.90 | 17.86 | +0.04 [-0.22, +0.30] | tie | +0.04 [-0.22, +0.30] | 17.90 | 20.76 | 0.33 | 0.51 | 0 |
| 20 | 100 | MCAR control of MNAR | 18.10 | 18.28 | -0.10 [-0.42, +0.21] | tie | -0.10 [-0.42, +0.21] | 18.10 | 20.76 | 0.61 | 0.48 | 2 |
| 20 | 100 | MNAR | 18.39 | 18.44 | -0.16 [-0.47, +0.12] | tie | -0.22 [-0.53, +0.08] | 18.31 | 20.76 | 0.63 | 0.50 | 1 |
| 20 | 1,000 | MAR | 7.98 | 10.58 | -2.60 [-3.20, -2.04] | better | -2.56 [-3.20, -1.96] | 8.48 | 20.76 | 0.73 | 0.95 | 0 |
| 20 | 1,000 | MCAR control of MAR | 6.54 | 9.32 | -2.78 [-3.40, -2.16] | better | -2.80 [-3.42, -2.18] | 6.54 | 20.76 | 0.63 | 0.77 | 0 |
| 20 | 1,000 | MCAR control of MNAR | 9.84 | 11.46 | -1.62 [-2.24, -1.02] | better | -1.62 [-2.24, -1.02] | 9.84 | 20.76 | 0.50 | 0.73 | 0 |
| 20 | 1,000 | MNAR | 10.32 | 11.68 | -1.36 [-2.06, -0.70] | better | -1.42 [-2.08, -0.80] | 10.48 | 20.76 | 0.58 | 0.84 | 0 |
| 20 | 5,000 | MAR | 4.18 | 6.48 | -2.30 [-2.86, -1.72] | better | -2.14 [-2.78, -1.52] | 6.98 | 20.76 | 0.74 | 3.18 | 0 |
| 20 | 5,000 | MCAR control of MAR | 2.76 | 5.86 | -3.10 [-3.84, -2.38] | better | -3.10 [-3.84, -2.38] | 2.74 | 20.76 | 0.54 | 2.14 | 0 |
| 20 | 5,000 | MCAR control of MNAR | 4.86 | 7.02 | -2.16 [-2.92, -1.44] | better | -2.14 [-2.90, -1.40] | 4.86 | 20.76 | 0.45 | 2.06 | 0 |
| 20 | 5,000 | MNAR | 5.56 | 7.82 | -2.26 [-2.94, -1.60] | better | -1.84 [-2.52, -1.12] | 7.98 | 20.76 | 0.70 | 2.90 | 0 |
| 20 | 10,000 | MAR | 3.64 | 5.48 | -1.84 [-2.50, -1.20] | better | -1.56 [-2.20, -0.92] | 8.16 | 20.76 | 0.88 | 5.71 | 0 |
| 20 | 10,000 | MCAR control of MAR | 2.68 | 4.42 | -1.74 [-2.38, -1.12] | better | -1.74 [-2.36, -1.14] | 2.68 | 20.76 | 0.61 | 4.05 | 0 |
| 20 | 10,000 | MCAR control of MNAR | 3.80 | 6.08 | -2.28 [-2.92, -1.70] | better | -2.28 [-2.92, -1.70] | 3.78 | 20.76 | 0.54 | 3.75 | 0 |
| 20 | 10,000 | MNAR | 5.28 | 7.28 | -2.00 [-2.76, -1.26] | better | -2.26 [-3.00, -1.54] | 8.58 | 20.76 | 0.88 | 5.32 | 0 |
| 50 | 100 | MAR | 40.66 | 41.36 | -0.70 [-1.24, -0.16] | better | -0.70 [-1.24, -0.16] | 40.66 | 50.10 | 0.46 | 1.17 | 0 |
| 50 | 100 | MCAR control of MAR | 40.44 | 41.26 | -0.82 [-1.32, -0.32] | better | -0.82 [-1.32, -0.32] | 40.44 | 50.10 | 0.43 | 1.08 | 0 |
| 50 | 100 | MCAR control of MNAR | 41.79 | 42.54 | -0.87 [-1.34, -0.43] | better | -0.87 [-1.34, -0.43] | 41.79 | 50.10 | 0.41 | 1.10 | 3 |
| 50 | 100 | MNAR | 42.00 | 43.18 | -1.16 [-1.65, -0.67] | better | -1.16 [-1.65, -0.69] | 42.00 | 50.10 | 0.43 | 1.16 | 1 |
| 50 | 1,000 | MAR | 15.72 | 22.10 | -6.38 [-7.44, -5.38] | better | -6.36 [-7.46, -5.32] | 16.30 | 50.10 | 0.69 | 7.34 | 0 |
| 50 | 1,000 | MCAR control of MAR | 14.24 | 21.12 | -6.88 [-7.96, -5.80] | better | -6.88 [-7.96, -5.80] | 14.22 | 50.10 | 0.53 | 6.56 | 0 |
| 50 | 1,000 | MCAR control of MNAR | 17.98 | 24.84 | -6.86 [-8.08, -5.66] | better | -6.88 [-8.10, -5.70] | 18.00 | 50.10 | 0.48 | 6.02 | 0 |
| 50 | 1,000 | MNAR | 20.46 | 25.86 | -5.40 [-6.42, -4.44] | better | -5.54 [-6.62, -4.54] | 20.52 | 50.10 | 0.65 | 6.69 | 0 |
| 50 | 5,000 | MAR | 6.44 | 13.12 | -6.68 [-7.74, -5.64] | better | -6.76 [-7.86, -5.64] | 11.60 | 50.10 | 1.43 | 40.73 | 0 |
| 50 | 5,000 | MCAR control of MAR | 6.16 | 12.32 | -6.16 [-7.54, -4.82] | better | -6.16 [-7.56, -4.80] | 6.12 | 50.10 | 0.92 | 34.83 | 0 |
| 50 | 5,000 | MCAR control of MNAR | 6.78 | 13.88 | -7.10 [-8.24, -5.96] | better | -7.12 [-8.26, -6.00] | 6.76 | 50.10 | 0.87 | 31.87 | 0 |
| 50 | 5,000 | MNAR | 9.34 | 15.92 | -6.58 [-7.94, -5.28] | better | -6.74 [-8.12, -5.34] | 13.76 | 50.10 | 1.29 | 36.10 | 0 |
| 50 | 10,000 | MAR | 5.14 | 10.82 | -5.68 [-6.84, -4.62] | better | -5.24 [-6.48, -4.06] | 14.28 | 50.10 | 2.48 | 85.50 | 0 |
| 50 | 10,000 | MCAR control of MAR | 3.68 | 9.50 | -5.82 [-7.20, -4.48] | better | -5.84 [-7.22, -4.48] | 3.68 | 50.10 | 1.45 | 70.72 | 0 |
| 50 | 10,000 | MCAR control of MNAR | 5.16 | 11.18 | -6.02 [-7.16, -4.94] | better | -6.04 [-7.18, -4.94] | 5.12 | 50.10 | 1.32 | 65.69 | 0 |
| 50 | 10,000 | MNAR | 7.36 | 12.98 | -5.62 [-6.92, -4.38] | better | -5.66 [-6.90, -4.40] | 14.92 | 50.10 | 2.22 | 78.47 | 0 |

MVPC is better in 28 cells and ties in the 4 cells of 20 variables and 100 rows; it is below the
empty graph in all 32, as the R code is. The control column shows the same margin between the two
packages' test-wise PC, so the margin is PC's orientation step and the correction layer ties: the
skeleton F1 of the two MVPC arms ties in 30 cells and differs by 0.002 and 0.004 in two MNAR cells
at 20 variables. The R code finished all 1,600 fits. Andrey's arm raised in 7, all at 100 rows,
where a test-wise test had fewer rows than it needs and test-wise PC raises too; they are the
`Andrey fits failed` column and no dataset of them is scored. Against the baselines in the same
cells:

| mask | cells | below the empty graph | errors / timeouts of fits | test-wise PC: better / tie / worse | worse again | listwise: better / tie / worse | worse again | mean imputation: better / tie / worse | worse again |
|---|---|---|---|---|---|---|---|---|---|
| MAR | 8 | 8 of 8 | 0 / 0 of 400 | 6 / 2 / 0 | 0 | 5 / 0 / 0 (3 not judged) | 0 | 6 / 2 / 0 | 0 |
| MCAR control of MAR | 8 | 8 of 8 | 0 / 0 of 400 | 0 / 8 / 0 | 0 | 3 / 0 / 0 (5 not judged) | 0 | 6 / 1 / 1 | not rerun |
| MCAR control of MNAR | 8 | 8 of 8 | 5 / 0 of 400 | 0 / 8 / 0 | 0 | 3 / 0 / 0 (5 not judged) | 0 | 6 / 2 / 0 | 0 |
| MNAR | 8 | 8 of 8 | 2 / 0 of 400 | 4 / 4 / 0 | 0 | 5 / 0 / 0 (3 not judged) | 0 | 6 / 2 / 0 | 0 |

The one cell worse than mean imputation is the MCAR control of MAR at 20 variables and 100 rows
(17.90 against 17.00); it was not rerun.

### MVPC on the generic grid

`m1-generic`, 72 cells and 1,440 fits per mask and level.

| alpha | mask | cells | below the empty graph | errors / timeouts of fits | test-wise PC: better / tie / worse | worse again | listwise: better / tie / worse | worse again | mean imputation: better / tie / worse | worse again |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | MAR, logistic | 72 | 72 of 72 | 0 / 20 of 1,440 | 0 / 65 / 7 | 2 of 7 rerun | 36 / 10 / 0 (26 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MAR, logistic | 72 | 71 of 72 | 0 / 57 of 1,440 | 4 / 60 / 5 (3 not judged) | 1 of 5 rerun | 28 / 17 / 0 (27 not judged) | 0 | 69 / 0 / 0 (3 not judged) | 0 |
| 0.01 | MAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 14 / 58 / 0 | 0 | 41 / 21 / 0 (10 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 28 / 44 / 0 | 0 | 46 / 16 / 0 (10 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.01 | MCAR | 72 | 72 of 72 | 0 / 0 of 1,440 | 1 / 69 / 2 | 0 of 2 rerun | 30 / 6 / 0 (36 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MCAR | 72 | 72 of 72 | 0 / 0 of 1,440 | 2 / 67 / 3 | 0 of 3 rerun | 30 / 6 / 0 (36 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.01 | MNAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 5 / 55 / 12 | 3 of 12 rerun | 35 / 19 / 0 (18 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MNAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 7 / 53 / 12 | 6 of 12 rerun | 37 / 17 / 0 (18 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.01 | self-masking | 72 | 72 of 72 | 0 / 0 of 1,440 | 1 / 65 / 6 | not rerun | 33 / 12 / 0 (27 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | self-masking | 72 | 72 of 72 | 0 / 0 of 1,440 | 2 / 61 / 9 | not rerun | 28 / 17 / 0 (27 not judged) | 0 | 72 / 0 / 0 | 0 |

- **Fits.** No fit raised. 77 of 14,400 ran past the cap of 1,800 s, all under logistic MAR at
  mean degree 2, 100 variables, and 10,000 rows: the mask gives each indicator up to 30 parents,
  and the search for them enumerates their subsets. Test-wise PC took 3.3 s (median) on those
  datasets. `max_depth` bounds the search.
- **Too few rows.** 962 of the 14,323 fits that returned answered at least one corrected test with
  the test-wise test, because fewer rows than the test needs observed its pair, its conditioning
  set, and the indicators' parents (a median of 27 parents). Nearly all are under logistic MAR,
  MNAR, and self-masking, where most columns go missing.
- **Against test-wise PC.** Under quantile MAR, the paper's mask, MVPC is better in 14 cells at
  `alpha` 0.01 and 28 at 0.05 and worse in none. Under MCAR it ties, as it should. Under logistic
  MAR, where every indicator has many weak parents, the correction does not help. Under quantile
  MNAR it is better in 12 rows and worse in 24 on seeds 0 to 19.
- **Floor.** Met in every cell but one: logistic MAR, 100 variables, 10,000 rows, 30% missing, at
  `alpha` 0.05, where most fits ran past the cap.

On seeds 0 to 19, 41 rows of the claimed masks are worse than test-wise PC: 24 under quantile MNAR,
12 under logistic MAR, and 5 under MCAR. All 41 ran again on seeds 20 to 39, and 12 are worse
again: 9 under quantile MNAR, 3 under logistic MAR, and none under MCAR. The 12 are at 50 or 100
variables with 20 or 30% missing, and the difference on the fresh seeds is 0.7 to 3.6 SHD:

| mask | mean degree | variables | rows | missing | `alpha` | MVPC | test-wise PC | difference, seeds 0 to 19 | difference, seeds 20 to 39 |
|---|---|---|---|---|---|---|---|---|---|
| MAR, logistic | 2 | 100 | 10,000 | 30% | 0.01 | 13.46 | 11.75 | +3.09 [+1.09, +6.00] | +3.09 [+1.82, +4.91] |
| MAR, logistic | 4 | 100 | 2,000 | 30% | 0.05 | 128.50 | 126.70 | +1.80 [+0.70, +3.00] | +3.50 [+1.75, +5.50] |
| MAR, logistic | 4 | 100 | 10,000 | 30% | 0.01 | 119.35 | 116.70 | +2.65 [+1.25, +3.95] | +3.50 [+1.60, +5.30] |
| MNAR, quantile | 2 | 50 | 500 | 30% | 0.05 | 19.30 | 18.30 | +1.00 [+0.55, +1.50] | +1.05 [+0.45, +1.65] |
| MNAR, quantile | 2 | 50 | 2,000 | 30% | 0.01 | 11.95 | 11.35 | +0.60 [+0.15, +1.10] | +0.95 [+0.30, +1.60] |
| MNAR, quantile | 2 | 100 | 2,000 | 20% | 0.05 | 20.80 | 19.40 | +1.40 [+0.65, +2.25] | +0.90 [+0.10, +1.90] |
| MNAR, quantile | 2 | 100 | 2,000 | 30% | 0.01 | 22.30 | 21.75 | +0.55 [+0.10, +1.25] | +0.70 [+0.10, +1.35] |
| MNAR, quantile | 2 | 100 | 2,000 | 30% | 0.05 | 23.00 | 22.25 | +0.75 [+0.25, +1.25] | +0.95 [+0.25, +1.80] |
| MNAR, quantile | 4 | 50 | 500 | 30% | 0.05 | 77.65 | 76.80 | +0.85 [+0.15, +1.60] | +0.75 [+0.20, +1.30] |
| MNAR, quantile | 4 | 100 | 2,000 | 20% | 0.05 | 123.60 | 121.50 | +2.10 [+0.55, +3.85] | +3.55 [+2.05, +5.15] |
| MNAR, quantile | 4 | 100 | 2,000 | 30% | 0.01 | 129.05 | 127.80 | +1.25 [+0.25, +2.30] | +1.25 [+0.35, +2.25] |
| MNAR, quantile | 4 | 100 | 10,000 | 20% | 0.05 | 117.35 | 115.05 | +2.30 [+0.65, +4.00] | +3.20 [+0.80, +5.70] |

The two SHD columns are the means on seeds 0 to 19. The first row is paired on the 11 datasets
that finished within the cap, on both sets of seeds, and the others on all 20. Of the other 29
rows, 28 tie on the fresh seeds and 1 is better.

In three of the 12 rows, all under quantile MNAR, the MVPC authors' R code ran on the same 20
datasets of seeds 20 to 39, its MVPC and its test-wise PC at the row's `alpha`:

| mean degree | variables | rows | missing | `alpha` | R code, MVPC | R code, test-wise PC | Andrey, MVPC | Andrey, test-wise PC | R code: MVPC minus test-wise PC | Andrey: MVPC minus test-wise PC | MVPC: Andrey minus R code | test-wise PC: Andrey minus R code |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 50 | 500 | 30% | 0.05 | 21.50 | 22.15 | 20.15 | 19.10 | +0.17 [-0.22, +0.61] tie (18 pairs) | +1.05 [+0.45, +1.65] worse | -1.56 [-3.17, +0.06] tie (18 pairs) | -3.05 [-4.75, -1.50] better |
| 2 | 50 | 2,000 | 30% | 0.01 | 15.75 | 15.50 | 11.60 | 10.65 | +0.25 [-0.40, +0.95] tie | +0.95 [+0.30, +1.60] worse | -4.15 [-5.75, -2.65] better | -4.85 [-6.45, -3.35] better |
| 4 | 100 | 2,000 | 20% | 0.05 | 136.37 | 135.15 | 124.10 | 120.55 | +0.42 [-1.32, +2.11] tie (19 pairs) | +3.55 [+2.05, +5.15] worse | -11.95 [-15.32, -8.53] better (19 pairs) | -14.60 [-17.65, -11.65] better |

SHD is each arm's mean over the datasets it finished, and a difference is paired on the datasets
both arms finished, 20 unless the cell says otherwise: the R code's MVPC raised on 2 datasets of
the first row and 1 of the third. In these settings the correction does not help in either
implementation. The R code's MVPC ties its own test-wise PC. Andrey's MVPC is behind Andrey's
test-wise PC by 0.95 to 3.55 SHD and is still no less accurate than the R code's MVPC, a tie in
one row and fewer errors in two, because Andrey's test-wise PC starts 3.05 to 14.6 SHD ahead of
the R code's. So in these settings test-wise deletion is the better choice in Andrey. The trace
below says why the correction costs Andrey more than it costs the R code. The median MVPC fit took
1.9, 2.2, and 52 s in Andrey against 4.3, 10.9, and 419 s in the R code.

#### Why the correction removes true edges

A trace of Andrey's fits on the datasets of these three rows, and of the R code's on those of the
first two, compares every test the correction answers with the true graph. It reruns the fits and is
not a new campaign: its mean SHD matches the table above in every row. The number of columns in `W`,
defined below, and the rows each test read were recorded for the first two rows only. MVPC minus
test-wise PC splits into extra edges, missing edges, and orientation errors (mean per dataset, 95%
bootstrap interval), and the correction's removals into true and extra edges:

| setting | extra edges | missing edges | orientation | true edges removed | extra edges removed |
|---|---|---|---|---|---|
| degree 2, 50 variables, 500 rows, `alpha` 0.05 | -0.15 [-0.30, +0.00] | +1.05 [+0.60, +1.55] | +0.15 [-0.15, +0.45] | 21 | 3 |
| degree 2, 50 variables, 2,000 rows, `alpha` 0.01 | -0.20 [-0.45, +0.00] | +1.05 [+0.65, +1.55] | +0.10 [-0.30, +0.50] | 21 | 4 |
| degree 4, 100 variables, 2,000 rows, `alpha` 0.05 | -1.55 [-2.20, -0.90] | +4.50 [+3.15, +5.90] | +0.60 [-0.30, +1.55] | 90 | 31 |

The whole loss is true edges that the correction removes: 21 of its 24 changes in the first setting,
21 of 25 in the second, and 90 of 121 in the third removed a true edge, and the orientation column
is within noise of 0 in all three. Two things cause the removals, and both are the paper's method.

**A corrected test reads few rows.** Under this mask every column that can go missing has an
indicator with a parent that can go missing too, so `W`, the parents of the indicators of the tested
pair and its conditioning set, closed under the parents of their own indicators (the paper's
condition (i)), holds a median of 8 columns in the first setting and 7 in the second, and up to 15.
A corrected test reads the rows where the pair, its conditioning set, and `W` are all observed: a
median of 13% and 19% of the rows the test-wise test reads. The corrected test that removed a true
edge read a median of 22 and 41 rows where the test-wise test read about 245 and 965, and 19 of 21
and 18 of 21 of those edges had a test-wise p-value of at most 1e-4. A corrected test on a true edge
separated it in 38% and 55% of the tests with under 30 rows, 11% and 12% with 30 to 100, and 8% and
5% above 100. If a test below half the test-wise rows keeps its test-wise answer, the correction
removes 1 true and 1 extra edge in the first setting, 0 and 1 in the second, and 0 and 0 in the
third, where it removed 21 and 3, 21 and 4, and 90 and 31, so nearly every removal, right or wrong,
comes from a test that read under half the rows. The paper has no such floor, and Andrey's is the
smaller `max(|S| + 3, |W| + 2)` rows.

**Step 3 re-tests pairs that falsely detected parents make candidates.** The paper's Step 3 marks an
adjacent pair for the correction when it shares a common adjacent variable or a missingness
indicator. Andrey applies both. The R code applies the first only, so on the same detected parents
(identical in 38 of 38 datasets) and the same test-wise skeleton it corrected 26 pairs where Andrey
corrected 68 in the first setting, and 44 where Andrey corrected 57 in the second. In the first two
settings 44 of 79 and 13 of 57 re-tested pairs are candidates by a shared indicator alone. All 57
are true edges, and in none are both columns true parents of that indicator: each pair exists
because the detection named a neighbor of the true parent as a second parent (precision 0.878 and
0.964, recall 0.99 and 1.00). The correction removed 16 and 8 of them. In the third setting the
detection has precision 0.803 and recall 0.985, and 99 of the 331 re-tested pairs are candidates by
a shared indicator alone: 94 are true edges, and the correction removed 28 of those. With the R
code's rule Andrey's MVPC minus test-wise PC falls from +1.05, +0.95, and +3.55 to +0.10 [-0.20,
+0.45], +0.55 [+0.05, +1.10], and +1.90 [+0.35, +3.50]; with the true parents in place of the
detected ones it falls to +0.05 [-0.20, +0.35] and +0.50 [-0.10, +1.15] in the first two. What
remains is the rows above.

The R code still removes true edges, 12 in 18 datasets of the first setting and 16 in 20 of the
second, against Andrey's 21 and 21 on the same datasets; in the first, 7 of its 12 come from blocks
of at most `|W| + 1` rows (a saturated or rank-deficient regression, or a p-value of 1), which
Andrey answers with the test-wise test. Its SHD also hides part of the loss: its test-wise PC starts
with 3 to 5 more orientation errors than Andrey's, and its orientation errors fall by 0.2 after the
correction where Andrey's rise by 0.1 to 0.15.

Each of the following was switched in Andrey on the first two settings and is not the cause (MVPC
minus test-wise PC, against +1.05 and +0.95):

- **The intercept.** Without it, as the paper's Eq. (2) has it: +0.90 [+0.40, +1.45] and +0.90
  [+0.30, +1.55].
- **One direction per pair.** Two independent shuffles per test, the edge going when either
  separates it, as the R code's two orders do: +1.10 [+0.50, +1.75] and +0.95 [+0.30, +1.60].
- **The seed.** Seeds 1 to 4 give +0.85, +1.15, +1.25, and +1.05, and +0.85, +0.90, +0.75, and
  +0.90.
- **Orientation.** The colliders over a removed pair read only the corrected separating set, and
  ranking the removed pairs last among the colliders changes no graph.
- **The fallback to the test-wise test.** It protects the fit: answering a test with too few rows
  with a p-value of 1, as the R code does, gives +2.45 [+1.55, +3.40] and +1.40 [+0.70, +2.15], with
  56 and 29 true edges removed.

The paper's experiments with values missing not at random use "MNAR: MAR + MCAR": the parents of the
chosen indicators go missing completely at random, so `W` closes after one level. It names the
general case, "MNAR: no self-masking", and does not run it (Tu et al., 2019, Section 5.1 of the
extended version, arXiv:1807.04010). Quantile MNAR here is that general case: 70% of the columns can
go missing, and the indicator of each depends on another of them.

The other six rows of quantile MNAR that repeat, and the three rows of logistic MAR, were not
traced. A logistic mask gives each indicator many parents, so `W` is large there too, but that is
inferred, not measured. In these settings use `missing="testwise"`.

### NOTEARS with EM against MissDAG

20 variables, 100 rows, 20 datasets per cell. The control is the difference between the two
NOTEARS on the complete data, and the time ratio is Andrey's fit time over MissDAG's.

| lane | graph | mask | rate | Andrey | MissDAG | difference | result | control | empty graph | below it | time ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `m2-a` | er, degree 1 | MCAR | 0.1 | 0.60 | 2.70 | -2.10 [-2.95, -1.25] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.27 |
| `m2-a` | er, degree 1 | MCAR | 0.2 | 1.05 | 4.65 | -3.60 [-4.70, -2.50] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.24 |
| `m2-a` | er, degree 1 | MCAR | 0.3 | 2.50 | 7.55 | -5.05 [-6.55, -3.65] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.25 |
| `m2-a` | er, degree 1 | MCAR | 0.4 | 4.65 | 10.70 | -6.05 [-7.85, -4.35] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.19 |
| `m2-a` | er, degree 1 | MCAR | 0.5 | 8.40 | 12.65 | -4.25 [-6.05, -2.40] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.16 |
| `m2-a` | er, degree 2 | MCAR | 0.1 | 1.80 | 3.90 | -2.10 [-2.95, -1.25] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.43 |
| `m2-a` | er, degree 2 | MCAR | 0.2 | 3.15 | 6.15 | -3.00 [-4.55, -1.60] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.41 |
| `m2-a` | er, degree 2 | MCAR | 0.3 | 6.50 | 11.55 | -5.05 [-8.45, -2.05] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.34 |
| `m2-a` | er, degree 2 | MCAR | 0.4 | 11.30 | 16.50 | -5.20 [-8.75, -2.00] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.41 |
| `m2-a` | er, degree 2 | MCAR | 0.5 | 18.05 | 22.35 | -4.30 [-8.60, -1.15] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.37 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.1 | 2.40 | 3.85 | -1.45 [-2.60, -0.45] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.40 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.2 | 3.35 | 5.50 | -2.15 [-3.85, -0.40] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.39 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.3 | 7.95 | 11.95 | -4.00 [-5.75, -2.20] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.41 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.4 | 13.00 | 17.05 | -4.05 [-6.95, -1.30] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.35 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.5 | 17.20 | 20.75 | -3.55 [-6.35, -0.90] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.32 |
| `m2-c` | er, degree 1 | MAR, logistic | 0.1 | 0.30 | 1.90 | -1.60 [-2.30, -0.90] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.26 |
| `m2-c` | er, degree 1 | MAR, logistic | 0.3 | 1.95 | 5.15 | -3.20 [-4.55, -1.80] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.22 |
| `m2-c` | er, degree 1 | MAR, logistic | 0.5 | 6.10 | 9.45 | -3.35 [-5.15, -1.60] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.18 |
| `m2-c` | er, degree 1 | self-masking | 0.1 | 0.40 | 1.75 | -1.35 [-1.90, -0.90] | not judged | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.20 |
| `m2-c` | er, degree 1 | self-masking | 0.3 | 2.55 | 5.45 | -2.90 [-4.35, -1.20] | not judged | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.21 |
| `m2-c` | er, degree 1 | self-masking | 0.5 | 6.20 | 10.30 | -4.10 [-6.20, -2.05] | not judged | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.19 |
| `m2-c` | er, degree 2 | MAR, logistic | 0.1 | 1.65 | 3.40 | -1.75 [-2.65, -0.90] | better | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.33 |
| `m2-c` | er, degree 2 | MAR, logistic | 0.3 | 5.50 | 8.05 | -2.55 [-5.10, -0.10] | better | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.38 |
| `m2-c` | er, degree 2 | MAR, logistic | 0.5 | 12.60 | 19.60 | -7.00 [-10.35, -3.80] | better | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.30 |
| `m2-c` | er, degree 2 | self-masking | 0.1 | 1.75 | 3.75 | -2.00 [-2.95, -1.10] | not judged | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.32 |
| `m2-c` | er, degree 2 | self-masking | 0.3 | 6.85 | 10.20 | -3.35 [-7.30, +0.00] | not judged | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.32 |
| `m2-c` | er, degree 2 | self-masking | 0.5 | 15.15 | 22.85 | -7.70 [-12.95, -3.05] | not judged | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.28 |
| `m2-c` | scale_free, degree 2 | MAR, logistic | 0.1 | 2.15 | 3.25 | -1.10 [-2.05, -0.20] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.34 |
| `m2-c` | scale_free, degree 2 | MAR, logistic | 0.3 | 9.00 | 10.20 | -1.20 [-4.80, +2.50] | tie | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.31 |
| `m2-c` | scale_free, degree 2 | MAR, logistic | 0.5 | 14.55 | 18.15 | -3.60 [-6.40, -0.75] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.37 |
| `m2-c` | scale_free, degree 2 | self-masking | 0.1 | 2.80 | 3.45 | -0.65 [-2.15, +0.85] | not judged | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.36 |
| `m2-c` | scale_free, degree 2 | self-masking | 0.3 | 9.35 | 10.80 | -1.45 [-5.10, +2.05] | not judged | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.35 |
| `m2-c` | scale_free, degree 2 | self-masking | 0.5 | 15.80 | 19.95 | -4.15 [-7.50, -1.00] | not judged | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.28 |

`notears(missing="em")` is better in 23 of the 24 cells of its claim and ties in 1 (19 and 5 with
the earlier scheme). Part of the margin is not EM: on the complete data Andrey's NOTEARS is ahead by
0.85 to 1.8 SHD. It is below the empty graph in every cell, at 50% missing on ER2 and the scale-free
graph too (18.05 against 20 and 17.2 against 19), where it was not with the earlier scheme and
MissDAG is not. The median fit took 8 to 41 s against MissDAG's 49 to 114 s.

### Hill climbing and Tabu search with EM against bnlearn

`m2-bnlearn`, 20 datasets per cell, SHD between equivalence classes, both packages on the plain
BIC.

| method | size | mask | cells | below the empty graph | errors / timeouts of fits | bnlearn `structural.em` at `max.iter = 10`: better / tie / worse | worse again | test-wise PC: better / tie / worse | worse again |
|---|---|---|---|---|---|---|---|---|---|
| `hc`, `ebic_gamma 0` | 15 variables, 100 rows | MAR, logistic | 9 | 4 of 9 | 0 / 0 of 180 | 7 / 2 / 0 | 0 | 0 / 1 / 8 | not rerun |
| `tabu`, `ebic_gamma 0` | 15 variables, 100 rows | MAR, logistic | 9 | 6 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 0 / 5 / 4 | not rerun |
| `hc`, `ebic_gamma 0` | 15 variables, 100 rows | MCAR | 15 | 6 of 15 | 0 / 0 of 300 | 13 / 2 / 0 | 0 | 0 / 5 / 10 | not rerun |
| `tabu`, `ebic_gamma 0` | 15 variables, 100 rows | MCAR | 15 | 11 of 15 | 0 / 0 of 300 | 13 / 2 / 0 | 0 | 0 / 10 / 5 | not rerun |
| `hc`, `ebic_gamma 0` | 15 variables, 100 rows | self-masking | 9 | 4 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 0 / 2 / 7 | not rerun |
| `tabu`, `ebic_gamma 0` | 15 variables, 100 rows | self-masking | 9 | 6 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 0 / 3 / 6 | not rerun |
| `hc`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MAR, logistic | 6 | 6 of 6 | 0 / 0 of 120 | 4 / 2 / 0 | 0 | 0 / 0 / 6 | not rerun |
| `tabu`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MAR, logistic | 6 | 6 of 6 | 0 / 0 of 120 | 5 / 1 / 0 | 0 | 0 / 5 / 1 | not rerun |
| `hc`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MCAR | 6 | 6 of 6 | 0 / 0 of 120 | 4 / 2 / 0 | 0 | 0 / 0 / 6 | not rerun |
| `tabu`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MCAR | 6 | 6 of 6 | 0 / 0 of 120 | 5 / 1 / 0 | 0 | 0 / 6 / 0 | not rerun |

Neither search is worse than bnlearn's `structural.em` in any cell: `hc` is better in 28 and ties in
8 of the 36 cells of its claim, `tabu` is better in 31 and ties in 5. With the earlier scheme the
counts were 29 and 7, and 32 and 4: in each search one cell moved from better to a tie. The
complete-data control separates the search from its EM layer:

| search | graph | variables | rows | Andrey | bnlearn | difference | verdict | empty graph |
|---|---|---|---|---|---|---|---|---|
| `hc` | er, degree 1 | 15 | 100 | 8.80 | 8.50 | +0.30 [-0.75, +1.45] | tie | 8.00 |
| `hc` | er, degree 2 | 15 | 100 | 13.30 | 10.90 | +2.40 [+0.20, +4.65] | worse; tie on fresh datasets | 15.00 |
| `hc` | er, degree 2 | 20 | 2,000 | 11.85 | 12.70 | -0.85 [-4.35, +2.65] | tie | 20.00 |
| `hc` | er, degree 2 | 20 | 500 | 13.75 | 13.05 | +0.70 [-2.80, +4.15] | tie | 20.00 |
| `hc` | scale_free, degree 2 | 15 | 100 | 13.05 | 13.10 | -0.05 [-1.75, +1.70] | tie | 14.00 |
| `tabu` | er, degree 1 | 15 | 100 | 8.45 | 8.30 | +0.15 [-0.85, +1.05] | tie | 8.00 |
| `tabu` | er, degree 2 | 15 | 100 | 12.95 | 11.30 | +1.65 [+0.10, +3.20] | worse; tie on fresh datasets | 15.00 |
| `tabu` | er, degree 2 | 20 | 2,000 | 7.30 | 7.15 | +0.15 [-2.85, +3.55] | tie | 20.00 |
| `tabu` | er, degree 2 | 20 | 500 | 10.80 | 9.50 | +1.30 [-1.95, +4.90] | tie | 20.00 |
| `tabu` | scale_free, degree 2 | 15 | 100 | 12.75 | 12.40 | +0.35 [-1.35, +2.05] | tie | 14.00 |

Without missing values the two packages' searches tie in four of the five settings. In the fifth,
ER2 with 15 variables and 100 rows, Andrey's `hc` and `tabu` are behind bnlearn's on these 20
datasets. That setting ran again on 20 fresh datasets, seeds 100 to 119, and both searches tie, so
the difference does not repeat. The rerun adds mean degree 4 at 20 variables with 500 and 2,000
rows, a density the lane does not hold: all four rows tie, with both packages near or above the
empty graph.

| search | graph | variables | rows | Andrey | bnlearn | difference | verdict | empty graph |
|---|---|---|---|---|---|---|---|---|
| `hc` | er, degree 2 | 15 | 100 | 12.85 | 13.40 | -0.55 [-2.40, +1.15] | tie | 15.00 |
| `tabu` | er, degree 2 | 15 | 100 | 12.45 | 13.10 | -0.65 [-2.40, +1.05] | tie | 15.00 |
| `hc` | er, degree 4 | 20 | 500 | 41.45 | 44.35 | -2.90 [-10.15, +3.95] | tie | 40.00 |
| `tabu` | er, degree 4 | 20 | 500 | 36.90 | 43.15 | -6.25 [-13.45, +1.25] | tie | 40.00 |
| `hc` | er, degree 4 | 20 | 2,000 | 43.85 | 46.65 | -2.80 [-13.20, +7.75] | tie | 40.00 |
| `tabu` | er, degree 4 | 20 | 2,000 | 35.80 | 39.25 | -3.45 [-15.20, +8.25] | tie | 40.00 |

The plain searches tie, so the margin with EM is the EM layer: `structural.em` fills each missing
value with its conditional mean, and its error grows with the missing rate (under MCAR on ER2,
from 12.4 to 30.1 SHD between 10 and 50%, where Andrey's `hc` goes from 12.05 to 14.75).

At 15 variables and 100 rows the plain BIC overfits: the complete-data fits are not clearly below
the empty graph (on ER1, 8.8 for `hc` and 8.45 for `tabu` against 8.0), and with EM `hc` is below it
in 10 of 24 cells and `tabu` in 17, where bnlearn is in 0 and 1. At 500 and 2,000 rows both are
below it in all 12 cells. Both are worse than test-wise PC in many cells at this penalty (`hc` in 30
of 36, `tabu` in 10), as their complete-data fits are. The median fit took 1.0 to 2.4 s against
bnlearn's 0.8 to 2.3 s at 100 rows, 2.2 to 5.0 s against 6 to 14 s at 500, and 4.1 to 17 s against
26 to 53 s at 2,000.

### GES, BOSS, and GRaSP with EM against Tetrad's EM covariance

`m2-tetrad`, 20 datasets per cell, 132 cells per search. Both packages score with the plain BIC:
Andrey at `ebic_gamma=0.0` and `penalty_discount=1.0`, Tetrad at `penaltyDiscount = 1.0`, and both
count every row as the sample size. Andrey's default is `ebic_gamma=1.0`, which has no matching
objective in Tetrad and is not compared here; the lanes below run it against the baselines. A
second Tetrad arm, which counts fewer rows, follows the results of this one.

| method | size | mask | cells | below the empty graph | Tetrad's EM arm below it | errors / timeouts of fits | Tetrad, EM covariance: better / tie / worse | worse again | Tetrad, test-wise: better / tie / worse | test-wise PC: better / tie / worse | listwise: better / tie / worse | mean imputation: better / tie / worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ges` | 15 variables, 100 rows | MAR, logistic | 9 | 7 of 9 | 1 of 9 | 0 / 0 of 180 | 9 / 0 / 0 | 0 | 9 / 0 / 0 | 1 / 7 / 1 | 3 / 0 / 0 (6 not judged) | 8 / 1 / 0 |
| `ges` | 15 variables, 100 rows | MCAR | 15 | 13 of 15 | 1 of 15 | 0 / 0 of 300 | 15 / 0 / 0 | 0 | 15 / 0 / 0 | 2 / 12 / 1 | 3 / 0 / 0 (12 not judged) | 15 / 0 / 0 |
| `ges` | 10 to 50 variables, 100 rows | MAR, logistic | 18 | 6 of 18 | 2 of 18 | 0 / 0 of 360 | 17 / 1 / 0 | 0 | 18 / 0 / 0 | 0 / 8 / 10 | 7 / 3 / 0 (8 not judged) | 16 / 2 / 0 |
| `ges` | 10 to 50 variables, 100 rows | MCAR | 18 | 9 of 18 | 1 of 18 | 0 / 0 of 360 | 16 / 2 / 0 | 0 | 18 / 0 / 0 | 1 / 12 / 5 | 5 / 1 / 0 (12 not judged) | 17 / 1 / 0 |
| `ges` | 10 to 50 variables, 500 rows | MAR, logistic | 18 | 10 of 18 | 5 of 18 | 0 / 0 of 360 | 16 / 2 / 0 | 0 | 18 / 0 / 0 | 0 / 11 / 7 | 11 / 3 / 0 (4 not judged) | 18 / 0 / 0 |
| `ges` | 10 to 50 variables, 500 rows | MCAR | 18 | 10 of 18 | 4 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 3 / 11 / 4 | 7 / 3 / 0 (8 not judged) | 18 / 0 / 0 |
| `ges` | 10 to 50 variables, 2,000 rows | MAR, logistic | 18 | 9 of 18 | 8 of 18 | 0 / 0 of 360 | 9 / 9 / 0 | 0 | 18 / 0 / 0 | 2 / 13 / 3 | 10 / 4 / 0 (4 not judged) | 18 / 0 / 0 |
| `ges` | 10 to 50 variables, 2,000 rows | MCAR | 18 | 9 of 18 | 6 of 18 | 0 / 0 of 360 | 14 / 4 / 0 | 0 | 18 / 0 / 0 | 0 / 14 / 4 | 8 / 4 / 0 (6 not judged) | 18 / 0 / 0 |
| `boss` | 15 variables, 100 rows | MAR, logistic | 9 | 7 of 9 | 1 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 9 / 0 / 0 | 0 / 6 / 3 | 3 / 0 / 0 (6 not judged) | 9 / 0 / 0 |
| `boss` | 15 variables, 100 rows | MCAR | 15 | 13 of 15 | 0 of 15 | 0 / 0 of 300 | 15 / 0 / 0 | 0 | 15 / 0 / 0 | 2 / 11 / 2 | 3 / 0 / 0 (12 not judged) | 15 / 0 / 0 |
| `boss` | 10 to 50 variables, 100 rows | MAR, logistic | 18 | 15 of 18 | 4 of 18 | 0 / 0 of 360 | 17 / 1 / 0 | 0 | 18 / 0 / 0 | 9 / 5 / 4 | 10 / 0 / 0 (8 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 100 rows | MCAR | 18 | 15 of 18 | 1 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 8 / 8 / 2 | 5 / 1 / 0 (12 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 500 rows | MAR, logistic | 18 | 18 of 18 | 14 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 18 / 0 / 0 | 9 / 8 / 1 | 11 / 3 / 0 (4 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 500 rows | MCAR | 18 | 18 of 18 | 11 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 18 / 0 / 0 | 12 / 6 / 0 | 9 / 1 / 0 (8 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 2,000 rows | MAR, logistic | 18 | 18 of 18 | 17 of 18 | 0 / 0 of 360 | 9 / 9 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 12 / 6 / 0 | 12 / 2 / 0 (4 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 2,000 rows | MCAR | 18 | 18 of 18 | 14 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 10 / 8 / 0 | 10 / 2 / 0 (6 not judged) | 18 / 0 / 0 |
| `grasp` | 15 variables, 100 rows | MAR, logistic | 9 | 8 of 9 | 1 of 9 | 0 / 0 of 180 | 9 / 0 / 0 | 0 | 9 / 0 / 0 | 0 / 8 / 1 | 3 / 0 / 0 (6 not judged) | 9 / 0 / 0 |
| `grasp` | 15 variables, 100 rows | MCAR | 15 | 13 of 15 | 1 of 15 | 0 / 0 of 300 | 15 / 0 / 0 | 0 | 15 / 0 / 0 | 4 / 9 / 2 | 3 / 0 / 0 (12 not judged) | 15 / 0 / 0 |
| `grasp` | 10 to 50 variables, 100 rows | MAR, logistic | 18 | 15 of 18 | 4 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 7 / 8 / 3 | 8 / 2 / 0 (8 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 100 rows | MCAR | 18 | 16 of 18 | 3 of 18 | 0 / 0 of 360 | 16 / 2 / 0 | 0 | 18 / 0 / 0 | 6 / 10 / 2 | 5 / 1 / 0 (12 not judged) | 17 / 1 / 0 |
| `grasp` | 10 to 50 variables, 500 rows | MAR, logistic | 18 | 18 of 18 | 13 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 18 / 0 / 0 | 14 / 4 / 0 | 12 / 2 / 0 (4 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 500 rows | MCAR | 18 | 18 of 18 | 10 of 18 | 0 / 0 of 360 | 13 / 5 / 0 | 0 | 18 / 0 / 0 | 16 / 2 / 0 | 8 / 2 / 0 (8 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 2,000 rows | MAR, logistic | 18 | 18 of 18 | 17 of 18 | 0 / 0 of 360 | 8 / 10 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 13 / 5 / 0 | 11 / 3 / 0 (4 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 2,000 rows | MCAR | 18 | 18 of 18 | 14 of 18 | 0 / 0 of 360 | 13 / 5 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 14 / 4 / 0 | 10 / 2 / 0 (6 not judged) | 18 / 0 / 0 |

No search is worse than Tetrad's EM arm in any cell, so no cell ran again on fresh seeds. `ges` is
better in 111 cells and ties in 21, `boss` is better in 100 and ties in 32, and `grasp` is better in
101 and ties in 31 (107 and 25, 95 and 37, and 89 and 43 with the earlier scheme). No EM fit of
either package raised or ran past the cap, in 7,920 fits each. Thirteen of Andrey's EM fits failed
on a read of the Python environment's files and were fit again; the record holds the second fits.

Much of the margin is the reference's weakness, not a strength of Andrey's. Tetrad's EM arm is below
the empty graph in 28 (FGES), 62 (BOSS), and 63 (GRaSP) of the 132 cells, most of them at 500 and
2,000 rows. With 100 rows and 15 or more variables, or with 50 variables and 500 rows, its mean SHD
is far above the empty graph's: the search returns dense graphs. The likely cause is the saturated
covariance, which is estimated without a graph and scored as if every row were complete (see Known
differences). In the cells where Tetrad's EM arm is itself below the empty graph, `ges` is better in
22 and ties in 6, `boss` is better in 32 and ties in 30, and `grasp` is better in 35 and ties in 28.
The mean SHD over the cells of each setting (two masks by three rates; the eight cells of each graph
at 15 variables):

| graph | variables | rows | cells | empty graph | `ges`: Andrey / Tetrad | `boss`: Andrey / Tetrad | `grasp`: Andrey / Tetrad |
|---|---|---|---|---|---|---|---|
| er, degree 1 | 15 | 100 | 8 | 8.00 | 5.23 / 35.69 | 4.99 / 37.27 | 4.70 / 36.62 |
| er, degree 2 | 15 | 100 | 8 | 15.00 | 10.56 / 37.16 | 9.42 / 39.44 | 9.33 / 38.90 |
| scale_free, degree 2 | 15 | 100 | 8 | 14.00 | 8.94 / 36.61 | 10.22 / 40.71 | 9.82 / 40.36 |
| er, degree 2 | 10 | 100 | 6 | 10.00 | 5.28 / 8.63 | 4.77 / 8.83 | 5.16 / 8.04 |
| er, degree 2 | 10 | 500 | 6 | 10.00 | 2.69 / 5.32 | 2.89 / 4.46 | 1.86 / 4.23 |
| er, degree 2 | 10 | 2,000 | 6 | 10.00 | 1.74 / 3.00 | 1.97 / 2.79 | 1.53 / 2.54 |
| er, degree 2 | 20 | 100 | 6 | 20.00 | 13.49 / 43.87 | 12.09 / 48.83 | 11.92 / 48.68 |
| er, degree 2 | 20 | 500 | 6 | 20.00 | 6.43 / 16.11 | 5.93 / 15.99 | 4.90 / 14.41 |
| er, degree 2 | 20 | 2,000 | 6 | 20.00 | 4.99 / 11.91 | 3.85 / 9.94 | 2.65 / 9.28 |
| er, degree 2 | 50 | 100 | 6 | 50.00 | 49.95 / 281.38 | 48.54 / 463.00 | 45.73 / 429.78 |
| er, degree 2 | 50 | 500 | 6 | 50.00 | 23.72 / 99.16 | 17.03 / 114.68 | 15.22 / 109.98 |
| er, degree 2 | 50 | 2,000 | 6 | 50.00 | 12.36 / 49.00 | 9.90 / 48.43 | 7.73 / 44.57 |
| er, degree 4 | 10 | 100 | 6 | 20.00 | 18.55 / 20.77 | 15.85 / 18.80 | 16.88 / 18.37 |
| er, degree 4 | 10 | 500 | 6 | 20.00 | 17.78 / 18.88 | 9.77 / 10.22 | 10.95 / 12.00 |
| er, degree 4 | 10 | 2,000 | 6 | 20.00 | 16.93 / 17.07 | 6.58 / 6.84 | 7.18 / 9.98 |
| er, degree 4 | 20 | 100 | 6 | 40.00 | 40.86 / 61.93 | 24.58 / 61.05 | 28.75 / 60.45 |
| er, degree 4 | 20 | 500 | 6 | 40.00 | 41.27 / 50.82 | 10.35 / 17.12 | 10.25 / 21.93 |
| er, degree 4 | 20 | 2,000 | 6 | 40.00 | 38.05 / 45.52 | 6.79 / 10.24 | 7.50 / 10.03 |
| er, degree 4 | 50 | 100 | 6 | 100.00 | 104.83 / 307.57 | 62.06 / 471.53 | 67.77 / 431.48 |
| er, degree 4 | 50 | 500 | 6 | 100.00 | 96.12 / 178.81 | 20.97 / 127.16 | 23.87 / 134.12 |
| er, degree 4 | 50 | 2,000 | 6 | 100.00 | 100.37 / 144.71 | 11.55 / 41.48 | 10.88 / 40.73 |

The floor decides a cell of 15 variables and 100 rows only where Tetrad's EM arm is below the empty
graph: 2 cells for `ges`, 1 for `boss`, and 2 for `grasp`, and each search is below it in all of
them. Over all 24 cells of that size `ges` and `boss` are below the empty graph in 20 and `grasp` in
21; the misses are at 50% missing, but for one of `ges` at 10% under logistic MAR on ER1 (6.75
against 8.00, the interval reaching 0). On the generic cells `boss` is below the empty graph in 102
of 108 and `grasp` in 103, every miss at 100 rows. `ges` is below it in 50 of the 54 cells of mean
degree 2 and in 3 of the 54 of mean degree 4, where the plain search at this penalty is near or
above the empty graph on the complete data too (the control below).

The complete-data control separates the plain searches from the missing-data layers: both
packages' searches on the data before the mask, the same 20 datasets per setting.

| graph | variables | rows | empty graph | `ges`: Andrey / Tetrad FGES, difference | `boss`: Andrey / Tetrad BOSS, difference | `grasp`: Andrey / Tetrad GRaSP, difference |
|---|---|---|---|---|---|---|
| er, degree 1 | 15 | 100 | 8.00 | 7.75 / 7.50, +0.25 [+0.00, +0.70] tie | 7.50 / 7.65, -0.15 [-0.70, +0.35] tie | 6.40 / 6.70, -0.30 [-1.10, +0.50] tie |
| er, degree 2 | 15 | 100 | 15.00 | 11.05 / 10.55, +0.50 [+0.00, +1.20] tie | 7.85 / 8.80, -0.95 [-3.55, +0.80] tie | 6.15 / 7.00, -0.85 [-2.40, +0.65] tie |
| scale_free, degree 2 | 15 | 100 | 14.00 | 8.75 / 8.75, +0.00 [-0.25, +0.20] tie | 9.50 / 8.50, +1.00 [-1.10, +3.55] tie | 9.65 / 7.75, +1.90 [-0.30, +4.10] tie |
| er, degree 2 | 10 | 100 | 10.00 | 5.50 / 4.10, +1.40 [+0.40, +2.50] **worse**; tie on fresh datasets | 5.00 / 5.50, -0.50 [-1.90, +0.80] tie | 4.85 / 4.90, -0.05 [-2.00, +1.90] tie |
| er, degree 2 | 10 | 500 | 10.00 | 2.75 / 2.90, -0.15 [-1.05, +0.60] tie | 3.10 / 1.85, +1.25 [-0.25, +3.00] tie | 1.85 / 2.35, -0.50 [-1.85, +0.70] tie |
| er, degree 2 | 10 | 2,000 | 10.00 | 1.20 / 1.05, +0.15 [-0.15, +0.60] tie | 2.00 / 1.25, +0.75 [-0.40, +2.00] tie | 2.05 / 1.25, +0.80 [+0.00, +1.95] tie |
| er, degree 2 | 20 | 100 | 20.00 | 15.25 / 14.55, +0.70 [-0.30, +1.75] tie | 14.65 / 14.40, +0.25 [-2.35, +2.75] tie | 13.65 / 13.25, +0.40 [-2.45, +3.35] tie |
| er, degree 2 | 20 | 500 | 20.00 | 8.55 / 8.45, +0.10 [-0.50, +0.65] tie | 6.45 / 7.35, -0.90 [-3.50, +1.70] tie | 5.45 / 6.40, -0.95 [-3.10, +1.25] tie |
| er, degree 2 | 20 | 2,000 | 20.00 | 4.10 / 4.35, -0.25 [-1.65, +1.00] tie | 5.20 / 4.50, +0.70 [-0.85, +2.35] tie | 4.10 / 3.45, +0.65 [-1.70, +3.15] tie |
| er, degree 2 | 50 | 100 | 50.00 | 66.40 / 63.85, +2.55 [+0.55, +4.55] **worse**; tie on fresh datasets | 71.20 / 65.10, +6.10 [+1.50, +10.60] **worse**; tie on fresh datasets | 64.15 / 63.70, +0.45 [-2.60, +3.60] tie |
| er, degree 2 | 50 | 500 | 50.00 | 36.85 / 34.65, +2.20 [-0.45, +5.05] tie | 29.85 / 28.65, +1.20 [-0.45, +2.85] tie | 23.80 / 24.05, -0.25 [-2.95, +2.45] tie |
| er, degree 2 | 50 | 2,000 | 50.00 | 17.05 / 17.70, -0.65 [-2.90, +1.10] tie | 12.65 / 15.10, -2.45 [-4.70, -0.35] **better** | 11.35 / 9.95, +1.40 [-1.00, +3.40] tie |
| er, degree 4 | 10 | 100 | 20.00 | 18.95 / 19.70, -0.75 [-1.55, +0.00] tie | 11.10 / 15.45, -4.35 [-8.45, -0.55] **better** | 13.40 / 11.90, +1.50 [-3.60, +6.35] tie |
| er, degree 4 | 10 | 500 | 20.00 | 16.75 / 16.40, +0.35 [-0.75, +1.40] tie | 7.35 / 8.25, -0.90 [-4.65, +3.00] tie | 7.50 / 7.00, +0.50 [-4.00, +4.80] tie |
| er, degree 4 | 10 | 2,000 | 20.00 | 16.50 / 15.90, +0.60 [-0.80, +2.05] tie | 5.25 / 4.95, +0.30 [-3.90, +4.55] tie | 1.60 / 7.35, -5.75 [-9.65, -2.25] **better** |
| er, degree 4 | 20 | 100 | 40.00 | 40.25 / 40.80, -0.55 [-3.25, +1.80] tie | 20.55 / 17.45, +3.10 [-4.30, +10.20] tie | 19.75 / 21.50, -1.75 [-8.55, +4.35] tie |
| er, degree 4 | 20 | 500 | 40.00 | 41.85 / 46.75, -4.90 [-9.85, -1.15] **better** | 9.50 / 8.35, +1.15 [-7.05, +7.35] tie | 8.15 / 7.85, +0.30 [-5.55, +6.00] tie |
| er, degree 4 | 20 | 2,000 | 40.00 | 35.05 / 41.55, -6.50 [-13.10, -0.90] **better** | 9.70 / 5.90, +3.80 [-1.90, +10.60] tie | 4.95 / 4.25, +0.70 [-2.15, +4.20] tie |
| er, degree 4 | 50 | 100 | 100.00 | 108.45 / 109.85, -1.40 [-6.40, +5.45] tie | 61.65 / 60.35, +1.30 [-5.80, +8.45] tie | 75.20 / 78.05, -2.85 [-22.00, +15.70] tie |
| er, degree 4 | 50 | 500 | 100.00 | 99.35 / 108.05, -8.70 [-16.75, -1.15] **better** | 26.60 / 26.45, +0.15 [-11.50, +12.10] tie | 31.20 / 20.50, +10.70 [+1.10, +21.55] **worse**; worse on fresh datasets |
| er, degree 4 | 50 | 2,000 | 100.00 | 102.00 / 106.85, -4.85 [-19.50, +9.25] tie | 8.35 / 14.15, -5.80 [-15.70, +0.60] tie | 14.85 / 8.25, +6.60 [-0.60, +17.60] tie |

The plain searches tie in 53 of the 63 comparisons. Andrey's is behind in 4: `ges` at mean degree
2 with 100 rows (10 and 50 variables), `boss` at mean degree 2 with 50 variables and 100 rows, and
`grasp` at mean degree 4 with 50 variables and 500 rows. It is ahead in 6: `ges` at mean degree 4
(20 variables with 500 and 2,000 rows, 50 with 500), `boss` at mean degree 2 with 50 variables and
2,000 rows and at mean degree 4 with 10 variables and 100 rows, and `grasp` at mean degree 4 with
10 variables and 2,000 rows. In those 6 settings a margin with EM may be the search and not the
EM layer. The 4 settings where it is behind ran again on 20 fresh datasets each, seeds 20 to 39 (60
datasets: `ges` and `boss` share the 50-variable ones): Andrey's plain search, Tetrad's, and the
empty graph, 220 fits, the complete-data units of this lane at `--first-seed 20`. The difference is
Andrey minus Tetrad.

| search | graph | variables | rows | seeds 0 to 19: Andrey / Tetrad, difference | seeds 20 to 39: Andrey / Tetrad, difference | empty graph | repeats |
|---|---|---|---|---|---|---|---|
| `ges` | er, degree 2 | 10 | 100 | 5.50 / 4.10, +1.40 [+0.40, +2.50] worse | 3.15 / 3.90, -0.75 [-2.30, +0.10] tie | 10.00 | no |
| `ges` | er, degree 2 | 50 | 100 | 66.40 / 63.85, +2.55 [+0.55, +4.55] worse | 65.65 / 65.70, -0.05 [-2.30, +2.30] tie | 50.00 | no |
| `boss` | er, degree 2 | 50 | 100 | 71.20 / 65.10, +6.10 [+1.50, +10.60] worse | 67.80 / 69.70, -1.90 [-5.20, +1.30] tie | 50.00 | no |
| `grasp` | er, degree 4 | 50 | 500 | 31.20 / 20.50, +10.70 [+1.10, +21.55] worse | 27.90 / 16.90, +11.00 [+2.25, +21.40] worse | 100.00 | yes |

In three of the four settings the difference does not repeat: the packages tie on the fresh
datasets (`ges` with 10 and with 50 variables, `boss` with 50 variables). In `grasp` at mean degree
4 with 50 variables and 500 rows it does: Andrey's plain search is behind Tetrad's by 10.7 SHD on
the first datasets and by 11.0 on the fresh ones, both intervals above 0, with SHD 31.2 and 27.9
against 20.5 and 16.9 and the empty graph at 100. With EM in that setting `grasp` is better than
Tetrad's EM arm in 5 of its 6 cells and ties in 1, because Tetrad's EM arm has a mean SHD of 32 to
398 there: the margin with EM is the reference's missing-data layer and not a better search. The 6
settings where Andrey is ahead were not rerun; with 63 comparisons at 95%, about 3 would differ by
chance. Elsewhere the searches tie, so the margin is the missing-data layer.

Against the baselines:

- **Tetrad's test-wise deletion**, its default. Every search with EM is better in every judged cell.
  At 50 variables and 2,000 rows, 170 of Tetrad's test-wise BOSS and GRaSP fits were killed for
  memory (a fit that finished there used up to 12 GB, and the benchmark ran 16 fits side by side in
  48 GB) and 6 GRaSP fits ran past the cap, so 4 cells of each of the two are not judged and the
  others pair fewer datasets. Tetrad's test-wise FGES finished every fit.
- **Test-wise PC.** At this penalty `ges` with EM is worse than `pc` in 35 cells and better in 9,
  `boss` is worse in 12 and better in 62, and `grasp` is worse in 8 and better in 74. The worse
  cells of `boss` and `grasp` are at 100 rows, but for one of `boss` at 500; those of `ges` are at
  every size, 20 of them at mean degree 4. They were not rerun.
- **Listwise deletion and mean imputation.** No search is worse than either in any cell. Listwise
  deletion kept fewer than 4 rows in 3,534 of 7,920 fits, which are errors; PC raised in 5 fits at
  15 variables and 100 rows, where a test had too few rows.

The median EM fit took 1.3 to 26 s for `ges`, 1.5 to 44 s for `boss`, and 1.2 to 40 s for `grasp`,
against 0.7 to 45 s, 0.7 to 48 s, and 0.7 to 70 s for Tetrad's, which include about 1.3 s to start
the Java process. By the median ratio of the paired fit times, Andrey's fit was the slower one in 82
(`ges`), 109 (`boss`), and 98 (`grasp`) of the 132 cells, by up to 5.6, 8.0, and 7.8 times, the
largest at 20 variables with 2,000 rows, and the faster one in the others, by up to 6.7, 8.9, and 18
times. With the earlier scheme it was the slower one in 17, 64, and 58 cells: the alternating scheme
runs 56 E-steps where the earlier one ran 11.

### Tetrad's EM arm with the effective sample size at the mean pairwise count

The first arm counts every row as the sample size, which Tetrad's documentation calls
"anti-conservative when much data is missing". Tetrad 8.0.0 has two other settings of that count,
introduced this way: "With missing entries, the nominal number of rows overstates the information
actually available; these modes offer conservative alternatives." `MEAN_PAIRWISE` is "the mean, over
variable pairs, of the number of rows on which both variables are observed", and `MIN_PAIRWISE` is
the minimum of those counts. Tetrad's BIC takes the count in place of the number of rows, in the
likelihood and in the penalty's `log n`. The lane fits `tetrad.<search>.em_covariance.mean_pairwise`
on the same 2,640 masked datasets as the first arm, with the same EM estimate, search, and plain
BIC; `essMode` is the only difference. Andrey's EM arm is compared with it by the same paired
bootstrap, the cells where Andrey is worse ran again on fresh seeds, and it counts for the status
as the first arm does.

Andrey's EM arm is better than it in 90 (`ges`), 72 (`boss`), and 68 (`grasp`) of the 132 cells,
ties in 42, 59, and 61, and is worse in 0, 1, and 3; with the earlier scheme it was better in 81,
70, and 64 and worse in 1, 4, and 6. All 2,640 fits of the arm per search finished.
It is a stronger arm than the first. Over the cells its mean SHD is 46.4 (FGES), 46.8 (BOSS), and
41.5 (GRaSP), against 68.7, 74.5, and 71.1 at `FULL_N` and 28.6, 14.0, and 14.2 for Andrey's EM arm,
with the empty graph at 35.0. It is below the empty graph in 41, 80, and 83 cells, against 28, 62,
and 63 at `FULL_N`; it still returns dense graphs at 100 rows, at 50 variables with a mean SHD of
154 to 300 against the empty graph's 50 at mean degree 2, and 192 to 295 against 100 at mean degree
4. At 15 variables and 100 rows, where the floor decides a cell, it is below the empty graph in 4,
3, and 4 cells, and Andrey's EM arm is below it in each of them.

| search | cells | Andrey below the empty graph | Tetrad at `FULL_N` below it | Tetrad at `mean_pairwise` below it | of them at 15 variables, 100 rows | Andrey vs `mean_pairwise`: better / tie / worse | with the earlier scheme | where `mean_pairwise` is below the empty graph: better / tie / worse | worse cells that are worse again |
|---|---|---|---|---|---|---|---|---|---|
| `ges` | 132 | 73 | 28 | 41 | 4 of 24 | 90 / 42 / 0 | 81 / 50 / 1 | 22 / 19 / 0 | 0 of 0 |
| `boss` | 132 | 122 | 62 | 80 | 3 of 24 | 72 / 59 / 1 | 70 / 58 / 4 | 25 / 54 / 1 | 0 of 1 |
| `grasp` | 132 | 124 | 63 | 83 | 4 of 24 | 68 / 61 / 3 | 64 / 62 / 6 | 22 / 58 / 3 | 0 of 3 |

The mean SHD over the cells of each setting (two masks by three rates; the eight cells of each graph
at 15 variables):

| graph | variables | rows | cells | empty graph | `ges`: Andrey / Tetrad `FULL_N` / Tetrad `mean_pairwise` | `boss`: Andrey / Tetrad `FULL_N` / Tetrad `mean_pairwise` | `grasp`: Andrey / Tetrad `FULL_N` / Tetrad `mean_pairwise` |
|---|---|---|---|---|---|---|---|
| er, degree 1 | 15 | 100 | 8 | 8.00 | 5.23 / 35.69 / 25.48 | 4.99 / 37.27 / 29.03 | 4.70 / 36.62 / 27.01 |
| er, degree 2 | 15 | 100 | 8 | 15.00 | 10.56 / 37.16 / 27.46 | 9.42 / 39.44 / 32.59 | 9.33 / 38.90 / 29.74 |
| scale_free, degree 2 | 15 | 100 | 8 | 14.00 | 8.94 / 36.61 / 26.29 | 10.22 / 40.71 / 32.89 | 9.82 / 40.36 / 31.61 |
| er, degree 2 | 10 | 100 | 6 | 10.00 | 5.28 / 8.63 / 6.65 | 4.77 / 8.83 / 6.95 | 5.16 / 8.04 / 6.12 |
| er, degree 2 | 10 | 500 | 6 | 10.00 | 2.69 / 5.32 / 3.67 | 2.89 / 4.46 / 2.85 | 1.86 / 4.23 / 2.36 |
| er, degree 2 | 10 | 2,000 | 6 | 10.00 | 1.74 / 3.00 / 1.98 | 1.97 / 2.79 / 1.72 | 1.53 / 2.54 / 1.23 |
| er, degree 2 | 20 | 100 | 6 | 20.00 | 13.49 / 43.87 / 32.68 | 12.09 / 48.83 / 40.01 | 11.92 / 48.68 / 35.99 |
| er, degree 2 | 20 | 500 | 6 | 20.00 | 6.43 / 16.11 / 9.82 | 5.93 / 15.99 / 9.23 | 4.90 / 14.41 / 6.88 |
| er, degree 2 | 20 | 2,000 | 6 | 20.00 | 4.99 / 11.91 / 6.67 | 3.85 / 9.94 / 5.12 | 2.65 / 9.28 / 4.11 |
| er, degree 2 | 50 | 100 | 6 | 50.00 | 49.95 / 281.38 / 153.95 | 48.54 / 463.00 / 299.98 | 45.73 / 429.78 / 261.17 |
| er, degree 2 | 50 | 500 | 6 | 50.00 | 23.72 / 99.16 / 51.57 | 17.03 / 114.68 / 53.84 | 15.22 / 109.98 / 45.25 |
| er, degree 2 | 50 | 2,000 | 6 | 50.00 | 12.36 / 49.00 / 20.57 | 9.90 / 48.43 / 18.01 | 7.73 / 44.57 / 13.87 |
| er, degree 4 | 10 | 100 | 6 | 20.00 | 18.55 / 20.77 / 19.87 | 15.85 / 18.80 / 17.07 | 16.88 / 18.37 / 17.43 |
| er, degree 4 | 10 | 500 | 6 | 20.00 | 17.78 / 18.88 / 18.32 | 9.77 / 10.22 / 8.09 | 10.95 / 12.00 / 9.87 |
| er, degree 4 | 10 | 2,000 | 6 | 20.00 | 16.93 / 17.07 / 16.71 | 6.58 / 6.84 / 5.74 | 7.18 / 9.98 / 8.27 |
| er, degree 4 | 20 | 100 | 6 | 40.00 | 40.86 / 61.93 / 53.19 | 24.58 / 61.05 / 49.88 | 28.75 / 60.45 / 50.40 |
| er, degree 4 | 20 | 500 | 6 | 40.00 | 41.27 / 50.82 / 45.62 | 10.35 / 17.12 / 9.28 | 10.25 / 21.93 / 12.96 |
| er, degree 4 | 20 | 2,000 | 6 | 40.00 | 38.05 / 45.52 / 42.21 | 6.79 / 10.24 / 6.91 | 7.50 / 10.03 / 5.85 |
| er, degree 4 | 50 | 100 | 6 | 100.00 | 104.83 / 307.57 / 192.28 | 62.06 / 471.53 / 294.98 | 67.77 / 431.48 / 243.70 |
| er, degree 4 | 50 | 500 | 6 | 100.00 | 96.12 / 178.81 / 125.76 | 20.97 / 127.16 / 57.61 | 23.87 / 134.12 / 56.69 |
| er, degree 4 | 50 | 2,000 | 6 | 100.00 | 100.37 / 144.71 / 113.17 | 11.55 / 41.48 / 16.12 | 10.88 / 40.73 / 13.50 |

The 4 cells where Andrey's EM arm is worse (1 of `boss`, 3 of `grasp`) ran again on 20 fresh
datasets each, seeds 20 to 39: Andrey's EM arm, Tetrad's arm, and the empty graph. Tetrad's fits on
the fresh datasets of three of the cells are the ones recorded with the earlier scheme's reruns, so
180 fits ran. The difference is Andrey minus Tetrad.

| search | graph | variables | rows | mask | rate | seeds 0 to 19: Andrey / Tetrad, difference | seeds 20 to 39: Andrey / Tetrad, difference | worse again |
|---|---|---|---|---|---|---|---|---|
| `grasp` | er, degree 2 | 10 | 2,000 | MAR, logistic | 0.3 | 2.20 / 1.15, +1.05 [+0.05, +2.35] worse | 1.15 / 1.65, -0.50 [-2.05, +1.05] tie | no |
| `boss` | er, degree 4 | 10 | 500 | MCAR | 0.3 | 13.60 / 8.25, +5.35 [+0.45, +10.40] worse | 13.30 / 11.65, +1.65 [-3.25, +7.10] tie | no |
| `grasp` | er, degree 4 | 20 | 2,000 | MAR, logistic | 0.2 | 10.70 / 3.20, +7.50 [+2.05, +13.60] worse | 6.60 / 11.25, -4.65 [-11.30, +1.15] tie | no |
| `grasp` | er, degree 4 | 50 | 500 | MAR, logistic | 0.1 | 33.25 / 19.75, +13.50 [+1.85, +27.35] worse | 26.60 / 21.60, +5.00 [-5.45, +18.75] tie | no |

None is worse again: all four tie on the fresh datasets, so no cell of the lane is worse than a
Tetrad arm on both groups of seeds. With the earlier scheme 11 cells were worse than this arm (1 of
`ges`, 4 of `boss`, 6 of `grasp`), and two of them were worse again on fresh datasets, both at mean
degree 4 with 2,000 rows under MCAR at 30%: `boss` at 20 variables (a mean SHD of 17.85 against
8.60, then 20.10 against 7.30) and `grasp` at 50 variables (42.00 against 17.60, then 43.80 against
19.80). With the alternating scheme `boss` ties in its cell on the first datasets (7.75 against
8.60, a difference of -0.85 [-5.90, +3.90]) and `grasp` is better in its own (8.80 against 17.60,
-8.80 [-13.55, -4.35]). Of the 11 cells, 3 are still worse on the first datasets, 6 tie, and 2 are
better; the fourth worse cell, `grasp` at mean degree 4 with 50 variables, 500 rows, and 10% missing
at random, was a tie with the earlier scheme. The record keeps the earlier scheme's verdicts, its 11
cells on both groups of seeds, and its rows under `earlier_scheme`.

### The alternating scheme against the earlier one

The first search of a fit reads the expected covariance under the model with no edge, where every
covariance is shrunk toward zero, and returns many extra edges. With one refit per round the later
rounds removed them slowly, and the graph was still changing after the 10 rounds; the parameter
steps of the alternating scheme let the expected covariance catch up with the current graph before
the next search. Andrey's EM arms of the lanes with a reference ran again with the alternating
scheme on the recorded datasets, 10,380 fits, and every other arm is the recorded fit. Each cell
pairs the two schemes on its 20 datasets by the rule of [Scores](#scores); the cells of `m2-bnlearn`
and of `notears` include the self-masking ones.

| search | lane | cells | alternating against the earlier scheme: better / tie / worse | fits with the same graph | mean SHD over the cells, earlier -> alternating |
|---|---|---|---|---|---|
| `ges` | `m2-tetrad` | 132 | 14 / 112 / 6 | 929 of 2,640 | 28.92 -> 28.56 |
| `boss` | `m2-tetrad` | 132 | 13 / 116 / 3 | 1,008 of 2,640 | 14.73 -> 14.01 |
| `grasp` | `m2-tetrad` | 132 | 17 / 109 / 6 | 951 of 2,640 | 15.40 -> 14.21 |
| `hc` | `m2-bnlearn` | 45 | 5 / 40 / 0 | 487 of 900 | 11.42 -> 11.18 |
| `tabu` | `m2-bnlearn` | 45 | 6 / 38 / 1 | 422 of 900 | 9.78 -> 9.41 |
| `notears` | `m2-a`, `m2-c` | 33 | 9 / 24 / 0 | 414 of 660 | 6.99 -> 6.56 |

The alternating scheme has fewer errors than the earlier one in 64 cells and more in 16 (6 of `ges`,
3 of `boss`, 6 of `grasp`, 1 of `tabu`), by 0.55 to 10.1 SHD, at every size of the lanes, and the
mean SHD over the cells is lower in every search. The 16 cells did not run again on fresh seeds, so
whether they repeat is not known. One of them is among the 4 cells where Andrey's EM arm is worse
than Tetrad's `MEAN_PAIRWISE` arm on the first datasets (`grasp`, mean degree 4, 20 variables, 2,000
rows, 20% missing at random, a tie on fresh datasets); in the other 15 Andrey's EM arm is not worse
than a reference. The record lists every cell under `against_earlier_scheme`.

A fit records in `metadata["em"]["graph_changed"]` whether each round's search returned another
graph than the round before, and it logs a warning when the last round did: EM has then not settled
on one graph within its 10 rounds.

| search | fits | last round changed the graph | share | 100 rows | 500 rows | 2,000 rows | most often |
|---|---|---|---|---|---|---|---|
| `ges` | 2,640 | 151 | 5.7% | 84 of 1,200 | 39 of 720 | 28 of 720 | 50 variables, 100 rows: 43 of 240 |
| `boss` | 2,640 | 170 | 6.4% | 113 of 1,200 | 36 of 720 | 21 of 720 | 50 variables, 100 rows: 77 of 240 |
| `grasp` | 2,640 | 533 | 20.2% | 324 of 1,200 | 151 of 720 | 58 of 720 | 50 variables, 100 rows: 210 of 240 |
| `hc` | 900 | 20 | 2.2% | 14 of 660 | 4 of 120 | 2 of 120 | 20 variables, 500 rows: 4 of 120 |
| `tabu` | 900 | 18 | 2.0% | 13 of 660 | 2 of 120 | 3 of 120 | 20 variables, 2,000 rows: 3 of 120 |
| `notears` | 660 | 13 | 2.0% | 13 of 660 |  |  | 20 variables, 100 rows: 13 of 660 |

The last round changed the graph in 2 to 6% of the fits of five searches and in 20% of `grasp`'s.
The share grows as the rows fall and the variables rise: at 50 variables and 100 rows it is 210 of
240 fits of `grasp`, 77 of `boss`, and 43 of `ges`.

A fit runs 56 E-steps where it ran 11, so it takes longer:

| search | fit hours of the lane's EM fits, earlier -> alternating | median ratio per fit | by size, lowest to highest | same machine, back to back |
|---|---|---|---|---|
| `ges` | 1.84 -> 5.20 | 2.33 | 1.45 to 2.94 | 2.07 to 4.07 |
| `boss` | 3.92 -> 6.62 | 1.77 | 1.24 to 2.48 | 1.54 to 2.84 |
| `grasp` | 3.60 -> 6.40 | 1.82 | 1.31 to 2.39 | 1.50 to 2.86 |
| `hc` | 0.32 -> 1.05 | 2.32 | 1.74 to 5.09 | 3.66 to 5.00 |
| `tabu` | 0.34 -> 1.03 | 2.06 | 1.54 to 4.84 | 2.88 to 4.87 |
| `notears` | 4.43 -> 4.94 | 1.10 | 1.10 | 1.04 |

The lanes' fits of the two schemes ran on different machines at different times, so their ratios
carry the noise of shared nodes. The last column times both schemes on one core of one machine, back
to back: for `ges`, `boss`, and `grasp` the median ratio over 20 datasets in each of 6 groups of
`m2-tetrad` (20 and 50 variables, 100 and 2,000 rows, MCAR at 30%), and for `hc`, `tabu`, and
`notears` the best of three fits of one dataset at each of three sizes, one for `notears`, whose own
optimizer takes most of its time. A fit takes about 1.5 to 5 times as long as with the earlier
scheme.

### EM against the baselines alone

These lanes run no reference. `ges`, `boss`, and `grasp` at the papers' size, 15 variables and 100
rows, in the cells of the claim (MCAR and logistic MAR), at the default `ebic_gamma=1.0` and at
0.0. The three tables of this section were measured with the earlier scheme, one refit per round,
and did not run again with the alternating one. The first:

| method | `ebic_gamma` | missing | cells | below the empty graph | test-wise PC: better / tie / worse | listwise or mean imputation: worse |
|---|---|---|---|---|---|---|
| `ges` | 1.0 | 10 to 30% | 15 | 15 | 8 / 7 / 0 | 0 |
| `ges` | 1.0 | 40 to 50% | 9 | 4 | 0 / 1 / 8 | 0 |
| `ges` | 0.0 | 10 to 30% | 15 | 14 | 2 / 11 / 2 | 0 |
| `ges` | 0.0 | 40 to 50% | 9 | 6 | 0 / 6 / 3 | 0 |
| `boss` | 1.0 | 10 to 30% | 15 | 15 | 11 / 4 / 0 | 0 |
| `boss` | 1.0 | 40 to 50% | 9 | 4 | 0 / 1 / 8 | 0 |
| `boss` | 0.0 | 10 to 30% | 15 | 15 | 2 / 11 / 2 | 0 |
| `boss` | 0.0 | 40 to 50% | 9 | 5 | 0 / 4 / 5 | 0 |
| `grasp` | 1.0 | 10 to 30% | 15 | 15 | 10 / 4 / 1 | 0 |
| `grasp` | 1.0 | 40 to 50% | 9 | 4 | 0 / 1 / 8 | 0 |
| `grasp` | 0.0 | 10 to 30% | 15 | 15 | 1 / 13 / 1 | 0 |
| `grasp` | 0.0 | 40 to 50% | 9 | 5 | 0 / 5 / 4 | 0 |

On the generic grid, `m2-generic`, measured with the earlier scheme, one refit per round:

| method | mask | cells | below the empty graph | errors / timeouts of fits | test-wise PC: better / tie / worse | worse again | listwise: better / tie / worse | worse again | mean imputation: better / tie / worse | worse again |
|---|---|---|---|---|---|---|---|---|---|---|
| `boss` | MAR, logistic | 72 | 72 of 72 | 0 / 0 of 1,440 | 65 / 7 / 0 | 0 | 28 / 10 / 0 (34 not judged) | 0 | 72 / 0 / 0 | 0 |
| `ges` | MAR, logistic | 72 | 53 of 72 | 0 / 0 of 1,440 | 33 / 39 / 0 | 0 | 20 / 18 / 0 (34 not judged) | 0 | 70 / 2 / 0 | 0 |
| `grasp` | MAR, logistic | 72 | 72 of 72 | 0 / 0 of 1,440 | 66 / 6 / 0 | 0 | 28 / 10 / 0 (34 not judged) | 0 | 72 / 0 / 0 | 0 |
| `boss` | MAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 70 / 2 / 0 | 0 | 39 / 17 / 0 (16 not judged) | 0 | 72 / 0 / 0 | 0 |
| `ges` | MAR, quantile | 72 | 53 of 72 | 0 / 0 of 1,440 | 36 / 33 / 3 | not rerun | 30 / 23 / 3 (16 not judged) | not rerun | 70 / 2 / 0 | 0 |
| `grasp` | MAR, quantile | 72 | 71 of 72 | 0 / 0 of 1,440 | 66 / 6 / 0 | 0 | 37 / 19 / 0 (16 not judged) | 0 | 72 / 0 / 0 | 0 |
| `boss` | MCAR | 72 | 70 of 72 | 0 / 0 of 1,440 | 62 / 10 / 0 | 0 | 20 / 8 / 0 (44 not judged) | 0 | 70 / 2 / 0 | 0 |
| `ges` | MCAR | 72 | 55 of 72 | 0 / 0 of 1,440 | 34 / 36 / 2 | not rerun | 16 / 12 / 0 (44 not judged) | 0 | 71 / 1 / 0 | 0 |
| `grasp` | MCAR | 72 | 70 of 72 | 0 / 0 of 1,440 | 61 / 11 / 0 | 0 | 17 / 11 / 0 (44 not judged) | 0 | 69 / 3 / 0 | 0 |
| `boss` | MNAR, quantile | 72 | 70 of 72 | 0 / 0 of 1,440 | 61 / 11 / 0 | 0 | 33 / 14 / 0 (25 not judged) | 0 | 72 / 0 / 0 | 0 |
| `ges` | MNAR, quantile | 72 | 52 of 72 | 0 / 0 of 1,440 | 33 / 34 / 5 | not rerun | 23 / 22 / 2 (25 not judged) | not rerun | 69 / 3 / 0 | 0 |
| `grasp` | MNAR, quantile | 72 | 70 of 72 | 0 / 0 of 1,440 | 60 / 12 / 0 | 0 | 31 / 16 / 0 (25 not judged) | 0 | 72 / 0 / 0 | 0 |
| `boss` | self-masking | 72 | 72 of 72 | 0 / 0 of 1,440 | 64 / 8 / 0 | 0 | 25 / 12 / 0 (35 not judged) | 0 | 72 / 0 / 0 | 0 |
| `ges` | self-masking | 72 | 58 of 72 | 0 / 0 of 1,440 | 36 / 35 / 1 | not rerun | 18 / 19 / 0 (35 not judged) | 0 | 72 / 0 / 0 | 0 |
| `grasp` | self-masking | 72 | 71 of 72 | 0 / 0 of 1,440 | 63 / 9 / 0 | 0 | 22 / 15 / 0 (35 not judged) | 0 | 72 / 0 / 0 | 0 |

No EM fit raised or ran past the cap, in 25,560 fits. `boss` and `grasp` with EM are never worse
than test-wise PC, listwise deletion, or mean imputation on the generic grid and are below the
empty graph in 214 and 213 of the 216 cells of the claim. `ges` with EM is below it in 161 and is
worse than test-wise PC in 5 and than listwise deletion in 3; those rows were not rerun. At 100
rows with 40 to 50% missing, all three return almost the empty graph at the default
`ebic_gamma=1.0` and miss the floor in 5 of 9 cells.

`m2-reduced`, 20 variables, 500 rows, 20% missing, each method at its defaults, with the verdict
of EM against each baseline in parentheses. It was measured with the earlier scheme, one refit per
round, in `hc`, `tabu`, and `notears` too:

| method | mask | EM | complete data | test-wise PC | listwise | mean imputation | empty graph | below it |
|---|---|---|---|---|---|---|---|---|
| `calm` | MAR, logistic | 10.45 | 8.45 | 7.45 (worse) | 16.20 (better) | 13.90 (better) | 20.00 | yes |
| `dagma` | MAR, logistic | 19.80 | 22.00 | 7.45 (worse) | 37.20 (better) | 19.60 (tie) | 20.00 | no |
| `exact_search` | MAR, logistic | 1.05 | 0.85 | 7.45 (better) | 37.55 (better) | 16.05 (better) | 20.00 | yes |
| `hc` | MAR, logistic | 11.30 | 10.75 | 7.45 (worse) | 17.80 (better) | 24.10 (better) | 20.00 | yes |
| `notears` | MAR, logistic | 16.00 | 15.25 | 7.45 (worse) | 19.80 (better) | 16.40 (tie) | 20.00 | yes |
| `tabu` | MAR, logistic | 5.85 | 5.70 | 7.45 (tie) | 17.85 (better) | 23.35 (better) | 20.00 | yes |
| `xges` | MAR, logistic | 2.20 | 2.00 | 7.45 (better) | 19.35 (better) | 18.95 (better) | 20.00 | yes |
| `calm` | MCAR | 10.85 | 8.45 | 7.50 (worse) | 19.87 (better) | 11.95 (tie) | 20.00 | yes |
| `dagma` | MCAR | 18.80 | 22.00 | 7.50 (worse) | 47.13 (better) | 16.70 (worse) | 20.00 | yes |
| `exact_search` | MCAR | 1.00 | 0.85 | 7.50 (better) | 83.47 (better) | 15.10 (better) | 20.00 | yes |
| `hc` | MCAR | 11.10 | 10.75 | 7.50 (worse) | 34.47 (better) | 20.95 (better) | 20.00 | yes |
| `notears` | MCAR | 15.35 | 15.25 | 7.50 (worse) | 31.73 (better) | 13.55 (worse) | 20.00 | yes |
| `tabu` | MCAR | 6.40 | 5.70 | 7.50 (tie) | 48.13 (better) | 20.10 (better) | 20.00 | yes |
| `xges` | MCAR | 2.20 | 2.00 | 7.50 (better) | 42.93 (better) | 15.80 (better) | 20.00 | yes |

## Status

The bar for a supported value of `missing`: never less accurate than its reference in a cell that
repeats, and below the empty graph in its claimed setting.

- `pc(missing="mvpc")` and `missing="em"` in `notears`, `hc`, and `tabu` are supported: none is
  worse than its reference in any cell, with the alternating scheme as with the earlier one, so no
  cell needed a rerun for that comparison, and each meets the floor wherever it decides a cell.
- `hc` and `tabu` with `missing="em"` are supported with a stated limit. At 15 variables and 100
  rows the result is often no better than the empty graph: `hc` meets the floor in 10 of those 24
  cells and `tabu` in 17. The cause is the plain search at that sample size, which misses the floor
  on the complete data too (on ER1, 8.8 for `hc` and 8.45 for `tabu` against the empty graph's
  8.0). At 20 variables with 500 and 2,000 rows both meet the floor in 12 of 12 cells.
- `ges` with `missing="em"` is supported against Tetrad's EM covariance policy at both of its
  sample sizes, at the plain BIC, up to 50 variables, under MCAR and logistic MAR. Against the
  default arm it is worse in no cell (better in 111, a tie in 21). Against the `MEAN_PAIRWISE` arm
  it is better in 90 cells, ties in 42, and is worse in none. It
  meets the floor in the 2 and the 4 cells where the two arms decide it. The reference is a
  one-pass covariance EM, and at its default it is not below the empty graph in most cells of 100
  rows, so there the comparison says little more than that Andrey's EM does not fail the same way.
  `ges` has a stated limit: at mean degree 4 and this penalty its result is rarely below the empty
  graph (3 of 54 cells), with EM and on the complete data.
- `boss` and `grasp` with `missing="em"` are supported against both arms, with the limits of the
  reference stated for `ges`. Against the default arm neither is worse in any cell (`boss` is better
  in 100 and ties in 32, `grasp` in 101 and 31). Against the `MEAN_PAIRWISE` arm `boss` is better in
  72 cells, ties in 59, and is worse in 1, and `grasp` is better in 68, ties in 61, and is worse in
  3; all 4 worse cells tie on fresh datasets. They are below the empty graph in 122 and 124 of the
  132 cells and meet the floor wherever an arm decides a cell. With the earlier scheme, one refit
  per round, each was worse than the `MEAN_PAIRWISE` arm in one cell on the first datasets and again
  on fresh ones, which kept both experimental; with the alternating scheme `boss` ties in its cell
  and `grasp` is better in its own.
- `xges`, `exact_search`, `calm`, and `dagma` with `missing="em"` stay experimental: no reference
  implementation exists to measure them against. Their rows on this page were measured with the
  earlier scheme.

## Limits

- **MVPC's conditions.** The correction holds when no indicator causes a variable and no value
  causes its own missingness, and its permutation test needs the paper's two conditions; it can
  keep an extra edge when the indicator of a variable it conditions on is a common effect of the
  tested pair. Under masks where every indicator has many weak parents (logistic MAR) it does not
  improve on test-wise PC.
- **MVPC at 50 and 100 variables.** With 20 or 30% missing, MVPC is worse than test-wise PC on the
  first seeds and again on fresh ones in 12 rows of the generic grid, by 0.7 to 3.6 SHD: 9 under
  quantile MNAR (mean degree 2 and 4; 50 and 100 variables; 500, 2,000, and 10,000 rows) and 3 under
  logistic MAR (100 variables, 30% missing; 2,000 and 10,000 rows). The table under "MVPC on the
  generic grid" lists them. In three of them the MVPC authors' R code ran on the same fresh
  datasets: its MVPC ties its own test-wise PC, and Andrey's MVPC is no less accurate than the R
  code's, so in these settings test-wise deletion is the better choice. Traced in three of the nine
  quantile-MNAR rows (see "Why the correction removes true edges"), the loss is true edges that the
  correction removes, for two reasons: under this mask the parents of the indicators have missing
  values too, so a corrected test reads a median of 13% and 19% of the rows the test-wise test reads
  at 50 variables, and Step 3's shared-indicator rule re-tests true edges that falsely detected
  parents make candidates. The other six quantile-MNAR rows and the three logistic-MAR rows were not
  traced. In the paper's setting, at 20 and 50 variables, it is worse than test-wise PC in no cell.
- **Too few rows.** A corrected test that lacks rows is answered by the test-wise test, so the
  pair keeps its test-wise result; `metadata["corrected_edges"]["uncorrected"]` lists it.
- **MVPC's cost.** At 100 variables and 10,000 rows under logistic MAR, 77 fits ran past 30
  minutes in the search for the indicators' parents.
- **EM at 100 rows.** With 40 to 50% missing and 100 rows, `ges`, `boss`, and `grasp` at the default
  `ebic_gamma=1.0` miss the floor in 5 of 9 cells (measured with the earlier scheme). `notears`
  meets it in every cell; with the earlier scheme it missed it at 50% under MCAR on two of three
  graphs. `hc` and `tabu` at the plain BIC miss it in 14 and 7 of the 24 cells with 100 rows, at low
  rates too, as their complete-data fits do on ER1.
- **GES, BOSS, and GRaSP with EM: what was not compared.** The 100-variable cells and the
  quantile MAR mask of the generic grid, self-masking, the default `ebic_gamma=1.0`, and Tetrad's
  `MIN_PAIRWISE` effective sample size did not run against Tetrad. At 100 variables one
  dataset of 100 rows took Tetrad's six fits 365 s. Those cells are in `m2-generic` against the
  baselines.
- **GES with EM on denser graphs.** At the plain BIC and mean degree 4, `ges` with EM is below the
  empty graph in 3 of 54 cells and worse than test-wise PC in 20.
- **EM that has not settled.** The last of the 10 rounds still changed the graph in 2 to 6% of the
  fits of `ges`, `boss`, `hc`, `tabu`, and `notears` and in 20% of `grasp`'s, most often at 50
  variables with 100 rows (210 of 240 fits of `grasp`). Such a fit logs a warning, and
  `metadata["em"]["graph_changed"]` records it.
- **EM's cost.** The alternating scheme runs 56 E-steps where the earlier one ran 11, and a fit
  takes about 1.5 to 5 times as long. Against Tetrad's one-pass EM, Andrey's fit is the slower one
  in 82 to 109 of the 132 cells.
- **Tables of the earlier scheme.** The lanes without a reference (`m2-b`, the `ges`, `boss`, and
  `grasp` rows of `m2-c`, `m2-generic`, and `m2-reduced`) did not run again. Their tables under "EM
  against the baselines alone", and the numbers this page quotes from them, were measured with one
  refit per round and are not the current fit's results: in the lanes that ran again the alternating
  scheme returned another graph in 37 to 65% of the fits.
- **The alternating scheme against the earlier one.** The 16 cells where the alternating scheme has
  more errors than the earlier one did not run again on fresh seeds.
- **Values missing not at random.** EM has no guarantee under self-masking; the rows of that mask
  are reported and not judged.

## Run

```shell
benchmarks/start.sh && benchmarks/start_r.sh && benchmarks/start_mvpc_r.sh && benchmarks/start_missdag.sh
benchmarks/start_tetrad_missing.sh
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m1-paper --out outputs/m1-paper-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m2-bnlearn --part 0 --parts 32 \
    --out outputs/m2-bnlearn-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m2-tetrad --part 0 --parts 64 \
    --out outputs/m2-tetrad-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m1-generic --first-seed 20 \
    --out outputs-rerun/m1-generic-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_summary.py --runs outputs/*/ --data outputs/data \
    --replications outputs-rerun/*/ --out missing-data.json
```

`--part i --parts k` splits a lane into independent jobs, and `--first-seed` moves the seeds for a
rerun. The scripts print the environment variables the R, MissDAG, and Tetrad arms read.

## Known differences

- **Orientation.** The MVPC authors' `mvpc` orients with pcalg's relaxed rule, where a later
  collider overwrites an earlier one on a shared edge. Andrey ranks conflicting colliders by
  p-value. The test-wise control shows the same SHD gap as the MVPC comparison in every cell, and
  the two packages' skeletons agree, so MVPC's margin over the R code is PC's orientation step, not
  the correction.
- **Candidates.** Andrey tests again the pairs that share an adjacent variable or an indicator they
  both cause, the paper's Step 3. The R code runs every edge again and corrects a test when one of
  its variables has an indicator with parents and the pair shares a neighbor.
- **Detection against detection.** Both arms detect the parents of each missingness indicator. The
  paper's experiment scripts pass the true parents to `mvpc`.
- **Two directions.** The R code's skeleton search visits each adjacent pair in both orders, and a
  corrected test draws a fresh permutation at every call, so the two orders of one test can answer
  differently and the edge goes when either does not reject. Andrey's draw depends on the test's
  columns and the seed alone, so both orders give one p-value.
- **Intercept.** Andrey's regressions on the indicators' parents have an intercept; the R code's
  have none, which assumes that every variable has mean 0.
- **Too few rows.** When a corrected test has fewer rows than it needs, Andrey answers it with the
  test-wise test and lists the pair in `metadata["corrected_edges"]["uncorrected"]`. The paper does
  not treat the case. The R code's test then returns 1 when the rows are at most `|S| + 3`, which
  removes the edge, a number from a rank-deficient regression when they are fewer than the
  regressors, and an error with no rows.
- **Generator.** The authors' generator runs at its defaults (a value is missing with probability
  0.9 below the parent's quantile and 0.1 above). On a graph with fewer collider-parent pairs than
  indicators asked for, its MNAR mode leaves the open slots `NA`; the benchmark drops them, and the
  data are unchanged. Its columns are in causal order, so the benchmark shuffles them by the seed.
- **MissDAG's model.** MissDAG fits equal noise variances (`equal_variances=True`) and thresholds
  the weights at 0.3; Andrey's EM fits one variance per variable and `notears` runs at its own
  defaults. The control, both NOTEARS on the complete data, measures what the two optimizers
  differ by without EM.
- **Unit variances in `m2-b`.** MissDAG's paper draws noise variances from U[1, 2] for its GES and
  A* experiments. The harness draws unit variances.
- **bnlearn's E-step.** `structural.em` fills each missing value with its conditional mean and
  searches on the completed data, so the filled values carry no conditional variance. Andrey's EM
  scores from the expected second moments. bnlearn runs 5 iterations by default; the lane sets
  `max.iter = 10`, the rounds Andrey's EM runs.
- **bnlearn's score.** `hc` and `tabu` run at `ebic_gamma=0.0`, the plain BIC, against bnlearn's
  `bic-g`; their default is `ebic_gamma=1.0`, which `m2-reduced` runs.
- **MissDAG's M-step.** MissDAG keeps the weights NOTEARS returns and one noise variance for every
  variable. Andrey fits the weights and a variance per variable again by least squares on the
  parents the fit returns.
- **Tetrad's EM.** Tetrad's EM covariance policy estimates the mean and covariance of a saturated
  Gaussian by EM once, to convergence, from the available-case means and variances, and the search
  scores from that matrix. Andrey's EM runs 10 rounds of an E-step under the current graph's
  linear-Gaussian model and the search, with 5 parameter-only EM steps on the current graph before
  each search after the first. Both count every row as the sample size; Tetrad's
  documentation notes that this under-penalizes a covariance estimated with missing values. The
  second arm counts the mean pairwise number of rows instead; `MIN_PAIRWISE` does not run.
- **Tetrad's searches.** FGES stands beside `ges`. Tetrad's BOSS and GRaSP start from the columns'
  order in the data, and the benchmark shuffles the columns by the seed. `m2-tetrad` runs
  tetrad-lib 8.0.0, the first release with the policy; the other Tetrad lanes run 7.6.10.
- **Tetrad's default.** Without a policy, Tetrad's score deletes rows per local score (test-wise
  deletion). That arm is a baseline and decides no cell.
- **The plain BIC.** `ges`, `boss`, and `grasp` run at `ebic_gamma=0.0` against Tetrad's
  `penaltyDiscount = 1.0`; their default is `ebic_gamma=1.0`, which `m2-b`, `m2-c`, and
  `m2-generic` run.

## References

- Friedman, N. (1997). Learning belief networks in the presence of missing values and hidden
  variables. Proceedings of the 14th International Conference on Machine Learning, 125-133.
- Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H. (2022).
  MissDAG: Causal discovery in the presence of missing data with continuous additive noise models.
  Advances in Neural Information Processing Systems 35.
- Ramsey, J. D., Zhang, K., Glymour, M., Sanchez Romero, R., Huang, B., Ebert-Uphoff, I.,
  Samarasinghe, S., Barnes, E. A., and Glymour, C. (2018). TETRAD: a toolbox for causal discovery.
  8th International Workshop on Climate Informatics.
- Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of Statistical
  Software 35(3), 1-22.
- Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
  discovery in the presence of missing data. Proceedings of the 22nd International Conference on
  Artificial Intelligence and Statistics, PMLR 89, 1762-1770.
