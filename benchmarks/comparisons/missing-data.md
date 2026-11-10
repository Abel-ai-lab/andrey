# Missing data

`missing_cpu.py` measures the two corrections Andrey has for missing values: Missing-Value PC,
`pc(missing="mvpc")` (Tu et al., 2019), and structural EM, `missing="em"` on the searches that read
the data through a covariance (Friedman, 1997; Gao et al., 2022). `missing_summary.py` pairs the
records into the rows of
[`published/0.16.0/missing-data.json`](../published/0.16.0/missing-data.json) and, for the lane
`m2-tetrad`, of
[`published/0.17.0/missing-data-em.json`](../published/0.17.0/missing-data-em.json).

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
| `m2-tetrad` | `ges`, `boss`, `grasp` with `missing="em"` at `ebic_gamma=0.0`; Tetrad's FGES, BOSS, GRaSP under its EM covariance policy and under test-wise deletion | the `m2-b` cells and the 15-variable logistic MAR cells of `m2-c`; the MCAR and logistic MAR cells of `m2-generic` at 10, 20, 50 variables | 132 | 20 |
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
| `ges(missing="em")`, `boss(missing="em")`, `grasp(missing="em")` | Tetrad's FGES, BOSS, and GRaSP with `SemBicScore` under `MissingDataPolicy.EM_COVARIANCE`; the same searches under `TESTWISE`, Tetrad's default, as a baseline, and on the complete data as the control | tetrad-lib 8.0.0, Java 21 | `MissingDataSpec.of(policy)`: ridge 0, tolerance 1e-6, at most 1,000 iterations, `FULL_N`; `penaltyDiscount = 1.0`; search settings as the `tetrad-continuous` lane |

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
re-search". It handles missing values by EM in the same searches with the same score, so it is the
arm the three searches are judged against. For `xges`, `exact_search`, `calm`, and `dagma` no
reference exists, and the page reports the baselines and the floor.

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
The record holds 192,700 fits; with the reruns on fresh seeds, the reruns of the fits that raised,
and the reference check on fresh seeds, 200,860 were run, in 232 CPU-hours of fit time. The lane
`m2-tetrad` holds 47,820 more; with its control on fresh seeds and the refit of 3 fits that raised,
48,043 were run, on AMD EPYC 7713 processors, in 50.5 CPU-hours of fit time.

| method | lane | reference | better / tie / worse | worse again | floor | median fit time, Andrey / reference | status |
|---|---|---|---|---|---|---|---|
| `pc(missing="mvpc")` | `m1-paper` | the MVPC authors' R code | 28 / 4 / 0 of 32 | 0 | 32 of 32 | 0.02 to 1.24 | supported |
| `notears(missing="em")` | `m2-a`, `m2-c` | MissDAG | 19 / 5 / 0 of 24 | 0 | 15 of the 15 cells where MissDAG meets it; 22 of 24 | 0.13 to 0.52 | supported |
| `hc(missing="em")` | `m2-bnlearn` | bnlearn `structural.em` | 29 / 7 / 0 of 36 | 0 | 12 of 12 at 500 and 2,000 rows; 11 of 24 at 100 rows, where bnlearn meets it in none | 0.04 to 1.25 | supported |
| `tabu(missing="em")` | `m2-bnlearn` | bnlearn `structural.em` | 32 / 4 / 0 of 36 | 0 | 12 of 12 at 500 and 2,000 rows; 15 of 24 at 100 rows, where bnlearn meets it in one | 0.04 to 1.24 | supported |
| `ges(missing="em")` | `m2-tetrad` | Tetrad FGES, EM covariance | 107 / 25 / 0 of 132 | 0 | 2 of the 2 cells where Tetrad meets it; 76 of 132 | 0.09 to 1.76 | supported |
| `boss(missing="em")` | `m2-tetrad` | Tetrad BOSS, EM covariance | 95 / 37 / 0 of 132 | 0 | 1 of the 1 cell where Tetrad meets it; 121 of 132 | 0.07 to 4.12 | supported |
| `grasp(missing="em")` | `m2-tetrad` | Tetrad GRaSP, EM covariance | 89 / 43 / 0 of 132 | 0 | 2 of the 2 cells where Tetrad meets it; 122 of 132 | 0.04 to 3.68 | supported |
| `xges`, `exact_search`, `calm`, `dagma` with `missing="em"` | `m2-reduced` | none exists | | | see below | | experimental |

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
the R code's. So in these settings test-wise deletion is the better choice in Andrey. Why the
correction costs Andrey more than it costs the R code is not traced. The median MVPC fit took 1.9,
2.2, and 52 s in Andrey against 4.3, 10.9, and 419 s in the R code.

### NOTEARS with EM against MissDAG

20 variables, 100 rows, 20 datasets per cell. The control is the difference between the two
NOTEARS on the complete data, and the time ratio is Andrey's fit time over MissDAG's.

| lane | graph | mask | rate | Andrey | MissDAG | difference | result | control | empty graph | below it | time ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `m2-a` | er, degree 1 | MCAR | 0.1 | 0.60 | 2.70 | -2.10 [-2.95, -1.25] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.36 |
| `m2-a` | er, degree 1 | MCAR | 0.2 | 1.05 | 4.65 | -3.60 [-4.75, -2.50] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.28 |
| `m2-a` | er, degree 1 | MCAR | 0.3 | 2.55 | 7.55 | -5.00 [-6.40, -3.65] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.24 |
| `m2-a` | er, degree 1 | MCAR | 0.4 | 5.25 | 10.70 | -5.45 [-6.95, -4.00] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.21 |
| `m2-a` | er, degree 1 | MCAR | 0.5 | 8.65 | 12.65 | -4.00 [-5.95, -2.10] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.17 |
| `m2-a` | er, degree 2 | MCAR | 0.1 | 1.80 | 3.90 | -2.10 [-2.95, -1.25] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.52 |
| `m2-a` | er, degree 2 | MCAR | 0.2 | 3.35 | 6.15 | -2.80 [-4.35, -1.40] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.42 |
| `m2-a` | er, degree 2 | MCAR | 0.3 | 7.00 | 11.55 | -4.55 [-7.60, -1.75] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.39 |
| `m2-a` | er, degree 2 | MCAR | 0.4 | 12.75 | 16.50 | -3.75 [-7.05, -0.90] | better | -1.80 [-2.60, -1.10] | 20.00 | yes | 0.37 |
| `m2-a` | er, degree 2 | MCAR | 0.5 | 19.50 | 22.35 | -2.85 [-6.05, -0.30] | better | -1.80 [-2.60, -1.10] | 20.00 | no | 0.35 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.1 | 3.25 | 3.85 | -0.60 [-2.35, +1.60] | tie | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.43 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.2 | 3.80 | 5.50 | -1.70 [-3.30, +0.00] | tie | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.39 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.3 | 8.65 | 11.95 | -3.30 [-5.15, -1.45] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.34 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.4 | 14.80 | 17.05 | -2.25 [-5.45, +0.75] | tie | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.27 |
| `m2-a` | scale_free, degree 2 | MCAR | 0.5 | 18.70 | 20.75 | -2.05 [-4.60, +0.65] | tie | -0.85 [-1.90, +0.05] | 19.00 | no | 0.23 |
| `m2-c` | er, degree 1 | MAR, logistic | 0.1 | 0.30 | 1.90 | -1.60 [-2.30, -0.90] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.21 |
| `m2-c` | er, degree 1 | MAR, logistic | 0.3 | 1.95 | 5.15 | -3.20 [-4.50, -1.85] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.21 |
| `m2-c` | er, degree 1 | MAR, logistic | 0.5 | 6.20 | 9.45 | -3.25 [-5.10, -1.50] | better | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.14 |
| `m2-c` | er, degree 1 | self-masking | 0.1 | 0.40 | 1.75 | -1.35 [-1.90, -0.85] | not judged | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.18 |
| `m2-c` | er, degree 1 | self-masking | 0.3 | 2.20 | 5.45 | -3.25 [-4.45, -2.00] | not judged | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.15 |
| `m2-c` | er, degree 1 | self-masking | 0.5 | 6.15 | 10.30 | -4.15 [-6.25, -2.05] | not judged | -0.85 [-1.25, -0.45] | 10.00 | yes | 0.13 |
| `m2-c` | er, degree 2 | MAR, logistic | 0.1 | 1.75 | 3.40 | -1.65 [-2.65, -0.65] | better | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.26 |
| `m2-c` | er, degree 2 | MAR, logistic | 0.3 | 5.45 | 8.05 | -2.60 [-4.95, -0.25] | better | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.29 |
| `m2-c` | er, degree 2 | MAR, logistic | 0.5 | 14.90 | 19.60 | -4.70 [-7.75, -1.85] | better | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.23 |
| `m2-c` | er, degree 2 | self-masking | 0.1 | 1.80 | 3.75 | -1.95 [-2.85, -1.05] | not judged | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.31 |
| `m2-c` | er, degree 2 | self-masking | 0.3 | 7.20 | 10.20 | -3.00 [-6.50, +0.35] | not judged | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.24 |
| `m2-c` | er, degree 2 | self-masking | 0.5 | 17.25 | 22.85 | -5.60 [-10.05, -1.75] | not judged | -1.80 [-2.55, -1.05] | 20.00 | yes | 0.22 |
| `m2-c` | scale_free, degree 2 | MAR, logistic | 0.1 | 2.20 | 3.25 | -1.05 [-2.00, -0.15] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.29 |
| `m2-c` | scale_free, degree 2 | MAR, logistic | 0.3 | 9.00 | 10.20 | -1.20 [-4.30, +2.10] | tie | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.24 |
| `m2-c` | scale_free, degree 2 | MAR, logistic | 0.5 | 14.60 | 18.15 | -3.55 [-6.25, -0.90] | better | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.25 |
| `m2-c` | scale_free, degree 2 | self-masking | 0.1 | 2.75 | 3.45 | -0.70 [-2.15, +0.80] | not judged | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.30 |
| `m2-c` | scale_free, degree 2 | self-masking | 0.3 | 8.45 | 10.80 | -2.35 [-5.80, +0.90] | not judged | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.25 |
| `m2-c` | scale_free, degree 2 | self-masking | 0.5 | 16.50 | 19.95 | -3.45 [-6.65, -0.45] | not judged | -0.85 [-1.90, +0.05] | 19.00 | yes | 0.21 |

`notears(missing="em")` is better in 19 of the 24 cells of its claim and ties in 5. Part of the
margin is not EM: on the complete data Andrey's NOTEARS is ahead by 0.85 to 1.8 SHD. At 50%
missing on ER2 and the scale-free graph neither package is below the empty graph. The median fit
took 8 to 42 s against MissDAG's 49 to 114 s.

### Hill climbing and Tabu search with EM against bnlearn

`m2-bnlearn`, 20 datasets per cell, SHD between equivalence classes, both packages on the plain
BIC.

| method | size | mask | cells | below the empty graph | errors / timeouts of fits | bnlearn `structural.em` at `max.iter = 10`: better / tie / worse | worse again | test-wise PC: better / tie / worse | worse again |
|---|---|---|---|---|---|---|---|---|---|
| `hc`, `ebic_gamma 0` | 15 variables, 100 rows | MAR, logistic | 9 | 4 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 0 / 1 / 8 | not rerun |
| `tabu`, `ebic_gamma 0` | 15 variables, 100 rows | MAR, logistic | 9 | 4 of 9 | 0 / 0 of 180 | 9 / 0 / 0 | 0 | 0 / 3 / 6 | not rerun |
| `hc`, `ebic_gamma 0` | 15 variables, 100 rows | MCAR | 15 | 7 of 15 | 0 / 0 of 300 | 13 / 2 / 0 | 0 | 0 / 1 / 14 | not rerun |
| `tabu`, `ebic_gamma 0` | 15 variables, 100 rows | MCAR | 15 | 11 of 15 | 0 / 0 of 300 | 13 / 2 / 0 | 0 | 0 / 8 / 7 | not rerun |
| `hc`, `ebic_gamma 0` | 15 variables, 100 rows | self-masking | 9 | 3 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 0 / 3 / 6 | not rerun |
| `tabu`, `ebic_gamma 0` | 15 variables, 100 rows | self-masking | 9 | 4 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 0 / 3 / 6 | not rerun |
| `hc`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MAR, logistic | 6 | 6 of 6 | 0 / 0 of 120 | 4 / 2 / 0 | 0 | 0 / 0 / 6 | not rerun |
| `tabu`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MAR, logistic | 6 | 6 of 6 | 0 / 0 of 120 | 5 / 1 / 0 | 0 | 0 / 4 / 2 | not rerun |
| `hc`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MCAR | 6 | 6 of 6 | 0 / 0 of 120 | 4 / 2 / 0 | 0 | 0 / 0 / 6 | not rerun |
| `tabu`, `ebic_gamma 0` | 20 variables, 500 and 2,000 rows | MCAR | 6 | 6 of 6 | 0 / 0 of 120 | 5 / 1 / 0 | 0 | 0 / 3 / 3 | not rerun |

Neither search is worse than bnlearn's `structural.em` in any cell: `hc` is better in 29 and ties
in 7 of the 36 cells of its claim, `tabu` is better in 32 and ties in 4. The complete-data control
separates the search from its EM layer:

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
from 12.4 to 30.1 SHD between 10 and 50%, where Andrey's `hc` goes from 12.0 to 15.2).

At 15 variables and 100 rows the plain BIC overfits: the complete-data fits are not clearly below
the empty graph (on ER1, 8.8 for `hc` and 8.45 for `tabu` against 8.0), and with EM `hc` is below
it in 11 of 24 cells and `tabu` in 15, where bnlearn is in 0 and 1. At 500 and 2,000 rows both
are below it in all 12 cells. Both are worse than test-wise PC in most cells at this penalty (`hc`
in 34 of 36, `tabu` in 18), as their complete-data fits are. The median fit took 0.7 to 3.1 s against
bnlearn's 0.8 to 2.2 s at 100 rows, 6 to 14 s at 500, and 26 to 53 s at 2,000.

### GES, BOSS, and GRaSP with EM against Tetrad's EM covariance

`m2-tetrad`, 20 datasets per cell, 132 cells per search. Both packages score with the plain BIC:
Andrey at `ebic_gamma=0.0` and `penalty_discount=1.0`, Tetrad at `penaltyDiscount = 1.0`, and both
count every row as the sample size. Andrey's default is `ebic_gamma=1.0`, which has no matching
objective in Tetrad and is not compared here; the lanes below run it against the baselines.

| method | size | mask | cells | below the empty graph | Tetrad's EM arm below it | errors / timeouts of fits | Tetrad, EM covariance: better / tie / worse | worse again | Tetrad, test-wise: better / tie / worse | test-wise PC: better / tie / worse | listwise: better / tie / worse | mean imputation: better / tie / worse |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `ges` | 15 variables, 100 rows | MAR, logistic | 9 | 7 of 9 | 1 of 9 | 0 / 0 of 180 | 9 / 0 / 0 | 0 | 9 / 0 / 0 | 1 / 6 / 2 | 3 / 0 / 0 (6 not judged) | 8 / 1 / 0 |
| `ges` | 15 variables, 100 rows | MCAR | 15 | 13 of 15 | 1 of 15 | 0 / 0 of 300 | 15 / 0 / 0 | 0 | 15 / 0 / 0 | 1 / 11 / 3 | 3 / 0 / 0 (12 not judged) | 15 / 0 / 0 |
| `ges` | 10 to 50 variables, 100 rows | MAR, logistic | 18 | 7 of 18 | 2 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 0 / 9 / 9 | 7 / 3 / 0 (8 not judged) | 16 / 2 / 0 |
| `ges` | 10 to 50 variables, 100 rows | MCAR | 18 | 9 of 18 | 1 of 18 | 0 / 0 of 360 | 16 / 2 / 0 | 0 | 17 / 1 / 0 | 0 / 12 / 6 | 5 / 1 / 0 (12 not judged) | 17 / 1 / 0 |
| `ges` | 10 to 50 variables, 500 rows | MAR, logistic | 18 | 10 of 18 | 5 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 0 / 11 / 7 | 10 / 4 / 0 (4 not judged) | 18 / 0 / 0 |
| `ges` | 10 to 50 variables, 500 rows | MCAR | 18 | 11 of 18 | 4 of 18 | 0 / 0 of 360 | 16 / 2 / 0 | 0 | 18 / 0 / 0 | 1 / 13 / 4 | 7 / 3 / 0 (8 not judged) | 18 / 0 / 0 |
| `ges` | 10 to 50 variables, 2,000 rows | MAR, logistic | 18 | 9 of 18 | 8 of 18 | 0 / 0 of 360 | 9 / 9 / 0 | 0 | 18 / 0 / 0 | 3 / 12 / 3 | 9 / 5 / 0 (4 not judged) | 18 / 0 / 0 |
| `ges` | 10 to 50 variables, 2,000 rows | MCAR | 18 | 10 of 18 | 6 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 18 / 0 / 0 | 0 / 13 / 5 | 9 / 3 / 0 (6 not judged) | 18 / 0 / 0 |
| `boss` | 15 variables, 100 rows | MAR, logistic | 9 | 7 of 9 | 1 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 9 / 0 / 0 | 0 / 6 / 3 | 3 / 0 / 0 (6 not judged) | 9 / 0 / 0 |
| `boss` | 15 variables, 100 rows | MCAR | 15 | 13 of 15 | 0 of 15 | 0 / 0 of 300 | 15 / 0 / 0 | 0 | 15 / 0 / 0 | 2 / 9 / 4 | 3 / 0 / 0 (12 not judged) | 15 / 0 / 0 |
| `boss` | 10 to 50 variables, 100 rows | MAR, logistic | 18 | 14 of 18 | 4 of 18 | 0 / 0 of 360 | 14 / 4 / 0 | 0 | 18 / 0 / 0 | 8 / 7 / 3 | 9 / 1 / 0 (8 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 100 rows | MCAR | 18 | 15 of 18 | 1 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 7 / 9 / 2 | 5 / 1 / 0 (12 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 500 rows | MAR, logistic | 18 | 18 of 18 | 14 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 18 / 0 / 0 | 9 / 8 / 1 | 11 / 3 / 0 (4 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 500 rows | MCAR | 18 | 18 of 18 | 11 of 18 | 0 / 0 of 360 | 11 / 7 / 0 | 0 | 18 / 0 / 0 | 11 / 7 / 0 | 9 / 1 / 0 (8 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 2,000 rows | MAR, logistic | 18 | 18 of 18 | 17 of 18 | 0 / 0 of 360 | 9 / 9 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 11 / 7 / 0 | 10 / 4 / 0 (4 not judged) | 18 / 0 / 0 |
| `boss` | 10 to 50 variables, 2,000 rows | MCAR | 18 | 18 of 18 | 14 of 18 | 0 / 0 of 360 | 11 / 7 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 8 / 10 / 0 | 9 / 3 / 0 (6 not judged) | 18 / 0 / 0 |
| `grasp` | 15 variables, 100 rows | MAR, logistic | 9 | 7 of 9 | 1 of 9 | 0 / 0 of 180 | 8 / 1 / 0 | 0 | 9 / 0 / 0 | 0 / 7 / 2 | 3 / 0 / 0 (6 not judged) | 9 / 0 / 0 |
| `grasp` | 15 variables, 100 rows | MCAR | 15 | 13 of 15 | 1 of 15 | 0 / 0 of 300 | 15 / 0 / 0 | 0 | 15 / 0 / 0 | 1 / 9 / 5 | 3 / 0 / 0 (12 not judged) | 15 / 0 / 0 |
| `grasp` | 10 to 50 variables, 100 rows | MAR, logistic | 18 | 16 of 18 | 4 of 18 | 0 / 0 of 360 | 13 / 5 / 0 | 0 | 18 / 0 / 0 | 6 / 9 / 3 | 8 / 2 / 0 (8 not judged) | 17 / 1 / 0 |
| `grasp` | 10 to 50 variables, 100 rows | MCAR | 18 | 15 of 18 | 3 of 18 | 0 / 0 of 360 | 15 / 3 / 0 | 0 | 18 / 0 / 0 | 4 / 12 / 2 | 5 / 1 / 0 (12 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 500 rows | MAR, logistic | 18 | 18 of 18 | 13 of 18 | 0 / 0 of 360 | 11 / 7 / 0 | 0 | 18 / 0 / 0 | 15 / 3 / 0 | 13 / 1 / 0 (4 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 500 rows | MCAR | 18 | 17 of 18 | 10 of 18 | 0 / 0 of 360 | 12 / 6 / 0 | 0 | 18 / 0 / 0 | 15 / 3 / 0 | 8 / 2 / 0 (8 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 2,000 rows | MAR, logistic | 18 | 18 of 18 | 17 of 18 | 0 / 0 of 360 | 5 / 13 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 15 / 3 / 0 | 10 / 4 / 0 (4 not judged) | 18 / 0 / 0 |
| `grasp` | 10 to 50 variables, 2,000 rows | MCAR | 18 | 18 of 18 | 14 of 18 | 0 / 0 of 360 | 10 / 8 / 0 | 0 | 16 / 0 / 0 (2 not judged) | 12 / 6 / 0 | 9 / 3 / 0 (6 not judged) | 18 / 0 / 0 |

No search is worse than Tetrad's EM arm in any cell, so no cell ran again on fresh seeds. `ges` is
better in 107 cells and ties in 25, `boss` is better in 95 and ties in 37, and `grasp` is better in
89 and ties in 43. No EM fit of either package raised or ran past the cap, in 7,920 fits each. Three
of Andrey's fits, one EM fit of `boss` and two baseline fits, failed on a read of the Python
environment's files and were fit again; the record holds the second fits.

Much of the margin is the reference's weakness, not a strength of Andrey's. Tetrad's EM arm is below
the empty graph in 28 (FGES), 62 (BOSS), and 63 (GRaSP) of the 132 cells, most of them at 500 and
2,000 rows. With 100 rows and 15 or more variables, or with 50 variables and 500 rows, its mean SHD
is far above the empty graph's: the search returns dense graphs. The likely cause is the saturated
covariance, which is estimated without a graph and scored as if every row were complete (see Known
differences). In the cells where Tetrad's EM arm is itself below the empty graph, `ges` is better in
20 and ties in 8, `boss` is better in 29 and ties in 33, and `grasp` is better in 25 and ties in 38.
The mean SHD over the cells of each setting (two masks by three rates; the eight cells of each graph
at 15 variables):

| graph | variables | rows | cells | empty graph | `ges`: Andrey / Tetrad | `boss`: Andrey / Tetrad | `grasp`: Andrey / Tetrad |
|---|---|---|---|---|---|---|---|
| er, degree 1 | 15 | 100 | 8 | 8.00 | 5.23 / 35.69 | 4.86 / 37.27 | 4.86 / 36.62 |
| er, degree 2 | 15 | 100 | 8 | 15.00 | 10.62 / 37.16 | 9.90 / 39.44 | 10.07 / 38.90 |
| scale_free, degree 2 | 15 | 100 | 8 | 14.00 | 9.47 / 36.61 | 10.93 / 40.71 | 10.44 / 40.36 |
| er, degree 2 | 10 | 100 | 6 | 10.00 | 5.38 / 8.63 | 4.89 / 8.83 | 5.32 / 8.04 |
| er, degree 2 | 10 | 500 | 6 | 10.00 | 2.92 / 5.32 | 2.93 / 4.46 | 1.88 / 4.23 |
| er, degree 2 | 10 | 2,000 | 6 | 10.00 | 2.08 / 3.00 | 2.22 / 2.79 | 1.65 / 2.54 |
| er, degree 2 | 20 | 100 | 6 | 20.00 | 13.65 / 43.87 | 12.17 / 48.83 | 10.86 / 48.68 |
| er, degree 2 | 20 | 500 | 6 | 20.00 | 6.58 / 16.11 | 6.08 / 15.99 | 4.77 / 14.41 |
| er, degree 2 | 20 | 2,000 | 6 | 20.00 | 5.30 / 11.91 | 4.38 / 9.94 | 2.89 / 9.28 |
| er, degree 2 | 50 | 100 | 6 | 50.00 | 49.38 / 281.38 | 48.02 / 463.00 | 45.76 / 429.78 |
| er, degree 2 | 50 | 500 | 6 | 50.00 | 23.27 / 99.16 | 17.35 / 114.68 | 14.92 / 109.98 |
| er, degree 2 | 50 | 2,000 | 6 | 50.00 | 13.34 / 49.00 | 10.88 / 48.43 | 7.77 / 44.57 |
| er, degree 4 | 10 | 100 | 6 | 20.00 | 18.78 / 20.77 | 16.74 / 18.80 | 16.41 / 18.37 |
| er, degree 4 | 10 | 500 | 6 | 20.00 | 17.62 / 18.88 | 10.61 / 10.22 | 11.02 / 12.00 |
| er, degree 4 | 10 | 2,000 | 6 | 20.00 | 16.77 / 17.07 | 7.67 / 6.84 | 8.45 / 9.98 |
| er, degree 4 | 20 | 100 | 6 | 40.00 | 40.19 / 61.93 | 25.68 / 61.05 | 31.66 / 60.45 |
| er, degree 4 | 20 | 500 | 6 | 40.00 | 41.95 / 50.82 | 10.54 / 17.12 | 14.28 / 21.93 |
| er, degree 4 | 20 | 2,000 | 6 | 40.00 | 40.32 / 45.52 | 9.23 / 10.24 | 10.12 / 10.03 |
| er, degree 4 | 50 | 100 | 6 | 100.00 | 103.88 / 307.57 | 61.76 / 471.53 | 68.29 / 431.48 |
| er, degree 4 | 50 | 500 | 6 | 100.00 | 95.99 / 178.81 | 22.40 / 127.16 | 28.94 / 134.12 |
| er, degree 4 | 50 | 2,000 | 6 | 100.00 | 105.19 / 144.71 | 16.37 / 41.48 | 19.94 / 40.73 |

The floor decides a cell of 15 variables and 100 rows only where Tetrad's EM arm is below the empty
graph: 2 cells for `ges`, 1 for `boss`, and 2 for `grasp`, and each search is below it in all of
them. Over all 24 cells of that size each search is below the empty graph in 20; the 4 misses are at
50% missing, but for one of `ges` at 10% under logistic MAR on ER1 (6.70 against 8.00, the interval
reaching 0). On the generic cells `boss` is below the empty graph in 101 of 108 and `grasp` in 102,
the misses at 100 rows but for one of `grasp` at 500. `ges` is below it in 51 of the 54 cells of
mean degree 2 and in 5 of the 54 of mean degree 4, where the plain search at this penalty is near or
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

- **Tetrad's test-wise deletion**, its default. Every search with EM is better in every judged
  cell but one tie (`ges`, mean degree 4, 10 variables, 100 rows, MCAR at 10%). At 50 variables
  and 2,000 rows, 170 of Tetrad's test-wise BOSS and GRaSP fits were killed for memory (a fit that
  finished there used up to 12 GB, and the benchmark ran 16 fits side by side in 48 GB) and 6
  GRaSP fits ran past the cap, so 4 cells of each of the two are not judged and the others pair
  fewer datasets. Tetrad's test-wise FGES finished every fit.
- **Test-wise PC.** At this penalty `ges` with EM is worse than `pc` in 39 cells and better in 6,
  `boss` is worse in 13 and better in 56, and `grasp` is worse in 12 and better in 68. The worse
  cells of `boss` and `grasp` are at 100 rows, but for one of `boss` at 500; those of `ges` are at
  every size, 22 of them at mean degree 4. They were not rerun.
- **Listwise deletion and mean imputation.** No search is worse than either in any cell. Listwise
  deletion kept fewer than 4 rows in 3,534 of 7,920 fits, which are errors; PC raised in 5 fits at
  15 variables and 100 rows, where a test had too few rows.

The median EM fit took 0.7 to 9.9 s for `ges`, 0.7 to 33 s for `boss`, and 0.8 to 34 s for `grasp`,
against 0.7 to 45 s, 0.7 to 48 s, and 0.7 to 70 s for Tetrad's, which include about 1.3 s to start
the Java process. By the median ratio of the paired fit times, Andrey's fit was the slower one in 17
(`ges`), 64 (`boss`), and 58 (`grasp`) of the 132 cells, by up to 1.8, 4.1, and 3.7 times, the
largest at mean degree 4 with 500 and 2,000 rows, and the faster one in the others, by up to 11, 15,
and 27 times.

### EM against the baselines alone

These lanes run no reference. `ges`, `boss`, and `grasp` at the papers' size, 15 variables and 100
rows, in the cells of the claim (MCAR and logistic MAR), at the default `ebic_gamma=1.0` and at
0.0:

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

On the generic grid, `m2-generic`:

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
of EM against each baseline in parentheses:

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

- `pc(missing="mvpc")` and `missing="em"` in `notears`, `hc`, `tabu`, `ges`, `boss`, and `grasp` are
  supported: none is worse than its reference in any cell, so no cell needed a rerun for that
  comparison, and each meets the floor wherever it decides a cell.
- `hc` and `tabu` with `missing="em"` are supported with a stated limit. At 15 variables and 100
  rows the result is often no better than the empty graph: `hc` meets the floor in 11 of those 24
  cells and `tabu` in 15. The cause is the plain search at that sample size, which misses the floor
  on the complete data too (on ER1, 8.8 for `hc` and 8.45 for `tabu` against the empty graph's
  8.0). At 20 variables with 500 and 2,000 rows both meet the floor in 12 of 12 cells.
- `ges`, `boss`, and `grasp` with `missing="em"` are supported against Tetrad's EM covariance
  policy, at the plain BIC, up to 50 variables, under MCAR and logistic MAR. The reference is a
  one-pass covariance EM and is not below the empty graph in most cells of 100 rows, so there the
  comparison says little more than that Andrey's EM does not fail the same way. `ges` has a
  stated limit: at mean degree 4 and this penalty its result is rarely below the empty graph (5 of
  54 cells), with EM and on the complete data.
- `xges`, `exact_search`, `calm`, and `dagma` with `missing="em"` stay experimental: no reference
  implementation exists to measure them against.

## Limits

- **MVPC's conditions.** The correction holds when no indicator causes a variable and no value
  causes its own missingness, and its permutation test needs the paper's two conditions; it can
  keep an extra edge when the indicator of a variable it conditions on is a common effect of the
  tested pair. Under masks where every indicator has many weak parents (logistic MAR) it does not
  improve on test-wise PC.
- **MVPC at 50 and 100 variables.** With 20 or 30% missing, MVPC is worse than test-wise PC on
  the first seeds and again on fresh ones in 12 rows of the generic grid, by 0.7 to 3.6 SHD: 9
  under quantile MNAR (mean degree 2 and 4; 50 and 100 variables; 500, 2,000, and 10,000 rows) and
  3 under logistic MAR (100 variables, 30% missing; 2,000 and 10,000 rows). The table under "MVPC
  on the generic grid" lists them. In three of them the MVPC authors' R code ran on the same fresh
  datasets: its MVPC ties its own test-wise PC, and Andrey's MVPC is no less accurate than the R
  code's, so in these settings test-wise deletion is the better choice. In the paper's setting, at
  20 and 50 variables, it is worse than test-wise PC in no cell.
- **Too few rows.** A corrected test that lacks rows is answered by the test-wise test, so the
  pair keeps its test-wise result; `metadata["corrected_edges"]["uncorrected"]` lists it.
- **MVPC's cost.** At 100 variables and 10,000 rows under logistic MAR, 77 fits ran past 30
  minutes in the search for the indicators' parents.
- **EM at 100 rows.** With 40 to 50% missing and 100 rows, EM misses the floor: `notears` at 50%
  under MCAR on two of three graphs, and `ges`, `boss`, and `grasp` in 5 of 9 cells. `hc` and
  `tabu` at the plain BIC miss it in 13 and 9 of the 24 cells with 100 rows, at low rates too, as
  their complete-data fits do on ER1.
- **GES, BOSS, and GRaSP with EM: what was not compared.** The 100-variable cells and the
  quantile MAR mask of the generic grid, self-masking, the default `ebic_gamma=1.0`, and Tetrad's
  effective-sample-size modes other than `FULL_N` did not run against Tetrad. At 100 variables one
  dataset of 100 rows took Tetrad's six fits 365 s. Those cells are in `m2-generic` against the
  baselines.
- **GES with EM on denser graphs.** At the plain BIC and mean degree 4, `ges` with EM is below the
  empty graph in 5 of 54 cells and worse than test-wise PC in 22.
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
  scores from that matrix. Andrey's EM alternates 10 rounds of an E-step under the current graph's
  linear-Gaussian model with the search. Both count every row as the sample size; Tetrad's
  documentation notes that this under-penalizes a covariance estimated with missing values, and it
  has other effective-sample-size modes, which the lane does not run.
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
