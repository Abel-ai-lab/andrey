# Missing data

`missing_cpu.py` measures the two corrections Andrey has for missing values: Missing-Value PC,
`pc(missing="mvpc")` (Tu et al., 2019), and structural EM, `missing="em"` on the searches that read
the data through a covariance (Friedman, 1997; Gao et al., 2022). `missing_summary.py` pairs the
records into the rows of
[`published/0.16.0/missing-data.json`](../published/0.16.0/missing-data.json).

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

The MVPC and MissDAG repositories carry no license, so `start_mvpc_r.sh` and `start_missdag.sh`
download each at its pinned commit, check the archive's SHA-256, and the benchmark runs the code
unmodified from a git-ignored folder. bnlearn is GPL (>= 2) and comes with the R lane
(`start_r.sh`).

No package implements structural EM around GES, BOSS, GRaSP, XGES, or an exact search. MissDAG's
release holds NOTEARS M-steps only, although its paper reports GES and A*; bnlearn's
`structural.em` takes `hc` and `tabu`; causal-learn, pcalg, XGES, pgmpy, and pyAgrum search
structures on complete data only. Tetrad's structural EM is for discrete Bayesian networks, and its
EM covariance policy estimates one covariance under a saturated Gaussian model before the search, a
different estimator that this benchmark does not run. For those five methods, and for
`calm` and `dagma`, the page reports the baselines and the floor.

## Scores

A comparison is the per-dataset difference `D = SHD(method) - SHD(other)` on the datasets both
finished, with the 95% percentile-bootstrap interval of its mean over 10,000 resamples: worse when
the interval lies wholly above 0, better when wholly below, a tie otherwise. Fewer than 10 paired
datasets are reported and not judged. A worse cell counts only if it is worse again on 20 fresh
seeds. The floor holds in a cell when the interval of `SHD(method) - SHD(empty graph)` lies below
0. It decides a judged cell at the papers' sizes, `m1-paper`, `m2-a`, `m2-b`, `m2-c`, and the
100-row cells of `m2-bnlearn`, where the reference is below the empty graph too; elsewhere it is
reported. A mask outside a method's claim, self-masking for both corrections, is reported and not
judged. `m2-bnlearn` pairs DAGs with CPDAGs, so every arm there is scored on the SHD between the
equivalence classes of the estimate and of the truth.

The bar for a supported method: never less accurate than its reference in a cell that repeats, and
below the empty graph in its claimed setting.

## Results

Every number is on simulated data, one core per fit, on AMD EPYC 7713 and AMD EPYC 9555 processors.
The record holds 192,700 fits; with the reruns on fresh seeds and of the fits that raised, 198,800
were run, in 223 CPU-hours of fit time.

| method | lane | reference | better / tie / worse | worse again | floor | median fit time, Andrey / reference | status |
|---|---|---|---|---|---|---|---|
| `pc(missing="mvpc")` | `m1-paper` | the MVPC authors' R code | 28 / 4 / 0 of 32 | 0 | 32 of 32 | 0.02 to 1.24 | supported |
| `notears(missing="em")` | `m2-a`, `m2-c` | MissDAG | 19 / 5 / 0 of 24 | 0 | 15 of the 15 cells where MissDAG meets it; 22 of 24 | 0.13 to 0.52 | supported |
| `hc(missing="em")` | `m2-bnlearn` | bnlearn `structural.em` | 29 / 7 / 0 of 36 | 0 | 12 of 12 at 500 and 2,000 rows; 11 of 24 at 100 rows, where bnlearn meets it in none | 0.04 to 1.25 | supported |
| `tabu(missing="em")` | `m2-bnlearn` | bnlearn `structural.em` | 32 / 4 / 0 of 36 | 0 | 12 of 12 at 500 and 2,000 rows; 15 of 24 at 100 rows, where bnlearn meets it in one | 0.04 to 1.24 | supported |
| `ges`, `boss`, `grasp`, `xges`, `exact_search`, `calm`, `dagma` with `missing="em"` | `m2-b`, `m2-c`, `m2-generic`, `m2-reduced` | none exists | | | see below | | experimental |

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
| 0.01 | MAR, logistic | 72 | 72 of 72 | 0 / 20 of 1,440 | 0 / 65 / 7 | 1 of 6 rerun | 36 / 10 / 0 (26 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MAR, logistic | 72 | 71 of 72 | 0 / 57 of 1,440 | 4 / 60 / 5 (3 not judged) | 0 of 4 rerun | 28 / 17 / 0 (27 not judged) | 0 | 69 / 0 / 0 (3 not judged) | 0 |
| 0.01 | MAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 14 / 58 / 0 | 0 | 41 / 21 / 0 (10 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 28 / 44 / 0 | 0 | 46 / 16 / 0 (10 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.01 | MCAR | 72 | 72 of 72 | 0 / 0 of 1,440 | 1 / 69 / 2 | 0 of 2 rerun | 30 / 6 / 0 (36 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MCAR | 72 | 72 of 72 | 0 / 0 of 1,440 | 2 / 67 / 3 | 0 of 3 rerun | 30 / 6 / 0 (36 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.01 | MNAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 5 / 55 / 12 | 0 of 4 rerun | 35 / 19 / 0 (18 not judged) | 0 | 72 / 0 / 0 | 0 |
| 0.05 | MNAR, quantile | 72 | 72 of 72 | 0 / 0 of 1,440 | 7 / 53 / 12 | 0 of 5 rerun | 37 / 17 / 0 (18 not judged) | 0 | 72 / 0 / 0 | 0 |
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
  MAR, where every indicator has many weak parents, the correction does not help.
- **Fresh seeds.** 41 rows of the claimed masks are worse than test-wise PC on seeds 0 to 19. The
  24 that were worse before the fits that raised were fit again ran on seeds 20 to 39, and 1 is
  worse again: logistic MAR at mean degree 2, 100 variables, 10,000 rows, 30% missing, `alpha`
  0.01, 13.5 against 11.8 on the 11 datasets that finished, and +3.1 again on 11 fresh ones. None
  of the 5 MCAR rows repeats. The other 17 rows, 15 under quantile MNAR and 2 under logistic MAR,
  turned worse when the fits that had raised were added and have no rerun yet.
- **Floor.** Met in every cell but one: logistic MAR, 100 variables, 10,000 rows, 30% missing, at
  `alpha` 0.05, where most fits ran past the cap.

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
| `hc` | er, degree 2 | 15 | 100 | 13.30 | 10.90 | +2.40 [+0.20, +4.65] | worse | 15.00 |
| `hc` | er, degree 2 | 20 | 2,000 | 11.85 | 12.70 | -0.85 [-4.35, +2.65] | tie | 20.00 |
| `hc` | er, degree 2 | 20 | 500 | 13.75 | 13.05 | +0.70 [-2.80, +4.15] | tie | 20.00 |
| `hc` | scale_free, degree 2 | 15 | 100 | 13.05 | 13.10 | -0.05 [-1.75, +1.70] | tie | 14.00 |
| `tabu` | er, degree 1 | 15 | 100 | 8.45 | 8.30 | +0.15 [-0.85, +1.05] | tie | 8.00 |
| `tabu` | er, degree 2 | 15 | 100 | 12.95 | 11.30 | +1.65 [+0.10, +3.20] | worse | 15.00 |
| `tabu` | er, degree 2 | 20 | 2,000 | 7.30 | 7.15 | +0.15 [-2.85, +3.55] | tie | 20.00 |
| `tabu` | er, degree 2 | 20 | 500 | 10.80 | 9.50 | +1.30 [-1.95, +4.90] | tie | 20.00 |
| `tabu` | scale_free, degree 2 | 15 | 100 | 12.75 | 12.40 | +0.35 [-1.35, +2.05] | tie | 14.00 |

Without missing values the two packages' searches tie in four of the five settings, and Andrey's
`hc` and `tabu` are behind bnlearn's at ER2 with 15 variables and 100 rows, on these 20 datasets.
With EM they are ahead there too, so the margin is the EM layer: `structural.em` fills each
missing value with its conditional mean, and its error grows with the missing rate (under MCAR on
ER2, from 12.4 to 30.1 SHD between 10 and 50%, where Andrey's `hc` goes from 12.0 to 15.2).

At 15 variables and 100 rows the plain BIC overfits: the complete-data fits are not clearly below
the empty graph (on ER1, 8.8 for `hc` and 8.45 for `tabu` against 8.0), and with EM `hc` is below
it in 11 of 24 cells and `tabu` in 15, where bnlearn is in 0 and 1. At 500 and 2,000 rows both
are below it in all 12 cells. Both are worse than test-wise PC in most cells at this penalty (`hc`
in 34 of 36, `tabu` in 18), as their complete-data fits are. The median fit took 0.7 to 3.1 s against
bnlearn's 0.8 to 2.2 s at 100 rows, 6 to 14 s at 500, and 26 to 53 s at 2,000.

### EM in the searches without a reference

`ges`, `boss`, and `grasp` at the papers' size, 15 variables and 100 rows, in the cells of the
claim (MCAR and logistic MAR):

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

- `pc(missing="mvpc")`, `notears(missing="em")`, `hc(missing="em")`, and `tabu(missing="em")` are
  supported: none is worse than its reference in any cell, so no cell needed a rerun for that
  comparison, and each meets the floor wherever it decides a cell. For `hc` and `tabu` the floor
  decides one of the 100-row cells, because bnlearn is above the empty graph in the others; their
  claimed setting is 500 rows and more with 10 to 30% missing (Friedman, 1997, runs 250 to 4,000
  rows), where they meet it in every cell.
- `ges`, `boss`, `grasp`, `xges`, `exact_search`, `calm`, and `dagma` with `missing="em"` stay
  experimental: no reference implementation exists to measure them against.

## Limits

- **MVPC's conditions.** The correction holds when no indicator causes a variable and no value
  causes its own missingness, and its permutation test needs the paper's two conditions; it can
  keep an extra edge when the indicator of a variable it conditions on is a common effect of the
  tested pair. Under masks where every indicator has many weak parents (logistic MAR) it does not
  improve on test-wise PC, and one such cell is worse on fresh seeds too.
- **Too few rows.** A corrected test that lacks rows is answered by the test-wise test, so the
  pair keeps its test-wise result; `metadata["corrected_edges"]["uncorrected"]` lists it.
- **MVPC's cost.** At 100 variables and 10,000 rows under logistic MAR, 77 fits ran past 30
  minutes in the search for the indicators' parents.
- **EM at 100 rows.** With 40 to 50% missing and 100 rows, EM misses the floor: `notears` at 50%
  under MCAR on two of three graphs, and `ges`, `boss`, and `grasp` in 5 of 9 cells. `hc` and
  `tabu` at the plain BIC miss it in about half of the 100-row cells at every rate.
- **Values missing not at random.** EM has no guarantee under self-masking; the rows of that mask
  are reported and not judged.

## Run

```shell
benchmarks/start.sh && benchmarks/start_r.sh && benchmarks/start_mvpc_r.sh && benchmarks/start_missdag.sh
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m1-paper --out outputs/m1-paper-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m2-bnlearn --part 0 --parts 32 \
    --out outputs/m2-bnlearn-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_cpu.py --lane m1-generic --first-seed 20 \
    --out outputs-rerun/m1-generic-0 --resume
$ANDREY_BENCH_PYTHON benchmarks/missing_summary.py --runs outputs/*/ --data outputs/data \
    --replications outputs-rerun/*/ --out missing-data.json
```

`--part i --parts k` splits a lane into independent jobs, and `--first-seed` moves the seeds for a
rerun. The scripts print the environment variables the R and MissDAG arms read.

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

## References

- Friedman, N. (1997). Learning belief networks in the presence of missing values and hidden
  variables. Proceedings of the 14th International Conference on Machine Learning, 125-133.
- Gao, E., Ng, I., Gong, M., Shen, L., Huang, W., Liu, T., Zhang, K., and Bondell, H. (2022).
  MissDAG: Causal discovery in the presence of missing data with continuous additive noise models.
  Advances in Neural Information Processing Systems 35.
- Scutari, M. (2010). Learning Bayesian networks with the bnlearn R package. Journal of Statistical
  Software 35(3), 1-22.
- Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
  discovery in the presence of missing data. Proceedings of the 22nd International Conference on
  Artificial Intelligence and Statistics, PMLR 89, 1762-1770.
