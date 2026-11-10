# SCORE and NoGAM

`stein_cpu.py` compares Andrey's `score_matching`, which is SCORE (Rolland et al. 2022), with the
SCORE authors' code, and Andrey's `nogam`, NoGAM (Montagna et al. 2023), with NoGAM in dodiscover,
on the simulation both papers use. Each method runs twice per dataset: once for its DAG
(`andrey.score_matching.numpy`, `andrey.nogam.numpy` against `rolland.score`, `dodiscover.nogam`)
and once for its causal order alone (the `.order` solutions). Andrey's `cam`, R2-SortnRegress
(Reisach et al. 2023), and the empty graph fit the same datasets: `cam` as context, the other two as
floors.

## Data

| Regime | Data | Sizes |
|---|---|---|
| `score_paper`, `score_paper_laplace`, `score_paper_gumbel` | the papers' simulation in its own units, with Gaussian, Laplace, or Gumbel noise | 10, 20; 50 with Gaussian noise |
| `score_paper_standardized`, `..._laplace_standardized`, `..._gumbel_standardized` | the same draws, every column z-scored | the same |
| `score_paper_rescaled`, `..._laplace_rescaled`, `..._gumbel_rescaled` | the same model, each variable divided by its standard deviation before its children read it | the same |

The simulation (`andrey_bench/gp_models.py`) draws an Erdos-Renyi graph with as many edges as
variables (ER1) or four times as many (ER4), makes each variable one draw of a Gaussian process over
its parents (RBF kernel, unit bandwidth) plus noise whose variance is uniform on [0.4, 0.8], and
shuffles the columns by the seed. Every dataset has 1,000 rows, and every cell holds 20 datasets,
seeds 0 to 19. SCORE's claim is Gaussian noise at 10, 20, and 50 variables: 6 cells per scale.
NoGAM's claim is additive noise of any distribution at 10 and 20 variables, here Gaussian, Laplace,
and Gumbel: 12 cells per scale; NoGAM did not run at 50 variables, which its paper does not report.

The z-scored and rescaled cells are the decision cells: 12 for SCORE and 24 for NoGAM. In the
simulation's own units a variable's variance grows along the causal order, which a method can read
without learning anything about the mechanism (Reisach et al. 2021), so the raw cells are reported
and not judged. Andrey standardizes the columns inside its order search, so its fit of a raw cell is
its fit of the z-scored cell of the same draw, and it returned the same graph and order on every
such pair (120 draws for SCORE, 240 for NoGAM); the references read the units they are given.

## Settings

|  | Andrey | Reference |
|---|---|---|
| SCORE | `andrey.score_matching(X)` at its defaults: `alpha=0.001`, ridge term 0.001 in the paper's form, pruning by the smooth-term tests of `cam` | the authors' code at commit `c2c2b8e0` at its defaults (ridge term 0.001, variances not divided by the mean), CAM pruning in R with mgcv 1.9-4 at cutoff 0.001, on torch 2.13.0+cpu |
| NoGAM | `andrey.nogam(X)` at its defaults: `alpha=0.001`, the paper's `gamma` 0.1, five folds, the same pruning | dodiscover at commit `6e644561`, the class defaults with the pruning cutoff set to 0.001, the paper's |
| Context and floors | `andrey.cam(X, alpha=0.001)`, the 0.16.0 development build | R2-SortnRegress in CausalDisco 0.3.0; the empty graph |

The authors' SCORE code has no license, so the harness runs it unmodified and keeps it out of the
repository: `start_stein.sh` downloads its files at the pinned commit and checks each one's SHA-256.
Its results move slightly with the torch version: refit with torch 2.13.0+cpu, the graph solution
returned the output it gave under torch 2.5.1 on 246 of 360 datasets and the order solution on 150
of 360, and a cell's mean moved by -0.25 to +0.55 in SHD and -0.035 to +0.005 in order share. Every
SCORE reference record below was fit with torch 2.13.0+cpu, the version `requirements-stein.txt`
pins.

## Scores

SHD is the structural Hamming distance of the returned DAG from the true one, a reversed edge
counting 1. The order share is the share of true edges that the method's causal order puts in the
true direction, the arrowhead recall of the full DAG of that order; a random order gives 0.5 on
average, and the papers' order divergence is the edge count times one minus the share. Per dataset,
D is Andrey's value minus the reference's. A cell's interval is the percentile bootstrap 95%
interval of the mean D over its paired datasets (10,000 resamples): on SHD the cell is worse when
the interval lies above 0 and better when below, and on the order share the reverse.

A method is no less accurate than its reference when no decision cell is worse on the first seeds
and again on 20 fresh datasets, seeds 20 to 39. Every cell worse on the first seeds on either
measure was rerun, and so was every NoGAM cell better on either measure. A cell counts as better
only where the fresh datasets agree (better again); the published summary records every other cell
as a tie. SCORE's one cell better on the first seeds, on order share, was not rerun and counts as a
tie. A method also has to clear two floors in its decision cells: the interval of its SHD minus the
empty graph's lies below 0, and the interval of its order share minus R2-SortnRegress's lies above
0. A fit past its 1,800 s cap would be a timeout and never scored as a graph; none occurred.

## Results

Andrey's pre-release build of 10 November 2026 on NumPy 2.5.4, one fit at a time on one core with
one BLAS thread, on AMD EPYC 7713 64-Core Processor and AMD EPYC 9534 64-Core Processor processors.
Every number below is in
[`published/0.17.0/stein-score.json`](../published/0.17.0/stein-score.json). All 4,640 fits of the
comparison finished.

| Method | Measure | Decision cells | First seeds: better / tied / worse | Better again | Better, not repeated | Tied | Worse again | Floors passed |
|---|---|---:|---|---:|---:|---:|---:|---:|
| SCORE | SHD | 12 | 0 / 12 / 0 | 0 | 0 | 12 | 0 | 12 of 12 |
| SCORE | Order share | 12 | 1 / 9 / 2 | 0 | 0 | 12 | 0 | 12 of 12 |
| NoGAM | SHD | 24 | 19 / 5 / 0 | 16 | 3 | 5 | 0 | 24 of 24 |
| NoGAM | Order share | 24 | 19 / 5 / 0 | 18 | 1 | 5 | 0 | 24 of 24 |

Better again is better than the reference on the first seeds and on the fresh ones; better, not
repeated is better on the first seeds and a tie on the fresh ones. Tied holds every other cell that
is not worse in both blocks: the ties on the first seeds, SCORE's one cell better on order share on
the first seeds, which was not rerun, and SCORE's two cells worse on order share on the first seeds,
which were not worse again.

### SCORE

Z-scored columns. Means over 20 datasets; D is Andrey minus the authors' code.

| Noise | Variables | Graph | SHD: Andrey | Authors' code | D [95%] | Verdict | Order share: Andrey | Authors' code | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 2.8 | 3.1 | -0.30 [-0.70, +0.05] | tie | 0.805 | 0.785 | +0.020 [-0.010, +0.055] | tie |
| Gaussian | 10 | ER4 | 22.1 | 22.4 | -0.35 [-0.85, +0.10] | tie | 0.970 | 0.961 | +0.009 [-0.003, +0.020] | tie |
| Gaussian | 20 | ER1 | 7.8 | 8.2 | -0.45 [-1.00, +0.10] | tie | 0.728 | 0.710 | +0.018 [-0.005, +0.040] | tie |
| Gaussian | 20 | ER4 | 51.9 | 51.5 | +0.35 [-0.30, +1.10] | tie | 0.904 | 0.905 | -0.001 [-0.015, +0.014] | tie |
| Gaussian | 50 | ER1 | 25.0 | 24.4 | +0.60 [-0.40, +1.60] | tie | 0.649 | 0.660 | -0.011 [-0.026, +0.003] | tie |
| Gaussian | 50 | ER4 | 142.7 | 141.2 | +1.45 [-0.25, +3.25] | tie | 0.855 | 0.871 | -0.016 [-0.025, -0.007] | worse |

Rescaled variables.

| Noise | Variables | Graph | SHD: Andrey | Authors' code | D [95%] | Verdict | Order share: Andrey | Authors' code | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 2.3 | 2.3 | +0.00 [-0.50, +0.55] | tie | 0.845 | 0.835 | +0.010 [-0.025, +0.045] | tie |
| Gaussian | 10 | ER4 | 19.7 | 19.7 | +0.00 [-0.30, +0.25] | tie | 0.980 | 0.976 | +0.004 [-0.004, +0.011] | tie |
| Gaussian | 20 | ER1 | 6.9 | 7.0 | -0.15 [-0.55, +0.30] | tie | 0.748 | 0.730 | +0.018 [+0.003, +0.033] | better |
| Gaussian | 20 | ER4 | 49.9 | 50.1 | -0.25 [-1.20, +0.70] | tie | 0.908 | 0.903 | +0.005 [-0.006, +0.018] | tie |
| Gaussian | 50 | ER1 | 22.6 | 22.1 | +0.50 [-0.45, +1.35] | tie | 0.688 | 0.697 | -0.009 [-0.025, +0.009] | tie |
| Gaussian | 50 | ER4 | 137.6 | 137.0 | +0.60 [-0.75, +2.05] | tie | 0.857 | 0.865 | -0.008 [-0.015, -0.002] | worse |

### NoGAM

Z-scored columns. Means over 20 datasets; D is Andrey minus dodiscover.

| Noise | Variables | Graph | SHD: Andrey | dodiscover | D [95%] | Verdict | Order share: Andrey | dodiscover | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 1.7 | 3.0 | -1.35 [-2.25, -0.55] | better | 0.895 | 0.775 | +0.120 [+0.070, +0.180] | better |
| Gaussian | 10 | ER4 | 21.1 | 22.1 | -0.95 [-2.05, +0.10] | tie | 0.984 | 0.959 | +0.025 [+0.005, +0.048] | better |
| Gaussian | 20 | ER1 | 6.0 | 7.7 | -1.65 [-2.65, -0.70] | better | 0.805 | 0.740 | +0.065 [+0.028, +0.105] | better |
| Gaussian | 20 | ER4 | 49.4 | 52.8 | -3.40 [-4.85, -2.05] | better | 0.922 | 0.895 | +0.028 [+0.014, +0.041] | better |
| Laplace | 10 | ER1 | 1.2 | 2.9 | -1.60 [-2.25, -0.95] | better | 0.900 | 0.765 | +0.135 [+0.085, +0.190] | better |
| Laplace | 10 | ER4 | 19.9 | 20.4 | -0.50 [-1.55, +0.50] | tie | 0.983 | 0.979 | +0.004 [-0.009, +0.016] | tie |
| Laplace | 20 | ER1 | 4.2 | 5.8 | -1.60 [-2.50, -0.70] | better | 0.863 | 0.805 | +0.058 [+0.035, +0.083] | better |
| Laplace | 20 | ER4 | 48.8 | 51.1 | -2.35 [-3.55, -1.20] | better | 0.944 | 0.928 | +0.016 [+0.006, +0.026] | better |
| Gumbel | 10 | ER1 | 1.6 | 2.6 | -1.05 [-1.65, -0.35] | better | 0.920 | 0.815 | +0.105 [+0.055, +0.150] | better |
| Gumbel | 10 | ER4 | 19.6 | 21.0 | -1.35 [-2.50, -0.35] | better | 0.991 | 0.969 | +0.022 [+0.006, +0.040] | better |
| Gumbel | 20 | ER1 | 4.8 | 5.6 | -0.85 [-1.90, +0.20] | tie | 0.848 | 0.810 | +0.037 [-0.007, +0.075] | tie |
| Gumbel | 20 | ER4 | 49.1 | 52.1 | -3.05 [-4.30, -1.80] | better | 0.954 | 0.934 | +0.020 [+0.011, +0.029] | better |

Rescaled variables.

| Noise | Variables | Graph | SHD: Andrey | dodiscover | D [95%] | Verdict | Order share: Andrey | dodiscover | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 0.8 | 2.1 | -1.25 [-1.85, -0.70] | better | 0.965 | 0.865 | +0.100 [+0.055, +0.155] | better |
| Gaussian | 10 | ER4 | 19.4 | 20.3 | -0.90 [-1.65, -0.15] | better | 0.988 | 0.978 | +0.010 [+0.000, +0.021] | tie |
| Gaussian | 20 | ER1 | 4.8 | 6.8 | -2.00 [-2.90, -1.20] | better | 0.838 | 0.760 | +0.077 [+0.050, +0.107] | better |
| Gaussian | 20 | ER4 | 47.5 | 51.4 | -3.95 [-4.80, -3.05] | better | 0.938 | 0.908 | +0.030 [+0.018, +0.043] | better |
| Laplace | 10 | ER1 | 0.8 | 2.1 | -1.30 [-1.95, -0.65] | better | 0.935 | 0.845 | +0.090 [+0.040, +0.140] | better |
| Laplace | 10 | ER4 | 18.2 | 18.4 | -0.10 [-0.95, +0.70] | tie | 0.981 | 0.979 | +0.003 [-0.009, +0.013] | tie |
| Laplace | 20 | ER1 | 3.5 | 5.0 | -1.50 [-2.20, -0.80] | better | 0.902 | 0.853 | +0.050 [+0.030, +0.072] | better |
| Laplace | 20 | ER4 | 46.4 | 48.3 | -1.90 [-3.25, -0.60] | better | 0.939 | 0.919 | +0.019 [+0.006, +0.033] | better |
| Gumbel | 10 | ER1 | 1.2 | 2.4 | -1.15 [-1.65, -0.70] | better | 0.935 | 0.860 | +0.075 [+0.035, +0.120] | better |
| Gumbel | 10 | ER4 | 18.6 | 18.9 | -0.30 [-0.80, +0.25] | tie | 0.983 | 0.985 | -0.003 [-0.014, +0.007] | tie |
| Gumbel | 20 | ER1 | 4.0 | 5.5 | -1.55 [-2.45, -0.70] | better | 0.882 | 0.830 | +0.052 [+0.015, +0.087] | better |
| Gumbel | 20 | ER4 | 46.0 | 48.6 | -2.60 [-3.95, -1.25] | better | 0.944 | 0.926 | +0.018 [+0.008, +0.029] | better |

### Fresh seeds

SCORE. Two decision cells were worse on the first seeds, both at 50 variables on ER4 graphs and both
on the order share alone. Neither was worse again on 20 fresh datasets. The estimate stays below 0
in both blocks, and over all 40 datasets the interval excludes 0, so a deficit of about 0.007 to
0.010 in order share against the authors' code is plausible there. SHD ties in every block.

| SCORE, 50 variables, ER4 | Measure | Seeds 0-19 | Seeds 20-39 | All 40 |
|---|---|---|---|---|
| z-scored | SHD | +1.45 [-0.25, +3.25] tie | +1.00 [-0.05, +2.05] tie | +1.23 [+0.23, +2.27] |
| z-scored | Order share | -0.016 [-0.025, -0.007] worse | -0.004 [-0.010, +0.002] tie | -0.010 [-0.016, -0.005] |
| rescaled | SHD | +0.60 [-0.75, +2.05] tie | +0.25 [-1.70, +2.35] tie | +0.42 [-0.80, +1.70] |
| rescaled | Order share | -0.008 [-0.015, -0.002] worse | -0.006 [-0.015, +0.003] tie | -0.007 [-0.013, -0.001] |

NoGAM. The 20 decision cells better than dodiscover on SHD or on order share on the first seeds were
each rerun on 20 fresh datasets; the 4 cells tied on both measures were not. On SHD, 16 of the 19
cells better on the first seeds were better again and 3 were not; on order share, 18 of 19 were
better again and 1 was not. A cell that did not repeat is a tie on the fresh datasets with its
estimate still on the better side, and over all 40 datasets its interval excludes 0. 18 of the 20
cells were better again on at least one measure, and no cell was worse than dodiscover in either
block. One cell, z-scored Gaussian at 10 variables on ER4 graphs, tied on SHD on the first seeds and
was better on the fresh ones; it counts as a tie.

| Scale | Noise | Variables | Graph | SHD: seeds 0-19 | Seeds 20-39 | All 40 | Result | Order share: seeds 0-19 | Seeds 20-39 | All 40 | Result |
|---|---|---:|---|---|---|---|---|---|---|---|---|
| z-scored | Gaussian | 10 | ER1 | -1.35 [-2.25, -0.55] better | -1.40 [-2.20, -0.75] better | -1.38 [-1.98, -0.82] | better again | +0.120 [+0.070, +0.180] better | +0.105 [+0.065, +0.145] better | +0.113 [+0.080, +0.147] | better again |
| z-scored | Gaussian | 10 | ER4 | -0.95 [-2.05, +0.10] tie | -1.30 [-2.25, -0.40] better | -1.12 [-1.85, -0.45] | tie | +0.025 [+0.005, +0.048] better | +0.022 [+0.005, +0.041] better | +0.024 [+0.010, +0.038] | better again |
| z-scored | Gaussian | 20 | ER1 | -1.65 [-2.65, -0.70] better | -1.35 [-2.10, -0.60] better | -1.50 [-2.12, -0.93] | better again | +0.065 [+0.028, +0.105] better | +0.050 [+0.030, +0.070] better | +0.058 [+0.036, +0.080] | better again |
| z-scored | Gaussian | 20 | ER4 | -3.40 [-4.85, -2.05] better | -4.10 [-5.80, -2.65] better | -3.75 [-4.88, -2.75] | better again | +0.028 [+0.014, +0.041] better | +0.031 [+0.014, +0.050] better | +0.029 [+0.018, +0.041] | better again |
| z-scored | Laplace | 10 | ER1 | -1.60 [-2.25, -0.95] better | -1.25 [-2.00, -0.65] better | -1.43 [-1.93, -0.97] | better again | +0.135 [+0.085, +0.190] better | +0.090 [+0.050, +0.135] better | +0.113 [+0.077, +0.147] | better again |
| z-scored | Laplace | 20 | ER1 | -1.60 [-2.50, -0.70] better | -2.30 [-3.50, -1.20] better | -1.95 [-2.73, -1.25] | better again | +0.058 [+0.035, +0.083] better | +0.077 [+0.045, +0.113] better | +0.068 [+0.046, +0.090] | better again |
| z-scored | Laplace | 20 | ER4 | -2.35 [-3.55, -1.20] better | -1.05 [-2.25, +0.30] tie | -1.70 [-2.58, -0.80] | better, not repeated | +0.016 [+0.006, +0.026] better | +0.012 [-0.006, +0.028] tie | +0.014 [+0.004, +0.023] | better, not repeated |
| z-scored | Gumbel | 10 | ER1 | -1.05 [-1.65, -0.35] better | -1.45 [-2.10, -0.85] better | -1.25 [-1.70, -0.78] | better again | +0.105 [+0.055, +0.150] better | +0.095 [+0.060, +0.130] better | +0.100 [+0.070, +0.130] | better again |
| z-scored | Gumbel | 10 | ER4 | -1.35 [-2.50, -0.35] better | -0.40 [-1.10, +0.30] tie | -0.88 [-1.55, -0.25] | better, not repeated | +0.022 [+0.006, +0.040] better | +0.014 [+0.001, +0.026] better | +0.018 [+0.008, +0.029] | better again |
| z-scored | Gumbel | 20 | ER4 | -3.05 [-4.30, -1.80] better | -2.40 [-4.00, -0.85] better | -2.73 [-3.75, -1.73] | better again | +0.020 [+0.011, +0.029] better | +0.018 [+0.005, +0.029] better | +0.019 [+0.011, +0.027] | better again |
| rescaled | Gaussian | 10 | ER1 | -1.25 [-1.85, -0.70] better | -0.80 [-1.45, -0.10] better | -1.02 [-1.48, -0.57] | better again | +0.100 [+0.055, +0.155] better | +0.090 [+0.055, +0.125] better | +0.095 [+0.068, +0.128] | better again |
| rescaled | Gaussian | 10 | ER4 | -0.90 [-1.65, -0.15] better | -0.70 [-1.50, +0.10] tie | -0.80 [-1.35, -0.25] | better, not repeated | +0.010 [+0.000, +0.021] tie | +0.006 [-0.003, +0.016] tie | +0.008 [+0.001, +0.016] | tie |
| rescaled | Gaussian | 20 | ER1 | -2.00 [-2.90, -1.20] better | -2.35 [-2.95, -1.75] better | -2.17 [-2.73, -1.68] | better again | +0.077 [+0.050, +0.107] better | +0.083 [+0.058, +0.107] better | +0.080 [+0.061, +0.099] | better again |
| rescaled | Gaussian | 20 | ER4 | -3.95 [-4.80, -3.05] better | -3.65 [-4.95, -2.40] better | -3.80 [-4.55, -3.05] | better again | +0.030 [+0.018, +0.043] better | +0.021 [+0.013, +0.030] better | +0.026 [+0.018, +0.033] | better again |
| rescaled | Laplace | 10 | ER1 | -1.30 [-1.95, -0.65] better | -1.20 [-1.70, -0.70] better | -1.25 [-1.68, -0.85] | better again | +0.090 [+0.040, +0.140] better | +0.105 [+0.070, +0.145] better | +0.098 [+0.068, +0.130] | better again |
| rescaled | Laplace | 20 | ER1 | -1.50 [-2.20, -0.80] better | -2.25 [-3.30, -1.10] better | -1.88 [-2.52, -1.20] | better again | +0.050 [+0.030, +0.072] better | +0.072 [+0.035, +0.110] better | +0.061 [+0.040, +0.083] | better again |
| rescaled | Laplace | 20 | ER4 | -1.90 [-3.25, -0.60] better | -2.25 [-3.60, -0.85] better | -2.08 [-3.05, -1.12] | better again | +0.019 [+0.006, +0.033] better | +0.019 [+0.010, +0.028] better | +0.019 [+0.011, +0.028] | better again |
| rescaled | Gumbel | 10 | ER1 | -1.15 [-1.65, -0.70] better | -1.75 [-2.50, -1.05] better | -1.45 [-1.90, -1.02] | better again | +0.075 [+0.035, +0.120] better | +0.140 [+0.105, +0.175] better | +0.107 [+0.080, +0.138] | better again |
| rescaled | Gumbel | 20 | ER1 | -1.55 [-2.45, -0.70] better | -2.50 [-3.40, -1.60] better | -2.02 [-2.67, -1.38] | better again | +0.052 [+0.015, +0.087] better | +0.070 [+0.040, +0.102] better | +0.061 [+0.039, +0.085] | better again |
| rescaled | Gumbel | 20 | ER4 | -2.60 [-3.95, -1.25] better | -2.50 [-3.55, -1.50] better | -2.55 [-3.38, -1.73] | better again | +0.018 [+0.008, +0.029] better | +0.014 [+0.006, +0.024] better | +0.016 [+0.010, +0.023] | better again |

### Floors

Both floors pass in every decision cell, 12 of 12 for SCORE and 24 of 24 for NoGAM. The reference
passes them in the same cells.

SCORE.

| Noise | Variables | Graph | Scale | SHD: Andrey | Empty graph | Andrey - empty [95%] | Order share: Andrey | R2-SortnRegress | Andrey - R2 [95%] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | z-scored | 2.8 | 10 | -7.20 [-7.80, -6.65] | 0.805 | 0.495 | +0.310 [+0.240, +0.380] |
| Gaussian | 10 | ER4 | z-scored | 22.1 | 40 | -17.95 [-19.15, -16.60] | 0.970 | 0.159 | +0.811 [+0.777, +0.844] |
| Gaussian | 20 | ER1 | z-scored | 7.8 | 20 | -12.20 [-13.20, -11.25] | 0.728 | 0.460 | +0.268 [+0.188, +0.350] |
| Gaussian | 20 | ER4 | z-scored | 51.9 | 80 | -28.15 [-30.35, -25.90] | 0.904 | 0.184 | +0.720 [+0.688, +0.754] |
| Gaussian | 50 | ER1 | z-scored | 25.0 | 50 | -25.00 [-26.90, -23.15] | 0.649 | 0.427 | +0.222 [+0.157, +0.290] |
| Gaussian | 50 | ER4 | z-scored | 142.7 | 200 | -57.35 [-61.20, -53.45] | 0.855 | 0.186 | +0.668 [+0.634, +0.699] |
| Gaussian | 10 | ER1 | rescaled | 2.3 | 10 | -7.70 [-8.45, -6.90] | 0.845 | 0.450 | +0.395 [+0.285, +0.490] |
| Gaussian | 10 | ER4 | rescaled | 19.7 | 40 | -20.30 [-21.60, -19.05] | 0.980 | 0.182 | +0.797 [+0.771, +0.824] |
| Gaussian | 20 | ER1 | rescaled | 6.9 | 20 | -13.10 [-14.15, -12.05] | 0.748 | 0.480 | +0.268 [+0.182, +0.347] |
| Gaussian | 20 | ER4 | rescaled | 49.9 | 80 | -30.10 [-32.25, -27.85] | 0.908 | 0.187 | +0.721 [+0.695, +0.746] |
| Gaussian | 50 | ER1 | rescaled | 22.6 | 50 | -27.40 [-29.70, -25.10] | 0.688 | 0.441 | +0.247 [+0.180, +0.315] |
| Gaussian | 50 | ER4 | rescaled | 137.6 | 200 | -62.40 [-67.20, -57.45] | 0.857 | 0.205 | +0.651 [+0.616, +0.682] |

NoGAM.

| Noise | Variables | Graph | Scale | SHD: Andrey | Empty graph | Andrey - empty [95%] | Order share: Andrey | R2-SortnRegress | Andrey - R2 [95%] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | z-scored | 1.7 | 10 | -8.30 [-8.90, -7.70] | 0.895 | 0.495 | +0.400 [+0.325, +0.470] |
| Gaussian | 10 | ER4 | z-scored | 21.1 | 40 | -18.90 [-19.85, -17.95] | 0.984 | 0.159 | +0.825 [+0.792, +0.856] |
| Gaussian | 20 | ER1 | z-scored | 6.0 | 20 | -14.00 [-14.90, -13.10] | 0.805 | 0.460 | +0.345 [+0.270, +0.417] |
| Gaussian | 20 | ER4 | z-scored | 49.4 | 80 | -30.60 [-32.70, -28.45] | 0.922 | 0.184 | +0.738 [+0.700, +0.775] |
| Laplace | 10 | ER1 | z-scored | 1.2 | 10 | -8.75 [-9.25, -8.15] | 0.900 | 0.470 | +0.430 [+0.340, +0.520] |
| Laplace | 10 | ER4 | z-scored | 19.9 | 40 | -20.10 [-21.25, -18.95] | 0.983 | 0.204 | +0.779 [+0.740, +0.819] |
| Laplace | 20 | ER1 | z-scored | 4.2 | 20 | -15.75 [-16.55, -14.95] | 0.863 | 0.460 | +0.403 [+0.340, +0.465] |
| Laplace | 20 | ER4 | z-scored | 48.8 | 80 | -31.25 [-32.95, -29.35] | 0.944 | 0.228 | +0.717 [+0.684, +0.749] |
| Gumbel | 10 | ER1 | z-scored | 1.6 | 10 | -8.45 [-9.05, -7.70] | 0.920 | 0.455 | +0.465 [+0.395, +0.540] |
| Gumbel | 10 | ER4 | z-scored | 19.6 | 40 | -20.35 [-21.80, -18.95] | 0.991 | 0.172 | +0.819 [+0.789, +0.846] |
| Gumbel | 20 | ER1 | z-scored | 4.8 | 20 | -15.25 [-16.45, -13.90] | 0.848 | 0.495 | +0.352 [+0.287, +0.412] |
| Gumbel | 20 | ER4 | z-scored | 49.1 | 80 | -30.90 [-33.25, -28.35] | 0.954 | 0.193 | +0.761 [+0.736, +0.787] |
| Gaussian | 10 | ER1 | rescaled | 0.8 | 10 | -9.15 [-9.60, -8.60] | 0.965 | 0.450 | +0.515 [+0.440, +0.590] |
| Gaussian | 10 | ER4 | rescaled | 19.4 | 40 | -20.60 [-21.80, -19.45] | 0.988 | 0.182 | +0.805 [+0.776, +0.834] |
| Gaussian | 20 | ER1 | rescaled | 4.8 | 20 | -15.25 [-16.05, -14.45] | 0.838 | 0.480 | +0.357 [+0.273, +0.435] |
| Gaussian | 20 | ER4 | rescaled | 47.5 | 80 | -32.55 [-34.55, -30.55] | 0.938 | 0.187 | +0.751 [+0.728, +0.775] |
| Laplace | 10 | ER1 | rescaled | 0.8 | 10 | -9.15 [-9.55, -8.65] | 0.935 | 0.445 | +0.490 [+0.405, +0.575] |
| Laplace | 10 | ER4 | rescaled | 18.2 | 40 | -21.75 [-23.30, -20.30] | 0.981 | 0.237 | +0.744 [+0.691, +0.792] |
| Laplace | 20 | ER1 | rescaled | 3.5 | 20 | -16.45 [-17.35, -15.50] | 0.902 | 0.475 | +0.427 [+0.365, +0.487] |
| Laplace | 20 | ER4 | rescaled | 46.4 | 80 | -33.60 [-35.50, -31.70] | 0.939 | 0.247 | +0.691 [+0.660, +0.726] |
| Gumbel | 10 | ER1 | rescaled | 1.2 | 10 | -8.80 [-9.25, -8.25] | 0.935 | 0.445 | +0.490 [+0.410, +0.575] |
| Gumbel | 10 | ER4 | rescaled | 18.6 | 40 | -21.40 [-23.20, -19.70] | 0.983 | 0.186 | +0.796 [+0.754, +0.838] |
| Gumbel | 20 | ER1 | rescaled | 4.0 | 20 | -16.00 [-17.15, -14.75] | 0.882 | 0.502 | +0.380 [+0.320, +0.438] |
| Gumbel | 20 | ER4 | rescaled | 46.0 | 80 | -33.95 [-36.05, -31.90] | 0.944 | 0.209 | +0.735 [+0.711, +0.760] |

### Beside CAM

`cam` fits the same model by a greedy search over regressions. On sparse graphs (ER1) it orders more
true edges than either method and returns a closer graph; on dense graphs (ER4) SCORE and NoGAM
order as many true edges or more. Over the decision cells, SCORE against `cam` on SHD is 1 better, 1
tied, 10 worse, and on order share 2 better, 4 tied, 6 worse; NoGAM on SHD is 4 better, 5 tied, 15
worse, and on order share 11 better, 5 tied, 8 worse. The z-scored cells:

SCORE.

| Noise | Variables | Graph | SHD: Andrey | `cam` | D [95%] | Verdict | Order share: Andrey | `cam` | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 2.8 | 0.7 | +2.15 [+1.35, +2.90] | worse | 0.805 | 0.980 | -0.175 [-0.225, -0.125] | worse |
| Gaussian | 10 | ER4 | 22.1 | 21.9 | +0.15 [-0.85, +1.15] | tie | 0.970 | 0.909 | +0.061 [+0.035, +0.091] | better |
| Gaussian | 20 | ER1 | 7.8 | 2.1 | +5.70 [+4.80, +6.55] | worse | 0.728 | 0.963 | -0.235 [-0.273, -0.195] | worse |
| Gaussian | 20 | ER4 | 51.9 | 46.5 | +5.40 [+4.00, +7.20] | worse | 0.904 | 0.897 | +0.007 [-0.019, +0.037] | tie |
| Gaussian | 50 | ER1 | 25.0 | 5.5 | +19.50 [+17.10, +21.85] | worse | 0.649 | 0.966 | -0.317 [-0.353, -0.280] | worse |
| Gaussian | 50 | ER4 | 142.7 | 123.3 | +19.30 [+16.35, +22.40] | worse | 0.855 | 0.826 | +0.029 [-0.006, +0.065] | tie |

NoGAM.

| Noise | Variables | Graph | SHD: Andrey | `cam` | D [95%] | Verdict | Order share: Andrey | `cam` | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 1.7 | 0.7 | +1.05 [+0.35, +1.65] | worse | 0.895 | 0.980 | -0.085 [-0.150, -0.020] | worse |
| Gaussian | 10 | ER4 | 21.1 | 21.9 | -0.80 [-1.55, -0.05] | better | 0.984 | 0.909 | +0.075 [+0.046, +0.109] | better |
| Gaussian | 20 | ER1 | 6.0 | 2.1 | +3.90 [+3.05, +4.75] | worse | 0.805 | 0.963 | -0.158 [-0.190, -0.128] | worse |
| Gaussian | 20 | ER4 | 49.4 | 46.5 | +2.95 [+1.45, +4.90] | worse | 0.922 | 0.897 | +0.026 [-0.004, +0.053] | tie |
| Laplace | 10 | ER1 | 1.2 | 0.8 | +0.50 [+0.00, +1.00] | tie | 0.900 | 0.955 | -0.055 [-0.100, -0.010] | worse |
| Laplace | 10 | ER4 | 19.9 | 20.4 | -0.50 [-1.15, +0.10] | tie | 0.983 | 0.941 | +0.041 [+0.019, +0.066] | better |
| Laplace | 20 | ER1 | 4.2 | 1.4 | +2.80 [+2.05, +3.55] | worse | 0.863 | 0.973 | -0.110 [-0.138, -0.083] | worse |
| Laplace | 20 | ER4 | 48.8 | 46.4 | +2.40 [+1.50, +3.30] | worse | 0.944 | 0.912 | +0.032 [+0.004, +0.062] | better |
| Gumbel | 10 | ER1 | 1.6 | 0.9 | +0.65 [-0.05, +1.45] | tie | 0.920 | 0.940 | -0.020 [-0.065, +0.015] | tie |
| Gumbel | 10 | ER4 | 19.6 | 21.3 | -1.65 [-2.50, -0.90] | better | 0.991 | 0.902 | +0.089 [+0.054, +0.125] | better |
| Gumbel | 20 | ER1 | 4.8 | 3.3 | +1.45 [+0.05, +2.90] | worse | 0.848 | 0.900 | -0.052 [-0.110, +0.005] | tie |
| Gumbel | 20 | ER4 | 49.1 | 47.8 | +1.35 [+0.10, +2.75] | worse | 0.954 | 0.873 | +0.081 [+0.045, +0.117] | better |

### Raw scale

Reported, not judged. The references read the simulation's units, where the variances carry the
order, and Andrey standardizes first, so here the references score higher: SCORE on SHD is 0 better,
1 tied, 5 worse and on order share 0 better, 1 tied, 5 worse; NoGAM on SHD is 0 better, 7 tied, 5
worse and on order share 0 better, 4 tied, 8 worse. These are Andrey's z-scored results beside the
references' raw-scale results.

SCORE.

| Noise | Variables | Graph | SHD: Andrey | Authors' code | D [95%] | Verdict | Order share: Andrey | Authors' code | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 2.8 | 1.1 | +1.75 [+0.95, +2.55] | worse | 0.805 | 0.955 | -0.150 [-0.200, -0.100] | worse |
| Gaussian | 10 | ER4 | 22.1 | 21.8 | +0.25 [-0.45, +1.00] | tie | 0.970 | 0.965 | +0.005 [-0.010, +0.022] | tie |
| Gaussian | 20 | ER1 | 7.8 | 3.5 | +4.25 [+3.10, +5.35] | worse | 0.728 | 0.920 | -0.193 [-0.237, -0.147] | worse |
| Gaussian | 20 | ER4 | 51.9 | 47.7 | +4.15 [+2.35, +6.25] | worse | 0.904 | 0.946 | -0.041 [-0.068, -0.014] | worse |
| Gaussian | 50 | ER1 | 25.0 | 11.1 | +13.95 [+11.60, +16.35] | worse | 0.649 | 0.905 | -0.256 [-0.288, -0.223] | worse |
| Gaussian | 50 | ER4 | 142.7 | 134.3 | +8.35 [+5.45, +11.35] | worse | 0.855 | 0.906 | -0.052 [-0.069, -0.035] | worse |

NoGAM.

| Noise | Variables | Graph | SHD: Andrey | dodiscover | D [95%] | Verdict | Order share: Andrey | dodiscover | D [95%] | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 1.7 | 0.9 | +0.80 [+0.10, +1.45] | worse | 0.895 | 0.960 | -0.065 [-0.120, -0.015] | worse |
| Gaussian | 10 | ER4 | 21.1 | 21.1 | +0.05 [-0.65, +0.80] | tie | 0.984 | 0.988 | -0.004 [-0.014, +0.006] | tie |
| Gaussian | 20 | ER1 | 6.0 | 3.4 | +2.60 [+1.55, +3.65] | worse | 0.805 | 0.917 | -0.113 [-0.155, -0.072] | worse |
| Gaussian | 20 | ER4 | 49.4 | 48.5 | +0.90 [-0.70, +3.05] | tie | 0.922 | 0.953 | -0.030 [-0.058, -0.009] | worse |
| Laplace | 10 | ER1 | 1.2 | 0.8 | +0.50 [+0.05, +0.95] | worse | 0.900 | 0.960 | -0.060 [-0.100, -0.025] | worse |
| Laplace | 10 | ER4 | 19.9 | 19.5 | +0.40 [-0.50, +1.30] | tie | 0.983 | 0.993 | -0.010 [-0.020, -0.001] | worse |
| Laplace | 20 | ER1 | 4.2 | 2.9 | +1.35 [+0.45, +2.25] | worse | 0.863 | 0.945 | -0.083 [-0.115, -0.048] | worse |
| Laplace | 20 | ER4 | 48.8 | 48.1 | +0.60 [-0.70, +1.90] | tie | 0.944 | 0.961 | -0.017 [-0.029, -0.004] | worse |
| Gumbel | 10 | ER1 | 1.6 | 1.2 | +0.35 [-0.30, +1.05] | tie | 0.920 | 0.955 | -0.035 [-0.075, +0.000] | tie |
| Gumbel | 10 | ER4 | 19.6 | 19.9 | -0.25 [-1.00, +0.45] | tie | 0.991 | 0.990 | +0.001 [-0.006, +0.007] | tie |
| Gumbel | 20 | ER1 | 4.8 | 3.2 | +1.50 [+0.35, +2.75] | worse | 0.848 | 0.932 | -0.085 [-0.133, -0.040] | worse |
| Gumbel | 20 | ER4 | 49.1 | 49.3 | -0.20 [-1.80, +1.65] | tie | 0.954 | 0.963 | -0.009 [-0.026, +0.006] | tie |

### Speed

Median seconds per fit of the DAG solution on the z-scored cells, and the median over the datasets
of the reference's time over Andrey's: 4.1 to 8.7 for SCORE and 1.9 to 3.0 for NoGAM. Andrey's fits,
the references' fits, and `cam`'s ran in separate runs on shared nodes, so the ratios are
indicative. The peak is the largest resident memory of an Andrey fit in the cell.

SCORE.

| Noise | Variables | Graph | Andrey, s | Authors' code, s | Ratio | `cam`, s | Andrey peak, MB |
|---|---|---|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 1.0 | 4.0 | 4.1 | 0.9 | 150 |
| Gaussian | 10 | ER4 | 0.9 | 4.0 | 4.6 | 1.3 | 150 |
| Gaussian | 20 | ER1 | 3.0 | 14.3 | 4.9 | 4.0 | 150 |
| Gaussian | 20 | ER4 | 2.7 | 14.1 | 5.0 | 4.4 | 146 |
| Gaussian | 50 | ER1 | 28.2 | 262.3 | 8.7 | 11.3 | 152 |
| Gaussian | 50 | ER4 | 32.4 | 251.8 | 7.5 | 12.1 | 152 |

NoGAM.

| Noise | Variables | Graph | Andrey, s | dodiscover, s | Ratio | `cam`, s | Andrey peak, MB |
|---|---|---|---:|---:|---:|---:|---:|
| Gaussian | 10 | ER1 | 7.5 | 20.0 | 2.7 | 0.9 | 155 |
| Gaussian | 10 | ER4 | 7.6 | 20.4 | 2.7 | 1.3 | 155 |
| Gaussian | 20 | ER1 | 29.5 | 83.5 | 2.8 | 4.0 | 155 |
| Gaussian | 20 | ER4 | 28.3 | 83.3 | 3.0 | 4.4 | 155 |
| Laplace | 10 | ER1 | 9.4 | 20.4 | 2.2 | 0.8 | 155 |
| Laplace | 10 | ER4 | 9.7 | 23.7 | 2.4 | 1.2 | 155 |
| Laplace | 20 | ER1 | 37.9 | 79.2 | 2.1 | 3.0 | 156 |
| Laplace | 20 | ER4 | 37.8 | 108.3 | 2.9 | 4.3 | 155 |
| Gumbel | 10 | ER1 | 8.8 | 20.7 | 2.2 | 0.8 | 155 |
| Gumbel | 10 | ER4 | 9.2 | 17.8 | 1.9 | 1.3 | 141 |
| Gumbel | 20 | ER1 | 39.9 | 82.5 | 2.1 | 3.5 | 155 |
| Gumbel | 20 | ER4 | 39.1 | 82.2 | 2.1 | 4.7 | 155 |

### Additive-noise simulation

Outside the papers' simulation, reported and not judged: Andrey's own additive-noise generator
(`andrey.data`'s `additive_noise`, each link `z + sin z` of a weighted parent), z-scored, ER1, 10
and 20 variables, 1,000 rows, seeds 0 to 19, under the three noises. In the 0.16.0 screen both
reference implementations failed the floors on these datasets; the reference columns below are that
screen's records, SCORE's fit with torch 2.5.1. Andrey's SCORE has an SHD interval below the empty
graph's in 0 of 6 cells and an order-share interval above R2-SortnRegress's in 0 of 6; NoGAM in 1
and 0 of 6. Their orders put 0.41 to 0.53 (SCORE) and 0.47 to 0.62 (NoGAM) of the true edges in the
true direction, where `cam` puts 0.76 to 0.94. Neither clears the floors there, as neither reference
did, and `cam` does. It is a stated limit of both methods, outside the setting their papers claim.

| Method | Noise | Variables | SHD: Andrey | Reference | `cam` | Empty graph | Andrey - empty [95%] | Order share: Andrey | Reference | `cam` | R2-SortnRegress | Andrey - R2 [95%] |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SCORE | Gaussian | 10 | 10.8 | 10.7 | 1.4 | 10 | +0.80 [-0.85, +2.60] | 0.465 | 0.470 | 0.935 | 0.575 | -0.110 [-0.220, -0.005] |
| SCORE | Gaussian | 20 | 27.4 | 26.5 | 4.2 | 20 | +7.35 [+5.45, +9.50] | 0.410 | 0.422 | 0.927 | 0.650 | -0.240 [-0.312, -0.168] |
| SCORE | Laplace | 10 | 10.3 | 10.1 | 1.6 | 10 | +0.35 [-1.30, +2.10] | 0.510 | 0.540 | 0.915 | 0.645 | -0.135 [-0.240, -0.035] |
| SCORE | Laplace | 20 | 25.8 | 25.4 | 5.5 | 20 | +5.75 [+3.85, +7.75] | 0.460 | 0.472 | 0.910 | 0.665 | -0.205 [-0.282, -0.127] |
| SCORE | Gumbel | 10 | 10.3 | 10.0 | 4.7 | 10 | +0.35 [-1.10, +2.00] | 0.525 | 0.525 | 0.755 | 0.635 | -0.110 [-0.215, -0.010] |
| SCORE | Gumbel | 20 | 25.1 | 25.4 | 8.7 | 20 | +5.10 [+3.50, +6.95] | 0.480 | 0.487 | 0.795 | 0.670 | -0.190 [-0.260, -0.120] |
| NoGAM | Gaussian | 10 | 9.7 | 10.7 | 1.4 | 10 | -0.30 [-2.20, +1.70] | 0.555 | 0.470 | 0.935 | 0.575 | -0.020 [-0.140, +0.100] |
| NoGAM | Gaussian | 20 | 25.9 | 27.3 | 4.2 | 20 | +5.90 [+3.75, +8.20] | 0.470 | 0.477 | 0.927 | 0.650 | -0.180 [-0.260, -0.102] |
| NoGAM | Laplace | 10 | 8.2 | 9.6 | 1.6 | 10 | -1.85 [-3.60, -0.25] | 0.625 | 0.600 | 0.915 | 0.645 | -0.020 [-0.135, +0.095] |
| NoGAM | Laplace | 20 | 23.5 | 23.6 | 5.5 | 20 | +3.50 [+1.30, +5.85] | 0.527 | 0.583 | 0.910 | 0.665 | -0.138 [-0.217, -0.060] |
| NoGAM | Gumbel | 10 | 9.2 | 10.1 | 4.7 | 10 | -0.75 [-2.60, +0.95] | 0.575 | 0.545 | 0.755 | 0.635 | -0.060 [-0.175, +0.050] |
| NoGAM | Gumbel | 20 | 23.4 | 26.1 | 8.7 | 20 | +3.35 [+1.25, +5.65] | 0.555 | 0.550 | 0.795 | 0.670 | -0.115 [-0.193, -0.040] |

The cause is the error of the two estimators, traced on these datasets with the exact derivatives of
the log density beside the estimates. A leaf is a variable with no children among those that remain,
and its variance-to-noise ratio is its variance over the variance of its noise term. Measured:
SCORE's estimate of a leaf's criterion, whose exact value is 0 under Gaussian noise, grows with
about the square of the leaf's ratio (slope 1.5 to 2.0 of log estimate on log ratio, correlation
0.96 to 0.98, under the three noises and at 1,000, 4,000, and 8,000 rows), and NoGAM's criterion of
a leaf grows about linearly with it (slope 0.8, correlation 0.91 to 0.93). Over the leaves of four
simulations with Gaussian noise and 10 variables, the median estimate of a leaf was 5.4 where its
ratio was below 3 and 2,350 where it was 30 or above, against an exact criterion of about 2.4 for
the closest variable that is not a leaf. The links `z + sin z` have a strong linear part, which
passes variance down the causal order, so the sinks have the largest ratios: a leaf's ratio had a
median of 9.7 at 10 variables and 13.3 at 20, where a redraw of the papers' simulation, with links
whose derivatives are closed forms, gives 1.6 and 1.7. On z-scored columns both searches then rank
the sinks last.

Order share with one setting changed at a time, the mean of 20 datasets (standard error 0.02 to
0.04; a random order gives 0.5), from a copy of each order search that returns the package's orders
on these datasets. A longer dataset is the same model's longer draw, and a dash is a setting that
was not run.

| Setting | SCORE: Gaussian, 10 variables | Gaussian, 20 | Laplace, 10 | Laplace, 20 | NoGAM: Gaussian, 10 variables |
|---|---:|---:|---:|---:|---:|
| As shipped: 1,000 rows, ridge term 0.001, median bandwidth | 0.465 | 0.410 | 0.510 | 0.460 | 0.555 |
| 4,000 rows | 0.480 | 0.430 | 0.570 | 0.485 | 0.625 |
| 16,000 rows | 0.515 | - | - | - | - |
| Half the bandwidth | 0.440 | 0.425 | 0.475 | 0.453 | 0.480 |
| Twice the bandwidth | 0.495 | 0.443 | 0.580 | 0.495 | 0.550 |
| Ridge term 0.0001 | 0.445 | 0.400 | 0.445 | 0.445 | 0.470 |
| Ridge term 0.01 | 0.470 | 0.440 | 0.585 | 0.487 | 0.615 |
| Links `sin z` alone, as shipped | 0.665 | - | - | - | 0.755 |
| A redraw of the papers' simulation, as shipped | 0.740 | - | - | - | 0.885 |

Measured: more rows, another ridge term, or another bandwidth leave both orders near a random one;
an order search at 16,000 rows took 420 s and 4.1 GB. Without the linear part of the links (`sin z`
alone, a median ratio of 1.9) the order share is 0.665 for SCORE and 0.755 for NoGAM, more than half
of the way to the redraw of the papers' simulation. No argument of `score_matching` or `nogam`
changes this. Inferred, not measured: the estimators' errors act as relative errors, a fixed share
of the quantity they estimate, so the variable with the largest entry carries the largest spurious
variation; from the trend between 4,000 and 8,000 rows, a leaf would need rows of the order of
100,000 to win, beyond one fit's memory, which grows with the square of the rows; and large, uneven
diagonal entries of the inverse correlation matrix, which mean some variables are almost determined
by the others, are a sign of such data that needs no graph. That sign was not tested as a rule.

## Run

```shell
benchmarks/start_stein.sh
export ANDREY_BENCH_STEIN_PYTHON=$PWD/benchmarks/.venv-stein/bin/python
export ANDREY_BENCH_SCORE=$PWD/benchmarks/.score
export ANDREY_BENCH_CAM_PYTHON=$ANDREY_BENCH_PYTHON
$ANDREY_BENCH_PYTHON benchmarks/stein_cpu.py --out outputs/stein-smoke \
    --regimes score_paper_standardized --ladder 10 --densities er1 --seeds 2 --cores 1
```

Andrey's solutions and `cam` run in the interpreter `$ANDREY_BENCH_CAM_PYTHON` names, bench-env's
for the build under test. The references run in `$ANDREY_BENCH_STEIN_PYTHON`, built from the lock
file `requirements-stein.txt`; SCORE prunes with mgcv through Rscript, so load R as `start_r.sh
--env` prints first. The defaults are 20 datasets per cell from seed 0, one fit without warm-up, and
a 1,800 s cap. Fill the shared dataset store once with `--prepare-only`, then cut a campaign into
small runs with `--regimes`, `--ladder`, `--densities`, `--solutions`, `--seeds`, and
`--seed-start`, each with its own `--out`; fresh datasets for a rerun start at seed 20:

```shell
export ANDREY_BENCH_DATA=$PWD/outputs/stein-data
$ANDREY_BENCH_PYTHON benchmarks/stein_cpu.py --out outputs/stein-prep --ladder 10 20 50 --prepare-only
$ANDREY_BENCH_PYTHON benchmarks/stein_cpu.py --out outputs/stein-nogam-d10 \
    --regimes score_paper_laplace_standardized --ladder 10 --densities er1 er4 --cores 1 \
    --solutions andrey.nogam.numpy andrey.nogam.order.numpy dodiscover.nogam dodiscover.nogam.order
```

A dataset's id is a hash of the generator call, so separate runs of the same cell pair on it. The
regimes `anm_er`, `anm_laplace_er`, and `anm_gumbel_er` are the additive-noise simulation above.

## Known differences

- **The ridge term of the Stein estimator.** Both papers write the estimator with the kernel
  `exp(-||x - y||^2 / (2 s^2))` and a ridge term added to the kernel matrix, and state no value.
  Both reference implementations divide the kernel by the bandwidth `s` before adding 0.001, which
  equals a ridge term of `0.001 s` on the papers' kernel. Andrey adds 0.001 to the papers' kernel.
- **NoGAM's `gamma`.** The paper gives the kernel-ridge parameter `gamma` as 0.1. The reference
  implementation passes the value of its ridge term, 0.01. Andrey uses the paper's 0.1.
- **Folds.** NoGAM's paper predicts the score from the residuals by K-fold cross-validation and
  names no K. Andrey uses five folds of consecutive rows and also predicts the residuals out of
  fold.
- **Pruning.** Both papers prune with CAM's significance tests at 0.001. SCORE's authors call the
  CAM package in R, dodiscover fits its own additive models, and Andrey uses the pruning of its
  `cam`, penalized thin plate regression splines with smoothing chosen by GCV.
- **Units.** Andrey centers the columns and divides them by their standard deviations before the
  order search, so its result does not depend on the columns' units; the references read the
  columns as given.
- **Precision and ties.** The authors' SCORE code computes in 32-bit floats. Andrey computes in
  64-bit floats, compares its criteria rounded to 9 decimal places, and breaks a tie by a rank
  computed from the data, so its result does not depend on the column order.

## Limits

- **Standardized sparse graphs.** On z-scored ER1 graphs SCORE orders 0.65 to 0.81 of the true edges
  correctly and NoGAM 0.81 to 0.92, where the references reach 0.91 to 0.95 and 0.92 to 0.96 on the
  same draws in the simulation's own units and `cam` 0.90 to 0.98 z-scored. The references lose as
  much when they read z-scored columns. On sparse graphs `cam` is the more accurate method.
- **Order share at 50 variables on dense graphs.** SCORE's order share was 0.855 against the
  authors' 0.871 on the first seeds (z-scored) and 0.864 against 0.868 on fresh ones: not worse
  again, with a pooled deficit of about 0.007 to 0.010.
- **Gaussian noise for SCORE.** SCORE's leaf rule is exact under Gaussian noise only, and it was
  judged there only. Under other additive noise use NoGAM.
- **NoGAM at 50 variables.** Not measured: its paper reports 10 and 20 variables, and one fit at 50
  variables and 1,000 rows took about three minutes.
- **Other additive-noise data.** On Andrey's additive-noise simulation with links `z + sin z`,
  z-scored, SCORE's order puts 41% to 53% of the true edges in the true direction and NoGAM's 47% to
  62%, near a random order's 50%, and at 20 variables both return a graph farther from the truth
  than the empty graph; `cam` orders 76% to 94%. The estimators' error on a leaf grows with the
  leaf's variance-to-noise ratio, about its square for SCORE and about linearly for NoGAM, and links
  with a strong linear part give the sinks the largest ratios (a median of 10 to 13 there against
  1.6 to 1.7 in a redraw of the papers' simulation), so both searches rank the sinks last; 16,000
  rows, another ridge term, or another bandwidth do not change it, as the additive-noise section
  above measures. Use `cam` on such data.
- **Time and memory.** Each step of either order search factors a matrix with one row and one column
  per sample, so time grows with the cube of the rows and memory with their square; every number
  here is at 1,000 rows.

## References

- Montagna, F., Noceti, N., Rosasco, L., Zhang, K., and Locatello, F. (2023). Causal discovery with
  score matching on additive models with arbitrary noise. CLeaR, PMLR 213, 726-751.
- Reisach, A. G., Seiler, C., and Weichwald, S. (2021). Beware of the simulated DAG! Causal
  discovery benchmarks may be easy to game. NeurIPS 34.
- Reisach, A. G., Tami, M., Seiler, C., Chambaz, A., and Weichwald, S. (2023). A scale-invariant
  sorting criterion to find a causal order in additive noise models. NeurIPS 36.
- Rolland, P., Cevher, V., Kleindessner, M., Russell, C., Janzing, D., Scholkopf, B., and
  Locatello, F. (2022). Score matching enables causal discovery of nonlinear additive noise models.
  ICML, PMLR 162, 18741-18753.
