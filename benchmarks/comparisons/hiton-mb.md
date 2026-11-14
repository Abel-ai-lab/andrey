# HITON-MB

This campaign compares Andrey's `hiton_mb` with the authors' HITON-MB in Causal Explorer 1.5 on
the Markov blanket each learns at every target, on the networks and sizes of its paper.
`blanket_cpu.py` fits the solutions, as [Markov blankets](blankets.md) describes, and
`blanket_summary.py` judges each cell. The rows are in
[`published/0.18.0/hiton-mb.json`](../published/0.18.0/hiton-mb.json), and the
[0.18.0 release card](../../docs/docs/benchmarks/0.18.0.md#hiton-mb) summarizes them.

## Data

The paper (Aliferis et al., 2010, Part I, Section 6) ran HITON-MB on samples of child, insurance,
alarm, hailfinder, and pigs at 200, 500, and 5,000 rows. We ran child (20 variables), insurance
(27), alarm (37), and hailfinder (56) at those three sizes with every variable a target, and pigs
(441 variables) at 500 rows with 50 targets drawn from each dataset's seed. Every cell has 20
datasets, seeds 0 to 19, and a cell rerun on fresh datasets has seeds 20 to 39. A column that holds
one value is left out of every fit and answers the empty set at its target. The paper also runs
Fisher's z on continuous data, which the toolbox does not read, so on linear-Gaussian graphs with
20, 50, and 100 variables at 1,000 and 5,000 rows (mean degree 2, and 4 on 50 variables)
`hiton_mb` is judged by the floor alone.

## Solutions

| Solution | Setting |
|---|---|
| `andrey.hiton_mb` | the defaults: `max_depth=3`, `degrees_of_freedom="mean"`, alpha 0.05, `"gsq"` (`"fisherz"` on Gaussian data) |
| Causal Explorer `HITON_MB` | max-k 3, G-square at 0.05, at each target; the toolbox's form runs no symmetry check |
| context | `mmmb` at `max_depth=3`; `iamb`, `inter_iamb`, `gs`, and `pcmb` at their defaults (seeds 0 to 9 on the networks); `hiton_mb` with `degrees_of_freedom="nonempty"` |

The toolbox stops at a target now and then with "Deseparating set S is not found". Such a target
is left out of both sides of its dataset's pair, and the tables count the targets left.

## Measures

Per dataset we average, over the targets with a nonempty true blanket, the distance of the set's
precision and recall from (1, 1), where an empty answer counts precision 1 and distance 1, and the
set's F1. D is Andrey's value minus the toolbox's per dataset. A cell is worse when the 95%
percentile-bootstrap interval of the mean D (10,000 resamples of the datasets) lies on Andrey's
worse side on seeds 0 to 19 and again on seeds 20 to 39, better when it lies on the better side on
seeds 0 to 19, and a tie otherwise. The floor counts the errors per target, missing plus extra
members, against the empty set's, which are the true set's size; `hiton_mb` passes when the paired
interval of the difference lies below 0.

## Results (0.18.0)

Against the toolbox, `hiton_mb` was better in 11 of the 13 settings and worse in 2, hailfinder and
insurance at 5,000 rows, where the first datasets and the 20 fresh ones agree: on the fresh ones the
distance was 0.600 against 0.517 and 0.343 against 0.327. The toolbox's HITON_MB keeps each run's
parents and children as they are, the form the paper's experiments evaluate; Andrey's runs the
symmetry check of the paper's definition, which keeps a variable only when the target is also among
the variable's. A trace with the same tests and the check left out ties the toolbox in both settings
on seeds 0 to 19. The check raises the precision, 0.89 against 0.70 on hailfinder and 0.96 against
0.89 on insurance, at a larger cost in recall, 0.44 against 0.65 and 0.68 against 0.76. On
hailfinder, pairs of variables are copies of each other in the data, so a test given one of a pair
finds the other independent of the pair's children, and the check drops it from each child's set.

Andrey's errors are below the toolbox's in every setting. They are below the empty set's in every
setting but hailfinder at 200 and 500 rows, 3.98 and 4.41 per target against 3.72 and 3.26, where
the toolbox's are 15.65 and 9.15. There HITON-PC's rule of five rows per cell leaves many tests
unrun: without it the errors are 2.38 and 2.02, below the empty set's, but the F1 at 500 rows on
hailfinder and insurance falls below the toolbox's. On the Gaussian graphs the errors are below the
empty set's in all eight settings, 0.39 to 1.00 per target at mean degree 2 and 5.96 and 6.10 at
mean degree 4.

With `degrees_of_freedom="nonempty"`, the tally against the toolbox is the same, 11 better and 2
worse on the first datasets. `hiton_mb` keeps `"mean"`: its paper takes its tests from Tsamardinos,
Brown, and Aliferis (2006), whose degrees of freedom follow Steck and Jaakkola (2002), a count from
the data's configurations rather than the rule of Spirtes et al. that `mmmb`'s paper names. A median
fit took 2.2 to 16 s on the four networks and 91 s on pigs; the toolbox's took 21 to 420 s and 150
s.

### Against Causal Explorer

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 20 | 0.522 / 0.592 | -0.069 [-0.094, -0.044] | 0.593 / 0.552 | +0.041 [+0.020, +0.062] | better / better | - | 2.78 / 4.63 (3.51) |
| alarm, 500 | 20 | 0.326 / 0.347 | -0.021 [-0.037, -0.005] | 0.749 / 0.753 | -0.004 [-0.016, +0.006] | better / tie | - | 1.60 / 2.15 (3.51) |
| alarm, 5,000 | 20 | 0.115 / 0.196 | -0.081 [-0.093, -0.071] | 0.918 / 0.869 | +0.049 [+0.042, +0.058] | better / better | - | 0.57 / 0.97 (3.51) |
| child, 200 | 20 | 0.450 / 0.632 | -0.183 [-0.203, -0.163] | 0.652 / 0.505 | +0.147 [+0.128, +0.166] | better / better | - | 2.37 / 5.64 (3.00) |
| child, 500 | 20 | 0.285 / 0.383 | -0.098 [-0.118, -0.077] | 0.797 / 0.726 | +0.071 [+0.055, +0.086] | better / better | - | 1.18 / 2.52 (3.00) |
| child, 5,000 | 20 | 0.022 / 0.142 | -0.120 [-0.135, -0.104] | 0.987 / 0.906 | +0.081 [+0.070, +0.091] | better / better | - | 0.10 / 0.44 (3.01) |
| hailfinder, 200 | 20 | 0.663 / 0.837 | -0.173 [-0.184, -0.161] | 0.506 / 0.315 | +0.191 [+0.179, +0.203] | better / better | - | 3.98 / 15.65 (3.72) |
| hailfinder, 500 | 20 | 0.610 / 0.653 | -0.043 [-0.057, -0.031] | 0.503 / 0.469 | +0.034 [+0.025, +0.045] | better / better | - | 4.41 / 9.15 (3.26) |
| hailfinder, 5,000 | 20 | 0.606 / 0.525 | +0.081 [+0.070, +0.092] | 0.496 / 0.576 | -0.080 [-0.085, -0.075] | worse / worse | worse / worse | 2.04 / 6.99 (3.52) |
| insurance, 200 | 20 | 0.658 / 0.703 | -0.045 [-0.060, -0.029] | 0.508 / 0.461 | +0.047 [+0.033, +0.061] | better / better | - | 4.35 / 8.16 (5.19) |
| insurance, 500 | 20 | 0.531 / 0.563 | -0.032 [-0.048, -0.016] | 0.605 / 0.592 | +0.013 [-0.002, +0.027] | better / tie | - | 2.99 / 4.15 (5.18) |
| insurance, 5,000 | 20 | 0.346 / 0.318 | +0.028 [+0.011, +0.045] | 0.755 / 0.787 | -0.032 [-0.046, -0.018] | worse / worse | worse / worse | 1.89 / 1.95 (5.20) |
| pigs, 500 | 20 | 0.068 / 0.355 | -0.287 [-0.300, -0.275] | 0.959 / 0.764 | +0.195 [+0.185, +0.204] | better / better | - | 0.28 / 1.87 (3.64) |

### Context

The context arms are reported and never judged; `iamb`, `inter_iamb`, `gs`, and `pcmb` ran on
seeds 0 to 9 on the networks.

#### `hiton_mb` against mmmb, max_depth 3

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 20 | 0.523 / 0.465 | +0.058 [+0.038, +0.078] | 0.592 / 0.623 | -0.031 [-0.049, -0.012] | worse / worse | - | 2.79 / 1.92 (3.51) |
| alarm, 500 | 20 | 0.327 / 0.302 | +0.024 [+0.012, +0.037] | 0.748 / 0.766 | -0.018 [-0.028, -0.008] | worse / worse | - | 1.60 / 1.31 (3.51) |
| alarm, 5,000 | 20 | 0.114 / 0.111 | +0.004 [-0.001, +0.009] | 0.918 / 0.921 | -0.002 [-0.005, +0.001] | tie / tie | - | 0.57 / 0.56 (3.51) |
| child, 200 | 20 | 0.450 / 0.363 | +0.087 [+0.064, +0.109] | 0.652 / 0.721 | -0.068 [-0.090, -0.046] | worse / worse | - | 2.37 / 1.33 (3.00) |
| child, 500 | 20 | 0.285 / 0.242 | +0.043 [+0.026, +0.059] | 0.797 / 0.823 | -0.027 [-0.039, -0.014] | worse / worse | - | 1.18 / 0.92 (3.00) |
| child, 5,000 | 20 | 0.022 / 0.021 | +0.000 [-0.012, +0.012] | 0.987 / 0.986 | +0.000 [-0.007, +0.009] | tie / tie | - | 0.10 / 0.08 (3.00) |
| hailfinder, 200 | 20 | 0.819 / 0.745 | +0.074 [+0.059, +0.086] | 0.402 / 0.390 | +0.012 [+0.004, +0.020] | worse / better | - | 5.41 / 2.61 (3.54) |
| hailfinder, 500 | 20 | 0.615 / 0.686 | -0.072 [-0.082, -0.061] | 0.502 / 0.439 | +0.062 [+0.057, +0.068] | better / better | - | 4.68 / 2.45 (3.54) |
| hailfinder, 5,000 | 20 | 0.606 / 0.619 | -0.013 [-0.018, -0.006] | 0.496 / 0.495 | +0.002 [-0.001, +0.005] | better / tie | better / tie | 2.05 / 2.12 (3.54) |
| insurance, 200 | 20 | 0.659 / 0.643 | +0.016 [-0.006, +0.037] | 0.507 / 0.485 | +0.022 [+0.005, +0.039] | tie / better | - | 4.35 / 3.38 (5.19) |
| insurance, 500 | 20 | 0.530 / 0.539 | -0.009 [-0.030, +0.014] | 0.607 / 0.589 | +0.018 [-0.001, +0.036] | tie / tie | - | 2.99 / 2.83 (5.19) |
| insurance, 5,000 | 20 | 0.344 / 0.336 | +0.008 [-0.006, +0.023] | 0.756 / 0.762 | -0.005 [-0.017, +0.006] | tie / tie | tie / tie | 1.87 / 1.84 (5.19) |
| d 100, 1,000, degree 2 | 20 | 0.178 / 0.182 | -0.004 [-0.008, +0.001] | 0.879 / 0.877 | +0.002 [-0.001, +0.006] | tie / tie | - | 1.00 / 1.01 (3.88) |
| d 100, 5,000, degree 2 | 20 | 0.133 / 0.135 | -0.002 [-0.005, +0.002] | 0.912 / 0.910 | +0.001 [-0.001, +0.004] | tie / tie | - | 0.69 / 0.70 (3.88) |
| d 20, 1,000, degree 2 | 20 | 0.128 / 0.125 | +0.003 [-0.008, +0.015] | 0.915 / 0.917 | -0.002 [-0.011, +0.005] | tie / tie | - | 0.63 / 0.63 (3.35) |
| d 20, 5,000, degree 2 | 20 | 0.082 / 0.080 | +0.002 [-0.013, +0.018] | 0.947 / 0.949 | -0.002 [-0.013, +0.008] | tie / tie | - | 0.39 / 0.38 (3.35) |
| d 50, 1,000, degree 2 | 20 | 0.149 / 0.151 | -0.003 [-0.011, +0.006] | 0.898 / 0.896 | +0.002 [-0.004, +0.008] | tie / tie | - | 0.85 / 0.86 (3.72) |
| d 50, 1,000, degree 4 | 20 | 0.589 / 0.583 | +0.006 [-0.004, +0.017] | 0.557 / 0.560 | -0.003 [-0.013, +0.006] | tie / tie | - | 6.10 / 5.99 (8.52) |
| d 50, 5,000, degree 2 | 20 | 0.101 / 0.100 | +0.001 [-0.007, +0.010] | 0.934 / 0.935 | -0.001 [-0.007, +0.004] | tie / tie | - | 0.56 / 0.54 (3.72) |
| d 50, 5,000, degree 4 | 20 | 0.551 / 0.529 | +0.022 [+0.009, +0.036] | 0.591 / 0.608 | -0.017 [-0.028, -0.006] | worse / worse | - | 5.96 / 5.67 (8.52) |

#### `hiton_mb` against iamb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 10 | 0.526 / 0.527 | -0.001 [-0.033, +0.028] | 0.592 / 0.617 | -0.025 [-0.048, +0.001] | tie / tie | - | 2.78 / 2.48 (3.51) |
| alarm, 500 | 10 | 0.340 / 0.428 | -0.088 [-0.125, -0.049] | 0.740 / 0.702 | +0.038 [+0.009, +0.065] | better / better | - | 1.64 / 2.17 (3.51) |
| alarm, 5,000 | 10 | 0.115 / 0.331 | -0.216 [-0.249, -0.186] | 0.918 / 0.774 | +0.144 [+0.123, +0.168] | better / better | - | 0.57 / 1.72 (3.51) |
| child, 200 | 10 | 0.466 / 0.391 | +0.075 [+0.024, +0.117] | 0.637 / 0.721 | -0.084 [-0.122, -0.042] | worse / worse | - | 2.43 / 1.62 (3.00) |
| child, 500 | 10 | 0.278 / 0.281 | -0.003 [-0.039, +0.026] | 0.803 / 0.810 | -0.007 [-0.029, +0.021] | tie / tie | - | 1.18 / 1.25 (3.00) |
| child, 5,000 | 10 | 0.015 / 0.207 | -0.192 [-0.220, -0.165] | 0.991 / 0.862 | +0.129 [+0.111, +0.148] | better / better | - | 0.07 / 0.84 (3.00) |
| hailfinder, 200 | 10 | 0.815 / 0.725 | +0.089 [+0.013, +0.162] | 0.405 / 0.472 | -0.067 [-0.118, -0.014] | worse / worse | - | 5.32 / 3.15 (3.54) |
| hailfinder, 500 | 10 | 0.613 / 0.627 | -0.014 [-0.096, +0.065] | 0.501 / 0.544 | -0.043 [-0.101, +0.015] | tie / tie | - | 4.75 / 2.88 (3.54) |
| hailfinder, 5,000 | 10 | 0.603 / 0.614 | -0.011 [-0.077, +0.052] | 0.500 / 0.548 | -0.048 [-0.091, -0.005] | tie / worse | - | 2.03 / 3.16 (3.54) |
| insurance, 200 | 10 | 0.649 / 0.622 | +0.027 [-0.010, +0.062] | 0.514 / 0.545 | -0.030 [-0.058, -0.001] | tie / worse | - | 4.29 / 3.66 (5.19) |
| insurance, 500 | 10 | 0.537 / 0.535 | +0.002 [-0.023, +0.031] | 0.600 / 0.621 | -0.021 [-0.043, -0.001] | tie / worse | - | 3.04 / 3.14 (5.19) |
| insurance, 5,000 | 10 | 0.354 / 0.361 | -0.007 [-0.042, +0.027] | 0.749 / 0.753 | -0.004 [-0.031, +0.025] | tie / tie | - | 1.93 / 2.34 (5.19) |
| d 100, 1,000, degree 2 | 20 | 0.178 / 0.498 | -0.320 [-0.341, -0.298] | 0.879 / 0.640 | +0.239 [+0.223, +0.255] | better / better | - | 1.00 / 3.55 (3.88) |
| d 100, 5,000, degree 2 | 20 | 0.133 / 0.488 | -0.356 [-0.373, -0.337] | 0.912 / 0.648 | +0.263 [+0.249, +0.276] | better / better | - | 0.69 / 3.42 (3.88) |
| d 20, 1,000, degree 2 | 20 | 0.128 / 0.152 | -0.024 [-0.070, +0.028] | 0.915 / 0.903 | +0.012 [-0.025, +0.044] | tie / tie | - | 0.63 / 0.59 (3.35) |
| d 20, 5,000, degree 2 | 20 | 0.082 / 0.153 | -0.071 [-0.108, -0.029] | 0.947 / 0.902 | +0.045 [+0.017, +0.069] | better / better | - | 0.39 / 0.61 (3.35) |
| d 50, 1,000, degree 2 | 20 | 0.149 / 0.318 | -0.169 [-0.200, -0.138] | 0.898 / 0.788 | +0.111 [+0.087, +0.133] | better / better | - | 0.85 / 1.67 (3.72) |
| d 50, 1,000, degree 4 | 20 | 0.589 / 0.216 | +0.373 [+0.332, +0.412] | 0.557 / 0.860 | -0.303 [-0.335, -0.270] | worse / worse | - | 6.10 / 2.10 (8.52) |
| d 50, 5,000, degree 2 | 20 | 0.101 / 0.309 | -0.208 [-0.231, -0.186] | 0.934 / 0.793 | +0.140 [+0.126, +0.156] | better / better | - | 0.56 / 1.59 (3.72) |
| d 50, 5,000, degree 4 | 20 | 0.551 / 0.159 | +0.393 [+0.352, +0.433] | 0.591 / 0.902 | -0.312 [-0.345, -0.278] | worse / worse | - | 5.96 / 1.38 (8.52) |

#### `hiton_mb` against inter_iamb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 10 | 0.526 / 0.529 | -0.003 [-0.037, +0.027] | 0.592 / 0.615 | -0.023 [-0.046, +0.003] | tie / tie | - | 2.78 / 2.50 (3.51) |
| alarm, 500 | 10 | 0.340 / 0.429 | -0.089 [-0.125, -0.053] | 0.740 / 0.701 | +0.039 [+0.011, +0.064] | better / better | - | 1.64 / 2.19 (3.51) |
| alarm, 5,000 | 10 | 0.115 / 0.344 | -0.229 [-0.254, -0.203] | 0.918 / 0.767 | +0.151 [+0.133, +0.170] | better / better | - | 0.57 / 1.77 (3.51) |
| child, 200 | 10 | 0.466 / 0.394 | +0.071 [+0.015, +0.117] | 0.637 / 0.718 | -0.081 [-0.121, -0.035] | worse / worse | - | 2.43 / 1.65 (3.00) |
| child, 500 | 10 | 0.278 / 0.282 | -0.004 [-0.040, +0.025] | 0.803 / 0.809 | -0.006 [-0.029, +0.022] | tie / tie | - | 1.18 / 1.25 (3.00) |
| child, 5,000 | 10 | 0.015 / 0.215 | -0.200 [-0.227, -0.170] | 0.991 / 0.857 | +0.134 [+0.115, +0.154] | better / better | - | 0.07 / 0.85 (3.00) |
| hailfinder, 200 | 10 | 0.815 / 0.725 | +0.090 [+0.014, +0.161] | 0.405 / 0.472 | -0.068 [-0.118, -0.015] | worse / worse | - | 5.32 / 3.17 (3.54) |
| hailfinder, 500 | 10 | 0.613 / 0.626 | -0.013 [-0.095, +0.066] | 0.501 / 0.545 | -0.044 [-0.101, +0.015] | tie / tie | - | 4.75 / 2.89 (3.54) |
| hailfinder, 5,000 | 10 | 0.603 / 0.616 | -0.013 [-0.081, +0.050] | 0.500 / 0.547 | -0.047 [-0.088, -0.003] | tie / worse | - | 2.03 / 3.20 (3.54) |
| insurance, 200 | 10 | 0.649 / 0.621 | +0.028 [-0.006, +0.062] | 0.514 / 0.546 | -0.032 [-0.058, -0.004] | tie / worse | - | 4.29 / 3.64 (5.19) |
| insurance, 500 | 10 | 0.537 / 0.537 | +0.001 [-0.024, +0.030] | 0.600 / 0.620 | -0.020 [-0.042, -0.000] | tie / worse | - | 3.04 / 3.16 (5.19) |
| insurance, 5,000 | 10 | 0.354 / 0.358 | -0.004 [-0.040, +0.029] | 0.749 / 0.755 | -0.006 [-0.033, +0.024] | tie / tie | - | 1.93 / 2.29 (5.19) |
| d 100, 1,000, degree 2 | 20 | 0.178 / 0.498 | -0.320 [-0.341, -0.298] | 0.879 / 0.640 | +0.239 [+0.223, +0.255] | better / better | - | 1.00 / 3.56 (3.88) |
| d 100, 5,000, degree 2 | 20 | 0.133 / 0.489 | -0.356 [-0.373, -0.337] | 0.912 / 0.648 | +0.263 [+0.250, +0.276] | better / better | - | 0.69 / 3.43 (3.88) |
| d 20, 1,000, degree 2 | 20 | 0.128 / 0.153 | -0.025 [-0.071, +0.027] | 0.915 / 0.903 | +0.012 [-0.025, +0.045] | tie / tie | - | 0.63 / 0.59 (3.35) |
| d 20, 5,000, degree 2 | 20 | 0.082 / 0.153 | -0.072 [-0.108, -0.030] | 0.947 / 0.902 | +0.045 [+0.017, +0.069] | better / better | - | 0.39 / 0.61 (3.35) |
| d 50, 1,000, degree 2 | 20 | 0.149 / 0.318 | -0.169 [-0.200, -0.138] | 0.898 / 0.788 | +0.111 [+0.087, +0.133] | better / better | - | 0.85 / 1.67 (3.72) |
| d 50, 1,000, degree 4 | 20 | 0.589 / 0.214 | +0.375 [+0.335, +0.414] | 0.557 / 0.861 | -0.304 [-0.336, -0.273] | worse / worse | - | 6.10 / 2.08 (8.52) |
| d 50, 5,000, degree 2 | 20 | 0.101 / 0.309 | -0.208 [-0.231, -0.186] | 0.934 / 0.793 | +0.140 [+0.126, +0.156] | better / better | - | 0.56 / 1.59 (3.72) |
| d 50, 5,000, degree 4 | 20 | 0.551 / 0.159 | +0.393 [+0.352, +0.433] | 0.591 / 0.903 | -0.312 [-0.345, -0.278] | worse / worse | - | 5.96 / 1.38 (8.52) |

#### `hiton_mb` against gs

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 10 | 0.526 / 0.533 | -0.007 [-0.047, +0.029] | 0.592 / 0.611 | -0.019 [-0.047, +0.013] | tie / tie | - | 2.78 / 2.53 (3.51) |
| alarm, 500 | 10 | 0.340 / 0.439 | -0.099 [-0.134, -0.067] | 0.740 / 0.693 | +0.047 [+0.023, +0.072] | better / better | - | 1.64 / 2.23 (3.51) |
| alarm, 5,000 | 10 | 0.115 / 0.344 | -0.229 [-0.258, -0.203] | 0.918 / 0.763 | +0.155 [+0.137, +0.176] | better / better | - | 0.57 / 1.82 (3.51) |
| child, 200 | 10 | 0.466 / 0.392 | +0.074 [+0.023, +0.121] | 0.637 / 0.719 | -0.083 [-0.126, -0.038] | worse / worse | - | 2.43 / 1.63 (3.00) |
| child, 500 | 10 | 0.278 / 0.294 | -0.016 [-0.041, +0.007] | 0.803 / 0.800 | +0.003 [-0.014, +0.022] | tie / tie | - | 1.18 / 1.31 (3.00) |
| child, 5,000 | 10 | 0.015 / 0.205 | -0.190 [-0.221, -0.159] | 0.991 / 0.863 | +0.128 [+0.107, +0.148] | better / better | - | 0.07 / 0.85 (3.00) |
| hailfinder, 200 | 10 | 0.815 / 0.745 | +0.070 [-0.009, +0.145] | 0.405 / 0.456 | -0.051 [-0.104, +0.004] | tie / tie | - | 5.32 / 3.20 (3.54) |
| hailfinder, 500 | 10 | 0.613 / 0.651 | -0.038 [-0.126, +0.045] | 0.501 / 0.526 | -0.025 [-0.085, +0.038] | tie / tie | - | 4.75 / 2.96 (3.54) |
| hailfinder, 5,000 | 10 | 0.603 / 0.570 | +0.033 [-0.027, +0.090] | 0.500 / 0.580 | -0.080 [-0.117, -0.041] | tie / worse | - | 2.03 / 2.82 (3.54) |
| insurance, 200 | 10 | 0.649 / 0.619 | +0.030 [+0.006, +0.055] | 0.514 / 0.547 | -0.033 [-0.054, -0.013] | worse / worse | - | 4.29 / 3.55 (5.19) |
| insurance, 500 | 10 | 0.537 / 0.533 | +0.005 [-0.019, +0.032] | 0.600 / 0.622 | -0.022 [-0.042, -0.004] | tie / worse | - | 3.04 / 3.12 (5.19) |
| insurance, 5,000 | 10 | 0.354 / 0.362 | -0.008 [-0.039, +0.022] | 0.749 / 0.751 | -0.002 [-0.028, +0.025] | tie / tie | - | 1.93 / 2.27 (5.19) |
| d 100, 1,000, degree 2 | 20 | 0.178 / 0.495 | -0.318 [-0.340, -0.296] | 0.879 / 0.642 | +0.238 [+0.221, +0.254] | better / better | - | 1.00 / 3.55 (3.88) |
| d 100, 5,000, degree 2 | 20 | 0.133 / 0.489 | -0.356 [-0.374, -0.338] | 0.912 / 0.647 | +0.265 [+0.250, +0.278] | better / better | - | 0.69 / 3.49 (3.88) |
| d 20, 1,000, degree 2 | 20 | 0.128 / 0.145 | -0.018 [-0.063, +0.033] | 0.915 / 0.907 | +0.008 [-0.028, +0.041] | tie / tie | - | 0.63 / 0.58 (3.35) |
| d 20, 5,000, degree 2 | 20 | 0.082 / 0.150 | -0.069 [-0.104, -0.027] | 0.947 / 0.903 | +0.044 [+0.016, +0.067] | better / better | - | 0.39 / 0.60 (3.35) |
| d 50, 1,000, degree 2 | 20 | 0.149 / 0.316 | -0.168 [-0.199, -0.137] | 0.898 / 0.788 | +0.110 [+0.087, +0.133] | better / better | - | 0.85 / 1.71 (3.72) |
| d 50, 1,000, degree 4 | 20 | 0.589 / 0.188 | +0.401 [+0.360, +0.439] | 0.557 / 0.879 | -0.322 [-0.354, -0.288] | worse / worse | - | 6.10 / 1.85 (8.52) |
| d 50, 5,000, degree 2 | 20 | 0.101 / 0.311 | -0.211 [-0.235, -0.188] | 0.934 / 0.791 | +0.142 [+0.127, +0.159] | better / better | - | 0.56 / 1.63 (3.72) |
| d 50, 5,000, degree 4 | 20 | 0.551 / 0.157 | +0.395 [+0.354, +0.436] | 0.591 / 0.903 | -0.312 [-0.345, -0.279] | worse / worse | - | 5.96 / 1.45 (8.52) |

#### `hiton_mb` against pcmb

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 10 | 0.526 / 0.449 | +0.077 [+0.054, +0.096] | 0.592 / 0.643 | -0.051 [-0.065, -0.031] | worse / worse | - | 2.78 / 1.90 (3.51) |
| alarm, 500 | 10 | 0.340 / 0.326 | +0.013 [+0.001, +0.025] | 0.740 / 0.746 | -0.006 [-0.018, +0.005] | worse / tie | - | 1.64 / 1.52 (3.51) |
| alarm, 5,000 | 10 | 0.115 / 0.116 | -0.001 [-0.006, +0.004] | 0.918 / 0.917 | +0.001 [-0.001, +0.003] | tie / tie | - | 0.57 / 0.59 (3.51) |
| child, 200 | 10 | 0.466 / 0.393 | +0.072 [+0.036, +0.107] | 0.637 / 0.706 | -0.069 [-0.095, -0.042] | worse / worse | - | 2.43 / 1.57 (3.00) |
| child, 500 | 10 | 0.278 / 0.274 | +0.004 [-0.015, +0.027] | 0.803 / 0.804 | -0.001 [-0.017, +0.014] | tie / tie | - | 1.18 / 1.11 (3.00) |
| child, 5,000 | 10 | 0.015 / 0.024 | -0.009 [-0.024, +0.005] | 0.991 / 0.983 | +0.008 [-0.002, +0.019] | tie / tie | - | 0.07 / 0.10 (3.00) |
| hailfinder, 200 | 10 | 0.815 / 0.754 | +0.061 [+0.044, +0.077] | 0.405 / 0.416 | -0.011 [-0.022, -0.001] | worse / worse | - | 5.32 / 5.05 (3.54) |
| hailfinder, 500 | 10 | 0.613 / 0.590 | +0.023 [+0.012, +0.034] | 0.501 / 0.527 | -0.027 [-0.035, -0.019] | worse / worse | - | 4.75 / 4.03 (3.54) |
| hailfinder, 5,000 | 10 | 0.603 / 0.618 | -0.016 [-0.023, -0.007] | 0.500 / 0.497 | +0.003 [+0.000, +0.006] | better / better | - | 2.03 / 2.13 (3.54) |
| insurance, 200 | 10 | 0.649 / 0.602 | +0.047 [+0.024, +0.072] | 0.514 / 0.532 | -0.018 [-0.038, +0.001] | worse / tie | - | 4.29 / 3.35 (5.19) |
| insurance, 500 | 10 | 0.537 / 0.542 | -0.004 [-0.031, +0.024] | 0.600 / 0.587 | +0.013 [-0.009, +0.036] | tie / tie | - | 3.04 / 2.90 (5.19) |
| insurance, 5,000 | 10 | 0.354 / 0.343 | +0.011 [-0.011, +0.034] | 0.749 / 0.755 | -0.005 [-0.025, +0.015] | tie / tie | - | 1.93 / 1.87 (5.19) |
| d 100, 1,000, degree 2 | 20 | 0.178 / 0.182 | -0.004 [-0.010, +0.001] | 0.879 / 0.877 | +0.003 [-0.001, +0.007] | tie / tie | - | 1.00 / 1.02 (3.88) |
| d 100, 5,000, degree 2 | 20 | 0.133 / 0.135 | -0.002 [-0.005, +0.001] | 0.912 / 0.910 | +0.001 [-0.001, +0.004] | tie / tie | - | 0.69 / 0.71 (3.88) |
| d 20, 1,000, degree 2 | 20 | 0.128 / 0.123 | +0.005 [-0.002, +0.015] | 0.915 / 0.919 | -0.004 [-0.011, +0.002] | tie / tie | - | 0.63 / 0.62 (3.35) |
| d 20, 5,000, degree 2 | 20 | 0.082 / 0.075 | +0.007 [-0.006, +0.021] | 0.947 / 0.952 | -0.005 [-0.015, +0.004] | tie / tie | - | 0.39 / 0.35 (3.35) |
| d 50, 1,000, degree 2 | 20 | 0.149 / 0.151 | -0.002 [-0.011, +0.007] | 0.898 / 0.896 | +0.002 [-0.004, +0.008] | tie / tie | - | 0.85 / 0.86 (3.72) |
| d 50, 1,000, degree 4 | 20 | 0.589 / 0.580 | +0.009 [-0.003, +0.021] | 0.557 / 0.561 | -0.004 [-0.014, +0.005] | tie / tie | - | 6.10 / 5.94 (8.52) |
| d 50, 5,000, degree 2 | 20 | 0.101 / 0.100 | +0.000 [-0.009, +0.010] | 0.934 / 0.935 | -0.001 [-0.008, +0.005] | tie / tie | - | 0.56 / 0.54 (3.72) |
| d 50, 5,000, degree 4 | 20 | 0.551 / 0.529 | +0.023 [+0.007, +0.039] | 0.591 / 0.607 | -0.016 [-0.029, -0.004] | worse / worse | - | 5.96 / 5.60 (8.52) |

#### `hiton_mb` with `degrees_of_freedom="nonempty"` against Causal Explorer HITON_MB

| Network, rows | Datasets | Distance | D (95%) | F1 | D (95%) | Seeds 0-19 | Seeds 20-39 | Errors (empty) |
|---|---:|---|---|---|---|---|---|---|
| alarm, 200 | 20 | 0.508 / 0.592 | -0.084 [-0.104, -0.063] | 0.604 / 0.552 | +0.051 [+0.033, +0.070] | better / better | - | 2.68 / 4.63 (3.51) |
| alarm, 500 | 20 | 0.314 / 0.347 | -0.033 [-0.050, -0.016] | 0.755 / 0.753 | +0.002 [-0.010, +0.014] | better / tie | - | 1.52 / 2.15 (3.51) |
| alarm, 5,000 | 20 | 0.109 / 0.196 | -0.087 [-0.098, -0.077] | 0.922 / 0.869 | +0.053 [+0.046, +0.060] | better / better | - | 0.55 / 0.97 (3.51) |
| child, 200 | 20 | 0.444 / 0.632 | -0.188 [-0.207, -0.169] | 0.664 / 0.505 | +0.158 [+0.141, +0.175] | better / better | - | 2.41 / 5.64 (3.00) |
| child, 500 | 20 | 0.298 / 0.383 | -0.085 [-0.107, -0.062] | 0.789 / 0.726 | +0.063 [+0.044, +0.080] | better / better | - | 1.28 / 2.52 (3.00) |
| child, 5,000 | 20 | 0.016 / 0.142 | -0.126 [-0.141, -0.111] | 0.990 / 0.906 | +0.084 [+0.074, +0.095] | better / better | - | 0.06 / 0.44 (3.01) |
| hailfinder, 200 | 20 | 0.683 / 0.837 | -0.153 [-0.167, -0.139] | 0.491 / 0.315 | +0.176 [+0.164, +0.188] | better / better | - | 4.34 / 15.65 (3.72) |
| hailfinder, 500 | 20 | 0.619 / 0.653 | -0.034 [-0.046, -0.023] | 0.498 / 0.469 | +0.030 [+0.021, +0.039] | better / better | - | 4.75 / 9.15 (3.26) |
| hailfinder, 5,000 | 20 | 0.623 / 0.525 | +0.098 [+0.089, +0.107] | 0.496 / 0.576 | -0.080 [-0.085, -0.075] | worse / worse | - | 2.17 / 6.99 (3.52) |
| insurance, 200 | 20 | 0.648 / 0.703 | -0.055 [-0.070, -0.040] | 0.517 / 0.461 | +0.057 [+0.042, +0.072] | better / better | - | 4.33 / 8.16 (5.19) |
| insurance, 500 | 20 | 0.536 / 0.563 | -0.027 [-0.041, -0.013] | 0.598 / 0.592 | +0.006 [-0.006, +0.018] | better / tie | - | 3.04 / 4.15 (5.18) |
| insurance, 5,000 | 20 | 0.339 / 0.318 | +0.021 [+0.006, +0.036] | 0.760 / 0.787 | -0.027 [-0.040, -0.014] | worse / worse | - | 1.85 / 1.95 (5.20) |

Each table gives Andrey's value / the other's, D with its 95% interval, the verdicts on distance /
F1, and the errors per target, missing plus extra members, of Andrey / the other (the empty set's).

## Run

Set up the toolbox as [The harness](../README.md) describes for Causal Explorer, with `matlab` on
the `PATH`. Each arm is one run of the driver:

```shell
B="benchmarks/blanket_cpu.py --mode target --drop-constant --discrete-test plain --warmup 0 --cores 1"
B="$B --first-seed 0 --seeds 20"
NETS=$(for x in child insurance alarm hailfinder; do
  for n in 200 500 5000; do echo -n "net:$x:$n "; done; done)
GAUSS="gauss:20:1000:2 gauss:20:5000:2 gauss:50:1000:2 gauss:50:5000:2 gauss:100:1000:2
  gauss:100:5000:2 gauss:50:1000:4 gauss:50:5000:4"
$ANDREY_BENCH_PYTHON $B --out out/hmb --cells $NETS --max-depth 3 \
  --solutions andrey.hiton_mb,causal-explorer.hiton-mb,andrey.mmmb
$ANDREY_BENCH_PYTHON $B --out out/hmb-pigs --cells net:pigs:500 --targets 50 --max-depth 3 \
  --solutions andrey.hiton_mb,causal-explorer.hiton-mb
$ANDREY_BENCH_PYTHON $B --out out/gh --cells $GAUSS --max-depth 3 --solutions andrey.hiton_mb,andrey.mmmb
$ANDREY_BENCH_PYTHON benchmarks/blanket_summary.py 'hmb=out/hmb*' 'gh=out/gh*' --json cells.json
```

The driver's defaults supply the rest: `--indep-test gsq`, `--alpha 0.05`, and `--cap-wall-s` of
1800 (3,600 on pigs). A rerun is the same command with `--first-seed 20` on the cells to rerun, into
a folder its arm's pattern matches; the context arms are `'ctx=out/ctx*'` and `'gctx=out/gctx*'`
with `--max-depth none` and the solutions `andrey.iamb,andrey.inter_iamb,andrey.gs,andrey.pcmb`.
Times will differ with the hardware; the sets should match closely.
