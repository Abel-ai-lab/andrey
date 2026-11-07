# BOSS-FCI and GRaSP-FCI

`boss_grasp_fci_cpu.py` compares Andrey's BOSS-FCI and GRaSP-FCI with Tetrad's, with the empty graph
as the floor. The claim: on the paper's simulation, linear-Gaussian data with hidden common causes,
at the paper's settings and at Andrey's defaults, Andrey's PAGs make no more endpoint errors than
those of tetrad-lib 7.6.10, the authors' current implementation, at Tetrad's documented defaults.

## The model

BOSS-FCI and GRaSP-FCI (Ramsey, Andrews, and Spirtes 2025) are GFCI with BOSS's or GRaSP's CPDAG in
place of FGES's: conditional-independence tests remove the adjacencies hidden common causes explain,
the CPDAG decides the colliders, and FCI's rules orient the PAG. The paper proves them correct with
GFCI's possible-d-separation step and conjectures the step unneeded, and its Algorithm 5 leaves the
step out and tests every edge of the CPDAG; Andrey's hybrids and tetrad-lib 7.6.10's follow
Algorithm 5. tetrad-lib 7.6.3 leaves the step out too, but tests only the edges that lie in a
triangle of the CPDAG, the rule of the GFCI paper (Ogarrio, Spirtes, and Ramsey 2016, Sec. 5). The
BOSS-FCI paper's simulation (Sec. 8.1.1) draws linear-Gaussian SEMs on random forward DAGs over 20
measured variables plus 0, 4, or 8 hidden ones, at mean degree 2, 4, or 6 and 200 to 5,000 rows,
with coefficients uniform on [-1, 1], noise variances uniform on [1, 3], no standardization, and the
columns shuffled; 20 runs per cell, at BIC penalty discount 2.0 and Fisher z alpha 0.01. A larger
cell has 100 measured variables, 10 hidden, degree 6, and 1,000 rows, with conditioning sets of at
most 7.

The driver draws the same data with `andrey.data`: a DAG over `d + L` nodes in a random forward
order with exactly `degree * d / 2` edges, as Tetrad's `RandomForward` draws, and the SEM as its
`SemSimulation` does. The hidden nodes are drawn among the nodes with two or more children. Where
fewer than `L` nodes have two children, Tetrad hides only those, and so does the driver: at degree
2 with 8 hidden, a graph hides 4.7 on average. Each dataset records the number hidden.

| block | measured + hidden | mean degree | rows | settings | seeds | claim |
|---|---|---|---|---|---|---|
| P20 | 20 + 0, 4, or 8 | 2, 4, 6 | 200, 500, 1,000, 5,000 | the paper's and Andrey's defaults | 0-19 | in |
| P100 | 100 + 10, depth 7 | 6 | 1,000 | the paper's | 0-19 | in |
| O1 | 100 + 10, depth 7 | 10 | 1,000 | the paper's | 0-9 | out |
| O2 | 200 + 20, depth 7 | 6 | 1,000 | the paper's | 0, time only | out |
| O3 | `sample_scm(d=24, latents=4, density=4)` | 4 | 1,000 | both | 0-9 | out |

P20 and P100 are the paper's setting, 73 rows. The paper gives degree 6 at 100 variables in its
setup and degree 10 in its results; O1 runs the second. O2 is the paper's 200-variable test of
speed, and O3 the setting of Andrey's recovery test for both hybrids, standardized data on
`sample_scm`'s defaults.

The truth is the PAG of the measured variables, `andrey.data.latent.marginal(graph, "pag")`.
`check-truth` compares it with pcalg's `dag2pag` on every distinct graph of up to 30 measured
variables, with the DAG renumbered in a topological order, which `dag2pag` needs (its ancestor
search reads the parents among lower-numbered nodes only).

## Settings

Both packages run each row's penalty discount and alpha: the paper's 2.0 and 0.01, or Andrey's
defaults, 1.0 and 0.05.

| | Andrey | tetrad-lib 7.6.10, gated | causal-cmd 1.12.0 (tetrad-lib 7.6.3), context |
|---|---|---|---|
| BOSS-FCI | `boss_fci(score_func="bic", indep_test="fisherz", seed=0)` | `BossFci` | `bfci` |
| GRaSP-FCI | `grasp_fci(...)`, `grasp_depth=3` | `GraspFci` | `grasp-fci`, `depth` as GRaSP's depth |
| depth 7 | `max_depth=7` | `depth 7` | `depth 7` |
| runs per dataset | 1 | 5, no seed | 5, no seed |

Tetrad runs its documented defaults (`parameter.definitions.html` in each jar): one start, no BES,
the complete rule set, no cap on discriminating paths, no structure prior, and no seed.
causal-cmd sets a boolean it is not given to false rather than to its default, so the adapter names
each one that is on by default (`completeRuleSetUsed`, `doDiscriminatingPathRule`,
`precomputeCovariances`, `graspOrderedAlg`, `useDataOrder`). The paper ran Tetrad's own
algorithm interface, which sets GRaSP-FCI's depth from `depth`, -1 by default; causal-cmd sets
`graspDepth` and then `depth`, so `depth` wins there too, and it caps GRaSP's tuck depth and the FCI
steps' separating sets alike.

The gated reference is tetrad-lib 7.6.10's `BossFci` and `GraspFci`, the paper's code, which removes
edges as Algorithm 5 does. causal-cmd 1.12.0's `bfci` and `grasp-fci` (tetrad-lib 7.6.3) run as
context, at the same settings. 7.6.10 orients with rules R5 and R9 that read a covered path as
uncovered, so they can put a tail where no Markov-equivalent MAG has one, a defect of the reference;
7.6.3 does not have that code. Its tails count against it in the scores below. Tetrad's searches
take no seed by default, and 7.6.10's removes edges in an unseeded shuffled order, so each Tetrad
hybrid runs five times per dataset with no seed (once on O2); its errors are the mean of the five,
and the summary counts the datasets whose runs differ.

The implementations differ in parts each adapter names and the lane measures rather than matches:

| part | Andrey | Tetrad |
|---|---|---|
| removing adjacencies | FCI's adjacency search from the CPDAG's adjacencies, level by level | every edge of the CPDAG, one pass in a shuffled order, at the first separating subset of either end's neighbors (7.6.10); only the edges in a triangle of the CPDAG (7.6.3) |
| BOSS's or GRaSP's start | one start from the columns' ranks | one start from the column order |
| GRaSP's tuck depth | 3, independent of `max_depth` | `depth`, -1 (none) by default, the FCI cap too |

Context arms at the paper's settings measure each part: Andrey's GRaSP-FCI at a tuck depth of 7 on
P20, P100, and O1, and GFCI and FCI from both packages on P20. A tuck depth of `d`, nearer Tetrad's,
runs past minutes at 200 rows, where many scores tie, so the lane stops at 7.

## Scores

Every estimate is a PAG over the measured variables, scored against the true PAG:

| column | what it scores |
|---|---|
| `shd` | the endpoint SHD, the verdict |
| `empty_shd` | the empty graph's endpoint SHD, the floor |
| `adjacency_precision`, `adjacency_recall` | the pairs joined |
| `arrowhead_*`, `tail_*` | the endpoints with that mark: precision, recall, and F1 |
| `arrowhead_common_*` | the arrowheads on the pairs both graphs join |
| `arrow_path_precision` | the estimated `x *-> y` with no directed path from `y` to `x` in the DAG |
| `tail_path_precision` | the estimated `x -> y` with a directed path from `x` to `y` |
| `bidirected_latent_precision` | the estimated `x <-> y` with a hidden parent of both |
| `possible_dsep_max` | the most candidates any pair's possible-d-separation search had (Andrey's GFCI and FCI) |

A pair is one dataset fitted by both packages. D is Andrey's SHD minus the mean of Tetrad's five
runs, and its interval the percentile bootstrap 95% interval of the mean D over the pairs (10,000
resamples). A row in claim is worse when the interval lies above 0, better when it lies below, and a
tie otherwise; a row with fewer than 10 pairs is reported, not judged. A worse or better row is run
again on seeds 100-119 and keeps its verdict only if that interval lies on the same side of 0. The
floor applies where 7.6.10's SHD lies below the empty graph's on the first seeds. A fit stopped at
its cap, 1,800 s (3,600 s at 100 variables and out of claim), is a timeout: it is counted in the
coverage and never scored.

## Results

Andrey's pre-release build of 7 November 2026, from the 0.15.0 release branch, against tetrad-lib
7.6.10 and causal-cmd 1.12.0 (tetrad-lib 7.6.3) on Temurin JDK 21.0.12.1, one fit at a time on one
core of an AMD EPYC 7713 or 9534 processor (an Intel Xeon Gold 6542Y for O1) with one BLAS thread.
causal-cmd's runs come from the lane's first campaign, on the same datasets. Every number below is
in [`published/0.15.0/boss-fci-grasp-fci.json`](../published/0.15.0/boss-fci-grasp-fci.json), with
the context arms, the other scores, and every row's coverage. The true PAG agreed with pcalg's
`dag2pag` on all 330 distinct graphs of up to 30 measured variables, and every fit of every package
finished within its cap.

### In the claim

Means over the 20 datasets of each cell, seeds 0 to 19, of the endpoint SHD against the PAG of the
measured variables; 7.6.10's SHD on a dataset is the mean of its five runs. D is Andrey's SHD minus
7.6.10's, with its 95% interval, and the verdict is the one on seeds 0 to 19.

| Settings | Measured | Degree | Hidden | n | Empty | BOSS-FCI | 7.6.10 | D [95%] | | GRaSP-FCI | 7.6.10 | D [95%] | |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper's | 20 | 2 | 0 | 200 | 40 | 19 | 18.82 | +0.18 [-0.82, +1.21] | tie | 19.45 | 19.22 | +0.23 [-1.41, +1.85] | tie |
| paper's | 20 | 2 | 0 | 500 | 40 | 13.85 | 13.53 | +0.32 [-0.2, +0.97] | tie | 13.1 | 13.73 | -0.63 [-1.61, +0.14] | tie |
| paper's | 20 | 2 | 0 | 1,000 | 40 | 10.8 | 10.45 | +0.35 [-0.2, +1] | tie | 11.2 | 10.6 | +0.6 [0, +1.5] | tie |
| paper's | 20 | 2 | 0 | 5,000 | 40 | 6.6 | 6.6 | 0 [-0.3, +0.3] | tie | 6.75 | 7.05 | -0.3 [-1.3, +0.45] | tie |
| paper's | 20 | 2 | 4 | 200 | 39.8 | 28.3 | 28.26 | +0.04 [-0.48, +0.69] | tie | 28 | 28.29 | -0.29 [-0.99, +0.35] | tie |
| paper's | 20 | 2 | 4 | 500 | 39.8 | 24.1 | 23.89 | +0.21 [-0.28, +0.85] | tie | 24.35 | 23.99 | +0.36 [-0.74, +1.41] | tie |
| paper's | 20 | 2 | 4 | 1,000 | 39.8 | 21.1 | 21.35 | -0.25 [-1.25, +0.85] | tie | 21.5 | 21.1 | +0.4 [-0.5, +1.3] | tie |
| paper's | 20 | 2 | 4 | 5,000 | 39.8 | 16.75 | 16.45 | +0.3 [-1, +1.7] | tie | 15.75 | 16.45 | -0.7 [-2.1, +0.5] | tie |
| paper's | 20 | 2 | 8 | 200 | 37.8 | 27.05 | 26.7 | +0.35 [0, +1.05] | tie | 27.15 | 27.46 | -0.31 [-1.18, +0.44] | tie |
| paper's | 20 | 2 | 8 | 500 | 37.8 | 22.5 | 22.74 | -0.24 [-0.68, 0] | tie | 22.5 | 22.29 | +0.21 [-0.59, +1.1] | tie |
| paper's | 20 | 2 | 8 | 1,000 | 37.8 | 19.75 | 19.29 | +0.46 [-0.08, +1.36] | tie | 20.3 | 19.91 | +0.39 [-0.41, +1.25] | tie |
| paper's | 20 | 2 | 8 | 5,000 | 37.8 | 14.4 | 14.58 | -0.18 [-0.9, +0.52] | tie | 14.65 | 14.3 | +0.35 [-1.4, +1.7] | tie |
| paper's | 20 | 4 | 0 | 200 | 80 | 55.1 | 54.63 | +0.47 [-1.7, +2.68] | tie | 54.9 | 54.94 | -0.04 [-1.49, +1.42] | tie |
| paper's | 20 | 4 | 0 | 500 | 80 | 41.9 | 42.1 | -0.2 [-1.7, +1.22] | tie | 43.05 | 42.72 | +0.33 [-0.71, +1.33] | tie |
| paper's | 20 | 4 | 0 | 1,000 | 80 | 37.3 | 37.25 | +0.05 [-1.67, +1.49] | tie | 38.6 | 37.06 | +1.54 [+0.21, +2.94] | worse; rerun tie |
| paper's | 20 | 4 | 0 | 5,000 | 80 | 31 | 30.91 | +0.09 [-1.44, +1.51] | tie | 31.3 | 31.59 | -0.29 [-1.98, +1.32] | tie |
| paper's | 20 | 4 | 4 | 200 | 91.6 | 69 | 68.88 | +0.12 [-1.05, +1.31] | tie | 69.45 | 68.95 | +0.5 [-1.14, +2.38] | tie |
| paper's | 20 | 4 | 4 | 500 | 91.6 | 60.35 | 60.03 | +0.32 [-1.92, +2.65] | tie | 61.3 | 61.23 | +0.07 [-2.46, +2.98] | tie |
| paper's | 20 | 4 | 4 | 1,000 | 91.6 | 55.3 | 54.38 | +0.92 [-1.24, +3.28] | tie | 55.1 | 56.42 | -1.32 [-3.71, +1.27] | tie |
| paper's | 20 | 4 | 4 | 5,000 | 91.6 | 48.2 | 46.25 | +1.95 [-0.49, +4.87] | tie | 46.6 | 46.07 | +0.53 [-1.11, +2.2] | tie |
| paper's | 20 | 4 | 8 | 200 | 97.5 | 80.4 | 80.36 | +0.04 [-0.56, +0.68] | tie | 80.3 | 80.33 | -0.03 [-1.2, +1.2] | tie |
| paper's | 20 | 4 | 8 | 500 | 97.5 | 74.95 | 73.71 | +1.24 [+0.34, +2.15] | worse; rerun tie | 74.85 | 73.94 | +0.91 [-0.24, +2.23] | tie |
| paper's | 20 | 4 | 8 | 1,000 | 97.5 | 67.95 | 67.79 | +0.16 [-1.14, +1.49] | tie | 69.15 | 68.75 | +0.4 [-1.47, +2.04] | tie |
| paper's | 20 | 4 | 8 | 5,000 | 97.5 | 58.55 | 58.21 | +0.34 [-0.98, +1.78] | tie | 58.65 | 59.15 | -0.5 [-2.57, +1.7] | tie |
| paper's | 20 | 6 | 0 | 200 | 120 | 93.2 | 93.46 | -0.26 [-1.83, +1.45] | tie | 93.35 | 94.66 | -1.31 [-3.27, +0.59] | tie |
| paper's | 20 | 6 | 0 | 500 | 120 | 83.9 | 84.91 | -1.01 [-2.85, +0.84] | tie | 83.8 | 83.61 | +0.19 [-1.55, +2.36] | tie |
| paper's | 20 | 6 | 0 | 1,000 | 120 | 75.5 | 78.5 | -3 [-5.09, -1.03] | better; rerun tie | 75.85 | 78.34 | -2.49 [-4.87, -0.27] | better; rerun better |
| paper's | 20 | 6 | 0 | 5,000 | 120 | 67.4 | 69.73 | -2.33 [-4.93, -0.11] | better; rerun tie | 67.85 | 69.91 | -2.06 [-3.88, -0.49] | better; rerun tie |
| paper's | 20 | 6 | 4 | 200 | 164.4 | 140.35 | 140.56 | -0.21 [-1.99, +1.54] | tie | 141.9 | 140.56 | +1.34 [-0.76, +3.63] | tie |
| paper's | 20 | 6 | 4 | 500 | 164.4 | 132.1 | 132.45 | -0.35 [-2.1, +1.3] | tie | 132.85 | 133.01 | -0.16 [-2.09, +1.62] | tie |
| paper's | 20 | 6 | 4 | 1,000 | 164.4 | 126.35 | 127.22 | -0.87 [-2.45, +0.6] | tie | 128.5 | 126.64 | +1.86 [+0.13, +3.56] | worse; rerun tie |
| paper's | 20 | 6 | 4 | 5,000 | 164.4 | 117.4 | 118.05 | -0.65 [-2.42, +1.02] | tie | 117.1 | 118.59 | -1.49 [-3.48, +0.22] | tie |
| paper's | 20 | 6 | 8 | 200 | 191.9 | 170.1 | 169.77 | +0.33 [-1.92, +2.36] | tie | 170.15 | 169.88 | +0.27 [-0.98, +1.65] | tie |
| paper's | 20 | 6 | 8 | 500 | 191.9 | 162.4 | 160.98 | +1.42 [-0.73, +3.48] | tie | 163.9 | 162.29 | +1.61 [-0.89, +4.09] | tie |
| paper's | 20 | 6 | 8 | 1,000 | 191.9 | 157.9 | 157.61 | +0.29 [-0.77, +1.38] | tie | 157.15 | 158.74 | -1.59 [-2.97, -0.21] | better; rerun tie |
| paper's | 20 | 6 | 8 | 5,000 | 191.9 | 148.7 | 149.5 | -0.8 [-2.65, +0.95] | tie | 149.8 | 149.53 | +0.27 [-1.17, +1.65] | tie |
| paper's | 100 | 6 | 10 | 1,000 | 950.3 | 636.65 | 640.95 | -4.3 [-8.32, -0.57] | better; rerun better | 638.85 | 643.11 | -4.26 [-8.82, +0.57] | tie |
| defaults | 20 | 2 | 0 | 200 | 40 | 21.95 | 21.22 | +0.73 [-0.63, +2.14] | tie | 22.8 | 21.57 | +1.23 [-0.08, +2.63] | tie |
| defaults | 20 | 2 | 0 | 500 | 40 | 14.75 | 14.55 | +0.2 [-0.7, +1.1] | tie | 15.1 | 15.16 | -0.06 [-0.82, +0.69] | tie |
| defaults | 20 | 2 | 0 | 1,000 | 40 | 12 | 11.97 | +0.03 [-0.3, +0.36] | tie | 13.4 | 11.74 | +1.66 [+0.06, +3.76] | worse; rerun tie |
| defaults | 20 | 2 | 0 | 5,000 | 40 | 6.05 | 6.2 | -0.15 [-0.9, +0.45] | tie | 6.35 | 6.4 | -0.05 [-0.4, +0.3] | tie |
| defaults | 20 | 2 | 4 | 200 | 39.8 | 31.2 | 31.06 | +0.14 [-0.45, +0.9] | tie | 31.5 | 31.37 | +0.13 [-0.56, +0.88] | tie |
| defaults | 20 | 2 | 4 | 500 | 39.8 | 27.25 | 27.89 | -0.64 [-2.4, +0.81] | tie | 26.5 | 26.79 | -0.29 [-1.38, +0.75] | tie |
| defaults | 20 | 2 | 4 | 1,000 | 39.8 | 21.8 | 21.85 | -0.05 [-0.95, +0.65] | tie | 21.85 | 21.55 | +0.3 [-0.65, +1.25] | tie |
| defaults | 20 | 2 | 4 | 5,000 | 39.8 | 16.95 | 15.55 | +1.4 [+0.15, +2.75] | worse; rerun tie | 16.3 | 15.75 | +0.55 [-0.35, +1.65] | tie |
| defaults | 20 | 2 | 8 | 200 | 37.8 | 34.35 | 34.53 | -0.18 [-0.98, +0.6] | tie | 33.3 | 33.7 | -0.4 [-1.8, +0.85] | tie |
| defaults | 20 | 2 | 8 | 500 | 37.8 | 28.35 | 28.5 | -0.15 [-0.7, +0.45] | tie | 28.05 | 28.1 | -0.05 [-0.75, +0.7] | tie |
| defaults | 20 | 2 | 8 | 1,000 | 37.8 | 22.45 | 22.9 | -0.45 [-1.35, +0.45] | tie | 22.35 | 23 | -0.65 [-3.2, +1.45] | tie |
| defaults | 20 | 2 | 8 | 5,000 | 37.8 | 14.8 | 15.34 | -0.54 [-1.64, +0.51] | tie | 15.8 | 15.63 | +0.17 [-0.48, +0.85] | tie |
| defaults | 20 | 4 | 0 | 200 | 80 | 51.55 | 52.07 | -0.52 [-1.97, +0.92] | tie | 52.05 | 51.62 | +0.43 [-1.28, +2.08] | tie |
| defaults | 20 | 4 | 0 | 500 | 80 | 41.25 | 40.71 | +0.54 [-0.58, +1.71] | tie | 40.75 | 40.75 | 0 [-1.31, +1.52] | tie |
| defaults | 20 | 4 | 0 | 1,000 | 80 | 37.3 | 36.47 | +0.83 [-1.18, +2.76] | tie | 35.35 | 36 | -0.65 [-2.02, +0.82] | tie |
| defaults | 20 | 4 | 0 | 5,000 | 80 | 29.95 | 30.46 | -0.51 [-2.19, +1.18] | tie | 29.8 | 30.26 | -0.46 [-1.98, +1.13] | tie |
| defaults | 20 | 4 | 4 | 200 | 91.6 | 66.6 | 66.61 | -0.01 [-1.73, +1.66] | tie | 67.4 | 66.2 | +1.2 [-0.08, +2.51] | tie |
| defaults | 20 | 4 | 4 | 500 | 91.6 | 58.3 | 58.27 | +0.03 [-1.34, +1.38] | tie | 59.4 | 58.86 | +0.54 [-0.71, +1.92] | tie |
| defaults | 20 | 4 | 4 | 1,000 | 91.6 | 52.7 | 52.39 | +0.31 [-1.66, +2.16] | tie | 53.5 | 53.76 | -0.26 [-2.49, +1.8] | tie |
| defaults | 20 | 4 | 4 | 5,000 | 91.6 | 45.65 | 45.93 | -0.28 [-2.57, +2.04] | tie | 45.1 | 45.42 | -0.32 [-3.13, +2.25] | tie |
| defaults | 20 | 4 | 8 | 200 | 97.5 | 79.95 | 80.85 | -0.9 [-1.78, +0.02] | tie | 81.05 | 81.02 | +0.03 [-1.27, +1.51] | tie |
| defaults | 20 | 4 | 8 | 500 | 97.5 | 73.7 | 71.89 | +1.81 [+0.75, +2.92] | worse; rerun tie | 72.55 | 71.82 | +0.73 [-0.67, +1.97] | tie |
| defaults | 20 | 4 | 8 | 1,000 | 97.5 | 68.55 | 68.49 | +0.06 [-2.06, +2.27] | tie | 68 | 68.92 | -0.92 [-2.57, +0.66] | tie |
| defaults | 20 | 4 | 8 | 5,000 | 97.5 | 57.65 | 57.28 | +0.37 [-1.1, +1.88] | tie | 57.85 | 59.34 | -1.49 [-3.56, +0.37] | tie |
| defaults | 20 | 6 | 0 | 200 | 120 | 89.35 | 90.67 | -1.32 [-3.51, +0.86] | tie | 89.3 | 90.93 | -1.63 [-3.86, +0.53] | tie |
| defaults | 20 | 6 | 0 | 500 | 120 | 81.15 | 81.63 | -0.48 [-2.95, +1.99] | tie | 80.75 | 82.39 | -1.64 [-3.74, +0.51] | tie |
| defaults | 20 | 6 | 0 | 1,000 | 120 | 75.25 | 75.04 | +0.21 [-1.32, +1.74] | tie | 73.5 | 74.98 | -1.48 [-3.31, +0.25] | tie |
| defaults | 20 | 6 | 0 | 5,000 | 120 | 66.2 | 66.22 | -0.02 [-1.92, +1.75] | tie | 66.25 | 67.66 | -1.41 [-4.5, +1.63] | tie |
| defaults | 20 | 6 | 4 | 200 | 164.4 | 136.5 | 137.25 | -0.75 [-2.66, +0.95] | tie | 137.15 | 136.97 | +0.18 [-1.81, +2.15] | tie |
| defaults | 20 | 6 | 4 | 500 | 164.4 | 130.3 | 129.65 | +0.65 [-0.96, +2.26] | tie | 129.85 | 130.57 | -0.72 [-2.5, +0.92] | tie |
| defaults | 20 | 6 | 4 | 1,000 | 164.4 | 124.4 | 125.69 | -1.29 [-3.25, +0.68] | tie | 124.7 | 124.74 | -0.04 [-1.94, +1.78] | tie |
| defaults | 20 | 6 | 4 | 5,000 | 164.4 | 115.25 | 117.45 | -2.2 [-4.99, +0.25] | tie | 116.35 | 117.79 | -1.44 [-3.6, +0.57] | tie |
| defaults | 20 | 6 | 8 | 200 | 191.9 | 167.65 | 167.07 | +0.58 [-1.39, +2.75] | tie | 165.95 | 167.36 | -1.41 [-2.76, +0.02] | tie |
| defaults | 20 | 6 | 8 | 500 | 191.9 | 159.1 | 159.31 | -0.21 [-1.77, +1.42] | tie | 161.35 | 160.29 | +1.06 [-0.97, +3.25] | tie |
| defaults | 20 | 6 | 8 | 1,000 | 191.9 | 155.7 | 154.89 | +0.81 [-1.23, +3.05] | tie | 155.65 | 155.86 | -0.21 [-1.67, +1.08] | tie |
| defaults | 20 | 6 | 8 | 5,000 | 191.9 | 147.8 | 147.76 | +0.04 [-0.99, +1.02] | tie | 149.25 | 150.3 | -1.05 [-2.87, +0.75] | tie |

Of the 146 rows, 134 tie on seeds 0 to 19, 6 are worse, and 6 better, as many on each side, and none
by more than 4.3 errors. Rerun on seeds 100 to 119, every worse row ties, and two better rows stay
better: GRaSP-FCI at the paper's settings, degree 6 with no hidden variable and 1,000 rows, and
BOSS-FCI at 100 variables. The card counts 2 better, 144 ties, and none worse.

| Method | Settings | Measured | Degree | Hidden | n | Seeds 0-19: D [95%] | Seeds 100-119: D [95%] | Verdict |
|---|---|---|---|---|---|---|---|---|
| GRaSP-FCI | paper's | 20 | 4 | 0 | 1,000 | +1.54 [+0.21, +2.94], worse | +1.33 [-0.44, +3.3] | tie |
| BOSS-FCI | paper's | 20 | 4 | 8 | 500 | +1.24 [+0.34, +2.15], worse | +0.34 [-0.37, +1.17] | tie |
| BOSS-FCI | paper's | 20 | 6 | 0 | 1,000 | -3 [-5.09, -1.03], better | +0.55 [-1.71, +2.83] | tie |
| GRaSP-FCI | paper's | 20 | 6 | 0 | 1,000 | -2.49 [-4.87, -0.27], better | -2.32 [-4.61, -0.13] | better |
| BOSS-FCI | paper's | 20 | 6 | 0 | 5,000 | -2.33 [-4.93, -0.11], better | -0.27 [-2, +1.41] | tie |
| GRaSP-FCI | paper's | 20 | 6 | 0 | 5,000 | -2.06 [-3.88, -0.49], better | +0.15 [-1.95, +2.07] | tie |
| GRaSP-FCI | paper's | 20 | 6 | 4 | 1,000 | +1.86 [+0.13, +3.56], worse | +1.05 [-0.89, +3.01] | tie |
| GRaSP-FCI | paper's | 20 | 6 | 8 | 1,000 | -1.59 [-2.97, -0.21], better | -0.53 [-2.7, +1.31] | tie |
| BOSS-FCI | paper's | 100 | 6 | 10 | 1,000 | -4.3 [-8.32, -0.57], better | -5.01 [-7.68, -2.36] | better |
| GRaSP-FCI | defaults | 20 | 2 | 0 | 1,000 | +1.66 [+0.06, +3.76], worse | +0.45 [-0.45, +1.55] | tie |
| BOSS-FCI | defaults | 20 | 2 | 4 | 5,000 | +1.4 [+0.15, +2.75], worse | +0.21 [-0.58, +1.31] | tie |
| BOSS-FCI | defaults | 20 | 4 | 8 | 500 | +1.81 [+0.75, +2.92], worse | -0.77 [-2.47, +1.05] | tie |

Both hybrids find almost the same graphs as 7.6.10: their mean adjacency recall over the rows was
0.53 at the paper's settings and 0.57 at the defaults for both packages, with adjacency precision
0.95 to 0.996. Andrey returned the same PAG as one of 7.6.10's runs on 475 of the 1,460 datasets
with BOSS-FCI and on 390 with GRaSP-FCI.

### Floors

7.6.10's SHD was below the empty graph's in all 146 rows in the claim, so the floor applies in each,
and Andrey's was below it too: the upper end of its interval of SHD minus the empty graph's was -1.0
or lower. The closest is degree 2 with 8 hidden and 200 rows at the defaults, where the graphs hide
5.1 nodes on average: Andrey's BOSS-FCI made 34.35 errors against 7.6.10's 34.53 and the empty
graph's 37.8.

### tetrad-lib 7.6.3

causal-cmd 1.12.0's `bfci` and `grasp-fci` (tetrad-lib 7.6.3) test only the CPDAG's edges that lie
in a triangle of it, the GFCI paper's rule (Ogarrio, Spirtes, and Ramsey 2016, Sec. 5), and keep
the other edges untested. That is a different published variant from Algorithm 5: it keeps more
edges, and it scores better at the paper's strict settings and worse at Andrey's defaults on sparse
graphs. At the paper's settings and degree 4, its BOSS-FCI kept 25.9 edges on average against
Andrey's 23.5, with adjacency recall 0.59 against 0.54 and precision 0.985 against 0.997: the
edges it keeps are mostly true. At the defaults and degree 2 it kept 17.6 against 15.9, with
precision 0.825 against 0.879 and recall 0.74 against 0.72: they are mostly false. Pooled over the
hidden counts and the four sizes on P20 (means of the row means), with the verdicts of Andrey
against 7.6.3 on seeds 0 to 19, as context:

| Method | Settings | Degree | Empty | Andrey | 7.6.10 | 7.6.3 | Against 7.6.3: better, tie, worse | Adjacency recall: Andrey, 7.6.3 |
|---|---|---|---|---|---|---|---|---|
| BOSS-FCI | paper's | 2 | 39.2 | 18.68 | 18.55 | 18.11 | 1, 9, 2 | 0.66, 0.68 |
| BOSS-FCI | paper's | 4 | 89.7 | 56.67 | 56.21 | 53.6 | 0, 6, 6 | 0.54, 0.59 |
| BOSS-FCI | paper's | 6 | 158.77 | 122.94 | 123.56 | 119.3 | 0, 6, 6 | 0.38, 0.41 |
| BOSS-FCI | defaults | 2 | 39.2 | 20.99 | 20.96 | 24.02 | 11, 1, 0 | 0.72, 0.74 |
| BOSS-FCI | defaults | 4 | 89.7 | 55.26 | 55.12 | 55.9 | 1, 11, 0 | 0.58, 0.62 |
| BOSS-FCI | defaults | 6 | 158.77 | 120.72 | 121.05 | 118.89 | 0, 6, 6 | 0.41, 0.43 |
| GRaSP-FCI | paper's | 2 | 39.2 | 18.73 | 18.7 | 18.46 | 0, 12, 0 | 0.66, 0.68 |
| GRaSP-FCI | paper's | 4 | 89.7 | 56.94 | 56.76 | 54.36 | 1, 3, 8 | 0.54, 0.58 |
| GRaSP-FCI | paper's | 6 | 158.77 | 123.52 | 123.81 | 119.99 | 0, 6, 6 | 0.38, 0.41 |
| GRaSP-FCI | defaults | 2 | 39.2 | 21.11 | 20.9 | 23.47 | 7, 5, 0 | 0.72, 0.74 |
| GRaSP-FCI | defaults | 4 | 89.7 | 55.23 | 55.33 | 55.97 | 1, 11, 0 | 0.58, 0.61 |
| GRaSP-FCI | defaults | 6 | 158.77 | 120.84 | 121.65 | 119.93 | 0, 12, 0 | 0.41, 0.43 |

At the paper's settings Andrey makes more errors than 7.6.3 in 26 of the 48 rows at degree 4 and 6
(both hybrids) and fewer in 1; at 100 variables 7.6.3's mean is 583.8 for BOSS-FCI and 589.45 for
GRaSP-FCI against Andrey's 636.65 and 638.85. At the defaults on degree 2, Andrey makes fewer errors
in 18 of the 24 rows and more in none: 7.6.3's mean is 24.02 and 23.47 against Andrey's 20.99 and
21.11, and at degree 2 with 8 hidden and 200 rows 7.6.3 scores above the empty graph (38.45 and
36.7 against 37.8). 7.6.10 and Andrey, which test every edge, are within 0.9 of each other in every
row of this table.

### The reference's tails and runs

tetrad-lib 7.6.10's rules R5 and R9 read a covered path as uncovered, so they can put a tail where
no Markov-equivalent MAG has one, a defect of that release that 7.6.3 does not have. On these data
it did not show in the averages: 7.6.10's tail precision over the P20 rows was 0.57 for BOSS-FCI and
0.56 for GRaSP-FCI at the paper's settings, against Andrey's 0.54 and 0.56, and 0.51 and 0.50 at the
defaults, against 0.48 and 0.48. Its five runs per dataset returned different PAGs on 1,640 of its
3,380 datasets, from the unseeded order of its edge tests; causal-cmd's five runs never differed.

### Out of the claim

| Cell | Settings | Method | Empty | Andrey | 7.6.10 | D [95%] | 7.6.3 | D [95%] | Andrey s | 7.6.10 s |
|---|---|---|---|---|---|---|---|---|---|---|
| O1 | paper's | BOSS-FCI | 2541.2 | 2197.9 | 2218.5 | -20.6 [-27.7, -14.06] | 2115.2 | +82.7 [+69.8, +95.5] | 5.137 | 16.987 |
| O1 | paper's | GRaSP-FCI | 2541.2 | 2201.3 | 2223.12 | -21.82 [-30.48, -13.72] | 2123.1 | +78.2 [+66.3, +89.8] | 5.766 | 51.326 |
| O2 | paper's | BOSS-FCI | 1716 | 1005 | 1006 | - | 898 | - | 9.278 | 16.866 |
| O2 | paper's | GRaSP-FCI | 1716 | 998 | 981 | - | 897 | - | 9.986 | 501.8 |
| O3 | paper's | BOSS-FCI | 113.6 | 81.3 | 83.28 | -1.98 [-6.06, +1.52] | 79.8 | +1.5 [-3, +5.3] | 0.231 | 0.279 |
| O3 | paper's | GRaSP-FCI | 113.6 | 82.9 | 83.94 | -1.04 [-5.5, +3.5] | 82.7 | +0.2 [-5.6, +5.8] | 0.227 | 0.328 |
| O3 | defaults | BOSS-FCI | 113.6 | 78.8 | 82.7 | -3.9 [-7.14, -0.54] | 83.2 | -4.4 [-9.3, +0.5] | 0.297 | 0.32 |
| O3 | defaults | GRaSP-FCI | 113.6 | 79.7 | 81.08 | -1.38 [-3.42, +0.54] | 77.9 | +1.8 [-1.3, +5.3] | 0.294 | 0.397 |

At 100 measured variables with 10 hidden and mean degree 10 (O1), both hybrids make about 21 fewer
errors than 7.6.10 on 10 datasets and about 80 more than 7.6.3. O2, 200 measured variables with 20
hidden, ran one dataset for time. On O3, the setting of Andrey's recovery test, every row ties but
BOSS-FCI at the defaults, where D against 7.6.10 is -3.9 [-7.14, -0.54] on 10 datasets.

### Speed

Andrey's time is the fit from the array in memory, and Tetrad's its search inside the JVM, which
leaves out the JVM's start and the data transfer (about 15 s a call). Over the P20 rows, the median
ratio of 7.6.10's time to Andrey's on a dataset ran from 1.05 to 5.8 for BOSS-FCI and from 1.16 to
5.83 for GRaSP-FCI, with Andrey's median fit at 0.014 to 0.5 s. At 100 variables Andrey's median fit
took 1.42 s and 1.3 s against 7.6.10's 2.6 s and 11.9 s, ratios of 1.85 and 7.6; at 200 variables
9.3 s and 10 s against 16.9 s and 502 s.

## Run

```shell
export ANDREY_BENCH_JAVA=...             # a JDK 21 java
export ANDREY_BENCH_TETRAD_JAR=...       # causal-cmd-1.12.0-jar-with-dependencies.jar
export ANDREY_BENCH_TETRAD_GIN_JAR=...   # tetrad-lib-7.6.10-shaded.jar
$ANDREY_BENCH_PYTHON benchmarks/boss_grasp_fci_cpu.py run --out outputs/bgf-P20-deg4 \
    --blocks P20 --degrees 4 --workers 4
$ANDREY_BENCH_PYTHON benchmarks/boss_grasp_fci_cpu.py check-truth --runs outputs/bgf-* \
    --out outputs/bgf-truth
$ANDREY_BENCH_PYTHON benchmarks/boss_grasp_fci_cpu.py summarize --runs outputs/bgf-* \
    --out outputs/bgf-summary.json
```

`run` fits each dataset's packages back to back, one child process per package, each pinned to one
core with one BLAS thread; Tetrad starts one JVM per run, with one untimed warm-up search before the
timed one, and Andrey runs each fit once untimed on the same data first. `--blocks`, `--settings`,
`--degrees`, and `--latents` cut a campaign into small runs, `--packages` picks the packages,
`--seeds 100-119` reruns cells on fresh seeds, and `--resume` continues a run. `check-truth` needs R
with pcalg (see the README). Tetrad's time is its search inside the JVM (`wall_s`); the whole call,
with the JVM's start, is `call_s`.

## References

- Ramsey, J., Andrews, B., Spirtes, P. (2025). Efficient latent variable causal discovery:
  combining score search and targeted testing. arXiv:2510.04263v3.
- Ogarrio, J. M., Spirtes, P., Ramsey, J. (2016). A hybrid causal search algorithm for latent
  variable models. PGM, PMLR 52, 368-379.
- Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence
  of latent confounders and selection bias. Artificial Intelligence 172(16-17), 1873-1896.
