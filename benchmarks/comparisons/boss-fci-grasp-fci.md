# BOSS-FCI and GRaSP-FCI

`boss_grasp_fci_cpu.py` compares Andrey's BOSS-FCI and GRaSP-FCI with Tetrad's, with the empty
graph as the floor. The claim: on the paper's simulation, linear-Gaussian data with hidden common
causes, at the paper's settings and at Andrey's defaults, Andrey's PAGs make no more endpoint errors
than Tetrad's at its documented defaults.

## The model

BOSS-FCI and GRaSP-FCI (Ramsey, Andrews, and Spirtes 2025) are GFCI with BOSS's or GRaSP's CPDAG in
place of FGES's: conditional-independence tests remove the adjacencies hidden common causes
explain, the CPDAG decides the colliders, and FCI's rules orient the PAG. The paper proves them
correct with GFCI's possible-d-separation step and conjectures the step unneeded. Its simulation
(Sec. 8.1.1) draws linear-Gaussian SEMs on random forward DAGs over 20 measured variables plus 0, 4,
or 8 hidden ones, at mean degree 2, 4, or 6 and 200 to 5,000 rows, with coefficients uniform on
[-1, 1], noise variances uniform on [1, 3], no standardization, and the columns shuffled; 20 runs
per cell, at BIC penalty discount 2.0 and Fisher z alpha 0.01. A larger cell has 100 measured
variables, 10 hidden, degree 6, and 1,000 rows, with conditioning sets of at most 7.

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

| | Andrey | causal-cmd 1.12.0 (tetrad-lib 7.6.3), gated | tetrad-lib 7.6.10, context |
|---|---|---|---|
| BOSS-FCI | `boss_fci(score_func="bic", indep_test="fisherz", seed=0)` | `bfci` | `BossFci` |
| GRaSP-FCI | `grasp_fci(...)`, `grasp_depth=3` | `grasp-fci`, `depth` as GRaSP's depth | `GraspFci` |
| depth 7 | `max_depth=7` | `depth 7` | `depth 7` |
| runs per dataset | 1 | 5, no seed | 1 |

Tetrad runs its documented defaults (`parameter.definitions.html` in each jar): one start, no BES,
the complete rule set, no cap on discriminating paths, no structure prior, and no seed.
causal-cmd sets a boolean it is not given to false rather than to its default, so the adapter names
each one that is on by default (`completeRuleSetUsed`, `doDiscriminatingPathRule`,
`precomputeCovariances`, `graspOrderedAlg`, `useDataOrder`). The paper ran Tetrad's own
algorithm interface, which sets GRaSP-FCI's depth from `depth`, -1 by default; causal-cmd sets
`graspDepth` and then `depth`, so `depth` wins there too, and it caps GRaSP's tuck depth and the FCI
steps' separating sets alike.

The gated reference is causal-cmd 1.12.0's. tetrad-lib 7.6.10, the paper's code, orients with rules
R5 and R9 that read a covered path as uncovered, so they can put a tail where no Markov-equivalent
MAG has one; tetrad-lib 7.6.3 does not have that code. 7.6.10's hybrids run once per dataset at the
paper's settings, as context, and its tails count against it in the scores below. Tetrad's
searches take no seed by default, and 7.6.10's removes edges in an unseeded shuffled order, so the
gated reference runs five times per dataset with no seed; its errors are the mean of the five, and
the summary counts the datasets whose runs differ.

The implementations differ in parts each adapter names and the lane measures rather than matches:

| part | Andrey | Tetrad |
|---|---|---|
| possible-d-separation | kept: GFCI's step, under which the paper proves the method correct | none, in both releases |
| removing adjacencies | FCI's adjacency search from the CPDAG's adjacencies, level by level | one pass of greedy separating sets over the CPDAG's edges (7.6.3); one pass in a shuffled order (7.6.10) |
| BOSS's or GRaSP's start | one start from the columns' ranks | one start from the column order |
| GRaSP's tuck depth | 3, independent of `max_depth` | `depth`, -1 (none) by default, the FCI cap too |

Context arms at the paper's settings measure each part: Andrey's hybrids with the
possible-d-separation step's removals skipped (`no_possible_dsep`) on P20, P100, O1, and O2,
Andrey's GRaSP-FCI at a tuck depth of 7 on P20, P100, and O1, and GFCI and FCI from both packages
on P20. A tuck depth of `d`,
nearer Tetrad's, runs past minutes at 200 rows, where many scores tie, so the lane stops at 7.

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
| `possible_dsep_max` | the most candidates any pair's possible-d-separation search had (Andrey) |

A pair is one dataset fitted by both packages. D is Andrey's SHD minus the mean of Tetrad's five
runs, and its interval the percentile bootstrap 95% interval of the mean D over the pairs (10,000
resamples). A row in claim is worse when the interval lies above 0, better when it lies below, and
a tie otherwise; a row with fewer than 10 pairs is reported, not judged. A worse or better row is
run again on seeds 100-119 and keeps its verdict only if that interval lies on the same side of
0. The floor applies where Tetrad's SHD lies below the empty graph's on seeds 0-19. A fit stopped
at its cap, 1,800 s (3,600 s at 100 variables and out of claim), is a timeout: it is counted in
the coverage and never scored.

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
`--degrees`, and `--latents` cut a campaign into small runs, `--seeds 100-119` reruns cells on fresh
seeds, and `--resume` continues a run. `check-truth` needs R with pcalg (see the README). Tetrad's
time is its search inside the JVM (`wall_s`); the whole call, with the JVM's start, is `call_s`.

## References

- Ramsey, J., Andrews, B., Spirtes, P. (2025). Efficient latent variable causal discovery:
  combining score search and targeted testing. arXiv:2510.04263.
- Ogarrio, J. M., Spirtes, P., Ramsey, J. (2016). A hybrid causal search algorithm for latent
  variable models. PGM, PMLR 52, 368-379.
- Zhang, J. (2008). On the completeness of orientation rules for causal discovery in the presence
  of latent confounders and selection bias. Artificial Intelligence 172(16-17), 1873-1896.
