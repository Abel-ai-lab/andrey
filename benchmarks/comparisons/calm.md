# CALM

`calm_cpu.py` compares Andrey's CALM with causal-learn's, which the CALM paper's first author
contributed, with the empty-graph baseline. The paper's code repository runs the same optimizer
but has no license, so the lane runs causal-learn's copy. It runs on the canonical sparse slice of
`ges_cpu.py`: linear-Gaussian data on an Erdos-Renyi graph at average degree 2, standardized, with
`n = 10 * d` rows. Each estimate is a DAG, scored against the true DAG (`shd`) and through its
equivalence class (`mec_shd`, the score the paper reports).

Andrey runs twice. `andrey.calm.torch-cpu` uses Andrey's defaults: 3,000 Adam steps per round at
learning rate 0.01, `rho` from 1e-3, at most 25 rounds, and an edge kept only when its weight is
at least 0.3. `andrey.calm.torch-cpu.paper_schedule` uses the paper's schedule, which is
causal-learn's default: 40,000 steps per round at learning rate 0.001, `rho` from 1e-5, at most
100 rounds, and no weight threshold. Compare causal-learn with the paper-schedule solution, which
runs its settings. Every CALM solution uses `lambda1 = 0.005`, a gate temperature of 0.5,
Fisher-Z moral-graph tests at `alpha = 0.01`, `rho` three times larger each round until the
acyclicity residual is below 1e-8, and seed 0; `params` records every setting. causal-learn's
names map to Andrey's:

| Andrey | causal-learn |
|---|---|
| `lambda1`, `tau`, `rho_init`, `rho_mult`, `subproblem_iter`, `standardize` | the same names |
| `screen_alpha` | `alpha` |
| `h_tol` | `htol` |
| `lr=0.001`, `max_iter=100`, `rho_max=1e16`, `gate_threshold=0.5`, `w_threshold=0` | fixed inside `calm` |
| `seed=0` | `torch.manual_seed(0)` before the fit |

causal-learn's `calm` stops after 100 rounds whether or not its graph is acyclic; a cyclic graph
fails the conversion to a DAG and ends the fit in an error record. Andrey removes the weakest edge
of any cycle left after its last round. causal-learn also runs every step of every round, while
Andrey ends a round early once, after 50 steps, its loss changes by less than a 1e-7 fraction from
one step to the next, so the paper-schedule solution can run fewer steps than causal-learn.

The defaults are `d` of 10, 20, and 50, five seeds, one repeat without a warm-up fit, and a wall
cap of 3600 s. A smoke run fits every solution once at `d = 5`, and a run can keep to some
solutions:

```shell
$ANDREY_BENCH_PYTHON benchmarks/calm_cpu.py --out outputs/calm-smoke --smoke
$ANDREY_BENCH_PYTHON benchmarks/calm_cpu.py --out outputs/calm-cpu --cap-mem-mb 30000
$ANDREY_BENCH_PYTHON benchmarks/calm_cpu.py --out outputs/calm-andrey --ladder 100 \
    --solutions andrey.calm.torch-cpu,baseline.empty
```
