# PCMCI+ and LPCMCI on series with same-step links

`pcmci_cpu.py --method pcmciplus` and `--method lpcmci` compare Andrey's PCMCI+ and LPCMCI with
tigramite's on `pcmciplus_paper` and `lpcmci_paper`, where some links act within a time step. The
truth is a window over lags 0 to the largest true lag: the same-step DAG and the lagged links.
`lpcmci_paper` generates N series and drops 30% of them, so its rows are scored against the true
PAG over the observed series instead (`andrey_bench/window_pag.py`), the graph LPCMCI estimates
(Gerhardus and Runge, 2020, Sect. 2.3). It is built from the generating graph over lags 0 to
`tau_max`: the unobserved series and the observed ones before the window are marginalized, a link
is kept only when all its copies in the window are adjacent, and time order, the FCI rules, and
repeating orientations mark its ends. A truth with hidden series is rescored without fitting by
`rescore_window_pag.py`, which writes `window_pag.parquet`, one row per `run_id`.

Every solution tests lags 0 to `tau_max` (default 5 for PCMCI+ and 3 for LPCMCI, each paper's
largest true lag) with the Student-t partial-correlation test and the method's defaults, which are
both packages' defaults:

| solution | settings |
|---|---|
| `andrey.pcmciplus.student_t.tau<k>` | `alpha=0.01`, `collider_rule="majority"` |
| `tigramite.pcmciplus.parcorr.tau<k>` | `run_pcmciplus(tau_min=0, pc_alpha=0.01, contemp_collider_rule="majority")` |
| `andrey.lpcmci.student_t.tau<k>` | `alpha=0.05`, `n_preliminary_iterations=1` |
| `tigramite.lpcmci.parcorr.tau<k>` | `run_lpcmci(tau_min=0, pc_alpha=0.05, n_preliminary_iterations=1)` |
| `baseline.empty.window.tau<k>` | none: no link at any lag |

A link carries a mark at each end, so these solutions return window marks
(`andrey_bench/window.py`), an `(L + 1, n, n)` int8 stack: lag 0 as endpoint marks, and at each
lag `k >= 1` the mark at the earlier end of each link, whose later end is always an arrowhead. Both
packages' marks come from the same three-character link strings (`-->`, `o->`, `<->`, `o-o`,
`x-x`, ...): tigramite's `results["graph"]`, Andrey's LPCMCI `metadata["links"]`, and the same
strings built from Andrey's PCMCI+ structure and its `metadata["conflicts"]`. Two solutions return
the same graph when their marks are equal, conflicts included (`structure_hash`, `output_hash`).

A window estimate is scored link by link against the truth (`scoring.window_scores`); a link is a
same-step pair or a lagged pair, self-links included:

| columns | what they score |
|---|---|
| `endpoint_shd`, `lag0_endpoint_shd`, `lagged_endpoint_shd` | marks that differ from the truth's: two for a missing or extra link, one per wrong end |
| `adjacency_precision`, `adjacency_recall`, `adjacency_f1` | the links, whatever their marks |
| `wrong_marks` | tails and arrowheads on true links that contradict the truth |
| `lagged_tail`, `lagged_circle`, `lagged_arrow` | the true lagged links found, by the mark at their earlier end |

Without hidden series the truth's lag 0 is the DAG, not the class of graphs the data can tell
apart, so an unoriented PCMCI+ link or a PAG circle costs its open ends in `endpoint_shd`;
`wrong_marks` counts only the marks a solution claims. Against a true PAG a circle in both is a
match, and a tail or an arrowhead where the true PAG has a circle costs one in `endpoint_shd` but
is not in `wrong_marks`. A CPDAG's conflicting link is read as undirected and a PAG's conflict mark
as a circle, as Andrey returns them.

```shell
ANDREY_BENCH_TIGRAMITE_PYTHON=$PWD/benchmarks/.venv-tigramite/bin/python \
    $ANDREY_BENCH_PYTHON benchmarks/pcmci_cpu.py --method lpcmci --out outputs/lpcmci-smoke --smoke
```

The driver refuses a series shorter than the shortest Andrey's method accepts:
`2 * tau_max * (N + 1) + N + 1` steps for PCMCI+, `N * (tau_max + 1) + 2 * tau_max + 1` for LPCMCI.
