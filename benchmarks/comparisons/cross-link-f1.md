# Cross-link F1 of a time-series run

Every time-series row is scored by one measure: precision, recall, and F1 of the window causal
graph over its cross-links (`scoring.cross_link_scores`), the F1 of Assaad et al. (2022), Sect. 9.1,
with the autodependencies left out, as Runge et al. (2019) leave them out. A cross-link joins two
different variables: at a lag, oriented by time, or within a time step, counted when the two
variables are adjacent (`cross_*`) and, in `cross_oriented_*`, only when it also points the same
way. A panel pools its time points from the second. `rescore_cross_links.py` adds these scores to a
finished `pcmci_cpu.py` or `lingam_ts.py` run without fitting anything: it reads the stored graphs
and the dataset truths and writes `cross_links.parquet`, one row per `run_id`. A truth with
unobserved series is left out of these scores; LPCMCI on `lpcmci_paper` is scored against its true
PAG instead (`rescore_window_pag.py`). `granger_var.py` records the same scores
as `cross_precision`, `cross_recall`, and `cross_f1`.

```shell
$ANDREY_BENCH_PYTHON benchmarks/rescore_cross_links.py outputs/pcmci outputs/pcmciplus
```
