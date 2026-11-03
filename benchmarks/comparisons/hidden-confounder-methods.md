# Hidden-confounder methods

`confounded_cpu.py` compares Andrey's `rcd` and `bottom_up_parce_lingam` with lingam's `RCD` and
`BottomUpParceLiNGAM`, with the empty-graph baseline. It holds both methods in one file because they
answer the same question: which variables share a hidden common cause. Both run on
`latent_lingam_er`, linear with uniform noise on an Erdos-Renyi graph that hides one node in five,
at least one, as in [Hidden confounders](../README.md#hidden-confounders); `d` includes the hidden
nodes. Both packages run each method with the same settings, listed in `params`.

The defaults are 2000 rows, `d` of 5, 10, and 15, five seeds, one repeat without warm-up, and
`--cap-wall-s` of 1800. A fit that reaches the cap is a `timeout` record with no scores. A smoke run
covers every solution on one small dataset:

```shell
$ANDREY_BENCH_PYTHON benchmarks/confounded_cpu.py --out outputs/confounded-smoke --smoke
```

Every estimate is a PAG and gets the observed-PAG scores of the FCI runs, but those do not rank
these methods: RCD uses its functional assumptions to mark `<->` where the observed PAG has `o-o`,
so the PAG scores count a correct `<->` as an error and BottomUpParceLiNGAM's `o-o` as a match. Read
two other scores instead.

The directed edges are scored against the latent projection, observed `i -> j` when a directed path
runs from `i` to `j` through hidden nodes only, less the edges whose endpoints share a hidden cause:
RCD and BottomUpParceLiNGAM estimate these, and mark a pair with a hidden cause instead of orienting
it. `directed_projection_*` holds `n_true`, `precision`, `recall`, `f1`, and `errors`, the edges in
one set and not the other (a reversed edge counts twice); `directed_n_est` counts the estimate's
edges.

A fit also names the pairs it attributes to a hidden common cause: the `<->` edges for RCD,
which match `metadata["confounded_pairs"]` of Andrey's result, and the `o-o` edges for
BottomUpParceLiNGAM. BottomUpParceLiNGAM's `o-o` edges join every pair of the one block of variables
it leaves unordered: the confounded variables, their observed ancestors, and any other confounded
pairs. Pairs in the block that share no cause count against its precision, and edges among the
block's variables stay unoriented, which caps its directed recall. Read its pair recall, the share
of true pairs it leaves unordered. `confounded_n_est` counts the pairs, and two sets of true pairs
score them:

| columns | true pairs |
|---|---|
| `confounded_hidden_cause_*` | the observed pairs that one hidden node reaches through hidden nodes only |
| `confounded_pag_*` | the observed pairs joined by `<->` in the observed PAG |

Each set has `n_true`, `precision`, `recall`, and `f1`. Score RCD and BottomUpParceLiNGAM on
`directed_projection_*` and `confounded_hidden_cause_*`: both mark a pair that shares a hidden
cause, even when one node is the other's parent. Compare each method across the two packages, not
RCD with BottomUpParceLiNGAM, whose block caps its scores as described above. The driver's summary
prints these. A PAG keeps `<->` only where conditional independences fix both
arrowheads, so `confounded_pag_*` has nothing to find on nearly every dataset at `d = 5`. A ratio
with no denominator is `NaN` rather than 1: report how many datasets define it, and read the
`n_true` columns beside the scores. The `confounder_*` columns from `andrey.metrics` score empty
against empty as 1 and do not see the `o-o` pairs of BottomUpParceLiNGAM.

The driver adds these columns to the record parts after each rung, from the estimates stored in
`structures/`. A resumed run scores only the units it measures, so a part scored by an older scorer
keeps its columns. A rescore recomputes every part's columns without fitting anything; pass the
first run's flags:

```shell
$ANDREY_BENCH_PYTHON benchmarks/confounded_cpu.py --out outputs/confounded-cpu --rescore
```
