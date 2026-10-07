# Ground-truth recovery and regression tests

This suite checks Andrey against fixed cases, known ground truth, and its own recorded outputs.
All algorithm tests run on every PR. GPU device parity lives in `tests/gpu/`.

Recorded outputs are regression baselines: a mismatch identifies a change to review.
Ground-truth tests establish whether the recovered structure is correct.

- `baselines/` holds 16 small recorded outputs and hashes of their seeded inputs.
- `helpers/` holds the fixed generators, output projection, tolerance ledger, and comparisons.
- `test_quality_gate.py` self-checks the pre-publish gate of `qa/quality_gate.py` without fitting
  a method: the gate covers the supported methods and rejects wrong answers. The gate itself
  measures method quality and runs with `uv run python -m qa.quality_gate`, not on pull requests.

Structural fields match exactly. Numeric fields use `helpers/tolerances.py`; ExactSearch compares
CPDAGs because score-equivalent DAG optima can differ in orientation. Ground-truth and independent
adapter tests cover errors that a shared capture/projection path could hide.

## Run and capture

```shell
uv run pytest tests/recovery -q
uv run python -m qa.capture_baselines --algorithms GES PC
uv run python -m qa.capture_baselines --out-dir outputs/baseline-review
```

Review a capture diff before committing it. Fixtures contain small, deterministic cases and no
machine details or timing. Schema v3 graph fields use Andrey endpoint marks:
`NULL=0`, `TAIL=1`, `ARROW=2`, `CIRCLE=3`. The entry at `[i, j]` is the mark at node `i`, so an
edge `i -> j` has `[i, j]=TAIL` and `[j, i]=ARROW`. Numeric fields keep their existing layouts.
Git history records fixture changes.
