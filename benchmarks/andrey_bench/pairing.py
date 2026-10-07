"""Whether two measurements may be compared at all.

Tables belong to a batch and live with its output; only the rules are here.
"""

from __future__ import annotations

import pandas as pd

#: Two arms whose fits differ by more than this are filtered by node contention unevenly.
SKEW_LIMIT = 5.0


def spans_runs(g: pd.DataFrame) -> bool:
    """Does this group pool rows from more than one run?"""
    return g.run_dir.nunique() > 1


def resumed(df: pd.DataFrame) -> list[str]:
    """Run directories written by more than one job."""
    if "run_sittings" not in df.columns:
        return []
    return sorted(df[df.run_sittings.fillna(1) > 1].run_dir.unique())


def ceiling(g: pd.DataFrame) -> str | None:
    """The status that stopped a group of rows, or None if it produced a measurement."""
    if len(g[g.status == "ok"]):
        return None
    # `blocked` stays itself: the fit did not start, so the size is unmeasured, not unreachable.
    statuses = sorted(set(g.status))
    return "/".join(statuses) if statuses else None


#: What makes two rows a pair: the same dataset, measured in the same run.
PAIR_KEY: tuple[str, ...] = ("run_dir", "dataset_id")


def shared_cells(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both frames cut down to the ``PAIR_KEY`` groups they both cover."""
    ka = set(map(tuple, a[list(PAIR_KEY)].itertuples(index=False)))
    kb = set(map(tuple, b[list(PAIR_KEY)].itertuples(index=False)))
    both = ka & kb

    def cut(df):
        keys = map(tuple, df[list(PAIR_KEY)].itertuples(index=False))
        return df[[k in both for k in keys]]

    return cut(a), cut(b)


def paired_ratios(
    fast: pd.DataFrame,
    slow: pd.DataFrame,
    by: tuple[str, ...] = PAIR_KEY,
) -> list[float]:
    """Ratios of ``slow`` over ``fast``, one per dataset both methods measured, sorted."""
    f, s = shared_cells(fast, slow)
    if not len(f):
        return []
    keys = list(by)
    out = []
    for key, fg in f.groupby(keys):
        sg = s
        for name, value in zip(keys, key if isinstance(key, tuple) else (key,)):
            sg = sg[sg[name] == value]
        if len(sg) and fg.wall_s.median():
            out.append(sg.wall_s.median() / fg.wall_s.median())
    return sorted(out)


def duration_skew(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """How many times longer one method's fit runs than the other's, over the paired rows."""
    a, b = shared_cells(a, b)
    if not len(a) or not len(b):
        return 1.0
    am, bm = a.wall_s.median(), b.wall_s.median()
    return max(am, bm) / max(min(am, bm), 1e-12)
