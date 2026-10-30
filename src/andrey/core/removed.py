"""The argument names and values that 0.11.0 renamed and 0.12.0 removed, with their current names.

0.11.0 kept the former names working for one release. A call that passes one now fails as it does
for any unknown argument or any value that is not a choice, and the error adds what it is called
now (:func:`removal_hint`). The upgrade guide's table of former names lists the same rows.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import NamedTuple

__all__ = ["REMOVED", "Removed", "removal_hint"]

#: The release that renamed the arguments and values below.
RENAMED_IN = "0.11.0"


class Removed(NamedTuple):
    """An argument name, or a value of one argument, that the ``methods`` no longer take.

    ``now`` is the current name, or the current value when ``value`` is set; ``None`` when the
    argument was dropped, in which case ``note`` says what to do instead.
    """

    methods: tuple[str, ...]
    name: str
    now: str | None
    value: str | None = None
    note: str = ""


SCORE_SEARCHES = ("gfci", "ges", "hc", "exact_search", "boss", "grasp")

#: One row for each removed name or value, as the upgrade guide's table lists it. A name removed
#: from some methods can still be a current name, or a name kept for one release, in others:
#: ``pc_alpha`` is gone from ``pcmciplus`` and ``lpcmci`` and still works in ``pcmci``.
REMOVED: tuple[Removed, ...] = (
    Removed(("ica_lingam",), "random_state", "seed"),
    Removed(
        ("direct_lingam", "multi_group_direct_lingam"), "random_state", None, note="it set nothing"
    ),
    Removed(("pcmci",), "alpha_level", "alpha"),
    Removed(("pcmciplus", "lpcmci"), "pc_alpha", "alpha"),
    Removed(("rcd",), "ind_alpha", "alpha"),
    Removed(("calm",), "alpha", "screen_alpha"),
    Removed(
        ("gfci", "ges", "gies", "xges", "hc", "tabu", "boss", "grasp"),
        "lambda_value",
        "penalty_discount",
    ),
    Removed(
        ("gfci", "ges", "xges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "bic",
        "local_score_BIC",
    ),
    Removed(
        ("gfci", "ges", "xges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "bic",
        "local_score_BIC_from_cov",
    ),
    Removed(
        ("gfci", "ges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "discrete_bic",
        "local_score_BIC_categorical",
    ),
    Removed(
        ("gfci", "ges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "bdeu",
        "local_score_BDeu",
    ),
    Removed(SCORE_SEARCHES, "score_func", "cond_gauss", "local_score_cond_gauss"),
    Removed(SCORE_SEARCHES, "score_func", "degen_gauss", "local_score_degen_gauss"),
    Removed(("granger",), "maxlag", "tau_max"),
    Removed(("longitudinal_lingam",), "n_lags", "tau_max"),
    Removed(("rcd",), "max_explanatory_vars", "max_depth"),
    Removed(("calm",), "max_outer", "max_iter"),
    Removed(("multi_group_direct_lingam",), "data_groups", "data"),
    Removed(("longitudinal_lingam",), "data_list", "data"),
    Removed(("gin", "calm"), "labels", None, note="the labels come from a DataFrame's columns"),
)


def removal_hint(
    method: str, name: str, value: object = None, *, flags: Collection[str] = ()
) -> str | None:
    """Return what ``method``'s removed argument ``name``, or its removed ``value``, is called now.

    The hint is a clause to add to the error, such as ``it was renamed `alpha` in 0.11.0``, ``it
    was renamed `"bic"` in 0.11.0``, or ``it was removed in 0.11.0: it set nothing``. A current
    name in ``flags`` is spelled as a command-line flag, ``--data``. Returns ``None`` when the
    method never took that name or value, or still takes it.
    """
    if value is not None and not isinstance(value, str):
        return None
    for row in REMOVED:
        if method not in row.methods or row.name != name or row.value != value:
            continue
        if row.now is None:
            return f"it was removed in {RENAMED_IN}: {row.note}"
        if row.value is not None:
            return f'it was renamed `"{row.now}"` in {RENAMED_IN}'
        now = f"--{row.now.replace('_', '-')}" if row.now in flags else row.now
        return f"it was renamed `{now}` in {RENAMED_IN}"
    return None
