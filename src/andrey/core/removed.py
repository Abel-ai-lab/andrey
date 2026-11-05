"""The argument names and values that a release renamed or dropped, with their current names.

A release that renames or drops an argument keeps the former name working, with a
``DeprecatedArgumentWarning``, until a later release removes it. A call that passes a removed name
fails as it does for any unknown argument or any value that is not a choice, and the error adds
what it is called now and the release that renamed it (:func:`removal_hint`). The upgrade guide's
tables of former names list the same rows.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import NamedTuple

__all__ = ["REMOVED", "Removed", "removal_hint"]


class Removed(NamedTuple):
    """An argument name, or a value of one argument, that the ``methods`` no longer take.

    ``release`` is the release that renamed or dropped it. ``now`` is the current name, or the
    current value when ``value`` is set; ``None`` when the argument was dropped, in which case
    ``note`` says what to do instead.
    """

    release: str
    methods: tuple[str, ...]
    name: str
    now: str | None
    value: str | None = None
    note: str = ""


#: One row for each removed name or value, as the upgrade guide's tables list it. A name removed
#: from some methods can still be a current name in others, or a former name with another current
#: one: ``pc_alpha`` became ``alpha`` in ``pcmciplus`` and ``lpcmci``, and ``pc1_alpha`` in
#: ``pcmci``.
REMOVED: tuple[Removed, ...] = (
    Removed("0.11.0", ("ica_lingam",), "random_state", "seed"),
    Removed(
        "0.11.0",
        ("direct_lingam", "multi_group_direct_lingam"),
        "random_state",
        None,
        note="it set nothing",
    ),
    Removed("0.11.0", ("pcmci",), "alpha_level", "alpha"),
    Removed("0.11.0", ("pcmciplus", "lpcmci"), "pc_alpha", "alpha"),
    Removed("0.11.0", ("rcd",), "ind_alpha", "alpha"),
    Removed("0.11.0", ("calm",), "alpha", "screen_alpha"),
    Removed(
        "0.11.0",
        ("gfci", "ges", "gies", "xges", "hc", "tabu", "boss", "grasp"),
        "lambda_value",
        "penalty_discount",
    ),
    Removed(
        "0.11.0",
        ("gfci", "ges", "xges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "bic",
        "local_score_BIC",
    ),
    Removed(
        "0.11.0",
        ("gfci", "ges", "xges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "bic",
        "local_score_BIC_from_cov",
    ),
    Removed(
        "0.11.0",
        ("gfci", "ges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "discrete_bic",
        "local_score_BIC_categorical",
    ),
    Removed(
        "0.11.0",
        ("gfci", "ges", "hc", "tabu", "exact_search", "boss", "grasp"),
        "score_func",
        "bdeu",
        "local_score_BDeu",
    ),
    Removed(
        "0.11.0",
        ("gfci", "ges", "hc", "exact_search", "boss", "grasp"),
        "score_func",
        "cond_gauss",
        "local_score_cond_gauss",
    ),
    Removed(
        "0.11.0",
        ("gfci", "ges", "hc", "exact_search", "boss", "grasp"),
        "score_func",
        "degen_gauss",
        "local_score_degen_gauss",
    ),
    Removed("0.11.0", ("granger",), "maxlag", "tau_max"),
    Removed("0.11.0", ("longitudinal_lingam",), "n_lags", "tau_max"),
    Removed("0.11.0", ("rcd",), "max_explanatory_vars", "max_depth"),
    Removed("0.11.0", ("calm",), "max_outer", "max_iter"),
    Removed("0.11.0", ("multi_group_direct_lingam",), "data_groups", "data"),
    Removed("0.11.0", ("longitudinal_lingam",), "data_list", "data"),
    Removed(
        "0.11.0",
        ("gin", "calm"),
        "labels",
        None,
        note="the labels come from a DataFrame's columns",
    ),
    Removed("0.12.0", ("grasp",), "depth", "grasp_depth"),
    Removed("0.12.0", ("pcmci",), "pc_alpha", "pc1_alpha"),
    Removed("0.12.0", ("pcmciplus",), "collider_rule", "sepsets", "none"),
    Removed(
        "0.12.0",
        ("granger",),
        "cv",
        None,
        note="`alpha` sets the level of the test of each link",
    ),
    Removed(
        "0.12.0",
        (
            "pc",
            "fci",
            "gfci",
            "cdnod",
            "iamb",
            "inter_iamb",
            "mmpc",
            "mmhc",
            "rsmax2",
            "ges",
            "gies",
            "xges",
            "hc",
            "tabu",
            "exact_search",
            "boss",
            "grasp",
            "pcmci",
            "pcmciplus",
            "lpcmci",
        ),
        "max_categories",
        None,
        note="declare the categorical columns with `categorical` or a column type",
    ),
    Removed("0.14.0", ("cdnod",), "domain_index", "env"),
    Removed("0.14.0", ("gies",), "interventions", "targets"),
)


def removal_hint(
    method: str, name: str, value: object = None, *, flags: Collection[str] = ()
) -> str | None:
    """Return what ``method``'s removed argument ``name``, or its removed ``value``, is called now.

    The hint is a clause to add to the error, such as ``it was renamed `alpha` in 0.11.0``, ``it
    was renamed `"bic"` in 0.11.0``, or ``it was removed in 0.11.0: it set nothing``, with the
    release that renamed or dropped it. A current name in ``flags`` is spelled as a command-line
    flag, ``--data``. Returns ``None`` when the method never took that name or value, or still
    takes it.
    """
    if value is not None and not isinstance(value, str):
        return None
    for row in REMOVED:
        if method not in row.methods or row.name != name or row.value != value:
            continue
        if row.now is None:
            return f"it was removed in {row.release}: {row.note}"
        if row.value is not None:
            return f'it was renamed `"{row.now}"` in {row.release}'
        now = f"--{row.now.replace('_', '-')}" if row.now in flags else row.now
        return f"it was renamed `{now}` in {row.release}"
    return None
