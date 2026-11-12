"""The answers of the treatment-outcome methods, and how the lane scores them.

A method of the adjustment group answers for a pair of variables: whether the treatment has an
effect on the outcome, whether that effect is identifiable, and the adjustment sets to estimate it
with. Its answer is the list of ``effects`` records of ``metadata["effects"]``
(:mod:`andrey.core.effects`). This module holds what every solution and the driver share: how an
adapter hands its records to the driver (:func:`pack_effects`), the contract an adapter fills
(:class:`PairAdapter`), what the records are scored against (:func:`oracle_records`,
:func:`score_answer`), and the effect estimate of a returned set (:func:`ols_effect`).

The answers are scored by :func:`andrey.metrics.effect_scores` against the oracle records of
:func:`andrey.metrics.effect_oracle` computed on the true graph, and each returned set by
:func:`andrey.metrics.adjustment_scores`; nothing of theirs is reimplemented here. The effect error
is the lane's own: least squares of the outcome on the treatment and the returned set, on a fresh
sample of the model, against the model's true effect, both in standardized units (the intervention
distance of Schubert et al., 2025, Def. 9.1: the mean over the returned sets of the absolute
difference). An answer that gives no set and does not say the effect is zero abstains and has no
error; it is never scored as the empty set.

The module imports NumPy only, so an adapter in a reference's own environment can use it; the
scoring functions import :mod:`andrey` when called.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: The keys of ``params`` that hold the pair's columns for a method that answers for an ordered
#: pair. The driver adds them to the parameters of a solution whose ``unordered`` is false.
TREATMENT_PARAM = "treatment"
OUTCOME_PARAM = "outcome"

#: The key of ``params`` that holds the two columns of a method that does not know which of them
#: is the treatment (SNAP, LOAD): ``[i, j]``, in column order. Its records answer both directions.
VARIABLES_PARAM = "variables"

#: The key of ``params`` that holds the effect the cell asks for, ``"total"`` or ``"direct"``;
#: read by the arms that answer either, and ignored by the methods that have one estimand.
ESTIMAND_PARAM = "estimand"

#: ``n_tests`` of an answer whose solution reports no count.
NO_COUNT = -1

#: The estimands and the kinds of set of a record, as :mod:`andrey.core.effects` lists them (a test
#: checks the copies, so this module needs no Andrey).
ESTIMANDS: tuple[str, ...] = ("total", "direct")
SET_KINDS: tuple[str, ...] = ("optimal", "valid", "parents_of_outcome", "local_parents")

#: A solution's ``set_kind`` when it answers with the optimal set where the class identifies the
#: effect and with the local parent sets where it does not (SNAP, LOAD, and the PC arm). The oracle
#: answer is built the same way per pair.
OPTIMAL_OR_LOCAL = "optimal+local_parents"

#: The columns of a packed answer before the membership flags.
_HEAD = 8
_ESTIMAND_CODE = {"total": 0, "direct": 1}
_FLAG = {None: -1, False: 0, True: 1}
_FROM_FLAG = {v: k for k, v in _FLAG.items()}


@runtime_checkable
class PairAdapter(Protocol):
    """What a solution that answers for a treatment-outcome pair provides, beside a
    ``SolutionAdapter``'s members.

    ``fit(data, params)`` receives the dataset's columns and ``params`` with the pair's columns
    (0-based positions): ``params["treatment"]`` and ``params["outcome"]`` when ``unordered`` is
    false, ``params["variables"]`` (two columns) when it is true. It returns :func:`pack_effects`
    of its ``effects`` records, which name columns by position, and the number of
    conditional-independence tests it evaluated, or ``None`` when it reports none. The driver fits a
    solution once per pair of a dataset.

    The class attributes tell the lane what the solution's answers mean, so the oracle is built the
    way the method defines its answer:

    - ``estimand``: ``"total"`` or ``"direct"``, the effect it answers for; ``"any"`` for an arm
      that answers either, which reads ``params["estimand"]``. The driver runs a solution on the
      cells that ask for its estimand.
    - ``set_kind``: the kind of set it returns (one of :data:`SET_KINDS`) or
      :data:`OPTIMAL_OR_LOCAL`; the oracle's sets are of that kind.
    - ``identified_from``: ``"cpdag"`` when the method identifies the effect from the Markov
      equivalence class (no hidden variables), ``"dag"`` when from observed sets in the true
      graph (hidden variables allowed): the choices of ``effect_oracle``.
    - ``unordered``: ``True`` for a method that is not told which variable is the treatment.
    - ``output_type``: ``"effects"``.

    Two attributes are optional. ``indep_tests``, the tests the solution can run
    (``"fisherz"``, ``"chisq"``, ``"gsq"``), keeps it off the cells whose test it lacks; it
    defaults to every test. ``takes_max_depth``, ``True`` for a solution that accepts the largest
    conditioning set as ``params["max_depth"]``, defaults to ``False``.

    A solution that runs in an environment of its own (``python_env``) cannot import Andrey, so a
    reference whose output is a graph rather than sets returns the packed graph from ``fit`` and
    reads the sets in the parent: its optional ``records(native, params)`` returns ``(records,
    n_tests)`` and runs untimed where Andrey is installed (:func:`answer_records`). Without it the
    native answer is :func:`pack_effects`' array.
    """

    estimand: str
    set_kind: str
    identified_from: str
    unordered: bool
    output_type: str

    def params(self) -> Mapping[str, Any]:
        """The solution's settings, without the pair."""
        ...

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Answer for the pair in ``params``; the timed region."""
        ...


def pack_effects(
    records: Sequence[Mapping[str, Any]], d: int, n_tests: int | None = None
) -> np.ndarray:
    """Pack ``effects`` records over ``d`` columns, named by position, and a test count.

    One row per record and set, ``[treatment, outcome, estimand, nonzero, identifiable, set_kind,
    n_sets, set_index]`` followed by ``d`` flags for the members of the set; a record with no set
    is one row with ``set_index`` -1. The estimand and the set kind are indices into
    :data:`ESTIMANDS` and :data:`SET_KINDS`, ``nonzero`` and ``identifiable`` are -1 for ``None``,
    0, or 1. The last row holds the count in its first entry, :data:`NO_COUNT` when the solution
    reports none, so the array is ``int64`` throughout and survives the runner's pickle.
    """
    rows: list[list[int]] = []
    for record in records:
        sets = [sorted(int(v) for v in s) for s in record["adjustment_sets"]]
        kind = record["set_kind"]
        head = [
            int(record["treatment"]),
            int(record["outcome"]),
            _ESTIMAND_CODE[record["estimand"]],
            _FLAG[record["nonzero"]],
            _FLAG[record["identifiable"]],
            -1 if kind is None else SET_KINDS.index(kind),
            len(sets),
        ]
        for i, members in enumerate(sets or [None]):
            flags = [0] * d
            for v in members or ():
                flags[v] = 1
            rows.append([*head, -1 if members is None else i, *flags])
    packed = np.zeros((len(rows) + 1, _HEAD + d), dtype=np.int64)
    if rows:
        packed[:-1] = rows
    packed[-1, 0] = NO_COUNT if n_tests is None else int(n_tests)
    return packed


def unpack_effects(native: np.ndarray) -> tuple[list[dict[str, Any]], int | None]:
    """Return ``(records, n_tests)`` of what ``fit`` returned: :func:`pack_effects`' inverse.

    The records hold the seven fields of :func:`andrey.core.effects.effect_record` with columns
    named by position; their sets are lists of positions in ascending order, and the records are in
    the order of first appearance.
    """
    native = np.asarray(native)
    if native.ndim != 2 or native.shape[1] < _HEAD or native.shape[0] < 1:
        raise ValueError(f"a packed answer is a (rows + 1, {_HEAD} + d) array, got {native.shape}")
    count = int(native[-1, 0])
    records: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in native[:-1]:
        key = (int(row[0]), int(row[1]), int(row[2]))
        record = records.get(key)
        if record is None:
            kind = int(row[5])
            record = records[key] = {
                "treatment": key[0],
                "outcome": key[1],
                "estimand": ESTIMANDS[key[2]],
                "nonzero": _FROM_FLAG[int(row[3])],
                "identifiable": _FROM_FLAG[int(row[4])],
                "adjustment_sets": [],
                "set_kind": None if kind < 0 else SET_KINDS[kind],
            }
        if row[7] >= 0:
            record["adjustment_sets"].append([int(v) for v in np.flatnonzero(row[_HEAD:])])
    return list(records.values()), None if count == NO_COUNT else count


def empty_structure(native: np.ndarray) -> Any:
    """The graph without edges over the columns of a packed answer: the ``to_structure`` of every
    solution of the lane, as a pair answer draws no graph; untimed."""
    from andrey import GraphStructure

    d = np.asarray(native).shape[1] - _HEAD
    return GraphStructure.from_numpy(np.zeros((d, d), dtype=np.int8), kind="digraph")


def answer_records(
    adapter: object, native: np.ndarray, params: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int | None]:
    """The records and the test count of ``adapter``'s native answer: its own ``records`` when it
    has one, else :func:`unpack_effects`."""
    reader = getattr(adapter, "records", None)
    return reader(native, params) if callable(reader) else unpack_effects(native)


def ols_effect(data: np.ndarray, treatment: int, outcome: int, members: Sequence[int]) -> float:
    """The effect of ``treatment`` on ``outcome`` adjusted for ``members``, in standardized units.

    Least squares of the outcome on an intercept, the treatment, and the ``members``; the
    treatment's coefficient times the treatment's standard deviation over the outcome's, both
    sample standard deviations of ``data``. ``nan`` when a column is constant.
    """
    data = np.asarray(data, dtype=np.float64)
    cols = [data[:, treatment], *(data[:, v] for v in members)]
    design = np.column_stack([np.ones(len(data)), *cols])
    beta = np.linalg.lstsq(design, data[:, outcome], rcond=None)[0]
    sx, sy = data[:, treatment].std(ddof=1), data[:, outcome].std(ddof=1)
    return float(beta[1] * sx / sy) if sx > 0 and sy > 0 else float("nan")


def effect_estimates(
    record: Mapping[str, Any], fresh: np.ndarray, true_effect: float
) -> dict[str, Any] | None:
    """The effect estimates of one record on a fresh sample, and their distance to the truth.

    One estimate per returned set (:func:`ols_effect`); a record with no set that says the effect
    is zero estimates 0. ``error`` is the mean absolute difference from ``true_effect`` over the
    estimates, ``worst`` the largest. ``None`` for a record that gives no set and does not say
    the effect is zero: an abstention has no estimate.
    """
    x, y = int(record["treatment"]), int(record["outcome"])
    sets = record["adjustment_sets"]
    if sets:
        estimates = [ols_effect(fresh, x, y, s) for s in sets]
    elif record["nonzero"] is False:
        estimates = [0.0]
    else:
        return None
    gaps = [abs(e - true_effect) for e in estimates]
    return {
        "estimates": estimates,
        "true": true_effect,
        "error": float(np.mean(gaps)),
        "worst": float(np.max(gaps)),
    }


def oracle_records(
    dag: np.ndarray,
    hidden: Sequence[int],
    pair: tuple[int, int],
    *,
    estimand: str,
    set_kind: str,
    identified_from: str,
    unordered: bool = False,
) -> list[dict[str, Any]]:
    """The records the method's definition gives on the true graph for ``pair``.

    ``dag`` is the true directed adjacency over every node, hidden ones included, and ``hidden``
    their positions. The answer is :func:`andrey.metrics.effect_oracle` with the method's
    ``set_kind`` and ``identified_from``; :data:`OPTIMAL_OR_LOCAL` asks for the optimal set and,
    where it says the effect is not identifiable, for the local parent sets instead. A method
    that does not know the order (``unordered``) answers both directions.
    """
    from andrey.metrics import effect_oracle

    directions = [pair, pair[::-1]] if unordered else [pair]
    latent = list(hidden) or None
    out = []
    for x, y in directions:
        common = {"estimand": estimand, "identified_from": identified_from, "latent": latent}
        if set_kind == OPTIMAL_OR_LOCAL:
            record = effect_oracle(dag, x, y, set_kind="optimal", **common)
            if not record["identifiable"]:
                record = effect_oracle(dag, x, y, set_kind="local_parents", **common)
        else:
            record = effect_oracle(dag, x, y, set_kind=set_kind, **common)
        out.append(record)
    return out


def score_answer(
    records: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    dag: np.ndarray,
    hidden: Sequence[int],
    fresh: np.ndarray | None,
    effects: Mapping[tuple[int, int], float | None],
) -> dict[str, Any]:
    """Score one fit's records for one pair.

    ``records`` and ``oracle`` are lists of ``effects`` records naming columns by position (the
    oracle's from :func:`oracle_records`); ``dag`` and ``hidden`` the true graph; ``fresh`` a fresh
    sample of the model, ``None`` for discrete data; ``effects`` maps each ordered pair the oracle
    names to its true standardized effect (``None`` where undefined). Returns:

    - ``"answer"``: ``nonzero``, ``identifiable``, and ``adjustment``, each
      :func:`andrey.metrics.effect_scores`' ``answer_scores`` dict, over the oracle's pairs;
    - ``"sets"``: for each total-effect record, each returned set's :func:`adjustment_scores` as
      ``valid`` and, with no hidden variable, ``optimal``, ``precision``, ``recall``, and ``f1``;
    - ``"error"``: per record, :func:`effect_estimates` (or ``None``), keyed ``"x>y"``.
    """
    from andrey.metrics import adjustment_scores, effect_scores

    out: dict[str, Any] = {
        "answer": effect_scores(records, oracle, true=dag),
        "sets": [],
        "error": {},
    }
    for record in records:
        x, y = int(record["treatment"]), int(record["outcome"])
        if record["estimand"] == "total":
            for members in record["adjustment_sets"]:
                s = adjustment_scores(members, dag, x, y)
                keep = ("valid",) if hidden else ("valid", "optimal", "precision", "recall", "f1")
                out["sets"].append({"pair": [x, y], "set": members, **{k: s[k] for k in keep}})
        truth = effects.get((x, y))
        if fresh is not None and truth is not None:
            out["error"][f"{x}>{y}"] = effect_estimates(record, fresh, truth)
    return plain(out)


def plain(value: Any) -> Any:
    """``value`` with NumPy scalars and arrays as Python numbers and lists, for a JSON record."""
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value
