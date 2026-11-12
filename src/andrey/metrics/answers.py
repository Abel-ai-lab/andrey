"""Score answers that may abstain: correct, wrong, incomplete, or abstained, against an oracle.

A local method answers questions one at a time: whether a variable is in a target's blanket,
which mark an edge has at the target, whether an effect is identifiable. Each question is a query;
the oracle answer is the method's own definition applied to the true graph. :func:`answer_scores`
sorts the queries by outcome and compares the answer with two reference policies on the same
queries; :func:`pooled_answer_scores` adds up several such scores.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

#: The outcomes a query can have, in the order the counts are reported.
OUTCOMES: tuple[str, ...] = ("correct", "wrong", "incomplete", "abstained")

#: The policy label of the constant answer ``None``, "cannot tell".
CANNOT_TELL = "cannot tell"

Judge = Callable[[Any, Any, Any], str]


def same(query: Any, answer: Any, oracle: Any) -> str:
    """The outcome of ``answer`` when it is right only by equality with ``oracle``.

    Equal answers are correct, ``None`` for both included; ``None`` where the oracle decides is
    incomplete; any other answer is wrong, a decided answer where the oracle is ``None`` included.
    """
    if answer == oracle:
        return "correct"
    return "incomplete" if answer is None else "wrong"


def label(answer: Any) -> str:
    """The name of a constant answer in ``policy_net``: :data:`CANNOT_TELL` for ``None``."""
    return CANNOT_TELL if answer is None else str(answer)


def answer_scores(
    answers: Mapping[Any, Any],
    oracle: Mapping[Any, Any],
    *,
    choices: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Score the answers to a set of queries against the oracle answers to the same queries.

    Each query of ``oracle`` has one outcome. A query ``answers`` does not hold is *abstained*.
    An answer equal to the oracle's is *correct*, ``None`` ("cannot tell") for both included. An
    answer ``None`` where the oracle decides is *incomplete*: the answer leaves open what the
    oracle settles, such as an undirected edge where the oracle directs it, and it is never
    counted as wrong. Any other answer is *wrong*, a decided answer where the oracle is ``None``
    included. The answered queries are the correct and the wrong ones.

    The answer is compared with two reference policies on the same queries. Always abstaining
    has net 0. Always giving one constant answer ``a`` has the net of ``a`` given to every
    query; the base-rate policy is the constant answer, among ``choices``, with the highest net.
    The floor is the better of the two policies.

    Parameters
    ----------
    answers : mapping
        The answer to each query answered, keyed by query. ``None`` is "cannot tell".
    oracle : mapping
        The oracle answer to every query, keyed by query; ``None`` where the oracle cannot tell.
        Answers and oracle answers compare by ``==``.
    choices : iterable, optional
        The constant answers the base-rate policy may give. Default: every distinct oracle answer.

    Returns
    -------
    dict
        ``"queries"``, the number of oracle queries, and the counts ``"correct"``, ``"wrong"``,
        ``"incomplete"``, ``"abstained"``, which add up to it, and ``"answered"``, correct plus
        wrong. ``"coverage"`` = answered / queries; ``"risk"`` = wrong / answered;
        ``"net"`` = correct - wrong, and ``"net_rate"`` = net / queries, which equals coverage
        times (1 - 2 risk). ``"policy_net"`` maps each constant answer of ``choices``, named by
        its ``str`` (``"cannot tell"`` for ``None``), to its net, and ``"policy_correct"`` to
        the queries it gets right; ``"base_rate_answer"`` names the one with the highest net (the
        first of equals), ``"base_rate"`` is the share of queries it gets right, and
        ``"floor_rate"`` = max(0, its net / queries). A ratio with no denominator is ``nan``.

    Raises
    ------
    ValueError
        If ``answers`` holds a query that ``oracle`` lacks.

    Examples
    --------
    >>> oracle = {"a": True, "b": False, "c": False, "d": None}
    >>> s = answer_scores({"a": True, "b": True, "d": None}, oracle)
    >>> s["correct"], s["wrong"], s["abstained"], s["net"]
    (2, 1, 1, 1)
    >>> s["base_rate_answer"], s["floor_rate"]  # "cannot tell" everywhere: 1 right, 3 incomplete
    ('cannot tell', 0.25)
    """
    return tally(answers, oracle, same, choices)


def outcomes(
    answers: Mapping[Any, Any], oracle: Mapping[Any, Any], judge: Judge = same
) -> dict[Any, str]:
    """The outcome of each oracle query, one of :data:`OUTCOMES`: ``judge``'s for an answered
    query (default :func:`same`), ``"abstained"`` for the rest.

    Raises ``ValueError`` if ``answers`` holds a query that ``oracle`` lacks.
    """
    unknown = [q for q in answers if q not in oracle]
    if unknown:
        raise ValueError(f"answers hold queries the oracle lacks: {unknown[:3]!r}")
    return {
        query: judge(query, answers[query], truth) if query in answers else "abstained"
        for query, truth in oracle.items()
    }


def tally(
    answers: Mapping[Any, Any],
    oracle: Mapping[Any, Any],
    judge: Judge,
    choices: Iterable[Any] | None,
) -> dict[str, Any]:
    """:func:`answer_scores` with the outcome of each answered query given by ``judge``."""
    counts = dict.fromkeys(OUTCOMES, 0)
    for outcome in outcomes(answers, oracle, judge).values():
        counts[outcome] += 1
    if choices is None:
        choices = _distinct(oracle.values())
    policy_net, policy_correct = {}, {}
    for constant in choices:
        results = [judge(query, constant, truth) for query, truth in oracle.items()]
        policy_net[label(constant)] = results.count("correct") - results.count("wrong")
        policy_correct[label(constant)] = results.count("correct")
    return rates(len(oracle), counts, policy_net, policy_correct)


def rates(
    queries: int,
    counts: Mapping[str, int],
    policy_net: Mapping[str, int],
    policy_correct: Mapping[str, int],
) -> dict[str, Any]:
    """The rates of :func:`answer_scores` from its counts."""
    answered = counts["correct"] + counts["wrong"]
    best = max(policy_net, key=lambda a: policy_net[a]) if policy_net else None
    best_net = policy_net[best] if best is not None else 0
    return {
        "queries": queries,
        **{k: int(counts[k]) for k in OUTCOMES},
        "answered": answered,
        "coverage": _ratio(answered, queries),
        "risk": _ratio(counts["wrong"], answered),
        "net": counts["correct"] - counts["wrong"],
        "net_rate": _ratio(counts["correct"] - counts["wrong"], queries),
        "policy_net": dict(policy_net),
        "policy_correct": dict(policy_correct),
        "base_rate_answer": best,
        "base_rate": _ratio(policy_correct[best], queries) if best is not None else float("nan"),
        "floor_rate": max(0.0, _ratio(best_net, queries)) if queries else float("nan"),
    }


def pooled_answer_scores(scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool several :func:`answer_scores` results into one, as if their queries were one set.

    The counts add up. A constant answer keeps one value across the pool, so its net is the sum
    of its nets; where a part did not list it among its choices, no oracle answer there equals
    it, and it counts as wrong on every query of that part (``"cannot tell"``: incomplete).

    Parameters
    ----------
    scores : sequence of dict
        Results of :func:`answer_scores` (or of this function).

    Returns
    -------
    dict
        The keys of :func:`answer_scores`, computed over the pooled queries.

    Examples
    --------
    >>> a = answer_scores({0: True}, {0: True, 1: False})
    >>> b = answer_scores({0: False}, {0: False})
    >>> p = pooled_answer_scores([a, b])
    >>> p["queries"], p["net"], p["base_rate_answer"]
    (3, 2, 'False')
    """
    names = _distinct(name for s in scores for name in s["policy_net"])
    counts = {k: sum(s[k] for s in scores) for k in OUTCOMES}
    net, right = {}, {}
    for name in names:
        net[name] = sum(
            s["policy_net"].get(name, 0 if name == CANNOT_TELL else -s["queries"]) for s in scores
        )
        right[name] = sum(s["policy_correct"].get(name, 0) for s in scores)
    return rates(sum(s["queries"] for s in scores), counts, net, right)


def _distinct(values: Iterable[Any]) -> list[Any]:
    """The distinct values in their first order."""
    found: list[Any] = []
    for value in values:
        if value not in found:
            found.append(value)
    return found


def _ratio(num: float, den: float) -> float:
    return num / den if den else float("nan")
