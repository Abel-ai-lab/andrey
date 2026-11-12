"""Campaign rules for local answers: the gate, paired comparisons, and risk-coverage tables.

The scores of one answer come from :mod:`andrey.metrics`: :func:`~andrey.metrics.answer_scores`
and the scorers built on it (``target_answer_scores``, ``effect_scores``, ``claim_scores``), each
query correct, wrong, incomplete, or abstained against the oracle answer, the method's definition
applied to the true graph. A lane pools a dataset's targets or pairs with
:func:`~andrey.metrics.pooled_answer_scores` and judges a cell's datasets here:

- :func:`gate`: the answer against its floor, the better of always abstaining and always giving
  the base-rate answer, paired by dataset with the release cards' bootstrap rule
  (:func:`andrey_bench.pairing.compare`). A failed fit (a timeout or an error) fails the gate.
- :func:`paired_risk`: two methods' risk on the queries both answered, paired by dataset, so a
  method that abstains on hard queries does not buy a lower risk.
- :func:`risk_coverage`: the risk and coverage of an answer kept down to each confidence
  threshold, the display of a method that ranks its answers.
- :func:`wilson_95`: the Wilson score interval of a share, for the precision of claims.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from andrey.metrics import pooled_answer_scores
from andrey.metrics.answers import CANNOT_TELL, outcomes, same

from .pairing import MIN_PAIRS, compare

#: The floor policy that answers nothing.
ABSTAIN = "abstain"

#: The outcomes that count as answered.
ANSWERED = ("correct", "wrong")


def gate(datasets: Sequence[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Judge an answer against its floor over the datasets of one cell.

    The floor policy is chosen once, on the pooled queries: the base-rate answer when its pooled
    net is above 0, else abstaining (net 0). Per dataset, ``D`` = the answer's net rate minus the
    floor policy's net rate on the same queries. The verdict is ``"fail"`` when any dataset
    failed, ``"reported"`` with fewer than :data:`~andrey_bench.pairing.MIN_PAIRS` datasets, and
    otherwise ``"pass"`` exactly when the bootstrap 95% interval of the mean ``D`` lies wholly
    above 0.

    Parameters
    ----------
    datasets : sequence of dict or None
        One :func:`~andrey.metrics.answer_scores` result per dataset, pooled over its queries;
        ``None`` for a dataset whose fit failed.

    Returns
    -------
    dict
        ``"datasets"``, ``"failed"``, ``"floor_policy"`` (:data:`ABSTAIN` or the constant
        answer's name), ``"net_rate"`` and ``"floor_rate"`` over the pooled queries, ``"paired"``,
        ``"mean_D"``, ``"bootstrap_95"``, and ``"verdict"``.
    """
    finished = [d for d in datasets if d is not None]
    failed = len(datasets) - len(finished)
    pooled = pooled_answer_scores(finished) if finished else None
    policy = ABSTAIN
    if pooled is not None and pooled["base_rate_answer"] is not None:
        if pooled["policy_net"][pooled["base_rate_answer"]] > 0:
            policy = pooled["base_rate_answer"]
    usable = [d for d in finished if d["queries"]]
    mine = [d["net"] / d["queries"] for d in usable]
    floor = [_policy_net(d, policy) / d["queries"] if policy != ABSTAIN else 0.0 for d in usable]
    paired = compare(mine, floor, higher_is_better=True) if usable else {"paired": 0}
    if failed:
        verdict = "fail"
    elif paired["paired"] < MIN_PAIRS:
        verdict = "reported"
    else:
        verdict = "pass" if paired["verdict"] == "better" else "fail"
    return {
        "datasets": len(datasets),
        "failed": failed,
        "floor_policy": policy,
        "net_rate": pooled["net_rate"] if pooled else float("nan"),
        "floor_rate": pooled["floor_rate"] if pooled else float("nan"),
        "paired": paired["paired"],
        "mean_D": paired.get("mean_D"),
        "bootstrap_95": paired.get("bootstrap_95"),
        "verdict": verdict,
    }


def _policy_net(scores: Mapping[str, Any], policy: str) -> int:
    """A constant policy's net on one dataset; missing there, as in ``pooled_answer_scores``."""
    default = 0 if policy == CANNOT_TELL else -scores["queries"]
    return scores["policy_net"].get(policy, default)


def paired_risk(
    datasets: Sequence[tuple[Mapping[Any, str] | None, Mapping[Any, str] | None]],
) -> dict[str, Any]:
    """Compare two methods' risk on the queries both answered, paired by dataset.

    Per dataset, the common queries are those both answered (each outcome ``"correct"`` or
    ``"wrong"``), and ``D`` = risk of ``a`` minus risk of ``b`` there; a dataset with no common
    query has no ``D``. The verdict is :func:`andrey_bench.pairing.compare`'s for ``a``, lower
    risk better; ``"failed_a"`` and ``"failed_b"`` count the datasets each side failed, which a
    caller reports beside the verdict, since a timeout never becomes a lower risk.

    Parameters
    ----------
    datasets : sequence of (dict or None, dict or None)
        Per dataset, each method's outcome per query, as
        :func:`andrey.metrics.answers.outcomes` returns them over the same queries; ``None``
        where that method's fit failed.

    Returns
    -------
    dict
        ``"common"``, the common queries over all datasets, ``"failed_a"``, ``"failed_b"``,
        ``"risk_a"`` and ``"risk_b"`` pooled over the common queries, and the keys of
        :func:`andrey_bench.pairing.compare`.
    """
    risk_a, risk_b, wrong_a, wrong_b, common = [], [], 0, 0, 0
    failed_a = sum(a is None for a, _ in datasets)
    failed_b = sum(b is None for _, b in datasets)
    for a, b in datasets:
        if a is None or b is None:
            continue
        if a.keys() != b.keys():
            raise ValueError("the two methods must be scored on the same queries")
        both = [q for q in a if a[q] in ANSWERED and b[q] in ANSWERED]
        if not both:
            continue
        wa = sum(a[q] == "wrong" for q in both)
        wb = sum(b[q] == "wrong" for q in both)
        risk_a.append(wa / len(both))
        risk_b.append(wb / len(both))
        wrong_a, wrong_b, common = wrong_a + wa, wrong_b + wb, common + len(both)
    paired = compare(risk_a, risk_b) if risk_a else {"paired": 0}
    return {
        "common": common,
        "failed_a": failed_a,
        "failed_b": failed_b,
        "risk_a": wrong_a / common if common else float("nan"),
        "risk_b": wrong_b / common if common else float("nan"),
        **paired,
    }


def risk_coverage(
    answers: Mapping[Any, Any],
    oracle: Mapping[Any, Any],
    confidence: Mapping[Any, float],
    judge: Callable[[Any, Any, Any], str] = same,
) -> list[dict[str, Any]]:
    """The risk-coverage table of an answer that ranks its answers by confidence.

    The answered queries (correct or wrong) are kept from the most confident down; queries of
    equal confidence enter together, as one threshold. Each row is one threshold: the answers
    kept, ``coverage`` = kept / queries, ``risk`` = wrong among them / kept, and ``net_rate`` =
    (correct - wrong) among them / queries.

    Parameters
    ----------
    answers, oracle : mapping
        As :func:`andrey.metrics.answer_scores` takes them.
    confidence : mapping
        A confidence for each answered query, such as a claim's score; higher is surer.
    judge : callable, optional
        The outcome of one answer, :func:`andrey.metrics.answers.same` by default.

    Returns
    -------
    list of dict
        Rows ``{"threshold", "kept", "coverage", "risk", "net_rate"}``, most confident first.
    """
    found = outcomes(answers, oracle, judge)
    ranked = sorted(
        (q for q, outcome in found.items() if outcome in ANSWERED),
        key=lambda q: -confidence[q],
    )
    rows, kept, wrong = [], 0, 0
    for i, query in enumerate(ranked):
        kept += 1
        wrong += found[query] == "wrong"
        last = i + 1 == len(ranked) or confidence[ranked[i + 1]] != confidence[query]
        if last:
            rows.append(
                {
                    "threshold": float(confidence[query]),
                    "kept": kept,
                    "coverage": kept / len(oracle),
                    "risk": wrong / kept,
                    "net_rate": (kept - 2 * wrong) / len(oracle),
                }
            )
    return rows


def wilson_95(successes: int, n: int) -> list[float]:
    """The Wilson score 95% interval of the share ``successes / n``; ``[0.0, 1.0]`` for ``n`` 0.

    References
    ----------
    Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference.
    Journal of the American Statistical Association 22(158), 209-212.
    """
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    share = successes / n
    center = (share + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, center - half), min(1.0, center + half)]
