"""Interventions on a simulated model: hard and soft changes to one variable's mechanism.

:func:`hard` cuts a variable from its parents and sets it from outside; :func:`soft` changes its
mechanism and keeps its parents. :meth:`~andrey.data.SCM.do` applies them to a model, and
``envs=`` on :meth:`~andrey.data.SCM.sample`, :func:`~andrey.data.sample_discrete`, and
:func:`~andrey.data.sample_mixed` draws several environments, each with its own interventions, into
one dataset:

>>> import andrey.data as data
>>> from andrey.data.interventions import hard, soft
>>> scm = data.SCM(graph=data.graphs.erdos_renyi(d=8, avg_degree=2),
...                functional=data.functional.linear(),
...                noise=data.noise.gaussian())
>>> ds = scm.sample(n=[500, 200, 200], seed=0, envs=[{}, {3: hard(2.0)}, {5: soft(shift=1.0)}])
>>> ds.data.shape, ds.targets
((900, 8), {0: [], 1: [3], 2: [5]})

A variable is named by its column in the returned data, after the column shuffle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

__all__ = ["HardIntervention", "SoftIntervention", "hard", "soft"]


@dataclass(frozen=True, kw_only=True)
class HardIntervention:
    """A hard intervention: the variable's parents no longer act on it (see :func:`hard`)."""

    value: float | None = None
    sd: float = 0.0
    probs: tuple[float, ...] | None = None


@dataclass(frozen=True, kw_only=True)
class SoftIntervention:
    """A soft intervention: the mechanism changes and the parents still act (:func:`soft`)."""

    shift: float = 0.0
    weight_scale: float = 1.0
    noise_scale: float | tuple[float, float] = 1.0
    weights: str = "keep"


Intervention = HardIntervention | SoftIntervention


def hard(
    value: float | None = None, *, sd: float = 0.0, probs: Sequence[float] | None = None
) -> HardIntervention:
    """Build a hard intervention: the variable is set from outside, and its parents no longer act.

    Parameters
    ----------
    value : float or None, default=None
        A continuous variable: its value, or the mean of its values when ``sd`` is positive, in
        the units of the returned data, whatever the ``scale``. A discrete variable: the category
        every row takes. ``None`` takes ``probs`` for a discrete variable.
    sd : float, default=0.0
        A continuous variable: the standard deviation of a Gaussian the values are drawn from,
        in the units of the returned data. ``0`` sets every row to ``value``.
    probs : sequence of float or None, default=None
        A discrete variable: the probability of each category, the same for every row. ``None``
        with ``value=None`` gives every category the same probability.

    Returns
    -------
    HardIntervention

    Raises
    ------
    ValueError
        If ``value`` or ``sd`` is not finite, ``sd`` is negative, ``probs`` is given with
        ``value`` or ``sd``, or ``probs`` has a negative entry or does not sum to 1.

    Examples
    --------
    >>> from andrey.data.interventions import hard
    >>> hard(2.0, sd=0.2)
    HardIntervention(value=2.0, sd=0.2, probs=None)
    """
    if value is not None and not np.isfinite(value):
        raise ValueError(f"hard() takes a finite value, got {value}")
    if not (np.isfinite(sd) and sd >= 0):
        raise ValueError(f"hard() takes a finite sd >= 0, got {sd}")
    if probs is None:
        return HardIntervention(value=None if value is None else float(value), sd=float(sd))
    if value is not None or sd:
        raise ValueError("hard() takes probs, or a value and an sd, not both")
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 1 or p.size < 2 or not np.all(p >= 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError(f"probs must hold two or more probabilities that sum to 1, got {probs}")
    return HardIntervention(probs=tuple(float(v) for v in p))


def soft(
    *,
    shift: float = 0.0,
    weight_scale: float = 1.0,
    noise_scale: float | Sequence[float] = 1.0,
    weights: str = "keep",
) -> SoftIntervention:
    """Build a soft intervention: the variable's mechanism changes, and its parents still act.

    On a continuous variable, its noise is multiplied by ``noise_scale`` and moved by ``shift``,
    and the weights of its parents are multiplied by ``weight_scale`` or drawn anew. On a discrete
    variable, ``soft()`` takes no arguments: every row of its conditional probability table is
    drawn anew, from the Dirichlet the sampler drew it from.

    Parameters
    ----------
    shift : float, default=0.0
        Added to the variable's noise, in the units of the returned data, whatever the ``scale``.
        Under the linear and additive-noise functionals it moves the variable's mean by
        ``shift``.
    weight_scale : float, default=1.0
        Multiplies the weight of each of the variable's parents; under the location-scale
        functional, its weights in the location and in the scale, and in mixed data, a categorical
        parent's shifts. A variable without parents has no weights to change.
    noise_scale : float or (float, float), default=1.0
        Multiplies the variable's noise, so its noise standard deviation. A pair ``(low, high)``,
        ``low < high``, multiplies the noise of each row by its own draw from ``U[low, high]``, so
        the noise follows a new distribution, a scale mixture of the one it had (Peters et al.
        2016, Appendix G).
    weights : {"keep", "redraw"}, default="keep"
        ``"redraw"`` draws the weight of each of the variable's parents anew, each on its own, from
        the functional's own weight distribution: magnitude ``U[weight_range]``, with a random sign,
        or positive when the functional draws positive weights. The new weights come from that
        ``weight_range``, not from a range of their own as in Peters et al. (2016, Appendix G).
        Under the location-scale functional, the weights in the location and in the scale are drawn
        anew, and in mixed data, each continuous parent's weight and categorical parent's shifts, as
        :func:`~andrey.data.sample_mixed` draws them. A variable without parents has no weights to
        draw anew. The new weights are part of the model: samples with the same model seed draw the
        same ones. Each environment draws its own: the sample under :meth:`~andrey.data.SCM.do` and
        each environment of ``envs=`` draw different new weights for the same intervention, and the
        dataset's ``params`` holds the weights without interventions. ``"redraw"`` takes
        ``weight_scale=1``.

    Returns
    -------
    SoftIntervention

    Raises
    ------
    ValueError
        If an argument is not finite, ``noise_scale`` is negative or a pair not ordered
        ``0 <= low < high``, ``weights`` is not ``"keep"`` or ``"redraw"``, or ``"redraw"`` is
        given with a ``weight_scale`` other than 1.

    References
    ----------
    Peters, J., Buhlmann, P., and Meinshausen, N. (2016). Causal inference by using invariant
    prediction: identification and confidence intervals. Journal of the Royal Statistical
    Society, Series B, 78(5), 947-1012.

    Examples
    --------
    >>> from andrey.data.interventions import soft
    >>> soft(shift=1.0)
    SoftIntervention(shift=1.0, weight_scale=1.0, noise_scale=1.0, weights='keep')
    >>> soft(noise_scale=(1.0, 3.0), weights="redraw")
    SoftIntervention(shift=0.0, weight_scale=1.0, noise_scale=(1.0, 3.0), weights='redraw')
    """
    values = np.asarray(noise_scale, dtype=float)
    if values.ndim == 0:
        noise = float(values)
        if not (np.isfinite(noise) and noise >= 0):
            raise ValueError(f"soft() takes a finite noise_scale >= 0, got {noise_scale}")
    else:
        if values.shape != (2,) or not (np.all(np.isfinite(values)) and 0 <= values[0] < values[1]):
            raise ValueError(
                f"soft() takes noise_scale as a number or a pair 0 <= low < high, got {noise_scale}"
            )
        noise = (float(values[0]), float(values[1]))
    if not all(np.isfinite(v) for v in (shift, weight_scale)):
        raise ValueError(
            f"soft() takes finite arguments, got shift={shift}, weight_scale={weight_scale}"
        )
    if weights not in ("keep", "redraw"):
        raise ValueError(f"soft() takes weights 'keep' or 'redraw', got {weights!r}")
    if weights == "redraw" and weight_scale != 1:
        raise ValueError("soft() takes weights='redraw' or a weight_scale, not both")
    return SoftIntervention(
        shift=float(shift), weight_scale=float(weight_scale), noise_scale=noise, weights=weights
    )


def as_interventions(spec: Mapping, n_columns: int | None = None) -> dict[int, Intervention]:
    """Check one environment's ``{column: intervention}`` and return it sorted by column.

    A number stands for ``hard(number)``. A column is a whole number, from 0 to ``n_columns - 1``
    when ``n_columns`` is given.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(
            f"an environment is a mapping {{column: hard(...) or soft(...)}}, got {spec!r}"
        )
    out: dict[int, Intervention] = {}
    for column, intervention in spec.items():
        if isinstance(column, bool) or not isinstance(column, int | np.integer):
            raise TypeError(f"a target is a column number, got {column!r}")
        if column < 0 or (n_columns is not None and column >= n_columns):
            raise ValueError(
                f"target column {column} is not one of the observed columns 0..{n_columns - 1}"
                if n_columns is not None
                else f"a target is a column number >= 0, got {column}"
            )
        if isinstance(intervention, int | float | np.integer | np.floating) and not isinstance(
            intervention, bool
        ):
            intervention = hard(float(intervention))
        if not isinstance(intervention, HardIntervention | SoftIntervention):
            raise TypeError(
                f"column {column}: an intervention is hard(...), soft(...), or a number, got "
                f"{intervention!r}"
            )
        out[int(column)] = intervention
    return dict(sorted(out.items()))


#: One environment: its interventions by column, and its number of rows.
Environment = tuple[dict[int, Intervention], int]


def environments(envs: Sequence[Mapping], n: int | Sequence[int]) -> list[Environment]:
    """Check ``envs`` and ``n``; return each environment's interventions and rows.

    ``n`` gives every environment ``n`` rows, or environment ``k`` ``n[k]`` rows.
    """
    if isinstance(envs, Mapping) or not isinstance(envs, Sequence) or not envs:
        raise TypeError(
            "envs is a non-empty list of mappings {column: intervention}, one per environment; "
            f"got {envs!r}"
        )
    sizes = [n] * len(envs) if isinstance(n, int | np.integer) else list(n)
    if len(sizes) != len(envs):
        raise ValueError(f"n gives {len(sizes)} row counts for {len(envs)} environments")
    if not all(
        isinstance(s, int | np.integer) and not isinstance(s, bool) and s >= 1 for s in sizes
    ):
        raise ValueError(f"every environment takes a whole number of rows >= 1, got n={n}")
    return [(as_interventions(spec), int(size)) for spec, size in zip(envs, sizes, strict=True)]


def pooled_rows(
    envs: Sequence[Mapping] | None, n: int | Sequence[int]
) -> tuple[int, list[Environment] | None]:
    """Return the rows of a sample and its :func:`environments`, ``None`` without ``envs``.

    Without ``envs``, ``n`` is one row count; with them, the rows are those of every environment.
    """
    if envs is None:
        if not isinstance(n, int | np.integer):
            raise TypeError(f"n is one row count without envs, got {n!r}")
        return int(n), None
    pooled = environments(envs, n)
    return sum(size for _, size in pooled), pooled


def check_columns(envs: Sequence[Environment], n_columns: int) -> None:
    """Raise unless every target of ``envs`` is one of ``n_columns`` columns."""
    for spec, _ in envs:
        as_interventions(spec, n_columns)


def env_labels(envs: Sequence[Environment]) -> np.ndarray:
    """Return the environment of each pooled row, ``0`` to ``K - 1``, in order."""
    return np.repeat(np.arange(len(envs), dtype=np.int64), [rows for _, rows in envs])


def env_targets(envs: Sequence[Environment]) -> dict[int, list[int]]:
    """Return each environment's intervened columns: ``{k: [columns]}``, ``[]`` for none."""
    return {k: sorted(spec) for k, (spec, _) in enumerate(envs)}


def record(envs: Sequence[Environment]) -> list[dict]:
    """Return the provenance record of the environments: each one's rows and interventions."""
    return [
        {
            "n": rows,
            "interventions": {
                str(column): {"kind": type(intervention).__name__.lower(), **asdict(intervention)}
                for column, intervention in spec.items()
            },
        }
        for spec, rows in envs
    ]
