"""Missing values on simulated data: masks whose missingness graph is known.

:func:`mcar`, :func:`mar`, :func:`mnar`, and :func:`self_masking` build a :class:`Mechanism`, the
rule by which values go missing. ``missing=`` on :meth:`~andrey.data.SCM.sample` and
:func:`~andrey.data.sample_scm` draws its mask over the sampled values, and :func:`draw` draws it
over any complete data. The missingness indicator ``R_i`` of column ``i`` is 1 where its value is
missing, and the m-graph adds each indicator to the causal graph with an edge from each of its
parents [1]_; the returned :class:`Missingness` records them:

>>> import numpy as np
>>> import andrey.data as data
>>> from andrey.data import missingness
>>> ds = data.sample_scm(d=6, n=500, seed=0, density=2, missing=missingness.mar(0.2))
>>> ds.missing.mechanism, sorted(ds.missing.parents)
('mar', [0, 2, 3, 5])
>>> int(ds.missing.mask.sum()) == int(np.isnan(ds.data).sum())
True

The four mechanisms differ in what an indicator may depend on [1]_:

- ``"mcar"``, missing completely at random: no indicator has a parent.
- ``"mar"``, missing at random: each indicator's parents are always observed.
- ``"mnar"``, missing not at random: each indicator's parents are other columns that can go
  missing, never its own column.
- ``"self_masking"``: each indicator's parent is its own column, the case in which the joint
  distribution of the variables is not recoverable in general [1]_.

An indicator depends on its parents in one of two forms. ``"logistic"``: ``P(R_i = 1)`` is the
logistic function of a weighted sum of its parents, with weights drawn from a standard normal and
scaled so that the sum has standard deviation 1 over the rows, and an intercept solved so that the
column's mean missing probability over the rows is ``rate``, as in the masks of Muzellec et al.
[2]_. ``"quantile"``: an indicator has one parent, and a value is missing with probability 0.9
when that parent is below its ``q`` quantile over the rows and with probability 0.01 otherwise, the
form of the simulations of Tu et al. [3]_; ``q`` is set so that the expected missing share is
``rate``.

References
----------
.. [1] Mohan, K., Pearl, J., and Tian, J. (2013). Graphical models for inference with missing data.
   Advances in Neural Information Processing Systems 26.
.. [2] Muzellec, B., Josse, J., Boyer, C., and Cuturi, M. (2020). Missing data imputation using
   optimal transport. Proceedings of Machine Learning Research 119 (ICML), 7130-7140.
.. [3] Tu, R., Zhang, C., Ackermann, P., Mohan, K., Kjellstrom, H., and Zhang, K. (2019). Causal
   discovery in the presence of missing data. Proceedings of Machine Learning Research 89
   (AISTATS), 1762-1770.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import expit

from .dataset import Missingness
from .scm import MISSING, component_rng

__all__ = ["Mechanism", "Missingness", "draw", "mar", "mcar", "mnar", "self_masking"]

MECHANISMS = ("mcar", "mar", "mnar", "self_masking")
FORMS = ("logistic", "quantile")

# The probabilities of a missing value when the parent is below its quantile and when it is not,
# in the quantile form: p_missing_h and p_missing_l of the simulation code of Tu et al. (2019),
# github.com/TURuibo/MVPC, src/SyntheticDataGeneration.R.
P_HIGH, P_LOW = 0.9, 0.01
# The fewest observed values a column that can go missing keeps; a draw that leaves fewer is
# drawn again, at most MAX_DRAWS times in all.
MIN_OBSERVED = 2
MAX_DRAWS = 100


@dataclass(frozen=True, kw_only=True)
class Mechanism:
    """How values go missing.

    Built by :func:`mcar`, :func:`mar`, :func:`mnar`, or :func:`self_masking`, and drawn by
    :func:`draw` or ``missing=`` on :meth:`~andrey.data.SCM.sample`.
    """

    name: str
    rate: float
    observed: float
    form: str | None
    parents: tuple[tuple[int, tuple[int, ...]], ...] | None = None


def mcar(rate: float, *, observed: float = 0.0) -> Mechanism:
    """Values missing completely at random.

    Each value of a column that can go missing is missing with probability ``rate``, whatever the
    data.

    Parameters
    ----------
    rate : float
        The probability that a value is missing, in ``(0, 1)``.
    observed : float, default=0.0
        The share of the columns that are always observed, in ``[0, 1)``, rounded to a whole
        number of columns; which ones is drawn.

    Returns
    -------
    Mechanism

    Raises
    ------
    ValueError
        If ``rate`` is not in ``(0, 1)`` or ``observed`` is not in ``[0, 1)``.

    Examples
    --------
    >>> from andrey.data.missingness import mcar
    >>> mcar(0.1).name
    'mcar'
    """
    return _mechanism("mcar", rate, observed, None, None)


def mar(
    rate: float,
    *,
    observed: float = 0.3,
    form: str = "logistic",
    parents: Mapping[int, Sequence[int]] | None = None,
) -> Mechanism:
    """Values missing at random.

    Each indicator depends only on columns that are always observed.

    Parameters
    ----------
    rate : float
        The expected share of missing values in each column that can go missing, in ``(0, 1)``;
        in ``(0.01, 0.9)`` for ``form="quantile"``.
    observed : float, default=0.3
        The share of the columns that are always observed, in ``(0, 1)``, rounded to a whole number
        of columns and at least one; which ones is drawn. ``parents``, when given, decides the
        columns instead.
    form : {"logistic", "quantile"}, default="logistic"
        How an indicator depends on its parents (see :mod:`andrey.data.missingness`). Drawn parents
        are every always-observed column under ``"logistic"``, and one of them under
        ``"quantile"``.
    parents : mapping of int to sequence of int, or None, default=None
        The columns that can go missing, each mapped to the always-observed columns its indicator
        depends on, by column of the data: ``{3: [0]}`` makes column 3 go missing with column 0.
        Every other column is always observed, and no parent can go missing. ``"quantile"`` takes
        one parent per column. ``None`` draws them.

    Returns
    -------
    Mechanism

    Raises
    ------
    ValueError
        If ``rate`` or ``observed`` is out of range, ``form`` is not one of the forms, or
        ``parents`` names a parent that can go missing, a column without parents, or more than one
        parent under ``"quantile"``.

    Examples
    --------
    >>> from andrey.data.missingness import mar
    >>> mar(0.2, form="quantile", parents={3: [0]}).parents
    ((3, (0,)),)
    """
    return _mechanism("mar", rate, observed, form, parents)


def mnar(
    rate: float,
    *,
    observed: float = 0.3,
    form: str = "logistic",
    parents: Mapping[int, Sequence[int]] | None = None,
) -> Mechanism:
    """Values missing not at random.

    Each indicator depends on other columns that can go missing, never on its own column.

    Parameters
    ----------
    rate : float
        The expected share of missing values in each column that can go missing, in ``(0, 1)``;
        in ``(0.01, 0.9)`` for ``form="quantile"``.
    observed : float, default=0.3
        The share of the columns that are always observed, in ``[0, 1)``, rounded to a whole number
        of columns; at least two columns must be able to go missing. Which ones is drawn.
        ``parents``, when given, decides the columns instead.
    form : {"logistic", "quantile"}, default="logistic"
        How an indicator depends on its parents (see :mod:`andrey.data.missingness`). Drawn parents
        are every other column that can go missing under ``"logistic"``, and one of them under
        ``"quantile"``.
    parents : mapping of int to sequence of int, or None, default=None
        The columns that can go missing, each mapped to the other columns that can go missing its
        indicator depends on, by column of the data: ``{3: [5], 5: [3]}``. Every other column is
        always observed. ``"quantile"`` takes one parent per column. ``None`` draws them.

    Returns
    -------
    Mechanism

    Raises
    ------
    ValueError
        If ``rate`` or ``observed`` is out of range, ``form`` is not one of the forms, or
        ``parents`` names a parent that is not another column that can go missing, a column
        without parents, or more than one parent under ``"quantile"``.

    Examples
    --------
    >>> from andrey.data.missingness import mnar
    >>> mnar(0.2, parents={3: [5], 5: [3]}).name
    'mnar'
    """
    return _mechanism("mnar", rate, observed, form, parents)


def self_masking(rate: float, *, observed: float = 0.3, form: str = "logistic") -> Mechanism:
    """Self-masking missing values.

    Each indicator depends on its own column.

    Parameters
    ----------
    rate : float
        The expected share of missing values in each column that can go missing, in ``(0, 1)``;
        in ``(0.01, 0.9)`` for ``form="quantile"``.
    observed : float, default=0.3
        The share of the columns that are always observed, in ``[0, 1)``, rounded to a whole
        number of columns; which ones is drawn.
    form : {"logistic", "quantile"}, default="logistic"
        How an indicator depends on its own column (see :mod:`andrey.data.missingness`).

    Returns
    -------
    Mechanism

    Raises
    ------
    ValueError
        If ``rate`` or ``observed`` is out of range or ``form`` is not one of the forms.

    Examples
    --------
    >>> from andrey.data.missingness import self_masking
    >>> self_masking(0.2).name
    'self_masking'
    """
    return _mechanism("self_masking", rate, observed, form, None)


def draw(data: npt.ArrayLike, mechanism: Mechanism, *, seed: int) -> Missingness:
    """Draw the missing values of ``mechanism`` over complete ``data``.

    The draw comes from its own stream of ``seed``, the one ``missing=`` on
    :meth:`~andrey.data.SCM.sample` takes, so ``draw(ds.data, mechanism, seed=s)`` on a complete
    sample ``ds`` with seed ``s`` and no ``screen`` gives the mask of the sample with
    ``missing=mechanism``. A draw
    that leaves a column that can go missing with fewer than 2 observed values is drawn again from
    the next stream of ``seed``.

    Parameters
    ----------
    data : array-like of shape (n, d)
        Complete data, one row per sample; the mask reads its values.
    mechanism : Mechanism
        How values go missing, from :func:`mcar`, :func:`mar`, :func:`mnar`, or
        :func:`self_masking`.
    seed : int
        The seed of the draw.

    Returns
    -------
    Missingness
        The mask and the parents of each indicator. ``np.where(m.mask, np.nan, data)`` is the data
        with the missing values.

    Raises
    ------
    ValueError
        If ``data`` is not a 2-D array of finite numbers with at least 2 rows, ``parents`` names a
        column ``data`` lacks, the columns cannot hold the mechanism (a ``"mar"`` mask needs one
        always-observed column and one that can go missing, an ``"mnar"`` mask two that can go
        missing), a parent is constant, or no draw in 100 keeps 2 observed values in every column.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey.data.missingness import draw, mcar
    >>> X = np.random.default_rng(0).standard_normal((200, 3))
    >>> m = draw(X, mcar(0.1), seed=0)
    >>> m.mask.shape, sorted(m.parents)
    ((200, 3), [0, 1, 2])
    """
    return draw_from(data, mechanism, np.random.SeedSequence(seed))


def draw_from(
    data: npt.ArrayLike, mechanism: Mechanism, root: np.random.SeedSequence
) -> Missingness:
    """:func:`draw` from the missingness streams of ``root``, ``(MISSING, k)`` for draw ``k``."""
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < MIN_OBSERVED or not np.all(np.isfinite(values)):
        raise ValueError(
            "draw takes complete data, a 2-D array of finite numbers with at least 2 rows"
        )
    n, d = values.shape
    for k in range(MAX_DRAWS):
        rng = component_rng(root, MISSING, k)
        parents = _parents(mechanism, d, rng)
        mask = np.zeros((n, d), dtype=bool)
        for column, pa in parents.items():
            mask[:, column] = rng.random(n) < _probabilities(mechanism, values[:, pa], rng)
        if all(n - mask[:, column].sum() >= MIN_OBSERVED for column in parents):
            return Missingness(
                mask=mask,
                parents=parents,
                mechanism=mechanism.name,
                form=mechanism.form,
                rate=mechanism.rate,
            )
    raise ValueError(
        f"no draw in {MAX_DRAWS} kept {MIN_OBSERVED} observed values in every column that can go "
        f"missing; lower rate={mechanism.rate}"
    )


def _mechanism(
    name: str,
    rate: float,
    observed: float,
    form: str | None,
    parents: Mapping[int, Sequence[int]] | None,
) -> Mechanism:
    """Check and build a mechanism; ``parents`` become sorted ``(column, parents)`` pairs."""
    if form is not None and form not in FORMS:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    low, high = (P_LOW, P_HIGH) if form == "quantile" else (0.0, 1.0)
    if not (np.isfinite(rate) and low < rate < high):
        raise ValueError(f"{name} takes a rate in ({low}, {high}), got {rate}")
    if not (np.isfinite(observed) and (0 < observed < 1 if name == "mar" else 0 <= observed < 1)):
        bounds = "(0, 1)" if name == "mar" else "[0, 1)"
        raise ValueError(f"{name} takes observed in {bounds}, got {observed}")
    pairs = None
    if parents is not None:
        pairs = tuple(
            (int(c), tuple(sorted(int(p) for p in pa))) for c, pa in sorted(parents.items())
        )
        _check_parents(name, form, pairs)
    return Mechanism(
        name=name, rate=float(rate), observed=float(observed), form=form, parents=pairs
    )


def _check_parents(
    name: str, form: str | None, pairs: tuple[tuple[int, tuple[int, ...]], ...]
) -> None:
    """Raise unless ``pairs`` are the indicator parents ``name`` and ``form`` allow."""
    missing = {c for c, _ in pairs}
    for column, pa in pairs:
        if not pa or len(set(pa)) != len(pa):
            raise ValueError(f"parents of column {column} must be distinct columns, got {list(pa)}")
        if form == "quantile" and len(pa) != 1:
            raise ValueError(
                f"form='quantile' takes one parent per column; column {column} has {list(pa)}"
            )
        if name == "mar" and missing & set(pa):
            raise ValueError(
                f"under mar a parent is always observed; column {column} has parents "
                f"{sorted(missing & set(pa))} that can go missing"
            )
        if name == "mnar" and (column in pa or not set(pa) <= missing):
            raise ValueError(
                f"under mnar each parent is another column that can go missing; column {column} "
                f"has parents {list(pa)}, and the columns that can go missing are {sorted(missing)}"
            )


def _parents(mechanism: Mechanism, d: int, rng: np.random.Generator) -> dict[int, list[int]]:
    """Return the parents of each indicator over ``d`` columns: the mechanism's own, or drawn."""
    if mechanism.parents is not None:
        named = {c for c, pa in mechanism.parents for c in (c, *pa)}
        if max(named) >= d or min(named) < 0:
            raise ValueError(f"parents name columns {sorted(named)}, and the data have {d}")
        return {c: list(pa) for c, pa in mechanism.parents}
    k = int(round(mechanism.observed * d))
    if mechanism.name == "mar":
        k = max(k, 1)
    least = {"mcar": 1, "mar": 1, "mnar": 2, "self_masking": 1}[mechanism.name]
    if d - k < least or (mechanism.name == "mar" and k < 1):
        raise ValueError(
            f"{mechanism.name} with observed={mechanism.observed} on {d} columns leaves "
            f"{d - k} that can go missing; it needs {least} and, under mar, one always observed"
        )
    always = np.sort(rng.choice(d, size=k, replace=False))
    partial = [int(c) for c in np.setdiff1d(np.arange(d), always)]
    one = mechanism.form == "quantile"
    out: dict[int, list[int]] = {}
    for column in partial:
        if mechanism.name == "mcar":
            out[column] = []
        elif mechanism.name == "self_masking":
            out[column] = [column]
        else:
            pool = always if mechanism.name == "mar" else [c for c in partial if c != column]
            out[column] = [int(rng.choice(pool))] if one else [int(c) for c in pool]
    return out


def _probabilities(
    mechanism: Mechanism, parents: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Each row's probability that the column is missing, given its ``(n, k)`` parent values."""
    n = parents.shape[0]
    if mechanism.name == "mcar":
        return np.full(n, mechanism.rate)
    if np.any(np.ptp(parents, axis=0) == 0):
        raise ValueError("a parent of a missingness indicator is constant over the rows")
    if mechanism.form == "quantile":
        q = (mechanism.rate - P_LOW) / (P_HIGH - P_LOW)
        below = parents[:, 0] < np.quantile(parents[:, 0], q)
        return np.where(below, P_HIGH, P_LOW)
    score = parents @ rng.standard_normal(parents.shape[1])
    if np.std(score) == 0:
        raise ValueError("the weighted sum of a missingness indicator's parents is constant")
    score = (score - score.mean()) / np.std(score)
    intercept = brentq(lambda b: float(np.mean(expit(score + b))) - mechanism.rate, -50.0, 50.0)
    return expit(score + intercept)
