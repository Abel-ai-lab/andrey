"""``andrey.markov_check``: test the independences a DAG or CPDAG implies against data.

The function's docstring describes the procedure and cites it.
"""

from __future__ import annotations

import heapq
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from .api._adapt import (
    categorical_metadata,
    checked_data,
    column_labels,
    copies_named,
    default_test,
)
from .core.ci import make_indep_test
from .core.ci_missing import rows_metadata, rows_tested
from .core.output import StructureOutput
from .core.seeding import resolve_seed
from .core.structure import GraphStructure
from .core.warning_policy import SampleSizeWarning

#: The rows per table cell below which a categorical test is called sparse (Cochran 1954).
SPARSE_ROWS_PER_CELL = 5
#: The categorical tests whose table has one cell per combination of levels.
TABLE_TESTS = ("gsq", "chisq")

# A fact: x is independent of w given the parents of x, as node positions.
Fact = tuple[int, int, tuple[int, ...]]


@dataclass(frozen=True, eq=False, repr=False)
class MarkovCheckOutput:
    """The result of :func:`~andrey.markov_check`.

    It holds one p-value for each independence the graph implies, and the test of whether those
    p-values look uniform.

    ``structure`` is the graph as checked, and ``dag`` the DAG whose independences were tested (the
    graph itself for a DAG, a consistent extension for a CPDAG). ``p_values`` and ``table`` list the
    tested independences in order, ``uniformity_statistic`` and ``uniformity_p_value`` are the
    Anderson-Darling test of those p-values against the uniform distribution, ``passed`` is the
    check's verdict, and ``metadata`` records the settings used.
    """

    structure: GraphStructure
    dag: GraphStructure
    alpha: float
    p_values: np.ndarray
    uniformity_statistic: float | None
    uniformity_p_value: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _tests: tuple[Fact, ...] = ()

    @property
    def n_tests(self) -> int:
        """The number of independences tested."""
        return len(self._tests)

    @property
    def share_dependent(self) -> float | None:
        """The share of the tested independences whose p-value is below ``alpha``.

        About ``alpha`` when the graph is right and the tests are calibrated; ``None`` when nothing
        was tested.
        """
        return float(np.mean(self.p_values < self.alpha)) if self.n_tests else None

    @property
    def passed(self) -> bool:
        """``False`` when the p-values are not uniform: the uniformity p-value is at most ``alpha``.

        ``True`` also when the graph implies no independence, so nothing was tested.
        """
        return self.uniformity_p_value is None or self.uniformity_p_value > self.alpha

    @property
    def verdict(self) -> str:
        """One sentence: what was tested and whether the data reject it."""
        n, a = self.n_tests, self.alpha
        if n == 0:
            return (
                "Not rejected: the graph implies no independence among its variables, so the data "
                "cannot reject it."
            )
        assert self.share_dependent is not None and self.uniformity_p_value is not None
        share = f"{100 * self.share_dependent:.1f}% have p < {a:g}, about {100 * a:g}% expected"
        if self.passed:
            return (
                f"Not rejected: the p-values of the {n} independences the graph implies are "
                f"consistent with uniform (Anderson-Darling p = {self.uniformity_p_value:.3g} > "
                f"{a:g}); {share}. A pass does not show the edges are right: a graph with extra "
                "edges also passes."
            )
        return (
            f"Rejected: the p-values of the {n} independences the graph implies are not uniform "
            f"(Anderson-Darling p = {self.uniformity_p_value:.3g} <= {a:g}); {share}."
        )

    def table(self, *, index: bool = False) -> list[dict[str, Any]]:
        """One row per tested independence: ``x``, ``y``, ``given``, and its ``p_value``.

        Nodes are named by label when the graph has labels, and by position with ``index=True``.
        """
        names = self._names(index)
        return [
            {"x": names[x], "y": names[y], "given": [names[z] for z in given], "p_value": float(p)}
            for (x, y, given), p in zip(self._tests, self.p_values.tolist(), strict=True)
        ]

    def _names(self, index: bool) -> Any:
        labels = self.structure.labels
        return range(self.structure.n_nodes) if index or labels is None else labels

    def __repr__(self) -> str:
        """Return the verdict in a constructor-like form."""
        return f"MarkovCheckOutput({self.verdict})"

    def __str__(self) -> str:
        """Return the verdict, then the test count, the statistic, and the rows per test."""
        if self.n_tests == 0:
            return self.verdict
        return (
            f"{self.verdict}\n  {self.n_tests} tests | Anderson-Darling A^2 = "
            f"{self.uniformity_statistic:.3g} | each test on "
            f"{self.metadata['rows_per_test']} rows | indep_test {self.metadata['indep_test']}"
        )


def markov_check(
    graph: StructureOutput | GraphStructure,
    data: npt.ArrayLike,
    *,
    indep_test: str | None = None,
    alpha: float = 0.05,
    categorical: Sequence[str | int] | None = None,
    sample_fraction: float = 0.5,
    seed: int | None = None,
) -> MarkovCheckOutput:
    """Test whether the independences a DAG or CPDAG implies hold in the data.

    A graph implies independences: for a DAG, each variable is independent of every variable that
    comes earlier in a topological order and is not its parent, given its parents (the ordered
    local Markov property [1]_). The check tests each one, a pair of variables and their
    conditioning set, with a conditional-independence test, and asks whether the p-values are
    uniform on ``(0, 1)``, as they are when every tested independence holds: the Anderson-Darling
    test of uniformity, with Marsaglia and Marsaglia's p-value [2]_. A small uniformity p-value, at
    most ``alpha``, rejects the graph together with the test's assumptions [1]_. A CPDAG is checked
    through a DAG of its class, which implies the same independences. The number of tests is
    ``d (d - 1) / 2`` minus the number of edges for ``d`` variables.

    Use rows the graph was not learned from [1]_: a graph fitted to the same rows can pass because
    the fit made its independences true there. Split the rows, learn on one half and check on the
    other.

    Parameters
    ----------
    graph : StructureOutput or GraphStructure
        The graph to check, of kind ``dag`` or ``cpdag``: a method's result, or its ``structure``.
        A PAG, MAG, undirected, or cyclic graph, and a time-series result, raise ``ValueError``
        (the check is validated for a DAG or CPDAG [1]_).
    data : array-like of shape (n_samples, n_variables)
        The data to check against, as the methods take it. When the graph and the data both have
        labels (a DataFrame's column names), a node is read from the column of its label and the
        two must name the same columns; otherwise the n-th node is the n-th column, and the counts
        must agree. Missing values (``NaN``) are deleted test by test, assuming they are missing
        completely at random.
    indep_test : str or None, default=None
        The conditional-independence test, as ``pc`` takes it. ``None`` picks from the declared
        column types: ``"gsq"`` when every column is categorical, ``"cond_gauss"`` when some are,
        else ``"fisherz"``. ``metadata["indep_test"]`` names the one used. See Notes.
    alpha : float, default=0.05
        The level of the uniformity test, in ``(0, 1)``, and the level at which
        ``share_dependent`` counts a p-value as small.
    categorical : sequence of str or int or None, default=None
        The categorical columns, by label or 0-based position, as the methods take them; ``None``
        reads them from a DataFrame's ``category``, ``bool``, and text columns, and ``[]`` declares
        none.
    sample_fraction : float, default=0.5
        The share of the rows each test reads, in ``(0, 1]``: a random subset drawn without
        replacement for each test, so that the tests do not all read the same rows [1]_. ``1.0``
        reads every row in every test. See Notes. The same meaning as in :func:`~andrey.bootstrap`.
    seed : int or None, default=None
        The seed of the row draws and of the randomized tests, a non-negative integer. ``None``
        uses ``ANDREY_SEED``, or 0 when it is unset, so a check repeats.

    Returns
    -------
    MarkovCheckOutput
        The uniformity p-value, the verdict ``passed``, and the p-value of every tested
        independence (``table()``). ``metadata`` records ``indep_test``, ``seed``,
        ``sample_fraction``, ``rows_per_test``, the topological ``order`` that fixed the
        independences, and ``rows_tested`` when missing values were deleted.

    Raises
    ------
    ValueError
        If ``graph`` is not a DAG or a CPDAG that has a DAG, if the graph and the data disagree on
        the columns, if an argument is out of range, if a test would read fewer than the
        conditioning set size plus four rows, or if a test returns a p-value outside ``[0, 1]``.

    Warns
    -----
    SampleSizeWarning
        With a categorical test (``"gsq"``, ``"chisq"``) that reads a table with fewer than 5 rows
        per cell: the test then rejects true independences more often than ``alpha``, so a
        rejection is not evidence against the graph. See Notes.

    Notes
    -----
    **Rows per test.** The tests read the same data, so their p-values are uniform one by one but
    not independent, and the Anderson-Darling p-value assumes independence. With every test on every
    row (``sample_fraction=1.0``, the script of [1]_ and the default of Tetrad), the check rejected
    a true graph in 9% to 21% of 1,000 simulated linear datasets per cell at ``alpha=0.05``, more
    with more tests (10 and 20 variables, average degree 2 and 4, 200 and 1,000 rows). With each
    test on its own random half of the rows (the default, as in the text of [1]_), the rate was 6%
    to 9%, and with each test on new rows it was 5%. A subsample costs power: one missing edge was
    rejected in 96% to 99% of the datasets at 1,000 rows (98% to 100% on all rows) and in 72% to
    82% at 200 rows (89% to 93%). The verdict also depends on ``seed``, because the row draws do.

    **The uniformity p-value.** The Anderson-Darling p-value is exact for one tested independence.
    For two or more it is Marsaglia and Marsaglia's approximation [2]_, which was within 0.002 of
    the level at 0.01, 0.05, and 0.2 in 200,000 or more simulated sets of uniform p-values for each
    of 2 to 400 independences, and looser in the middle of its range (off by up to 0.017 at two).

    **What a pass shows.** A graph with extra edges also passes, since an added edge removes
    independences without making any other false, so a pass is not evidence that the edges are right
    [1]_; compare graphs of the same data by the check and by their number of edges. Missing edges
    and wrongly oriented v-structures are rejected. Which independences are tested follows a
    topological order broken by the nodes' labels (their positions without labels), so the order
    that fixed them is in ``metadata["order"]``.

    **Tests on categorical data.** The G-square test (``"gsq"``) and the chi-square test
    (``"chisq"``) reject true independences more often than ``alpha`` on tables with few rows per
    cell, so the check does too: with 3 to 5 categories per variable and 500 to 2,000 rows, 69% to
    86% of the true graphs were rejected, and with 2 to 3 categories, 24% at 500 rows, 12% at 2,000,
    and 10% at 5,000. A warning names the case. A mixed-data or kernel test is allowed and was not
    measured; a test whose p-values are discrete or concentrated at 1 departs from the uniform
    distribution the check assumes.

    **Limitations.** Measured on linear data with Gaussian and uniform noise simulated on random
    graphs (10 to 40 variables, average degree 2 and 4, 200 to 2,000 rows, ``"fisherz"``), at
    ``alpha=0.05``, with every test on all rows unless noted. A graph in the true graph's
    equivalence class passes, as it must: reversing an edge without changing the CPDAG was rejected
    in 0% to 24% of the datasets in cells of 25 or more, and a reversal that changes it in 70% to
    100%. Two extra edges were rejected in 6% to 20%, the rate of the true graph. The DAG of the
    observed variables of a model with hidden confounders was rejected in every one of 500
    datasets. ``pc`` outputs were rejected in 57% to 70% of the datasets (89% to 94% when an
    adjacency was missing, 15% to 26% when none was); ``ges`` outputs near the rate of a true
    graph, since its misses are weak edges whose loss leaves no detectable independence; and
    ``direct_lingam`` outputs on Gaussian data near that rate too. Many tests dilute one false
    independence: one missing edge among 740 tests (40 variables) was rejected in 75% to 80% of the
    datasets, against 97% to 100% among 35 to 170. The pairwise tests miss a dependence that shows
    only in a set of variables [1]_, and a test misses a dependence it cannot see, such as
    ``"fisherz"`` on a nonlinear one, so the check misses that missing edge too. The check was not
    measured on data with mixed types, on small categorical tables beyond the cases above, or for
    effect sizes.

    References
    ----------
    .. [1] Ramsey, J. D., Andrews, B., and Spirtes, P. (2024). Choosing DAG models using Markov
       and minimal edge count in the absence of ground truth. arXiv:2409.20187.
    .. [2] Marsaglia, G., and Marsaglia, J. (2004). Evaluating the Anderson-Darling distribution.
       Journal of Statistical Software 9(2), 1-5.

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import GraphStructure, markov_check
    >>> rng = np.random.default_rng(0)
    >>> a = rng.standard_normal(500)
    >>> b = a + rng.standard_normal(500)
    >>> c = b + rng.standard_normal(500)
    >>> data = np.column_stack([a, b, c])
    >>> chain = GraphStructure.from_numpy(
    ...     np.array([[0, 1, 0], [2, 0, 1], [0, 2, 0]]), kind="dag", labels=("a", "b", "c")
    ... )
    >>> markov_check(chain, data, sample_fraction=1.0).passed
    True
    """
    from .spec.concepts import MARKOV_CHECK_PARAMS

    checked: dict[str, Any] = {
        name: MARKOV_CHECK_PARAMS[name].check(value, "markov_check")
        for name, value in (
            ("alpha", alpha),
            ("sample_fraction", sample_fraction),
            ("seed", seed),
            ("indep_test", indep_test),
        )
    }
    alpha, sample_fraction = float(checked["alpha"]), float(checked["sample_fraction"])
    structure = _graph_of(graph)
    dag = structure if structure.kind == "dag" else structure.to_dag(tie_break="label")
    seed = resolve_seed(checked["seed"])

    test_name, categorical = default_test(data, checked["indep_test"], categorical=categorical)
    X, mask = checked_data(
        data, categorical, "indep_test", test_name, method="markov_check", missing=True
    )
    columns = _columns(dag, column_labels(data), X.shape[1])
    parents = [[int(u) for u in dag.parents(v, index=True)] for v in range(dag.n_nodes)]
    order = _order(dag, parents)
    facts = _facts(order, parents)
    rows_per_test = round(sample_fraction * X.shape[0])
    if facts and rows_per_test < max(len(z) for _, _, z in facts) + 4:
        raise ValueError(
            f"markov_check: each test reads {rows_per_test} rows, fewer than the largest "
            f"conditioning set size plus four; use more rows or a larger sample_fraction"
        )

    with copies_named(column_labels(data)), rows_tested() as rows:
        p_values = _p_values(X, mask, columns, facts, test_name, sample_fraction, seed)
    if not np.all((p_values >= 0) & (p_values <= 1)):  # also catches NaN
        raise ValueError(f"markov_check: the {test_name!r} test returned a p-value outside [0, 1]")
    if test_name in TABLE_TESTS:
        _warn_sparse(X, columns, facts, test_name, rows_per_test)

    statistic, p_uniform = anderson_darling_uniform(p_values) if facts else (None, None)
    labels = dag.labels
    metadata: dict[str, Any] = {
        "indep_test": test_name,
        "seed": seed,
        "sample_fraction": sample_fraction,
        "rows_per_test": rows_per_test,
        "order": [labels[v] if labels is not None else v for v in order],
        **categorical_metadata(mask, data),
        **rows_metadata(rows),
    }
    return MarkovCheckOutput(
        structure=structure,
        dag=dag,
        alpha=alpha,
        p_values=p_values,
        uniformity_statistic=statistic,
        uniformity_p_value=p_uniform,
        metadata=metadata,
        _tests=tuple(facts),
    )


def _graph_of(graph: StructureOutput | GraphStructure) -> GraphStructure:
    """Return the graph to check: a result's structure, which must be a dag or a cpdag."""
    structure = graph.structure if isinstance(graph, StructureOutput) else graph
    if not isinstance(structure, GraphStructure) or structure.kind not in ("dag", "cpdag"):
        kind = getattr(structure, "kind", type(structure).__name__)
        raise ValueError(
            f"markov_check checks a dag or a cpdag, got {kind!r}: the check is validated for "
            "those two kinds (Ramsey, Andrews, and Spirtes 2024)"
        )
    return structure


def _columns(dag: GraphStructure, labels: tuple[str, ...] | None, n_columns: int) -> list[int]:
    """Return the data column of each node: by label if both have labels, else by position."""
    if n_columns != dag.n_nodes:
        raise ValueError(f"the graph has {dag.n_nodes} nodes but the data has {n_columns} columns")
    if dag.labels is None or labels is None:
        return list(range(n_columns))
    if set(dag.labels) != set(labels):
        missing = sorted(set(dag.labels) - set(labels))
        extra = sorted(set(labels) - set(dag.labels))
        raise ValueError(
            f"the graph's labels and the data's column names differ: the data lack {missing} "
            f"and have {extra} the graph lacks"
        )
    where = {label: j for j, label in enumerate(labels)}
    return [where[label] for label in dag.labels]


def _order(dag: GraphStructure, parents: list[list[int]]) -> list[int]:
    """Return Kahn's topological order of ``dag``, the lowest label (position) first among ties.

    The columns' order does not change it, only the nodes' labels do.
    """
    children: list[list[int]] = [[] for _ in parents]
    waiting = [len(p) for p in parents]
    for j, ps in enumerate(parents):
        for i in ps:
            children[i].append(j)
    rank: list[Any] = list(dag.labels) if dag.labels is not None else list(range(dag.n_nodes))
    ready = [(rank[v], v) for v, k in enumerate(waiting) if k == 0]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        _, v = heapq.heappop(ready)
        order.append(v)
        for c in children[v]:
            waiting[c] -= 1
            if waiting[c] == 0:
                heapq.heappush(ready, (rank[c], c))
    if len(order) != dag.n_nodes:
        raise ValueError("markov_check: the dag has a directed cycle")
    return order


def _facts(order: list[int], parents: list[list[int]]) -> list[Fact]:
    """Return the ordered local Markov independences.

    They are ``x`` and each earlier non-parent ``w``, given the parents of ``x``.
    """
    facts: list[Fact] = []
    for k, x in enumerate(order):
        given = tuple(parents[x])
        facts += [(x, w, given) for w in order[:k] if w not in given]
    return facts


def _p_values(
    X: np.ndarray,
    mask: np.ndarray | None,
    columns: list[int],
    facts: list[Fact],
    test_name: str,
    sample_fraction: float,
    seed: int,
) -> np.ndarray:
    """Return the p-value of each fact.

    It comes from one test over all rows, or from a test over a random subset of the rows drawn for
    the fact alone.
    """
    if sample_fraction == 1.0:
        test = make_indep_test(test_name, X, categorical=mask, seed=seed)
        return np.array(
            [test(columns[x], columns[w], [columns[z] for z in given]) for x, w, given in facts],
            dtype=np.float64,
        )
    n = X.shape[0]
    kept = round(sample_fraction * n)
    rng = np.random.default_rng(seed)
    out = np.empty(len(facts), dtype=np.float64)
    for k, (x, w, given) in enumerate(facts):
        cols = [columns[x], columns[w], *(columns[z] for z in given)]
        rows = np.sort(rng.choice(n, size=kept, replace=False))
        test = make_indep_test(
            test_name,
            X[rows[:, None], cols],
            categorical=None if mask is None else mask[cols],
            seed=seed,
        )
        out[k] = test(0, 1, list(range(2, len(cols))))
    return out


def _warn_sparse(
    X: np.ndarray, columns: list[int], facts: list[Fact], test_name: str, rows_per_test: int
) -> None:
    """Warn when a categorical test reads a table with few rows per cell.

    A table is sparse below ``SPARSE_ROWS_PER_CELL`` rows per cell, counting the cells as the
    product of the variables' numbers of levels.
    """
    levels = [max(1, len(np.unique(X[~np.isnan(X[:, j]), j]))) for j in range(X.shape[1])]
    per_cell = [
        rows_per_test / math.prod(levels[columns[v]] for v in (x, w, *given))
        for x, w, given in facts
    ]
    sparse = [r for r in per_cell if r < SPARSE_ROWS_PER_CELL]
    if sparse:
        warnings.warn(
            f"{len(sparse)} of the {len(facts)} tests read a table with fewer than "
            f"{SPARSE_ROWS_PER_CELL} rows per cell (as few as {min(sparse):.2g}). On sparse tables "
            f"the {test_name!r} test rejects true independences more often than alpha, so a "
            "rejection of the graph here is not evidence against it.",
            SampleSizeWarning,
            stacklevel=3,
        )


def anderson_darling_uniform(p_values: npt.ArrayLike) -> tuple[float, float]:
    """Return the Anderson-Darling statistic of ``p_values`` against U(0, 1) and its p-value.

    The p-value is exact for one value. For two or more it is one minus the distribution function of
    Marsaglia and Marsaglia (2004): their asymptotic ``ADinf``, corrected for the number of values
    by ``errfix``. That approximation is close where a verdict is made: the share of null statistics
    with a p-value at most 0.01, 0.05, and 0.2 was within 0.002 of those levels for 2 to 400 values
    (200,000 or more simulated samples each). Its middle is looser, off by up to 0.017 at two values
    and 0.003 at twenty.
    """
    u = np.sort(np.clip(np.asarray(p_values, dtype=np.float64), 1e-300, 1 - 1e-16))
    n = u.size
    i = np.arange(1, n + 1)
    a2 = float(-n - np.sum((2 * i - 1) * (np.log(u) + np.log1p(-u[::-1]))) / n)
    if n == 1:  # A^2 = -1 - ln(u (1 - u)) >= ln 4 - 1, so P(A^2 >= a) = 1 - sqrt(1 - 4 e^(-1-a))
        return a2, 1.0 - math.sqrt(max(0.0, 1.0 - 4.0 * math.exp(-1.0 - a2)))
    if a2 <= 0:
        return a2, 1.0
    x = _adinf(a2)
    return a2, min(1.0, max(0.0, 1.0 - (x + _errfix(n, x))))


def _adinf(z: float) -> float:
    """Marsaglia and Marsaglia's (2004) asymptotic distribution function of A^2 at ``z > 0``."""
    if z < 2:
        poly = (
            2.00012
            + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z
        )
        return math.exp(-1.2337141 / z) / math.sqrt(z) * poly
    inner = 1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z
    return math.exp(-math.exp(inner))


def _errfix(n: int, x: float) -> float:
    """Marsaglia and Marsaglia's (2004) correction of ``_adinf``'s value ``x`` for ``n`` values."""
    c = 0.01265 + 0.1757 / n
    if x < c:
        t = x / c
        t = math.sqrt(t) * (1 - t) * (49 * t - 102)
        return t * (0.0037 / n**2 + 0.00078 / n + 0.00006) / n
    if x < 0.8:
        t = (x - c) / (0.8 - c)
        t = -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t
        return t * (0.04213 / n + 0.01365 / n**2) / n
    poly = (
        -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * x) * x) * x) * x) * x
    )
    return poly / n
