"""J-PCMCI+: the causal links of time series from several datasets with shared contexts.

J-PCMCI+ runs PCMCI+ on ``M`` datasets of the same system variables. A **time context** varies over
time and is the same in every dataset at each step, a **space context** is fixed within a dataset
and can differ between datasets, and no system variable causes a context. Two dummies stand in for
the contexts nobody measured: the **space dummy** labels the dataset and the **time dummy** the time
step, each one-hot. A dummy link says the variable changes with the dataset or the step; it is not
a causal link.

The nodes are the system variables, the time contexts, the space contexts, and the dummies, in that
order. The system variables and the time contexts have lagged copies; the other nodes have none,
and they enter a test as the same columns at every lag. The phases are those of
:mod:`~andrey.temporal.pcmciplus`, with the skeleton split in three:

1. **Lagged skeleton.** :func:`~andrey.temporal.pcmci.pc1` selects the lagged parents of each system
   variable among the lagged system variables and time contexts, and of each time context among
   the lagged time contexts.
2. **Context pass.** The skeleton (:func:`~andrey.temporal.pcmciplus.skeleton`) tests the links from
   the time contexts, at lags ``0`` to ``tau_max``, and from the space contexts, at lag ``0``, into
   the system variables. Every link between two system variables stays, untested. A subset ``S``
   draws from the lag-0 neighbors, contexts included. The context links that survive are directed
   into the system and join the parents of their system variable.
3. **Dummy pass.** The skeleton tests the links from the dummies into the system variables, at lag
   ``0``. ``S`` draws from the system variables. The links that survive join the parents.
4. **System pass.** The skeleton tests the links between system variables, given the lagged,
   context, and dummy parents of both ends. ``S`` draws from the system variables.
5. **Colliders and rules.** Phases 3 and 4 of PCMCI+ orient the lag-0 links between system
   variables. A triple's last two nodes are system variables, and its first can be a context or a
   dummy, so a link ``C -> X_t - Y_t`` helps orient ``X_t - Y_t``.

Every test reads one stacked design: each dataset's lagged rows, one block after another, so that no
lag window crosses from one dataset into the next. Conditioning on a dummy centers the data within
each dataset or each step. A dummy with ``K`` levels enters a conditioning set as ``K - 1`` columns,
and a test of a dummy is one test of each level against the rest, with the smallest p-value
multiplied by the number of levels (Bonferroni); two levels are one test. Given its dummy, a context
is fixed: it leaves the conditioning set, and a test of it has nothing to read. Only the collider
votes run such a test, and it separates nothing, so a triple that hangs on it stays ambiguous.

A test whose tested column lies in the span of its conditions has no residual to read. A dummy's
level there is a function of the conditions, so it separates. A context there, such as a sine that
its own lags determine, runs again without the older time contexts, where its parents lie; if the
test still has nothing to read, the context cannot be told apart from its conditions, and the link
stays. A system variable there raises, as in :func:`~andrey.temporal.pcmciplus.pcmciplus`, and so
does one with a copy among the conditions, such as a pulse every dataset shares, which copies a
level of the time dummy; a dummy level or a context with a copy there separates.

References
----------
Gunther, W., Ninad, U., and Runge, J. (2023). Causal discovery for time series from multiple
datasets with latent contexts. Proceedings of the 39th Conference on Uncertainty in Artificial
Intelligence (UAI), PMLR 216, 766-776. https://proceedings.mlr.press/v216/gunther23a.html

Runge, J. (2020). Discovering contemporaneous and lagged causal relations in autocorrelated
nonlinear time series datasets. Proceedings of the 36th Conference on Uncertainty in Artificial
Intelligence (UAI), PMLR 124, 1388-1397. https://proceedings.mlr.press/v124/runge20a.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from andrey.constraint._skeleton import check_collider_rule
from andrey.core import backend
from andrey.core.ci import SingularCorrelationError, normal_scores
from andrey.core.fdr import check_fdr_method, corrected_pvalues
from andrey.core.knowledge import LaggedConstraints
from andrey.core.ties import tie_ranks
from andrey.temporal.pcmci import DesignTest, Link, LinkTest, column_test, lagged_design, pc1
from andrey.temporal.pcmciplus import PCMCIPlusResult, Recorder, colliders, orient, skeleton

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

# A context link ``(context, lag, variable)``: ``context`` counts the time contexts, then the space
# contexts, from ``0``.
ContextLink = tuple[int, int, int]


@dataclass(frozen=True)
class JPCMCIPlusResult:
    """The links of one J-PCMCI+ run.

    ``system`` holds the graph over the system variables, as PCMCI+ reports it. ``context_links``
    lists the links from the contexts into the system variables. ``space_links`` and ``time_links``
    list the system variables linked to the space dummy and the time dummy.
    """

    system: PCMCIPlusResult
    context_links: tuple[ContextLink, ...]
    space_links: tuple[int, ...]
    time_links: tuple[int, ...]


@dataclass(frozen=True)
class Nodes:
    """The node numbers of a run: system variables, time contexts, space contexts, dummies."""

    n_system: int
    n_time: int
    n_space: int
    time_dummy: bool

    @property
    def n_series(self) -> int:
        """The nodes with lagged copies, which come first."""
        return self.n_system + self.n_time

    @property
    def space_dummy_node(self) -> int:
        return self.n_series + self.n_space

    @property
    def time_dummy_node(self) -> int | None:
        return self.space_dummy_node + 1 if self.time_dummy else None

    @property
    def count(self) -> int:
        return self.space_dummy_node + 1 + int(self.time_dummy)

    def mask(self, *spans: tuple[int, int]) -> np.ndarray:
        """Mark the nodes in each ``[start, stop)`` span."""
        marked = np.zeros(self.count, dtype=bool)
        for start, stop in spans:
            marked[start:stop] = True
        return marked

    @property
    def system(self) -> np.ndarray:
        return self.mask((0, self.n_system))

    @property
    def contexts(self) -> np.ndarray:
        return self.mask((self.n_system, self.space_dummy_node))

    @property
    def dummies(self) -> np.ndarray:
        return self.mask((self.space_dummy_node, self.count))


def one_hot(levels: np.ndarray) -> np.ndarray:
    """The ``(len(levels), K)`` indicator columns of the ``K`` values of ``levels``."""
    return (levels[:, None] == np.unique(levels)).astype(np.float64)


def pooled_design(
    series: Sequence[np.ndarray], spaces: np.ndarray, nodes: Nodes, tau_max: int
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return the stacked design and the design columns of each node without lagged copies.

    The design stacks each dataset's lagged design over lags ``0`` to ``2 * tau_max``, then holds
    the space contexts, repeated over a dataset's rows, the levels of the space dummy, and the
    levels of the time dummy. A row's time level counts its dataset's rows from ``0``, so equal
    lengths align the steps.
    """
    lagged = [lagged_design(block, 2 * tau_max) for block in series]
    rows = [len(block) for block in lagged]
    dataset = np.repeat(np.arange(len(series)), rows)
    parts = [np.vstack(lagged), spaces[dataset], one_hot(dataset)]
    if nodes.time_dummy:
        parts.append(one_hot(np.concatenate([np.arange(n) for n in rows])))
    start = nodes.n_series * (2 * tau_max + 1)
    columns: list[tuple[int, ...]] = []
    for part in parts[1:]:
        widths = (1,) * part.shape[1] if part is parts[1] else (part.shape[1],)
        for width in widths:
            columns.append(tuple(range(start, start + width)))
            start += width
    return np.hstack(parts), tuple(columns)


SPAN_NOTE = (
    "A system variable is a linear function of other columns, or of a dummy: list one that is the "
    "same in every dataset as a time context, and drop one that sums others."
)


class ContextTest:
    """A :class:`~andrey.temporal.pcmci.LinkTest` over the columns ``lag * nodes.count + node``.

    Each query is translated to the columns of the stacked design and answered by ``test``, as the
    module's docstring describes. ``fixed`` holds the columns of each node without lagged copies,
    and ``rows`` is the number of rows of the design.
    """

    def __init__(
        self,
        test: LinkTest,
        nodes: Nodes,
        fixed: tuple[tuple[int, ...], ...],
        tau_max: int,
        rows: int,
    ) -> None:
        self.test = test
        self.nodes = nodes
        self.fixed = fixed
        self.tau_max = tau_max
        self.rows = rows
        lags = range(2 * tau_max + 1)
        self.system_columns = {
            lag * nodes.n_series + i for lag in lags for i in range(nodes.n_system)
        }
        self.time_columns = {
            lag * nodes.n_series + i for lag in lags for i in range(nodes.n_system, nodes.n_series)
        }
        self.space_columns = {block[0] for block in fixed[: nodes.n_space]}
        self.dummy_nodes = set(range(nodes.space_dummy_node, nodes.count))
        self.copies: dict[int, set[int]] = {}
        if isinstance(test, DesignTest):
            for group in test.test.copies():
                for column in group:
                    self.copies[column] = set(group) - {column}

    def _columns(self, column: int) -> tuple[int, ...]:
        """The design columns of the lagged node ``column``."""
        lag, node = divmod(column, self.nodes.count)
        if node < self.nodes.n_series:
            return (lag * self.nodes.n_series + node,)
        return self.fixed[node - self.nodes.n_series]

    def _levels(self, column: int) -> tuple[int, ...]:
        """The columns a test of ``column`` runs on: two levels of a dummy are one test."""
        columns = self._columns(column)
        return columns[:1] if len(columns) == 2 else columns

    def _given(self, row: Iterable[int]) -> tuple[tuple[int, ...], set[int]]:
        """Return the design columns of the conditioning set ``row``, and its nodes."""
        given: set[int] = set()
        present = set()
        for column in row:
            present.add(column % self.nodes.count)
            columns = self._columns(column)
            given.update(columns[:-1] if column % self.nodes.count in self.dummy_nodes else columns)
        if self.nodes.space_dummy_node in present:
            given -= self.space_columns
        if self.nodes.time_dummy_node in present:
            given -= self.time_columns
        return tuple(sorted(given)), present

    def _check_rows(self, given: tuple[int, ...]) -> None:
        """Raise ``ValueError`` unless a test given the design columns ``given`` keeps a degree of
        freedom, of which Fisher-Z spends three."""
        needed = len(given) + 4
        if self.rows < needed:
            raise ValueError(
                f"jpcmciplus: a test given {len(given)} columns of lagged values, contexts, and "
                f"dummy levels needs at least {needed} rows after the first "
                f"2 * tau_max = {2 * self.tau_max} of each dataset, and the datasets have "
                f"{self.rows}; add datasets or steps, cap the conditions with max_p_global and "
                "max_depth, or pass time_dummy=False"
            )

    def _fixed_by(self, column: int, present: set[int]) -> bool:
        """Whether a dummy in ``present`` fixes the context of the lagged node ``column``."""
        node = column % self.nodes.count
        if self.nodes.n_system <= node < self.nodes.n_series:
            return self.nodes.time_dummy_node in present
        if self.nodes.n_series <= node < self.nodes.space_dummy_node:
            return self.nodes.space_dummy_node in present
        return False

    def _in_span(self, column: np.ndarray, conditions: np.ndarray) -> bool:
        """Whether the design column ``column`` lies in the span of the columns ``conditions``."""
        assert isinstance(self.test, DesignTest)  # only it reports a singular query
        try:
            self.test.test.residual_statistic_pvalues(self.test.design, column, column, conditions)
        except SingularCorrelationError:
            return True
        return False

    def _answer(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """``test``'s answers, with a tested column in the span of its conditions read as the
        module's docstring says."""
        try:
            return self.test.statistic_pvalues(x, y, conditions)
        except SingularCorrelationError:
            pass
        statistics, p_values = np.zeros(len(x)), np.ones(len(x))
        for k in range(len(x)):
            query = (x[k : k + 1], y[k : k + 1], conditions[k : k + 1])
            try:
                statistic, p_value = self.test.statistic_pvalues(*query)
            except SingularCorrelationError as error:
                held = {int(c[0]) for c in query[:2] if self._in_span(c, query[2])}
                if held & self.system_columns:
                    error.add_note(SPAN_NOTE)
                    raise
                if held & (self.time_columns | self.space_columns):
                    statistics[k], p_values[k] = self._determined(*query, held)
                continue
            statistics[k], p_values[k] = statistic[0], p_value[0]
        return statistics, p_values

    def _determined(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray, held: set[int]
    ) -> tuple[float, float]:
        """The answer for a context in ``held``, in the span of its conditions.

        The query runs again without the time contexts older than the tested one, where its
        parents lie. A link that query cannot read either stays.
        """
        n_series = self.nodes.n_series
        lag = max((c // n_series for c in held & self.time_columns), default=None)
        kept = [
            c
            for c in conditions[0].tolist()
            if lag is None or c not in self.time_columns or c // n_series <= lag
        ]
        if len(kept) == conditions.shape[1]:
            return 1.0, 0.0
        try:
            statistic, p_value = self.test.statistic_pvalues(
                x, y, np.array(kept, dtype=np.intp).reshape(1, len(kept))
            )
        except SingularCorrelationError:
            return 1.0, 0.0
        return float(statistic[0]), float(p_value[0])

    def statistic_pvalues(
        self, x: np.ndarray, y: np.ndarray, conditions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Test ``x[i]`` against ``y[i]`` given the columns ``conditions[i]``, for every row."""
        statistics = np.zeros(len(x))
        smallest = np.ones(len(x))
        tests = np.ones(len(x))
        batches: dict[int, list[tuple[int, int, int, tuple[int, ...]]]] = {}
        for query, (a, b, row) in enumerate(
            zip(np.asarray(x).tolist(), np.asarray(y).tolist(), np.asarray(conditions).tolist())
        ):
            given, present = self._given(row)
            if self._fixed_by(a, present) or self._fixed_by(b, present):
                statistics[query], smallest[query] = 1.0, 0.0
                continue
            self._check_rows(given)
            left, right = self._levels(a), self._levels(b)
            tests[query] = len(left) * len(right)
            for column in {left[0], right[0]} & self.system_columns:
                if self.copies.get(column, set()) & set(given):
                    error = SingularCorrelationError(
                        "a tested column lies in the span of its conditioning columns; cannot run "
                        "a partial-correlation test"
                    )
                    error.add_note(SPAN_NOTE)
                    raise error
            for u in left:
                for v in right:
                    if not (self.copies.get(u, set()) | self.copies.get(v, set())) & set(given):
                        batches.setdefault(len(given), []).append((query, u, v, given))
        for size, batch in batches.items():
            queries = [query for query, _, _, _ in batch]
            answers = self._answer(
                np.array([u for _, u, _, _ in batch], dtype=np.intp),
                np.array([v for _, _, v, _ in batch], dtype=np.intp),
                np.array([given for _, _, _, given in batch], dtype=np.intp).reshape(
                    len(batch), size
                ),
            )
            for query, statistic, p_value in zip(queries, *answers):
                if p_value < smallest[query]:
                    smallest[query], statistics[query] = p_value, statistic
        return statistics, np.minimum(1.0, smallest * tests)


def _linked(parents: Sequence[Sequence[Link]], n_nodes: int, tau_max: int) -> np.ndarray:
    """The ``[lag, cause, effect]`` mask of the links in ``parents``."""
    linked = np.zeros((tau_max + 1, n_nodes, n_nodes), dtype=bool)
    for j, links in enumerate(parents):
        for i, lag in links:
            linked[lag, i, j] = True
    return linked


def _with_links(
    parents: Sequence[Sequence[Link]],
    adjacency: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
) -> tuple[tuple[Link, ...], ...]:
    """``parents`` with exactly the links of ``adjacency`` from ``sources`` into ``targets``."""
    updated = []
    for j, links in enumerate(parents):
        kept = [(i, lag) for i, lag in links if not (sources[i] and targets[j])]
        if targets[j]:
            kept += [
                (int(i), lag)
                for lag in range(adjacency.shape[0])
                for i in np.flatnonzero(adjacency[lag, :, j] & sources)
            ]
        updated.append(tuple(kept))
    return tuple(updated)


def _pass_constraints(
    free: np.ndarray, kept: np.ndarray, known: LaggedConstraints
) -> LaggedConstraints:
    """Constraints for a skeleton pass: it tests ``free``, keeps ``kept``, and has no other link.

    ``free`` and ``kept`` are ``[lag, cause, effect]`` masks. ``known`` adds the knowledge.
    """
    forbidden = ~(free | kept) | known.forbidden
    return LaggedConstraints(forbidden, (kept & ~known.forbidden) | known.required, known.tier)


def _embedded(
    constraints: LaggedConstraints | None, nodes: Nodes, tau_max: int
) -> LaggedConstraints:
    """The knowledge over the system variables as constraints over every node."""
    shape = (tau_max + 1, nodes.count, nodes.count)
    forbidden, required = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    tier = np.full(nodes.count, -1, dtype=np.intp)
    if constraints is not None:
        a = nodes.n_system
        forbidden[:, :a, :a] = constraints.forbidden
        required[:, :a, :a] = constraints.required
        tier[:a] = constraints.tier
    return LaggedConstraints(forbidden, required, tier)


def _lagged_knowledge(known: LaggedConstraints, nodes: Nodes) -> LaggedConstraints:
    """Constraints for the lagged phase: JCI's rule that no system variable causes a context."""
    a, n_series = nodes.n_system, nodes.n_series
    forbidden = np.ones_like(known.forbidden)
    forbidden[:, :n_series, :a] = False
    forbidden[:, a:n_series, a:n_series] = False
    return LaggedConstraints(forbidden | known.forbidden, known.required, known.tier)


def run_phases(
    test: LinkTest,
    nodes: Nodes,
    tau_max: int,
    pc_alpha: float,
    collider_rule: str,
    ranks: Sequence[int],
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    fdr_method: str = "none",
) -> JPCMCIPlusResult:
    """Run the phases with ``test`` over the node columns (:class:`ContextTest`)."""
    n_nodes, a = nodes.count, nodes.n_system
    system, contexts, dummies = nodes.system, nodes.contexts, nodes.dummies
    known = _embedded(constraints, nodes, tau_max)
    recorder = Recorder(test)
    lagged = _lagged_knowledge(known, nodes)
    parents = tuple(
        pc1(recorder, n_nodes, j, tau_max, pc_alpha, ranks, lagged, max_depth)
        if j < nodes.n_series
        else ()
        for j in range(n_nodes)
    )

    distinct = ~np.eye(n_nodes, dtype=bool)
    between_system = np.outer(system, system) & distinct

    def tested(lag_0: np.ndarray, lagged_sources: np.ndarray) -> np.ndarray:
        """The links of a pass: ``lag_0`` at lag 0; from ``lagged_sources`` into the system."""
        free = _linked(parents, n_nodes, tau_max) & lagged_sources[None, :, None]
        free &= system[None, None, :]
        free[0] = lag_0
        return free

    # The context pass.
    free = tested(np.outer(contexts, system), contexts)
    kept = _linked(parents, n_nodes, tau_max) & ~free
    kept[0] = between_system
    links, _, _, sepsets = skeleton(
        test,
        n_nodes,
        parents,
        tau_max,
        pc_alpha,
        _pass_constraints(free, kept, known),
        max_p_global,
    )
    parents = _with_links(parents, links, contexts, system)

    # The dummy pass.
    free = tested(np.outer(dummies, system), np.zeros(n_nodes, dtype=bool))
    kept = _linked(parents, n_nodes, tau_max) & ~free
    kept[0] |= between_system
    links, _, _, dummy_sepsets = skeleton(
        test,
        n_nodes,
        parents,
        tau_max,
        pc_alpha,
        _pass_constraints(free, kept, known),
        max_p_global,
        pool=system,
    )
    parents = _with_links(parents, links, dummies, system)

    # The system pass.
    free = tested(between_system, system)
    kept = _linked(parents, n_nodes, tau_max) & ~free
    adjacency, p_values, statistics, system_sepsets = skeleton(
        test,
        n_nodes,
        parents,
        tau_max,
        pc_alpha,
        _pass_constraints(free, kept, known),
        max_p_global,
        pool=system,
    )

    sepsets = {**sepsets, **dummy_sepsets, **system_sepsets}
    for (x, j), (statistic, p_value) in recorder.largest.items():
        lag, i = divmod(x, n_nodes)
        if p_values[lag, i, j] < 0:  # removed in the lagged phase
            p_values[lag, i, j], statistics[lag, i, j] = p_value, statistic
    p_values[p_values < 0] = 1.0
    required = known.required.copy()
    required[0] |= required[0].T
    p_values[required], statistics[required] = 0.0, 1.0

    # Contexts and dummies point into the system; a pair they leave apart keeps its separating set.
    directions = np.zeros_like(known.forbidden)
    directions[0] = np.outer(system, ~system)
    orientation = LaggedConstraints(directions | known.forbidden, known.required, known.tier)
    middles = range(a)
    marks, ambiguous = colliders(
        test,
        n_nodes,
        parents,
        adjacency,
        sepsets,
        pc_alpha,
        collider_rule,
        orientation,
        max_p_global,
        middles,
    )
    orient(marks, adjacency, ambiguous)

    p_system = p_values[:, :a, :a].copy()
    if fdr_method != "none":
        p_system[1:] = corrected_pvalues(
            p_system[1:], fdr_method, None if constraints is None else constraints.forbidden[1:]
        )
    result = PCMCIPlusResult(
        parents=tuple(
            tuple((i, lag) for i, lag in links if i < a and lag > 0) for links in parents[:a]
        ),
        adjacency=adjacency[:, :a, :a],
        marks=marks[:a, :a],
        p_values=p_system,
        statistics=statistics[:, :a, :a],
        sepsets={key: tuple(s) for key, s in sepsets.items() if key[0] < a and key[2] < a},
        ambiguous_triples=tuple(
            sorted(triple for triple in ambiguous if max(triple[0], triple[2], triple[3]) < a)
        ),
    )
    time_node = nodes.time_dummy_node
    return JPCMCIPlusResult(
        system=result,
        context_links=tuple(
            (int(k) - a, lag, int(j))
            for lag in range(tau_max + 1)
            for k, j in np.argwhere(adjacency[lag] & np.outer(contexts, system))
        ),
        space_links=tuple(
            int(j) for j in np.flatnonzero(adjacency[0, nodes.space_dummy_node] & system)
        ),
        time_links=()
        if time_node is None
        else tuple(int(j) for j in np.flatnonzero(adjacency[0, time_node] & system)),
    )


def jpcmciplus(
    series: Sequence[np.ndarray],
    spaces: np.ndarray,
    n_system: int,
    *,
    time_dummy: bool = True,
    tau_max: int = 1,
    pc_alpha: float = 0.01,
    fdr_method: str = "none",
    collider_rule: str = "majority",
    indep_test: str = "student_t",
    constraints: LaggedConstraints | None = None,
    max_depth: int | None = None,
    max_p_global: int | None = None,
    seed: int | None = None,
) -> JPCMCIPlusResult:
    """Run J-PCMCI+ on ``M`` datasets of ``n_system`` system variables and their contexts.

    ``series`` holds one ``(T_m, n_system + n_time)`` array per dataset, the system variables and
    then the time contexts. ``spaces`` holds the ``(M, n_space)`` values of the space contexts.
    ``time_dummy`` adds the time dummy, which needs datasets of one length, aligned in time.
    ``constraints``, background knowledge over the system variables, reach every phase, and
    ``fdr_method``, ``max_depth``, and ``max_p_global`` act as in
    :func:`~andrey.temporal.pcmciplus.pcmciplus`. ``seed`` reaches the test.

    Raises what :func:`~andrey.temporal.pcmci.column_test` raises, and :class:`ValueError` when
    ``collider_rule`` or ``fdr_method`` is not one of its choices.
    """
    check_collider_rule(collider_rule)
    check_fdr_method(fdr_method)
    blocks = [np.asarray(block, dtype=np.float64) for block in series]
    spaces = np.asarray(spaces, dtype=np.float64)
    n_series = blocks[0].shape[1]
    nodes = Nodes(n_system, n_series - n_system, spaces.shape[1], time_dummy)
    design, fixed = pooled_design(blocks, spaces, nodes, tau_max)
    if indep_test == "normal_scores":  # Fisher-Z on the scores, as a test over the design
        design, indep_test = normal_scores(design), "fisherz"
    lagged = n_series * (2 * tau_max + 1)
    built = column_test(indep_test, design, n_series, n_lagged=lagged, seed=seed)
    test = ContextTest(built, nodes, fixed, tau_max, len(design))
    ranks = np.zeros(nodes.count, dtype=np.intp)
    ranks[:n_series] = tie_ranks(np.vstack(blocks))
    with backend.one_blas_thread():  # the stacked inversions then give the same bits on any host
        return run_phases(
            test,
            nodes,
            tau_max,
            pc_alpha,
            collider_rule,
            ranks.tolist(),
            constraints,
            max_depth,
            max_p_global,
            fdr_method,
        )
