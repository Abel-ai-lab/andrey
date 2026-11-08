"""Time series from several datasets with shared contexts, and their true links.

:func:`sample_context_series` draws the linear model that Gunther, Ninad, and Runge (2023) describe:
each system variable depends on its own past, on other system variables at lag ``0`` or a positive
lag, and on at most one context. A time context varies over time and is the same in every dataset; a
space context is fixed within a dataset. Contexts are standard normal and have no parents. Each
context drives exactly one system variable here, so no hidden context is a common cause of two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_DRAWS = 1000  # draws of the weights before giving up on a stationary model
_BURN_IN = 100  # steps dropped from the start of each series


@dataclass(frozen=True)
class ContextSeries:
    """Datasets of system variables and observed contexts, with the model's links.

    ``datasets`` holds one ``(n_steps, n_columns)`` array per dataset: the system variables, then
    the observed time contexts, then the observed space contexts, whose positions are
    ``time_contexts`` and ``space_contexts``. ``links`` holds the ``(cause, effect, lag)`` links
    between system variables, a variable's link to its own past included. ``context_links`` holds
    the ``(context, lag, variable)`` links from the observed contexts, by column position, and
    ``hidden_links`` the ``(lag, variable)`` links from the contexts the datasets leave out.
    """

    datasets: tuple[np.ndarray, ...]
    time_contexts: tuple[int, ...]
    space_contexts: tuple[int, ...]
    links: frozenset[tuple[int, int, int]]
    context_links: frozenset[tuple[int, int, int]]
    hidden_links: frozenset[tuple[int, int]]


def _lag(rng: np.random.Generator, tau_max: int) -> int:
    """Lag ``0`` half the time, else a uniform lag from ``1`` to ``tau_max``."""
    return 0 if rng.random() < 0.5 else int(rng.integers(1, tau_max + 1))


def _system_weights(
    n_system: int, n_links: int, tau_max: int, rng: np.random.Generator
) -> np.ndarray:
    """Return the ``(tau_max + 1, n_system, n_system)`` weights ``[lag, effect, cause]``."""
    weights = np.zeros((tau_max + 1, n_system, n_system))
    weights[1][np.diag_indices(n_system)] = rng.uniform(0.3, 0.8, n_system)
    position = np.argsort(rng.permutation(n_system))
    placed = 0
    while placed < n_links:
        cause, effect = (int(i) for i in rng.choice(n_system, 2, replace=False))
        lag = _lag(rng, tau_max)
        if lag == 0 and position[cause] > position[effect]:
            cause, effect = effect, cause
        if weights[:, effect, cause].any() or weights[:, cause, effect].any():
            continue
        weights[lag, effect, cause] = rng.uniform(0.5, 0.9)
        placed += 1
    return weights


def _stationary(weights: np.ndarray) -> bool:
    """Whether the reduced form of ``weights`` has spectral radius below one."""
    n = weights.shape[1]
    mixing = np.linalg.inv(np.eye(n) - weights[0])
    blocks = [mixing @ block for block in weights[1:]]
    top = np.hstack(blocks)
    lower = np.eye(n * (len(blocks) - 1), n * len(blocks))
    return bool(np.abs(np.linalg.eigvals(np.vstack([top, lower]))).max() < 1)


def sample_context_series(
    n_system: int = 5,
    n_time: int = 2,
    n_space: int = 1,
    *,
    n_datasets: int = 10,
    n_steps: int = 100,
    tau_max: int = 2,
    n_links: int | None = None,
    observed: float = 1.0,
    seed: int = 0,
) -> ContextSeries:
    """Draw linear time series from ``n_datasets`` datasets with shared contexts.

    Each system variable has a link to its own past with weight in ``[0.3, 0.8]``. ``n_links``
    further links, ``n_system`` by default, and one link from each context, into distinct system
    variables, have weights in ``[0.5, 0.9]``; half are at lag ``0`` and the rest uniform in ``1``
    to ``tau_max``, and a space context's is at lag ``0``. The noise is standard normal. Each
    system variable is rescaled to variance one across the datasets, so its variance does not tell
    the causal order. ``observed`` is the fraction of the contexts the datasets hold, the rest
    chosen at random.

    Raises :class:`ValueError` when there are more contexts than system variables or more links
    than pairs of them, or when no draw of the weights is stationary.
    """
    n_contexts = n_time + n_space
    if n_contexts > n_system:
        raise ValueError(f"at most {n_system} contexts, one per system variable, got {n_contexts}")
    pairs = n_system * (n_system - 1) // 2
    n_links = n_system if n_links is None else n_links
    if n_links > pairs:
        raise ValueError(f"at most {pairs} links between {n_system} variables, got {n_links}")
    rng = np.random.default_rng(seed)
    for _ in range(_DRAWS):
        weights = _system_weights(n_system, n_links, tau_max, rng)
        if _stationary(weights):
            break
    else:
        raise ValueError(f"no stationary model in {_DRAWS} draws; lower n_links or tau_max")

    targets = rng.permutation(n_system)[:n_contexts].tolist()
    lags = [_lag(rng, tau_max) for _ in range(n_time)] + [0] * n_space
    strengths = rng.uniform(0.5, 0.9, n_contexts)
    shown = rng.permutation(n_contexts)[: round(observed * n_contexts)].tolist()

    steps = n_steps + _BURN_IN
    time_values = rng.standard_normal((steps, n_time))
    space_values = rng.standard_normal((n_datasets, n_space))
    mixing = np.linalg.inv(np.eye(n_system) - weights[0])
    series = np.zeros((steps, n_datasets, n_system))
    for t in range(steps):
        drive = rng.standard_normal((n_datasets, n_system))
        for lag in range(1, tau_max + 1):
            if t >= lag:
                drive += series[t - lag] @ weights[lag].T
        for k in range(n_time):
            if t >= lags[k]:
                drive[:, targets[k]] += strengths[k] * time_values[t - lags[k], k]
        for k in range(n_space):
            drive[:, targets[n_time + k]] += strengths[n_time + k] * space_values[:, k]
        series[t] = drive @ mixing.T
    series = series[_BURN_IN:]
    series /= series.reshape(-1, n_system).std(axis=0)

    time_shown = [k for k in range(n_time) if k in shown]
    space_shown = [k for k in range(n_space) if n_time + k in shown]
    start = n_system + len(time_shown)
    time_columns = tuple(range(n_system, start))
    space_columns = tuple(range(start, start + len(space_shown)))
    datasets = tuple(
        np.column_stack(
            [
                series[:, m],
                time_values[_BURN_IN:, time_shown],
                np.tile(space_values[m, space_shown], (n_steps, 1)),
            ]
        )
        for m in range(n_datasets)
    )
    column = {
        **{k: time_columns[i] for i, k in enumerate(time_shown)},
        **{n_time + k: space_columns[i] for i, k in enumerate(space_shown)},
    }
    return ContextSeries(
        datasets=datasets,
        time_contexts=time_columns,
        space_contexts=space_columns,
        links=frozenset(
            (int(cause), int(effect), int(lag)) for lag, effect, cause in np.argwhere(weights != 0)
        ),
        context_links=frozenset(
            (column[k], lags[k], targets[k]) for k in range(n_contexts) if k in column
        ),
        hidden_links=frozenset((lags[k], targets[k]) for k in range(n_contexts) if k not in column),
    )
