"""The simulated time series of the J-PCMCI+ paper's numerical experiments, from its authors' code.

Günther, Ninad, and Runge (2023, UAI) evaluate J-PCMCI+ on data from the script ``generate_data.py``
of their repository (guenwi/J-PCMCIplus). :func:`jpcmciplus2023` follows that script's
``generate_random_contemp_context_model`` and ``generate_data``, in the order of its random draws,
so a seed gives the model the script draws for ``SeedSequence(seed)``:

- ``n_system`` system variables, each with a link to its own past at lag 1 with weight 0.5 or 0.7,
  and ``n_system`` cross-links between them along one random causal order, with weight -0.5 or
  0.5. The script asks for all of them at lag 0, but its lag draw compares a pair with a list of
  triples, a test that never holds, so each gets a lag drawn from 1 to ``tau_max``: the data the
  paper's figures come from have no same-step links between system variables, and this model
  draws the same.
- two time contexts, each with a link to its own past at lag 1 (0.5 or 0.7) and one link between
  them at a lag from 1 to ``tau_max``; one series, the same in every dataset.
- two space contexts with one same-step link between them, one draw per dataset.
- ``2 * k`` links from contexts into system variables (weight -0.5 or 0.5), where ``k`` counts the
  context slots, a context at a lag from 0 to ``tau_max`` (a space context at lag 0), that are not
  a parent of another context. A system variable can have several context parents, and a context
  several children.
- Gaussian noise of standard deviation 0.5 or 2, drawn per variable; a burn-in of 20% of the steps,
  dropped; no rescaling.

``int(observed * 2)`` contexts of each kind are observed, chosen at random as the script chooses
them. The cross-links follow one causal order and the contexts drive the system without feedback,
so every draw is stationary and the script's redraw of a model whose data overflow never applies.

:func:`jpcmciplus2023_nonlinear` is the script's fixed nonlinear model (``is_simple_exp=2``), the
data of the supplement's Figure 6: two system variables, one time and one space context, a square
link, and standard normal noise. The supplement's equation (4) writes another model; the figure
comes from this one.

The paper's Assumption 3 forbids a hidden context that confounds an observed context and a system
variable. Here that happens when a hidden context is a parent of an observed one and reaches a
system variable through hidden contexts only; ``breaks_assumption_3`` records it.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

#: The script's draws: link weights, autodependencies, and noise standard deviations.
WEIGHTS = (-0.5, 0.5)
AUTO_WEIGHTS = (0.5, 0.7)
NOISE_SD = (0.5, 2.0)
N_TIME = N_SPACE = 2
BURN_IN = 0.2


@dataclass(frozen=True)
class ContextDraw:
    """One draw: the observed datasets and the true links over their columns.

    ``datasets`` has shape ``(n_datasets, n_steps, n_system + observed contexts)``: the system
    variables, then the observed time contexts at ``time_contexts``, then the observed space
    contexts at ``space_contexts``. ``links`` holds the ``(cause, effect, lag)`` links between
    system variables, a variable's link to its own past included; ``context_links`` the ``(column,
    lag, variable)`` links from observed contexts, a space context's at lag 0. ``model`` lists every
    link of the model as ``[cause, effect, lag, weight, function]`` over its nodes, the system
    variables first, then the time and the space contexts, and ``observed`` the contexts observed,
    by node.
    """

    datasets: np.ndarray
    n_system: int
    time_contexts: tuple[int, ...]
    space_contexts: tuple[int, ...]
    links: frozenset[tuple[int, int, int]]
    context_links: frozenset[tuple[int, int, int]]
    breaks_assumption_3: bool
    model: tuple[tuple, ...]
    observed: tuple[int, ...]
    noise_sd: tuple[float, ...]


# A model is ``{node: [((cause, -lag), weight, function), ...]}``, the script's layout.

_FUNCTIONS = {"linear": lambda x: x, "square": np.square}


def _structural_process(nodes, n_links, contemp_fraction, max_lag, rng):
    """The script's ``generate_structural_causal_process``: links and noise sds over `nodes`."""
    if max_lag == 0:
        contemp_fraction = 1.0
    pairs = list(itertools.combinations(nodes, 2))
    possible = min(n_links, len(pairs))
    n_contemp = int(contemp_fraction * possible)
    n_lagged = possible - n_contemp
    order = list(rng.permutation(nodes))
    links = {i: [] for i in nodes}
    if max_lag > 0:
        for i in order:
            links[int(i)].append(((int(i), -1), float(rng.choice(AUTO_WEIGHTS)), "linear"))
    ordered = list(itertools.combinations(range(len(nodes)), 2))
    rng.shuffle(ordered)
    contemp = [(order[a], order[b]) for a, b in ordered[:n_contemp]]
    unordered = list(set(itertools.permutations(range(len(nodes)), 2)) - set(ordered[:n_contemp]))
    rng.shuffle(unordered)
    lagged = [(order[a], order[b]) for a, b in unordered[:n_lagged]]
    for i, j in lagged + contemp:
        # The script tests membership in a list of triples here, which never holds: every link
        # takes a positive lag unless max_lag is 0.
        lag = 0 if max_lag == 0 else int(rng.integers(1, max_lag + 1))
        weight = float(rng.choice(WEIGHTS))
        rng.choice(["linear"])  # the script's draw of the function, kept for the stream
        links[int(j)].append(((int(i), -lag), weight, "linear"))
    sds = []
    for _ in links:
        rng.choice(["gaussian"]), rng.choice([0.0])  # the script's distribution and mean draws
        sds.append(float(rng.choice(NOISE_SD)))
    return links, sds


def _non_parents(links, tau_max):
    """The script's ``get_non_parents``: the ``(node, lag)`` slots no link of `links` starts at."""
    parents = {cause for node in links.values() for cause, _, _ in node}
    return [(v, tau) for v in links for tau in range(tau_max + 1) if (v, -tau) not in parents]


def _context_links(system, n_links, slots, tau_max, rng):
    """The script's ``set_links``: `n_links` distinct links from context `slots` into `system`."""
    chosen = []
    for _ in range(n_links):
        while True:
            cause, lag = (int(v) for v in rng.choice(slots))
            effect = int(rng.choice(system))
            if (cause, effect, lag) not in chosen:
                break
        chosen.append((cause, effect, lag))
    links = {j: [] for j in system}
    for cause, effect, lag in chosen:
        weight = float(rng.choice(WEIGHTS))
        rng.choice(["linear"])
        links[effect].append(((cause, -lag), weight, "linear"))
    return links


def _same_step_order(links, nodes):
    """`nodes` in an order where every same-step parent comes first."""
    order, placed = [], set()
    while len(order) < len(nodes):
        ready = [
            j
            for j in nodes
            if j not in placed
            and all(c in placed or c not in nodes for (c, lag), _, _ in links[j] if lag == 0)
        ]
        if not ready:
            raise ValueError("the same-step links form a cycle")
        order += ready
        placed.update(ready)
    return order


def _simulate(links, nodes, values, noise):
    """Run the links of `nodes` forward over every dataset at once.

    `values` holds the ``(n_datasets, n_rows)`` series of the nodes given from outside (the
    contexts); `noise` the ``(n_datasets, n_rows, n_nodes)`` innovations. Rows before the largest
    lag keep their noise, as in the script.
    """
    x = noise.copy()
    for node, series in values.items():
        x[:, :, node] = series
    max_lag = max((-lag for j in nodes for (_, lag), _, _ in links[j]), default=0)
    order = _same_step_order(links, nodes)
    for t in range(max_lag, x.shape[1]):
        for j in order:
            for (cause, lag), weight, function in links[j]:
                x[:, t, j] += weight * _FUNCTIONS[function](x[:, t + lag, cause])
    return x


def _parents(model, nodes):
    """Each node's parents at any lag, itself excluded."""
    out = {j: set() for j in nodes}
    for j, node_links in model.items():
        out[j] |= {cause for (cause, _), _, _ in node_links if cause != j}
    return out


def breaks_assumption_3(model: dict, n_system: int, observed: set[int]) -> bool:
    """Whether a hidden context is a parent of an observed context and reaches a system variable
    through hidden contexts only, so that it confounds the two."""
    nodes = list(model)
    parents = _parents(model, nodes)
    children = {u: {v for v in nodes if u in parents[v]} for u in nodes}
    hidden = {c for c in nodes if c >= n_system and c not in observed}
    for h in hidden:
        if not children[h] & observed:
            continue
        stack, seen = [h], {h}
        while stack:
            for v in children[stack.pop()]:
                if v < n_system:
                    return True
                if v in hidden and v not in seen:
                    seen.add(v)
                    stack.append(v)
    return False


def _draw(model, sds, n_system, time_nodes, space_nodes, observed, n_datasets, n_steps, rng):
    """Sample `model` and keep the system variables and the `observed` contexts."""
    burn = math.floor(BURN_IN * n_steps)
    rows = n_steps + burn
    time_series = {}
    if time_nodes:
        # The script runs the time contexts with a burn-in of their own, 20% of their length.
        long = rows + math.floor(BURN_IN * rows)
        noise = np.stack([sds[c] * rng.standard_normal(long) for c in time_nodes], axis=-1)
        sub = {i: model[c] for i, c in enumerate(time_nodes)}
        shifted = {i: [((c - time_nodes[0], lag), w, f) for (c, lag), w, f in sub[i]] for i in sub}
        x = _simulate(shifted, list(shifted), {}, noise[None])[0, long - rows :]
        time_series = {
            c: np.broadcast_to(x[:, i], (n_datasets, rows)) for i, c in enumerate(time_nodes)
        }
    space_series = {}
    if space_nodes:
        noise = np.stack([sds[c] * rng.standard_normal(n_datasets) for c in space_nodes], axis=-1)
        sub = {i: model[c] for i, c in enumerate(space_nodes)}
        shifted = {i: [((c - space_nodes[0], lag), w, f) for (c, lag), w, f in sub[i]] for i in sub}
        x = _simulate(shifted, list(shifted), {}, noise[:, None, :])[:, 0]
        space_series = {
            c: np.repeat(x[:, i : i + 1], rows, axis=1) for i, c in enumerate(space_nodes)
        }
    n_nodes = len(model)
    noise = rng.standard_normal((n_datasets, rows, n_nodes)) * np.asarray(sds)
    system = list(range(n_system))
    x = _simulate(model, system, time_series | space_series, noise)[:, burn:]
    obs_time = [c for c in time_nodes if c in observed]
    obs_space = [c for c in space_nodes if c in observed]
    columns = system + obs_time + obs_space
    position = {c: k for k, c in enumerate(columns)}
    links, context_links = set(), set()
    for j in system:
        for (cause, lag), _, _ in model[j]:
            if cause < n_system:
                links.add((cause, j, -lag))
            elif cause in position:
                context_links.add((position[cause], -lag, j))
    if not np.isfinite(x).all():
        raise RuntimeError("the draw overflowed")
    return ContextDraw(
        datasets=np.ascontiguousarray(x[:, :, columns]),
        n_system=n_system,
        time_contexts=tuple(range(n_system, n_system + len(obs_time))),
        space_contexts=tuple(
            range(n_system + len(obs_time), n_system + len(obs_time) + len(obs_space))
        ),
        links=frozenset(links),
        context_links=frozenset(context_links),
        breaks_assumption_3=breaks_assumption_3(model, n_system, set(observed)),
        model=tuple(
            (int(c), int(j), int(-lag), float(w), f)
            for j in sorted(model)
            for (c, lag), w, f in model[j]
        ),
        observed=tuple(int(c) for c in sorted(observed)),
        noise_sd=tuple(float(s) for s in sds),
    )


def jpcmciplus2023(
    n_system: int,
    n_datasets: int,
    n_steps: int,
    observed: float,
    seed: int,
    tau_max: int = 2,
) -> ContextDraw:
    """Draw the linear model of the paper's experiments; see the module docstring.

    `observed` is the fraction of each kind of context observed: ``1.0`` observes all, ``0.5`` one
    of the two of each kind.
    """
    # The script's seven child seeds: its model draws read 1, 3, 5, and 6 (the others seed its
    # noise), and an eighth seeds this draw's data.
    seeds = np.random.SeedSequence(seed).spawn(8)
    generators = {k: np.random.default_rng(seeds[k]) for k in (1, 3, 5, 6)}
    system = list(range(n_system))
    time_nodes = list(range(n_system, n_system + N_TIME))
    space_nodes = list(range(n_system + N_TIME, n_system + N_TIME + N_SPACE))
    # The script draws one link between two contexts of a kind (and k links for k != 2 of them).
    time_links, time_sds = _structural_process(
        time_nodes, 1 if N_TIME == 2 else N_TIME, 0.5, tau_max, generators[1]
    )
    space_links, space_sds = _structural_process(
        space_nodes, 1 if N_SPACE == 2 else N_SPACE, 1.0, 0, generators[3]
    )
    system_links, system_sds = _structural_process(
        system, 1 if n_system == 2 else n_system, 1.0, tau_max, generators[5]
    )
    slots = _non_parents(space_links, 0) + _non_parents(time_links, tau_max)
    from_contexts = _context_links(system, 2 * len(slots), slots, tau_max, generators[6])
    model = time_links | space_links | {j: system_links[j] + from_contexts[j] for j in system}
    model = {j: model[j] for j in sorted(model)}
    sds = system_sds + time_sds + space_sds
    choose = np.random.default_rng(np.random.SeedSequence(seed))
    seen = choose.choice(time_nodes, size=int(observed * N_TIME), replace=False).tolist()
    seen += choose.choice(space_nodes, size=int(observed * N_SPACE), replace=False).tolist()
    rng = np.random.default_rng(seeds[7])
    return _draw(model, sds, n_system, time_nodes, space_nodes, seen, n_datasets, n_steps, rng)


def jpcmciplus2023_nonlinear(n_datasets: int, n_steps: int, seed: int) -> ContextDraw:
    """Draw the script's fixed nonlinear model; see the module docstring. Both contexts observed."""
    model = {
        0: [((1, -1), 0.3, "square"), ((2, -1), 0.5, "linear"), ((3, 0), -0.5, "linear")],
        1: [((1, -1), 0.3, "linear"), ((2, -1), -0.5, "linear"), ((3, 0), 0.5, "linear")],
        2: [],
        3: [],
    }
    rng = np.random.default_rng(seed)
    return _draw(model, [1.0] * 4, 2, [2], [3], [2, 3], n_datasets, n_steps, rng)
