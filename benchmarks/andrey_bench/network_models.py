"""The cells of the adjustment lane that replay a published network.

MAGIC-NIAB and SANGIOVESE are fitted linear-Gaussian networks (:mod:`andrey_bench.gaussian_networks`);
the data are drawn from their fitted parameters, intercepts and residual standard deviations, and the
true effects are read from the fitted coefficients. MILDEW is a discrete network
(:mod:`andrey_bench.networks`); its rows are drawn from its tables, so a cell on it scores the
answers and has no linear effect.

The pair of a cell is the one its paper names, or, for ``pair=random``, a pair drawn the way the
SNAP and LOAD papers draw their targets (an explicit ancestor of the other with an effect
identifiable from the CPDAG, :func:`andrey_bench.adjustment_models.explicit_ancestor_pair`), a
different one for each seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from andrey_bench import adjustment_models as M
from andrey_bench.gaussian_networks import GaussianNetwork, load_gaussian_network

#: The pair each network's paper names: MAGIC-NIAB, the treatment ``G266`` and the outcome ``HT``
#: (Gupta et al., 2023, Sect. 5); SANGIOVESE, ``pH`` and ``GrapeW`` (Maasch et al., 2025, Table D.5);
#: MILDEW, ``mikro_1`` and ``meldug_2`` (Maasch et al., 2024, Appendix F.1), with the two variables
#: the paper drops from the data (``dm_1``, ``foto_1``) hidden.
PAIRS = {
    "magic-niab": ("G266", "HT"),
    "sangiovese": ("pH", "GrapeW"),
    "mildew": ("mikro_1", "meldug_2"),
}
MILDEW_HIDDEN = ("dm_1", "foto_1")


def gaussian_model(
    seed: int,
    network: GaussianNetwork,
    *,
    pair: str = "paper",
    estimand: str = "total",
    setting: str | None = None,
) -> M.LinearModel:
    """The linear-Gaussian model of ``network`` with one pair.

    ``pair="paper"`` is the pair :data:`PAIRS` names for the network; ``"random"`` draws one per
    seed as the SNAP and LOAD papers do.
    """
    setting = setting or network.name
    if pair == "paper":
        x, y = (network.index(v) for v in PAIRS[network.name])
    elif pair == "random":
        found = M.explicit_ancestor_pair(
            M.rng_for(setting, seed, M.MODEL), (network.weights != 0).astype(np.int8)
        )
        if found is None:
            raise RuntimeError(f"{network.name} has no qualifying pair (seed {seed})")
        x, y = found
    else:
        raise ValueError(f"pair must be 'paper' or 'random', got {pair!r}")
    return M.finish(
        setting,
        seed,
        network.weights,
        network.noise_sd,
        [],
        [(x, y, estimand)],
        intercepts=network.intercepts,
        params={"pair": pair, "estimand": estimand},
        meta={"names": list(network.nodes)},
    )


def magic_niab(seed: int, *, pair: str = "paper", estimand: str = "total") -> M.LinearModel:
    """MAGIC-NIAB (44 nodes, 66 arcs), the fitted linear-Gaussian network of Gupta et al. (2023),
    Sect. 5: treatment ``G266``, outcome ``HT``, whose one directed path is the arc between them
    (weight -0.78). The SNAP and LOAD papers use it with random identifiable target pairs
    (``pair=random``)."""
    return gaussian_model(seed, load_gaussian_network("magic-niab"), pair=pair, estimand=estimand)


def sangiovese(seed: int, *, pair: str = "paper", estimand: str = "direct") -> M.LinearModel:
    """SANGIOVESE without its discrete root (:mod:`andrey_bench.gaussian_networks`): Maasch et al.
    (2025), Sect. 5, treatment ``pH`` and outcome ``GrapeW``, a sink of the network. Their pair is
    scored for the direct effect, the one LD3 answers."""
    return gaussian_model(seed, load_gaussian_network("sangiovese"), pair=pair, estimand=estimand)


@dataclass(frozen=True)
class TableModel(M.LinearModel):
    """A discrete network replayed from its tables, with the pair asked about.

    ``weights`` holds the arcs as 0/1 so :attr:`dag` is the network's DAG in the model's node
    order; ``order`` maps a position to the network's node index, so the rows are the network's
    columns in that order, the observed ones first and the hidden ones left out.
    """

    network: Any = None
    order: tuple[int, ...] = ()

    def rows(self, n: int, stream: int) -> np.ndarray:
        """``n`` rows of category codes of the observed columns, from the network's tables."""
        rng = M.rng_for(self.setting, self.seed, stream, n)
        full = self.network.sample(n, int(rng.integers(2**31)))
        return full[:, list(self.order[: self.p])]


def table_model(
    seed: int,
    network: Any,
    pair: tuple[str, str],
    hidden: tuple[str, ...] = (),
    *,
    setting: str,
    estimand: str = "total",
    indep_test: str = "chisq",
) -> TableModel:
    """The discrete ``network`` (:class:`andrey_bench.networks.Network`) with the pair named by
    node, ``hidden`` nodes left out of the data, the observed columns in an order drawn from the
    seed."""
    index = {v.lower(): i for i, v in enumerate(network.nodes)}
    out = [index[v.lower()] for v in hidden]
    x, y = (index[v.lower()] for v in pair)
    rng = M.rng_for(setting, seed, M.MODEL, 1)
    observed = [v for v in range(network.d) if v not in out]
    observed = [observed[i] for i in rng.permutation(len(observed))]
    order = (*observed, *out)
    place = {old: new for new, old in enumerate(order)}
    return TableModel(
        setting=setting,
        seed=seed,
        weights=network.adjacency.astype(float)[np.ix_(order, order)],
        noise_sd=np.ones(network.d),
        p=len(observed),
        pairs=(M.Pair(place[x], place[y], estimand),),
        params={"estimand": estimand},
        meta={"names": [network.nodes[v] for v in order]},
        family="table",
        indep_test=indep_test,
        network=network,
        order=order,
    )


def mildew(seed: int, *, estimand: str = "total") -> TableModel:
    """MILDEW (35 nodes, 46 arcs), the discrete network of Maasch et al. (2024), Appendix F.1:
    treatment ``mikro_1`` and outcome ``meldug_2``, with ``dm_1`` and ``foto_1`` removed from the
    data as the paper removes them, so they are hidden variables of the model. Rows are drawn from
    the network's tables; the paper's sizes are 25000 to 75000 rows with the chi-square test at
    ``alpha = 0.001`` (the paper's Table G.8 says 0.005)."""
    from andrey_bench.networks import load_network

    return table_model(
        seed,
        load_network("mildew"),
        PAIRS["mildew"],
        MILDEW_HIDDEN,
        setting="mildew",
        estimand=estimand,
    )


for _name, _make, _estimand, _paper, _hidden in (
    ("magic-niab", magic_niab, "total", "Gupta, Childers, and Lipton (2023)", False),
    ("sangiovese", sangiovese, "direct", "Maasch et al. (2025)", False),
    ("mildew", mildew, "total", "Maasch et al. (2024)", True),
):
    M.register(M.Setting(_name, _make, _estimand, _paper, {}, hidden=_hidden))
