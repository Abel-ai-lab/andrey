"""Measurement columns on simulated data: noisy children whose measured column is known.

:func:`noisy_children` describes measurement columns, and ``measurements=`` on
:meth:`~andrey.data.SCM.sample` and :func:`~andrey.data.sample_scm` appends them to the sampled
data: each is a selected variable plus independent noise, the random measurement error model
[1]_ [2]_, and a child of that variable in the true graph. The returned
:class:`~andrey.data.dataset.Measurements` records which column each one measures:

>>> import andrey.data as data
>>> from andrey.data import measurement
>>> ds = data.sample_scm(d=5, n=500, seed=0, density=2,
...                      measurements=measurement.noisy_children([0, 3]))
>>> ds.data.shape, ds.measurements.parents
((500, 7), {5: 0, 6: 3})

A measurement of column ``i`` is ``X_i + noise_scale * sd(X_i) * e`` with ``e`` of mean 0 and
variance 1, independent of every variable and of the other measurement noises, and ``sd(X_i)`` the
standard deviation of the returned column over the rows. The share of a measurement's variance
that comes from its variable, its reliability, is therefore ``1 / (1 + noise_scale**2)`` whatever
the units of the data. The measurement is then scaled as the sample's ``scale`` scales a
variable: z-scored under ``"standardize"``, divided by its standard deviation under ``"rescale"``,
divided by ``sqrt(1 + noise_scale**2)`` under ``"unitless"``, and kept under ``"raw"``.

To study a variable observed only with error, drop its column and keep its measurement: the
variable is then a hidden parent of the measurement.

References
----------
.. [1] Scheines, R., and Ramsey, J. (2016). Measurement error and causal discovery. CEUR Workshop
   Proceedings 1792, 1-7.
.. [2] Zhang, K., Gong, M., Ramsey, J., Batmanghelich, K., Spirtes, P., and Glymour, C. (2018).
   Causal discovery with linear non-Gaussian models under measurement error: structural
   identifiability results. Proceedings of the 34th Conference on Uncertainty in Artificial
   Intelligence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from andrey.core.structure import OBSERVED

from .dataset import CausalDataset, Measurements
from .graphs import dag_truth
from .noise import Noise, gaussian
from .scm import MEASUREMENT, component_rng
from .sortability import standardize

__all__ = ["Measurements", "NoisyChildren", "noisy_children"]

# The streams under MEASUREMENT: the choice of the measured columns, part of the model, and the
# measurement noise, part of the rows.
_CHOICE, _NOISE = 0, 1


@dataclass(frozen=True, kw_only=True)
class NoisyChildren:
    """Which variables get a measurement column, and its noise.

    Built by :func:`noisy_children` and drawn by ``measurements=`` on
    :meth:`~andrey.data.SCM.sample`.
    """

    columns: tuple[int, ...] | None
    share: float | None
    noise_scale: float
    noise: Noise


def noisy_children(
    columns: Sequence[int] | None = None,
    *,
    share: float | None = None,
    noise_scale: float = 0.5,
    noise: Noise | None = None,
) -> NoisyChildren:
    """Measurement columns: for each selected variable, a child that is it plus independent noise.

    Parameters
    ----------
    columns : sequence of int or None, default=None
        The measured columns, as columns of the sample without measurements, after its column
        shuffle. ``None`` measures the ``share`` of the columns, or every column when ``share``
        is ``None`` too.
    share : float or None, default=None
        The share of the columns to measure, in ``(0, 1]``, rounded to a whole number of at least
        one column; which ones is drawn from the model's seed. Given without ``columns``.
    noise_scale : float, default=0.5
        The standard deviation of the measurement noise as a multiple of the standard deviation
        of the measured column, finite and positive. The default gives each measurement a
        reliability of ``1 / (1 + 0.5**2) = 0.8``.
    noise : Noise or None, default=None
        The family of the measurement noise, from :mod:`andrey.data.noise`, without
        ``heteroscedastic`` or ``variance_range``. ``None`` is Gaussian.

    Returns
    -------
    NoisyChildren

    Raises
    ------
    ValueError
        If ``columns`` is empty or holds a repeated, negative, or fractional column, ``columns``
        and ``share`` are both given, ``share`` is not in ``(0, 1]``, ``noise_scale`` is not
        finite and positive, or ``noise`` sets its own scales.

    Examples
    --------
    >>> from andrey.data.measurement import noisy_children
    >>> noisy_children([0, 2], noise_scale=1.0).columns
    (0, 2)
    """
    chosen = None
    if columns is not None:
        if share is not None:
            raise ValueError("noisy_children takes columns or a share, not both")
        listed = list(columns)
        whole = all(
            isinstance(c, int | np.integer) and not isinstance(c, bool) and c >= 0 for c in listed
        )
        if not listed or not whole or len(set(listed)) != len(listed):
            raise ValueError(
                f"columns must be distinct whole numbers >= 0, at least one, got {listed!r}"
            )
        chosen = tuple(sorted(int(c) for c in listed))
    if share is not None and not (np.isfinite(share) and 0 < share <= 1):
        raise ValueError(f"share must be in (0, 1], got {share!r}")
    if not (np.isfinite(noise_scale) and noise_scale > 0):
        raise ValueError(f"noise_scale must be finite and positive, got {noise_scale!r}")
    family = gaussian() if noise is None else noise
    if not isinstance(family, Noise):
        raise ValueError(f"noise must be a family of andrey.data.noise, got {noise!r}")
    if family.heteroscedastic or family.variance_range is not None:
        raise ValueError(
            "measurement noise takes its scale from noise_scale; give a noise family without "
            "heteroscedastic=True or a variance_range"
        )
    return NoisyChildren(
        columns=chosen,
        share=None if share is None else float(share),
        noise_scale=float(noise_scale),
        noise=family,
    )


def append(
    ds: CausalDataset,
    spec: NoisyChildren,
    root: np.random.SeedSequence,
    model: np.random.SeedSequence,
    *,
    scale: str,
) -> CausalDataset:
    """Return ``ds`` with the measurement columns of ``spec`` appended.

    The measured columns come from the measurement stream of ``model`` and the noise from that of
    ``root``, so no other draw depends on them. The measurement nodes follow the observed nodes
    in the true graph and precede the latent ones.
    """
    n, n_obs = ds.data.shape
    measured = _measured(spec, n_obs, component_rng(model, MEASUREMENT, _CHOICE))
    k = measured.size
    values = ds.data[:, measured].astype(np.float64)
    sd = values.std(axis=0)
    if np.any(sd == 0):
        raise ValueError(
            f"columns {measured[sd == 0].tolist()} are constant over the rows; a measurement "
            "takes its noise scale from the measured column's standard deviation"
        )
    noise_sd = spec.noise_scale * sd
    children = values + spec.noise.draw(n, noise_sd, component_rng(root, MEASUREMENT, _NOISE))
    if scale == "standardize":
        children = standardize(children)
    elif scale == "rescale":
        children /= children.std(axis=0)
    elif scale == "unitless":
        children /= np.sqrt(1.0 + spec.noise_scale**2)
    data = np.concatenate([ds.data, children.astype(ds.data.dtype)], axis=1)

    # The measurement nodes take the labels n_obs .. n_obs + k - 1, and the latent nodes move up.
    edges = np.where(ds.params.edges >= n_obs, ds.params.edges + k, ds.params.edges)
    nodes = n_obs + np.arange(k)
    types = ds.graph.node_types
    if types is not None:
        types = np.concatenate([types[:n_obs], np.full(k, OBSERVED, types.dtype), types[n_obs:]])
    graph = dag_truth(
        np.concatenate([edges[:, 0], measured]),
        np.concatenate([edges[:, 1], nodes]),
        ds.graph.n_nodes + k,
        node_types=types,
    )
    scales = ds.params.noise_scales
    params = replace(
        ds.params,
        edges=edges,
        noise_scales=np.concatenate([scales[:n_obs], noise_sd, scales[n_obs:]]),
    )
    record = Measurements(
        parents={int(node): int(column) for node, column in zip(nodes, measured)},
        noise=spec.noise.name,
        noise_scale=spec.noise_scale,
    )
    options = spec.noise.options
    config = {
        **ds.provenance["config"],
        "measurements": {
            "columns": measured.tolist(),
            "noise_scale": spec.noise_scale,
            "noise": spec.noise.name,
            **({"noise_params": options} if options else {}),
        },
    }
    return replace(
        ds,
        data=data,
        graph=graph,
        params=params,
        measurements=record,
        provenance={**ds.provenance, "config": config},
    )


def _measured(spec: NoisyChildren, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """Return the measured columns of ``n_obs``, ascending: the spec's own, or drawn."""
    if spec.columns is not None:
        if spec.columns[-1] >= n_obs:
            raise ValueError(
                f"measurements name columns {list(spec.columns)}, and the data have {n_obs}"
            )
        return np.array(spec.columns, dtype=np.int64)
    if spec.share is None:
        return np.arange(n_obs, dtype=np.int64)
    k = max(1, int(round(spec.share * n_obs)))
    return np.sort(rng.choice(n_obs, size=k, replace=False)).astype(np.int64)
