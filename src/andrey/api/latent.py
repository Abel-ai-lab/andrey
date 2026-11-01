"""Latent-variable family facade: GIN.

The callable runs the GIN engine and returns one uniform ``StructureOutput``. GIN recovers a
linear non-Gaussian latent-variable model from the observed data alone, so the wrapped graph spans
both the observed indicators and the discovered latents: each latent points to its observed
children, an ordered latent also points to every later latent, and ``node_types`` flags which
nodes are latent. The recovered clustering and latent causal order are stored in ``metadata``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from andrey.api._adapt import column_labels, data_matrix
from andrey.api._method import method_spec

if TYPE_CHECKING:
    import numpy.typing as npt

    from andrey.core import StructureOutput


@method_spec("gin")
def gin(data: npt.ArrayLike, *, alpha: float = 0.05) -> StructureOutput:
    """Learn a latent-variable DAG over observed indicators and their hidden latents with GIN.

    GIN exploits the Generalized Independent Noise condition in a linear non-Gaussian model to
    group observed variables under shared latent parents and to order those latents [1]_. It
    applies when the observed variables are noisy indicators of a few unobserved common causes
    rather than direct causes of one another.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_variables)
        Observed indicator data, one row per sample. Coerced to ``float64``. A DataFrame's column
        names become the observed variables' labels, else ``X1..Xn``; latents are named ``L1``,
        ``L2``, and so on, skipping a name an observed variable has.
    alpha : float, default=0.05
        Significance level of the pooled independence test, in ``(0, 1)``. A group of variables
        forms a cluster, and a latent is the next in the order, when the test's p-value is at
        least ``alpha``, so larger values make each test harder to pass.

    Returns
    -------
    StructureOutput
        Wraps a ``dag`` :class:`~andrey.core.GraphStructure` over the observed variables (indices
        ``0`` to ``n_variables - 1``) and one latent per cluster, named ``L1``, ``L2``, and so on:
        the ordered latents first, in causal order, then the unordered ones;
        ``structure.node_types`` flags the latents. Each latent points to the observed variables in
        its cluster. An ordered latent also points to every later latent and to every unordered one,
        and unordered latents have no edge between them: edges between latents follow the order only
        and do not mean a direct effect was found. An observed variable in no cluster has no edges.
        ``metadata["algorithm"]`` is ``"gin"``, ``metadata["clusters"]`` lists the clusters as lists
        of column indices, by smallest member, ``metadata["cluster_order"]`` lists the ordered ones
        earliest latent first, and ``metadata["unordered"]`` lists the rest by smallest member.

    Raises
    ------
    ValueError
        If an argument has the wrong type, lies outside its range, or is not one of its choices; the
        message names the method, the argument, what it accepts, and the value. Also if two of
        ``data``'s column names are equal as text.

    Notes
    -----
    **When to use.** GIN suits i.i.d. continuous observational data in which the observed variables
    are noisy indicators of a few unobserved common causes, the latents, rather than causes of one
    another [1]_. It assumes linear effects with non-Gaussian noise, no edge between two observed
    variables, and at least two observed indicators for each latent. It returns a DAG over the
    observed variables and one latent per cluster of indicators, flagged in
    ``structure.node_types``: each latent points to its cluster, and each ordered latent to every
    later or unordered latent, edges that follow the order rather than direct effects found. When
    the observed variables cause one another, use :func:`~andrey.direct_lingam` with no hidden
    variables, and :func:`~andrey.rcd` or :func:`~andrey.fci` with hidden common causes.

    A group of observed variables forms a cluster when the linear combination of the group with
    the smallest correlation with the other variables not yet in a cluster is independent of each of
    them; groups grow from pairs upward, and clusters that share a variable merge. The latents are
    then ordered one at a time, given the latents already ordered: the next latent is a root, whose
    cluster passes the independence test at ``alpha`` paired with each other remaining cluster [1]_.
    Each such test runs once per member, as many times as the largest cluster involved has members:
    run ``k`` builds the linear combination from member ``k`` of each cluster, cycling through a
    smaller cluster's members, and Fisher's method pools the runs, so every member of every cluster
    enters the test. When several candidates pass, the one whose lowest p-value is highest is taken,
    so latents with no causal path between them are still ordered, arbitrarily. When none passes,
    the ordering stops and the remaining latents stay unordered. A single remaining latent is
    ordered last without a test. Fisher's method treats
    the runs, and the variables each run is tested against, as independent tests, though they share
    the data, so a comparison can reject a true root more often than ``alpha``, and a correct model
    can still leave the latents from that step on unordered.

    **Column order.** Each column gets a tie rank computed from the data, as for
    :func:`~andrey.hc`, and the search runs on the columns sorted by it: the order of the groups it
    tests, the member each run of a root test takes, and the choice between root candidates whose
    lowest p-values agree to nine decimals follow the rank. Reordering the columns therefore
    relabels the clusters and changes nothing else, unless two columns are identical. The rank is
    arbitrary with respect to causal structure. The
    :doc:`column-order guide </docs/guides/column-order>` says what reordering the columns keeps.

    **Missing values.** A ``NaN`` in ``data`` raises ``ValueError``: its tests compare clusters of
    variables over the same rows, so a row with a missing value has no place in them. Drop the
    incomplete rows first, which suits values missing completely at random, or use
    :func:`~andrey.pc` or :func:`~andrey.fci`, whose tests delete rows per test.

    **Limitations.** Measured on simulated linear data with uniform noise from latent variables in a
    complete DAG, each measured by 3 observed indicators, on 4 cores with a cap of 1 h and 28 GB per
    fit. The median adjusted Rand index of its clusters, which is 0 for the empty graph, was 0.77 to
    1 at every size through 16 latents (48 indicators) with 1,000 rows and 1 through 8 latents (24
    indicators) with 5,000 rows, the largest sizes its fits finished within the caps. The ladder
    with 1,000 rows ended at the largest planned size, and the one with 5,000 rows when three fits
    of the next size ran past 1 h. The median fit took 340 s at 16 latents (48 indicators) with
    1,000 rows. Given the true clusters and 1,000 rows, over 10 seeds each, the order of a chain of
    three latents never stopped early and that of a complete three-latent model stopped early once,
    with no latent put after one it causes; with the benchmark's random weights instead, 2 of 10
    complete three-latent models put a latent after one it causes. It tests every group of observed
    variables of each size, from pairs upward, so the number of tests grows combinatorially with the
    variable count, and each HSIC test's time and memory grow with the square of the row count; the
    fit keeps one such kernel matrix per variable, 8 MB each at 1,000 rows. Clusters have at least
    two variables, so a latent with a single indicator is not found. Each cluster gets one latent,
    so indicators that share two or more latents appear as one. It assumes every observed variable
    measures a latent: variables that depend on no other variable pass the cluster test trivially
    and are grouped into one cluster. The numbers are a reference, not a hard limit: they change
    with the data generator, the density, the method's settings, and the machine.

    References
    ----------
    .. [1] Xie, F., Cai, R., Huang, B., Glymour, C., Hao, Z., and Zhang, K. (2020). Generalized
       independent noise condition for estimating latent variable causal graphs. Advances in
       Neural Information Processing Systems 33 (NeurIPS 2020).

    Examples
    --------
    >>> import numpy as np
    >>> from andrey import gin
    >>> rng = np.random.default_rng(0)
    >>> latent = rng.uniform(-1, 1, 500)
    >>> data = np.column_stack([
    ...     0.8 * latent + 0.1 * rng.uniform(-1, 1, 500),
    ...     1.2 * latent + 0.1 * rng.uniform(-1, 1, 500),
    ...     0.9 * latent + 0.1 * rng.uniform(-1, 1, 500),
    ... ])
    >>> out = gin(data)
    >>> out.metadata["clusters"], out.structure.labels
    ([[0, 1, 2]], ('X1', 'X2', 'X3', 'L1'))
    """
    from andrey.latent.gin import gin_structure

    X = data_matrix(data, method="gin")
    return gin_structure(X, alpha=alpha, labels=column_labels(data))
