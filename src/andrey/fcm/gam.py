"""Additive models of penalized thin plate regression splines, with per-term p-values.

A response ``y`` is modeled as ``y = c + sum_j f_j(x_j) + e``. Each ``f_j`` is a thin plate
regression spline of rank ``k`` (Wood 2003) whose identifiability constraint ``sum_i f_j(x_ij) = 0``
is absorbed into the basis, and its wiggliness ``integral f_j''(t)^2 dt`` is penalized with its
own smoothing parameter. The smoothing parameters minimize the GCV score
``n * RSS / (n - tr(A))^2`` (``A`` the hat matrix) by Newton's method on their logarithms with exact
first and second derivatives (Wood 2004). Each term's p-value tests ``f_j = 0`` with the Wood (2013)
statistic, whose reference distribution is a weighted sum of chi-squares evaluated by Imhof's
(1961) inversion formula.

The basis of a term depends only on its covariate and ``k``, so :class:`BasisCache` builds it once
per variable and rank and reuses it across fits. :func:`fit_gam` fits one variable on a set of
others with the rank rule of the CAM method (Buhlmann, Peters, Ernest 2014); :func:`fit_additive`
fits a response on given bases. Every step is deterministic.

References
----------
Wood. "Thin plate regression splines." JRSS-B 65(1):95-114, 2003.

Wood. "Stable and efficient multiple smoothing parameter estimation for generalized additive
models." JASA 99:673-686, 2004.

Wood. "On p-values for smooth components of an extended generalized additive model." Biometrika
100(1):221-228, 2013.

Imhof. "Computing the distribution of quadratic forms in normal variables." Biometrika
48:419-426, 1961.

Buhlmann, Peters, Ernest. "CAM: Causal additive models, high-dimensional order search and
penalized regression." Annals of Statistics 42(6):2526-2556, 2014.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy import integrate, linalg, stats
from scipy.sparse import linalg as sparse_linalg

_log = logging.getLogger(__name__)

#: Distinct covariate values above which a basis is built on a fixed subset of this many knots.
MAX_KNOTS = 2000

#: Bounds of the log smoothing parameters. At either bound a term is, to working precision, its
#: penalty null space (a linear function) or an unpenalized regression spline.
_LOG_SP_BOUND = 25.0
#: GCV convergence: every free gradient component below this fraction of the score.
_GRAD_TOL = 1e-7
_MAX_NEWTON = 200
_MAX_HALVINGS = 30
#: Largest change of a log smoothing parameter in one step.
_MAX_STEP = 5.0
#: Midpoint-rule nodes above which a tail probability falls back to adaptive quadrature.
_MAX_NODES = 200_000


@dataclass(frozen=True)
class SmoothBasis:
    """One term's constrained thin plate regression spline basis and its penalty.

    Attributes
    ----------
    X : np.ndarray
        ``(n, k - 1)`` design matrix with the sum-to-zero constraint absorbed.
    S : np.ndarray
        ``(k - 1, k - 1)`` wiggliness penalty, divided by ``scale``.
    root : np.ndarray
        ``(k - 2, k - 1)`` square root of ``S``: ``S == root.T @ root``.
    scale : float
        Factor the raw penalty was divided by, ``norm_1(S_raw) / norm_inf(X_raw)^2``, which puts
        the penalty on the scale of ``X.T @ X``.
    """

    X: np.ndarray
    S: np.ndarray
    root: np.ndarray
    scale: float


def tp_basis(x: np.ndarray, k: int = 10, *, max_knots: int = MAX_KNOTS) -> SmoothBasis:
    """Build the rank-``k`` thin plate regression spline basis of one covariate.

    The covariate is centered. The full thin plate spline (second-derivative penalty, kernel
    ``eta(r) = r^3 / 12``) is built on the distinct values, or, when there are more than
    ``max_knots`` of them, on a fixed subset of ``max_knots`` drawn by R's Mersenne-Twister
    ``sample()`` seeded with ``1``. The basis keeps the ``k`` eigenvectors of the kernel matrix
    with the largest absolute eigenvalues, constrained orthogonal to the linear functions, plus the
    constant and linear functions (Wood 2003). The penalty is divided by
    ``norm_1(S) / norm_inf(X)^2``, then the constraint ``sum_i f(x_i) = 0`` is absorbed by a
    Householder reflection of the column means.

    Parameters
    ----------
    x : np.ndarray
        ``(n,)`` covariate values.
    k : int, default 10
        Basis rank before the constraint, at least ``3``.
    max_knots : int, default 2000
        Most knots the basis is built on.

    Returns
    -------
    SmoothBasis
        The ``(n, k - 1)`` design and its ``(k - 1, k - 1)`` penalty.

    Raises
    ------
    ValueError
        If ``k < 3`` or ``x`` has fewer than ``k`` distinct values.
    """
    x = np.asarray(x, dtype=np.float64)
    if k < 3:
        raise ValueError(f"a thin plate spline basis needs k >= 3, got {k}")
    xc = x - x.mean()
    knots = np.unique(xc)
    if knots.size < k:
        raise ValueError(f"k = {k} needs at least {k} distinct values, got {knots.size}")
    if x.size > max_knots and knots.size > max_knots:
        knots = np.sort(knots[_r_sample(knots.size, max_knots, seed=1)])
    width = float(np.abs(knots).max())
    values, vectors = _top_eigen(knots / width, k)
    values *= width**3 / 12.0
    polynomial = np.column_stack([np.ones(knots.size), knots])
    Q, _ = np.linalg.qr(vectors.T @ polynomial, mode="complete")
    Z = Q[:, 2:]
    X = np.empty((x.size, k))
    X[:, : k - 2] = _cubic_cross(xc / width, knots / width, vectors @ Z) * (width**3 / 12.0)
    X[:, k - 2] = 1.0
    X[:, k - 1] = xc
    S = np.zeros((k, k))
    S[: k - 2, : k - 2] = Z.T @ (values[:, None] * Z)
    S = (S + S.T) / 2
    scale = float(np.abs(S).sum(axis=0).max() / np.abs(X).sum(axis=1).max() ** 2)
    S /= scale
    H, _ = np.linalg.qr(X.mean(axis=0)[:, None], mode="complete")
    X = X @ H[:, 1:]
    S = H[:, 1:].T @ S @ H[:, 1:]
    S = (S + S.T) / 2
    eig, vec = np.linalg.eigh(S)
    keep = np.argsort(eig)[1:]
    root = np.sqrt(np.clip(eig[keep], 0.0, None))[:, None] * vec[:, keep].T
    return SmoothBasis(X=X, S=S, root=root, scale=scale)


def _top_eigen(t: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` largest-magnitude eigenpairs of ``|t_i - t_j|^3`` for sorted ``t``.

    Lanczos iteration (ARPACK) from a fixed start vector, on products that cost ``O(m)`` through
    prefix sums of ``v t^p``; dense ``eigh`` for small ``m`` or when Lanczos does not converge.
    """
    m = t.size
    if m > max(100, 4 * k):
        start = np.cos(0.7 * np.arange(m)) + 1.0
        try:
            values, vectors = sparse_linalg.eigsh(_CubicKernel(t), k=k, which="LM", v0=start, tol=0)
        except sparse_linalg.ArpackNoConvergence:
            _log.info("Lanczos did not converge on %d knots; using a dense eigensolver", m)
        else:
            order = np.argsort(-np.abs(values), kind="stable")
            return values[order], vectors[:, order]
    values, vectors = linalg.eigh(np.abs(t[:, None] - t[None, :]) ** 3)
    top = np.argsort(-np.abs(values), kind="stable")[:k]
    return values[top], vectors[:, top]


class _CubicKernel(sparse_linalg.LinearOperator):
    """The matrix ``|t_i - t_j|^3`` for sorted ``t``, applied through :func:`_cubic_cross`."""

    def __init__(self, t: np.ndarray) -> None:
        super().__init__(dtype=np.dtype(np.float64), shape=(t.size, t.size))
        self.t = t

    def _matvec(self, x: np.ndarray) -> np.ndarray:
        return _cubic_cross(self.t, self.t, x.reshape(-1, 1)).ravel()

    def _matmat(self, X: np.ndarray) -> np.ndarray:
        return _cubic_cross(self.t, self.t, X)


def _cubic_cross(x: np.ndarray, t: np.ndarray, A: np.ndarray) -> np.ndarray:
    """``K @ A`` for ``K_ij = |x_i - t_j|^3`` and sorted ``t``, in ``O((n + m) c)``.

    ``|x - t_j|^3`` expands in powers of ``x`` and ``t_j`` on either side of ``x``, so each row
    needs only prefix sums of ``A_j t_j^p`` for ``p = 0..3``.
    """
    powers = np.stack([np.ones_like(t), t, t * t, t * t * t])
    terms = powers[:, :, None] * A[None, :, :]
    prefix = np.concatenate([np.zeros((4, 1, A.shape[1])), np.cumsum(terms, axis=1)], axis=1)
    below = prefix[:, np.searchsorted(t, x), :]
    above = prefix[:, -1:, :] - below
    xi = x[:, None]
    left = xi**3 * below[0] - 3.0 * xi**2 * below[1] + 3.0 * xi * below[2] - below[3]
    right = above[3] - 3.0 * xi * above[2] + 3.0 * xi**2 * above[1] - xi**3 * above[0]
    return left + right


class BasisCache:
    """The thin plate bases of one data matrix's columns, built once per column and rank.

    Parameters
    ----------
    data : np.ndarray
        ``(n, d)`` data matrix; column ``j`` is variable ``j``.
    max_knots : int, default 2000
        Passed to :func:`tp_basis`.
    """

    def __init__(self, data: np.ndarray, *, max_knots: int = MAX_KNOTS) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2-D, got shape {self.data.shape}")
        self.max_knots = max_knots
        self._bases: dict[tuple[int, int], SmoothBasis] = {}

    def __call__(self, column: int, k: int) -> SmoothBasis:
        """Return column ``column``'s rank-``k`` basis, building it on first use."""
        key = (int(column), int(k))
        if key not in self._bases:
            self._bases[key] = tp_basis(self.data[:, column], k, max_knots=self.max_knots)
        return self._bases[key]


@dataclass
class GAMFit:
    """A fitted additive model: fitted values, smoothing, and per-term tests.

    Attributes
    ----------
    fitted, residuals : np.ndarray
        ``(n,)`` fitted values and ``y - fitted``.
    coef : np.ndarray
        ``(p,)`` coefficients: the intercept, then each term's ``k - 1`` coefficients.
    sp : np.ndarray
        ``(J,)`` smoothing parameters multiplying each term's scaled penalty ``SmoothBasis.S``.
    edf, edf1 : np.ndarray
        ``(J,)`` effective degrees of freedom of each term, ``tr(F)`` and ``tr(2F - FF)`` over its
        coefficients, with ``F = (X'X + S_sp)^{-1} X'X``.
    scale : float
        Residual variance estimate ``RSS / (n - tr(A))``.
    gcv : float
        GCV score ``n * RSS / (n - tr(A))^2``.
    residual_df : float
        ``n - tr(A)``: the number of rows less the model's effective degrees of freedom.
    k : int
        Basis rank of every term.
    bases : tuple of SmoothBasis
        The terms' bases, in the order of ``sp``.
    """

    fitted: np.ndarray
    residuals: np.ndarray
    coef: np.ndarray
    sp: np.ndarray
    edf: np.ndarray
    edf1: np.ndarray
    scale: float
    gcv: float
    residual_df: float
    k: int
    bases: tuple[SmoothBasis, ...]
    _cov_unscaled: np.ndarray = field(repr=False)
    _tests: list[tuple[float, float, float]] | None = field(default=None, repr=False)

    @property
    def residual_variance(self) -> float:
        """Sample variance of the residuals with denominator ``n - 1``."""
        return float(np.var(self.residuals, ddof=1))

    def p_values(self) -> np.ndarray:
        """Return each term's p-value for ``f_j = 0`` (Wood 2013), shape ``(J,)``."""
        return np.array([t[0] for t in self._term_tests()])

    def statistics(self) -> np.ndarray:
        """Return each term's Wood (2013) test statistic, shape ``(J,)``."""
        return np.array([t[1] for t in self._term_tests()])

    def ref_df(self) -> np.ndarray:
        """Return the rank each term's test uses, ``min(k - 1, edf1)`` or less, shape ``(J,)``."""
        return np.array([t[2] for t in self._term_tests()])

    def _term_tests(self) -> list[tuple[float, float, float]]:
        if self._tests is None:
            tests = []
            start = 1
            for j, basis in enumerate(self.bases):
                cols = slice(start, start + basis.X.shape[1])
                start = cols.stop
                cov = self._cov_unscaled[cols, cols] * self.scale
                rank = min(basis.X.shape[1], float(self.edf1[j]))
                tests.append(_test_stat(self.coef[cols], basis.X, cov, rank, self.residual_df))
            self._tests = tests
        return self._tests


def effective_k(n: int, n_parents: int, k: int = 10) -> int:
    """Return the basis rank for ``n_parents`` smooth terms on ``n`` rows.

    With fewer than ``3 k`` rows per term, the rank drops to ``ceil(n / (3 n_parents))`` so that
    each basis function keeps at least three rows; it never drops below ``3``, the smallest thin
    plate basis with a wiggly part.
    """
    if n_parents > 0 and n / n_parents < 3 * k:
        k = math.ceil(n / (3 * n_parents))
    return max(k, 3)


def fit_gam(
    bases: BasisCache, child: int, parents: Sequence[int] | np.ndarray, *, k: int = 10
) -> GAMFit:
    """Fit column ``child`` of ``bases.data`` as an additive model of columns ``parents``.

    Every parent gets a smooth term of rank :func:`effective_k`. When the GCV search fails
    numerically, the model is refit with every smoothing parameter ``0``, an unpenalized regression
    spline.

    Parameters
    ----------
    bases : BasisCache
        Bases of the data matrix's columns.
    child : int
        Response column.
    parents : sequence of int
        Covariate columns, at least one, excluding ``child``.
    k : int, default 10
        Basis rank before :func:`effective_k` adjusts it.

    Returns
    -------
    GAMFit
        The fitted model; term ``j`` belongs to ``parents[j]``.

    Raises
    ------
    ValueError
        If ``parents`` is empty or contains ``child``, or a parent has fewer than the rank's
        distinct values.
    """
    parents = [int(p) for p in parents]
    if not parents:
        raise ValueError("fit_gam needs at least one parent")
    if child in parents:
        raise ValueError(f"column {child} cannot be its own parent")
    n = bases.data.shape[0]
    rank = effective_k(n, len(parents), k)
    if rank != k:
        _log.debug("basis rank %d -> %d for %d parents on %d rows", k, rank, len(parents), n)
    terms = [bases(p, rank) for p in parents]
    y = bases.data[:, child]
    try:
        return fit_additive(y, terms)
    except (np.linalg.LinAlgError, FloatingPointError) as err:
        _log.info("GCV search failed (%s); refitting with zero smoothing parameters", err)
        return fit_additive(y, terms, sp=np.zeros(len(terms)))


def fit_additive(
    y: np.ndarray, bases: Sequence[SmoothBasis], *, sp: np.ndarray | None = None
) -> GAMFit:
    """Fit ``y = c + sum_j f_j`` with one smooth term per basis.

    Parameters
    ----------
    y : np.ndarray
        ``(n,)`` response.
    bases : sequence of SmoothBasis
        One basis per term, each with ``n`` rows.
    sp : np.ndarray or None, default None
        Fixed smoothing parameters, one per term, each ``>= 0``; ``None`` selects them by GCV.

    Returns
    -------
    GAMFit
        The fitted model.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the penalized least-squares system is singular.
    FloatingPointError
        If the GCV score is not finite.
    """
    y = np.asarray(y, dtype=np.float64)
    problem = _Problem(y, bases)
    if sp is None:
        state = problem.optimize()
    else:
        sp = np.asarray(sp, dtype=np.float64)
        if sp.shape != (len(bases),) or np.any(sp < 0):
            raise ValueError(f"sp must hold {len(bases)} values >= 0")
        state = problem.solve(sp)
    return problem.finish(state, k=bases[0].X.shape[1] + 1 if bases else 0)


@dataclass
class _State:
    """The penalized fit at one set of smoothing parameters."""

    sp: np.ndarray
    P: np.ndarray  # inverse of the triangular factor of [R; sqrt(sp) roots]
    Q1: np.ndarray  # R @ P
    coef: np.ndarray
    rss: float
    trace: float  # tr(A)
    gcv: float


class _Problem:
    """GCV smoothing-parameter selection for one response and set of bases (Wood 2004)."""

    def __init__(self, y: np.ndarray, bases: Sequence[SmoothBasis]) -> None:
        n = y.size
        if not bases:
            raise ValueError("an additive model needs at least one term")
        for b in bases:
            if b.X.shape[0] != n:
                raise ValueError(f"a basis has {b.X.shape[0]} rows, the response {n}")
        self.n = n
        self.y = y
        self.bases = tuple(bases)
        self.X = np.column_stack([np.ones(n), *(b.X for b in bases)])
        self.p = self.X.shape[1]
        if self.p >= n:
            raise ValueError(f"{self.p} coefficients need more than {n} rows")
        self.cols: list[slice] = []
        self.rows: list[slice] = []
        c, r = 1, 0
        for b in bases:
            self.cols.append(slice(c, c + b.X.shape[1]))
            self.rows.append(slice(r, r + b.root.shape[0]))
            c += b.X.shape[1]
            r += b.root.shape[0]
        self.n_root = r
        Q, self.R = np.linalg.qr(self.X)
        self.f = Q.T @ y
        resid = y - Q @ self.f
        self.rss0 = float(resid @ resid)

    def solve(self, sp: np.ndarray) -> _State:
        """Penalized least squares at ``sp`` through the QR factor of ``[R; sqrt(sp) roots]``."""
        B = np.zeros((self.n_root, self.p))
        for j, b in enumerate(self.bases):
            B[self.rows[j], self.cols[j]] = math.sqrt(sp[j]) * b.root
        Qs, Rs = np.linalg.qr(np.vstack([self.R, B]))
        P = linalg.solve_triangular(Rs, np.eye(self.p), check_finite=False)
        Q1 = Qs[: self.p]
        Q1f = Q1.T @ self.f
        coef = P @ Q1f
        resid = self.f - Q1 @ Q1f
        rss = self.rss0 + float(resid @ resid)
        trace = float(np.sum(Q1 * Q1))
        gcv = self.n * rss / (self.n - trace) ** 2
        if not (np.isfinite(gcv) and np.all(np.isfinite(coef))):
            raise FloatingPointError(f"GCV score is not finite at sp = {sp}")
        return _State(sp, P, Q1, coef, rss, trace, gcv)

    def derivatives(self, st: _State) -> tuple[np.ndarray, np.ndarray]:
        """Gradient and Hessian of the GCV score in the log smoothing parameters."""
        n, sp, P, Q1 = self.n, st.sp, st.P, st.Q1
        J = sp.size
        K = Q1.T @ Q1
        G = np.vstack([b.root @ P[self.cols[j]] for j, b in enumerate(self.bases)])
        A = np.empty((self.p, J))
        for j, b in enumerate(self.bases):
            g = b.root @ st.coef[self.cols[j]]
            A[:, j] = sp[j] * (G[self.rows[j]].T @ g)
        a = A.sum(axis=1)
        GK = G @ K
        CH = (G @ G.T) * (GK @ G.T)
        starts = np.array([r.start for r in self.rows])
        tr1 = -sp * np.add.reduceat(np.sum(G * GK, axis=1), starts)
        tr2 = np.add.reduceat(np.add.reduceat(CH, starts, axis=0), starts, axis=1)
        tr2 = np.diag(tr1) + 2.0 * np.outer(sp, sp) * tr2
        W = sp[:, None] * np.add.reduceat((G @ a)[:, None] * (G @ A), starts, axis=0)
        d1 = 2.0 * A.T @ a
        d2 = 2.0 * A.T @ K @ A - 2.0 * (W + W.T) + 2.0 * np.diag(A.T @ a)
        rss, df = st.rss, n - st.trace
        grad = n * d1 / df**2 + 2.0 * n * rss * tr1 / df**3
        hess = (
            n * d2 / df**2
            + 2.0 * n * (np.outer(d1, tr1) + np.outer(tr1, d1)) / df**3
            + 2.0 * n * rss * tr2 / df**3
            + 6.0 * n * rss * np.outer(tr1, tr1) / df**4
        )
        return grad, hess

    def initial_log_sp(self) -> np.ndarray:
        """Return the starting log smoothing parameters.

        Each term starts at its mean diagonal of ``X'X`` over that of ``S``, and all are rescaled
        by powers of ten until the penalized share of the diagonal is about 0.4.
        """
        xx = np.sum(self.X * self.X, axis=0)
        ss = np.zeros(self.p)
        penalized = np.zeros(self.p, dtype=bool)
        sp = np.zeros(len(self.bases))
        for j, b in enumerate(self.bases):
            aS = np.abs(b.S)
            tiny = np.finfo(float).eps ** 0.8 * aS.max()
            on = (aS.mean(axis=1) > tiny) & (aS.mean(axis=0) > tiny) & (np.diag(aS) > tiny)
            sp[j] = xx[self.cols[j]][on].mean() / np.diag(b.S)[on].mean()
            ss[self.cols[j]] += sp[j] * np.diag(b.S)
            penalized[self.cols[j]] |= on
        use = (ss > 0) & penalized & (xx > 0)
        xx, ss = xx[use], ss[use]
        while np.mean(xx / (xx + ss)) > 0.4:
            sp *= 10.0
            ss = ss * 10.0
        while np.mean(xx / (xx + ss)) < 0.4:
            sp /= 10.0
            ss = ss / 10.0
        return np.clip(np.log(sp), -_LOG_SP_BOUND, _LOG_SP_BOUND)

    def optimize(self) -> _State:
        """Minimize the GCV score over ``log sp`` in ``[-25, 25]``.

        A step is Newton's when the Hessian of the free parameters is positive definite and steepest
        descent otherwise, capped at 5 per parameter and halved until the score falls. A parameter
        at a bound whose gradient points outward is held there.
        """
        rho = self.initial_log_sp()
        st = self.solve(np.exp(rho))
        for _ in range(_MAX_NEWTON):
            grad, hess = self.derivatives(st)
            free = ~(((rho >= _LOG_SP_BOUND) & (grad < 0)) | ((rho <= -_LOG_SP_BOUND) & (grad > 0)))
            g = grad[free]
            if g.size == 0 or np.max(np.abs(g)) <= _GRAD_TOL * st.gcv:
                break
            directions = []
            try:
                np.linalg.cholesky(hess[np.ix_(free, free)])
                directions.append(-np.linalg.solve(hess[np.ix_(free, free)], g))
            except np.linalg.LinAlgError:
                pass
            directions.append(-g / np.max(np.abs(g)))
            moved = False
            for direction in directions:
                step = np.zeros_like(rho)
                step[free] = direction * min(1.0, _MAX_STEP / np.max(np.abs(direction)))
                for _ in range(_MAX_HALVINGS):
                    trial_rho = np.clip(rho + step, -_LOG_SP_BOUND, _LOG_SP_BOUND)
                    trial = self.solve(np.exp(trial_rho))
                    if trial.gcv < st.gcv:
                        moved = True
                        break
                    step /= 2.0
                if moved:
                    break
            if not moved:
                break
            rho, st = trial_rho, trial
        return st

    def finish(self, st: _State, *, k: int) -> GAMFit:
        """Package a solved state as a :class:`GAMFit`."""
        F = st.P @ (st.Q1.T @ self.R)
        edf = np.diag(F)
        edf1 = 2.0 * edf - np.sum(F * F.T, axis=1)
        fitted = self.X @ st.coef
        return GAMFit(
            fitted=fitted,
            residuals=self.y - fitted,
            coef=st.coef,
            sp=st.sp,
            edf=np.array([edf[c].sum() for c in self.cols]),
            edf1=np.array([edf1[c].sum() for c in self.cols]),
            scale=st.rss / (self.n - st.trace),
            gcv=st.gcv,
            residual_df=self.n - st.trace,
            k=k,
            bases=self.bases,
            _cov_unscaled=st.P @ st.P.T,
        )


def _test_stat(
    coef: np.ndarray, X: np.ndarray, cov: np.ndarray, rank: float, residual_df: float
) -> tuple[float, float, float]:
    """Wood's (2013) test of ``f = 0`` for one term: ``(p-value, statistic, rank)``.

    ``cov`` is the term's Bayesian posterior covariance. The statistic is ``f' V_f^{r-} f`` with
    ``f = X coef`` and ``V_f^{r-}`` the rank-``r`` pseudoinverse of the covariance of ``f``. A
    non-integer rank ``r = k + nu`` blends the ``k``-th and ``(k+1)``-th eigen-directions, and the
    reference distribution is then ``sum_i w_i chi2_1`` against an F-type scaled ``chi2_{df}``; the
    p-value averages the two sign choices of the blended direction.
    """
    _, R = np.linalg.qr(X)
    V = R @ cov @ R.T
    V = (V + V.T) / 2
    values, vectors = np.linalg.eigh(V)
    values, vectors = values[::-1], vectors[:, ::-1]
    sign = np.sign(vectors[0])
    sign[sign == 0] = 1.0
    vectors = vectors * sign
    k = max(0, math.floor(rank))
    nu = abs(rank - k)
    k1 = k + 1 if nu > 0 else k
    numerical_rank = int(np.sum(values > values[0] * np.finfo(float).eps ** 0.9))
    if numerical_rank < k1:
        k1 = k = numerical_rank
        nu = 0.0
        rank = float(numerical_rank)
    vec = vectors[:, : max(k1, 1)].copy()
    if nu > 0 and k > 0:
        if k > 1:
            vec[:, : k - 1] /= np.sqrt(values[: k - 1])
        b12 = math.sqrt(max(0.0, 0.5 * nu * (1.0 - nu)))
        scale = np.diag(values[k - 1 : k1] ** -0.5)
        B = scale @ np.array([[1.0, b12], [b12, nu]]) @ scale
        w, U = np.linalg.eigh(B)
        root = U @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ U.T
        vec1 = vec.copy()
        vec1[:, k - 1 : k1] = vec[:, k - 1 : k1] @ np.diag([-1.0, 1.0]) @ root.T
        vec[:, k - 1 : k1] = vec[:, k - 1 : k1] @ root.T
    else:
        vec = vec / np.sqrt(values[0]) if k == 0 else vec / np.sqrt(values[:k])
        vec1 = vec
        if k == 1:
            rank = 1.0
    f = R @ coef
    stat = float(np.sum((vec.T @ f) ** 2))
    stat1 = float(np.sum((vec1.T @ f) ** 2))
    rank1 = rank
    p = 2.0
    if nu > 0:
        if k1 == 1:
            rank1 = 1.0
            weights = [1.0]
        else:
            rp = nu + 1.0
            wk = (rp + math.sqrt(rp * (2.0 - rp))) / 2.0
            weights = [1.0] * (k - 1) + [wk, rp - wk]
        df = max(1, round(residual_df))
        dofs = [1.0] * len(weights) + [float(df)]
        p = (
            _quadratic_form_sf([*weights, -stat / df], dofs)
            + _quadratic_form_sf([*weights, -stat1 / df], dofs)
        ) / 2.0
    if p > 1.0:
        f_sf = stats.f.sf
        p = float(f_sf(stat / rank1, rank1, residual_df) + f_sf(stat1 / rank1, rank1, residual_df))
        p /= 2.0
    return min(1.0, max(0.0, p)), stat, rank


def _quadratic_form_sf(weights: Sequence[float], dofs: Sequence[float]) -> float:
    """``P(sum_i w_i chi2_{h_i} > 0)`` for independent central chi-squares (Imhof 1961).

    Imhof's integral ``1/2 + (1/pi) int_0^inf sin(theta(u)) / (u rho(u)) du`` is evaluated by the
    midpoint rule with step ``4 pi / L``, where ``L`` bounds the range of the quadratic form's mass,
    so the rule's aliasing error is the form's probability beyond ``L`` (Davies 1973), and truncated
    where ``rho`` exceeds ``e^40``.
    """
    w = np.asarray(weights, dtype=np.float64)
    h = np.asarray(dofs, dtype=np.float64)
    w, h = w[w != 0], h[w != 0]
    spread = float(np.sum(np.abs(w) * h) + 12.0 * math.sqrt(2.0 * np.sum(w * w * h)))
    spread += 60.0 * float(np.max(np.abs(w)))
    step = 4.0 * math.pi / spread

    def log_rho(u: float) -> float:
        return float(0.25 * np.sum(h * np.log1p((w * u) ** 2)))

    top = 1.0 / float(np.max(np.abs(w)))
    while log_rho(top) + math.log(top) < 40.0:
        top *= 2.0
    nodes = math.ceil(top / step)
    if nodes > _MAX_NODES:
        return _quadratic_form_sf_quad(w, h, top)
    u = (np.arange(nodes) + 0.5) * step
    wu = w[:, None] * u[None, :]
    theta = 0.5 * (h[:, None] * np.arctan(wu)).sum(axis=0)
    rho = 0.25 * (h[:, None] * np.log1p(wu * wu)).sum(axis=0)
    return float(0.5 + step * np.sum(np.sin(theta) * np.exp(-rho) / u) / math.pi)


def _quadratic_form_sf_quad(w: np.ndarray, h: np.ndarray, top: float) -> float:
    """:func:`_quadratic_form_sf` by adaptive quadrature in ``s = log u`` up to ``log(top)``.

    In ``s`` the integrand's algebraic tail decays exponentially. Below ``u0`` the integrand is
    its value at ``0``, ``sum(h w) / 2``, to third order.
    """
    u0 = 1e-8 / float(np.sum(np.abs(w) * h))

    def integrand(s: float) -> float:
        u = math.exp(s)
        theta = 0.5 * np.sum(h * np.arctan(w * u))
        return float(np.sin(theta) * np.exp(-0.25 * np.sum(h * np.log1p((w * u) ** 2))))

    value, _ = integrate.quad(
        integrand, math.log(u0), math.log(top), epsabs=1e-13, epsrel=1e-11, limit=4000
    )
    return 0.5 + (value + 0.5 * float(np.sum(h * w)) * u0) / math.pi


def _r_sample(n: int, size: int, *, seed: int) -> np.ndarray:
    """0-based indices of R's ``set.seed(seed); sample.int(n, size)`` for ``n <= 1e7``.

    R seeds its Mersenne-Twister state by a scrambled linear congruential generator, draws uniforms
    as 32-bit outputs times ``2^-32``, and samples an index below ``n`` by rejection on the low
    ``ceil(log2 n)`` bits of 16-bit chunks; ``sample.int`` without replacement draws each index
    from the shrinking pool and moves the pool's last element into its slot.
    """
    state = seed & 0xFFFFFFFF
    for _ in range(50):
        state = (69069 * state + 1) & 0xFFFFFFFF
    key = np.empty(625, dtype=np.uint32)
    for i in range(625):
        state = (69069 * state + 1) & 0xFFFFFFFF
        key[i] = state
    generator = np.random.MT19937()
    generator.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}
    tiny = 2.328306437080797e-10

    def uniform() -> float:
        v = int(generator.random_raw()) * 2.3283064365386963e-10
        return 0.5 * tiny if v <= 0.0 else (1.0 - 0.5 * tiny if 1.0 - v <= 0.0 else v)

    def index_below(m: int) -> int:
        bits = math.ceil(math.log2(m))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):
                v = 65536 * v + math.floor(uniform() * 65536)
            v &= (1 << bits) - 1
            if v < m:
                return v

    pool = list(range(n))
    out = np.empty(size, dtype=np.intp)
    m = n
    for i in range(size):
        j = index_below(m)
        out[i] = pool[j]
        m -= 1
        pool[j] = pool[m]
    return out
