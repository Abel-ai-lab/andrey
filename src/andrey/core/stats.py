"""Statistical primitives with automatic backend selection.

``cov``, ``corrcoef``, and ``entropy`` accept numpy arrays and return ``float64`` arrays. Numpy is
the exact oracle for ``cov`` and ``corrcoef`` and the reference for ``entropy`` (``atol 1e-3``).

Auto-dispatch uses per-device element-count cutoffs because compute and host-to-device transfer
scale with that metric. ``cov`` and ``corrcoef`` gate on ``n*d``; ``entropy`` gates on ``n*B``.
Only CUDA has validated defaults. Entropy may bootstrap the GPU runtime because it recurs throughout
a DirectLiNGAM fit. The one-shot covariance operations use a GPU only when its runtime is live.

``ANDREY_COV_GPU_THRESHOLD`` and ``ANDREY_ENTROPY_GPU_THRESHOLD`` set manual cutoffs.
``ANDREY_GPU_CALIBRATE`` measures and caches local crossovers. See
``docs/docs/configuration.md``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import pathlib
import platform
import statistics
import threading
import time

import numpy as np

from . import backend, env

_log = logging.getLogger(__name__)

# Per-device auto-dispatch cutoffs use element count because compute and host-to-device transfer
# scale with it. DirectLiNGAM uses tall `n` by `B <= ~2d` batches. Column count alone is unsuitable.
#
# Five `qa/gpu_calibration_validate.py` runs on A100-SXM4-80GB GPUs produced these brackets:
#
#   entropy   7,433   7,433   6,250   7,433   7,433
#   cov     148,651 125,000  88,388  37,163 250,000
#
# Each result is a lattice point within `_BRACKET_TOL`, not an exact crossover. Defaults are at the
# low end of each band. A cutoff that is too low costs bounded overhead near the cutoff. A cutoff
# that is too high forfeits the speedup. Entropy forms a narrow band. Cov spans nearly `7x`.
# Its minimum is one noisy draw of five, so its default is above that minimum. The liveness rule,
# not this value, keeps one-shot calls off a cold device. `auto` never reaches `mps`
# (`backend._TORCH_AUTO`); its entries stay `None` because no Metal crossover has been measured,
# and `qa/gpu_calibration_validate.py` reads them. Re-measure after a change to the timed path:
# the torch or numpy kernels, `backend.to_numpy`, or the probe.
_COV_GPU_THRESHOLDS: dict[str, int | None] = {"cuda": 90_000, "mps": None}  # elements n*d
_ENTROPY_GPU_THRESHOLDS: dict[str, int | None] = {"cuda": 6_250, "mps": None}  # elements n*B

# Maximum-entropy approximation of differential entropy (Hyvarinen 1998), DirectLiNGAM's measure.
_K1 = 79.047
_K2 = 7.4129
_GAMMA = 0.37457
_HALF_LOG_2PI_E = (1.0 + math.log(2.0 * math.pi)) / 2.0


# Opt-in calibration probes the gate's element-count metric with workload-realistic tall arrays.
# It steps up to a device win, down to a numpy win, and bisects the resulting bracket.
_COV_PROBE_START = 500_000  # elements n*d, the first size tried
_ENTROPY_PROBE_START = 25_000  # elements n*B
_PROBE_FLOOR = 1_024  # below this both paths are call overhead, and a crossover is meaningless
# The ceiling limits the float64 input to about 134 MB. Numpy also allocates a centered copy and
# temporaries, so larger probes risk an uncatchable OOM kill.
_PROBE_CEILING = 1 << 24
_BRACKET_TOL = 1.2  # stop bisecting once the bracket is this tight; tighter is noise, not signal
_MAX_PROBES = 90  # a hard bound on measurements, so a pathological curve cannot sweep forever
_ROUNDS = 3  # independent timing rounds per decision; the median decides
_NEVER_TTL_S = 3600  # how long a "the device never wins" verdict stays cached on disk
_MARGIN = 1.25  # the device must beat numpy by this factor to count as a real win (timing noise)
_NEVER = 1 << 62  # a cutoff no real workload reaches: the device stays on numpy
_CALIBRATED: dict[str, int | None] = {}  # fingerprint -> cutoff, in-process cache
_CALIBRATION_VERSION = 3  # bump when the probe or search changes; retires every cached entry
_calibrating = threading.Lock()  # one sweep at a time in this process


def _reset_after_fork() -> None:
    """Replace the inherited calibration lock in a forked child.

    A child forked during calibration can inherit a lock whose owning thread does not exist there.
    """
    global _calibrating
    _calibrating = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


def _probe_array(metric: int, aspect: int) -> np.ndarray:
    """A ``metric``-element standard-normal array with ``rows ~ aspect * columns``."""
    cols = max(2, round(math.sqrt(metric / aspect)))
    return np.random.default_rng(0).standard_normal((max(2, metric // cols), cols))


def _cov_probe(metric: int) -> np.ndarray:
    # cov inputs are data matrices (n samples >> d variables): n:d ~ 8 spans the realistic range.
    return _probe_array(metric, aspect=8)


# Match the `rowvar=False` orientation used by `BICScore` and Fisher-Z.
def _cov_probe_call(x: np.ndarray) -> np.ndarray:
    return cov(x, rowvar=False)


def _corrcoef_probe_call(x: np.ndarray) -> np.ndarray:
    return corrcoef(x, rowvar=False)


def _entropy_probe(metric: int) -> np.ndarray:
    # entropy batches are residual stacks (n samples >> B columns): n:B ~ 32 matches DirectLiNGAM
    # (n in the hundreds-to-thousands, B <= ~2d) around the crossover region.
    return _probe_array(metric, aspect=32)


def _available_accelerator() -> str | None:
    """The first available device ``auto`` may select, or ``None``."""
    for dev in backend._TORCH_AUTO:
        if dev != "numpy" and backend._is_available(dev):
            return dev
    return None


def _live_accelerator() -> str | None:
    """The ``auto`` device already running in this process, without importing torch to find out."""
    for dev in backend._TORCH_AUTO:
        if dev != "numpy" and backend._is_live(dev):
            return dev
    return None


def _timed(call, x: np.ndarray) -> float:
    start = time.perf_counter()
    call(x)
    return time.perf_counter() - start


def _cpu_threads() -> str:
    """Describe CPU parallelism for the numpy side of a crossover.

    Includes torch's intra-op count and the OpenMP, OpenBLAS, and MKL thread variables.
    """
    torch_threads = "?"
    try:
        t = backend.torch()
        if t is not None:
            torch_threads = str(t.get_num_threads())
    except Exception:  # noqa: BLE001 - identification must not break a stats call
        pass
    blas = ",".join(
        os.environ.get(v, "?")
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    )
    return f"{torch_threads}/{blas}"


def _device_fingerprint(device: str) -> str:
    """Identify the machine state used to measure a cached crossover.

    The fingerprint includes the probe version, host, accelerator, device name, dtype, CPU thread
    settings, and the torch and numpy versions, so a library upgrade re-measures. Unknown fields use
    placeholders to preserve the key structure.
    """
    name = ""
    torch_version = "?"
    try:
        t = backend.torch()
        if t is not None:
            torch_version = t.__version__
            if device == "cuda" and t.cuda.is_available():
                name = t.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - identification must not break a stats call
        name = ""
    return (
        f"v{_CALIBRATION_VERSION}|{platform.node()}|{device}|{name or '?'}|f8|t{_cpu_threads()}"
        f"|torch{torch_version}|numpy{np.__version__}"
    )


def _calibration_file() -> pathlib.Path:
    root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return pathlib.Path(root) / "andrey" / "gpu_calibration.json"


def _load_calibration(fingerprint: str) -> int | None:
    """The cached cutoff for ``fingerprint``, or ``None`` when there is no usable entry.

    A "never wins" entry expires after ``_NEVER_TTL_S`` so transient load does not disable device
    dispatch indefinitely.
    """
    try:
        entry = json.loads(_calibration_file().read_text())[fingerprint]
    except Exception:  # noqa: BLE001 - missing/corrupt cache means "not calibrated yet"
        return None
    if isinstance(entry, dict):
        if time.time() - float(entry.get("at", 0)) > _NEVER_TTL_S:
            return None
        return _NEVER
    try:
        return int(entry)
    except (TypeError, ValueError):
        return None


def _store_calibration(fingerprint: str, cutoff: int | dict) -> None:
    """Best-effort write-through of a measured cutoff (atomic replace; failure only logs)."""
    try:
        path = _calibration_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                data = {}
        except Exception:  # noqa: BLE001 - a corrupt cache is rebuilt, not fatal
            data = {}
        data[fingerprint] = cutoff
        # The pid alone collides across nodes sharing one home directory; add the host.
        tmp = path.with_suffix(f".{platform.node()}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)
    except Exception as exc:  # noqa: BLE001 - persistence is an optimization, never an error
        _log.debug("andrey: could not persist the gpu calibration (%s)", exc)


@contextlib.contextmanager
def _calibration_gate(fingerprint: str):
    """Hold an exclusive lock while measuring, so only one process probes a given ``fingerprint``.

    This prevents fresh pool workers from measuring against one another. Waiters re-read the cache
    after acquiring the lock. Filesystems without lock support continue without cross-process
    serialization.
    """
    path = _calibration_file().with_suffix(".lock")
    handle = None
    try:
        # Import locally so Windows can use the CPU path without `fcntl`.
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "w")  # noqa: SIM115 - released in the finally below
        fcntl.flock(handle, fcntl.LOCK_EX)
    except Exception as exc:  # noqa: BLE001 - locking is an optimization, never an error
        _log.debug("andrey: could not lock the calibration cache (%s)", exc)
    try:
        yield
    finally:
        if handle is not None:
            handle.close()  # closing the descriptor releases the flock


def _crossover(device: str, make, call, start: int) -> int | None:
    """Return the smallest probed element count where ``device`` beats numpy by ``_MARGIN``.

    Steps up until the device wins, steps down until numpy wins, and bisects on the log scale. The
    downward search permits a result below ``start``.

    An ascent win must hold at twice the size. Each decision uses the median of ``_ROUNDS`` timing
    rounds.

    Returns ``None`` if no win is confirmed. A probed win at ``_PROBE_FLOOR`` can be returned.
    """
    probes = 0

    def ratio(metric: int) -> float:
        """``device * _MARGIN / numpy`` at ``metric`` elements; below 1 the device wins."""
        nonlocal probes
        probes += 1
        x = make(metric)
        with backend.unrecorded_dispatch():
            with backend.config(backend=device):
                for _ in range(2):  # warmup: JIT, cuBLAS handle, first transfer
                    call(x)
                dev_s = min(_timed(call, x) for _ in range(3))
            with backend.config(backend="numpy"):
                for _ in range(2):
                    call(x)
                cpu_s = min(_timed(call, x) for _ in range(3))
        return dev_s * _MARGIN / cpu_s if cpu_s > 0 else math.inf

    def wins(metric: int) -> bool:
        """Return whether the median of ``_ROUNDS`` timings favors the device."""
        return statistics.median(ratio(metric) for _ in range(_ROUNDS)) < 1.0

    # Ascent: find a size where the device wins. Confirm each win at twice the size.
    hi = max(_PROBE_FLOOR, start)
    confirmed = False
    while probes < _MAX_PROBES:
        if wins(hi) and wins(min(hi * 2, _PROBE_CEILING)):
            confirmed = True
            break
        if hi >= _PROBE_CEILING:
            return None  # the device never wins at any size worth offloading
        hi = min(hi * 4, _PROBE_CEILING)  # clamp, so the ceiling itself is always tried
    if not confirmed:
        # The probe budget ended without a confirmed device win.
        return None

    # Descent: decrease the size from the confirmed win until numpy wins.
    lo = max(_PROBE_FLOOR, hi // 4)
    while probes < _MAX_PROBES and wins(lo):
        hi = lo
        if lo <= _PROBE_FLOOR:
            return _PROBE_FLOOR  # a probed win at the floor: offload as soon as it is worth it
        lo = max(_PROBE_FLOOR, lo // 4)

    # Bisection: halve the bracket on the log scale.
    while probes < _MAX_PROBES and hi / lo > _BRACKET_TOL:
        mid = int(round(math.sqrt(lo * hi)))
        if mid <= lo or mid >= hi:
            break
        if wins(mid):
            hi = mid
        else:
            lo = mid
    return hi


def _calibrate(op: str, device: str, make, call, start: int, default: int | None) -> int | None:
    """Measure the numpy->``device`` crossover for ``op`` on this machine; cache and log it.

    Device and OOM errors use ``default`` for the current process. Successful measurements are
    cached by machine fingerprint for reuse across processes. A "never wins" result is cached with
    a timestamp and expires after ``_NEVER_TTL_S``.
    """
    with _calibrating:  # check-then-act: two threads must not both start a sweep
        # Include the machine fingerprint in the in-process cache key.
        key = fingerprint = f"{op}|{_device_fingerprint(device)}"
        if key in _CALIBRATED:
            return _CALIBRATED[key]
        cached = _load_calibration(fingerprint)
        if cached is not None:
            _CALIBRATED[key] = cached
            _log.info(
                "andrey: %s on %s: reusing the cached calibration (%s)", op, device, fingerprint
            )
            return cached
        with _calibration_gate(fingerprint):
            # Re-read after acquiring the cross-process lock.
            cached = _load_calibration(fingerprint)
            if cached is not None:
                _CALIBRATED[key] = cached
                _log.info(
                    "andrey: %s on %s: another process calibrated it (%s)", op, device, cached
                )
                return cached
            _log.info(
                "andrey: ANDREY_GPU_CALIBRATE: measuring numpy->%s crossover for %s", device, op
            )
            try:
                cutoff = _crossover(device, make, call, start)
            except Exception as exc:  # noqa: BLE001 - a device error must not break a stats call
                _log.warning(
                    "andrey: calibrating %s on %s failed (%s); using the default cutoff",
                    op,
                    device,
                    exc,
                )
                _CALIBRATED[key] = default
                return default
            if cutoff is None:
                _CALIBRATED[key] = _NEVER
                _store_calibration(fingerprint, {"never": True, "at": time.time()})
                _log.info(
                    "andrey: %s never beat numpy on %s in calibration; keeping it on numpy",
                    op,
                    device,
                )
                return _NEVER
            _CALIBRATED[key] = cutoff
            _store_calibration(fingerprint, cutoff)
            _log.info("andrey: calibrated %s on %s -> offload at metric >= %d", op, device, cutoff)
            return cutoff


def _resolve_thresholds(
    op: str,
    setting: env.Setting[int],
    defaults: dict[str, int | None],
    make,
    call,
    start: int,
    *,
    amortized: bool,
) -> int | dict[str, int | None]:
    """Resolve cutoffs from an environment override, calibration, or per-device defaults.

    Calibration requires ``auto``. Repeated operations may initialize an available accelerator;
    one-shot operations calibrate only on an accelerator whose runtime is already live.
    """
    manual = setting.read()
    if manual is not None:
        return manual
    if env.GPU_CALIBRATE.read():
        if backend.config.backend == "auto":
            dev = _available_accelerator() if amortized else _live_accelerator()
            if dev is not None:
                return {**defaults, dev: _calibrate(op, dev, make, call, start, defaults[dev])}
    return defaults


def cov(X: np.ndarray, *, rowvar: bool = True, ddof: int = 1) -> np.ndarray:
    """Covariance matrix, matching ``np.cov(X, rowvar=rowvar, ddof=ddof)``.

    ``rowvar=True`` treats each row as a variable (numpy's default); ``ddof=1`` is the sample
    covariance. numpy is the exact oracle; the torch path equals it within device tolerance.
    """
    X = np.asarray(X, dtype=np.float64)
    thr = _resolve_thresholds(
        "cov",
        env.COV_GPU_THRESHOLD,
        _COV_GPU_THRESHOLDS,
        _cov_probe,
        _cov_probe_call,
        _COV_PROBE_START,
        amortized=False,
    )
    # Covariance runs once per fit and offloads only to an already-live device under `auto`.
    dev = backend.resolve(X.size, thr, op="cov")
    if dev == "numpy":
        return np.cov(X, rowvar=rowvar, ddof=ddof)
    torch = backend.torch()
    t = backend.as_tensor(X if rowvar else X.T, dev)
    return backend.to_numpy(torch.cov(t, correction=ddof))


def corrcoef(X: np.ndarray, *, rowvar: bool = True) -> np.ndarray:
    """Pearson correlation matrix, matching ``np.corrcoef(X, rowvar=rowvar)``."""
    X = np.asarray(X, dtype=np.float64)
    thr = _resolve_thresholds(
        "corrcoef",
        env.COV_GPU_THRESHOLD,
        _COV_GPU_THRESHOLDS,
        _cov_probe,
        _corrcoef_probe_call,
        _COV_PROBE_START,
        amortized=False,
    )
    # One-shot, like cov: FisherZ computes its correlation matrix once per fit.
    dev = backend.resolve(X.size, thr, op="corrcoef")
    if dev == "numpy":
        return np.corrcoef(X, rowvar=rowvar)
    torch = backend.torch()
    t = backend.as_tensor(X if rowvar else X.T, dev)
    return backend.to_numpy(torch.corrcoef(t))


def entropy(U: np.ndarray) -> np.ndarray:
    """Differential-entropy approximation for each column of ``U`` (shape ``(n_samples, d)``).

    The maximum-entropy approximation used by DirectLiNGAM; inputs are assumed standardized. The
    numpy path is the reference; the torch path matches it within ``atol 1e-3``.
    """
    U = np.asarray(U, dtype=np.float64)
    if U.ndim == 1:
        U = U[:, None]
    thr = _resolve_thresholds(
        "entropy",
        env.ENTROPY_GPU_THRESHOLD,
        _ENTROPY_GPU_THRESHOLDS,
        _entropy_probe,
        entropy,
        _ENTROPY_PROBE_START,
        amortized=True,
    )
    # Entropy gates on `U.size = n*B` and may bootstrap the device because it recurs per fit.
    dev = backend.resolve(U.size, thr, amortized=True, op="entropy")
    if dev == "numpy":
        return _entropy_numpy(U)
    return _entropy_torch(U, dev)


def _entropy_numpy(U: np.ndarray) -> np.ndarray:
    t1 = np.mean(np.log(np.cosh(U)), axis=0) - _GAMMA
    t2 = np.mean(U * np.exp(-(U**2) / 2.0), axis=0)
    return _HALF_LOG_2PI_E - _K1 * t1**2 - _K2 * t2**2


def _entropy_torch(U: np.ndarray, dev: str) -> np.ndarray:
    torch = backend.torch()
    t = backend.as_tensor(U, dev)
    t1 = torch.mean(torch.log(torch.cosh(t)), dim=0) - _GAMMA
    t2 = torch.mean(t * torch.exp(-(t**2) / 2.0), dim=0)
    ent = _HALF_LOG_2PI_E - _K1 * t1**2 - _K2 * t2**2
    return backend.to_numpy(ent)
