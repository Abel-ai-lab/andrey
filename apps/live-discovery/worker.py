"""One persistent library worker, controlled through a small JSON protocol on a private pipe."""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import time

#: The pipe descriptor the parent passes; library output on stdout never reaches it.
PROTOCOL_FD = "ANDREY_DEMO_PROTOCOL_FD"
MESSAGE_CHARS = 300


def main(package):
    """Import and warm the library, then time only its fit on each request."""
    protocol = os.fdopen(int(os.environ[PROTOCOL_FD]), "w", encoding="utf-8")

    def emit(message):
        protocol.write(json.dumps(message) + "\n")
        protocol.flush()

    import numpy as np
    from runners import prepare
    from threadpoolctl import threadpool_limits

    # Environment pins are installed by the parent before this interpreter starts.
    limits = threadpool_limits(limits=1)
    warm = np.random.default_rng(17).uniform(-1, 1, (200, 5))
    methods = {
        "Andrey": ("PC", "DirectLiNGAM", "GES", "FCI"),
        "causal-learn": ("PC", "DirectLiNGAM", "FCI"),
        "gCastle": ("PC",),
        "lingam": ("DirectLiNGAM",),
    }[package]
    for method in methods:
        fit, convert = prepare(package, method, warm)
        convert(fit())
    distribution = {"Andrey": "andrey-core", "gCastle": "gcastle"}.get(package, package)
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        import andrey

        version = andrey.__version__
    emit({"event": "ready", "version": version})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            data = np.load(request["data"], allow_pickle=False)
            fit, convert = prepare(package, request["method"], data)
            emit({"event": "fit"})
            start = time.perf_counter()
            result = fit()
            seconds = time.perf_counter() - start
            np.save(request["output"], convert(result), allow_pickle=False)
            emit({"event": "result", "seconds": seconds})
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            emit({"event": "error", "error": message[:MESSAGE_CHARS]})
    limits.restore_original_limits()


if __name__ == "__main__":
    main(sys.argv[1])
