"""Shared support for adapters that use the embedded R session.

Each R adapter declares a package, R source defining an entry point, and the entry point's name.
:meth:`RAdapter.setup` starts R and loads the package once per worker, outside timing.
:meth:`RAdapter.call` times data transfer, the R call, and retrieval of the result.

Every entry point returns a square matrix with `dimnames`.
:meth:`~andrey_bench.rsession.RSession.square_matrix` uses these names to handle variable reordering
without producing a permuted graph.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from andrey_bench import rsession


class RAdapter:
    """Base for a solution fitted through R.

    Subclasses declare :attr:`r_package`, :attr:`r_source`, and :attr:`r_function`, then the
    :class:`~andrey_bench.contracts.SolutionAdapter` metadata and a ``fit`` that calls
    :meth:`call`.
    """

    #: The R package `library()` loads in `setup`.
    r_package: str
    #: R source defining the entry point. Evaluated once per worker in `setup`.
    r_source: str
    #: The R function `call` invokes, defined by :attr:`r_source`.
    r_function: str

    def setup(self) -> None:
        """Start R, load the package, and define the entry point. Untimed; see `rsession`."""
        rsession.start(self.r_package).define(self.r_source)

    def call(self, data: np.ndarray, *args: Any) -> np.ndarray:
        """Transfer `data` to R, run the entry point, and retrieve its matrix. The timed region."""
        session = rsession.session()
        names = rsession.columns(np.asarray(data).shape[1])
        result = session.function(self.r_function)(session.matrix(data), *args)
        return session.square_matrix(result, names=names)
