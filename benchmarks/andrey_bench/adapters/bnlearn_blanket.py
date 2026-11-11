"""bnlearn's Markov blanket and parents-and-children learners, the reference of the set-learner lane.

bnlearn 5.2.1 (:data:`PINNED_VERSION`; the driver stops on any other) runs the same learners as
Andrey's, in the same two ways as :mod:`andrey_blanket`:

- **One target at a time**: ``learn.mb(method=)`` for ``gs``, ``iamb``, ``fast.iamb``, and
  ``inter.iamb``; ``learn.nbr(method=)`` for ``mmpc`` and ``si.hiton.pc``; and ``pc.stable``, run
  once and read at each target, ``mb()`` for its Markov blanket and ``nbr()`` for its parents and
  children. ``learn.mb`` and ``learn.nbr`` return the answer of a search from one target, with no
  help from the searches of the others. ``test.counter()`` gives the tests of each call.
- **For the whole graph**: the learner's own function, ``undirected=TRUE`` for the skeleton and its
  CPDAG otherwise (``mmpc`` and ``si.hiton.pc`` return a skeleton only); the moral graph, which
  bnlearn does not return, is the AND rule over ``learn.mb`` of every variable.

The settings that reach both sides are the same: the test, ``alpha``, and the largest conditioning
set (Andrey's ``max_depth``, bnlearn's ``max.sx``). Where bnlearn's defaults differ from Andrey's,
the adapter does not default them:

- **The continuous test.** bnlearn's default is the exact t-test of the correlation (``cor``); the
  lane passes ``zf``, Fisher's z, Andrey's ``fisherz``.
- **The categorical tests.** ``mi`` is the G statistic, 2 N times the mutual information, Andrey's
  ``gsq``, and ``x2`` is Pearson's statistic, ``chisq``. ``adf``, the lane's ``sparse_rule``, picks
  ``mi-adf`` and ``x2-adf`` over ``mi`` and ``x2``, with two differences. The ``-adf`` tests take
  the empty rows and columns of a table out of the degrees of freedom; and they assume independence
  when a table averages fewer than 5 rows per cell (``n < 5 * r * c * q`` for ``r`` and ``c``
  levels of the two variables and ``q`` configurations of the conditioning set), so such a test
  reports ``p = 1``. The plain tests have neither. Andrey's tests have no such rule, and take the
  mean of the statistic under a shuffle as the degrees of freedom. Every row records the test used.
- **Conditioning-set size.** No limit is ``max.sx = Inf``, bnlearn's default. A limit must be at
  least 1 for bnlearn, so a run with ``max_depth=0`` has no reference. A limit means the same in
  the blanket learners: the blanket stops growing once it holds more than that many members. The
  lane always passes the limit, since a learner of Andrey can have its own default (HITON's 3).
  The parents-and-children learners differ: bnlearn's ``mmpc`` and ``si.hiton.pc`` also stop their
  forward phase once the candidate set holds more than ``max.sx`` members, and drop the candidates
  left, while Andrey's ``mmpc`` caps only the size of the conditioning sets. With a limit the two
  can learn different sets, so those learners agree only with no limit.
- **Symmetry of parents and children.** ``learn.nbr`` returns a one-sided set, and the whole-graph
  ``mmpc`` and ``si.hiton.pc`` keep a pair when each variable is in the other's set. Andrey's
  ``mmpc(target=)`` applies that rule, so the adapter does too: a variable stays in the set of a
  target when the target is in the variable's own ``learn.nbr``, and the tests of those runs are
  counted. A blanket has no such rule on either side.
- **Candidate order.** bnlearn breaks ties in the order of the columns, and Andrey in the order of
  a rank computed from the data, which does not depend on the columns' order. Where several
  p-values agree, for example when they underflow to zero, the two can add variables in a
  different order, and a learner can end with a different set.
- **Fast-IAMB's batch rule.** ``fast.iamb`` adds a candidate of a batch only when the data holds at
  least 5 observations per cell of the table of the target, the candidate, and the blanket so far
  (``obs.per.cell``). It applies to every asymptotic test, so to ``mi``, ``mi-adf``, ``x2``, and
  ``x2-adf`` alike, and not to continuous data, where the count is infinite. The sparse-table rule
  of the ``-adf`` tests is a separate rule, and applies to every learner.
- **Deprecation.** ``fast.iamb`` is deprecated in 5.2.1 and warns on every call; the adapter
  silences that one warning.

Discrete columns become R factors inside the timed call, as in the discrete lane's adapters.

To add a learner, name its function in :data:`FUNCTIONS` and, for the learner of Andrey it is the
reference of, in :data:`COUNTERPART`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from andrey_bench import rsession
from andrey_bench.adapters.r_adapter import RAdapter
from andrey_bench.blanket_lane import (
    NO_COUNT,
    STRUCTURE_PARAM,
    TARGETS_PARAM,
    answer_structure,
    pack_answer,
)

#: The bnlearn version the lane is written for; ``fast.iamb`` is removed after it.
PINNED_VERSION = "5.2.1"

_DEFAULT_ALPHA = 0.05

#: bnlearn's test for each of Andrey's, by whether the sparse-table rule is on (the ``-adf`` form).
TESTS: dict[tuple[str, bool], str] = {
    ("fisherz", True): "zf",
    ("fisherz", False): "zf",
    ("gsq", True): "mi-adf",
    ("gsq", False): "mi",
    ("chisq", True): "x2-adf",
    ("chisq", False): "x2",
}


@dataclass(frozen=True)
class Function:
    """What the lane needs to know about one bnlearn function."""

    #: ``"mb"`` for a learner of Markov blankets, ``"pc"`` for one of parents and children, and
    #: ``None`` for one that gives both, which the adapter is told to read.
    kind: str | None
    #: The graphs it learns for the whole graph, from :data:`~andrey_bench.blanket_lane.STRUCTURES`.
    structures: tuple[str, ...]


#: The functions the lane runs, by bnlearn's name.
FUNCTIONS: dict[str, Function] = {
    "gs": Function("mb", ("moral", "skeleton", "cpdag")),
    "iamb": Function("mb", ("moral", "skeleton", "cpdag")),
    "fast.iamb": Function("mb", ("moral", "skeleton", "cpdag")),
    "inter.iamb": Function("mb", ("moral", "skeleton", "cpdag")),
    "mmpc": Function("pc", ("skeleton",)),
    "si.hiton.pc": Function("pc", ("skeleton",)),
    "pc.stable": Function(None, ("skeleton", "cpdag")),
}

#: The function that runs the algorithm of each learner of Andrey, by the learner's name. The
#: learners without a counterpart (``hiton_mb``, ``mmmb``, ``pcmb``) have no bnlearn function.
COUNTERPART: dict[str, str] = {
    "iamb": "iamb",
    "inter_iamb": "inter.iamb",
    "mmpc": "mmpc",
    "gs": "gs",
    "fast_iamb": "fast.iamb",
    "hiton_pc": "si.hiton.pc",
}

_R_SOURCE = """
andrey_bnlearn_frame <- function(X, discrete) {
    df <- as.data.frame(X)
    if (discrete) df[] <- lapply(df, factor)
    df
}
# fast.iamb is deprecated in bnlearn 5.2.1 and says so on every call.
andrey_bnlearn_quiet <- function(expr) {
    withCallingHandlers(expr, warning = function(w) {
        if (grepl("deprecated", conditionMessage(w))) invokeRestart("muffleWarning")
    })
}
andrey_bnlearn_mb <- function(df, node, fn, test, alpha, max_sx) {
    members <- andrey_bnlearn_quiet(
        bnlearn::learn.mb(df, node, method = fn, test = test, alpha = alpha, max.sx = max_sx))
    list(members = members, tests = bnlearn::test.counter())
}
# The parents and children of `node`: its own search, kept to the variables whose search holds it.
andrey_bnlearn_pc <- function(df, node, fn, test, alpha, max_sx) {
    own <- bnlearn::learn.nbr(df, node, method = fn, test = test, alpha = alpha, max.sx = max_sx)
    tests <- bnlearn::test.counter()
    keep <- character(0)
    for (y in own) {
        back <- bnlearn::learn.nbr(df, y, method = fn, test = test, alpha = alpha, max.sx = max_sx)
        tests <- tests + bnlearn::test.counter()
        if (node %in% back) keep <- c(keep, y)
    }
    list(members = keep, tests = tests)
}
andrey_bnlearn_sets <- function(X, discrete, fn, kind, test, alpha, max_sx, targets) {
    df <- andrey_bnlearn_frame(X, discrete)
    nodes <- names(df)
    d <- length(nodes)
    answer <- matrix(0, d, d, dimnames = list(nodes, nodes))
    counts <- rep(-1, d)
    total <- -1
    if (fn == "pc.stable") {
        fit <- bnlearn::pc.stable(df, test = test, alpha = alpha, max.sx = max_sx)
        total <- fit$learning$ntests
        for (i in targets) {
            members <- if (kind == "mb") bnlearn::mb(fit, nodes[i]) else bnlearn::nbr(fit, nodes[i])
            answer[i, members] <- 1
        }
    } else {
        total <- 0
        for (i in targets) {
            run <- if (kind == "mb") andrey_bnlearn_mb else andrey_bnlearn_pc
            found <- run(df, nodes[i], fn, test, alpha, max_sx)
            answer[i, found$members] <- 1
            counts[i] <- found$tests
            total <- total + found$tests
        }
    }
    list(answer, counts, total)
}
andrey_bnlearn_graph <- function(X, discrete, fn, structure, test, alpha, max_sx) {
    df <- andrey_bnlearn_frame(X, discrete)
    nodes <- names(df)
    d <- length(nodes)
    if (structure == "moral") {
        answer <- matrix(0, d, d, dimnames = list(nodes, nodes))
        total <- 0
        for (i in seq_len(d)) {
            found <- andrey_bnlearn_mb(df, nodes[i], fn, test, alpha, max_sx)
            answer[i, found$members] <- 1
            total <- total + found$tests
        }
        answer <- answer * t(answer)
    } else {
        learn <- getExportedValue("bnlearn", fn)
        fit <- andrey_bnlearn_quiet(learn(df, test = test, alpha = alpha, max.sx = max_sx,
                                          undirected = (structure == "skeleton")))
        answer <- bnlearn::amat(fit)
        total <- fit$learning$ntests
    }
    list(answer, rep(-1, d), total)
}
"""


def check_version() -> None:
    """Stop unless the library bnlearn is installed in is the pinned release.

    The records say which build they come from, but the lane's claims are about 5.2.1: its
    ``fast.iamb`` is deprecated and goes after it, and the other learners' details are read from
    its source.
    """
    found = rsession.package_version("bnlearn")
    if found != PINNED_VERSION:
        raise SystemExit(
            f"the set-learner lane is written for bnlearn {PINNED_VERSION}, and the R library has "
            f"{found}. Install {PINNED_VERSION} from the CRAN archive "
            f"(https://cran.r-project.org/src/contrib/Archive/bnlearn/bnlearn_{PINNED_VERSION}.tar.gz)"
            " into the library of $ANDREY_BENCH_R_LIBS."
        )


class BnlearnBlanket(RAdapter):
    """Solution adapter for one function of :data:`FUNCTIONS`, at one setting.

    Satisfies :class:`~andrey_bench.contracts.SolutionAdapter` and
    :class:`~andrey_bench.contracts.SetupAdapter`. ``structure`` is the graph a whole-graph run
    answers with, and ``None`` for a run that answers for targets; ``kind`` is the set a function
    that gives both, ``pc.stable``, is read for. ``sparse_rule`` picks the ``-adf`` form of a
    categorical test (see the module docstring).
    """

    package: str = "bnlearn"
    package_version: str = rsession.package_version("bnlearn")
    backend: str = "native"
    mode: str = "serial"
    r_package: str = "bnlearn"
    r_source: str = _R_SOURCE
    r_function: str = "andrey_bnlearn_sets"

    def __init__(
        self,
        function: str,
        *,
        indep_test: str,
        alpha: float = _DEFAULT_ALPHA,
        max_depth: int | None = None,
        sparse_rule: bool = True,
        structure: str | None = None,
        kind: str | None = None,
    ) -> None:
        if function not in FUNCTIONS:
            raise ValueError(f"function must be one of {sorted(FUNCTIONS)}, got {function!r}")
        known = FUNCTIONS[function]
        if (indep_test, sparse_rule) not in TESTS:
            raise ValueError(f"indep_test must be one of fisherz, gsq, chisq, got {indep_test!r}")
        if max_depth is not None and max_depth < 1:
            raise ValueError("bnlearn's max.sx is at least 1; run max_depth=0 without a reference")
        if structure is not None and structure not in known.structures:
            raise ValueError(f"{function} gives the graphs {known.structures}, got {structure!r}")
        if known.kind is None:
            if kind is None and structure is None:
                raise ValueError(f"{function} gives either set; pass kind='mb' or kind='pc'")
        elif kind is not None and kind != known.kind:
            raise ValueError(f"{function} learns the {known.kind!r} set, not {kind!r}")
        self.function = function
        self.kind = known.kind or kind
        self.indep_test = indep_test
        self.alpha = alpha
        self.max_depth = max_depth
        self.sparse_rule = sparse_rule
        self.structure = structure
        self.name = (
            "bnlearn."
            + function.replace(".", "-")
            + (f".{self.kind or 'graph'}" if known.kind is None else "")
        )
        self.algorithm = function
        self.output_type = "cpdag" if structure == "cpdag" else "undirected"

    @property
    def bnlearn_test(self) -> str:
        """bnlearn's name of the test the run uses, ``mi-adf`` for ``gsq`` with the rule on."""
        return TESTS[(self.indep_test, self.sparse_rule)]

    def params(self) -> dict[str, Any]:
        """Every setting of the run, in Andrey's names where the two sides share one.

        ``indep_test`` is Andrey's test, ``bnlearn_test`` bnlearn's, and ``sparse_rule`` whether it
        is the ``-adf`` form that assumes independence on a sparse table.
        """
        return {
            "indep_test": self.indep_test,
            "bnlearn_test": self.bnlearn_test,
            "sparse_rule": self.sparse_rule and self.indep_test != "fisherz",
            "alpha": self.alpha,
            "max_depth": self.max_depth,
            "kind": self.kind,
            STRUCTURE_PARAM: self.structure,
        }

    def fit(self, data: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
        """Run the function; return :func:`~andrey_bench.blanket_lane.pack_answer`'s array.

        With ``params["targets"]``, each target's set is its row of the answer and the tests of its
        search are in the counts; ``pc.stable`` runs once and reports the fit's tests only.
        Without, the answer is the graph ``params["structure"]`` names, as a 0/1 adjacency. The
        only timed region.
        """
        import rpy2.robjects as ro

        session = rsession.session()
        data = np.asarray(data)
        d = data.shape[1]
        names = rsession.columns(d)
        test = TESTS[(str(params["indep_test"]), bool(params.get("sparse_rule", True)))]
        discrete = str(params["indep_test"]) != "fisherz"
        max_sx = float("inf") if params.get("max_depth") is None else int(params["max_depth"])
        alpha = float(params.get("alpha", self.alpha))
        targets = params.get(TARGETS_PARAM)
        matrix = session.matrix(data)
        if targets is None:
            structure = params.get(STRUCTURE_PARAM, self.structure)
            if structure is None:
                raise ValueError("a whole-graph run needs params['structure']")
            result = session.function("andrey_bnlearn_graph")(
                matrix, discrete, self.function, str(structure), test, alpha, max_sx
            )
        else:
            result = session.function("andrey_bnlearn_sets")(
                matrix,
                discrete,
                self.function,
                str(params.get("kind", self.kind)),
                test,
                alpha,
                max_sx,
                ro.IntVector([int(t) + 1 for t in targets]),
            )
        answer = session.square_matrix(result[0], names=names).astype(np.int8)
        counts = np.rint(np.asarray(result[1], dtype=np.float64)).astype(np.int64)
        total = int(round(float(np.asarray(result[2])[0])))
        return pack_answer(answer, counts, None if total == NO_COUNT else total)

    def to_structure(self, native: np.ndarray) -> Any:
        """The answer as a ``GraphStructure``; untimed. See ``answer_structure``."""
        return answer_structure(native, self.structure)


__all__ = ["COUNTERPART", "FUNCTIONS", "PINNED_VERSION", "BnlearnBlanket", "Function"]
