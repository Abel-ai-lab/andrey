"""Run the UT-IGSP reference in its own environment: one request in, one JSON line per draw out.

``ut_igsp_cpu.py`` and ``igsp_cpu.py`` start this file with ``ANDREY_BENCH_UTIGSP_PYTHON``, the
interpreter ``start_utigsp.sh`` builds, or with ``ANDREY_BENCH_GML_DEF42_PYTHON``, the same
packages with IGSP's I-contradicting test patched to Definition 4.2 of Yang, Katcoff, and Uhler
(2018), as ``python gml_interventional.py <request.npz> <out.jsonl>``. It imports NumPy and the
authors' library alone: graphical-model-learning 0.1a8, conditional-independence 0.1a6, and
graphical-models 0.1a21. Nothing here imports Andrey.

A ``fit`` request holds the pooled rows ``X``, each row's environment ``env`` (0 the observational
one, 1..K the settings), ``known``, the 0/1 mask of each row's known targets, and ``fits``, a JSON
list. Each fit names an ``algorithm``, a ``config``, its ``alpha``, ``nruns``, ``tester``, and
``draws``:

- ``ut_igsp``: ``unknown_target_igsp`` at the library defaults (depth 4, ``use_lowest``,
  ``tup_score``), each setting's known targets as ``known_interventions``. Its targets are the
  known ones and the learned ones, as the authors score them; ``learned`` holds the learned sets
  alone and ``known_missing`` the known targets missing from them.
- ``igsp``: ``igsp`` with each setting's known targets as its ``interventions``.
- ``jci_gsp``: :func:`jci_gsp_fixed`, with the authors' combined tester. Its targets are the
  children of each setting's context node.

``tester`` is ``gauss`` (the partial-correlation CI test and the Gaussian invariance test),
``kci`` (``kci_test`` and ``kci_invariance_test`` at the library defaults, which regress on the
conditioning set with a GAM and test the residuals with HSIC), or ``kci_conditional`` (the same
with ``regress=False``). Both tests run at ``alpha``.

Each draw seeds Python's ``random`` and NumPy's global generator with the draw number, and the
clock covers the testers' construction and the search. Before the first draw of each fit, one
untimed warm-up runs the same call on the first ``warmup_rows`` rows of each environment. Each draw
appends one line as it finishes, so a killed child keeps the draws it finished.

An ``icpdag`` request holds ``pairs``, a JSON list of ``{"p", "arcs", "interventions"}``, and
answers each with graphical-models' ``DAG.interventional_cpdag``: its ``arcs`` and ``edges``.

An ``oracle`` request holds ``cases``, a JSON list of ``{"p", "arcs", "family", "queries",
"seeds"}``: a DAG, its settings' targets, and the tests answered by d-separation in the DAG with one
context node per setting into its targets (:class:`OracleTesters`). Each query ``{"arcs", "arc"}``
asks whether IGSP's I-contradicting test, as the installed package builds it for ``family``, marks
``arc`` in the DAG ``arcs``; each seed runs ``igsp`` with the oracle testers at the library
defaults. One line per case: ``contradicting`` (one answer per query) and ``found`` (the arcs of
each seed's DAG).

A ``probe`` request holds ``Xs``, several datasets of one shape with one ``env``, and one ``fits``
entry; each dataset runs one draw, its ``seeds`` entry, and answers one line with its ``arcs``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import itertools
import json
import random
import resource
import sys
import time
import traceback

import numpy as np

PACKAGES = ("graphical-model-learning", "conditional-independence", "graphical-models")


def versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in PACKAGES}


def combined_gauss_ci_test(suffstat, i, j, cond_set=None, alpha=0.01, alpha_inv=0.01):
    """The authors' JCI-GSP tester: a CI test between two variables, an invariance test between a
    context node ``c<k>`` and a variable; context nodes are dropped from the conditioning set."""
    from conditional_independence import gauss_invariance_test, partial_correlation_test

    cond_set = {c for c in (cond_set or ()) if not isinstance(c, str)}
    if isinstance(i, int) and isinstance(j, int):
        return partial_correlation_test(suffstat["ci"], i, j, cond_set=cond_set, alpha=alpha)
    context, node = (int(i[1:]), j) if isinstance(i, str) else (int(j[1:]), i)
    return gauss_invariance_test(
        suffstat["invariance"], context, node, cond_set=cond_set, alpha=alpha_inv
    )


def jci_gsp_fixed(setting_list, nodes, combined_ci_tester, depth=4, nruns=5):
    """The 0.1a8 body of ``jci_gsp`` with two call fixes.

    The shipped function raises before any test: it passes tuple pairs as ``fixed_adjacencies``,
    which ``permutation2dag`` rejects, and unpacks two values from ``gsp``, which returns the DAG
    alone unless ``summarize=True``. Here the minimal-I-MAP builder receives the pairs as
    frozensets while ``update_minimal_imap`` keeps the tuples it checks, and ``gsp`` runs with
    ``summarize=True``.
    """
    # The package exports a function named gsp, which hides the module of that name.
    gsp_module = importlib.import_module("graphical_model_learning.algorithms.dag.gsp")
    from graphical_model_learning.algorithms.undirected import threshold_ug

    context_nodes = [f"c{i}" for i in range(len(setting_list))]
    context_adjacencies = set(itertools.permutations(context_nodes, r=2))
    known_iv_adjacencies = set.union(
        *(
            {(f"c{i}", node) for node in s["known_interventions"]}
            for i, s in enumerate(setting_list)
        )
    )
    fixed_orders = set(itertools.combinations(context_nodes, 2)) | set(
        itertools.product(context_nodes, nodes)
    )
    amat = threshold_ug(set(nodes), combined_ci_tester).to_amat()
    initial_permutations = [
        context_nodes + gsp_module.min_degree_alg_amat(amat) for _ in range(nruns)
    ]
    original = gsp_module.permutation2dag

    def permutation2dag(
        perm, ci_tester, fixed_adjacencies=frozenset(), fixed_gaps=frozenset(), **kw
    ):
        return original(
            perm,
            ci_tester,
            fixed_adjacencies={frozenset(a) for a in fixed_adjacencies},
            fixed_gaps={frozenset(g) for g in fixed_gaps},
            **kw,
        )

    gsp_module.permutation2dag = permutation2dag
    try:
        meta_dag, _ = gsp_module.gsp(
            nodes | set(context_nodes),
            combined_ci_tester,
            depth=depth,
            nruns=nruns,
            initial_permutations=initial_permutations,
            fixed_orders=fixed_orders,
            fixed_adjacencies=context_adjacencies | known_iv_adjacencies,
            summarize=True,
        )
    finally:
        gsp_module.permutation2dag = original
    learned = [
        {c for c in meta_dag.children_of(f"c{i}") if not isinstance(c, str)}
        for i in range(len(setting_list))
    ]
    dag = meta_dag.induced_subgraph({v for v in meta_dag.nodes if not isinstance(v, str)})
    return dag, learned


def testers(fit: dict, obs: np.ndarray, ivs: list[np.ndarray]):
    """The CI tester and the invariance tester of ``fit``, built on the stored rows."""
    from conditional_independence import (
        MemoizedCI_Tester,
        MemoizedInvarianceTester,
        gauss_invariance_suffstat,
        gauss_invariance_test,
        kci_invariance_test,
        kci_test,
        partial_correlation_suffstat,
        partial_correlation_test,
    )

    alpha, tester = fit["alpha"], fit["tester"]
    if tester == "gauss":
        ci = MemoizedCI_Tester(
            partial_correlation_test, partial_correlation_suffstat(obs), alpha=alpha
        )
        inv = MemoizedInvarianceTester(
            gauss_invariance_test, gauss_invariance_suffstat(obs, ivs), alpha=alpha
        )
        return ci, inv
    if tester not in ("kci", "kci_conditional"):
        raise ValueError(f"unknown tester {tester!r}")
    regress = tester == "kci"
    # kci_test indexes its suffstat by column, so it takes the array itself.
    ci = MemoizedCI_Tester(kci_test, obs, alpha=alpha, regress=regress)
    contexts = {"obs_samples": obs, **dict(enumerate(ivs))}
    inv = MemoizedInvarianceTester(kci_invariance_test, contexts, alpha=alpha, regress=regress)
    return ci, inv


def run_fit(fit: dict, X: np.ndarray, env: np.ndarray, known: list[list[int]]) -> dict:
    """One call of the fit's algorithm on ``X``: its arcs and targets."""
    import graphical_model_learning as gml
    from conditional_independence import (
        MemoizedCI_Tester,
        gauss_invariance_suffstat,
        partial_correlation_suffstat,
    )

    p, settings = X.shape[1], len(known)
    nodes = set(range(p))
    obs = X[env == 0]
    ivs = [X[env == k] for k in range(1, settings + 1)]
    algorithm, nruns = fit["algorithm"], fit["nruns"]
    if algorithm == "ut_igsp":
        ci, inv = testers(fit, obs, ivs)
        dag, learned = gml.unknown_target_igsp(
            [{"known_interventions": k} for k in known], nodes, ci, inv, nruns=nruns
        )
        learned = [sorted(int(v) for v in t) for t in learned]
        return {
            "arcs": sorted([int(i), int(j)] for i, j in dag.arcs),
            "targets": [sorted(set(k) | set(t)) for k, t in zip(known, learned, strict=True)],
            "learned": learned,
            "known_missing": sum(len(set(k) - set(t)) for k, t in zip(known, learned, strict=True)),
        }
    if algorithm == "igsp":
        ci, inv = testers(fit, obs, ivs)
        dag = gml.igsp([{"interventions": k} for k in known], nodes, ci, inv, nruns=nruns)
        return {"arcs": sorted([int(i), int(j)] for i, j in dag.arcs), "targets": known}
    if algorithm == "jci_gsp":
        if fit["tester"] != "gauss":
            raise ValueError("jci_gsp runs the authors' Gaussian tester only")
        suffstat = {
            "ci": partial_correlation_suffstat(obs),
            "invariance": gauss_invariance_suffstat(obs, ivs),
        }
        tester = MemoizedCI_Tester(
            combined_gauss_ci_test, suffstat, alpha=fit["alpha"], alpha_inv=fit["alpha"]
        )
        dag, learned = jci_gsp_fixed(
            [{"known_interventions": k} for k in known], nodes, tester, nruns=nruns
        )
        return {
            "arcs": sorted([int(i), int(j)] for i, j in dag.arcs),
            "targets": [sorted(int(v) for v in t) for t in learned],
        }
    raise ValueError(f"unknown algorithm {algorithm!r}")


def head_rows(env: np.ndarray, rows: int) -> np.ndarray:
    """The first ``rows`` rows of each environment, for the warm-up."""
    return np.concatenate([np.flatnonzero(env == k)[:rows] for k in np.unique(env)])


def fit_request(request, sink) -> None:
    X, env, mask = request["X"], request["env"], request["known"]
    settings = int(env.max())
    known = [
        sorted(int(v) for v in np.flatnonzero(mask[np.flatnonzero(env == k)[0]]))
        for k in range(1, settings + 1)
    ]
    head = head_rows(env, int(request["warmup_rows"]))
    for fit in json.loads(str(request["fits"])):
        warmup_s, warmup_error = None, None
        t0 = time.perf_counter()
        try:
            random.seed(0)
            np.random.seed(0)
            run_fit(fit, X[head], env[head], known)
            warmup_s = time.perf_counter() - t0
        except Exception as exc:  # a warm-up failure is recorded; the timed draws still run
            warmup_error = f"{type(exc).__name__}: {exc}"
        for draw in fit["draws"]:
            line = {"algorithm": fit["algorithm"], "config": fit["config"], "draw": draw}
            random.seed(draw)
            np.random.seed(draw)
            t0 = time.perf_counter()
            try:
                result = run_fit(fit, X, env, known)
                line.update(status="ok", wall_s=time.perf_counter() - t0, **result)
            except Exception:
                line.update(status="error", error_text=traceback.format_exc()[-2000:])
            line.update(
                warmup_s=warmup_s,
                warmup_error=warmup_error,
                peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                versions=versions(),
            )
            sink.write(json.dumps(line) + "\n")
            sink.flush()


class OracleTesters:
    """IGSP's two tests answered by d-separation: the DAG ``arcs`` on ``p`` variables with context
    node ``p + k`` into the targets of setting ``k``. Every context but the tested one is given, as
    in the observational distribution."""

    def __init__(self, p: int, arcs: list, family: list[list[int]]):
        import graphical_models as gm

        self.p, self.contexts = p, set(range(p, p + len(family)))
        edges = {(int(i), int(j)) for i, j in arcs}
        edges |= {(p + k, int(v)) for k, targets in enumerate(family) for v in targets}
        self.dag = gm.DAG(nodes=set(range(p + len(family))), arcs=edges)

    def is_ci(self, i, j, cond_set=()) -> bool:
        return self.dag.dsep({i}, {j}, set(cond_set) | self.contexts)

    def is_invariant(self, node, context, cond_set=()) -> bool:
        given = set(cond_set) | (self.contexts - {self.p + context})
        return self.dag.dsep({self.p + context}, {node}, given)


class _Found(Exception):
    pass


def i_contradicting(setting_list: list[dict], nodes: set, invariance_tester):
    """The I-contradicting test ``igsp`` builds for ``setting_list``, taken from its frame.

    ``igsp`` defines the test inside its body before its first CI test; a CI tester that raises on
    its first call stops ``igsp`` there and reads the test from the frame.
    """
    import inspect

    import graphical_model_learning as gml

    found: dict = {}

    class Stop:
        def is_ci(self, *args, **kwargs):
            frame = inspect.currentframe()
            while frame is not None and "_is_i_contradicting" not in frame.f_locals:
                frame = frame.f_back
            if frame is not None:
                found.update(frame.f_locals)
            raise _Found

    try:
        gml.igsp(setting_list, nodes, Stop(), invariance_tester, nruns=1)
    except _Found:
        pass
    return found["_is_i_contradicting"]


def oracle_request(request, sink) -> None:
    import graphical_model_learning as gml
    import graphical_models as gm

    for case in json.loads(str(request["cases"])):
        p, family = case["p"], case["family"]
        testers = OracleTesters(p, case["arcs"], family)
        settings = [{"interventions": list(t)} for t in family]
        nodes = set(range(p))
        test = i_contradicting(settings, nodes, testers)
        answers = []
        for query in case["queries"]:
            dag = gm.DAG(nodes=nodes, arcs={(int(i), int(j)) for i, j in query["arcs"]})
            answers.append(bool(test(*query["arc"], dag)))
        found = []
        for seed in case["seeds"]:
            random.seed(seed)
            np.random.seed(seed)
            dag = gml.igsp(settings, nodes, testers, testers)
            found.append(sorted([int(i), int(j)] for i, j in dag.arcs))
        sink.write(json.dumps({"contradicting": answers, "found": found}) + "\n")
        sink.flush()


def probe_request(request, sink) -> None:
    Xs, env = request["Xs"], request["env"]
    (fit,) = json.loads(str(request["fits"]))
    known = json.loads(str(request["known"]))
    for X, seed in zip(Xs, json.loads(str(request["seeds"])), strict=True):
        random.seed(seed)
        np.random.seed(seed)
        result = run_fit(fit, X, env, known)
        sink.write(json.dumps({"seed": seed, "arcs": result["arcs"]}) + "\n")
        sink.flush()


def icpdag_request(request, sink) -> None:
    import graphical_models as gm

    for pair in json.loads(str(request["pairs"])):
        dag = gm.DAG(nodes=set(range(pair["p"])), arcs={tuple(a) for a in pair["arcs"]})
        cpdag = dag.interventional_cpdag([set(t) for t in pair["interventions"]], cpdag=dag.cpdag())
        line = {
            "arcs": sorted([int(i), int(j)] for i, j in cpdag.arcs),
            "edges": sorted(sorted(int(v) for v in e) for e in cpdag.edges),
        }
        sink.write(json.dumps(line) + "\n")
        sink.flush()


def main(argv: list[str]) -> int:
    import warnings

    warnings.simplefilter("ignore")
    request = np.load(argv[1])
    with open(argv[2], "a") as sink:
        mode = str(request["mode"])
        if mode == "icpdag":
            icpdag_request(request, sink)
        elif mode == "oracle":
            oracle_request(request, sink)
        elif mode == "probe":
            probe_request(request, sink)
        else:
            fit_request(request, sink)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
