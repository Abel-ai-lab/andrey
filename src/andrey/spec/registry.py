"""The method registry -- one :class:`MethodSpec` per public facade.

Keyed on ``andrey.api.__all__``; a drift-guard test asserts the registry and that list stay in
lockstep and that every spec's inputs / params / defaults match the live facade signature. Summaries
are bounded "when to reach for this" notes; parameter defaults are the source of truth the CLI
and JSON schema render.
"""

from __future__ import annotations

from andrey.core.ci import SHIPPED_INDEP_TESTS

from .models import DataSpec, InputSpec, MethodSpec, OutputSpec, ParamSpec

# Shared input + parameter fragments (kept identical where the facades share a contract).
_DATA_MATRIX = InputSpec(
    name="data",
    kind="matrix",
    doc="Observed data, one row per sample.",
)
_ALPHA = ParamSpec(
    name="alpha",
    type="float",
    default=0.05,
    doc="Significance level of the conditional-independence test.",
)
_INDEP_TEST = ParamSpec(
    name="indep_test",
    type="str",
    default="fisherz",
    doc="Conditional-independence test. Only Fisher-Z (Gaussian partial correlation) is supported.",
    choices=SHIPPED_INDEP_TESTS,  # single source of truth: the shipped CI-registry names
)
_COLLIDER_RULE = ParamSpec(
    name="collider_rule",
    type="str",
    default="sepsets",
    choices=("sepsets", "majority"),
    doc="How colliders are decided on x - z - y: sepsets = z is a collider when it is not in the "
    "separating set of x and y; majority = test x and y given every subset of each one's "
    "neighbors, and z is a collider when fewer than half of the separating subsets contain it "
    "(exactly half: unoriented). Majority costs up to 2**degree tests per triple.",
)
_LAMBDA_VALUE = ParamSpec(
    name="lambda_value",
    type="float",
    default=1.0,
    doc="Multiplier on the BIC penalty of log(n_samples) per parent. Larger values give sparser "
    "graphs; 1.0 is the standard BIC.",
)
_SEED = ParamSpec(
    name="seed",
    type="int",
    default=None,
    doc="Seed for the search's random orders. null uses ANDREY_SEED, or 0 when it is unset.",
    nullable=True,
)
_RANDOM_STATE = ParamSpec(
    name="random_state",
    type="int",
    default=None,
    doc="Ignored: the pwling order search is deterministic. Accepted so that every LiNGAM method "
    "takes random_state.",
    nullable=True,
)

_MATRIX_LAYOUT = "(n_samples, n_variables), row = sample"

REGISTRY: dict[str, MethodSpec] = {}


def _add(spec: MethodSpec) -> None:
    REGISTRY[spec.name] = spec


# --- constraint-based -----------------------------------------------------------------------------

_add(
    MethodSpec(
        name="pc",
        family="constraint",
        summary="General-purpose structure learning from i.i.d. observational data: "
        "conditional-independence tests remove edges, then orientation rules direct the edges the "
        "data determine; returns a CPDAG. A good first method when nothing is known about the "
        "graph.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(structure_type="graph", graph_kind="cpdag"),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        params=(_ALPHA, _INDEP_TEST),
        references=(
            "Spirtes & Glymour 1991",
            "Spirtes, Glymour & Scheines 2000",
            "Colombo & Maathuis 2014",
            "Meek 1995",
        ),
    )
)
_add(
    MethodSpec(
        name="fci",
        family="constraint",
        summary="Structure learning that allows latent confounders and selection bias; returns a "
        "PAG. A circle mark is an endpoint the data cannot determine; a bidirected edge (<->) "
        "marks a latent common cause. Use when unmeasured confounding is plausible.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(structure_type="graph", graph_kind="pag"),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        params=(_ALPHA, _INDEP_TEST, _COLLIDER_RULE),
        references=(
            "Spirtes, Meek & Richardson 1995",
            "Spirtes, Glymour & Scheines 2000",
            "Zhang 2008",
            "Colombo & Maathuis 2014",
        ),
    )
)
_add(
    MethodSpec(
        name="gfci",
        family="constraint",
        summary="Score-then-FCI hybrid: a GES-scored skeleton refined by FCI orientation. Latent-"
        "aware like FCI but usually more accurate on the skeleton. Returns a PAG.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(structure_type="graph", graph_kind="pag"),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC",
                doc="Local score for the GES phase.",
                choices=("local_score_BIC",),
            ),
            _LAMBDA_VALUE,
            _INDEP_TEST,
            _ALPHA,
            _COLLIDER_RULE,
        ),
        references=("Ogarrio, Spirtes & Ramsey 2016", "Colombo & Maathuis 2014"),
    )
)
_add(
    MethodSpec(
        name="cdnod",
        family="constraint",
        summary="Constraint-based discovery for data pooled from several domains or regimes: adds "
        "the domain index as a variable that can only be a cause, which can orient more edges. "
        "Returns a CPDAG over the data variables.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="c_indx",
                    kind="context",
                    doc="Domain / context index, (n_samples,) or (n_samples, 1), aligned to the "
                    "data rows.",
                ),
            ),
            layout=_MATRIX_LAYOUT,
            notes="c_indx becomes a trailing pseudo-node, then projected out of the result.",
        ),
        output=OutputSpec(structure_type="graph", graph_kind="cpdag"),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(_ALPHA, _INDEP_TEST),
        references=("Huang et al. 2020",),
    )
)

# --- score-based ----------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="ges",
        family="score",
        summary="Greedy equivalence search: a BIC-scored greedy search over CPDAGs. Fast and "
        "accurate on linear-Gaussian data; the BIC score is in metadata['score'].",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="deterministic",
        accel="numba",
        status="supported",
        params=(
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC",
                doc="Local score objective.",
                choices=("local_score_BIC",),
            ),
            _LAMBDA_VALUE,
        ),
        references=("Chickering 2002",),
    )
)
_add(
    MethodSpec(
        name="gies",
        family="score",
        summary="Greedy interventional equivalence search. Only the observational "
        "(no-intervention) case is exposed, where it coincides with GES; returns a CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="deterministic",
        accel="numba",
        status="experimental",
        params=(_LAMBDA_VALUE,),
        references=("Hauser & Buhlmann 2012",),
    )
)
_add(
    MethodSpec(
        name="hc",
        family="score",
        summary="Hill climbing over DAGs: adds, removes, or reverses one edge at a time while the "
        "BIC score improves; a simple greedy baseline. Stops after 200 moves, so at most 200 "
        "edges. Returns the CPDAG of the final DAG and its score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC_from_cov",
                doc="Local score objective (covariance-based BIC).",
                choices=("local_score_BIC_from_cov",),
            ),
            _LAMBDA_VALUE,
        ),
        references=("Chickering, Geiger & Heckerman 1995", "Scutari 2010"),
    )
)
_add(
    MethodSpec(
        name="exact_search",
        family="score",
        summary="Globally optimal BIC search over DAGs (A* or dynamic programming). Exact but "
        "exponential in the number of variables; keep to about 20 or fewer. Returns one DAG from "
        "the best-scoring equivalence class.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(structure_type="graph", graph_kind="dag"),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(
            ParamSpec(
                name="search_method",
                type="str",
                default="astar",
                doc="Exact search strategy: A* or dynamic programming.",
                choices=("astar", "dp"),
            ),
        ),
        references=("Yuan & Malone 2013", "Silander & Myllymaki 2006"),
    )
)
_add(
    MethodSpec(
        name="calm",
        family="score",
        summary="Continuous-optimization structure learning: an L0-penalized likelihood with an "
        "acyclicity constraint, searched within an estimated moral graph. Returns a weighted DAG. "
        "Needs the [torch] extra. Seeded on every call: seed, else ANDREY_SEED, else 0.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            weighted=True,
            doc="weighted DAG; no score. Extra optimizer knobs (lambda1, lr, ...) pass through.",
        ),
        determinism="stochastic",
        accel="torch",
        status="experimental",
        seeded=True,
        passthrough=True,
        params=(
            ParamSpec(
                name="seed",
                type="int",
                default=None,
                doc="Seed for the torch optimizer. null uses ANDREY_SEED, or 0 when it is unset; "
                "metadata['seed'] records the seed used.",
                nullable=True,
            ),
        ),
        references=("Jin, Ng, Zhang & Huang 2026",),
    )
)

# --- permutation-based ----------------------------------------------------------------------------

_BOSS_SCORE = ParamSpec(
    name="score_func",
    type="str",
    default="local_score_BIC_from_cov",
    doc="Local score objective (covariance-based BIC).",
    choices=("local_score_BIC_from_cov",),
)
_add(
    MethodSpec(
        name="boss",
        family="permutation",
        summary="Best-order score search: a permutation search over the linear-Gaussian BIC, fast "
        "and accurate on linear-Gaussian data. Returns a CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        seeded=True,
        params=(_BOSS_SCORE, _LAMBDA_VALUE, _SEED),
        references=("Andrews, Ramsey, Sanchez Romero, Camchong & Kummerfeld 2023",),
    )
)
_add(
    MethodSpec(
        name="grasp",
        family="permutation",
        summary="Greedy relaxation of the sparsest permutation: a permutation search over the "
        "linear-Gaussian BIC, like BOSS with a different move (tucks). Returns a CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        seeded=True,
        params=(
            _BOSS_SCORE,
            _LAMBDA_VALUE,
            ParamSpec(
                name="depth",
                type="int",
                default=3,
                doc="Maximum number of tucks in one chain of moves. Only 3 is supported.",
            ),
            _SEED,
        ),
        references=("Lam, Andrews & Ramsey 2022",),
    )
)

# --- LiNGAM family --------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="direct_lingam",
        family="lingam",
        summary="DirectLiNGAM for linear non-Gaussian data: recovers a fully-oriented DAG plus a "
        "causal order and edge weights. Use when noise is non-Gaussian and effects are linear.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="dag", weighted=True, has_ordering=True
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        params=(
            _RANDOM_STATE,
            ParamSpec(
                name="measure",
                type="str",
                default="pwling",
                doc="How the order search picks the most exogenous variable at each step; pwling "
                "is the pairwise likelihood-ratio measure.",
                choices=("pwling",),
            ),
        ),
        references=("Shimizu et al. 2011", "Hyvarinen & Smith 2013"),
    )
)
_add(
    MethodSpec(
        name="ica_lingam",
        family="lingam",
        summary="ICA-based LiNGAM for linear non-Gaussian data: FastICA gives a causal order, "
        "adaptive Lasso the edge weights; returns a DAG, the order, and the weights. Seeded "
        "(ANDREY_SEED, else 0), so repeated runs match. DirectLiNGAM is more accurate on the "
        "published benchmarks.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="dag", weighted=True, has_ordering=True
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        seeded=True,
        params=(
            ParamSpec(
                name="random_state",
                type="int",
                default=None,
                doc="Seed for FastICA, from 0 to 2**32 - 1. null uses ANDREY_SEED, or 0 when it is "
                "unset.",
                nullable=True,
            ),
            ParamSpec(
                name="max_iter",
                type="int",
                default=1000,
                doc="Maximum FastICA iterations; at least 1.",
            ),
        ),
        references=("Shimizu et al. 2006",),
    )
)
_add(
    MethodSpec(
        name="multi_group_direct_lingam",
        family="lingam",
        summary="DirectLiNGAM fit jointly across >= 2 datasets sharing the same variables: one "
        "shared causal order and a per-group weighted DAG. Returns one result per group.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data_groups",
                    kind="panel",
                    doc="Two or more datasets over the same d variables (repeat the flag).",
                ),
            ),
            layout="each group (n_samples, n_variables); all groups share n_variables",
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            weighted=True,
            has_ordering=True,
            multi=True,
            doc="one StructureOutput per group; shared ordering, per-group weights.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(_RANDOM_STATE,),
        references=("Shimizu 2012",),
    )
)

# --- temporal family ------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="varma_lingam",
        family="temporal",
        summary="VARMA-LiNGAM for a single multivariate time series: instantaneous + lagged causal "
        "effects (autoregressive and moving-average). Returns a temporal structure.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data", kind="matrix", doc="A multivariate time series, row = step."
                ),
            ),
            layout="(n_samples, n_variables), row = time step",
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            has_ordering=True,
            doc="lag_weights (AR, lag 0 = instantaneous) + lag_weights_ma (MA) + order.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(
            ParamSpec(
                name="order",
                type="int_pair",
                default=(1, 1),
                doc="(p, q): autoregressive and moving-average lag counts.",
            ),
            ParamSpec(
                name="criterion",
                type="str",
                default=None,
                doc="Order-selection criterion: pick the best (p, q) up to order by this "
                "criterion; null uses order as given.",
                choices=("aic", "bic", "hqic"),
                nullable=True,
            ),
            ParamSpec(
                name="prune",
                type="bool",
                default=False,
                doc="Re-estimate the lagged weights by adaptive Lasso on the causal order, which "
                "sets small weights to zero. false keeps every lagged weight, so every lag graph "
                "links every pair of variables.",
            ),
            ParamSpec(
                name="structural_ma",
                type="bool",
                default=False,
                doc="Map the moving-average weights onto the independent noise terms. Ignored "
                "when prune is true.",
            ),
        ),
        references=("Hyvarinen et al. 2010", "Kawahara, Shimizu & Washio 2011"),
    )
)
_add(
    MethodSpec(
        name="longitudinal_lingam",
        family="temporal",
        summary="Longitudinal LiNGAM for panel data (the same units measured at several time "
        "points): instantaneous + lagged effects. Returns a temporal structure.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data_list",
                    kind="panel",
                    doc="One dataset per time point, same units and shape (repeat the flag).",
                ),
            ),
            layout="each (n_samples, n_variables); one array per time point",
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            doc="instantaneous-plus-lagged lag_weights; per-time causal orders in metadata.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(
            ParamSpec(
                name="n_lags",
                type="int",
                default=1,
                doc="Number of past time points regressed out before the instantaneous fit.",
            ),
            ParamSpec(
                name="measure",
                type="str",
                default="pwling",
                doc="How the instantaneous order search picks the most exogenous variable; "
                "pwling is the pairwise likelihood-ratio measure.",
                choices=("pwling",),
            ),
        ),
        references=("Kadowaki, Shimizu & Washio 2013",),
    )
)

# --- pairwise family ------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="pnl",
        family="pairwise",
        summary="Post-nonlinear pairwise direction test: decides whether column 0 causes "
        "column 1, the reverse, or neither, from a two-column table.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data",
                    kind="matrix",
                    doc="The two variables, one per column: column 0 is x, column 1 is y.",
                ),
            ),
            layout="(n_samples, 2), rows are samples",
        ),
        params=(
            ParamSpec(
                name="alpha",
                type="float",
                default=None,
                doc="How the two p-values decide the edge. null: the direction with the larger "
                "p-value. A level such as 0.05: an edge only when exactly one direction's p-value "
                "is above it.",
                nullable=True,
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="two-node graph with the decided edge, or none; metadata pval_forward (x->y) "
            "and pval_backward (y->x).",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        references=(
            "Zhang & Hyvarinen 2009",
            "Zhang & Hyvarinen 2010",
            "Hoyer et al. 2008",
            "Mooij et al. 2016",
        ),
    )
)

# --- latent-variable family -----------------------------------------------------------------------

_add(
    MethodSpec(
        name="gin",
        family="latent",
        summary="Generalized independent noise (GIN) for linear non-Gaussian models with latent "
        "variables: recovers latents and their causal order from observed indicators alone. The "
        "graph spans observed + latent nodes; node_types flags the latents.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            has_latents=True,
            doc="observed + latent nodes; metadata clusters + causal_order.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        params=(
            ParamSpec(
                name="alpha",
                type="float",
                default=0.05,
                doc="Significance level of the independence test that finds the causal clusters.",
            ),
            ParamSpec(
                name="labels",
                type="str_list",
                default=None,
                doc="Names for the observed variables, for example, labels=[x1,x2]. null uses the "
                "data file's header, else X1..Xn; latents are named L1..Lm.",
                nullable=True,
            ),
        ),
        references=("Xie et al. 2020",),
    )
)


def get_spec(name: str) -> MethodSpec:
    """The spec for one method name, or a ``KeyError`` with the valid names."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; valid: {', '.join(sorted(REGISTRY))}") from None


def list_specs() -> list[MethodSpec]:
    """All specs in registration (family-grouped) order."""
    return list(REGISTRY.values())
