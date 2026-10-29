"""Method definitions used to generate CLI help and JSON schemas.

Each public function in ``andrey.api.__all__`` has one :class:`MethodSpec` describing its inputs,
parameters, defaults, and results. Tests check that these definitions cover the public functions
and match their signatures. Summaries explain when to use each method; the CLI and JSON schemas
display the parameter defaults defined here.
"""

from __future__ import annotations

from andrey.constraint.pc import COLLIDER_ORDERS
from andrey.core.knowledge import KNOWLEDGE_PARTS, KnowledgePart
from andrey.core.score import LINEAR_SCORES
from andrey.search.hybrid import MAXIMIZES, RESTRICTS

from .concepts import CONCEPTS, SCORE_ALIASES, concept_param
from .models import DataSpec, InputSpec, MethodSpec, OutputSpec, ParamSpec

# Shared inputs and parameters; a parameter that implements a concept comes from its template.
_DATA_MATRIX = InputSpec(
    name="data",
    kind="matrix",
    doc="Observed data, one row per sample.",
)
_ALPHA = concept_param("alpha")
#: How a null test or score picks from the declared column types.
_DECLARED = (
    "null picks from the declared column types: {every} when every column is categorical, "
    "cond_gauss when some are, else {fallback}. A column is declared categorical by categorical "
    "or a DataFrame category, bool, or text type, never by its values."
)
_INDEP_TEST = concept_param(
    "indep_test",
    doc="Conditional-independence test. For continuous data, on the partial correlation: "
    "fisherz = Fisher's z-transform and the normal distribution; student_t = the Student-t "
    "distribution, exact for Gaussian data at any sample size. For categorical data coded by whole "
    "numbers: chisq = Pearson's chi-square; gsq = G-square. For mixed data, with the categorical "
    "columns named by categorical: degen_gauss = degenerate Gaussian and "
    "cond_gauss = conditional Gaussian likelihood ratio. For nonlinear effects in continuous "
    "data, each far slower per test: kci = kernel test; rcot = its random-feature approximation; "
    "cmiknn = nearest-neighbor conditional mutual information; "
    "gpdc = Gaussian-process regression and distance correlation. These four rank the columns as "
    "the searches do, so ANDREY_TIE_BREAK=position makes their p-values depend on the column "
    "order. " + _DECLARED.format(every="gsq", fallback="fisherz"),
)
_CATEGORICAL = concept_param("categorical")
_MAX_CATEGORIES = concept_param("max_categories")
_COLLIDER_RULE = concept_param("collider_rule")
_POSSIBLE_DSEP = ParamSpec(
    name="possible_dsep",
    type="str",
    default=None,
    choices=("order_free", "fast"),
    doc="The order of the possible-d-separation step, which removes edges one at a time: "
    "order_free = by a rank computed from each column's own data, so the result does not depend "
    "on the column order unless two columns are identical; fast = column order. null follows "
    "ANDREY_TIE_BREAK: order_free under rank, fast under position.",
    nullable=True,
)
_COLLIDER_ORDER = ParamSpec(
    name="collider_order",
    type="str",
    default=None,
    choices=COLLIDER_ORDERS,
    doc="The order in which colliders are oriented, which decides between two that would orient "
    "one edge in opposite directions: p_value = the collider whose endpoints were separated with "
    "the largest p-value first, independent of the column order; column = in column order, the "
    "first of two conflicting colliders kept. null follows ANDREY_TIE_BREAK: p_value under rank, "
    "column under position.",
    nullable=True,
)
_PENALTY_DISCOUNT = concept_param("penalty_discount")
#: The same, for the methods that shipped it as lambda_value.
_PENALTY_DISCOUNT_RENAMED = concept_param("penalty_discount", aliases=("lambda_value",))
_SCORE_DOC = (
    "Local score: bic = linear-Gaussian BIC; bdeu = BDeu and discrete_bic = multinomial BIC, "
    "for categorical data coded by whole numbers; degen_gauss and cond_gauss = degenerate and "
    "conditional Gaussian BIC, for mixed data. " + _DECLARED.format(every="bdeu", fallback="bic")
)
_SCORE_FUNC = concept_param("score_func", doc=_SCORE_DOC)
#: The same, for the methods whose score names shipped as local_score_*.
_SCORE_FUNC_RENAMED = concept_param("score_func", doc=_SCORE_DOC, choice_aliases=SCORE_ALIASES)
_EBIC_GAMMA = concept_param("ebic_gamma")
_EBIC_GAMMA_STANDARD = ParamSpec(
    name="ebic_gamma",
    type="float",
    default=0.0,
    doc="Weight on the extended-BIC term 2 * ebic_gamma * log C(n_variables - 1, n_parents) that "
    "the linear-Gaussian BIC adds per variable; it grows with the number of variables. Larger "
    "values give sparser graphs; 0.0 is the standard BIC. Other scores ignore it.",
    minimum=0,
    concept="ebic_gamma",
)
_SEED = concept_param(
    "seed",
    doc="Seed for the search's random orders. null uses ANDREY_SEED, or 0 when it is unset.",
)
_TEST_SEED = concept_param(
    "seed",
    doc="Seed for the random draws of indep_test, which only rcot, cmiknn, and gpdc make; the "
    "other tests ignore it. null uses ANDREY_SEED, or 0 when it is unset.",
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
        "graph; indep_test=chisq or gsq for categorical data, degen_gauss or cond_gauss for mixed "
        "data.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT, missing=True),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="metadata['categorical'] lists the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position",
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires Meek\'s rules one at a time in column order, and it or '
            '`collider_order="column"` orients colliders in column order.'
        ),
        seeded=True,
        params=(
            _ALPHA,
            _INDEP_TEST,
            _COLLIDER_ORDER,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _TEST_SEED,
        ),
        references=(
            "Spirtes & Glymour 1991",
            "Spirtes, Glymour & Scheines 2000",
            "Colombo & Maathuis 2014",
            "Meek 1995",
            "Ramsey, Zhang & Spirtes 2006",
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
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT, missing=True),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            doc="metadata['categorical'] lists the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position",
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires the orientation rules one at a time in column order, and '
            'it or `possible_dsep="fast"` visits the possible-d-separation step in column order.'
        ),
        seeded=True,
        params=(
            _ALPHA,
            _INDEP_TEST,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _POSSIBLE_DSEP,
            _TEST_SEED,
        ),
        references=(
            "Spirtes, Meek & Richardson 1995",
            "Spirtes, Glymour & Scheines 2000",
            "Zhang 2008",
            "Colombo & Maathuis 2014",
            "Ramsey, Zhang & Spirtes 2006",
        ),
    )
)
_add(
    MethodSpec(
        name="gfci",
        family="constraint",
        summary="Score-then-FCI hybrid: GES adjacencies, pruned by independence tests and oriented "
        "with FCI's rules. Latent-aware like FCI but usually more accurate on the skeleton. "
        "Returns a PAG.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            doc="metadata['categorical'] lists the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position",
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires the orientation rules one at a time in column order and '
            'records the first separating subset found, and it or `possible_dsep="fast"` visits '
            "the possible-d-separation step in column order."
        ),
        seeded=True,
        params=(
            _SCORE_FUNC_RENAMED,
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA_STANDARD,
            _INDEP_TEST,
            _ALPHA,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _POSSIBLE_DSEP,
            _TEST_SEED,
        ),
        references=(
            "Ogarrio, Spirtes & Ramsey 2016",
            "Zhang 2008",
            "Colombo & Maathuis 2014",
            "Spirtes, Glymour & Scheines 2000",
            "Ramsey, Zhang & Spirtes 2006",
        ),
    )
)
_add(
    MethodSpec(
        name="cdnod",
        family="constraint",
        summary="Constraint-based discovery for data pooled from several domains or regimes: adds "
        "the domain index as a variable that can only be a cause, which can orient more edges. "
        "Returns a partially directed graph over the data variables, tagged cpdag.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="domain_index",
                    kind="context",
                    doc="Domain / context index, (n_samples,) or (n_samples, 1), aligned to the "
                    "data rows.",
                ),
            ),
            layout=_MATRIX_LAYOUT,
            notes="domain_index becomes a trailing pseudo-node, then projected out of the result.",
            missing=True,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="metadata['categorical'] lists the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position",
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires Meek\'s rules one at a time in column order, and it or '
            '`collider_order="column"` orients colliders in column order.'
        ),
        seeded=True,
        params=(
            _ALPHA,
            _INDEP_TEST,
            _COLLIDER_ORDER,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            ParamSpec(
                name="dependence_threshold",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                type="float",
                default=None,
                doc="Runs the independent-change step, which orients edges between variables that "
                "both change with the index, at this threshold on its module dependence measure. "
                "null skips the step.",
                nullable=True,
            ),
            _TEST_SEED,
        ),
        references=("Huang et al. 2020", "Ramsey, Zhang & Spirtes 2006", "Colombo & Maathuis 2014"),
    )
)

# --- local structure: one variable's Markov blanket or parents and children ----------------------

_TARGET = ParamSpec(
    name="target",
    type="str",
    default=None,
    doc="The variable whose set to learn: a column name from the data file's header or a 0-based "
    "position. null learns every variable's set and keeps a pair when each is in the other's set.",
    nullable=True,
)
_LOCAL_OUTPUT_DOC = (
    "undirected edges only, from target to each member, or between each pair in each other's "
    "set when target is null; metadata['target'] names the target and metadata['{key}'] lists the "
    "members in the order they joined"
)
for _name, _summary, _key, _refs in (
    (
        "iamb",
        "One variable's Markov blanket (its parents, children, and children's other parents) by "
        "incremental association: grow by conditional-independence tests, then shrink. With "
        "target=null, the moral graph of every variable's blanket.",
        "markov_blanket",
        ("Tsamardinos, Aliferis & Statnikov 2003",),
    ),
    (
        "inter_iamb",
        "One variable's Markov blanket by interleaved IAMB, which shrinks after every addition. "
        "With target=null, the moral graph of every variable's blanket.",
        "markov_blanket",
        ("Tsamardinos, Aliferis & Statnikov 2003",),
    ),
    (
        "mmpc",
        "One variable's parents and children, undirected, by max-min parents and children: "
        "conditional-independence tests over subsets of the members so far, with a symmetry "
        "check. With target=null, the skeleton that mmhc searches.",
        "parents_and_children",
        ("Tsamardinos, Brown & Aliferis 2006",),
    ),
):
    _add(
        MethodSpec(
            name=_name,
            family="constraint",
            summary=_summary,
            data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT, missing=True),
            output=OutputSpec(
                structure_type="graph",
                graph_kind="undirected",
                doc=_LOCAL_OUTPUT_DOC.format(key=_key),
            ),
            determinism="stochastic",
            accel="numpy",
            status="experimental",
            column_order="relabels",
            tie_break=True,
            seeded=True,
            params=(_TARGET, _ALPHA, _INDEP_TEST, _CATEGORICAL, _MAX_CATEGORIES, _TEST_SEED),
            references=_refs,
        )
    )

# --- score-based ----------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="ges",
        family="score",
        summary="Greedy equivalence search: a score-based greedy search over CPDAGs. Fast and "
        "accurate on linear-Gaussian data with the BIC; bdeu, the default when every "
        "column is categorical, or discrete_bic for categorical data, "
        "cond_gauss, the default when some columns are, or degen_gauss for "
        "mixed data. The score is in metadata['score'].",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC_RENAMED,
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
            concept_param(
                "max_parents",
                doc="A cap on every node's parents, at least 0: no DAG the returned CPDAG "
                "represents gives a node more than max_parents parents, and some DAG reaches each "
                "node's count. A move whose CPDAG would "
                "break the cap, in either phase, is skipped for the next best move. A cap can "
                "stop the search short of the lowest-scoring graph, which GES otherwise finds. "
                "null sets no cap; the forward phase then keeps its gate of n_variables / 2.",
            ),
            _CATEGORICAL,
            _MAX_CATEGORIES,
        ),
        references=("Chickering 2002",),
    )
)
_add(
    MethodSpec(
        name="gies",
        family="score",
        summary="Greedy interventional equivalence search: GES for data that mixes observation "
        "with interventions, scoring each variable on the rows where it was not intervened on. "
        "Returns the interventional essential graph + score.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="interventions",
                    kind="context",
                    doc="(n_samples, n_variables) 0/1 matrix aligned to the data: 1 where the "
                    "row's value of the variable was set by an intervention. Omit it for "
                    "observational data.",
                    required=False,
                ),
            ),
            layout=_MATRIX_LAYOUT,
        ),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "score_func",
                doc="Local score: bic = linear-Gaussian BIC, the only score GIES supports; null "
                "picks it.",
                choices=LINEAR_SCORES,
            ),
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
        ),
        references=("Hauser & Buhlmann 2012",),
    )
)
_add(
    MethodSpec(
        name="xges",
        family="score",
        summary="Extremely greedy equivalence search: GES's moves and BIC score, deleting an edge "
        "before reversing one and reversing before adding, then retrying each deletion. Returns a "
        "CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="cpdag", doc="score in metadata['score']"
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "score_func",
                doc="Local score: bic = linear-Gaussian BIC, the only score XGES supports; null "
                "picks it.",
                choices=LINEAR_SCORES,
                choice_aliases=SCORE_ALIASES[:2],
            ),
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
            ParamSpec(
                name="extended_search",
                type="bool",
                default=True,
                doc="Run the extended search: after the greedy search converges, delete each edge "
                "in turn, forbid adding it back, search again, and keep a better result. "
                "false returns the converged graph.",
            ),
        ),
        references=("Nazaret & Blei 2024",),
    )
)
_add(
    MethodSpec(
        name="hc",
        family="score",
        summary="Hill climbing over DAGs: adds, removes, or reverses one edge at a time while the "
        "score improves; a simple greedy baseline. BIC by default, BDeu when every column is "
        "categorical, conditional Gaussian when some are; score_func=bdeu or "
        "discrete_bic for categorical data, degen_gauss or "
        "cond_gauss for mixed data. Stops at a local optimum, or after max_iter moves "
        "when set. Returns the CPDAG of the final DAG and its score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC_RENAMED,
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
            concept_param(
                "max_iter",
                default=None,
                doc="Most moves the search takes, at least 1; null sets no limit (the search still "
                "ends). Reaching a set limit warns with SearchLimitWarning.",
                nullable=True,
            ),
            _CATEGORICAL,
            _MAX_CATEGORIES,
        ),
        references=("Chickering, Geiger & Heckerman 1995", "Scutari 2010"),
    )
)
_add(
    MethodSpec(
        name="tabu",
        family="score",
        summary="Tabu search over DAGs: hill climbing that, at a local optimum, takes the best "
        "non-improving move, never returning to a recently visited DAG, and keeps the best DAG "
        "seen. With the same settings its score is never worse than hc's. BIC by default, BDeu "
        "when every column is categorical, conditional Gaussian when some are; score_func=bdeu or "
        "discrete_bic for categorical data, degen_gauss or cond_gauss for mixed data. Returns "
        "the CPDAG of that DAG and its score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "score_func",
                doc=_SCORE_DOC,
                # Before 0.11.0 tabu took the linear and the categorical scores, not the mixed ones.
                choice_aliases=SCORE_ALIASES[:4],
            ),
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
            ParamSpec(
                name="tabu_length",
                minimum=1,
                type="int",
                default=10,
                doc="Length of the tabu list: how many recently visited DAGs a move may not lead "
                "to, at least 1.",
            ),
            ParamSpec(
                name="max_tabu",
                minimum=1,
                type="int",
                default=None,
                doc="Non-improving moves allowed without a better DAG before the search stops, "
                "from 1 to tabu_length. null uses tabu_length.",
                nullable=True,
            ),
            concept_param(
                "max_iter",
                default=None,
                doc="Most moves the search takes, improving or not, at least 1; null sets no limit "
                "(the search still ends). Reaching a set limit warns with SearchLimitWarning.",
                nullable=True,
            ),
            _CATEGORICAL,
            _MAX_CATEGORIES,
        ),
        references=("Glover 1986", "Russell & Norvig 2009", "Scutari 2010"),
    )
)
_add(
    MethodSpec(
        name="exact_search",
        family="score",
        summary="Globally optimal score search over DAGs (A* or dynamic programming). Exact but "
        "exponential in the number of variables; keep to about 20 or fewer. Returns the CPDAG of "
        "the best-scoring equivalence class. BIC by default, BDeu when every column is "
        "categorical, conditional Gaussian when some are; score_func=bdeu or "
        "discrete_bic for categorical data, degen_gauss or "
        "cond_gauss for mixed data.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="metadata['dag'] lists the edges [i, j] of the optimal DAG the search found; "
            "metadata['categorical'] lists the categorical columns a mixed-data setting read",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            ParamSpec(
                name="search_method",
                type="str",
                default="astar",
                doc="Exact search strategy: A* or dynamic programming.",
                choices=("astar", "dp"),
            ),
            _SCORE_FUNC_RENAMED,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _CATEGORICAL,
            _MAX_CATEGORIES,
        ),
        references=("Yuan & Malone 2013", "Silander & Myllymaki 2006"),
    )
)
_add(
    MethodSpec(
        name="calm",
        family="score",
        summary="Continuous-optimization structure learning in the line of NOTEARS: an "
        "L0-penalized likelihood with an acyclicity constraint, searched within an estimated moral "
        "graph. Returns a weighted DAG. Needs the [torch] extra. Seeded on every call: seed, else "
        "ANDREY_SEED, else 0.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            weighted=True,
            doc="weighted DAG; no score.",
        ),
        determinism="stochastic",
        accel="torch",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            "Repeats on one machine with one torch version; on another platform the graph "
            "can differ."
        ),
        seeded=True,
        dropped_args=("labels",),
        params=(
            concept_param(
                "lambda1",
                default=0.005,
                doc="Weight of the l1 penalty on the sum of the edge gates, which approximate a "
                "count of edges, at least 0. Larger values give sparser graphs.",
            ),
            ParamSpec(
                name="screen_alpha",
                type="float",
                default=0.01,
                doc="Significance level of the Fisher-Z tests that estimate the moral graph. Only "
                "pairs the screen keeps can become edges; smaller values keep fewer.",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                aliases=("alpha",),
            ),
            ParamSpec(
                name="use_moral_graph",
                type="bool",
                default=True,
                doc="false makes every pair a candidate edge.",
            ),
            ParamSpec(
                name="standardize",
                type="bool",
                default=False,
                doc="Scale each column to unit variance first.",
            ),
            concept_param(
                "w_threshold",
                doc="An edge is kept when its weight magnitude reaches this value, at least 0.",
            ),
            ParamSpec(
                name="gate_threshold",
                type="float",
                default=0.5,
                doc="An edge is kept when its gate probability reaches this value.",
                minimum=0,
                maximum=1,
            ),
            concept_param(
                "max_iter",
                default=25,
                doc="Most quadratic-penalty rounds, at least 1. Stopping on it before the "
                "acyclicity residual clears h_tol warns; raise it to let the fit run on.",
                nullable=False,
                aliases=("max_outer",),
            ),
            ParamSpec(
                name="subproblem_iter",
                type="int",
                default=3000,
                doc="Most Adam steps per round, at least 1.",
                minimum=1,
            ),
            ParamSpec(
                name="lr",
                type="float",
                default=0.01,
                doc="Adam's learning rate, above 0.",
                minimum=0,
                exclusive_minimum=True,
            ),
            ParamSpec(
                name="tau",
                type="float",
                default=0.5,
                doc="The gate's Gumbel-sigmoid temperature, above 0.",
                minimum=0,
                exclusive_minimum=True,
            ),
            ParamSpec(
                name="rho_init",
                type="float",
                default=0.001,
                doc="The acyclicity penalty of the first round, above 0.",
                minimum=0,
                exclusive_minimum=True,
            ),
            ParamSpec(
                name="rho_mult",
                type="float",
                default=3.0,
                doc="The factor by which the acyclicity penalty grows between rounds, at least 1.",
                minimum=1,
            ),
            ParamSpec(
                name="rho_max",
                type="float",
                default=1e16,
                doc="The rounds end when the acyclicity penalty reaches this value, above 0.",
                minimum=0,
                exclusive_minimum=True,
            ),
            ParamSpec(
                name="h_tol",
                type="float",
                default=1e-8,
                doc="The rounds end when the acyclicity residual of the noise-free gates is at "
                "most this value, at least 0.",
                minimum=0,
            ),
            concept_param(
                "seed",
                doc="Seed for the torch optimizer. null uses ANDREY_SEED, or 0 when it is unset; "
                "metadata['seed'] records the seed used.",
            ),
        ),
        references=("Jin, Ng, Zhang & Huang 2026",),
    )
)

# --- continuous optimization of a weighted DAG: NOTEARS and DAGMA ------------------------------

_CONTINUOUS_PATH_PARAMS = (
    ParamSpec(
        name="robust",
        type="bool",
        default=False,
        doc="true follows the first fit with the likelihood path of Deng et al. 2024: the Gaussian "
        "likelihood with one noise variance per variable and a quasi-MCP penalty, refit while the "
        "likelihood falls and the penalty shrinks.",
    ),
    ParamSpec(
        name="mcp_lambda",
        minimum=0,
        exclusive_minimum=True,
        type="float",
        default=0.4,
        doc="Penalty level of the path's first step, above 0. Used only with robust.",
    ),
    ParamSpec(
        name="mcp_delta",
        minimum=0,
        exclusive_minimum=True,
        type="float",
        default=0.2,
        doc="Weight magnitude above which the quasi-MCP stops growing, at the path's first step, "
        "above 0. Used only with robust.",
    ),
    ParamSpec(
        name="mcp_decay",
        minimum=0,
        maximum=1,
        exclusive_minimum=True,
        exclusive_maximum=True,
        type="float",
        default=0.8,
        doc="Factor in (0, 1) by which mcp_lambda and mcp_delta shrink after each accepted step. "
        "Used only with robust.",
    ),
    ParamSpec(
        name="max_path",
        minimum=1,
        type="int",
        default=50,
        doc="Most accepted steps of the path, at least 1. Used only with robust.",
    ),
)
_CONTINUOUS_OUTPUT = OutputSpec(
    structure_type="graph",
    graph_kind="dag",
    weighted=True,
    doc="weighted DAG; no score. metadata['h_final'] is the acyclicity function before the "
    "threshold and metadata['path_steps'] the accepted steps of the robust path.",
)
_add(
    MethodSpec(
        name="notears",
        family="score",
        summary="NOTEARS: least squares with an l1 penalty under the acyclicity equation "
        "tr(exp(W*W)) = d, by L-BFGS-B. Returns a weighted DAG. robust=true adds the likelihood "
        "path of Deng et al. 2024, which does not depend on the scale of the columns. "
        "Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=_CONTINUOUS_OUTPUT,
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "lambda1",
                default=0.1,
                doc="Weight of the l1 penalty of the first fit, at least 0.",
            ),
            concept_param(
                "w_threshold",
                doc="An edge is kept when its weight magnitude reaches this value, at least 0.",
            ),
            concept_param(
                "max_iter",
                default=100,
                doc="Most dual steps of the augmented Lagrangian, at least 1.",
                nullable=False,
            ),
            ParamSpec(
                name="h_tol",
                minimum=0,
                exclusive_minimum=True,
                type="float",
                default=1e-8,
                doc="The fit stops when the acyclicity function falls to this value, above 0.",
            ),
            ParamSpec(
                name="rho_max",
                minimum=0,
                exclusive_minimum=True,
                type="float",
                default=1e16,
                doc="The fit stops when the penalty weight on the acyclicity function reaches "
                "this value, above 0.",
            ),
            *_CONTINUOUS_PATH_PARAMS,
        ),
        references=(
            "Zheng, Aragam, Ravikumar & Xing 2018",
            "Deng, Bello, Ravikumar & Aragam 2024",
            "Kaiser & Sipos 2022",
            "Reisach, Seiler & Weichwald 2021",
            "Ng, Huang & Zhang 2024",
        ),
    )
)
_add(
    MethodSpec(
        name="dagma",
        family="score",
        summary="DAGMA: least squares with an l1 penalty and a log-determinant acyclicity barrier "
        "lowered along a central path, by Adam. Returns a weighted DAG. robust=true adds the "
        "likelihood path of Deng et al. 2024, which does not depend on the scale of the columns. "
        "Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=_CONTINUOUS_OUTPUT,
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "lambda1",
                default=0.03,
                doc="Weight of the l1 penalty of the first fit, at least 0.",
            ),
            concept_param(
                "w_threshold",
                doc="An edge is kept when its weight magnitude reaches this value, at least 0.",
            ),
            ParamSpec(
                name="stages",
                minimum=1,
                type="int",
                default=5,
                doc="Stages of the central path, at least 1 (T in the paper).",
            ),
            ParamSpec(
                name="warm_iter",
                minimum=1,
                type="int",
                default=30000,
                doc="Most Adam steps of each stage but the last, at least 1.",
            ),
            concept_param(
                "max_iter",
                default=60000,
                doc="Most Adam steps of the last stage, at least 1.",
                nullable=False,
            ),
            ParamSpec(
                name="lr",
                type="float",
                default=3e-4,
                doc="Step size of Adam, above 0.",
                minimum=0,
                exclusive_minimum=True,
            ),
            *_CONTINUOUS_PATH_PARAMS,
        ),
        references=(
            "Bello, Aragam & Ravikumar 2022",
            "Deng, Bello, Ravikumar & Aragam 2024",
            "Kaiser & Sipos 2022",
            "Reisach, Seiler & Weichwald 2021",
            "Ng, Huang & Zhang 2024",
        ),
    )
)

# --- hybrid: a restrict phase of CI tests, then a score search among its pairs ------------------

_HYBRID_MAX_ITER = concept_param(
    "max_iter",
    default=None,
    doc="Most moves the score search takes, at least 1; null sets no limit (the search still "
    "ends). Reaching a set limit warns with SearchLimitWarning.",
    nullable=True,
)
_add(
    MethodSpec(
        name="mmhc",
        family="score",
        summary="Max-min hill climbing: MMPC's conditional-independence tests learn a skeleton, "
        "then hill climbing searches DAGs whose edges join only its pairs. Faster than hc on "
        "sparse graphs with many variables. BIC with Fisher-z tests by default; "
        "indep_test=chisq with score_func=bdeu for categorical data. Returns the "
        "CPDAG of the final DAG and its score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']",
        ),
        determinism="stochastic",
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
        seeded=True,
        params=(
            _ALPHA,
            _INDEP_TEST,
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _HYBRID_MAX_ITER,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _TEST_SEED,
        ),
        references=("Tsamardinos, Brown & Aliferis 2006",),
    )
)
_add(
    MethodSpec(
        name="rsmax2",
        family="score",
        summary="Restrict-maximize search: conditional-independence tests keep the candidate pairs "
        "(restrict=mmpc, iamb, inter_iamb, or pc), then a score search (maximize=hc or tabu) "
        "searches DAGs whose edges join only those pairs. mmhc is restrict=mmpc with maximize=hc. "
        "Returns the CPDAG of the final DAG and its score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']",
        ),
        determinism="stochastic",
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
        seeded=True,
        params=(
            ParamSpec(
                name="restrict",
                type="str",
                default="mmpc",
                doc="The restrict phase: mmpc = each variable in the other's MMPC parents and "
                "children; iamb and inter_iamb = each in the other's Markov blanket (the moral "
                "graph); pc = PC's skeleton.",
                choices=RESTRICTS,
            ),
            ParamSpec(
                name="maximize",
                type="str",
                default="hc",
                doc="The maximize phase: hc = hill climbing; tabu = tabu search.",
                choices=MAXIMIZES,
            ),
            _ALPHA,
            _INDEP_TEST,
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _HYBRID_MAX_ITER,
            ParamSpec(
                name="tabu_length",
                minimum=1,
                type="int",
                default=10,
                doc="With maximize=tabu, the number of most recently visited DAGs a move may not "
                "return to, at least 1.",
            ),
            ParamSpec(
                name="max_tabu",
                minimum=1,
                type="int",
                default=None,
                doc="With maximize=tabu, the non-improving moves allowed without a better DAG, "
                "from 1 to tabu_length; null uses tabu_length.",
                nullable=True,
            ),
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _TEST_SEED,
        ),
        references=(
            "Friedman, Nachman & Pe'er 1999",
            "Tsamardinos, Brown & Aliferis 2006",
            "Tsamardinos, Aliferis & Statnikov 2003",
            "Scutari 2010",
        ),
    )
)

# --- permutation-based ----------------------------------------------------------------------------

_add(
    MethodSpec(
        name="boss",
        family="permutation",
        summary="Best-order score search: a permutation search over the linear-Gaussian BIC, fast "
        "and accurate on linear-Gaussian data; bdeu, the default when every column is "
        "categorical, or discrete_bic for categorical data, cond_gauss, "
        "the default when some columns are, or degen_gauss for mixed data. Returns a "
        "CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        seeded=True,
        params=(
            _SCORE_FUNC_RENAMED,
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
            _SEED,
            _CATEGORICAL,
            _MAX_CATEGORIES,
        ),
        references=("Andrews, Ramsey, Sanchez Romero, Camchong & Kummerfeld 2023",),
    )
)
_add(
    MethodSpec(
        name="grasp",
        family="permutation",
        summary="Greedy relaxation of the sparsest permutation: a permutation search over the "
        "linear-Gaussian BIC, like BOSS with a different move (tucks); bdeu, the "
        "default when every column is categorical, or discrete_bic for categorical "
        "data, cond_gauss, the default when some columns are, or "
        "degen_gauss for mixed data. Returns a CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        determinism="stochastic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        seeded=True,
        params=(
            _SCORE_FUNC_RENAMED,
            _PENALTY_DISCOUNT_RENAMED,
            _EBIC_GAMMA,
            ParamSpec(
                name="depth",
                minimum=1,
                type="int",
                default=3,
                doc="Maximum number of tucks in one chain of moves, at least 1; larger depths "
                "search more chains and run longer.",
            ),
            _SEED,
            _CATEGORICAL,
            _MAX_CATEGORIES,
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
        column_order="relabels",
        tie_break=True,
        dropped_args=("random_state",),
        params=(
            concept_param(
                "measure",
                doc="How the order search picks the most exogenous variable at each step; pwling "
                "is the pairwise likelihood-ratio measure.",
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
        column_order="relabels_if_converged",
        tie_break=True,
        order_note=(
            "At few rows per variable, FastICA can reach another solution under a new "
            "column order or on another machine."
        ),
        seeded=True,
        params=(
            concept_param(
                "seed",
                doc="Seed for FastICA's starting point. null uses ANDREY_SEED, or 0 when it is "
                "unset.",
                aliases=("random_state",),
            ),
            concept_param(
                "max_iter",
                default=1000,
                doc="Maximum FastICA iterations; at least 1.",
                nullable=False,
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
                    name="data",
                    kind="panel",
                    doc="Two or more datasets over the same d variables (repeat the flag).",
                    aliases=("data_groups",),
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
        status="supported",
        column_order="relabels",
        tie_break=True,
        dropped_args=("random_state",),
        params=(
            concept_param(
                "measure",
                doc="How the order search picks the most exogenous variable at each step; pwling "
                "is the pairwise likelihood-ratio measure.",
            ),
        ),
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
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            "Where the likelihood fit stops depends on the BLAS kernels the CPU runs, so another "
            "type of CPU can give another graph, and so can other library versions when the fit "
            "stops before it converges."
        ),
        params=(
            ParamSpec(
                name="order",
                minimum=0,
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
            concept_param(
                "max_iter",
                default=100,
                doc="Most iterations of the likelihood fit, at least 1. A fit stopped before it "
                "converges gives no warning; raise this to let it converge.",
                nullable=False,
            ),
            concept_param(
                "measure",
                doc="How the order search on the innovations picks the most exogenous variable; "
                "pwling is the pairwise likelihood-ratio measure.",
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
                    name="data",
                    kind="panel",
                    doc="One dataset per time point, same units and shape (repeat the flag).",
                    aliases=("data_list",),
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
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag: the number of past time points regressed out before the "
                "instantaneous fit, at least 1.",
                aliases=("n_lags",),
            ),
            concept_param(
                "measure",
                doc="How the instantaneous order search picks the most exogenous variable; "
                "pwling is the pairwise likelihood-ratio measure.",
            ),
        ),
        references=("Kadowaki, Shimizu & Washio 2013",),
    )
)
_SERIES_TEST = concept_param(
    "indep_test",
    doc="Conditional-independence test behind every p-value, a partial-correlation test: "
    "fisherz = Fisher's z-transform; student_t = Student-t distribution, as in the authors' "
    "reference implementation. null picks fisherz.",
    choices=("student_t", "fisherz"),
)
_add(
    MethodSpec(
        name="pcmci",
        family="temporal",
        summary="PCMCI for a single multivariate time series: lagged causal links, tested with "
        "partial correlation given the parents of both ends, which controls false links from "
        "autocorrelation. No same-step links. Returns a temporal structure.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data", kind="matrix", doc="A multivariate time series, row = step."
                ),
            ),
            layout="(n_samples, n_variables), row = time step",
            missing=True,
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            doc="lag graphs for lags 1..tau_max; MCI p_values and statistics, and the PC1 parents, "
            "in metadata.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag tested; links at lags 1 to tau_max are tested.",
            ),
            ParamSpec(
                name="pc_alpha",
                type="float",
                default=0.2,
                doc="Significance level of the PC1 parent selection; larger keeps more parents as "
                "conditions.",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
            ),
            concept_param(
                "alpha",
                doc="Significance level of the MCI test, the level that decides the returned "
                "links; a link is kept when its p-value is at most this.",
                aliases=("alpha_level",),
            ),
            _SERIES_TEST,
        ),
        references=("Runge et al. 2019",),
    )
)
_add(
    MethodSpec(
        name="pcmciplus",
        family="temporal",
        summary="PCMCI+ for a single multivariate time series: lagged and same-step causal links. "
        "Lagged links point forward in time; same-step links are oriented by collider and Meek "
        "rules where the data allow. Returns a temporal structure.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data", kind="matrix", doc="A multivariate time series, row = step."
                ),
            ),
            layout="(n_samples, n_variables), row = time step",
            missing=True,
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            doc="a cpdag at lag 0 and lag graphs for lags 1..tau_max; p_values, statistics, the "
            "lagged parents, conflicting and ambiguous orientations in metadata.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note='With `collider_rule="none"`, the orientations can follow the column order.',
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag tested; links at lags 0 to tau_max are tested.",
            ),
            concept_param(
                "alpha",
                default=0.01,
                doc="Significance level of every test; a link is removed when a test gives it a "
                "p-value above this. The default is the level of the published experiments "
                "(Runge 2020).",
                aliases=("pc_alpha",),
            ),
            ParamSpec(
                name="collider_rule",
                type="str",
                default="majority",
                choices=("majority", "conservative", "none"),
                doc="How a same-step collider is judged: majority = fewer than half of the "
                "separating subsets hold the middle; conservative = none does; none = the middle "
                "is not in the one separating set found.",
                concept="collider_rule",
            ),
            _SERIES_TEST,
            concept_param(
                "max_p_global",
                doc="The most conditions a skeleton or collider test takes beyond the default "
                "conditions, the lagged parents of both ends, at least 0 (tigramite's name); "
                "max_depth caps the lagged phase. null sets no limit. A cap bounds the number of "
                "tests and can keep links that a larger conditioning set would remove.",
            ),
        ),
        references=("Runge 2020",),
    )
)
_add(
    MethodSpec(
        name="lpcmci",
        family="temporal",
        summary="LPCMCI for a single multivariate time series with possible hidden common causes: "
        "lagged and same-step links as a partial ancestral graph over a time window, where a "
        "two-headed link marks a hidden common cause. Returns a temporal structure.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data", kind="matrix", doc="A multivariate time series, row = step."
                ),
            ),
            layout="(n_samples, n_variables), row = time step",
            missing=True,
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            doc="a pag at lag 0 and the definite lagged causes for lags 1..tau_max; every link "
            "with its marks, p_values, statistics, the preliminary parents, and the conflicting "
            "marks in metadata.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag; the graph covers time steps t - tau_max to t.",
            ),
            concept_param(
                "alpha",
                doc="Significance level of every test; a link is removed when a test gives it a "
                "p-value above this.",
                aliases=("pc_alpha",),
            ),
            ParamSpec(
                name="n_preliminary_iterations",
                minimum=0,
                type="int",
                default=1,
                doc="Preliminary rounds before the final one; each passes the parents it found "
                "to the next as conditions.",
            ),
            _SERIES_TEST,
            concept_param(
                "max_p_global",
                doc="The most conditions a test takes beyond the default conditions, the parents "
                "of both ends, at least 0 (tigramite's name); max_depth elsewhere counts the whole "
                "conditioning set. null sets no limit. A cap bounds the number of tests and can "
                "keep links that a larger conditioning set would remove.",
            ),
        ),
        references=("Gerhardus & Runge 2020",),
    )
)

_add(
    MethodSpec(
        name="granger",
        family="temporal",
        summary="Lasso Granger causality for a single multivariate time series: lagged effects "
        "only, one cross-validated lasso per variable. Returns a temporal structure.",
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
            doc="lag_weights at lags 1..tau_max (the lasso coefficients, no lag 0); no ordering.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag: each regression uses the past tau_max time steps, at least "
                "1.",
                aliases=("maxlag",),
            ),
            ParamSpec(
                name="cv",
                minimum=2,
                type="int",
                default=5,
                doc="Cross-validation folds that choose each regression's lasso penalty, at "
                "least 2.",
            ),
        ),
        references=("Granger 1969", "Arnold, Liu & Abe 2007", "Shojaie & Michailidis 2010"),
    )
)

# --- pairwise family ------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="anm",
        family="pairwise",
        summary="Additive-noise-model pairwise direction test: decides whether column 0 causes "
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
            missing=True,
        ),
        params=(
            ParamSpec(
                name="alpha",
                type="float",
                default=None,
                doc="How the two p-values decide the edge. null: the direction with the larger "
                "p-value, or with the smaller HSIC statistic when the p-values are equal. A level "
                "such as 0.05: an edge only when exactly one direction's p-value is above it.",
                nullable=True,
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                concept="alpha",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="two-node graph with the decided edge, or none; metadata p_values and "
            "statistics map forward (x->y) and backward (y->x) to the p-value and the HSIC "
            "statistic, divided by its mean under independence, of each test.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        references=("Hoyer et al. 2008", "Gretton et al. 2008", "Mooij et al. 2016"),
    )
)
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
            missing=True,
        ),
        params=(
            ParamSpec(
                name="alpha",
                type="float",
                default=None,
                doc="How the two p-values decide the edge. null: the direction with the larger "
                "p-value, or with the smaller HSIC statistic when the p-values are equal. A level "
                "such as 0.05: an edge only when exactly one direction's p-value is above it.",
                nullable=True,
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                concept="alpha",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="two-node graph with the decided edge, or none; metadata p_values and "
            "statistics map forward (x->y) and backward (y->x) to the p-value and the HSIC "
            "statistic, divided by its mean under independence, of each test.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
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
            doc="observed + latent nodes; metadata clusters, cluster_order + unordered.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="can_change",
        tie_break=True,
        order_note=(
            "Reordering whole clusters relabels the result; reordering the columns inside a "
            "cluster can change the latents' order."
        ),
        dropped_args=("labels",),
        params=(
            concept_param(
                "alpha",
                doc="Significance level of the independence test that finds the causal clusters "
                "and each next latent in the order.",
            ),
        ),
        references=("Xie et al. 2020",),
    )
)


# --- hidden-confounder family: a PAG over the observed variables ---------------------------------


_add(
    MethodSpec(
        name="rcd",
        family="latent",
        summary="Repetitive causal discovery (RCD) for linear non-Gaussian data with hidden common "
        "causes: direct causes as ->, confounded pairs as <->.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        params=(
            ParamSpec(
                name="max_depth",
                type="int",
                default=2,
                doc="The maximum conditioning-set size of an ancestor test, the most variables "
                "regressed out together, at least 1; larger values find more ancestors and take "
                "longer. The default 2 is the setting of Maeda and Shimizu's experiments.",
                minimum=1,
                aliases=("max_explanatory_vars",),
                concept="max_depth",
            ),
            ParamSpec(
                name="cor_alpha",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                type="float",
                default=0.01,
                doc="Significance level of the Pearson correlation tests.",
            ),
            concept_param(
                "alpha",
                default=0.01,
                doc="Significance level of the HSIC independence tests, which decide the ancestors "
                "and so the returned edges. The default is the level of the published experiments "
                "(Maeda & Shimizu 2020).",
                aliases=("ind_alpha",),
            ),
            ParamSpec(
                name="shapiro_alpha",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                type="float",
                default=0.01,
                doc="Significance level of the Shapiro-Wilk non-Gaussianity tests.",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            weighted=True,
            doc="PAG over the observed variables; metadata confounded_pairs lists the <-> pairs, "
            "which share a hidden common cause.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        references=("Maeda & Shimizu 2020",),
    )
)

_add(
    MethodSpec(
        name="bottom_up_parce_lingam",
        family="latent",
        summary="BottomUpParceLiNGAM for linear non-Gaussian data with hidden common causes: a "
        "causal order that leaves confounded variables unordered (o-o).",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        params=(
            concept_param(
                "alpha",
                doc="Significance level of the independence test that accepts a sink, "
                "Bonferroni-corrected across the remaining variables. The default is the level of "
                "the published experiments (Tashiro et al. 2014).",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            weighted=True,
            doc="PAG over the observed variables; o-o joins the variables a hidden common "
            "cause leaves unordered, and metadata partial_order nests them.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        references=("Tashiro et al. 2014",),
    )
)

# --- background knowledge and the conditioning-set cap ------------------------------------------

_SKELETON_RULE = (
    "The adjacency search starts from the allowed pairs and never removes a required pair; the "
    "tests can remove any other allowed pair. A pair the knowledge makes non-adjacent is tested "
    "for a separating set, so colliders over it are oriented as usual; when the search finds none "
    "within max_depth, no rule orients through its unshielded triples. "
)
_PAG_MARKS = (
    "In the PAG, a forbidden a -> b on an adjacent pair puts an arrowhead at a and can leave a "
    "circle at b, and a required one is a -> b. "
)
_CLASS_RULE = (
    "The result directs every edge the knowledge decides, then Meek's rules direct the edges that "
    "follow."
)
_MOVE_RULE = (
    "No move adds a forbidden edge or removes a required one; the search starts from the required "
    "edges. "
)
_CANDIDATE_RULE = (
    "The score search considers the pairs the restrict phase keeps and the pairs of the required "
    "edges, less the pairs the knowledge makes non-adjacent; the restrict phase reads the data "
    "alone. "
)
_PARENT_RULE = "Each variable's parents exclude the forbidden ones and include the required ones. "
_ORDER_RULE = "Causal orders put the cause of each required edge, and every earlier tier, first. "
_MASK_RULE = (
    "Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are held at "
    "zero in the fit; a required edge leaves the penalty and the threshold, and the cycle repair "
    "never removes it. "
)
_REGRESSION_RULE = (
    "The regression drops forbidden parents and does not penalize required ones; every required "
    "edge is in the DAG."
)
_LINK_RULE = (
    "An edge [cause, effect, lag] is the link from cause at time t - lag to effect at time t; a "
    "forbidden [cause, effect] pair forbids it at every lag, and a required link names its lag. "
    "Tiers and allowed hold at every lag; a variable's links to its own past stay allowed. "
)
_KNOWLEDGE_HOW: dict[str, str] = {
    "pc": _SKELETON_RULE + _CLASS_RULE,
    "cdnod": _SKELETON_RULE + _CLASS_RULE + " The domain index is unconstrained.",
    "fci": _SKELETON_RULE + _PAG_MARKS,
    "gfci": "The GES phase moves as ges does; the test phase starts from the adjacencies GES found "
    "and never removes a required pair. " + _PAG_MARKS,
    "ges": _MOVE_RULE + _CLASS_RULE,
    "gies": _MOVE_RULE + _CLASS_RULE,
    "xges": _MOVE_RULE + _CLASS_RULE,
    "hc": _MOVE_RULE + _CLASS_RULE,
    "tabu": _MOVE_RULE + _CLASS_RULE,
    "mmhc": _CANDIDATE_RULE + _MOVE_RULE + _CLASS_RULE,
    "rsmax2": _CANDIDATE_RULE + _MOVE_RULE + _CLASS_RULE,
    "boss": _PARENT_RULE + _ORDER_RULE + _CLASS_RULE,
    "grasp": _PARENT_RULE + _ORDER_RULE + _CLASS_RULE,
    "exact_search": _PARENT_RULE,
    "notears": _MASK_RULE,
    "dagma": _MASK_RULE,
    "direct_lingam": _ORDER_RULE + _REGRESSION_RULE,
    "ica_lingam": "The order search reads every forbidden edge's unmixing weight as 0. "
    + _ORDER_RULE
    + _REGRESSION_RULE,
    "multi_group_direct_lingam": "It holds for every group. " + _ORDER_RULE + _REGRESSION_RULE,
    "pcmci": _LINK_RULE + "No test runs on a forbidden or a required link: a forbidden link is not "
    "in the graph, and a required one is.",
    "pcmciplus": _LINK_RULE + "The lagged and skeleton phases test no forbidden or required link: "
    "a forbidden link is never linked, and a required one stays. Same-step links the knowledge "
    "decides are directed before the colliders, which never reverse them; rules R1-R3 follow.",
    "varma_lingam": _LINK_RULE + "Lags run over the autoregressive part, 0 to order[0]. At lag 0 "
    "the innovations' DirectLiNGAM honors the knowledge as direct_lingam does. With prune=true "
    "each regression drops forbidden lagged regressors and keeps required ones unpenalized; with "
    "prune=false every lagged weight stays, so a forbidden lagged link raises.",
    "granger": _LINK_RULE + "Lags run from 1 to tau_max; a required link needs its lag, and "
    "same-step links are not learned. Each regression drops forbidden lagged regressors and keeps "
    "required ones unpenalized.",
    "longitudinal_lingam": _LINK_RULE + "At lag 0 each time point's DirectLiNGAM honors the "
    "knowledge as direct_lingam does. A forbidden lagged link is left out of the least-squares "
    "fit of its effect's lagged weights; every other lagged weight stays.",
    "rcd": "The ancestor search runs as without knowledge; the parent sets then drop forbidden "
    "parents and gain required ones, and a pair forbidden both ways is never marked <->.",
    "bottom_up_parce_lingam": "A sink is taken only when no remaining variable is in a later "
    "tier or is the effect of a required edge out of it, and each regression drops forbidden "
    "parents and keeps "
    "required ones unpenalized. In the unordered block, a pair forbidden both ways is not "
    "adjacent, a pair forbidden one way has an arrowhead at the cause, and a required edge is "
    "directed.",
    "lpcmci": _LINK_RULE + "A forbidden lagged link, and a same-step pair forbidden both ways, is "
    "never linked; a same-step link forbidden one way has an arrowhead at its cause. A required "
    "link is -->, and no test or rule removes it. No rule changes a mark the knowledge sets.",
}
#: The Knowledge parts each method honors: every part.
_KNOWLEDGE_PARTS: dict[str, tuple[KnowledgePart, ...]] = dict.fromkeys(
    _KNOWLEDGE_HOW, KNOWLEDGE_PARTS
)
_MAX_DEPTH = concept_param("max_depth")

for _name, _how in _KNOWLEDGE_HOW.items():
    _knowledge = concept_param(
        "knowledge",
        doc="Background knowledge: required and forbidden edges, the pairs allowed to be "
        f"adjacent, and temporal tiers. Honors {', '.join(_KNOWLEDGE_PARTS[_name])}. {_how}",
    )
    _depth = (_MAX_DEPTH,) if _name in CONCEPTS["max_depth"].conforming else ()
    REGISTRY[_name] = REGISTRY[_name].model_copy(
        update={
            "params": (*REGISTRY[_name].params, *_depth, _knowledge),
            "knowledge": _KNOWLEDGE_PARTS[_name],
        }
    )

#: The parent cap of the score searches that take it as the concept's template; ges has its own doc.
_MAX_PARENTS = concept_param(
    "max_parents",
    doc="A cap on every node's parents, at least 0: no DAG the returned graph represents gives a "
    "node more than max_parents parents. A cap can stop the search short of the lowest-scoring "
    "graph. null sets no cap.",
)
# The caps the knowledge loop above leaves out: max_depth on the methods without knowledge, and
# max_parents, before knowledge where a method has it.
for _concept, _cap in (("max_depth", _MAX_DEPTH), ("max_parents", _MAX_PARENTS)):
    for _name in CONCEPTS[_concept].conforming:
        _params = REGISTRY[_name].params
        if any(p.concept == _concept for p in _params):
            continue
        _cut = next((i for i, p in enumerate(_params) if p.name == "knowledge"), len(_params))
        REGISTRY[_name] = REGISTRY[_name].model_copy(
            update={"params": (*_params[:_cut], _cap, *_params[_cut:])}
        )


#: Each method's own metadata keys; the loop below adds the keys that follow from its spec.
_METADATA: dict[str, tuple[str, ...]] = {
    "pc": ("conflicts",),
    "cdnod": ("conflicts",),
    "iamb": ("target", "markov_blanket"),
    "inter_iamb": ("target", "markov_blanket"),
    "mmpc": ("target", "parents_and_children"),
    **dict.fromkeys(
        ("ges", "gies", "xges", "hc", "tabu", "boss", "grasp", "mmhc", "rsmax2"), ("score",)
    ),
    "exact_search": ("dag",),
    "calm": ("h_final", "objective"),
    **dict.fromkeys(("notears", "dagma"), ("h_final", "edges_removed", "path_steps")),
    "multi_group_direct_lingam": ("group_index",),
    "longitudinal_lingam": ("causal_orders",),
    "pcmci": ("p_values", "statistics", "parents"),
    "pcmciplus": ("p_values", "statistics", "parents", "conflicts"),
    "lpcmci": ("links", "p_values", "statistics", "parents", "conflicts"),
    **dict.fromkeys(("anm", "pnl"), ("p_values", "statistics")),
    "gin": ("clusters", "cluster_order", "unordered"),
    "rcd": ("confounded_pairs",),
    "bottom_up_parce_lingam": ("partial_order",),
}
# The shared keys: the name, the seed of a seeded fit, the test and the score it used, the column
# types a mixed-data setting read, the fewest rows a test used, and the triples a collider vote left
# ambiguous.
for _name, _spec in REGISTRY.items():
    _concepts = {p.concept for p in _spec.params}
    _shared = (
        "algorithm",
        *(("seed",) if _spec.seeded else ()),
        *(("indep_test",) if "indep_test" in _concepts else ()),
        *(("score_func",) if "score_func" in _concepts else ()),
        *(("categorical",) if "categorical" in _concepts else ()),
        *(("rows_tested",) if _spec.data.missing else ()),
        *(("ambiguous_triples",) if "collider_rule" in _concepts else ()),
    )
    _output = _spec.output.model_copy(update={"metadata": (*_shared, *_METADATA.get(_name, ()))})
    REGISTRY[_name] = _spec.model_copy(update={"output": _output})


def get_spec(name: str) -> MethodSpec:
    """The spec for one method name, or a ``KeyError`` with the valid names."""
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; valid: {', '.join(sorted(REGISTRY))}") from None


def list_specs() -> list[MethodSpec]:
    """All specs in registration (family-grouped) order."""
    return list(REGISTRY.values())
