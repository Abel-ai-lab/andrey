"""Method definitions used to generate CLI help and JSON schemas.

Each public function in ``andrey.api.__all__`` has one :class:`MethodSpec` describing its inputs,
parameters, defaults, and results. Tests check that these definitions cover the public functions
and match their signatures. Summaries explain when to use each method; the CLI and JSON schemas
display the parameter defaults defined here.
"""

from __future__ import annotations

from andrey.constraint._skeleton import COLLIDER_RULES
from andrey.constraint.pc import COLLIDER_ORDERS
from andrey.core.ci import SHIPPED_INDEP_TESTS
from andrey.core.knowledge import KNOWLEDGE_PARTS, KnowledgePart
from andrey.core.score import LINEAR_SCORES
from andrey.search.boss import PERMUTATION_SCORES
from andrey.search.exact import EXACT_SCORES
from andrey.search.ges import GES_SCORES
from andrey.search.hc import HC_SCORES
from andrey.search.hybrid import MAXIMIZES, RESTRICTS
from andrey.search.tabu import TABU_SCORES

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
    "order.",
    choices=SHIPPED_INDEP_TESTS,  # single source of truth: the shipped CI-registry names
)
_CATEGORICAL = ParamSpec(
    name="categorical",
    type="str_list",
    default=None,
    doc="The categorical columns, for a mixed-data test or score: column names from the data "
    "file's header or 0-based positions, for example, categorical=[smoker,region]; [] marks none. "
    "null infers them: every column of at most max_categories distinct whole numbers, named in a "
    "DataTypeWarning; with none, the setting runs as its linear-Gaussian form and warns. Other "
    "tests and scores take only null or [].",
    nullable=True,
)
_MAX_CATEGORIES = ParamSpec(
    name="max_categories",
    type="int",
    default=10,
    doc="The most distinct whole numbers a column may hold to be inferred as categorical, at least "
    "2. With a continuous test or score, such columns get a one-time DataTypeWarning that they may "
    "be category codes.",
)
_COLLIDER_RULE = ParamSpec(
    name="collider_rule",
    type="str",
    default="sepsets",
    choices=COLLIDER_RULES,
    doc="How colliders are decided on x - z - y: sepsets = z is a collider when it is not in the "
    "separating set of x and y; conservative and majority = test x and y given every subset of "
    "each one's neighbors, and z is a collider when none (conservative) or fewer than half "
    "(majority) of the separating subsets contain it, a non-collider when all or more than half "
    "do, and ambiguous otherwise, never oriented through. Both cost up to 2**degree tests per "
    "pair.",
)
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
_LAMBDA_VALUE = ParamSpec(
    name="lambda_value",
    type="float",
    default=1.0,
    doc="Multiplier on the BIC penalty of log(n_samples) per parent. Larger values give sparser "
    "graphs; 1.0 with ebic_gamma=0.0 is the standard BIC.",
)
_EBIC_GAMMA = ParamSpec(
    name="ebic_gamma",
    type="float",
    default=1.0,
    doc="Weight on the extended-BIC term 2 * ebic_gamma * log C(n_variables - 1, n_parents) that "
    "the linear-Gaussian BIC adds per variable; it grows with the number of variables. Larger "
    "values give sparser graphs; 0.0 is the standard BIC. Other scores ignore it.",
)
_EBIC_GAMMA_STANDARD = ParamSpec(
    name="ebic_gamma",
    type="float",
    default=0.0,
    doc="Weight on the extended-BIC term 2 * ebic_gamma * log C(n_variables - 1, n_parents) that "
    "the linear-Gaussian BIC adds per variable; it grows with the number of variables. Larger "
    "values give sparser graphs; 0.0 is the standard BIC. Other scores ignore it.",
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
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires Meek\'s rules one at a time in column order, and it or '
            '`collider_order="column"` orients colliders in column order.'
        ),
        params=(
            _ALPHA,
            _INDEP_TEST,
            _COLLIDER_ORDER,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
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
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires the orientation rules one at a time in column order, and '
            'it or `possible_dsep="fast"` visits the possible-d-separation step in column order.'
        ),
        params=(
            _ALPHA,
            _INDEP_TEST,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _POSSIBLE_DSEP,
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
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires the orientation rules one at a time in column order and '
            'records the first separating subset found, and it or `possible_dsep="fast"` visits '
            "the possible-d-separation step in column order."
        ),
        params=(
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC",
                doc="Local score for the GES phase: local_score_BIC or local_score_BIC_from_cov = "
                "linear-Gaussian BIC; "
                "local_score_BDeu = BDeu and local_score_BIC_categorical = multinomial BIC, for "
                "categorical data coded by whole numbers; "
                "local_score_degen_gauss and local_score_cond_gauss = degenerate and conditional "
                "Gaussian BIC, for mixed data (experimental). The test must read the same kind of "
                "data.",
                choices=GES_SCORES,
            ),
            _LAMBDA_VALUE,
            _EBIC_GAMMA_STANDARD,
            _INDEP_TEST,
            _ALPHA,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            _POSSIBLE_DSEP,
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
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            '`tie_break="position"` fires Meek\'s rules one at a time in column order, and it or '
            '`collider_order="column"` orients colliders in column order.'
        ),
        params=(
            _ALPHA,
            _INDEP_TEST,
            _COLLIDER_ORDER,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _MAX_CATEGORIES,
            ParamSpec(
                name="dependence_threshold",
                type="float",
                default=None,
                doc="Runs the independent-change step, which orients edges between variables that "
                "both change with the index, at this threshold on its module dependence measure. "
                "null skips the step.",
                nullable=True,
            ),
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
                graph_kind="cpdag",
                doc=_LOCAL_OUTPUT_DOC.format(key=_key),
            ),
            determinism="deterministic",
            accel="numpy",
            status="experimental",
            column_order="relabels",
            tie_break=True,
            params=(_TARGET, _ALPHA, _INDEP_TEST, _CATEGORICAL, _MAX_CATEGORIES),
            references=_refs,
        )
    )

# --- score-based ----------------------------------------------------------------------------------

_add(
    MethodSpec(
        name="ges",
        family="score",
        summary="Greedy equivalence search: a score-based greedy search over CPDAGs. Fast and "
        "accurate on linear-Gaussian data with the BIC; local_score_BDeu, the default when every "
        "column is categorical, or local_score_BIC_categorical for categorical data, "
        "local_score_cond_gauss, the default when some columns are, or local_score_degen_gauss for "
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
            ParamSpec(
                name="score_func",
                type="str",
                default=None,
                doc="Local score objective: local_score_BIC or local_score_BIC_from_cov = "
                "linear-Gaussian BIC; "
                "local_score_BDeu = BDeu and local_score_BIC_categorical = multinomial BIC, for "
                "categorical data coded by whole numbers; "
                "local_score_degen_gauss and local_score_cond_gauss = degenerate and conditional "
                "Gaussian BIC, for mixed data; "
                "null picks local_score_BDeu when every column is categorical (a category, bool, "
                "or text column, or at most max_categories distinct whole numbers, or a column "
                "categorical lists) and "
                "local_score_cond_gauss when some are, each with a DataTypeWarning, else "
                "local_score_BIC.",
                choices=GES_SCORES,
                nullable=True,
            ),
            _LAMBDA_VALUE,
            _EBIC_GAMMA,
            ParamSpec(
                name="max_parents",
                type="int",
                default=None,
                doc="A cap on every node's parents, at least 0: no DAG the returned CPDAG "
                "represents gives a node more than max_parents parents, and some DAG reaches each "
                "node's count. A move whose CPDAG would "
                "break the cap, in either phase, is skipped for the next best move. A cap can "
                "stop the search short of the lowest-scoring graph, which GES otherwise finds. "
                "null sets no cap; the forward phase then keeps its gate of n_variables / 2.",
                nullable=True,
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
        params=(_LAMBDA_VALUE, _EBIC_GAMMA),
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
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC",
                doc="Local score objective: local_score_BIC or local_score_BIC_from_cov = "
                "linear-Gaussian BIC, the only score XGES supports.",
                choices=LINEAR_SCORES,
            ),
            ParamSpec(
                name="lambda_value",
                type="float",
                default=1.0,
                doc="Multiplier on the BIC penalty of log(n_samples) per parent. Larger values "
                "give sparser graphs; 1.0 with ebic_gamma=0.0 is the standard BIC.",
            ),
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
        "categorical, conditional Gaussian when some are; score_func=local_score_BDeu or "
        "local_score_BIC_categorical for categorical data, local_score_degen_gauss or "
        "local_score_cond_gauss for mixed data. Stops at a local optimum, or after max_iter moves "
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
            ParamSpec(
                name="score_func",
                type="str",
                default=None,
                doc="Local score objective: local_score_BIC_from_cov or local_score_BIC = "
                "covariance-based BIC; "
                "local_score_BDeu = BDeu and local_score_BIC_categorical = multinomial BIC, for "
                "categorical data coded by whole numbers; "
                "local_score_degen_gauss and local_score_cond_gauss = degenerate and conditional "
                "Gaussian BIC, for mixed data; "
                "null picks local_score_BDeu when every column is categorical (a category, bool, "
                "or text column, or at most max_categories distinct whole numbers, or a column "
                "categorical lists) and "
                "local_score_cond_gauss when some are, each with a DataTypeWarning, else "
                "local_score_BIC_from_cov.",
                choices=HC_SCORES,
                nullable=True,
            ),
            _LAMBDA_VALUE,
            _EBIC_GAMMA,
            ParamSpec(
                name="max_iter",
                type="int",
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
        "when every column is categorical; score_func=local_score_BDeu or "
        "local_score_BIC_categorical for categorical data. Returns "
        "the CPDAG of that DAG and its score.",
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
            ParamSpec(
                name="score_func",
                type="str",
                default=None,
                doc="Local score objective: local_score_BIC_from_cov or local_score_BIC = "
                "covariance-based BIC; "
                "local_score_BDeu = BDeu and local_score_BIC_categorical = multinomial BIC, for "
                "categorical data coded by whole numbers; "
                "null picks local_score_BDeu, with a DataTypeWarning, when every column is "
                "categorical (at most 10 distinct whole numbers), else local_score_BIC_from_cov, "
                "mixed data included: tabu has no mixed-data score.",
                choices=TABU_SCORES,
                nullable=True,
            ),
            _LAMBDA_VALUE,
            _EBIC_GAMMA,
            ParamSpec(
                name="tabu_length",
                type="int",
                default=10,
                doc="Length of the tabu list: how many recently visited DAGs a move may not lead "
                "to, at least 1.",
            ),
            ParamSpec(
                name="max_tabu",
                type="int",
                default=None,
                doc="Non-improving moves allowed without a better DAG before the search stops, "
                "from 1 to tabu_length. null uses tabu_length.",
                nullable=True,
            ),
            ParamSpec(
                name="max_iter",
                type="int",
                default=None,
                doc="Most moves the search takes, improving or not, at least 1; null sets no limit "
                "(the search still ends). Reaching a set limit warns with SearchLimitWarning.",
                nullable=True,
            ),
        ),
        references=("Glover 1986", "Russell & Norvig 2009", "Scutari 2010"),
    )
)
_add(
    MethodSpec(
        name="exact_search",
        family="score",
        summary="Globally optimal score search over DAGs (A* or dynamic programming). Exact but "
        "exponential in the number of variables; keep to about 20 or fewer. Returns one DAG from "
        "the best-scoring equivalence class. BIC by default, BDeu when every column is "
        "categorical, conditional Gaussian when some are; score_func=local_score_BDeu or "
        "local_score_BIC_categorical for categorical data, local_score_degen_gauss or "
        "local_score_cond_gauss for mixed data.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="metadata['categorical'] lists the categorical columns a mixed-data setting read",
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
            ParamSpec(
                name="score_func",
                type="str",
                default=None,
                doc="Local score: local_score_BIC or local_score_BIC_from_cov = linear-Gaussian "
                "BIC; local_score_BDeu = BDeu "
                "and local_score_BIC_categorical = multinomial BIC, for categorical data coded by "
                "whole numbers; local_score_degen_gauss and local_score_cond_gauss = degenerate "
                "and conditional Gaussian BIC, for mixed data; "
                "null picks local_score_BDeu when every column is categorical (a category, bool, "
                "or text column, or at most max_categories distinct whole numbers, or a column "
                "categorical lists) and "
                "local_score_cond_gauss when some are, each with a DataTypeWarning, else "
                "local_score_BIC.",
                choices=EXACT_SCORES,
                nullable=True,
            ),
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
            doc="weighted DAG; no score. Extra optimizer knobs (lambda1, lr, ...) pass through.",
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
        type="float",
        default=0.4,
        doc="Penalty level of the path's first step, above 0. Used only with robust.",
    ),
    ParamSpec(
        name="mcp_delta",
        type="float",
        default=0.2,
        doc="Weight magnitude above which the quasi-MCP stops growing, at the path's first step, "
        "above 0. Used only with robust.",
    ),
    ParamSpec(
        name="mcp_decay",
        type="float",
        default=0.8,
        doc="Factor in (0, 1) by which mcp_lambda and mcp_delta shrink after each accepted step. "
        "Used only with robust.",
    ),
    ParamSpec(
        name="max_path",
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
            ParamSpec(
                name="lambda1",
                type="float",
                default=0.1,
                doc="Weight of the l1 penalty of the first fit, at least 0.",
            ),
            ParamSpec(
                name="w_threshold",
                type="float",
                default=0.3,
                doc="An edge is kept when its weight magnitude reaches this value, at least 0.",
            ),
            ParamSpec(
                name="max_iter",
                type="int",
                default=100,
                doc="Most dual steps of the augmented Lagrangian, at least 1.",
            ),
            ParamSpec(
                name="h_tol",
                type="float",
                default=1e-8,
                doc="The fit stops when the acyclicity function falls to this value, above 0.",
            ),
            ParamSpec(
                name="rho_max",
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
            ParamSpec(
                name="lambda1",
                type="float",
                default=0.03,
                doc="Weight of the l1 penalty of the first fit, at least 0.",
            ),
            ParamSpec(
                name="w_threshold",
                type="float",
                default=0.3,
                doc="An edge is kept when its weight magnitude reaches this value, at least 0.",
            ),
            ParamSpec(
                name="stages",
                type="int",
                default=5,
                doc="Stages of the central path, at least 1 (T in the paper).",
            ),
            ParamSpec(
                name="warm_iter",
                type="int",
                default=30000,
                doc="Most Adam steps of each stage but the last, at least 1.",
            ),
            ParamSpec(
                name="max_iter",
                type="int",
                default=60000,
                doc="Most Adam steps of the last stage, at least 1.",
            ),
            ParamSpec(name="lr", type="float", default=3e-4, doc="Step size of Adam, above 0."),
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

_HYBRID_SCORE_DOC = (
    "Local score of the maximize phase: local_score_BIC_from_cov or local_score_BIC = "
    "covariance-based BIC; "
    "local_score_BDeu = BDeu and local_score_BIC_categorical = multinomial BIC, for categorical "
    "data coded by whole numbers; local_score_degen_gauss and local_score_cond_gauss = degenerate "
    "and conditional Gaussian BIC, for mixed data (not with maximize=tabu). It must read the same "
    "kind of data as indep_test."
)
_HYBRID_MAX_ITER = ParamSpec(
    name="max_iter",
    type="int",
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
        "indep_test=chisq with score_func=local_score_BDeu for categorical data. Returns the "
        "CPDAG of the final DAG and its score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['restrict'] and metadata['maximize'] name "
            "the phases",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
        params=(
            _ALPHA,
            _INDEP_TEST,
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC_from_cov",
                doc=_HYBRID_SCORE_DOC,
                choices=HC_SCORES,
            ),
            _LAMBDA_VALUE,
            _EBIC_GAMMA,
            _HYBRID_MAX_ITER,
            _CATEGORICAL,
            _MAX_CATEGORIES,
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
            doc="score in metadata['score']; metadata['restrict'] and metadata['maximize'] name "
            "the phases",
        ),
        determinism="deterministic",
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
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
            ParamSpec(
                name="score_func",
                type="str",
                default="local_score_BIC_from_cov",
                doc=_HYBRID_SCORE_DOC,
                choices=HC_SCORES,
            ),
            _LAMBDA_VALUE,
            _EBIC_GAMMA,
            _HYBRID_MAX_ITER,
            ParamSpec(
                name="tabu_length",
                type="int",
                default=10,
                doc="With maximize=tabu, the number of most recently visited DAGs a move may not "
                "return to, at least 1.",
            ),
            ParamSpec(
                name="max_tabu",
                type="int",
                default=None,
                doc="With maximize=tabu, the non-improving moves allowed without a better DAG, "
                "from 1 to tabu_length; null uses tabu_length.",
                nullable=True,
            ),
            _CATEGORICAL,
            _MAX_CATEGORIES,
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

_BOSS_SCORE = ParamSpec(
    name="score_func",
    type="str",
    default=None,
    doc="Local score objective: local_score_BIC_from_cov or local_score_BIC = covariance-based "
    "BIC; "
    "local_score_BDeu = BDeu and local_score_BIC_categorical = multinomial BIC, for categorical "
    "data coded by whole numbers; "
    "local_score_degen_gauss and local_score_cond_gauss = degenerate and conditional Gaussian BIC, "
    "for mixed data (experimental); "
    "null picks local_score_BDeu when every column is categorical (a category, bool, or text "
    "column, or at most max_categories distinct whole numbers, or a column categorical lists) "
    "and local_score_cond_gauss when some are, each with a DataTypeWarning, else "
    "local_score_BIC_from_cov.",
    choices=PERMUTATION_SCORES,
    nullable=True,
)
_add(
    MethodSpec(
        name="boss",
        family="permutation",
        summary="Best-order score search: a permutation search over the linear-Gaussian BIC, fast "
        "and accurate on linear-Gaussian data; local_score_BDeu, the default when every column is "
        "categorical, or local_score_BIC_categorical for categorical data, local_score_cond_gauss, "
        "the default when some columns are, or local_score_degen_gauss for mixed data. Returns a "
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
            _BOSS_SCORE,
            _LAMBDA_VALUE,
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
        "linear-Gaussian BIC, like BOSS with a different move (tucks); local_score_BDeu, the "
        "default when every column is categorical, or local_score_BIC_categorical for categorical "
        "data, local_score_cond_gauss, the default when some columns are, or "
        "local_score_degen_gauss for mixed data. Returns a CPDAG + score.",
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
            _BOSS_SCORE,
            _LAMBDA_VALUE,
            _EBIC_GAMMA,
            ParamSpec(
                name="depth",
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
        column_order="relabels_if_converged",
        tie_break=True,
        order_note=(
            "At few rows per variable, FastICA can reach another solution under a new "
            "column order or on another machine."
        ),
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
        status="supported",
        column_order="relabels",
        tie_break=True,
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
            ParamSpec(
                name="max_iter",
                type="int",
                default=100,
                doc="Most iterations of the likelihood fit, at least 1. A fit stopped before it "
                "converges gives no warning; raise this to let it converge.",
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
        status="supported",
        column_order="relabels",
        tie_break=True,
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
            ParamSpec(
                name="tau_max",
                type="int",
                default=1,
                doc="The largest lag tested; links at lags 1 to tau_max are tested.",
            ),
            ParamSpec(
                name="pc_alpha",
                type="float",
                default=0.2,
                doc="Significance level of the PC1 parent selection; larger keeps more parents as "
                "conditions.",
            ),
            ParamSpec(
                name="alpha_level",
                type="float",
                default=0.05,
                doc="Significance level of the MCI test; a link is kept when its p-value is at "
                "most this.",
            ),
            ParamSpec(
                name="indep_test",
                type="str",
                default="student_t",
                choices=("student_t", "fisherz"),
                doc="Partial-correlation test behind every p-value: student_t = Student-t "
                "distribution, as in the authors' reference implementation; fisherz = Fisher's "
                "z-transform.",
            ),
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
            ParamSpec(
                name="tau_max",
                type="int",
                default=1,
                doc="The largest lag tested; links at lags 0 to tau_max are tested.",
            ),
            ParamSpec(
                name="pc_alpha",
                type="float",
                default=0.01,
                doc="Significance level of every test; a link is removed when a test gives it a "
                "p-value above this.",
            ),
            ParamSpec(
                name="collider_rule",
                type="str",
                default="majority",
                choices=("majority", "conservative", "none"),
                doc="How a same-step collider is judged: majority = fewer than half of the "
                "separating subsets hold the middle; conservative = none does; none = the middle "
                "is not in the one separating set found.",
            ),
            ParamSpec(
                name="indep_test",
                type="str",
                default="student_t",
                choices=("student_t", "fisherz"),
                doc="Partial-correlation test behind every p-value: student_t = Student-t "
                "distribution, as in the authors' reference implementation; fisherz = Fisher's "
                "z-transform.",
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
            ParamSpec(
                name="tau_max",
                type="int",
                default=1,
                doc="The largest lag; the graph covers time steps t - tau_max to t.",
            ),
            ParamSpec(
                name="pc_alpha",
                type="float",
                default=0.05,
                doc="Significance level of every test; a link is removed when a test gives it a "
                "p-value above this.",
            ),
            ParamSpec(
                name="n_preliminary_iterations",
                type="int",
                default=1,
                doc="Preliminary rounds before the final one; each passes the parents it found "
                "to the next as conditions.",
            ),
            ParamSpec(
                name="indep_test",
                type="str",
                default="student_t",
                choices=("student_t", "fisherz"),
                doc="Partial-correlation test behind every p-value: student_t = Student-t "
                "distribution, as in the authors' reference implementation; fisherz = Fisher's "
                "z-transform.",
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
            doc="lag_weights at lags 1..maxlag (the lasso coefficients, no lag 0); no ordering.",
        ),
        determinism="deterministic",
        accel="numpy",
        status="supported",
        column_order="relabels",
        params=(
            ParamSpec(
                name="maxlag",
                type="int",
                default=2,
                doc="Number of past time steps each regression uses, at least 1.",
            ),
            ParamSpec(
                name="cv",
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
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="two-node graph with the decided edge, or none; metadata pval_forward (x->y) "
            "and pval_backward (y->x), and stat_forward and stat_backward, each HSIC statistic "
            "divided by its mean under independence.",
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
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="two-node graph with the decided edge, or none; metadata pval_forward (x->y) "
            "and pval_backward (y->x), and stat_forward and stat_backward, each HSIC statistic "
            "divided by its mean under independence.",
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
            doc="observed + latent nodes; metadata clusters, causal_order + unordered.",
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
        params=(
            ParamSpec(
                name="alpha",
                type="float",
                default=0.05,
                doc="Significance level of the independence test that finds the causal clusters "
                "and each next latent in the order.",
            ),
            ParamSpec(
                name="labels",
                type="str_list",
                default=None,
                doc="Names for the observed variables, for example, labels=[x1,x2]. null uses the "
                "data file's header, else X1..Xn; latents are named L1, L2, ..., skipping a name "
                "an observed variable has.",
                nullable=True,
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
                name="max_explanatory_vars",
                type="int",
                default=2,
                doc="Most variables regressed out together in an ancestor test, at least 1; "
                "larger values find more ancestors and take longer.",
            ),
            ParamSpec(
                name="cor_alpha",
                type="float",
                default=0.01,
                doc="Significance level of the Pearson correlation tests.",
            ),
            ParamSpec(
                name="ind_alpha",
                type="float",
                default=0.01,
                doc="Significance level of the HSIC independence tests.",
            ),
            ParamSpec(
                name="shapiro_alpha",
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
            ParamSpec(
                name="alpha",
                type="float",
                default=0.1,
                doc="Significance level of the independence test that accepts a sink, "
                "Bonferroni-corrected across the remaining variables.",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            weighted=True,
            doc="PAG over the observed variables; o-o joins the variables a hidden common "
            "cause leaves unordered, and metadata causal_order nests them.",
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
    "longitudinal_lingam": _LINK_RULE + "At lag 0 each time point's DirectLiNGAM honors the "
    "knowledge as direct_lingam does. A forbidden lagged link is left out of the least-squares "
    "fit of its effect's lagged weights; every other lagged weight stays.",
    "rcd": "The ancestor search runs as without knowledge; the parent sets then drop forbidden "
    "parents and gain required ones, and a pair forbidden both ways is never marked <->.",
    "bottom_up_parce_lingam": "A sink is taken only when no remaining variable is in a later "
    "tier, and each regression drops forbidden parents. In the unordered block, a pair forbidden "
    "both ways is not adjacent and a pair forbidden one way has an arrowhead at the cause. "
    "Required edges raise.",
    "lpcmci": _LINK_RULE + "A forbidden lagged link, and a same-step pair forbidden both ways, is "
    "never linked; a same-step link forbidden one way has an arrowhead at its cause. A required "
    "link is -->, and no test or rule removes it. No rule changes a mark the knowledge sets.",
}
#: The Knowledge parts each method honors: every part, but for BottomUpParceLiNGAM.
_KNOWLEDGE_PARTS: dict[str, tuple[KnowledgePart, ...]] = {
    **dict.fromkeys(_KNOWLEDGE_HOW, KNOWLEDGE_PARTS),
    "bottom_up_parce_lingam": ("forbidden", "allowed", "tiers"),
}
_MAX_DEPTH = ParamSpec(
    name="max_depth",
    type="int",
    default=None,
    doc="The maximum conditioning-set size of every conditional-independence test, at least 0: 0 "
    "runs only unconditional tests, and null sets no limit. A smaller cap runs fewer tests and can "
    "keep edges that a larger conditioning set would remove.",
    nullable=True,
)

for _name, _how in _KNOWLEDGE_HOW.items():
    _knowledge = ParamSpec(
        name="knowledge",
        type="knowledge",
        default=None,
        doc="Background knowledge: required and forbidden edges, the pairs allowed to be "
        f"adjacent, and temporal tiers. Honors {', '.join(_KNOWLEDGE_PARTS[_name])}. {_how}",
        nullable=True,
    )
    _depth = (_MAX_DEPTH,) if _name in ("pc", "cdnod", "fci", "gfci") else ()
    REGISTRY[_name] = REGISTRY[_name].model_copy(
        update={
            "params": (*REGISTRY[_name].params, *_depth, _knowledge),
            "knowledge": _KNOWLEDGE_PARTS[_name],
        }
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
