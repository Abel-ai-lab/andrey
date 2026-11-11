"""Method definitions used to generate CLI help and JSON schemas.

Each public function in ``andrey.api.__all__`` has one :class:`MethodSpec` describing its inputs,
parameters, defaults, and results. Tests check that these definitions cover the public functions
and match their signatures. Summaries explain when to use each method; the CLI and JSON schemas
display the parameter defaults defined here.
"""

from __future__ import annotations

from andrey.constraint.pc import COLLIDER_ORDERS
from andrey.core.ci import CONTINUOUS_INDEP_TESTS
from andrey.core.ci_missing import EM, EM_METHODS, EM_SUPPORTED
from andrey.core.knowledge import EDGE_PARTS, PATH_PARTS, KnowledgePart
from andrey.search.hybrid import MAXIMIZES, RESTRICTS

from .concepts import CONCEPTS, concept_param
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
    "distribution, exact for Gaussian data at any sample size; normal_scores = fisherz on each "
    "column's normal scores, for data whose columns are monotone transformations of Gaussian "
    "variables. For categorical data coded by whole "
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
_COLLIDER_RULE = concept_param("collider_rule")
_POSSIBLE_DSEP = ParamSpec(
    name="possible_dsep",
    type="str",
    default=None,
    choices=("order_free", "fast"),
    doc="The order of the possible-d-separation step, which removes edges one at a time: "
    "order_free = by a rank computed from the data, so the result does not depend "
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
_SCORE_DOC = (
    "Local score: bic = linear-Gaussian BIC; bdeu = BDeu and discrete_bic = multinomial BIC, "
    "for categorical data coded by whole numbers; degen_gauss and cond_gauss = degenerate and "
    "conditional Gaussian BIC, for mixed data. " + _DECLARED.format(every="bdeu", fallback="bic")
)
_SCORE_FUNC = concept_param("score_func", doc=_SCORE_DOC)
_EBIC_GAMMA = concept_param("ebic_gamma")
_EBIC_GAMMA_STANDARD = concept_param("ebic_gamma", default=0.0)
_EQUIVALENT_SAMPLE_SIZE = concept_param("equivalent_sample_size")
_SEED = concept_param(
    "seed",
    doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh random one, "
    "so a fit repeats. Seed for the search's random orders.",
)
_N_STARTS = concept_param("n_starts")
#: The seed of hc and tabu, whose only random draws are the changes before a later start.
_RESTART_SEED = concept_param(
    "seed",
    doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh random one, "
    "so a fit repeats. Seed for the random edge changes before each search after the first, which "
    "only n_starts above 1 makes.",
)
_N_PERTURBATIONS = concept_param("n_perturbations")
_RESTART_DOC = (
    "; with n_starts above 1, metadata['seed'] is the seed used and metadata['n_perturbations'] "
    "the number of random edge changes before each later search"
)
_TEST_SEED = concept_param(
    "seed",
    doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh random one, "
    "so a fit repeats. Seed for the random draws of indep_test, which only rcot, cmiknn, and gpdc "
    "make; the other tests ignore it.",
)
_MATRIX_LAYOUT = "(n_samples, n_variables), row = sample"
#: The environment index the score searches and their FCI hybrids take for data pooled across
#: environments.
_ENV_INDEX = InputSpec(
    name="env",
    kind="context",
    doc="The environment of each row, for data pooled across environments: one label per row, "
    "(n_samples,) or (n_samples, 1), or a column of the data file, which is then not a variable. "
    "The index joins the search as a variable no data variable causes. Numbers are read as "
    "numbers; text labels are coded 0, 1, ... in sorted order. Leave it out for one environment.",
    required=False,
)
_ENV_NOTES = (
    "With env, the index becomes a trailing node no data variable causes, then is projected out of "
    "the result; its edges go in metadata['index_edges']."
)
_ENV_DATA = DataSpec(inputs=(_DATA_MATRIX, _ENV_INDEX), layout=_MATRIX_LAYOUT, notes=_ENV_NOTES)
#: The ``metadata`` a CPDAG search and a PAG search add with ``env``.
_ENV_CPDAG_DOC = (
    "; with env, metadata['index_edges'] lists the index's edges as [variable, 'directed'] pairs "
    "and metadata['changing_variables'] the same variables, whose causal module changes across "
    "the environments, in column order"
)
_ENV_PAG_DOC = (
    ", the index numbered n_variables; with env, metadata['index_edges'] lists the index's edges "
    "as [variable, kind] pairs, kind directed, partially_directed, or bidirected, in column order"
)

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
            "[x, z, y] triples the rule left ambiguous, by 0-based position; with missing=mvpc on "
            "data with missing values, metadata['missingness_parents'] maps each column whose "
            "missingness indicator has parents to them, and metadata['corrected_edges'] lists the "
            "edges tested again (retested), removed (removed), and left with a test that had too "
            "few rows to correct (uncorrected), by name or 0-based position",
        ),
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
            concept_param(
                "missing",
                choices=("testwise", "mvpc"),
                doc="How missing values (NaN) are handled: testwise = each test reads only the "
                "rows where all of its variables are observed (test-wise deletion); mvpc = "
                "Missing-Value PC: after the test-wise skeleton, the parents of each missingness "
                "indicator are found and the edges they could have kept are tested again on "
                "virtual data that correct for them, with fisherz or student_t; null picks the "
                "method's default, testwise.",
            ),
            _TEST_SEED,
        ),
        references=(
            "Spirtes & Glymour 1991",
            "Spirtes, Glymour & Scheines 2000",
            "Colombo & Maathuis 2014",
            "Meek 1995",
            "Ramsey, Zhang & Spirtes 2006",
            "Tu et al. 2019",
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
        data=_ENV_DATA,
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            doc="metadata['categorical'] lists the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position" + _ENV_PAG_DOC,
        ),
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
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA_STANDARD,
            _EQUIVALENT_SAMPLE_SIZE,
            _INDEP_TEST,
            _ALPHA,
            _COLLIDER_RULE,
            _CATEGORICAL,
            _POSSIBLE_DSEP,
            concept_param(
                "edge_removal",
                doc="Which edges of the GES CPDAG the adjacency search and the "
                "possible-d-separation step may remove: all = every edge; triangles = only the "
                "edges in a triangle of the CPDAG, whose ends share a neighbor in it (Ogarrio, "
                "Spirtes and Ramsey 2016, Section 5). An edge in no triangle then stays and still "
                "conditions the tests of the other pairs.",
            ),
            concept_param("confounded_index"),
            _TEST_SEED,
            concept_param(
                "max_parents",
                doc="A cap on every node's parents in the GES phase, at least 0: no DAG the GES "
                "CPDAG represents gives a node more than max_parents parents. The test phase only "
                "removes edges from that CPDAG; the cap does not bound the arrowheads at a node of "
                "the PAG. null sets no cap; the GES phase then keeps its gate of n_variables / 2.",
            ),
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
                    name="env",
                    kind="context",
                    doc="The environment (domain) of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "Numbers, such as a time step, are read as numbers; text labels are coded 0, "
                    "1, ... in sorted order.",
                ),
            ),
            layout=_MATRIX_LAYOUT,
            notes="env becomes a trailing pseudo-node, then projected out of the result.",
            missing=True,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="metadata['index_edges'] lists the index's edges as [variable, 'directed'] "
            "pairs and metadata['changing_variables'] the same variables, whose causal module "
            "changes across the environments, in column order; metadata['categorical'] lists "
            "the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position",
        ),
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
_add(
    MethodSpec(
        name="fci_jci",
        family="constraint",
        summary="FCI for data pooled from several domains or regimes, allowing hidden common "
        "causes: adds the environment index as a variable no data variable causes (Joint Causal "
        "Inference), which removes edges pooling adds and can orient more. Returns a PAG over the "
        "data variables and lists the index's edges.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="env",
                    kind="context",
                    doc="The environment (domain) of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "Numbers are read as numbers; text labels are coded 0, 1, ... in sorted order.",
                ),
            ),
            layout=_MATRIX_LAYOUT,
            notes="env becomes a trailing context node, then projected out of the result; its "
            "edges go in metadata['index_edges'].",
            missing=True,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            doc="metadata['index_edges'] lists the index's edges as [variable, kind] pairs, kind "
            "directed, partially_directed, or bidirected, in column order; "
            "metadata['categorical'] lists the categorical columns a mixed-data setting read; "
            "under collider_rule=conservative or majority, metadata['ambiguous_triples'] lists the "
            "[x, z, y] triples the rule left ambiguous, by 0-based position, the index numbered "
            "n_variables",
        ),
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
            _POSSIBLE_DSEP,
            concept_param("confounded_index"),
            _TEST_SEED,
        ),
        references=(
            "Mooij, Magliacane & Claassen 2020",
            "Spirtes, Glymour & Scheines 2000",
            "Zhang 2008",
            "Colombo & Maathuis 2014",
            "Ramsey, Zhang & Spirtes 2006",
        ),
    )
)
_add(
    MethodSpec(
        name="icp",
        family="constraint",
        summary="Invariant causal prediction: the causal parents of one target from continuous "
        "data pooled from several environments, such as observation and experiments, that leave "
        "the target's mechanism alone. Tests every set of the other variables for a regression "
        "that is the same in every environment; the variables in every accepted set are parents "
        "with probability at least 1 - alpha. Returns a DAG of edges into the target, with "
        "per-variable p-values and confidence intervals.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="env",
                    kind="context",
                    doc="The environment of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "At least two environments.",
                ),
            ),
            layout=_MATRIX_LAYOUT,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="edges into the target from its estimated parents; metadata['parents'], "
            "metadata['p_values'] and metadata['confidence_intervals'] per column, "
            "metadata['accepted_sets'], metadata['model_rejected']",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        order_note="The listing order of `metadata['accepted_sets']` follows the column order.",
        params=(
            ParamSpec(
                name="target",
                type="str",
                default=None,
                doc="Required: the variable whose parents to find, a column name from the data "
                "file's header or a 0-based position.",
                nullable=True,
            ),
            ParamSpec(
                name="targets",
                type="targets",
                default=None,
                doc="The variables intervened on in each environment of env, as for gies: a JSON "
                "object mapping each label of env to a list of variables. An environment that "
                "intervened on target is left out; one without an entry is kept. null keeps every "
                "environment.",
                nullable=True,
            ),
            concept_param(
                "alpha",
                doc="The significance level of every set's invariance test: the estimated parents "
                "are a subset of the true parents with probability at least 1 - alpha.",
            ),
            ParamSpec(
                name="invariance_test",
                type="str",
                default="approximate",
                choices=("approximate", "exact"),
                concept="invariance_test",
                doc="approximate = one regression on every row, each environment's residuals "
                "against the others' by Welch's t-test and the F-test of their variances; exact = "
                "the regression on the other rows predicts each environment's rows, by Chow's "
                "F-test, exact for Gaussian noise.",
            ),
            concept_param(
                "max_parents",
                doc="A cap on the target's parents, at least 0: only sets of at most max_parents "
                "variables are tested, and the guarantee holds when the target has at most that "
                "many parents. null tests every set.",
            ),
        ),
        references=("Peters, Buhlmann & Meinshausen 2016", "Chow 1960"),
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
    "without a target, the graph structure names: the CPDAG, the skeleton, or, for the blankets, "
    "the moral graph; with a target, undirected edges from target to each member, "
    "metadata['target'] naming the target and metadata['{key}'] listing the members in the order "
    "they joined"
)
_BLANKET_STRUCTURE = concept_param("structure")
for _name, _summary, _key, _structure, _kind, _refs in (
    (
        "iamb",
        "One variable's Markov blanket (its parents, children, and children's other parents) by "
        "incremental association: grow by conditional-independence tests, then shrink. With "
        "target=null, every variable's blanket joined into a CPDAG by the Grow-Shrink structure "
        "learner, or into the skeleton or the moral graph (structure).",
        "markov_blanket",
        _BLANKET_STRUCTURE,
        "cpdag",
        ("Tsamardinos, Aliferis & Statnikov 2003", "Margaritis 2003"),
    ),
    (
        "inter_iamb",
        "One variable's Markov blanket by interleaved IAMB, which shrinks after every addition. "
        "With target=null, every variable's blanket joined into a CPDAG by the Grow-Shrink "
        "structure learner, or into the skeleton or the moral graph (structure).",
        "markov_blanket",
        _BLANKET_STRUCTURE,
        "cpdag",
        ("Tsamardinos, Aliferis & Statnikov 2003", "Margaritis 2003"),
    ),
    (
        "mmpc",
        "One variable's parents and children, undirected, by max-min parents and children: "
        "conditional-independence tests over subsets of the members so far, with a symmetry "
        "check. With target=null, the skeleton that mmhc searches, or, with structure=cpdag, that "
        "skeleton oriented into a CPDAG by the Grow-Shrink structure learner.",
        "parents_and_children",
        concept_param(
            "structure",
            default="skeleton",
            choices=("skeleton", "cpdag"),
            doc="The graph returned without a target: skeleton = the adjacencies, each variable "
            "joined to its parents and children; cpdag = the skeleton with its colliders "
            "oriented and Meek's rules applied (Margaritis 2003), which costs more. With a "
            "target, the target's set, whatever the value.",
        ),
        "undirected",
        ("Tsamardinos, Brown & Aliferis 2006", "Margaritis 2003"),
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
                graph_kind=_kind,
                doc=_LOCAL_OUTPUT_DOC.format(key=_key),
            ),
            accel="numpy",
            status="supported",
            column_order="relabels",
            tie_break=True,
            params=(_TARGET, _structure, _ALPHA, _INDEP_TEST, _CATEGORICAL, _TEST_SEED),
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
        data=_ENV_DATA,
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read" + _ENV_CPDAG_DOC,
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
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
        "BIC by default, BDeu when every column is categorical, conditional Gaussian when some "
        "are; score_func=bdeu or discrete_bic for categorical data, degen_gauss or cond_gauss for "
        "mixed data. Returns the interventional essential graph + score.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="env",
                    kind="context",
                    doc="The environment of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "Needs targets. Omit both for observational data.",
                    required=False,
                ),
            ),
            layout=_MATRIX_LAYOUT,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            ParamSpec(
                name="targets",
                type="targets",
                default=None,
                doc="The variables intervened on in each environment of env: a JSON object "
                "mapping each label of env to a list of variables, by the data file's header or "
                'by 0-based position, such as {"0": [], "1": ["x"]}; [] marks an '
                "observational environment. Every environment needs an entry. null, with no env, "
                "is observational data.",
                nullable=True,
            ),
            concept_param(
                "score_func",
                doc=_SCORE_DOC + " Each variable is scored on the rows without an intervention "
                "on it.",
            ),
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            _CATEGORICAL,
        ),
        references=("Hauser & Buhlmann 2012", "Cooper & Yoo 1999"),
    )
)
_add(
    MethodSpec(
        name="gnies",
        family="score",
        summary="Greedy noise-interventional equivalence search: learns the variables whose noise "
        "variance changes between environments (unknown targets) together with the graph, for "
        "linear-Gaussian data from two or more environments that share one set of weights. "
        "Returns the interventional essential graph + score; the learned targets are "
        "metadata['changing_variables'].",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="env",
                    kind="context",
                    doc="The environment of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "Two or more environments of 2+ rows each. null takes the environments from "
                    "targets given as a column or a 0/1 matrix.",
                    required=False,
                ),
            ),
            layout=_MATRIX_LAYOUT,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; the targets, known and learned, in "
            "metadata['changing_variables']: the variables whose noise variance differs between "
            "some two environments, in column order; with env, its labels in "
            "metadata['environments']",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            ParamSpec(
                name="targets",
                type="targets",
                default=None,
                doc="The variables known to be intervened on: a JSON object mapping labels of env "
                "to lists of variables, by the data file's header or 0-based position, such as "
                '{"0": [], "1": ["x"]}; null, or a label left out, marks an environment whose '
                "targets are unknown, and the search may add targets there. null for the whole "
                "argument (the default) learns every target.",
                nullable=True,
            ),
            concept_param(
                "penalty_discount",
                doc="Multiplier on the BIC penalty of log(n_samples) per parent and per extra "
                "noise variance of a target. Larger values give sparser graphs and fewer targets; "
                "1.0 with ebic_gamma=0.0 is the standard BIC.",
            ),
            concept_param(
                "ebic_gamma",
                doc="Weight on the extended-BIC term 2 * ebic_gamma * log C(n_variables - 1, "
                "n_parents) added per variable, on parents only, from 0 to 1. Larger values give "
                "sparser graphs; 0.0 is the standard BIC, the paper's score.",
            ),
        ),
        references=("Gamella et al. 2022", "Hauser & Buhlmann 2012"),
    )
)
_add(
    MethodSpec(
        name="xges",
        family="score",
        summary="Extremely greedy equivalence search: GES's moves and scores, deleting an edge "
        "before reversing one and reversing before adding, then retrying each deletion. BIC by "
        "default, BDeu when every column is categorical, conditional Gaussian when some are; "
        "score_func=bdeu or discrete_bic for categorical data, degen_gauss or cond_gauss for mixed "
        "data. Returns a CPDAG + score.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            ParamSpec(
                name="extended_search",
                type="bool",
                default=True,
                doc="Run the extended search: after the greedy search converges, delete each edge "
                "in turn, forbid adding it back, search again, and keep a better result. "
                "false returns the converged graph.",
            ),
            _CATEGORICAL,
        ),
        references=("Nazaret & Blei 2024",),
    )
)
_add(
    MethodSpec(
        name="ages",
        family="score",
        summary="Aggregative greedy equivalence search for linear-Gaussian data with weak edges: "
        "the CPDAGs GES returns for every penalty from penalty_discount up, overlaid on the CPDAG "
        "of that minimum penalty, so the graph keeps the GES skeleton and can direct more of its "
        "edges. Standard linear-Gaussian BIC only. Returns a partially directed graph, tagged "
        "cpdag.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="the aggregated partially directed graph (APDAG), a maximally oriented one, as a "
            "CPDAG narrowed by background knowledge is; metadata['path_penalties'] lists the "
            "penalty_discount of each CPDAG of the solution path and metadata['path_aggregated'] "
            "whether its orientations were taken",
        ),
        accel="numpy",
        status="experimental",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "penalty_discount",
                doc="Multiplier on the BIC penalty of log(n) per parent, the minimum penalty of "
                "the solution path: the result has the skeleton of the ges CPDAG with this "
                "penalty_discount, ebic_gamma=0.0, and no gate on a node's parents. 1.0 is the "
                "standard BIC, the paper's minimum.",
            ),
        ),
        references=("Eigenmann, Nandy & Maathuis 2017", "Chickering 2002", "Meek 1995"),
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
        "when set; n_starts above 1 climbs again from the best DAG after random edge changes. "
        "Returns the CPDAG of the best DAG and its score.",
        data=_ENV_DATA,
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read" + _RESTART_DOC + _ENV_CPDAG_DOC,
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            concept_param(
                "max_iter",
                default=None,
                doc="Most moves each climb takes, at least 1; null sets no limit (the search still "
                "ends). Reaching a set limit warns with SearchLimitWarning.",
                nullable=True,
            ),
            _RESTART_SEED,
            _N_STARTS,
            _N_PERTURBATIONS,
            _CATEGORICAL,
        ),
        references=(
            "Chickering, Geiger & Heckerman 1995",
            "Scutari 2010",
            "Scutari, Graafland & Gutierrez 2019",
        ),
    )
)
_add(
    MethodSpec(
        name="tabu",
        family="score",
        summary="Tabu search over DAGs: hill climbing that, at a local optimum, takes the best "
        "non-improving move, never returning to a recently visited DAG, and keeps the best DAG "
        "seen. With one start and the same settings its score is never worse than hc's. BIC by "
        "default, BDeu when every column is categorical, conditional Gaussian when some are; "
        "score_func=bdeu or discrete_bic for categorical data, degen_gauss or cond_gauss for "
        "mixed data; n_starts above 1 searches again from the best DAG after random edge changes. "
        "Returns the CPDAG of the best DAG and its score.",
        data=_ENV_DATA,
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read" + _RESTART_DOC + _ENV_CPDAG_DOC,
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
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
                doc="Most moves each search takes, improving or not, at least 1; null sets no "
                "limit (the search still ends). Reaching a set limit warns with "
                "SearchLimitWarning.",
                nullable=True,
            ),
            _RESTART_SEED,
            _N_STARTS,
            _N_PERTURBATIONS,
            _CATEGORICAL,
        ),
        references=(
            "Glover 1986",
            "Russell & Norvig 2009",
            "Scutari 2010",
            "Chickering, Geiger & Heckerman 1995",
            "Scutari, Graafland & Gutierrez 2019",
        ),
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
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            _CATEGORICAL,
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
        accel="torch",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note=(
            "Repeats on one machine with one torch version; on another platform the graph "
            "can differ."
        ),
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
            ),
            ParamSpec(
                name="use_moral_graph",
                type="bool",
                default=True,
                doc="false makes every pair a candidate edge.",
            ),
            concept_param("standardize"),
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
                doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh "
                "random one, so a fit repeats. Seed for the torch optimizer; metadata['seed'] "
                "records the seed used.",
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
        accel="numpy",
        status="supported",
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
                minimum=1,
                exclusive_minimum=True,
                type="float",
                default=1e16,
                doc="The fit stops when the penalty weight on the acyclicity function reaches "
                "this value, above 1, the weight the fit starts from.",
            ),
            *_CONTINUOUS_PATH_PARAMS,
            concept_param("standardize"),
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
            concept_param("standardize"),
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
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _ALPHA,
            _INDEP_TEST,
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            _HYBRID_MAX_ITER,
            _CATEGORICAL,
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
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            ParamSpec(
                name="restrict",
                type="str",
                default="mmpc",
                doc="The restrict phase: mmpc = each variable in the other's MMPC parents and "
                "children; iamb and inter_iamb = each in the other's Markov blanket and not "
                "separated by a subset of the smaller blanket (their skeleton); pc = PC's "
                "skeleton.",
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
            _EQUIVALENT_SAMPLE_SIZE,
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
        data=_ENV_DATA,
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read" + _ENV_CPDAG_DOC,
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            _SEED,
            _N_STARTS,
            _CATEGORICAL,
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
        data=_ENV_DATA,
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="score in metadata['score']; metadata['categorical'] lists the categorical columns "
            "a mixed-data setting read" + _ENV_CPDAG_DOC,
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            _SCORE_FUNC,
            _PENALTY_DISCOUNT,
            _EBIC_GAMMA,
            _EQUIVALENT_SAMPLE_SIZE,
            ParamSpec(
                name="grasp_depth",
                minimum=1,
                type="int",
                default=3,
                doc="Maximum number of tucks in one chain of moves, at least 1; larger depths "
                "search more chains and run longer.",
            ),
            _SEED,
            _N_STARTS,
            _CATEGORICAL,
        ),
        references=("Lam, Andrews & Ramsey 2022",),
    )
)
#: The parent cap of the BOSS or GRaSP phase of boss_fci and grasp_fci.
_PERMUTATION_FCI_MAX_PARENTS = concept_param(
    "max_parents",
    doc="A cap on every node's parents in the BOSS or GRaSP phase, at least 0. The test phase only "
    "removes edges from that CPDAG; the cap does not bound the arrowheads at a node of the PAG. "
    "null sets no cap.",
)
#: The edges of the BOSS or GRaSP CPDAG that the adjacency search of boss_fci and grasp_fci tests.
_PERMUTATION_FCI_EDGE_REMOVAL = concept_param(
    "edge_removal",
    doc="Which edges of the BOSS or GRaSP CPDAG the adjacency search may remove: all = every edge; "
    "triangles = only the edges in a triangle of the CPDAG, whose ends share a neighbor in it "
    "(Ogarrio, Spirtes and Ramsey 2016, Section 5). An edge in no triangle then stays and still "
    "conditions the tests of the other pairs.",
)
#: The seed of boss_fci and grasp_fci: the order search's and a randomized test's.
_PERMUTATION_FCI_SEED = concept_param(
    "seed",
    doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh random one, "
    "so a fit repeats. Seed for the random order in which the BOSS or GRaSP phase visits "
    "variables, for GRaSP's random starting order, and for the random draws of rcot, cmiknn, and "
    "gpdc.",
)
for _name, _summary, _own, _refs in (
    (
        "boss_fci",
        "GFCI with BOSS in place of GES: BOSS adjacencies, pruned by independence tests and "
        "oriented with FCI's rules. Allows hidden common causes like FCI, with BOSS's search for "
        "highly connected variables. Returns a PAG.",
        (),
        (
            "Ramsey, Andrews & Spirtes 2025",
            "Ogarrio, Spirtes & Ramsey 2016",
            "Andrews, Ramsey, Sanchez Romero, Camchong & Kummerfeld 2023",
            "Zhang 2008",
        ),
    ),
    (
        "grasp_fci",
        "GFCI with GRaSP in place of GES: GRaSP adjacencies, pruned by independence tests and "
        "oriented with FCI's rules. Allows hidden common causes like FCI. Returns a PAG.",
        (
            ParamSpec(
                name="grasp_depth",
                minimum=1,
                type="int",
                default=3,
                doc="Maximum number of tucks in one chain of moves of the GRaSP phase, at least 1; "
                "larger depths search more chains and run longer.",
            ),
        ),
        (
            "Ramsey, Andrews & Spirtes 2025",
            "Ogarrio, Spirtes & Ramsey 2016",
            "Lam, Andrews & Ramsey 2022",
        ),
    ),
):
    _add(
        MethodSpec(
            name=_name,
            family="permutation",
            summary=_summary,
            data=_ENV_DATA,
            output=OutputSpec(
                structure_type="graph",
                graph_kind="pag",
                doc="metadata['categorical'] lists the categorical columns a mixed-data setting "
                "read; under collider_rule=conservative or majority, metadata['ambiguous_triples'] "
                "lists the [x, z, y] triples the rule left ambiguous, by 0-based position"
                + _ENV_PAG_DOC,
            ),
            accel="numpy",
            status="supported",
            column_order="relabels",
            tie_break=True,
            order_note=(
                '`tie_break="position"` fires the orientation rules one at a time in column order '
                "and records the first separating subset found."
            ),
            params=(
                _SCORE_FUNC,
                _PENALTY_DISCOUNT,
                _EBIC_GAMMA_STANDARD,
                _EQUIVALENT_SAMPLE_SIZE,
                *_own,
                _INDEP_TEST,
                _ALPHA,
                _COLLIDER_RULE,
                _CATEGORICAL,
                _PERMUTATION_FCI_EDGE_REMOVAL,
                concept_param("confounded_index"),
                _PERMUTATION_FCI_SEED,
                _N_STARTS,
                _PERMUTATION_FCI_MAX_PARENTS,
            ),
            references=_refs,
        )
    )
_add(
    MethodSpec(
        name="igsp",
        family="permutation",
        summary="Interventional greedy sparsest permutation: a permutation search over minimal "
        "I-MAPs for data from an observational environment and settings whose intervention "
        "targets are known, minimizing the edges, then the edges the settings contradict. Fisher "
        "z and Gaussian invariance tests by default. Returns the interventional essential graph.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="env",
                    kind="context",
                    doc="The environment of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "Needs targets that map every label, at least one to [], the observational "
                    "rows.",
                    required=False,
                ),
            ),
            layout=_MATRIX_LAYOUT,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="metadata['score'] is the edges of the DAG found; metadata['n_contradicting'] its "
            "I-contradicting edges; metadata['invariance_test'] names the invariance test; "
            "metadata['alpha'] is the level of the conditional-independence tests",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            ParamSpec(
                name="targets",
                type="targets",
                default=None,
                doc="The variables intervened on in each environment of env: a JSON object "
                "mapping each label of env to a list of variables, by the data file's header or "
                'by 0-based position, such as {"0": [], "1": ["x"]}; [] marks an observational '
                "environment. Every environment needs an entry.",
                nullable=True,
            ),
            concept_param(
                "alpha",
                default=None,
                nullable=True,
                doc="Significance level of the conditional-independence tests, and of the "
                "invariance tests when invariance_alpha is null; null uses one over the number "
                "of observational rows.",
            ),
            ParamSpec(
                name="invariance_alpha",
                type="float",
                default=None,
                doc="Significance level of the invariance tests: a variable's distribution "
                "differs between a setting and the observational rows when the test rejects at "
                "it. null uses alpha.",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                nullable=True,
            ),
            _INDEP_TEST,
            concept_param("invariance_test"),
            concept_param(
                "n_starts",
                default=10,
                doc="Number of searches, at least 1, each from a minimum-degree order: the first "
                "breaks ties by column rank, each later one draws them from seed and its index; "
                "returns the best-scoring search.",
            ),
            concept_param(
                "seed",
                doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh "
                "random one, so a fit repeats. Seed for the searches after the first and the "
                "randomized tests.",
            ),
            _CATEGORICAL,
        ),
        references=(
            "Wang, Solus, Yang & Uhler 2017",
            "Yang, Katcoff & Uhler 2018",
            "Hauser & Buhlmann 2012",
        ),
    )
)
_add(
    MethodSpec(
        name="ut_igsp",
        family="permutation",
        summary="Unknown-target interventional greedy sparsest permutation: a permutation search "
        "over minimal I-MAPs for data from an observational environment and settings whose "
        "intervention targets are known in part or not at all, minimizing the edges plus the "
        "targets. Fisher z and Gaussian invariance tests by default. Returns the interventional "
        "essential graph + each setting's targets.",
        data=DataSpec(
            inputs=(
                _DATA_MATRIX,
                InputSpec(
                    name="env",
                    kind="context",
                    doc="The environment of each row: one label per row, (n_samples,) or "
                    "(n_samples, 1), or a column of the data file, which is then not a variable. "
                    "Needs targets that map at least one label to [], the observational rows.",
                    required=False,
                ),
            ),
            layout=_MATRIX_LAYOUT,
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="cpdag",
            doc="metadata['targets'] maps each label of env to its targets, known and learned, "
            "and passes back as targets=; "
            "metadata['score'] is the edges plus the targets of the DAG found; "
            "metadata['invariance_test'] names the invariance test; metadata['alpha'] is the "
            "level of the conditional-independence tests",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            ParamSpec(
                name="targets",
                type="targets",
                default=None,
                doc="The known intervention targets of each environment of env: a JSON object "
                "mapping each label of env to a list of variables, by the data file's header or "
                'by 0-based position, such as {"0": [], "1": ["x"], "2": null}; [] marks an '
                "observational environment, and null, or a label left out, a setting whose "
                "targets are all unknown.",
                nullable=True,
            ),
            concept_param(
                "alpha",
                default=None,
                nullable=True,
                doc="Significance level of the conditional-independence tests, and of the "
                "invariance tests when invariance_alpha is null; null uses one over the number "
                "of observational rows.",
            ),
            ParamSpec(
                name="invariance_alpha",
                type="float",
                default=None,
                doc="Significance level of the invariance tests: a variable is a target of a "
                "setting when the test rejects at it. null uses alpha.",
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                nullable=True,
            ),
            _INDEP_TEST,
            concept_param("invariance_test"),
            concept_param(
                "n_starts",
                default=10,
                doc="Number of searches, at least 1, each from a minimum-degree order: the first "
                "breaks ties by column rank, each later one draws them from seed and its index; "
                "returns the best-scoring search.",
            ),
            concept_param(
                "seed",
                doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh "
                "random one, so a fit repeats. Seed for the searches after the first and the "
                "randomized tests.",
            ),
            _CATEGORICAL,
        ),
        references=("Squires, Wang & Uhler 2020", "Yang, Katcoff & Uhler 2018"),
    )
)

# --- functional causal models (Glymour, Zhang & Spirtes 2019) -------------------------------------
# Linear effects with non-Gaussian noise (LiNGAM).

_add(
    MethodSpec(
        name="direct_lingam",
        family="functional",
        summary="DirectLiNGAM for linear non-Gaussian data: recovers a fully-oriented DAG plus a "
        "causal order and edge weights. Use when noise is non-Gaussian and effects are linear.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="dag", weighted=True, has_ordering=True
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "measure",
                doc="How the order search picks the most exogenous variable at each step; pwling "
                "is the maximum-entropy approximation of the pairwise likelihood ratio, tanh and "
                "skew their first-order approximations by a nonlinear correlation.",
            ),
        ),
        references=("Shimizu et al. 2011", "Hyvarinen & Smith 2013"),
    )
)
_add(
    MethodSpec(
        name="ica_lingam",
        family="functional",
        summary="ICA-based LiNGAM for linear non-Gaussian data: FastICA gives a causal order, "
        "adaptive Lasso the edge weights; returns a DAG, the order, and the weights. Seeded "
        "(ANDREY_SEED, else 0), so repeated runs match. DirectLiNGAM is more accurate on the "
        "published benchmarks.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph", graph_kind="dag", weighted=True, has_ordering=True
        ),
        accel="numpy",
        status="supported",
        column_order="relabels_if_converged",
        tie_break=True,
        order_note=(
            "At few rows per variable, FastICA can reach another solution under a new "
            "column order or on another machine."
        ),
        params=(
            concept_param(
                "seed",
                doc="null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh "
                "random one, so a fit repeats. Seed for FastICA's starting point.",
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
        family="functional",
        summary="DirectLiNGAM fit jointly across >= 2 datasets sharing the same variables: one "
        "shared causal order and a per-group weighted DAG. Returns one result per group.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data",
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
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "measure",
                doc="How the order search picks the most exogenous variable at each step; pwling "
                "is the maximum-entropy approximation of the pairwise likelihood ratio, tanh and "
                "skew their first-order approximations by a nonlinear correlation.",
            ),
        ),
        references=("Shimizu 2012",),
    )
)

# Nonlinear effects with additive noise (Hoyer et al. 2009; Peters et al. 2014).

_add(
    MethodSpec(
        name="cam",
        family="functional",
        summary="Causal additive model (CAM) for nonlinear additive effects with additive noise: "
        "boosting preselects each variable's possible parents, a greedy search adds edges by the "
        "drop in log residual variance of additive spline fits, and smooth-term tests prune them. "
        "Returns a DAG and a causal order. Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            has_ordering=True,
            doc="DAG after pruning; ordering is a causal order of the search's DAG before pruning.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "alpha",
                default=0.001,
                doc="Significance level of the pruning test: a parent stays when its smooth "
                "term's p-value is at most alpha. 0.001 is the paper's value.",
            ),
        ),
        references=(
            "Buhlmann, Peters & Ernest 2014",
            "Hoyer et al. 2009",
            "Peters, Mooij, Janzing & Scholkopf 2014",
        ),
    )
)
_add(
    MethodSpec(
        name="resit",
        family="functional",
        summary="Regression with subsequent independence test (RESIT) for nonlinear effects with "
        "additive noise of any distribution: a causal order from the sink up, each sink the "
        "variable whose additive-model residual an HSIC test finds most independent of the "
        "others, then each parent removed whose removal leaves the residual independent. Returns "
        "a DAG and a causal order. Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            has_ordering=True,
            doc="DAG after the edge removal; ordering is the causal order of the sink search.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "alpha",
                doc="Significance level of the edge-removal tests: a parent goes when the "
                "residual without it is independent of the variable's predecessors at a p-value "
                "above alpha. 0.05 is the paper's value.",
            ),
        ),
        references=("Peters, Mooij, Janzing & Scholkopf 2014", "Gretton et al. 2007"),
    )
)
_add(
    MethodSpec(
        name="score_matching",
        family="functional",
        summary="SCORE for nonlinear effects with additive Gaussian noise: a causal order from the "
        "leaf up, each leaf the variable whose diagonal entry of the score's Jacobian, estimated "
        "by a Stein kernel estimator on the standardized columns, varies least over the rows, then "
        "the complete DAG of the order pruned by smooth-term tests. Returns a DAG and a causal "
        "order. Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            has_ordering=True,
            doc="DAG after pruning; ordering is the causal order of the leaf search.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "alpha",
                default=0.001,
                doc="Significance level of the pruning test: a parent stays when its smooth "
                "term's p-value is at most alpha. 0.001 is the value of the paper's experiments.",
            ),
        ),
        references=(
            "Rolland et al. 2022",
            "Li & Turner 2018",
            "Buhlmann, Peters & Ernest 2014",
        ),
    )
)
_add(
    MethodSpec(
        name="nogam",
        family="functional",
        summary="NoGAM for nonlinear effects with additive noise of any distribution: a causal "
        "order from the leaf up, each leaf the variable whose entry of the score, estimated by a "
        "Stein kernel estimator, is best predicted from the variable's kernel-ridge residual on "
        "the others, on the standardized columns, then the complete DAG of the order pruned by "
        "smooth-term tests. Returns a DAG and a causal order. Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            has_ordering=True,
            doc="DAG after pruning; ordering is the causal order of the leaf search.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "alpha",
                default=0.001,
                doc="Significance level of the pruning test: a parent stays when its smooth "
                "term's p-value is at most alpha. 0.001 is the value of the paper's experiments.",
            ),
        ),
        references=(
            "Montagna et al. 2023",
            "Li & Turner 2018",
            "Buhlmann, Peters & Ernest 2014",
        ),
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
                "pwling is the maximum-entropy approximation of the pairwise likelihood ratio, "
                "tanh and skew their first-order approximations by a nonlinear correlation.",
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
                ),
            ),
            layout="each (n_samples, n_variables); one array per time point",
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            doc="instantaneous-plus-lagged lag_weights; per-time causal orders in metadata.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag: the number of past time points regressed out before the "
                "instantaneous fit, at least 1.",
            ),
            concept_param(
                "measure",
                doc="How the instantaneous order search picks the most exogenous variable; "
                "pwling is the maximum-entropy approximation of the pairwise likelihood ratio, "
                "tanh and skew their first-order approximations by a nonlinear correlation.",
            ),
        ),
        references=("Kadowaki, Shimizu & Washio 2013",),
    )
)
_add(
    MethodSpec(
        name="pcmci",
        family="temporal",
        summary="PCMCI for a single multivariate time series: lagged causal links, each tested "
        "given the parents of both ends, which controls false links from autocorrelation. No "
        "same-step links. Returns a temporal structure.",
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
                name="pc1_alpha",
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
            ),
            concept_param(
                "fdr_method",
                doc="How the MCI p-values are corrected for the number of links tested: none "
                "keeps them as tested; fdr_bh replaces them with their Benjamini-Hochberg "
                "adjusted values over all the links together, and a link is kept when its "
                "adjusted p-value is at most alpha.",
            ),
            _INDEP_TEST,
            _CATEGORICAL,
            _TEST_SEED,
            concept_param(
                "max_depth",
                doc="The maximum conditioning-set size of every test of the PC1 step, which "
                "selects the lagged parents, at least 0: 0 runs only unconditional tests, and null "
                "sets no limit. The MCI tests condition on the selected parents of both ends, "
                "however many.",
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
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note='With `collider_rule="sepsets"`, the orientations can follow the column order.',
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
            ),
            concept_param(
                "fdr_method",
                doc="How the p-values of the lagged links are corrected for the number of links "
                "tested: none keeps them as tested; fdr_bh replaces them with their "
                "Benjamini-Hochberg adjusted values over all the lagged links together. The "
                "links do not change: the skeleton phase decides them at alpha.",
            ),
            concept_param(
                "collider_rule",
                default="majority",
                doc="How a same-step collider is judged: sepsets = the middle is not in the one "
                "separating set found; majority and conservative = at least one subset separates "
                "the ends and fewer than half (majority) or none (conservative) of them hold the "
                "middle. The default majority does not depend on the order of the variables.",
            ),
            _INDEP_TEST,
            _CATEGORICAL,
            _TEST_SEED,
            concept_param(
                "max_p_global",
                doc="The most conditions a skeleton or collider test takes beyond the default "
                "conditions, the lagged parents of both ends, at least 0; max_depth caps the "
                "lagged phase. null sets no limit. A cap bounds the number of "
                "tests and can keep links that a larger conditioning set would remove.",
            ),
            concept_param(
                "max_depth",
                doc="The maximum conditioning-set size of every test of the lagged phase, PC1 as "
                "in pcmci, at least 0: 0 runs only unconditional tests, and null sets no limit. "
                "The later phases add the lagged parents of both ends to every test, which "
                "max_p_global caps.",
            ),
        ),
        references=("Runge 2020",),
    )
)
_add(
    MethodSpec(
        name="jpcmciplus",
        family="temporal",
        summary="J-PCMCI+ for time series from several datasets that differ through measured or "
        "hidden contexts: lagged and same-step causal links between the system variables, and "
        "the context links. Returns a temporal structure.",
        data=DataSpec(
            inputs=(
                InputSpec(
                    name="data",
                    kind="panel",
                    doc="One time series per dataset, same columns, row = step (repeat the flag).",
                ),
            ),
            layout="each (n_samples, n_variables); one array per dataset, row = time step",
            steps=True,
        ),
        output=OutputSpec(
            structure_type="temporal",
            graph_kind=None,
            doc="a cpdag at lag 0 and lag graphs for lags 1..tau_max over the system variables; "
            "p_values, statistics, the lagged parents, conflicting and ambiguous orientations, "
            "context_links, and the dummies' links in metadata.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        order_note='With `collider_rule="sepsets"`, the orientations can follow the column order.',
        params=(
            ParamSpec(
                name="time_contexts",
                type="str_list",
                default=None,
                doc="The columns that hold time contexts, each the same in every dataset at each "
                "step: column names or 0-based positions. null lists none.",
                nullable=True,
            ),
            ParamSpec(
                name="space_contexts",
                type="str_list",
                default=None,
                doc="The columns that hold space contexts, each constant within a dataset: column "
                "names or 0-based positions. null lists none.",
                nullable=True,
            ),
            ParamSpec(
                name="time_dummy",
                type="bool",
                default=None,
                doc="Add the one-hot dummy of the time step, which stands in for hidden time "
                "contexts and needs datasets of one length, aligned in time, with more than one "
                "step after the first 2 * tau_max. null adds it when the datasets allow it, true "
                "requires it and raises when they do not, and false leaves it out.",
                nullable=True,
            ),
            concept_param(
                "tau_max",
                doc="The largest lag tested; links at lags 0 to tau_max are tested.",
            ),
            concept_param(
                "alpha",
                default=0.01,
                doc="Significance level of every test; a link is removed when a test gives it a "
                "p-value above this. The default is pcmciplus's.",
            ),
            concept_param(
                "fdr_method",
                doc="How the p-values of the lagged links between system variables are corrected "
                "for the number of links tested: none keeps them as tested; fdr_bh replaces them "
                "with their Benjamini-Hochberg adjusted values over all those links together. The "
                "links do not change: the skeleton passes decide them at alpha.",
            ),
            concept_param(
                "collider_rule",
                default="majority",
                doc="How a same-step collider between system variables is judged, as in "
                "pcmciplus; a triple can open with a context or a dummy.",
            ),
            concept_param(
                "indep_test",
                choices=CONTINUOUS_INDEP_TESTS,
                doc="Conditional-independence test, as in pcmciplus: a test of continuous data, as "
                "the dummies enter it as numeric one-hot columns. null picks fisherz.",
            ),
            _TEST_SEED,
            concept_param(
                "max_p_global",
                doc="The most conditions a skeleton or collider test takes beyond the default "
                "conditions, the parents of both ends, at least 0; max_depth caps the lagged "
                "phase. null sets no limit. A cap bounds the number of tests and can keep links "
                "that a larger conditioning set would remove.",
            ),
            concept_param(
                "max_depth",
                doc="The maximum conditioning-set size of every test of the lagged phase, PC1 as "
                "in pcmci, at least 0: 0 runs only unconditional tests, and null sets no limit. "
                "The later passes add the parents of both ends to every test, which max_p_global "
                "caps.",
            ),
        ),
        references=("Gunther et al. 2023", "Runge 2020"),
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
            ),
            ParamSpec(
                name="n_preliminary_iterations",
                minimum=0,
                type="int",
                default=1,
                doc="Preliminary rounds before the final one; each passes the parents it found "
                "to the next as conditions.",
            ),
            _INDEP_TEST,
            _CATEGORICAL,
            _TEST_SEED,
            concept_param(
                "max_p_global",
                doc="The most conditions a test takes beyond the default conditions, the parents "
                "of both ends, at least 0; max_depth elsewhere counts the whole "
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
        summary="Granger causality for a single multivariate time series: lagged effects only, "
        "a t-test on each lagged coefficient of a vector autoregression at the level alpha. "
        "Returns a temporal structure.",
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
            doc="lag_weights at lags 1..tau_max (the regression coefficients of the significant "
            "links, no lag 0); no ordering.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag: each regression uses the past tau_max time steps, at least "
                "1.",
            ),
            _ALPHA,
            concept_param(
                "fdr_method",
                doc="How the p-values of the lagged links are corrected for the number of links "
                "tested: none keeps them as tested; fdr_bh replaces them with their "
                "Benjamini-Hochberg adjusted values over all the lagged links together, and a "
                "link is kept when its adjusted p-value is at most alpha.",
            ),
        ),
        references=("Granger 1969", "Lutkepohl 2005"),
    )
)
_add(
    MethodSpec(
        name="dynotears",
        family="temporal",
        summary="DYNOTEARS for a single multivariate time series: a linear structural vector "
        "autoregression, same-step and lagged effects, by least squares with an l1 penalty under "
        "the NOTEARS acyclicity constraint on the same-step weights. Returns a weighted temporal "
        "structure. Deterministic; no torch.",
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
            doc="a dag at lag 0 and lag graphs for lags 1..tau_max, with their lag_weights; "
            "h_final and edges_removed in metadata.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        params=(
            concept_param(
                "tau_max",
                doc="The largest lag: each step is fit on the same step and the tau_max steps "
                "before it, at least 1.",
            ),
            concept_param(
                "lambda1",
                default=0.05,
                doc="Weight of the l1 penalty on every same-step and lagged weight, at least 0. "
                "0.05 is the paper's value for 500 time steps.",
            ),
            concept_param(
                "w_threshold",
                doc="A same-step edge is kept when its weight magnitude reaches this value, at "
                "least 0.",
            ),
            ParamSpec(
                name="lag_threshold",
                minimum=0,
                type="float",
                default=0.1,
                doc="A lagged edge is kept when its weight magnitude reaches this value, at least "
                "0. 0.1 is the paper's value for 500 time steps.",
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
                doc="The fit stops when the acyclicity function of the same-step weights falls to "
                "this value, above 0.",
            ),
            ParamSpec(
                name="rho_max",
                minimum=1,
                exclusive_minimum=True,
                type="float",
                default=1e16,
                doc="The fit stops when the penalty weight on the acyclicity function reaches "
                "this value, above 1, the weight the fit starts from.",
            ),
            concept_param("standardize"),
        ),
        references=(
            "Pamfil et al. 2020",
            "Zheng, Aragam, Ravikumar & Xing 2018",
            "Deng, Bello, Ravikumar & Aragam 2024",
        ),
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
_add(
    MethodSpec(
        name="loci",
        family="pairwise",
        summary="Location-scale-noise-model pairwise direction test: decides whether column 0 "
        "causes column 1, the reverse, or neither, from a two-column table.",
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
                doc="How the two p-values decide the edge under criterion independence. null: the "
                "direction with the larger p-value, or with the smaller HSIC statistic when the "
                "p-values are equal. A level such as 0.05: an edge only when exactly one "
                "direction's p-value is above it. Must be null under criterion likelihood.",
                nullable=True,
                minimum=0,
                maximum=1,
                exclusive_minimum=True,
                exclusive_maximum=True,
                concept="alpha",
            ),
            ParamSpec(
                name="criterion",
                type="str",
                default="independence",
                choices=("independence", "likelihood"),
                doc="independence = an HSIC test of each direction's scaled residual against its "
                "proposed cause decides (LOCI-H); likelihood = the direction whose model has the "
                "larger likelihood (LOCI-M).",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="dag",
            doc="two-node graph with the decided edge, or none; metadata log_likelihoods maps "
            "forward (x->y) and backward (y->x) to each model's mean log-likelihood, and under "
            "criterion independence p_values and statistics map them to the p-value and the HSIC "
            "statistic, divided by its mean under independence, of each test.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        references=("Immer et al. 2023", "Gretton et al. 2008", "Mooij et al. 2016"),
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
        accel="numpy",
        status="supported",
        column_order="can_change",
        tie_break=True,
        order_note=(
            "Reordering whole clusters relabels the result; reordering the columns inside a "
            "cluster can change the latents' order."
        ),
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
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        references=("Tashiro et al. 2014",),
    )
)

_add(
    MethodSpec(
        name="camuv",
        family="latent",
        summary="Causal additive models with unobserved variables (CAM-UV) for nonlinear additive "
        "effects where some causes are hidden: direct causes as ->, and as o-o the pairs a hidden "
        "common cause or a causal path through a hidden variable joins, whose relation cannot be "
        "determined. Deterministic; no torch.",
        data=DataSpec(inputs=(_DATA_MATRIX,), layout=_MATRIX_LAYOUT),
        params=(
            concept_param(
                "alpha",
                default=0.01,
                doc="Significance level of every HSIC independence test: the sink tests, the "
                "tests that remove a candidate parent, and the tests that mark a pair. 0.01 is "
                "the paper's value.",
            ),
            ParamSpec(
                name="max_depth",
                type="int",
                default=2,
                doc="The maximum conditioning-set size of a sink test, the most variables tested "
                "together as a variable's new candidate parents, at least 1: the search runs over "
                "sets of up to max_depth + 1 variables. Each sink test also regresses on the "
                "variable's candidates found so far, however many. Larger values find more parents "
                "and take longer. The default 2 is the paper's setting, sets of at most 3 "
                "variables.",
                minimum=1,
                concept="max_depth",
            ),
        ),
        output=OutputSpec(
            structure_type="graph",
            graph_kind="pag",
            doc="PAG over the observed variables; o-o joins a pair through a hidden variable, a "
            "common cause or a causal path, metadata latent_pairs lists those pairs, and "
            "cycle_edges the edges dropped from directed cycles.",
        ),
        accel="numpy",
        status="supported",
        column_order="relabels",
        tie_break=True,
        references=("Maeda & Shimizu 2021", "Gretton et al. 2007"),
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
    "The restrict phase keeps the pair of each required edge without a test and conditions the "
    "tests of the other pairs on it, and never keeps a pair the knowledge makes non-adjacent; the "
    "score search considers the pairs it keeps. "
)
_RESTRICT_RULE = (
    "With restrict=mmpc or pc, the restrict phase keeps the pair of each required edge without a "
    "test and conditions the tests of the other pairs on it, and never keeps a pair the knowledge "
    "makes non-adjacent; with restrict=iamb or inter_iamb it reads the data alone. The score "
    "search considers the pairs the restrict phase keeps and the pairs of the required edges, "
    "less the pairs the knowledge makes non-adjacent. "
)
_PARENT_RULE = "Each variable's parents exclude the forbidden ones and include the required ones. "
_ORDER_RULE = "Causal orders put the cause of each required edge, and every earlier tier, first. "
_MASK_RULE = (
    "Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are held at "
    "zero in the fit; a required edge leaves the penalty and the threshold, and the cycle repair "
    "never removes it. "
)
_GATE_RULE = (
    "The candidate edges are the pairs the moral-graph screen keeps and the pairs of the required "
    "edges; the screen reads the data alone. Forbidden edges, edges into an earlier tier, and "
    "pairs outside the allowed set have their gates held at zero; a required edge's gate is held "
    "at one, so the penalty and the thresholds leave it, and the cycle repair never removes it."
)
_MEMBER_RULE = (
    "Each end of a required edge is in the other's set from the start, and no test removes it. "
)
_BLANKET_ORIENT_RULE = (
    "Under structure=cpdag, every edge the knowledge decides is directed before the colliders, "
    "which never reverse it, and Meek's rules R1-R4 follow; the undirected structures do not read "
    "tiers or edges forbidden one way. With target or an undirected structure, required_paths and "
    "forbidden_paths raise NotImplementedError."
)
_BLANKET_RULE = (
    _MEMBER_RULE
    + "A pair the knowledge makes non-adjacent stays a candidate, since a child's other parent "
    "belongs to the blanket, and is left out of the result. " + _BLANKET_ORIENT_RULE
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
_ENV_FREE = "With env, the environment index is unconstrained."
_KNOWLEDGE_HOW: dict[str, str] = {
    "pc": _SKELETON_RULE + _CLASS_RULE,
    "cdnod": _SKELETON_RULE + _CLASS_RULE + " The domain index is unconstrained.",
    "fci": _SKELETON_RULE + _PAG_MARKS,
    "fci_jci": _SKELETON_RULE + _PAG_MARKS + "The environment index is unconstrained.",
    "gfci": "The GES phase moves as ges does; the test phase starts from the adjacencies GES found "
    "and never removes a required pair. " + _PAG_MARKS + _ENV_FREE,
    "boss_fci": "The BOSS phase searches as boss does; the test phase starts from the adjacencies "
    "BOSS found and never removes a required pair. " + _PAG_MARKS + _ENV_FREE,
    "grasp_fci": "The GRaSP phase searches as grasp does; the test phase starts from the "
    "adjacencies GRaSP found and never removes a required pair. " + _PAG_MARKS + _ENV_FREE,
    "ges": _MOVE_RULE + _CLASS_RULE + " " + _ENV_FREE,
    "gies": _MOVE_RULE + _CLASS_RULE,
    "gnies": _MOVE_RULE + _CLASS_RULE,
    "xges": _MOVE_RULE + _CLASS_RULE,
    "hc": _MOVE_RULE + _CLASS_RULE + " " + _ENV_FREE,
    "tabu": _MOVE_RULE + _CLASS_RULE + " " + _ENV_FREE,
    "mmhc": _CANDIDATE_RULE + _MOVE_RULE + _CLASS_RULE,
    "rsmax2": _RESTRICT_RULE + _MOVE_RULE + _CLASS_RULE,
    "boss": _PARENT_RULE + _ORDER_RULE + _CLASS_RULE + " " + _ENV_FREE,
    "grasp": _PARENT_RULE + _ORDER_RULE + _CLASS_RULE + " " + _ENV_FREE,
    **dict.fromkeys(
        ("igsp", "ut_igsp"),
        "Every DAG the search visits keeps the required edges and holds no forbidden one. "
        + _ORDER_RULE
        + _CLASS_RULE,
    ),
    "exact_search": _PARENT_RULE,
    "iamb": _BLANKET_RULE,
    "inter_iamb": _BLANKET_RULE,
    "mmpc": _MEMBER_RULE
    + "A pair the knowledge makes non-adjacent is never a candidate. "
    + _BLANKET_ORIENT_RULE,
    "calm": _GATE_RULE,
    "notears": _MASK_RULE,
    "dagma": _MASK_RULE,
    "cam": "Forbidden edges, edges into an earlier tier, and pairs outside the allowed set are "
    "never possible parents; the boosting that preselects the possible parents reads the data "
    "alone. The search starts from the required edges, and the pruning never removes one. "
    + _ORDER_RULE.rstrip(),
    "resit": "A variable is the next sink only when no remaining variable is in a later tier or is "
    "the effect of a required edge out of it. Forbidden edges, edges into an earlier tier, and "
    "pairs outside the allowed set are in no regression and never in the result; each test still "
    "reads every remaining variable or predecessor. The edge removal never removes a required "
    "edge.",
    "score_matching": "A variable is the next leaf only when no remaining variable is in a later "
    "tier or is the effect of a required edge out of it. Forbidden edges, edges into an earlier "
    "tier, and pairs outside the allowed set are in no pruning fit and never in the result; the "
    "score estimate still reads every remaining variable. The pruning never removes a required "
    "edge.",
    "nogam": "A variable is the next leaf only when no remaining variable is in a later tier or is "
    "the effect of a required edge out of it. Forbidden edges, edges into an earlier tier, and "
    "pairs outside the allowed set are in no pruning fit and never in the result; the score "
    "estimate and the residual regressions still read every remaining variable. The pruning never "
    "removes a required edge.",
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
    "jpcmciplus": _LINK_RULE + "It reads the knowledge over the system variables, by name or by "
    "position among them, as pcmciplus does; the contexts bring their own rule, that no system "
    "variable causes a context and no two contexts are linked.",
    "varma_lingam": _LINK_RULE + "Lags run over the autoregressive part, 0 to order[0]. At lag 0 "
    "the innovations' DirectLiNGAM honors the knowledge as direct_lingam does. With prune=true "
    "each regression drops forbidden lagged regressors and keeps required ones unpenalized; with "
    "prune=false every lagged weight stays, so a forbidden lagged link raises.",
    "granger": _LINK_RULE + "Lags run from 1 to tau_max; a required link needs its lag, and "
    "same-step links are not learned. Each regression drops forbidden lagged regressors and keeps "
    "required ones unpenalized.",
    "dynotears": _LINK_RULE + "Lags run from 0 to tau_max. Forbidden links, links into an "
    "earlier tier, and pairs outside the allowed set are held at zero in the fit; a required "
    "link leaves the penalty and the threshold, and the cycle repair of the same-step graph "
    "never removes it.",
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
    "camuv": "Forbidden edges, edges into an earlier tier, and pairs outside the allowed set never "
    "make a variable a candidate parent, so a variable is not tried as the sink of a set that "
    "holds a forbidden parent of it. A required edge's cause is a candidate parent from the start "
    "and no step removes it; its pair is never marked. A pair forbidden both ways is never marked; "
    "on a marked pair, a forbidden i -> j whose reverse is not forbidden, such as an edge into an "
    "earlier tier, puts an arrowhead at i.",
    "lpcmci": _LINK_RULE + "A forbidden lagged link, and a same-step pair forbidden both ways, is "
    "never linked; a same-step link forbidden one way has an arrowhead at its cause. A required "
    "link is -->, and no test or rule removes it. No rule changes a mark the knowledge sets.",
}
#: The methods whose result is a CPDAG (for iamb, inter_iamb, and mmpc, with structure="cpdag" and
#: no target, else the path parts raise): they honor the path parts by restricting the class they
#: found (Fang et al. 2025).
_PATH_METHODS = (
    "pc",
    "cdnod",
    "ges",
    "xges",
    "hc",
    "tabu",
    "mmhc",
    "rsmax2",
    "boss",
    "grasp",
    "exact_search",
    "iamb",
    "inter_iamb",
    "mmpc",
)
_PATH_RULE = (
    " With required_paths and forbidden_paths, the result is the MPDAG of the DAGs in the class "
    "found that meet them, and metadata ancestor_clauses lists the clauses no MPDAG can show (Fang "
    "et al. 2025). A claim no such DAG meets is dropped: taken in order, required_paths then "
    "forbidden_paths, each claim is kept when some DAG meets it with those kept before it. "
    "metadata unmet_knowledge lists the dropped claims, and the fit warns with KnowledgeWarning."
)
#: The searches that direct edges by tests can return a graph with no consistent extension.
_NO_EXTENSION_RULE = (
    " A graph found with no consistent extension stands for no DAG: there a claim on a pair joined "
    "by an undirected edge directs that edge (Fang and He 2020), Meek's rules follow (Meek 1995), "
    "a claim on a pair that is not adjacent, against a directed edge, or in conflict with other "
    "claims is dropped, and metadata edge_claims lists the claims applied to an edge."
)
_PAG_PATHS = (
    " On an adjacent pair, required_paths (a, b) makes the edge a -> b and forbidden_paths (a, b) "
    "puts an arrowhead at a, before the rules; the tests can still remove the pair. No test "
    "removes a confounded pair. A required path the forbidden edges or the tiers deny on an "
    "adjacent pair sets no mark. A claim on a pair the PAG leaves non-adjacent sets no mark, and "
    "one the PAG does not meet is dropped: a required path without the edge a -> b or, on a "
    "non-adjacent pair, a possibly directed path, or a forbidden path with a directed path. "
    "metadata unmet_knowledge lists the dropped claims, and the fit warns with KnowledgeWarning. "
    "The marks and the checks assume no selection bias, as an "
    "arrowhead at a says that a is an ancestor of neither b nor the selection variables "
    "(Richardson and Spirtes 2002)."
)
_SCORE_PAG_PATHS = " The test phase also starts from every confounded pair." + _PAG_PATHS
#: boss_fci and grasp_fci: their order search reads required_paths as it reads the tiers.
_ORDER_PAG_PATHS = (
    " The order search puts the cause of each required_paths pair before its effect in every order "
    "it visits." + _SCORE_PAG_PATHS
)
_ORDER_PATHS = (
    " With required_paths and forbidden_paths, the order search follows Shimizu et al. (2011, "
    "Section 3.4): when the claims show that no remaining variable is an ancestor of some of "
    "them, the next variable is one of those; otherwise it is none with a claimed ancestor among "
    "the remaining. The regression drops each forbidden path as a parent. A claim the DAG does not "
    "meet is dropped: metadata unmet_knowledge lists it, and the fit warns with KnowledgeWarning."
)
_LAGGED_PATHS = (
    " required_paths [a, b, lag] puts a tail at a and an arrowhead at b on that link, and "
    "forbidden_paths [a, b, lag] an arrowhead at a; the tests can still remove the link. "
    "confounded [a, b, lag] keeps the link from every removal. Each claim names its lag. A claim "
    "on a link the PAG lacks sets no mark, and one the PAG does not meet is dropped: a required "
    "path with no possibly directed path, or a forbidden path with a directed path. metadata "
    "unmet_knowledge lists the dropped claims as [part, a, b, lag], and the fit warns with "
    "KnowledgeWarning."
)
_PAG_PARTS: tuple[KnowledgePart, ...] = (*PATH_PARTS, "confounded")
#: The methods that read the path parts and confounded in the search, with the sentence that says
#: how.
_ENGINE_PATHS: dict[str, tuple[tuple[KnowledgePart, ...], str]] = {
    "fci": (_PAG_PARTS, _PAG_PATHS),
    "fci_jci": (_PAG_PARTS, _PAG_PATHS),
    "gfci": (_PAG_PARTS, _SCORE_PAG_PATHS),
    "boss_fci": (_PAG_PARTS, _ORDER_PAG_PATHS),
    "grasp_fci": (_PAG_PARTS, _ORDER_PAG_PATHS),
    "lpcmci": (_PAG_PARTS, _LAGGED_PATHS),
    "direct_lingam": (PATH_PARTS, _ORDER_PATHS),
    "multi_group_direct_lingam": (PATH_PARTS, _ORDER_PATHS),
}
_MAX_DEPTH = concept_param("max_depth")

#: How missing values are handled, on every method the concept's template fits, after its
#: categorical columns where it has them.
_MISSING = concept_param("missing")
for _name in CONCEPTS["missing"].conforming:
    _params = REGISTRY[_name].params
    _cut = next((i + 1 for i, p in enumerate(_params) if p.name == "categorical"), len(_params))
    REGISTRY[_name] = REGISTRY[_name].model_copy(
        update={"params": (*_params[:_cut], _MISSING, *_params[_cut:])}
    )

#: Structural EM on the searches whose fit reads only a covariance and the row count, after their
#: categorical columns where they have them: supported in EM_SUPPORTED, experimental in the others.
_MISSING_EM = {
    supported: concept_param(
        "missing",
        choices=(EM,),
        doc=f"How missing values (NaN) are handled: em = {status}structural EM, which alternates "
        "the search on the expected covariance of the data with a linear-Gaussian fit of the "
        "graph it returns, 10 times; it assumes values missing completely at random or at random "
        "given observed values. null refuses missing values.",
    )
    for supported, status in ((True, ""), (False, "experimental "))
}
for _name in EM_METHODS:
    _params = REGISTRY[_name].params
    _cut = next((i + 1 for i, p in enumerate(_params) if p.name == "categorical"), len(_params))
    _missing_em = _MISSING_EM[_name in EM_SUPPORTED]
    REGISTRY[_name] = REGISTRY[_name].model_copy(
        update={"params": (*_params[:_cut], _missing_em, *_params[_cut:])}
    )

#: The path and confounder parts a method honors as supported: the method ran in the lanes of
#: benchmarks/knowledge_paths.py, and the part passed them, against the same fit without the claims
#: and, where a reference package takes the same knowledge, against it. A part a method honors
#: beyond the edge parts and does not list here is experimental: tabu, mmhc, and rsmax2 made more
#: errors than their reference given the required edges, on repeated seeds, and iamb, inter_iamb,
#: and mmpc with forbidden paths alone were farther from the class the claims leave at 50
#: variables, over two seed groups pooled. In pc, cdnod, iamb, inter_iamb, and mmpc the parts were
#: measured with the rule for a graph with no consistent extension, on which a claim acts on an
#: edge and most claims are dropped.
_SUPPORTED_PARTS: dict[str, tuple[KnowledgePart, ...]] = {
    "pc": PATH_PARTS,
    "cdnod": PATH_PARTS,
    "ges": PATH_PARTS,
    "xges": PATH_PARTS,
    "hc": PATH_PARTS,
    "tabu": ("forbidden_paths",),
    "mmhc": ("forbidden_paths",),
    "rsmax2": ("forbidden_paths",),
    "boss": PATH_PARTS,
    "grasp": PATH_PARTS,
    "exact_search": PATH_PARTS,
    "iamb": ("required_paths",),
    "inter_iamb": ("required_paths",),
    "mmpc": ("required_paths",),
    "fci": _PAG_PARTS,
    "fci_jci": _PAG_PARTS,
    "gfci": _PAG_PARTS,
    "boss_fci": _PAG_PARTS,
    "grasp_fci": _PAG_PARTS,
    "lpcmci": _PAG_PARTS,
    "direct_lingam": PATH_PARTS,
    "multi_group_direct_lingam": PATH_PARTS,
}

for _name, _how in _KNOWLEDGE_HOW.items():
    _extra, _rule = (
        (PATH_PARTS, _PATH_RULE) if _name in _PATH_METHODS else _ENGINE_PATHS.get(_name, ((), ""))
    )
    if _name in _PATH_METHODS and REGISTRY[_name].family == "constraint":
        _rule += _NO_EXTENSION_RULE
    _parts = (*EDGE_PARTS, *_extra)
    _knowledge = concept_param(
        "knowledge",
        doc="Background knowledge: required and forbidden edges, the pairs allowed to be "
        f"adjacent, and temporal tiers. Honors {', '.join(_parts)}. {_how}{_rule}",
    )
    _depth = (_MAX_DEPTH,) if _name in CONCEPTS["max_depth"].conforming else ()
    REGISTRY[_name] = REGISTRY[_name].model_copy(
        update={
            "params": (*REGISTRY[_name].params, *_depth, _knowledge),
            "knowledge_extra": _extra,
            "knowledge_experimental": tuple(
                part for part in _extra if part not in _SUPPORTED_PARTS.get(_name, ())
            ),
            "restricts_paths": _name in _PATH_METHODS,
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
    "pc": ("conflicts", "missingness_parents", "corrected_edges"),
    "cdnod": ("conflicts", "index_edges", "changing_variables"),
    "fci_jci": ("index_edges",),
    "icp": (
        "target",
        "parents",
        "p_values",
        "confidence_intervals",
        "accepted_sets",
        "best_p_value",
        "model_rejected",
        "environments_left_out",
    ),
    "iamb": ("target", "markov_blanket"),
    "inter_iamb": ("target", "markov_blanket"),
    "mmpc": ("target", "parents_and_children"),
    **dict.fromkeys(("gies", "xges", "mmhc", "rsmax2"), ("score",)),
    **dict.fromkeys(
        ("ges", "hc", "tabu", "boss", "grasp"), ("score", "index_edges", "changing_variables")
    ),
    **dict.fromkeys(("gfci", "boss_fci", "grasp_fci"), ("index_edges",)),
    "gnies": ("score", "changing_variables", "environments"),
    "ages": ("path_penalties", "path_aggregated"),
    "exact_search": ("dag",),
    "igsp": ("score", "n_contradicting", "invariance_test", "alpha"),
    "ut_igsp": ("targets", "score", "invariance_test", "alpha"),
    "calm": ("h_final", "objective"),
    **dict.fromkeys(("notears", "dagma"), ("h_final", "edges_removed", "path_steps")),
    "multi_group_direct_lingam": ("group_index",),
    "longitudinal_lingam": ("causal_orders",),
    "granger": ("p_values",),
    "dynotears": ("h_final", "edges_removed"),
    "pcmci": ("p_values", "statistics", "parents"),
    "pcmciplus": ("p_values", "statistics", "parents", "conflicts"),
    "jpcmciplus": (
        *("p_values", "statistics", "parents", "conflicts"),
        *("context_links", "index_edges", "time_index_edges", "time_dummy"),
    ),
    "lpcmci": ("links", "p_values", "statistics", "parents", "conflicts"),
    **dict.fromkeys(("anm", "pnl"), ("p_values", "statistics")),
    "loci": ("log_likelihoods", "p_values", "statistics"),
    "gin": ("clusters", "cluster_order", "unordered"),
    "rcd": ("confounded_pairs",),
    "bottom_up_parce_lingam": ("partial_order",),
    "camuv": ("latent_pairs", "cycle_edges"),
}
# The shared keys: the name, the seed of a seeded fit, the changes before each later start, the
# test and the score it used with the score's penalties, the configuration's tie-break rule, the
# column types a mixed-data setting read, the fewest rows a test used and how missing values were
# handled, the record of a structural EM, the triples a collider vote left ambiguous, the clauses
# the path parts of knowledge leave, and the claims a causal order search does not meet.
for _name, _spec in REGISTRY.items():
    _concepts = {p.concept for p in _spec.params}
    _shared = (
        "algorithm",
        *(("seed",) if _spec.seeded else ()),
        *(("n_perturbations",) if "n_perturbations" in _concepts else ()),
        *(("indep_test",) if "indep_test" in _concepts else ()),
        *(("score_func",) if "score_func" in _concepts else ()),
        *(
            k
            for k in ("penalty_discount", "ebic_gamma", "equivalent_sample_size")
            if k in _concepts
        ),
        *(("tie_break",) if _spec.tie_break else ()),
        *(("categorical",) if "categorical" in _concepts else ()),
        *(("rows_tested",) if _spec.data.missing else ()),
        *(("missing",) if "missing" in _concepts else ()),
        *(("em",) if _name in EM_METHODS else ()),
        *(("ambiguous_triples",) if "collider_rule" in _concepts else ()),
        *(("ancestor_clauses", "edge_claims") if _spec.restricts_paths else ()),
        *(("unmet_knowledge",) if "required_paths" in _spec.knowledge else ()),
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
