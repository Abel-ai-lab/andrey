"""Package-wide parameter rules: one :class:`Concept` per problem that many methods share.

A concept has one parameter name, one meaning, and one default convention, and goes to every
method its theory makes suitable. Each concept lists:

- its template :class:`~andrey.spec.models.ParamSpec` (name, type, default, choices, bounds), or
  none for a behavior such as a warning or a metadata key;
- the suitable methods, and the exempt ones with the reason;
- the recorded deviations: each cell that differs today (another name, default, meaning, or
  choices, or the parameter missing) with today's value, the release that removes it, and the
  former names or values that the change keeps working for one release.

A cell with two differences has one record, for the change that comes first; that change deletes
the record or rewrites it to the difference that remains. A deviation with ``fixed_in=None`` is
kept, and its reason says why. The registry takes each conforming cell's parameter from here, and
``tests/unit/test_concepts.py`` checks every concept against every method: a new method fails
there until each concept lists it as suitable or exempt.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import model_validator

from andrey.constraint._skeleton import COLLIDER_RULES
from andrey.core.ci import SHIPPED_INDEP_TESTS, STOCHASTIC_INDEP_TESTS
from andrey.core.score import LINEAR_SCORES, SCORES

from .models import ParamSpec, ParamType, Spec

Release = Literal["0.11.0", "0.12.0"]

#: The score names every score search takes.
SCORE_NAMES = SCORES
#: The score names that shipped before 0.11.0, each with its current name, kept for one release.
SCORE_ALIASES = (
    ("local_score_BIC", "bic"),
    ("local_score_BIC_from_cov", "bic"),
    ("local_score_BDeu", "bdeu"),
    ("local_score_BIC_categorical", "discrete_bic"),
    ("local_score_cond_gauss", "cond_gauss"),
    ("local_score_degen_gauss", "degen_gauss"),
)
#: The largest seed NumPy and torch accept.
SEED_MAX = 2**32 - 1


class Exemption(Spec):
    """A method the concept does not suit, and why."""

    method: str
    reason: str


class Deviation(Spec):
    """One cell that differs from its concept today."""

    method: str
    reason: str
    fixed_in: Release | None  # None: kept, for the reason given
    name: str | None = None  # today's parameter or input name; None: absent
    default: Any = None  # today's default
    choices: tuple[str, ...] | None = None  # today's choices
    observed: str | None = None  # a behavior concept's value today
    aliases: tuple[str, ...] = ()  # former names or values kept for one release by the change
    fragment: str | None = None  # the docstring phrase when the meaning differs; "" checks none


class Concept(Spec):
    """One shared parameter or behavior and the methods it applies to."""

    key: str
    summary: str
    param: ParamSpec | None = None  # the template; None for a behavior
    fragment: str = ""  # a phrase every conforming docstring entry contains
    default_rule: str = ""  # set when each method keeps its own default, saying which
    suitable: tuple[str, ...]
    exempt: tuple[Exemption, ...] = ()
    others_exempt: str = ""  # the reason for every method not listed, when one reason covers all
    deviations: tuple[Deviation, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Concept:
        exempt = [e.method for e in self.exempt]
        if set(exempt) & set(self.suitable) or len(set(exempt)) != len(exempt):
            raise ValueError(f"{self.key}: a method is listed twice")
        methods = [d.method for d in self.deviations]
        if len(set(methods)) != len(methods):
            raise ValueError(f"{self.key}: two deviations for one method")
        if self.param is not None and self.param.concept != self.key:
            raise ValueError(f"{self.key}: the template is not tagged with its concept")
        return self

    def deviation(self, method: str) -> Deviation | None:
        """The method's recorded deviation, if any."""
        return next((d for d in self.deviations if d.method == method), None)

    @property
    def conforming(self) -> tuple[str, ...]:
        """The suitable methods whose cell is the template."""
        return tuple(m for m in self.suitable if self.deviation(m) is None)


def _template(key: str, type: ParamType, default: Any, doc: str, **fields: Any) -> ParamSpec:
    return ParamSpec(name=key, type=type, default=default, doc=doc, concept=key, **fields)


def _exempt(reason: str, *methods: str) -> tuple[Exemption, ...]:
    return tuple(Exemption(method=m, reason=reason) for m in methods)


def _absent(fixed_in: Release, reason: str, *methods: str) -> tuple[Deviation, ...]:
    return tuple(Deviation(method=m, fixed_in=fixed_in, reason=reason) for m in methods)


# Method groups the rules below name.
CI_SEARCHES = ("pc", "fci", "gfci", "cdnod", "iamb", "inter_iamb", "mmpc", "mmhc", "rsmax2")
PCMCI_FAMILY = ("pcmci", "pcmciplus", "lpcmci")
SCORE_SEARCHES = (
    "ges",
    "gies",
    "xges",
    "hc",
    "tabu",
    "exact_search",
    "boss",
    "grasp",
    "mmhc",
    "rsmax2",
    "gfci",
)
_PURE_SCORE = ("ges", "gies", "xges", "hc", "tabu", "exact_search", "boss", "grasp")
_CONTINUOUS = ("notears", "dagma")
_LINGAM = (
    "direct_lingam",
    "ica_lingam",
    "multi_group_direct_lingam",
    "varma_lingam",
    "longitudinal_lingam",
)
_RESIDUAL = ("gin", "rcd", "bottom_up_parce_lingam", "anm", "pnl")
ALL_METHODS = (
    *("pc", "fci", "gfci", "cdnod", "iamb", "inter_iamb", "mmpc"),
    *_PURE_SCORE,
    "calm",
    *_CONTINUOUS,
    "mmhc",
    "rsmax2",
    *_LINGAM,
    *PCMCI_FAMILY,
    "granger",
    *_RESIDUAL,
)

_CONCEPTS = (
    Concept(
        key="alpha",
        summary="The significance level of the test that decides the returned edges, or of the "
        "only test. A second test step's level is <step>_alpha.",
        param=_template(
            "alpha",
            "float",
            0.05,
            "Significance level of the conditional-independence test.",
            minimum=0,
            maximum=1,
            exclusive_minimum=True,
            exclusive_maximum=True,
        ),
        fragment="significance level",
        suitable=(*CI_SEARCHES, *PCMCI_FAMILY, *_RESIDUAL),
        exempt=(
            *_exempt("A score search: no test decides an edge.", *_PURE_SCORE),
            *_exempt("Continuous optimization: no test decides an edge.", *_CONTINUOUS),
            *_exempt(
                "Its edges come from the gated likelihood; its moral-graph screen is a step "
                "level, screen_alpha.",
                "calm",
            ),
            *_exempt(
                "The order comes from an independence measure and the edges from the adaptive "
                "Lasso by BIC: no test at a level.",
                *_LINGAM,
            ),
            *_exempt("A Lasso chosen by cross-validation: no test.", "granger"),
        ),
        deviations=(
            Deviation(
                method="pcmciplus",
                name="alpha",
                default=0.01,
                fixed_in=None,
                reason="0.01 is the level of the method's published experiments (Runge 2020: "
                "tau_max = 5 and alpha = 0.01).",
            ),
            Deviation(
                method="rcd",
                name="alpha",
                default=0.01,
                fixed_in=None,
                reason="0.01 is the level of the method's published experiments (Maeda and Shimizu "
                "2020 set every level to 0.01).",
            ),
            *(
                Deviation(
                    method=m,
                    name="alpha",
                    default=None,
                    fixed_in=None,
                    fragment="p-value",
                    reason="None decides without a level: the direction with the larger p-value.",
                )
                for m in ("anm", "pnl")
            ),
        ),
    ),
    Concept(
        key="indep_test",
        summary="The conditional-independence test. None picks it from the declared column types.",
        param=_template(
            "indep_test",
            "str",
            None,
            "Conditional-independence test; null picks one from the declared column types.",
            choices=SHIPPED_INDEP_TESTS,
            nullable=True,
        ),
        fragment="conditional-independence test",
        suitable=(*CI_SEARCHES, *PCMCI_FAMILY),
        exempt=(
            *_exempt(
                "It tests the independence of residuals as part of its own model.", *_RESIDUAL
            ),
            *_exempt("No conditional-independence test.", *_PURE_SCORE, *_CONTINUOUS, *_LINGAM),
            *_exempt("Its moral-graph screen is part of its own model.", "calm"),
            *_exempt("No conditional-independence test.", "granger"),
        ),
        deviations=tuple(
            Deviation(
                method=m,
                name="indep_test",
                default=None,
                choices=("student_t", "fisherz"),
                fixed_in="0.12.0",
                reason="The partial-correlation tests only; the categorical and nonlinear tests "
                "come in 0.12.0.",
            )
            for m in PCMCI_FAMILY
        ),
    ),
    Concept(
        key="score_func",
        summary="The local score of a score search. None picks it from the declared column types.",
        param=_template(
            "score_func",
            "str",
            None,
            "Local score; null picks one from the declared column types.",
            choices=SCORE_NAMES,
            nullable=True,
        ),
        fragment="score",
        suitable=SCORE_SEARCHES,
        others_exempt="Not a search over parent sets scored by a decomposable score.",
        deviations=tuple(
            Deviation(
                method=m,
                name="score_func",
                default=None,
                choices=LINEAR_SCORES,
                fixed_in="0.12.0",
                reason="The linear-Gaussian BIC only; the categorical scores are published and "
                "come in 0.12.0.",
            )
            for m in ("gies", "xges")
        ),
    ),
    Concept(
        key="penalty_discount",
        summary="The multiplier on the BIC penalty per parameter.",
        param=_template(
            "penalty_discount",
            "float",
            1.0,
            "Multiplier on the BIC penalty of log(n_samples) per parent. Larger values give "
            "sparser graphs; 1.0 with ebic_gamma=0.0 is the standard BIC.",
            minimum=0,
        ),
        fragment="BIC penalty",
        suitable=SCORE_SEARCHES,
        others_exempt="Does not score with BIC.",
    ),
    Concept(
        key="ebic_gamma",
        summary="The weight of the extended-BIC term.",
        param=_template(
            "ebic_gamma",
            "float",
            1.0,
            "Weight on the extended-BIC term 2 * ebic_gamma * log C(n_variables - 1, n_parents) "
            "that the linear-Gaussian BIC adds per variable; it grows with the number of "
            "variables. Larger values give sparser graphs; 0.0 is the standard BIC. Other scores "
            "ignore it.",
            minimum=0,
        ),
        fragment="extended-BIC term",
        suitable=SCORE_SEARCHES,
        others_exempt="Does not score with BIC.",
        deviations=(
            Deviation(
                method="gfci",
                name="ebic_gamma",
                default=0.0,
                fixed_in=None,
                reason="The standard BIC: the extended term lowered its errors only slightly and "
                "raised them on some dense graphs; its docstring gives the measurements.",
            ),
        ),
    ),
    Concept(
        key="lambda1",
        summary="The L1 weight of a continuous objective.",
        param=_template(
            "lambda1",
            "float",
            None,
            "Weight of the l1 penalty of the first fit, at least 0.",
            minimum=0,
        ),
        fragment="l1 penalty",
        default_rule="Each paper's default.",
        suitable=(*_CONTINUOUS, "calm"),
        others_exempt="No continuous objective.",
        deviations=_absent("0.11.0", "Passed through **params.", "calm"),
    ),
    Concept(
        key="w_threshold",
        summary="The weight magnitude below which a continuous fit drops an edge.",
        param=_template(
            "w_threshold",
            "float",
            0.3,
            "An edge is kept when its weight magnitude reaches this value, at least 0.",
            minimum=0,
        ),
        fragment="weight magnitude",
        suitable=(*_CONTINUOUS, "calm"),
        others_exempt="No weighted continuous fit to threshold.",
        deviations=_absent("0.11.0", "Passed through **params.", "calm"),
    ),
    Concept(
        key="max_depth",
        summary="The largest conditioning set of every conditional-independence test.",
        param=_template(
            "max_depth",
            "int",
            None,
            "The maximum conditioning-set size of every conditional-independence test, at least "
            "0: 0 runs only unconditional tests, and null sets no limit. A smaller cap runs fewer "
            "tests and can keep edges that a larger conditioning set would remove.",
            nullable=True,
            minimum=0,
        ),
        fragment="conditioning-set size",
        suitable=(*CI_SEARCHES, "pcmci", "pcmciplus", "rcd"),
        exempt=(
            *_exempt(
                "Every test includes the default conditions; its cap counts beyond them, "
                "max_p_global.",
                "lpcmci",
            ),
            *_exempt(
                "No conditional-independence tests over growing conditioning sets.",
                *_PURE_SCORE,
                "calm",
                *_CONTINUOUS,
                *_LINGAM,
                "granger",
                "gin",
                "bottom_up_parce_lingam",
                "anm",
                "pnl",
            ),
        ),
        deviations=(
            Deviation(
                method="rcd",
                name="max_depth",
                default=2,
                fixed_in=None,
                reason="Maeda and Shimizu set the cap to 2 in their experiments and advise a small "
                "one when there are fewer rows than variables; the minimum is 1, since the "
                "smallest ancestor test regresses one variable on another.",
            ),
        ),
    ),
    Concept(
        key="max_p_global",
        summary="The largest number of conditions beyond the default conditions every test "
        "includes (tigramite's LPCMCI name).",
        param=_template(
            "max_p_global",
            "int",
            None,
            "The most conditions a test takes beyond the default conditions, at least 0; null sets "
            "no limit.",
            nullable=True,
            minimum=0,
        ),
        fragment="beyond the default condition",
        suitable=("lpcmci", "pcmciplus"),
        exempt=_exempt(
            "MCI conditions on the default conditions alone, one test per link; PC1 has none "
            "(max_depth).",
            "pcmci",
        ),
        others_exempt="Its tests include no default conditions to count beyond.",
    ),
    Concept(
        key="max_parents",
        summary="The most parents a node may have in any graph the result represents.",
        param=_template(
            "max_parents",
            "int",
            None,
            "A cap on every node's parents, at least 0; null sets no limit.",
            nullable=True,
            minimum=0,
        ),
        fragment="parents",
        suitable=SCORE_SEARCHES,
        exempt=(
            *_exempt(
                "A constraint-based search: max_depth bounds the same cost.",
                "pc",
                "fci",
                "cdnod",
                "iamb",
                "inter_iamb",
                "mmpc",
            ),
            *_exempt(
                "A cardinality cap is not part of the continuous objective.", "calm", *_CONTINUOUS
            ),
            *_exempt("Parents come from the adaptive Lasso on the order.", *_LINGAM),
            *_exempt("No search over parent sets.", *PCMCI_FAMILY, "granger", *_RESIDUAL),
        ),
        deviations=(
            *_absent(
                "0.12.0",
                "A cap on the represented DAGs interacts with the forward phase's gate.",
                "gies",
                "xges",
                "gfci",
            ),
        ),
    ),
    Concept(
        key="max_iter",
        summary="The cap on the method's main loop, in the method's own unit.",
        param=_template(
            "max_iter",
            "int",
            None,
            "The most iterations of the main loop, at least 1.",
            minimum=1,
        ),
        fragment="at least 1",
        default_rule="None where the loop ends on its own (a score search's moves); the method's "
        "own count where it may not (an optimizer's steps). The docstring names the unit.",
        suitable=(
            "hc",
            "tabu",
            "mmhc",
            "rsmax2",
            *_CONTINUOUS,
            "calm",
            "ica_lingam",
            "varma_lingam",
        ),
        others_exempt="No iterative loop that a cap stops early.",
        deviations=_absent("0.11.0", "The main loop's cap is max_outer, in **params.", "calm"),
    ),
    Concept(
        key="cap_warning",
        summary="Stopping on max_iter rather than on the method's own rule warns "
        "SearchLimitWarning.",
        suitable=(
            "hc",
            "tabu",
            "mmhc",
            "rsmax2",
            *_CONTINUOUS,
            "calm",
            "ica_lingam",
            "varma_lingam",
        ),
        others_exempt="No iterative loop that a cap stops early.",
    ),
    Concept(
        key="tau_max",
        summary="The largest lag; the smallest follows the method's theory.",
        param=_template("tau_max", "int", 1, "The largest lag, at least 1.", minimum=1),
        fragment="largest lag",
        suitable=(*PCMCI_FAMILY, "granger", "longitudinal_lingam"),
        exempt=(
            *_exempt("order=(p, q) is the ARMA order, a different concept.", "varma_lingam"),
            *_exempt(
                "i.i.d. data.",
                *(
                    m
                    for m in ALL_METHODS
                    if m not in (*PCMCI_FAMILY, "granger", "longitudinal_lingam", "varma_lingam")
                ),
            ),
        ),
        deviations=(
            Deviation(
                method="granger",
                name="maxlag",
                default=2,
                fixed_in="0.11.0",
                aliases=("maxlag",),
                fragment="past time steps",
                reason="Another name, and the default 2 moves to 1.",
            ),
            Deviation(
                method="longitudinal_lingam",
                name="n_lags",
                default=1,
                fixed_in="0.11.0",
                aliases=("n_lags",),
                fragment="past time points",
                reason="Another name.",
            ),
        ),
    ),
    Concept(
        key="seed",
        summary="The seed of every random draw; None uses ANDREY_SEED, or 0 when it is unset. A "
        "method that tests independence takes it for the randomized tests: "
        f"{', '.join(STOCHASTIC_INDEP_TESTS)}.",
        param=_template(
            "seed",
            "int",
            None,
            "Seed for the method's random draws. null uses ANDREY_SEED, or 0 when it is unset.",
            nullable=True,
            minimum=0,
            maximum=SEED_MAX,
        ),
        fragment="ANDREY_SEED",
        suitable=(*CI_SEARCHES, "boss", "grasp", "calm", "ica_lingam"),
        exempt=(
            *_exempt(
                "Deterministic: student_t and fisherz draw no random numbers. It becomes suitable "
                "with the nonlinear tests.",
                *PCMCI_FAMILY,
            ),
            *_exempt(
                "Deterministic.",
                *(m for m in _PURE_SCORE if m not in ("boss", "grasp")),
                *_CONTINUOUS,
                "direct_lingam",
                "multi_group_direct_lingam",
                "varma_lingam",
                "longitudinal_lingam",
                "granger",
                *_RESIDUAL,
            ),
        ),
    ),
    Concept(
        key="num_workers",
        summary="Parallelism and the device come from andrey.config or ANDREY_*, never from a "
        "method argument.",
        param=_template(
            "num_workers",
            "int",
            None,
            "Not a method argument: set it with andrey.config(num_workers=...).",
            nullable=True,
        ),
        suitable=(),
        others_exempt="Execution settings come from andrey.config or ANDREY_* only.",
    ),
    Concept(
        key="knowledge",
        summary="Background knowledge, an andrey.Knowledge, every part honored.",
        param=_template(
            "knowledge",
            "knowledge",
            None,
            "Background knowledge: required and forbidden edges, the pairs allowed to be "
            "adjacent, and temporal tiers.",
            nullable=True,
        ),
        fragment="background knowledge",
        suitable=tuple(m for m in ALL_METHODS if m not in ("gin", "anm", "pnl")),
        exempt=(
            *_exempt("Its edges run from latent variables, which Knowledge cannot name.", "gin"),
            *_exempt("Two variables: knowledge of the edge would be the answer.", "anm", "pnl"),
        ),
        deviations=(
            *_absent(
                "0.12.0",
                "Required pairs join the set; a pair forbidden both ways leaves it.",
                "iamb",
                "inter_iamb",
                "mmpc",
            ),
            *_absent("0.12.0", "A mask in the gated objective.", "calm"),
            *_absent(
                "0.11.0",
                "Drop forbidden lagged regressors and keep required ones unpenalized.",
                "granger",
            ),
            Deviation(
                method="bottom_up_parce_lingam",
                name="knowledge",
                default=None,
                observed="forbidden,allowed,tiers",
                fixed_in="0.11.0",
                reason="Required edges raise.",
            ),
        ),
    ),
    Concept(
        key="categorical",
        summary="The categorical columns, for a categorical or mixed-data test or score.",
        param=_template(
            "categorical",
            "str_list",
            None,
            "The categorical columns, for a mixed-data test or score: column names from the data "
            "file's header or 0-based positions, for example, categorical=[smoker,region]; [] "
            "marks none. With a null test or score they declare the column types the default "
            "picks from. null takes a DataFrame's category, bool, and text columns; a mixed-data "
            "setting passed explicitly also infers every column of at most max_categories distinct "
            "whole numbers, named in a DataTypeWarning; with none, the setting runs as its "
            "linear-Gaussian form and warns. Other tests and scores take only null or [].",
            nullable=True,
        ),
        fragment="categorical columns",
        suitable=(
            *CI_SEARCHES,
            "ges",
            "hc",
            "exact_search",
            "boss",
            "grasp",
            "tabu",
            "gies",
            "xges",
            *PCMCI_FAMILY,
        ),
        others_exempt="A continuous model.",
        deviations=(
            *_absent("0.12.0", "Categorical scores are published; engine work.", "gies", "xges"),
            *_absent("0.12.0", "Discrete conditional-independence tests.", *PCMCI_FAMILY),
        ),
    ),
    Concept(
        key="max_categories",
        summary="The most distinct whole numbers a column may hold to be inferred as categorical.",
        param=_template(
            "max_categories",
            "int",
            10,
            "The most distinct whole numbers a column may hold to be inferred as categorical, at "
            "least 2. With a continuous test or score, such columns get a one-time "
            "DataTypeWarning that they may be category codes.",
            minimum=2,
        ),
        fragment="inferred as categorical",
        suitable=(
            *CI_SEARCHES,
            "ges",
            "hc",
            "exact_search",
            "boss",
            "grasp",
            "tabu",
            "gies",
            "xges",
            *PCMCI_FAMILY,
        ),
        others_exempt="A continuous model.",
        deviations=(
            *_absent("0.12.0", "Categorical scores are published; engine work.", "gies", "xges"),
            *_absent("0.12.0", "Discrete conditional-independence tests.", *PCMCI_FAMILY),
        ),
    ),
    Concept(
        key="category_codes",
        summary="Columns that look like category codes get a DataTypeWarning.",
        suitable=ALL_METHODS,
    ),
    Concept(
        key="missing",
        summary="NaN is a missing value, and each test reads the rows that observe its variables.",
        suitable=(
            "pc",
            "fci",
            "cdnod",
            "iamb",
            "inter_iamb",
            "mmpc",
            *PCMCI_FAMILY,
            "anm",
            "pnl",
        ),
        exempt=(
            *_exempt("Its score phase reads every column.", "gfci", "mmhc", "rsmax2"),
            *_exempt(
                "A score or a regression reads every column.",
                *_PURE_SCORE,
                "calm",
                *_CONTINUOUS,
                *_LINGAM,
                "gin",
                "rcd",
                "bottom_up_parce_lingam",
            ),
            *_exempt("A series cannot drop a step.", "granger"),
        ),
    ),
    Concept(
        key="units",
        summary="Rescaling a column changes no graph and no order: a numerical step that would "
        "read the data's units runs on unit-scale columns (andrey.core.units).",
        suitable=tuple(m for m in ALL_METHODS if m not in ("ica_lingam", *_CONTINUOUS, "calm")),
        exempt=(
            *_exempt(
                "Matching the unmixing rows to the variables by 1/|W| and pruning |B| from the "
                "smallest read the data's units (Shimizu et al. 2011, Sec. 2.4).",
                "ica_lingam",
            ),
            *_exempt(
                "The least-squares score with an l1 penalty and a weight threshold reads the "
                "data's units (Zheng et al. 2018; Bello et al. 2022). robust=True is under review.",
                *_CONTINUOUS,
            ),
            *_exempt("Not measured yet: its fit needs torch.", "calm"),
        ),
    ),
    Concept(
        key="labels",
        summary="Node labels come from a DataFrame's columns, never from an argument.",
        param=_template(
            "labels",
            "str_list",
            None,
            "Not a method argument: name the columns of a DataFrame.",
            nullable=True,
        ),
        suitable=(),
        others_exempt="Labels come from a DataFrame's columns.",
        deviations=(
            Deviation(
                method="gin",
                name="labels",
                default=None,
                fixed_in="0.11.0",
                aliases=("labels",),
                fragment="",
                reason="Names the observed columns.",
            ),
        ),
    ),
    Concept(
        key="data",
        summary="The first argument is data: a matrix, or a sequence of matrices for a panel.",
        suitable=ALL_METHODS,
        deviations=(
            Deviation(
                method="multi_group_direct_lingam",
                name="data_groups",
                fixed_in="0.11.0",
                aliases=("data_groups",),
                reason="Another name for a panel.",
            ),
            Deviation(
                method="longitudinal_lingam",
                name="data_list",
                fixed_in="0.11.0",
                aliases=("data_list",),
                reason="Another name for a panel.",
            ),
        ),
    ),
    Concept(
        key="measure",
        summary="The independence measure of a DirectLiNGAM order search.",
        param=_template(
            "measure",
            "str",
            "pwling",
            "How the order search picks the most exogenous variable.",
            choices=("pwling",),
        ),
        fragment="most exogenous",
        suitable=(
            "direct_lingam",
            "multi_group_direct_lingam",
            "varma_lingam",
            "longitudinal_lingam",
        ),
        others_exempt="No DirectLiNGAM order search.",
        deviations=_absent(
            "0.11.0",
            "Runs DirectLiNGAM's order search with one measure.",
            "multi_group_direct_lingam",
            "varma_lingam",
        ),
    ),
    Concept(
        key="collider_rule",
        summary="How a separating-set search decides the colliders.",
        param=_template(
            "collider_rule",
            "str",
            "sepsets",
            "How colliders are decided on x - z - y: sepsets = z is a collider when it is not in "
            "the separating set of x and y; conservative and majority = test x and y given every "
            "subset of each one's neighbors, and z is a collider when none (conservative) or fewer "
            "than half (majority) of the separating subsets contain it, a non-collider when all or "
            "more than half do, and ambiguous otherwise, never oriented through. Both cost up to "
            "2**degree tests per pair.",
            choices=COLLIDER_RULES,
        ),
        fragment="collider",
        suitable=("pc", "fci", "gfci", "cdnod", "pcmciplus"),
        exempt=(
            *_exempt(
                "Its collider phase follows LPCMCI's own orientation rules, not yet compared with "
                "the separating-set rules.",
                "lpcmci",
            ),
            *_exempt(
                "No separating-set orientation.",
                *(
                    m
                    for m in ALL_METHODS
                    if m not in ("pc", "fci", "gfci", "cdnod", "pcmciplus", "lpcmci")
                ),
            ),
        ),
        deviations=(
            Deviation(
                method="pcmciplus",
                name="collider_rule",
                default="majority",
                choices=("majority", "conservative", "none"),
                fixed_in="0.11.0",
                aliases=("none",),
                reason="none is sepsets under another name; the default majority is tigramite's.",
            ),
        ),
    ),
    Concept(
        key="graph_kind",
        summary="The graph kind names what an edge means.",
        suitable=ALL_METHODS,
        deviations=(
            Deviation(
                method="exact_search",
                observed="dag",
                fixed_in="0.11.0",
                reason="One DAG of the equivalence class; the CPDAG is what the data determine.",
            ),
            *(
                Deviation(
                    method=m,
                    observed="cpdag",
                    fixed_in="0.11.0",
                    reason="An undirected Markov-blanket or moral graph, not a CPDAG.",
                )
                for m in ("iamb", "inter_iamb", "mmpc")
            ),
        ),
    ),
    Concept(
        key="algorithm",
        summary="metadata['algorithm'] is the method's public name.",
        suitable=ALL_METHODS,
    ),
    Concept(
        key="invalid_choice",
        summary="A value outside a parameter's choices raises ValueError.",
        suitable=ALL_METHODS,
    ),
)

#: Every concept, by key.
CONCEPTS: dict[str, Concept] = {c.key: c for c in _CONCEPTS}


def concept_param(key: str, **update: Any) -> ParamSpec:
    """The concept's template parameter, with ``update`` applied (a method's own ``doc``)."""
    param = CONCEPTS[key].param
    if param is None:
        raise KeyError(f"{key} is a behavior, not a parameter")
    return param.model_copy(update=update)
