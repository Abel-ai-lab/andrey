"""Package-wide parameter rules: one :class:`Concept` per problem that many methods share.

A concept has one parameter name, one meaning, and one default convention, and goes to every
method its theory makes suitable. Each concept lists:

- its template :class:`~andrey.spec.models.ParamSpec` (name, type, default, choices, bounds), or
  none for a behavior such as a warning or a metadata key;
- the suitable methods, and the exempt ones with the reason;
- the recorded deviations: each cell that differs (another name, default, meaning, or choices, or
  the parameter missing), its value, and why, with the former names or values that a change keeps
  working for one release.

A cell with two differences has one record. The registry takes each conforming cell's parameter
from here, and ``tests/unit/test_concepts.py`` checks every concept against every method: a new
method fails there until each concept lists it as suitable or exempt.
"""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from andrey import api
from andrey.constraint._skeleton import COLLIDER_RULES
from andrey.core.ci import SHIPPED_INDEP_TESTS, STOCHASTIC_INDEP_TESTS
from andrey.core.env import SEED_MAX
from andrey.core.fdr import FDR_METHODS
from andrey.core.invariance import TWO_GROUP_INVARIANCE_TESTS
from andrey.core.score import SCORES
from andrey.core.structure import Kind

from .models import ParamSpec, ParamType, Spec

#: The graphs a Markov-blanket search returns without a target (``structure``), and the kind of
#: each.
BLANKET_STRUCTURES: dict[str, Kind] = {
    "cpdag": "cpdag",
    "skeleton": "undirected",
    "moral": "undirected",
}


class Exemption(Spec):
    """A method the concept does not suit, and why."""

    method: str
    reason: str


class Deviation(Spec):
    """One cell that differs from its concept, and why."""

    method: str
    reason: str
    name: str | None = None  # the parameter or input name; None: absent
    default: Any = None  # its default
    choices: tuple[str, ...] | None = None  # its choices
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


# Method groups the rules below name.
CI_SEARCHES = (
    *("pc", "fci", "gfci", "boss_fci", "grasp_fci", "cdnod", "fci_jci"),
    *("iamb", "inter_iamb", "mmpc", "mmhc", "rsmax2"),
)
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
    "boss_fci",
    "grasp_fci",
)
PURE_SCORE = ("ges", "gies", "xges", "hc", "tabu", "exact_search", "boss", "grasp")
CONTINUOUS = ("notears", "dagma")
LINGAM = (
    "direct_lingam",
    "ica_lingam",
    "multi_group_direct_lingam",
    "varma_lingam",
    "longitudinal_lingam",
)
RESIDUAL = ("gin", "rcd", "bottom_up_parce_lingam", "anm", "pnl")
#: The searches that take n_starts.
STARTS = ("boss", "grasp", "boss_fci", "grasp_fci", "hc", "tabu", "igsp", "ut_igsp")
#: The methods that take a conditional-independence test.
TESTED = (*CI_SEARCHES, *PCMCI_FAMILY, "igsp", "ut_igsp")
#: The methods with no edge to name in knowledge, and ICP, whose tests take none.
NO_KNOWLEDGE = ("gin", "anm", "pnl", "icp")
#: The methods whose main loop a cap can stop early.
CAPPED_LOOPS = (
    "hc",
    "tabu",
    "mmhc",
    "rsmax2",
    *CONTINUOUS,
    "calm",
    "ica_lingam",
    "varma_lingam",
    "dynotears",
)
#: Every method, in ``andrey.api.__all__`` order.
ALL_METHODS: tuple[str, ...] = tuple(api.__all__)

_CONCEPTS = (
    Concept(
        key="alpha",
        summary="The significance level of the test that decides the returned edges, or of the "
        "only test, in the open interval (0, 1). A second test step's level is <step>_alpha.",
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
        suitable=(
            *CI_SEARCHES,
            *PCMCI_FAMILY,
            "granger",
            *RESIDUAL,
            "cam",
            "resit",
            "camuv",
            "icp",
            "igsp",
            "ut_igsp",
        ),
        exempt=(
            *_exempt("A score search: no test decides an edge.", *PURE_SCORE, "gnies"),
            *_exempt("Continuous optimization: no test decides an edge.", *CONTINUOUS, "dynotears"),
            *_exempt(
                "Its edges come from the gated likelihood; its moral-graph screen is a step "
                "level, screen_alpha.",
                "calm",
            ),
            *_exempt(
                "The order comes from an independence measure and the edges from the adaptive "
                "Lasso by BIC: no test at a level.",
                *LINGAM,
            ),
        ),
        deviations=(
            Deviation(
                method="pcmciplus",
                name="alpha",
                default=0.01,
                reason="0.01 is the level of the method's published experiments (Runge 2020: "
                "tau_max = 5 and alpha = 0.01).",
            ),
            Deviation(
                method="rcd",
                name="alpha",
                default=0.01,
                reason="0.01 is the level of the method's published experiments (Maeda and Shimizu "
                "2020 set every level to 0.01).",
            ),
            *(
                Deviation(
                    method=m,
                    name="alpha",
                    default=None,
                    fragment="p-value",
                    reason="None decides without a level: the direction with the larger p-value.",
                )
                for m in ("anm", "pnl")
            ),
            Deviation(
                method="cam",
                name="alpha",
                default=0.001,
                reason="0.001 is the paper's cutoff of the pruning tests, at every sample size "
                "(Buhlmann, Peters and Ernest 2014, Section 5.3).",
            ),
            Deviation(
                method="camuv",
                name="alpha",
                default=0.01,
                reason="0.01 is the level of the method's published experiments (Maeda and Shimizu "
                "2021 set every level to 0.01).",
            ),
            Deviation(
                method="igsp",
                name="alpha",
                default=None,
                reason="None is one over the number of observational rows, for both tests, as for "
                "ut_igsp. On simulated data in the settings of Wang et al. (2017) and Yang, "
                "Katcoff and Uhler (2018), 1e-3 had the fewest errors of the levels its authors "
                "swept, 1e-4 to 0.05. The default equals 1e-3 at 1,000 observational rows, and in "
                "the 50 groups at 10,000 and 100,000 rows per setting it had fewer SHD errors "
                "than 1e-3 in 1, by 0.28 edges, and more in none.",
            ),
            Deviation(
                method="ut_igsp",
                name="alpha",
                default=None,
                reason="None is one over the number of observational rows, for both tests. On "
                "simulated data in the paper's setting, with 5 starts, 0.001 had fewer SHD errors "
                "than the paper's level, 1e-5 (Squires, Wang and Uhler 2020, Section 5.1), in 7 of "
                "15 groups and more in 2, those by about 0.3 edges at 3,000 rows per setting; on "
                "56 further groups, the default had fewer SHD errors than 0.001 in 4 comparisons, "
                "all at 5,000 rows per setting, and more in none.",
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
        suitable=TESTED,
        exempt=(
            *_exempt(
                "It tests the independence of residuals as part of its own model.",
                *RESIDUAL,
                "resit",
                "camuv",
            ),
            *_exempt(
                "No conditional-independence test.", *PURE_SCORE, "gnies", *CONTINUOUS, *LINGAM
            ),
            *_exempt("Its moral-graph screen is part of its own model.", "calm"),
            *_exempt("No conditional-independence test.", "granger", "dynotears"),
            *_exempt("Its pruning tests the smooth terms of its own additive model.", "cam"),
            *_exempt(
                "It tests the invariance of regressions across environments, invariance_test.",
                "icp",
            ),
        ),
    ),
    Concept(
        key="invariance_test",
        summary="The invariance test: whether a variable's distribution given a set of others is "
        "the same across environments. None picks it to match indep_test.",
        param=_template(
            "invariance_test",
            "str",
            None,
            "Invariance test: gaussian = Chow and variance F-tests of the linear regressions, "
            "not with a categorical or mixed-data indep_test; indicator = indep_test against an "
            "environment indicator; null picks gaussian under the partial-correlation tests, else "
            "indicator.",
            choices=TWO_GROUP_INVARIANCE_TESTS,
            nullable=True,
        ),
        fragment="invariance test",
        suitable=("igsp", "ut_igsp"),
        others_exempt="Takes no invariance test: its data come from one environment, or its "
        "environments enter its own tests or score.",
        deviations=(
            Deviation(
                method="icp",
                name="invariance_test",
                default="approximate",
                choices=("approximate", "exact"),
                reason="Its tests are the paper's two, which compare every environment with the "
                "others' rows, two or more of them; the two-group tests compare two environments.",
            ),
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
            choices=SCORES,
            nullable=True,
        ),
        fragment="score",
        suitable=SCORE_SEARCHES,
        exempt=_exempt(
            "Its score is the method: the noise-intervention likelihood has no other local score.",
            "gnies",
        ),
        others_exempt="Not a search over parent sets scored by a decomposable score.",
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
        suitable=(*SCORE_SEARCHES, "gnies"),
        others_exempt="Does not score with BIC.",
    ),
    Concept(
        key="ebic_gamma",
        summary="The weight of the extended-BIC term of the linear-Gaussian BIC, from 0 to 1 "
        "(Chen and Chen 2008). A call that passes it with another score raises ValueError.",
        param=_template(
            "ebic_gamma",
            "float",
            1.0,
            "Weight on the extended-BIC term 2 * ebic_gamma * log C(n_variables - 1, n_parents) "
            "that the linear-Gaussian BIC adds per variable; it grows with the number of "
            "variables, from 0 to 1. Larger values give sparser graphs; 0.0 is the standard BIC. "
            "Only bic uses it: passing it with another score raises ValueError.",
            minimum=0,
            maximum=1,
        ),
        fragment="extended-BIC term",
        suitable=(*SCORE_SEARCHES, "gnies"),
        others_exempt="Does not score with BIC.",
        deviations=(
            Deviation(
                method="gfci",
                name="ebic_gamma",
                default=0.0,
                reason="The standard BIC: the extended term lowered its errors only slightly and "
                "raised them on some dense graphs; its docstring gives the measurements.",
            ),
            *(
                Deviation(
                    method=m,
                    name="ebic_gamma",
                    default=0.0,
                    reason="The standard BIC, as in gfci, the same pipeline with GES, and in the "
                    "published method.",
                )
                for m in ("boss_fci", "grasp_fci")
            ),
        ),
    ),
    Concept(
        key="equivalent_sample_size",
        summary="The equivalent sample size of the BDeu score: the weight, in rows, of its uniform "
        "Dirichlet prior (Heckerman, Geiger and Chickering 1995; Buntine 1991), in (0, inf). A "
        "call that passes it with another score raises ValueError.",
        param=_template(
            "equivalent_sample_size",
            "float",
            1.0,
            "The equivalent sample size of the BDeu score: how many rows the uniform prior over "
            "each variable's conditional distributions is worth. A larger value weighs the prior "
            "more against the data, and the graph can change with it. Only bdeu uses it: passing "
            "it with another score raises ValueError.",
            minimum=0,
            exclusive_minimum=True,
        ),
        fragment="equivalent sample size",
        suitable=SCORE_SEARCHES,
        others_exempt="Does not score with BDeu.",
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
        suitable=(*CONTINUOUS, "calm", "dynotears"),
        others_exempt="No continuous objective.",
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
        suitable=(*CONTINUOUS, "calm", "dynotears"),
        others_exempt="No weighted continuous fit to threshold.",
    ),
    Concept(
        key="standardize",
        summary="Scale each column to unit variance before a continuous fit whose penalty and "
        "weight thresholds read the data's units; its weights are then those of the scaled "
        "columns.",
        param=_template(
            "standardize",
            "bool",
            False,
            "Scale each column to unit variance first, so the fit does not read the columns' "
            "units; the weights are then those of the scaled columns.",
        ),
        fragment="unit variance",
        suitable=(*CONTINUOUS, "calm", "dynotears"),
        others_exempt="Not a continuous-optimization fit.",
    ),
    Concept(
        key="fdr_method",
        summary="The correction of the lagged links' p-values for the number of links tested: "
        "fdr_bh, the default, replaces them with their Benjamini-Hochberg "
        "adjusted values over all the lagged links together, and none leaves them as tested. A "
        "method whose graph is thresholded on those p-values (pcmci, granger) thresholds the "
        "adjusted ones; a method that decides its links before it reports them (pcmciplus) "
        "reports the adjusted values and keeps its links.",
        param=_template(
            "fdr_method",
            "str",
            "fdr_bh",
            "How the p-values of the lagged links are corrected for the number of links tested: "
            "fdr_bh replaces them with their Benjamini-Hochberg adjusted values over all the "
            "lagged links together; none keeps them as tested.",
            choices=FDR_METHODS,
        ),
        fragment="benjamini-hochberg",
        suitable=("pcmci", "pcmciplus", "granger"),
        exempt=_exempt(
            "The marks come from the removal phases and the orientation rules at alpha, not from "
            "a threshold on the p-values the method reports.",
            "lpcmci",
        ),
        others_exempt="Reports no p-values of a family of lagged links.",
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
        suitable=(*CI_SEARCHES, "pcmci", "pcmciplus", "rcd", "camuv"),
        exempt=(
            *_exempt(
                "Every test includes the default conditions; its cap counts beyond them, "
                "max_p_global.",
                "lpcmci",
            ),
            *_exempt(
                "No conditional-independence tests over growing conditioning sets.",
                *PURE_SCORE,
                "gnies",
                "calm",
                "cam",
                "resit",
                *CONTINUOUS,
                *LINGAM,
                "granger",
                "dynotears",
                "gin",
                "bottom_up_parce_lingam",
                "anm",
                "pnl",
            ),
            *_exempt(
                "No conditional-independence tests; max_parents caps the sets it tests.", "icp"
            ),
            *_exempt(
                "The minimal I-MAP conditions each pair on every earlier variable; no "
                "conditioning-set cap.",
                "igsp",
                "ut_igsp",
            ),
        ),
        deviations=(
            Deviation(
                method="rcd",
                name="max_depth",
                default=2,
                reason="Maeda and Shimizu set the cap to 2 in their experiments and advise a small "
                "one when there are fewer rows than variables; the minimum is 1, since the "
                "smallest ancestor test regresses one variable on another.",
            ),
            Deviation(
                method="camuv",
                name="max_depth",
                default=2,
                reason="It caps the variables a sink test adds together as candidate parents; each "
                "test also regresses on the variable's candidates found so far, however many. "
                "Maeda and Shimizu 2021 search sets of at most 3 variables in their experiments, a "
                "cap of 2; the minimum is 1, since the smallest set is a pair.",
            ),
            *(
                Deviation(
                    method=m,
                    name="max_depth",
                    default=None,
                    fragment="pc1",
                    reason="It caps the PC1 step that selects the lagged parents; the later tests "
                    "condition on those parents, however many, which max_p_global caps in "
                    "pcmciplus.",
                )
                for m in ("pcmci", "pcmciplus")
            ),
        ),
    ),
    Concept(
        key="max_p_global",
        summary="The largest number of conditions beyond the default conditions every test "
        "includes.",
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
        suitable=(*SCORE_SEARCHES, "cam", "gnies", "icp"),
        exempt=(
            *_exempt(
                "A constraint-based search: max_depth bounds the same cost.",
                "pc",
                "fci",
                "cdnod",
                "fci_jci",
                "iamb",
                "inter_iamb",
                "mmpc",
            ),
            *_exempt(
                "A cardinality cap is not part of the continuous objective.",
                "calm",
                *CONTINUOUS,
                "dynotears",
            ),
            *_exempt("Parents come from the adaptive Lasso on the order.", *LINGAM),
            *_exempt("No search over parent sets.", *PCMCI_FAMILY, "granger", *RESIDUAL, "resit"),
            *_exempt(
                "The paper's search sets no cap on a variable's parents; max_depth caps the "
                "candidates one sink test adds.",
                "camuv",
            ),
            *_exempt(
                "Parents are those the tests keep in the minimal I-MAP of an order; the paper "
                "sets no cap.",
                "igsp",
                "ut_igsp",
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
        suitable=CAPPED_LOOPS,
        others_exempt="No iterative loop that a cap stops early.",
    ),
    Concept(
        key="cap_warning",
        summary="Stopping on max_iter rather than on the method's own rule warns "
        "SearchLimitWarning.",
        suitable=CAPPED_LOOPS,
        others_exempt="No iterative loop that a cap stops early.",
    ),
    Concept(
        key="tau_max",
        summary="The largest lag; the smallest follows the method's theory.",
        param=_template("tau_max", "int", 1, "The largest lag, at least 1.", minimum=1),
        fragment="largest lag",
        suitable=(*PCMCI_FAMILY, "granger", "longitudinal_lingam", "dynotears"),
        exempt=_exempt("order=(p, q) is the ARMA order, a different concept.", "varma_lingam"),
        others_exempt="i.i.d. data.",
    ),
    Concept(
        key="seed",
        summary="The seed of every random draw. None is the configured seed, ANDREY_SEED or 0 "
        "when it is unset, not a fresh random one, so a fit repeats. A method that tests "
        "independence takes it for the randomized tests: "
        f"{', '.join(STOCHASTIC_INDEP_TESTS)}.",
        param=_template(
            "seed",
            "int",
            None,
            "null is the configured seed, ANDREY_SEED or 0 when it is unset, not a fresh random "
            "one, so a fit repeats. Seed for the method's random draws.",
            nullable=True,
            minimum=0,
            maximum=SEED_MAX,
        ),
        fragment="not a fresh random one",
        suitable=(*CI_SEARCHES, *PCMCI_FAMILY, *STARTS, "calm", "ica_lingam"),
        exempt=(
            *_exempt(
                "Deterministic.",
                *(m for m in PURE_SCORE if m not in STARTS),
                "gnies",
                *CONTINUOUS,
                "cam",
                "resit",
                "direct_lingam",
                "multi_group_direct_lingam",
                "varma_lingam",
                "longitudinal_lingam",
                "granger",
                "dynotears",
                *RESIDUAL,
                "camuv",
                "icp",
            ),
        ),
    ),
    Concept(
        key="n_starts",
        summary="The number of searches a fit runs, each later one from a start drawn from seed: "
        "a random order (boss, grasp, boss_fci, grasp_fci), a minimum-degree order with ties "
        "drawn from seed (igsp, ut_igsp), or the best DAG so far after n_perturbations random edge "
        "changes (hc, tabu); the best-scoring one is returned, and 1, the default, runs one "
        "search.",
        param=_template(
            "n_starts",
            "int",
            1,
            "Number of searches, the first from the method's usual start and each later one from a "
            "start drawn from seed and its index: a random order, a minimum-degree order with ties "
            "drawn from seed, or the best DAG so far after random edge changes; returns the "
            "best-scoring one.",
            minimum=1,
        ),
        fragment="best-scoring search",
        suitable=STARTS,
        others_exempt="Runs one search per fit.",
        deviations=(
            Deviation(
                method="igsp",
                name="n_starts",
                default=10,
                reason="Further searches from minimum-degree orders whose ties are drawn from "
                "seed guard against a poor first order; each one runs the paper's Algorithm 1, "
                "which restarts its depth-first search within one run from a single starting "
                "order (Yang, Katcoff and Uhler 2018). On one set of simulated datasets in its "
                "papers' settings, the SHD of 10 searches was lower than one search's in 21 of 87 "
                "groups, that of 5 in 19, and neither was higher than the other's anywhere.",
            ),
            Deviation(
                method="ut_igsp",
                name="n_starts",
                default=10,
                reason="Restarts are part of the method's search (Squires, Wang and Uhler 2020); "
                "on 15 groups of simulated datasets, 12 in the paper's setting, 10 searches had "
                "fewer SHD errors than 5 in 4 groups, by 0.17 to 0.42, 2 of them in that setting, "
                "and more in none, and fewer target errors in 2 and more in none.",
            ),
        ),
    ),
    Concept(
        key="n_perturbations",
        summary="The number of random edge additions, removals, or reversals made to the best DAG "
        "so far before each later start of n_starts; None, the default, is the number of "
        "variables.",
        param=_template(
            "n_perturbations",
            "int",
            None,
            "Number of random edge additions, removals, or reversals made to the best DAG so far "
            "before each search after the first, at least 1; null uses the number of variables. "
            "Changes nothing when n_starts is 1.",
            nullable=True,
            minimum=1,
        ),
        fragment="random edge",
        suitable=("hc", "tabu"),
        others_exempt="No later start from a changed DAG.",
    ),
    Concept(
        key="n_jobs",
        summary="Parallelism and the device come from andrey.config or ANDREY_*, never from a "
        "method argument.",
        param=_template(
            "n_jobs",
            "int",
            None,
            "Not a method argument: set it with andrey.config(n_jobs=...).",
            nullable=True,
        ),
        suitable=(),
        others_exempt="Execution settings come from andrey.config or ANDREY_* only.",
    ),
    Concept(
        key="resampling",
        summary="How andrey.bootstrap draws a resample of the rows: a subsample, sample_fraction "
        "of the rows without replacement, or a bootstrap resample, as many rows with "
        "replacement. An argument of the bootstrap function, not of a method.",
        param=_template(
            "resampling",
            "str",
            "subsample",
            "How a resample is drawn: subsample (sample_fraction of the rows, without "
            "replacement) or bootstrap (as many rows, with replacement).",
            choices=("subsample", "bootstrap"),
        ),
        suitable=(),
        others_exempt="Resampling belongs to andrey.bootstrap, which refits any method.",
    ),
    Concept(
        key="sample_fraction",
        summary="The share of the rows in a subsample, drawn without replacement: each resample "
        "of andrey.bootstrap (not used by the bootstrap resampling) and each test of "
        "andrey.markov_check. An argument of those functions, not of a method.",
        param=_template(
            "sample_fraction",
            "float",
            0.5,
            "The share of the rows in a subsample, in (0, 1].",
            minimum=0,
            maximum=1,
            exclusive_minimum=True,
        ),
        suitable=(),
        others_exempt="Row subsampling belongs to andrey.bootstrap, which refits any method, and "
        "andrey.markov_check.",
    ),
    Concept(
        key="threshold",
        summary="The share of fits in which a pair of andrey.bootstrap's averaged graph must be "
        "adjacent, on top of its most frequent edge outnumbering no edge. An argument of the "
        "bootstrap function, not of a method.",
        param=_template(
            "threshold",
            "float",
            0.5,
            "A pair of the averaged graph must be adjacent in more than this share of the fits, "
            "or in all of them, in [0.5, 1].",
            minimum=0.5,
            maximum=1,
        ),
        suitable=(),
        others_exempt="Averaging belongs to andrey.bootstrap, which refits any method.",
    ),
    Concept(
        key="strict",
        summary="What andrey.bootstrap does when the vote of the fits has no valid graph of the "
        "method's kind: return the resample graph nearest the others with a warning (False), or "
        "raise (True). An argument of the bootstrap function, not of a method.",
        param=_template(
            "strict",
            "bool",
            False,
            "Raise when the vote of the fits has no valid graph of the method's kind, instead of "
            "returning the resample graph nearest the others with a warning.",
        ),
        suitable=(),
        others_exempt="Averaging belongs to andrey.bootstrap, which refits any method.",
    ),
    Concept(
        key="averaged_graph",
        summary="BootstrapOutput.metadata['averaged_graph'] records which graph is the averaged "
        "graph: 'vote', the vote of the fits made a valid graph of the method's kind, or "
        "'nearest_resample', the valid resample graph nearest the others when the vote has no "
        "valid graph of that kind, or 'nearest_resample_invalid', the resample graph nearest the "
        "others when no resample is a valid graph of that kind either.",
        suitable=(),
        others_exempt="Averaging belongs to andrey.bootstrap, which refits any method.",
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
        suitable=tuple(m for m in ALL_METHODS if m not in NO_KNOWLEDGE),
        exempt=(
            *_exempt("Its edges run from latent variables, which Knowledge cannot name.", "gin"),
            *_exempt("Two variables: knowledge of the edge would be the answer.", "anm", "pnl"),
            *_exempt(
                "Its tests read each set of the target's predictors as the data give it; no "
                "knowledge of edges enters an invariance test.",
                "icp",
            ),
        ),
    ),
    Concept(
        key="required_edges",
        summary="Every required edge is in the output, whatever its estimated weight, its test "
        "result, or its score: a required edge whose least-squares weight is 0, or whose "
        "variables look independent, is in the graph all the same.",
        suitable=tuple(m for m in ALL_METHODS if m not in NO_KNOWLEDGE),
        exempt=(
            *_exempt(
                "Has no edge for knowledge to name (the knowledge concept).",
                *(m for m in NO_KNOWLEDGE if m != "icp"),
            ),
            *_exempt("Takes no knowledge (the knowledge concept).", "icp"),
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
            "marks none. null takes a DataFrame's category, bool, and text columns. Values never "
            "declare a column. With a null test or score they declare the column types the "
            "default picks from; a mixed-data setting reads the declared columns as categorical, "
            "and with none it runs as its linear-Gaussian form and warns. A setting for continuous "
            "data raises ValueError on a declared column, by either route.",
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
            "igsp",
            "ut_igsp",
        ),
        others_exempt="A continuous model.",
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
            "fci_jci",
            "iamb",
            "inter_iamb",
            "mmpc",
            *PCMCI_FAMILY,
            "anm",
            "pnl",
        ),
        exempt=(
            *_exempt(
                "Its score phase reads every column.",
                *("gfci", "boss_fci", "grasp_fci", "mmhc", "rsmax2"),
            ),
            *_exempt(
                "A score or a regression reads every column.",
                *PURE_SCORE,
                "gnies",
                "calm",
                *CONTINUOUS,
                "cam",
                "resit",
                *LINGAM,
                "gin",
                "rcd",
                "bottom_up_parce_lingam",
                "camuv",
                "icp",
                "igsp",
                "ut_igsp",
            ),
            *_exempt("A series cannot drop a step.", "granger", "dynotears"),
        ),
    ),
    Concept(
        key="units",
        summary="Rescaling a column changes no graph and no order: a numerical step that would "
        "read the data's units runs on unit-scale columns (andrey.core.units).",
        suitable=tuple(
            m for m in ALL_METHODS if m not in ("ica_lingam", *CONTINUOUS, "calm", "dynotears")
        ),
        exempt=(
            *_exempt(
                "Matching the unmixing rows to the variables by 1/|W| and pruning |B| from the "
                "smallest read the data's units (Shimizu et al. 2011, Sec. 2.4).",
                "ica_lingam",
            ),
            *_exempt(
                "The least-squares score with an l1 penalty and a weight threshold reads the "
                "data's units (Zheng et al. 2018; Bello et al. 2022), and so do the quasi-MCP "
                "penalty and the weight threshold of robust=True (Deng et al. 2024).",
                *CONTINUOUS,
            ),
            *_exempt(
                "The least-squares score with an l1 penalty and weight thresholds on the same-step "
                "and lagged weights reads the data's units (Pamfil et al. 2020, Sec. 4.1, "
                "standardize their returns because the penalty would favor low-variance series; "
                "Zheng et al. 2018).",
                "dynotears",
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
    ),
    Concept(
        key="data",
        summary="The first argument is data: a matrix, or a sequence of matrices for a panel.",
        suitable=ALL_METHODS,
    ),
    Concept(
        key="confounded_index",
        summary="With an environment index, whether a hidden variable may cause both the index and "
        "a data variable (FCI-JCI1, the default), or the index is set from outside, as in a "
        "randomized experiment, so every edge it keeps points out of it (FCI-JCI123).",
        param=_template(
            "confounded_index",
            "bool",
            True,
            "Whether a hidden variable may cause both the index and a data variable (FCI-JCI1). "
            "false assumes the index is set from outside, as in a randomized experiment, so every "
            "edge it keeps points out of it (FCI-JCI123).",
        ),
        fragment="hidden variable may cause both the index and a data variable",
        suitable=("fci_jci",),
        others_exempt="Not an FCI-JCI search: only those choose between JCI's assumptions on the "
        "index.",
    ),
    Concept(
        key="measure",
        summary="The pairwise likelihood-ratio measure of a DirectLiNGAM order search.",
        param=_template(
            "measure",
            "str",
            "pwling",
            "How the order search picks the most exogenous variable: pwling = the maximum-entropy "
            "approximation of the pairwise likelihood ratios; tanh and skew = their first-order "
            "approximations by a nonlinear correlation (Hyvarinen and Smith 2013).",
            choices=("pwling", "skew", "tanh"),
        ),
        fragment="most exogenous",
        suitable=(
            "direct_lingam",
            "multi_group_direct_lingam",
            "varma_lingam",
            "longitudinal_lingam",
        ),
        others_exempt="No DirectLiNGAM order search.",
    ),
    Concept(
        key="structure",
        summary="The graph a Markov-blanket search returns without a target: iamb and inter_iamb "
        "return the CPDAG (cpdag) by default, and mmpc the skeleton, the graph its sets of "
        "parents and children give. cpdag is the CPDAG of the Grow-Shrink structure learner "
        "(Margaritis 2003); skeleton, its adjacencies; moral, the moral graph of the Markov "
        "blankets.",
        param=_template(
            "structure",
            "str",
            "cpdag",
            "The graph returned without a target: cpdag = the skeleton with its colliders "
            "oriented and Meek's rules applied (Margaritis 2003); skeleton = the adjacencies, each "
            "variable joined to its parents and children; moral = the moral graph, each variable "
            "joined to its Markov blanket. With a target, the target's set, whatever the value.",
            choices=tuple(BLANKET_STRUCTURES),
        ),
        fragment="cpdag",
        suitable=("iamb", "inter_iamb", "mmpc"),
        others_exempt="Learns the whole graph of one kind.",
        deviations=(
            Deviation(
                method="mmpc",
                name="structure",
                default="skeleton",
                choices=("skeleton", "cpdag"),
                reason="The skeleton is the graph MMPC's sets give (Tsamardinos, Brown and "
                "Aliferis 2006); cpdag stays available. MMPC learns parents and children, which "
                "hold no child's other parents, so it has no moral graph to return.",
            ),
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
            "subset of each one's neighbors and, if at least one subset separates them, z is a "
            "collider when none (conservative) or fewer than half (majority) of the separating "
            "subsets contain it, a non-collider when all or more than half do, and ambiguous "
            "otherwise, never oriented through. Both cost up to 2**degree tests per pair.",
            choices=COLLIDER_RULES,
        ),
        fragment="collider",
        suitable=("pc", "fci", "gfci", "boss_fci", "grasp_fci", "cdnod", "fci_jci", "pcmciplus"),
        exempt=(
            *_exempt(
                "Its collider phase follows LPCMCI's own orientation rules, not yet compared with "
                "the separating-set rules.",
                "lpcmci",
            ),
            *_exempt(
                "The Grow-Shrink structure learner (Margaritis 2003) decides each triple by its "
                "own test, of the endpoints given the middle node and every subset of the smaller "
                "Markov blanket, not by a separating set.",
                "iamb",
                "inter_iamb",
                "mmpc",
            ),
        ),
        others_exempt="No separating-set orientation.",
        deviations=(
            Deviation(
                method="pcmciplus",
                name="collider_rule",
                default="majority",
                choices=COLLIDER_RULES,
                reason="majority orients the same-step links without regard to the order of the "
                "variables, where sepsets reads the one separating set the skeleton phase stored, "
                "which can change with it (Runge 2020).",
            ),
        ),
    ),
    Concept(
        key="graph_kind",
        summary="The graph kind names what an edge means: a search identified up to its "
        "equivalence class returns the class (cpdag, pag), and a graph whose edges claim no "
        "direction is undirected. A fit returns the kind its spec declares; a method with "
        "structure returns the kind of its default, and cpdag for structure=cpdag and undirected "
        "for skeleton and moral.",
        suitable=ALL_METHODS,
    ),
    Concept(
        key="algorithm",
        summary="metadata['algorithm'] is the method's public name.",
        suitable=ALL_METHODS,
    ),
    Concept(
        key="metadata",
        summary="A fit's metadata holds only the keys its OutputSpec declares, and a key means one "
        "thing with one shape in every method: p_values and statistics for test results, "
        "conflicts for the links colliders ask to orient both ways, score for a score search's "
        "score. An order that is not a total order of the variables has its own name "
        "(partial_order, cluster_order), and no key repeats an argument, but for the ones a "
        "default or the configuration leaves open: seed, indep_test, score_func, the score's "
        "penalty_discount, ebic_gamma, and equivalent_sample_size, and n_perturbations hold the "
        "value the fit used, and tie_break the rule of andrey.config that a fit it can change ran "
        "under.",
        suitable=ALL_METHODS,
    ),
    Concept(
        key="chosen_settings",
        summary="A method with an indep_test or a score_func records the one it used in "
        "metadata['indep_test'] or metadata['score_func'], as its current short name: the name "
        "the call passed, or, for None, the one picked from the declared column types. A "
        "registered test records its registered name. A score search records the penalties its "
        "score used: penalty_discount for every score but bdeu, ebic_gamma for bic, and "
        "equivalent_sample_size for bdeu.",
        suitable=tuple(m for m in ALL_METHODS if m in (*TESTED, *SCORE_SEARCHES)),
        others_exempt="Neither a conditional-independence test nor a score to choose.",
    ),
    Concept(
        key="warnings",
        summary="Every warning a fit raises is an AndreyWarning, one class per situation: "
        "ExperimentalWarning, BoundaryWarning, SearchLimitWarning, DataTypeWarning, "
        "SampleSizeWarning, PerformanceWarning, BackendFallbackWarning, DeprecatedArgumentWarning.",
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


#: The resampling and averaging arguments of ``andrey.bootstrap``, the "bootstrap spec" it checks.
BOOTSTRAP_PARAMS: dict[str, ParamSpec] = {
    "resampling": concept_param("resampling"),
    "sample_fraction": concept_param("sample_fraction"),
    "threshold": concept_param("threshold"),
    "strict": concept_param("strict"),
}


#: The arguments of ``andrey.markov_check`` that are package-wide concepts, which it checks.
MARKOV_CHECK_PARAMS: dict[str, ParamSpec] = {
    "alpha": concept_param(
        "alpha", doc="The level of the Anderson-Darling test of the p-values' uniformity."
    ),
    "indep_test": concept_param("indep_test"),
    "seed": concept_param("seed"),
    "sample_fraction": concept_param("sample_fraction"),
}
