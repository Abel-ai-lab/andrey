"""Generate agent interfaces from the method registry.

- :func:`to_json_schema`: a method's parameter schema.
- :func:`help_schema`: methods, the ``ANDREY_*`` variables, and CLI invocation, discovery, and
  exit-code contracts, emitted by ``andrey --help --json``.
- :func:`render_skill`: the generated ``SKILL.md`` guide.

``SCHEMA_VERSION`` versions structured help, configuration reports, and run-result envelopes
independently of ``andrey.core.output``'s on-disk ``format_version``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import andrey
from andrey.core.structure import EDGE_GLYPHS

from .concepts import CONCEPTS
from .environment import ENV_VARS, PRECEDENCE
from .registry import REGISTRY, get_spec, list_specs

if TYPE_CHECKING:
    from .models import MethodSpec, ParamSpec

SCHEMA_VERSION = "2"

# Semantic exit codes (mirrored by the CLI) -- an agent branches on these without parsing English.
EXIT_CODES = {
    "0": "success",
    "2": "usage or validation error (fix the call)",
    "3": "data error: bad shape, NaN, or singular input (fix the input)",
    "4": "algorithm failure: non-convergence or numerical breakdown (retry / adjust)",
    "1": "unexpected internal error",
}

EXPERIMENTAL_NOTE = (
    "Methods marked experimental have no published benchmark, may change their API without "
    "deprecation, and warn with `andrey.ExperimentalWarning` on their first call."
)

_JSON_TYPES: dict[str, dict] = {
    "float": {"type": "number"},
    "int": {"type": "integer"},
    "str": {"type": "string"},
    "bool": {"type": "boolean"},
    "int_pair": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
    "str_list": {"type": "array", "items": {"type": "string"}},
    "knowledge": {
        "type": "object",
        "properties": {
            **{
                part: {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": ["string", "integer"]},
                        "minItems": 2,
                        "maxItems": 2 if part == "allowed" else 3,
                    },
                }
                for part in ("required", "forbidden", "allowed")
            },
            "tiers": {
                "type": "array",
                "items": {"type": "array", "items": {"type": ["string", "integer"]}},
            },
        },
        "additionalProperties": False,
    },
}

#: How `knowledge=` reads on the command line, in the skill and in each method's help.
KNOWLEDGE_NOTE = (
    "knowledge= takes a JSON object, inline or as a .json file path: required and forbidden list "
    "directed edges [cause, effect], allowed lists the only pairs that may be adjacent, and tiers "
    "lists groups of columns in time order (no edge from a later tier into an earlier one). For a "
    "time-series method, [cause, effect, lag] is the link from cause at time t - lag to effect at "
    "time t. Name columns by the data file's header or by 0-based position, for example, "
    'knowledge=\'{"forbidden": [["y", "x"]], "tiers": [["age"], ["x", "y"]]}\'.'
)


def _param_schema(p: ParamSpec) -> dict:
    schema = dict(_JSON_TYPES[p.type])
    if p.choices:  # a nullable choice param must admit null as a valid enum member (its default)
        schema["enum"] = [*p.choices, None] if p.nullable else list(p.choices)
    if p.nullable and isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]
    schema["default"] = list(p.default) if isinstance(p.default, tuple) else p.default
    if p.minimum is not None:
        schema["exclusiveMinimum" if p.exclusive_minimum else "minimum"] = p.minimum
    if p.maximum is not None:
        schema["exclusiveMaximum" if p.exclusive_maximum else "maximum"] = p.maximum
    schema["description"] = p.doc
    schema["concept"] = p.concept
    schema["aliases"] = list(p.aliases)
    schema["choice_aliases"] = dict(p.choice_aliases)
    return schema


def concepts_dict() -> dict:
    """The package-wide parameter concepts: each one's meaning and the name each method uses."""
    return {
        key: {
            "parameter": concept.param.name,
            "description": concept.summary,
            "methods": {
                spec.name: p.name for spec in list_specs() for p in spec.params if p.concept == key
            },
        }
        for key, concept in CONCEPTS.items()
        if concept.param is not None
    }


def grammar_default(p: ParamSpec) -> str:
    """Render a parameter default in ``key=value`` grammar.

    Uses ``null`` for None, ``[1,1]`` for int_pair, ``[a,b]`` for str_list, and ``true``/``false``
    for bool.
    """
    d = p.default
    if d is None:
        return "null"
    if p.type == "int_pair":
        return "[" + ",".join(str(i) for i in d) + "]"
    if p.type == "str_list":
        return "[" + ",".join(d) + "]"
    if p.type == "bool":
        return "true" if d else "false"
    return str(d)


def grammar_example(p: ParamSpec) -> str:
    """Return an example value in parameter grammar.

    Lists use generic values because their defaults may be empty, and a number whose default is
    ``None`` uses ``1`` or ``0.05``; other types use their defaults.
    """
    if p.type == "int_pair":
        return "[1,1]"
    if p.type == "str_list":
        return "[a,b]"
    if p.type == "knowledge":
        return "knowledge.json"
    if p.default is None and p.type in ("int", "float"):
        return "1" if p.type == "int" else "0.05"
    return grammar_default(p)


def _example_token(p: ParamSpec) -> str:
    """One `key=value` example token; list values are quoted so a shell keeps the brackets."""
    val = grammar_example(p)
    return f"'{p.name}={val}'" if p.type in ("int_pair", "str_list") else f"{p.name}={val}"


def cli_example(spec: MethodSpec) -> str:
    parts = ["andrey", "run", spec.name]
    multi_input = len(spec.data.inputs) > 1  # only one input can read stdin, so no `-` when >1
    for inp in spec.data.inputs:
        if inp.kind == "panel":  # >= 2 datasets: repeat the flag
            parts += [inp.flag, "<path>", inp.flag, "<path>"]
        else:
            parts += [inp.flag, "<path>" if multi_input else "<path|->"]
    if spec.params:  # one runnable key=value override
        parts.append(_example_token(spec.params[0]))
    return " ".join(parts)


def to_json_schema(spec: MethodSpec) -> dict:
    """One method's machine tool-contract: identity, data inputs, param JSON-schema, output."""
    return {
        "name": spec.name,
        "family": spec.family,
        "status": spec.status,
        "summary": spec.summary,
        "determinism": spec.determinism,
        "accel": spec.accel,
        "seeded": spec.seeded,
        "dropped_args": list(spec.dropped_args),
        "data": {
            "layout": spec.data.layout,
            "notes": spec.data.notes,
            "missing": spec.data.missing,
            "inputs": [
                {
                    "name": inp.name,
                    "flag": inp.flag,
                    "kind": inp.kind,
                    "required": inp.required,
                    "description": inp.doc,
                }
                for inp in spec.data.inputs
            ],
        },
        "params": {
            "type": "object",
            "properties": {p.name: _param_schema(p) for p in spec.params},
            "additionalProperties": spec.passthrough,
        },
        "output": {
            "structure_type": spec.output.structure_type,
            "graph_kind": spec.output.graph_kind,
            "weighted": spec.output.weighted,
            "has_ordering": spec.output.has_ordering,
            "has_latents": spec.output.has_latents,
            "multi": spec.output.multi,
            "description": spec.output.doc,
        },
        "knowledge": list(spec.knowledge),
        "references": list(spec.references),
        "cli": cli_example(spec),
    }


def env_dict() -> dict:
    return {
        "variables": [
            {
                "name": v.name,
                "values": v.values,
                "default": v.default,
                "effect": v.effect,
                "group": v.group,
            }
            for v in ENV_VARS
        ],
        "precedence": list(PRECEDENCE),
    }


def cli_meta() -> dict:
    return {
        "invocation": "andrey run <method> --data <path|-> [key=value ...]",
        "params": (
            "hyperparameters are `key=value` after the method (or a `--config <file>` YAML/JSON "
            "base; overrides win). Types / choices are in each method's `params` schema below -- "
            "for example, alpha=0.05, indep_test=fisherz, 'order=[1,1]', criterion=null. "
            "Data arrays stay --flags (--data, --data-groups, --data-list, --domain-index, "
            "--interventions), never key=value."
        ),
        "introspection": {
            "list": "andrey list -- enumerate every method with its status and a one-line summary",
            "help": "andrey --help --json -- the complete package contract",
            "method": "andrey run <method> --help --json -- one method contract",
            "config": "andrey config --json -- current configuration and available backends",
            "environment": "andrey config --help --json -- environment defaults and effects",
            "calibrate": "andrey calibrate -- measure the worker-pool cutoffs (GES, skeleton) here",
            "skill": "andrey --skill -- the SKILL.md guide",
        },
        "output": (
            "runs and listings -> JSON on a non-interactive stdout (piped, an agent, CI); "
            "an interactive terminal gets a human summary, and --json forces JSON. failure -> a "
            "JSON error envelope on stderr plus a non-zero exit. Warnings from the fit go in "
            "run.warnings, never to stderr."
        ),
        "bounded_output": (
            "large arrays are capped at 1000 on stdout: a graph's inline edges (edges_truncated) "
            "and a temporal method's dense weight stacks (weights_truncated, *_shape kept). "
            "n_edges and edge_types always stay complete; --full or -o <file> return everything."
        ),
        "result_envelope": {
            "schema_version": SCHEMA_VERSION,
            "method": "<name>",
            "status": "supported or experimental",
            "structure": (
                "graph or temporal: kind, n_nodes, labels (a CSV / TSV header's names, else null), "
                "n_edges, edge_types, edges[] (source and target as 0-based column indices, type, "
                "and weight when the method estimates one), latent_nodes (GIN), ordering"
            ),
            "metadata": "scores / p-values / provenance from the run",
            "run": "resolved params, seed, versions, elapsed, n_samples, n_variables, and the "
            "fit's warnings (category, message)",
        },
        "bootstrap": {
            "invocation": (
                "andrey bootstrap <method> --data <path|-> [--n-boot N] [--seed S] "
                "[--threshold T] [--block-length ROWS] [key=value ...]"
            ),
            "what": (
                "refits the method on bootstrap resamples of the rows (blocks of rows for a "
                "time-series method on one series) and reports how often each edge appears; "
                "takes the inputs, params, --config, -o, --json, and --full of `run`, and not "
                "gin. The method's own seed= is the bootstrap's --seed. Workers follow "
                "ANDREY_NUM_WORKERS and the result does not depend on their number."
            ),
            "envelope": (
                "the result envelope of `run`, with `structure` the averaged graph, plus "
                "bootstrap (n_boot, threshold, seed, block_length) and frequencies[] (source, "
                "target, type, count, and frequency for each edge that appeared in a fit, most "
                "frequent first, and lag for a time-series method); capped at 1000 "
                "(frequencies_truncated)"
            ),
        },
        "error_envelope": {
            "error": {
                "code": "machine-readable error code",
                "message": "one-line human explanation",
                "valid_options": "enumerated when a value was out of range",
                "next": "runnable follow-up commands, when there is an obvious fix",
            }
        },
        "exit_codes": EXIT_CODES,
    }


def help_schema(name: str | None = None) -> dict:
    """The complete machine surface. ``name=None`` returns every method; a name returns that method.

    Always carries the shared header: schema version, Andrey version, the ``ANDREY_*`` environment
    variables, and the CLI contract (invocation, introspection ladder, exit codes).
    """
    ctx: dict = {
        "schema_version": SCHEMA_VERSION,
        "andrey_version": andrey.__version__,
        "cli": cli_meta(),
        "environment": env_dict(),
    }
    if name is None:
        ctx["methods"] = {n: to_json_schema(s) for n, s in REGISTRY.items()}
        ctx["concepts"] = concepts_dict()
    else:
        ctx["method"] = to_json_schema(get_spec(name))
    return ctx


# --- text surfaces --------------------------------------------------------------------------------

_FAMILY_TITLES = {
    "constraint": "Constraint-based",
    "score": "Score-based",
    "permutation": "Permutation-based",
    "lingam": "LiNGAM (linear non-Gaussian)",
    "temporal": "Temporal",
    "latent": "Latent-variable",
    "pairwise": "Pairwise",
}


def by_family() -> dict[str, list[MethodSpec]]:
    groups: dict[str, list[MethodSpec]] = {}
    for spec in list_specs():
        groups.setdefault(spec.family, []).append(spec)
    return groups


# What each edge ``type`` in a result means, read from the edge's source to its target.
EDGE_MEANINGS = {
    "directed": "the source causes the target.",
    "undirected": "adjacent; the data do not settle the direction.",
    "bidirected": "neither causes the other; a hidden common cause links them.",
    "partially_directed": "the target does not cause the source; the circle end is open.",
    "partially_undirected": "the tail end is settled; the circle end is open.",
    "circle": "adjacent; neither end is settled, so a hidden common cause is possible.",
}


def _data_types_section() -> str:
    """The agent guide's paragraph on column types, from the registry's mixed-data settings."""
    from andrey.api._adapt import MAX_CATEGORIES
    from andrey.core.ci import DISCRETE_INDEP_TESTS, MIXED_INDEP_TESTS
    from andrey.core.score import DISCRETE_SCORES, MIXED_SCORES

    def code(names: tuple[str, ...]) -> str:
        return " or ".join(f"`{n}`" for n in names)

    typed = [
        name for name, spec in REGISTRY.items() if any(p.name == "categorical" for p in spec.params)
    ]
    return (
        "The default tests read every column as continuous. A score search left at "
        "`score_func=null` picks `local_score_BDeu` when every column is categorical and "
        "`local_score_cond_gauss` when some are (`tabu`, with no mixed-data score, keeps the BIC), "
        "inferring the columns by the rule below and saying so in a `DataTypeWarning`; otherwise "
        "it picks the linear-Gaussian BIC. For categorical columns "
        f"coded by whole numbers, choose `indep_test` {code(DISCRETE_INDEP_TESTS)} or `score_func` "
        f"{code(DISCRETE_SCORES)}. For mixed data, choose `indep_test` {code(MIXED_INDEP_TESTS)} "
        f"or `score_func` {code(MIXED_SCORES)} (on {', '.join(f'`{n}`' for n in typed)}), "
        "and name the categorical columns with `categorical=[col,...]` (header names or 0-based "
        "positions), or `categorical=[]` for none. Without `categorical`, the setting infers them: "
        "a DataFrame's `category`, `bool`, and text columns, and every column of at most "
        f"`max_categories` (default {MAX_CATEGORIES}) distinct whole numbers. `andrey run` reads "
        "numbers only, so its data file codes each category as a whole number. The result's "
        "`metadata.categorical` lists the categorical columns read, and `run.warnings` holds a "
        "`DataTypeWarning` that names the inferred columns, or says that none was found and the "
        "setting ran as its linear-Gaussian form. Under a continuous setting, a `DataTypeWarning` "
        "names whole-number columns that may be category codes: check them before reading the "
        "graph."
    )


def render_skill() -> str:
    """The ``SKILL.md`` agent skill: how to select, call, and read Andrey from an agent."""
    out: list[str] = []
    a = out.append
    a("---")
    a("name: andrey")
    a("description: >-")
    a("  Causal discovery from observational data with the `andrey` command line. Use it to learn")
    a("  a causal graph (DAG, CPDAG, or PAG) from a dataset, find which variables drive which,")
    a("  check for hidden confounders, or analyze time-series, panel, or multi-domain data, even")
    a("  when the user names no method.")
    a("---")
    a("")
    a("# Andrey -- causal discovery for agents")
    a("")
    a(
        "Andrey learns causal structure from data and **never prompts**. Runs and method listings "
        "print JSON on stdout when piped; `--json` also selects JSON on a terminal. Help and "
        "configuration reports use readable text unless `--json` is given. Each run returns one "
        "uniform result envelope wrapping a graph (or a temporal structure)."
    )
    a("")
    a("## Discover and run")
    a("")
    a("1. `andrey list` -- enumerate every method with its status and summary (start here).")
    a("2. `andrey run <method> --help --json` -- the method contract: inputs, params, output.")
    a("3. `andrey run <method> --data <path|-> [key=value ...]` -- execute; envelope to stdout.")
    a("")
    a(
        "Data is passed by file path or `-` for stdin (CSV / TSV / .npy), never inline. A header "
        "row names the variables: the result's `labels`, which each edge's `source` / `target` "
        "indexes. Hyperparameters are `key=value` after the method (for example, `alpha=0.05`, "
        "`'order=[1,1]'`, `criterion=null`); a `--config <file>` (YAML/JSON) is a base that "
        "`key=value` overrides. "
        "`andrey run <method> --help` lists its params; adding `--json` returns its versioned "
        "contract without reading data or running the method. `andrey --help --json` returns "
        "all contracts. On failure the exit code is non-zero and a JSON error envelope goes to "
        "stderr: `valid_options` lists the accepted values, `did_you_mean` suggests a close name, "
        "and `next` lists runnable follow-up commands."
    )
    a("")
    a("## Reading the result")
    a("")
    a(
        "A run's envelope holds `structure` (or `results`, one per group): its `kind` (`dag`, "
        "`cpdag`, `pag`, or temporal), `labels`, `n_edges`, an `edge_types` count, and `edges`. "
        "Each edge has a `source` and a `target`, indexes into `labels`, a `type`, and, from the "
        "LiNGAM methods, a `weight`. A causal order is in `structure.ordering`, scores and "
        "p-values in `metadata`, and the resolved call (params, seed, versions) in `run`. A large "
        "graph's inline `edges` are capped (`edges_truncated: true`); `--full` or `-o <file>` "
        f'keeps every edge. Every envelope carries `schema_version` ("{SCHEMA_VERSION}"): check it '
        "before reading the fields."
    )
    a("")
    a("Edge types, read from `source` to `target`:")
    a("")
    for name, glyph in EDGE_GLYPHS.items():
        a(f"- `{name}` (`{glyph}`): {EDGE_MEANINGS[name]}")
    a("")
    a(
        "A CPDAG's undirected edges are directions the data cannot settle, not missing work; "
        "report them as such."
    )
    a("")
    a("## How stable an edge is")
    a("")
    a(
        "`andrey bootstrap <method> --data <path|-> [--n-boot N] [--seed S] [--threshold T] "
        "[key=value ...]` refits the method on resamples of the rows and answers how often each "
        "edge appears. It takes the inputs and `key=value` params of `andrey run`, and not `gin`. "
        "Its envelope is a run's, with `structure` the averaged graph: a pair is in it when it is "
        "adjacent in at least `--threshold` of the fits (default 0.5), with the edge it has most "
        "often. `bootstrap` gives `n_boot`, `threshold`, `seed`, and `block_length`, and "
        "`frequencies` lists each edge that appeared in some fit, most frequent first, with its "
        "`source`, `target`, `type`, `count`, and `frequency` (a time-series method adds `lag`). "
        "Report an edge with its frequency: 0.95 is stable, 0.3 is not. A time-series method on "
        "one series resamples blocks of `--block-length` rows. Set `--seed` when the result must "
        "repeat; `ANDREY_NUM_WORKERS` spreads the fits over processes without changing it."
    )
    a("")
    a("## Data types")
    a("")
    a(_data_types_section())
    a("")
    a("## Missing values")
    a("")
    a(
        "A `NaN` cell, or an empty, `NA`, or `N/A` field of the data file, is a missing value. "
        "The methods whose schema sets `data.missing` accept it and delete rows per test: each "
        "test reads the rows observed in all of its variables, and `metadata.rows_tested` gives "
        "the fewest (`min`) and the median rows a test read. They are "
        + ", ".join(f"`{s.name}`" for s in REGISTRY.values() if s.data.missing)
        + ". Every other method refuses a missing value with a data error that says what to do "
        "instead. The deletion is sound when values are missing completely at random; when "
        "whether a value is missing depends on the variables, `pc` can keep extra edges, and "
        "`fci` stays sound "
        "when no missingness mechanism causally affects another."
    )
    a("")
    a("## Background knowledge")
    a("")
    a(
        KNOWLEDGE_NOTE + " A method honors the parts its schema lists under knowledge and rejects "
        "the others. Methods that take it: "
        + ", ".join(f"`{s.name}`" for s in REGISTRY.values() if s.knowledge)
        + "."
    )
    a("")
    a("## Choosing a method")
    a("")
    a(
        f"{EXPERIMENTAL_NOTE} A supported method is benchmarked and checked before each release: "
        "measured, not accurate on every workload. When a supported and an experimental method "
        "both fit the data, prefer the supported one."
    )
    a("")
    for family, specs in by_family().items():
        a(f"**{_FAMILY_TITLES[family]}**")
        for spec in specs:
            kind = spec.output.graph_kind or spec.output.structure_type
            status = ", experimental" if spec.status == "experimental" else ""
            a(f"- `{spec.name}` ({kind}{status}): {spec.summary}")
        a("")
    a("## Environment")
    a("")
    a(
        "Andrey reads a few `ANDREY_*` environment variables for backend selection, seeding, and "
        "testing. "
        "`andrey config` reports current settings and available backends; `--json` selects a "
        "structured report. Availability does not determine which backend an operation uses. "
        "`andrey config --help` explains every variable, default, effect, and precedence rule."
    )
    a("")
    a(
        "Before a large run, read `andrey config --json` once. Its `system` section gives the "
        "platform's support tier, the usable CPU cores, each optional extra with the command that "
        "installs it, and why each unavailable backend is off. Suggest a missing extra to the user "
        "rather than installing it. On many cores, `ANDREY_NUM_WORKERS=-1` runs GES on all "
        "of them, and `andrey calibrate` measures the pool cutoffs of GES and the skeleton "
        "search on this machine. Set "
        "`ANDREY_SEED` when a stochastic method's result must repeat."
    )
    a("")
    a("| Variable | Default | Effect |")
    a("|---|---|---|")
    for v in ENV_VARS:
        effect = v.effect.replace("\n", " ")
        a(f"| `{v.name}` | `{v.default}` | {effect} |")
    a("")
    a("Precedence:")
    a("")
    for rule in PRECEDENCE:
        a(f"- {rule}")
    a("")
    a("## Exit codes")
    a("")
    for code, meaning in EXIT_CODES.items():
        a(f"- `{code}`: {meaning}")
    a("")
    return "\n".join(out)
