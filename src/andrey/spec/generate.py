"""Generate agent interfaces from the method registry.

- :func:`to_json_schema`: a method's parameter schema.
- :func:`help_schema`: methods, the ``ANDREY_*`` variables, and CLI invocation, discovery, and
  exit-code contracts, emitted by ``andrey --help --json``.
- :func:`render_llms_txt`: the compact discovery index.
- :func:`render_skill`: the generated ``SKILL.md`` guide.

``SCHEMA_VERSION`` versions structured help, configuration reports, and run-result envelopes
independently of ``andrey.core.output``'s on-disk ``format_version``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import andrey
from andrey.core.structure import EDGE_GLYPHS

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
}


def _param_schema(p: ParamSpec) -> dict:
    schema = dict(_JSON_TYPES[p.type])
    if p.choices:  # a nullable choice param must admit null as a valid enum member (its default)
        schema["enum"] = [*p.choices, None] if p.nullable else list(p.choices)
    if p.nullable and isinstance(schema.get("type"), str):
        schema["type"] = [schema["type"], "null"]
    schema["default"] = list(p.default) if isinstance(p.default, tuple) else p.default
    schema["description"] = p.doc
    return schema


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
        "data": {
            "layout": spec.data.layout,
            "notes": spec.data.notes,
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
            "Data arrays stay --flags (--data, --data-groups, --data-list, --c-indx), never "
            "key=value."
        ),
        "introspection": {
            "list": "andrey list -- enumerate every method with its status and a one-line summary",
            "help": "andrey --help --json -- the complete package contract",
            "method": "andrey run <method> --help --json -- one method contract",
            "config": "andrey config --json -- current configuration and available backends",
            "environment": "andrey config --help --json -- environment defaults and effects",
            "calibrate": "andrey calibrate -- measure the GES and HC worker-pool cutoffs here",
            "skill": "andrey --skill -- the SKILL.md guide",
            "llms": "andrey --llms -- the compact discovery index",
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


def render_llms_txt() -> str:
    """The compact discovery index (llms.txt): one line per method, and the ANDREY_* variables."""
    lines = [
        "# Andrey",
        "",
        "> Scalable causal discovery with a familiar API, accelerated on CPU and GPU. Every method "
        "returns one uniform StructureOutput. `andrey run <method> --data <path|-> "
        "[key=value ...] --json` prints a JSON result envelope on stdout.",
        "",
        f"Schema version: {SCHEMA_VERSION}. Discover: `andrey list`, "
        "`andrey run <method> --help --json`, `andrey --help --json`, `andrey --skill`.",
        "",
        "## Methods",
        "",
        EXPERIMENTAL_NOTE,
        "",
    ]
    for family, specs in by_family().items():
        lines.append(f"### {_FAMILY_TITLES[family]}")
        for spec in specs:
            status = " (experimental)" if spec.status == "experimental" else ""
            lines.append(f"- `{spec.name}`{status}: {spec.summary}")
        lines.append("")
    lines += ["## Environment (`ANDREY_*`)", ""]
    for v in ENV_VARS:
        lines.append(f"- `{v.name}` (default `{v.default}`): {v.effect}")
    lines.append("")
    return "\n".join(lines)


# What each edge ``type`` in a result means, read from the edge's source to its target.
EDGE_MEANINGS = {
    "directed": "the source causes the target.",
    "undirected": "adjacent; the data do not settle the direction.",
    "bidirected": "neither causes the other; a hidden common cause links them.",
    "partially_directed": "the target does not cause the source; the circle end is open.",
    "partially_undirected": "the tail end is settled; the circle end is open.",
    "circle": "adjacent; neither end is settled, so a hidden common cause is possible.",
}


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
        "rather than installing it. On many cores, `ANDREY_NUM_WORKERS=-1` runs GES and HC on all "
        "of them, and `andrey calibrate` measures their pool cutoffs on this machine. Set "
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
