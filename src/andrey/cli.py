"""The ``andrey`` command-line interface, generated from method and configuration definitions.

``andrey list`` enumerates methods, ``andrey run <method> --help`` explains one method, and
``--help --json`` returns its versioned contract without execution. Root ``--help --json`` returns
all contracts; ``--skill`` prints the agent guide. ``andrey config`` reports current configuration
and backend availability; ``--json`` selects a structured report.

``andrey run <method> --data <path|-> [key=value ...]`` executes; a CSV / TSV header names the
nodes. Runs and listings use JSON when piped and readable text on a terminal; ``--json`` forces
JSON. Large result arrays are capped on stdout; ``--full`` or ``-o <file>`` preserves the complete
result. Failures print a structured JSON error on stderr and exit non-zero.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import os
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Optional

import typer

import andrey
from andrey.api._adapt import shared_labels, with_labels
from andrey.core import LATENT, GraphStructure
from andrey.core.backend import usable_cpus
from andrey.core.env import SETTINGS
from andrey.core.knowledge import KnowledgeError
from andrey.core.structure import SUMMARY_EDGES, node_labels
from andrey.spec import ENV_VARS, REGISTRY, SCHEMA_VERSION, get_spec, help_schema, list_specs
from andrey.spec.generate import (
    env_dict,
    grammar_default,
    grammar_example,
    render_skill,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    import numpy as np

    from andrey import BootstrapOutput
    from andrey.core import StructureOutput
    from andrey.spec.models import InputSpec, MethodSpec, ParamSpec

# Exit codes (see andrey.spec.generate.EXIT_CODES).
EXIT_OK = 0
EXIT_INTERNAL = 1
EXIT_USAGE = 2
EXIT_DATA = 3
EXIT_ALGORITHM = 4

# Output bounds: a large graph must not flood stdout / an agent's context. The full count and the
# edge-type breakdown survive; only the inline edge list is capped (--full / -o keep everything).
EDGE_LIMIT = 1000  # max inline edges in the JSON envelope on stdout


# --- error reporting ------------------------------------------------------------------------------


class CliError(Exception):
    """A handled failure carrying an exit code and a structured error envelope."""

    def __init__(self, code: str, message: str, exit_code: int, **extra: object) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.exit_code = exit_code
        self.extra = extra


def emit_error(err: CliError) -> int:
    payload = {"error": {"code": err.code, "message": err.message, **err.extra}}
    print(json.dumps(payload, indent=2), file=sys.stderr)
    return err.exit_code


# --- data loading ---------------------------------------------------------------------------------


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _row_fields(line: str, delimiter: str | None) -> list[tuple[str, bool]]:
    """Split one text row into ``(field, quoted)`` pairs, one per field.

    As in CSV, a field in double quotes may hold the delimiter, and ``""`` inside it is one quote.
    Whitespace-separated text splits at every run of whitespace.
    """
    if delimiter is None:
        return [(field.strip('"'), field.startswith('"')) for field in line.split()]
    field = re.compile(rf' *(?:"((?:[^"]|"")*)" *|([^{delimiter}]*))(?={delimiter}|$)')
    fields, start = [], 0
    while True:
        match = field.match(line, start)
        assert match is not None  # the unquoted branch takes any text up to the delimiter
        quoted, plain = match.groups()
        fields.append((quoted.replace('""', '"'), True) if quoted is not None else (plain, False))
        start = match.end() + 1
        if start > len(line):
            return fields


def _parse_text(raw: str) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Parse delimited text into a 2-D float64 array and its header's column names, if any.

    A header is a first row whose every field is text: not a number, or in double quotes, so
    ``a,b,c`` and ``"1","0","2"`` name the columns. A first row of unquoted numbers is data. A
    first row that mixes unquoted numbers with text (``a,1,b`` or ``1,2,bad``) is neither, so it
    surfaces as ``bad_data`` rather than being read as a header or as a truncated dataset.
    """
    import numpy as np

    first = next((ln for ln in raw.splitlines() if ln.strip()), "")
    unquoted = re.sub(r'"(?:[^"]|"")*"', "", first)
    delimiter = "," if "," in unquoted else ("\t" if "\t" in unquoted else None)
    fields = _row_fields(first, delimiter)
    numbers = [text for text, quoted in fields if not quoted and _is_number(text)]
    if numbers and len(numbers) < len(fields):
        text = next(text for text, quoted in fields if quoted or not _is_number(text))
        raise CliError(
            "bad_data",
            f"the first row mixes numbers, such as {numbers[0].strip()!r}, with text, such as "
            f"{text!r}: a header names every column with text (quote a name that is a number, "
            'such as "1"), and a data row holds numbers only',
            EXIT_DATA,
        )
    header = bool(fields) and not numbers
    try:
        arr = np.loadtxt(
            io.StringIO(raw),
            delimiter=delimiter,
            dtype=np.float64,
            ndmin=2,
            skiprows=1 if header else 0,
        )
    except ValueError as exc:
        raise CliError("bad_data", f"could not parse numeric data: {exc}", EXIT_DATA) from exc
    if not header:
        return arr, None
    names = [text if quoted else text.strip() for text, quoted in fields]
    if arr.shape[1] != len(names):
        raise CliError(
            "bad_data",
            f"the header names {len(names)} columns but the rows have {arr.shape[1]}",
            EXIT_DATA,
        )
    try:
        return arr, node_labels(names)
    except ValueError as exc:
        raise CliError("bad_data", str(exc), EXIT_DATA) from exc


def load(path: str) -> tuple[np.ndarray, tuple[str, ...] | None]:
    """Load one array and its header's column names from a path (CSV / TSV / .npy) or ``-``."""
    import numpy as np

    if path == "-":
        if sys.stdin.isatty():  # never block waiting on a keyboard -- "never prompts"
            raise CliError(
                "no_stdin", "no data on stdin (use --data <path> or pipe in)", EXIT_USAGE
            )
        return _parse_text(sys.stdin.read())
    if path.endswith(".npy"):
        try:
            return np.asarray(np.load(path), dtype=np.float64), None
        except FileNotFoundError as exc:
            raise CliError("input_not_found", f"no such file: {path!r}", EXIT_USAGE) from exc
        except (OSError, ValueError) as exc:  # a corrupt .npy is a data error
            raise CliError("bad_data", f"could not load {path!r}: {exc}", EXIT_DATA) from exc
    try:
        with open(path, encoding="utf-8") as fh:
            return _parse_text(fh.read())
    except FileNotFoundError as exc:
        raise CliError("input_not_found", f"no such file: {path!r}", EXIT_USAGE) from exc
    except OSError as exc:
        raise CliError("input_unreadable", f"could not read {path!r}: {exc}", EXIT_USAGE) from exc


def load_input(
    spec_input: InputSpec, value: str | list[str] | None
) -> tuple[np.ndarray | list[np.ndarray], tuple[str, ...] | None]:
    """Load one method input to the shape its kind requires, with its header's column names."""
    if spec_input.kind == "panel":
        paths = value if isinstance(value, list) else ([value] if value else [])
        if len(paths) < 2:
            raise CliError(
                "too_few_groups",
                f"{spec_input.flag} needs at least two datasets (repeat the flag)",
                EXIT_USAGE,
            )
        loaded = [load(p) for p in paths]
        try:
            labels = shared_labels(names for _, names in loaded)
        except ValueError as exc:
            raise CliError("bad_data", str(exc), EXIT_DATA) from exc
        return [arr for arr, _ in loaded], labels
    path = value[0] if isinstance(value, list) else value
    assert isinstance(path, str)  # non-panel inputs are required single paths
    arr, labels = load(path)
    if spec_input.kind == "context":
        return arr.reshape(arr.shape[0], -1), None
    return arr, labels


# --- result envelope ------------------------------------------------------------------------------


def graph_view(graph, weighted: np.ndarray | None = None) -> dict:
    edges = []
    edge_types: dict[str, int] = {}
    for source, target, edge_type in graph.oriented_edges(index=True):
        rec = {"source": source, "target": target, "type": edge_type}
        if weighted is not None and edge_type in ("directed", "partially_directed"):
            rec["weight"] = float(weighted[source, target])
        edges.append(rec)
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    view = {
        "type": "graph",
        "kind": graph.kind,
        "n_nodes": int(graph.n_nodes),
        "n_edges": len(edges),
        "edge_types": edge_types,  # a bounded summary that survives edge truncation
        "labels": list(graph.labels) if graph.labels is not None else None,
        "edges": edges,
    }
    if graph.node_types is not None:
        view["latent_nodes"] = [i for i, t in enumerate(graph.node_types.tolist()) if t == LATENT]
    return view


def temporal_view(struct) -> dict:
    lags = []
    agg_types: dict[str, int] = {}
    for k in struct.lags:  # lag *values*, one per stacked graph
        gview = graph_view(struct.lag(k))
        lags.append(
            {
                "lag": int(k),
                "n_edges": gview["n_edges"],
                "edge_types": gview["edge_types"],
                "edges": gview["edges"],
            }
        )
        for t, n in gview["edge_types"].items():
            agg_types[t] = agg_types.get(t, 0) + n
    view = {
        "type": "temporal",
        "n_nodes": int(struct.n_nodes),
        "n_lags": int(struct.n_lags),
        "n_edges": sum(lag["n_edges"] for lag in lags),  # aggregate, like a graph structure
        "edge_types": agg_types,  # aggregate breakdown; survives edge/weight truncation
        "labels": list(struct.labels) if struct.labels is not None else None,
        "lags": lags,
    }
    if struct.lag_weights is not None:  # the AR coefficient stack (lag 0 = instantaneous)
        view["lag_weights"] = struct.lag_weights.tolist()
    if struct.lag_weights_ma is not None:  # the MA coefficient stack (VARMA)
        view["lag_weights_ma"] = struct.lag_weights_ma.tolist()
    return view


def structure_view(struct, weighted: np.ndarray | None = None) -> dict:
    return temporal_view(struct) if struct.type == "temporal" else graph_view(struct, weighted)


def result_view(out) -> dict:
    view = structure_view(out.structure, out.weighted_adjacency)
    if out.ordering is not None:
        view["ordering"] = list(out.ordering)
    return {"structure": view, "metadata": dict(out.metadata)}


def envelope(spec: MethodSpec, out, run: dict) -> dict:
    env: dict = {"schema_version": SCHEMA_VERSION, "method": spec.name, "status": spec.status}
    if isinstance(out, list):  # multi-output (multi_group_direct_lingam)
        env["multi"] = True
        env["results"] = [result_view(o) for o in out]
    else:
        result = result_view(out)
        env["structure"] = result["structure"]
        env["metadata"] = result["metadata"]
    env["run"] = run
    return env


def bootstrap_envelope(
    spec: MethodSpec, out: BootstrapOutput | list[BootstrapOutput], run: dict
) -> dict:
    """The result envelope of ``andrey bootstrap``: ``run``'s, with each graph's edge frequencies.

    ``structure`` is the averaged graph; ``frequencies`` lists every edge that appeared in a fit,
    most frequent first, as ``run``'s ``edges`` do plus ``count`` and ``frequency``.
    """
    outs = out if isinstance(out, list) else [out]
    first = outs[0]
    env: dict = {"schema_version": SCHEMA_VERSION, "method": spec.name, "status": spec.status}
    env["bootstrap"] = {
        "n_boot": first.n_boot,
        "threshold": first.threshold,
        "seed": first.metadata["seed"],
        "block_length": first.metadata["block_length"],
    }
    views = [
        {"structure": structure_view(o.structure), "frequencies": o.table(index=True)} for o in outs
    ]
    if isinstance(out, list):  # multi-output (multi_group_direct_lingam)
        env["multi"] = True
        env["results"] = views
    else:
        env.update(views[0])
    env["run"] = run
    return env


def _nested_shape(x: object) -> list[int]:
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        x = x[0] if x else None
    return shape


def _size(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _truncate_output(struct: dict, limit: int) -> bool:
    """Cap a structure's inline edges (and temporal weight stacks); return if anything was cut.

    The summary fields (``n_edges``, ``edge_types``, the weight ``*_shape``) always survive, so a
    large graph or a dense coefficient stack never floods stdout or an agent's context. For a
    temporal structure the edge budget is shared across all lags: many small lags cannot
    collectively exceed ``limit`` inline edges without the cap engaging.
    """
    cut = False
    if struct.get("type") == "temporal":
        remaining = limit  # one budget across every lag, not `limit` per lag
        for t in struct["lags"]:
            edges = t.get("edges", [])
            if len(edges) > remaining:
                t["edges"] = edges[:remaining]
                t["edges_truncated"] = True
                cut = True
                remaining = 0
            else:
                remaining -= len(edges)
        for key in ("lag_weights", "lag_weights_ma"):  # dense coefficient stacks are O((p+q)*d^2)
            weights = struct.get(key)
            if weights is not None and _size(_nested_shape(weights)) > limit:
                struct[f"{key}_shape"] = _nested_shape(weights)
                struct[key] = None
                struct["weights_truncated"] = True
                cut = True
        return cut
    edges = struct.get("edges", [])
    if len(edges) > limit:
        struct["edges"] = edges[:limit]
        struct["edges_truncated"] = True
        cut = True
    return cut


def _apply_output_cap(env: dict, limit: int) -> None:
    """Cap inline edges and dense weight stacks across a single- or multi-output envelope.

    Counts and shape summaries always survive -- only bulk arrays are cut, so a large result never
    floods stdout. A bootstrap's frequency table keeps its most frequent edges. A note tells the
    caller how to get the whole thing back.
    """
    results = env["results"] if env.get("multi") else [env]
    cut = [_truncate_output(r["structure"], limit) for r in results]
    for r in results:
        if len(r.get("frequencies", ())) > limit:  # a bootstrap's edge table: its most frequent
            r["frequencies"] = r["frequencies"][:limit]
            r["frequencies_truncated"] = True
            cut.append(True)
    if any(cut):
        env["note"] = (
            f"inline edges / dense weight stacks capped at {limit}; counts and shapes stay "
            "complete. Re-run with --full or -o <file> for the entire result."
        )


# --- human render (interactive terminal) ----------------------------------------------------------

_MORE_HINT = " (--full or -o <file> for all)"  # ends the "... and N more" line


def _run_footer(run: dict) -> str:
    notes = "".join(f"\n  {w['category']}: {w['message']}" for w in run["warnings"])
    return (
        f"\n\n  {run['n_samples']} samples x {run['n_variables']} variables"
        f" | {run['elapsed_s']}s | andrey {run['versions']['andrey']}{notes}"
    )


def render_human(
    spec: MethodSpec, out: StructureOutput | list[StructureOutput], run: dict, *, full: bool = False
) -> str:
    """Render a terminal summary of the result and run details.

    The method name heads each result. ``full`` lists every edge; otherwise each graph lists at
    most ``SUMMARY_EDGES`` edges. The run footer follows the result.
    """
    limit = None if full else SUMMARY_EDGES
    if not isinstance(out, list):
        lines = out._summary(name=spec.name, limit=limit, more=_MORE_HINT)
        return "\n".join(lines) + _run_footer(run)
    lines = [f"{spec.name}  |  {len(out)} groups"]  # multi-group: one block per group
    for k, o in enumerate(out):
        graph = o.structure
        assert isinstance(graph, GraphStructure)  # every group's result is a graph
        weights = o.to_scipy_sparse()  # every group's result is weighted
        lines += ["", f"  group {k}  {graph.kind}  |  {graph._edge_count_text()}"]
        edges = graph._edge_lines(weights=weights, limit=limit, more=_MORE_HINT)
        lines += ["    " + line for line in edges]
    return "\n".join(lines) + _run_footer(run)


def render_bootstrap_human(
    spec: MethodSpec, out: BootstrapOutput | list[BootstrapOutput], run: dict, *, full: bool = False
) -> str:
    """Render a terminal summary of a bootstrap: each averaged graph and its most frequent edges.

    ``full`` lists every edge; otherwise each graph lists at most ``SUMMARY_EDGES`` of them.
    """
    limit = None if full else SUMMARY_EDGES
    if not isinstance(out, list):
        return "\n".join(out._summary(name=spec.name, limit=limit, more=_MORE_HINT)) + _run_footer(
            run
        )
    lines = [f"{spec.name}  |  {len(out)} groups"]
    for k, o in enumerate(out):
        lines += ["", *o._summary(name=f"group {k}", limit=limit, more=_MORE_HINT)]
    return "\n".join(lines) + _run_footer(run)


# --- run: `key=value` overrides + `--config`, coerced against the spec ---------------------


def _unquote(raw: str) -> tuple[str, bool]:
    """Strip one matched pair of double quotes; a quoted value is a literal string (no coercion)."""
    if len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
        return raw[1:-1], True
    return raw, False


def _bracket_items(value: str, key: str) -> list[str]:
    """Parse a `[a,b]` literal into trimmed, unquoted string items (`[]` -> empty); brackets
    required, so a JSON-style `["a","b"]` yields `a`, `b` not `"a"`, `"b"`."""
    if not (value.startswith("[") and value.endswith("]")):
        raise CliError("bad_value", f"{key} expects a bracketed list like [a,b]", EXIT_USAGE)
    inner = value[1:-1].strip()
    return [_unquote(item.strip())[0] for item in inner.split(",")] if inner else []


def _check_choice(value: str, param: ParamSpec) -> str:
    """A string value must be one of the param's choices, when it enumerates any."""
    if param.choices and value not in param.choices:
        raise CliError(
            "unsupported_value",
            f"{param.name} must be one of: {', '.join(param.choices)}",
            EXIT_USAGE,
            valid_options=list(param.choices),
        )
    return value


def _coerce(raw: str, param: ParamSpec) -> object:
    """Coerce a raw CLI string against a ParamSpec type -- the spec drives the parse (no sniff)."""
    if param.type == "knowledge":
        return _knowledge_value(raw, param.name)
    value, quoted = _unquote(raw)
    if quoted:  # a quoted value is a literal string -- only a str param can accept one
        if param.type != "str":
            raise CliError(
                "bad_value", f"{param.name} is a {param.type}, not a quoted string", EXIT_USAGE
            )
        return _check_choice(value, param)
    if param.nullable and value.lower() == "null":
        return None
    if value == "":
        raise CliError(
            "bad_value", f"{param.name}: empty value (use {param.name}=null for none)", EXIT_USAGE
        )
    kind = param.type
    if kind == "bool":
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise CliError(
            "bad_value",
            f"{param.name} expects true or false",
            EXIT_USAGE,
            valid_options=["true", "false"],
        )
    if kind == "int":
        try:
            return int(value)
        except ValueError:
            raise CliError(
                "bad_value", f"{param.name} expects an integer, got {value!r}", EXIT_USAGE
            ) from None
    if kind == "float":
        try:
            return float(value)
        except ValueError:
            raise CliError(
                "bad_value", f"{param.name} expects a number, got {value!r}", EXIT_USAGE
            ) from None
    if kind == "int_pair":
        items = _bracket_items(value, param.name)
        try:
            pair = tuple(int(i) for i in items)
        except ValueError:
            raise CliError(
                "bad_value", f"{param.name} expects two integers like [1,1]", EXIT_USAGE
            ) from None
        if len(pair) != 2:
            raise CliError(
                "bad_value",
                f"{param.name} expects two integers like [1,1]",
                EXIT_USAGE,
                expected_shape="[int, int]",
            )
        return pair
    if kind == "str_list":
        return tuple(_bracket_items(value, param.name))
    return _check_choice(value, param)


def _knowledge_value(raw: str, name: str) -> dict | None:
    """Read a `knowledge=` value: inline JSON (`{...}`), `null`, or the path of a JSON file, bare
    or in one pair of single or double quotes."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        text = text[1:-1].strip()
    if text.lower() == "null":
        return None
    if not text.startswith("{"):
        try:
            text = Path(text).read_text()
        except OSError as exc:
            raise CliError(
                "bad_value",
                f"{name} expects a JSON object or the path of a JSON file; cannot read {raw!r}: "
                f"{exc.strerror}",
                EXIT_USAGE,
            ) from None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CliError("bad_value", f"{name} is not valid JSON: {exc}", EXIT_USAGE) from None
    return _knowledge_object(value, name)


def _knowledge_object(value: object, name: str) -> dict:
    """Check a knowledge object's keys; the facade checks its contents."""
    keys = ("required", "forbidden", "allowed", "tiers")
    if not isinstance(value, dict) or set(value) - set(keys):
        raise CliError(
            "bad_value",
            f"{name} must be a JSON object whose keys are among {', '.join(keys)}; each is "
            "optional",
            EXIT_USAGE,
            valid_options=list(keys),
        )
    return value


def _coerce_passthrough(raw: str) -> object:
    """Sniff an open passthrough value (only where there is no spec, such as CALM): quoted -> str,
    then null, true/false, int, float, [list], else str."""
    value, quoted = _unquote(raw)
    if quoted:
        return value
    if value == "":
        raise CliError("bad_value", "empty value (use key=null for none)", EXIT_USAGE)
    if value.lower() == "null":
        return None
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.startswith("[") and value.endswith("]"):
        return [_coerce_passthrough(item) for item in _bracket_items(value, "value")]
    return value


def _validate_config_value(value: object, param: ParamSpec) -> object:
    """A `--config` value is already typed (YAML/JSON): validate against the spec, do not coerce a
    string. Returns tuples for int_pair / str_list."""
    if value is None:
        if param.nullable:
            return None
        raise CliError("bad_value", f"{param.name} may not be null", EXIT_USAGE)
    if param.type == "knowledge":
        if isinstance(value, str):  # a path or inline JSON, as on the command line
            return _knowledge_value(value, param.name)
        return _knowledge_object(value, param.name)
    ok = {
        "bool": isinstance(value, bool),
        "int": isinstance(value, int) and not isinstance(value, bool),
        "float": isinstance(value, (int, float)) and not isinstance(value, bool),
        "str": isinstance(value, str),
        "int_pair": isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(i, int) and not isinstance(i, bool) for i in value),
        "str_list": isinstance(value, (list, tuple)) and all(isinstance(i, str) for i in value),
    }[param.type]
    if not ok:
        raise CliError(
            "bad_value", f"{param.name} in --config is not a valid {param.type}", EXIT_USAGE
        )
    if param.type == "str" and param.choices and value not in param.choices:
        raise CliError(
            "unsupported_value",
            f"{param.name} must be one of: {', '.join(param.choices)}",
            EXIT_USAGE,
            valid_options=list(param.choices),
        )
    if param.type in ("int_pair", "str_list"):
        return tuple(value)  # ty: ignore[invalid-argument-type]  # validated list/tuple above
    return value


def _flag_hint(token: str) -> dict:
    """A structured error for a stale `--flag` token, with a did_you_mean when it names a param."""
    name = token.lstrip("-").replace("-", "_")
    if name in {p.name for s in REGISTRY.values() for p in s.params}:
        return {
            "param": token,
            "message": "params are key=value, not flags",
            "did_you_mean": f"{name}=<value>",
        }
    return {"param": token, "message": f"unexpected token {token!r}; params are key=value"}


def _parse_overrides(tokens: Sequence[str]) -> tuple[dict[str, str], list[dict]]:
    """Split `key=value` tokens, collecting (not raising) errors. Dotted keys reserved; a repeated
    key within the CLI layer is an error."""
    raw: dict[str, str] = {}
    errors: list[dict] = []
    for tok in tokens or ():
        if tok.startswith("-"):
            errors.append(_flag_hint(tok))
        elif "=" not in tok:
            errors.append(
                {
                    "param": tok,
                    "message": f"expected key=value (quote spaces/brackets), got {tok!r}",
                }
            )
        else:
            key, _, value = tok.partition("=")
            if not key:
                errors.append({"param": tok, "message": "empty key"})
            elif "." in key:
                errors.append({"param": key, "message": f"dotted keys are reserved: {key}"})
            elif key in raw:
                errors.append({"param": key, "message": f"repeated override {key!r} (set it once)"})
            else:
                raw[key] = value
    return raw, errors


def _load_config(path: str) -> dict:
    """Load a `--config` YAML/JSON param file into a dict (yaml.safe_load parses JSON too)."""
    import yaml

    if path == "-":
        raise CliError("bad_config", "--config does not read stdin (that is --data -)", EXIT_USAGE)
    try:
        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except FileNotFoundError as exc:
        raise CliError("input_not_found", f"no such config file: {path!r}", EXIT_USAGE) from exc
    except yaml.YAMLError as exc:
        raise CliError("bad_config", f"could not parse config {path!r}: {exc}", EXIT_USAGE) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise CliError(
            "bad_config",
            f"config must be a mapping of param: value, got {type(data).__name__}",
            EXIT_USAGE,
        )
    bad_keys = [k for k in data if not isinstance(k, str)]
    if bad_keys:
        raise CliError(
            "bad_config", f"config keys must be param names, got {bad_keys!r}", EXIT_USAGE
        )
    return data


def _resolve_params(spec: MethodSpec, config: dict, tokens: Sequence[str]) -> tuple[dict, dict]:
    """Layer spec defaults < --config < key=value overrides; coerce / validate against the spec. All
    problems collect into one `invalid_params` envelope (a `details` list)."""
    specs = {p.name: p for p in spec.params}
    input_names = {i.name for i in spec.data.inputs}
    raw_overrides, errors = _parse_overrides(tokens)
    fkwargs: dict = {p.name: p.default for p in spec.params}  # manifest = defaults + config + CLI

    def _assign(key: str, value: object, *, from_config: bool) -> None:
        # A data array or `method` reaches the facade positionally; letting it also arrive as a
        # keyword would raise "multiple values" -- guard both layers, so it is a clean usage error.
        if key in input_names or key == "method":
            errors.append(
                {"param": key, "message": f"{key} is a data input / reserved, not a param"}
            )
            return
        param = specs.get(key)
        if param is None and not spec.passthrough:
            errors.append(
                {"param": key, "message": f"unknown param {key!r}", "valid_options": sorted(specs)}
            )
            return
        try:
            if param is None:  # passthrough (CALM)
                # a CLI override value is always str (raw_overrides); a config value is pre-typed
                fkwargs[key] = value if from_config else _coerce_passthrough(value)  # ty: ignore[invalid-argument-type]
            elif from_config:
                fkwargs[key] = _validate_config_value(value, param)
            else:
                fkwargs[key] = _coerce(value, param)  # ty: ignore[invalid-argument-type]
        except CliError as exc:
            errors.append({"param": key, "message": exc.message, **exc.extra})

    for key, value in config.items():  # config layer: typed values
        _assign(key, value, from_config=True)
    for key, value in raw_overrides.items():  # CLI overrides win over config
        _assign(key, value, from_config=False)

    if errors:
        raise CliError(
            "invalid_params",
            f"{len(errors)} invalid parameter(s) for {spec.name}",
            EXIT_USAGE,
            details=errors,
            next=[f"andrey run {spec.name} --help --json"],
        )
    resolved = {k: (list(v) if isinstance(v, tuple) else v) for k, v in fkwargs.items()}
    return fkwargs, resolved


def _load_method_inputs(
    spec: MethodSpec, provided: dict[str, list[str] | None]
) -> tuple[list, tuple[str, ...] | None]:
    """Load a method's array inputs from their `--flag` value(s).

    Errors on a missing required input or an unexpected input flag. Also returns the first input's
    header, whose column names name the nodes.
    """
    used = {i.name for i in spec.data.inputs}
    extra = [name for name, val in provided.items() if val and name not in used]
    if extra:
        flags = ", ".join("--" + e.replace("_", "-") for e in extra)
        raise CliError("unexpected_input", f"{spec.name} does not take {flags}", EXIT_USAGE)
    positional, headers = [], []
    for inp in spec.data.inputs:
        vals = provided.get(inp.name)
        if not vals and not inp.required:
            continue  # optional inputs come last, so the facade's default fills the gap
        if not vals:
            raise CliError(
                "missing_input",
                f"{spec.name} needs {inp.flag} <path>",
                EXIT_USAGE,
                next=[f"andrey run {spec.name} --help --json"],
            )
        if inp.kind != "panel" and len(vals) != 1:
            raise CliError("bad_input", f"{inp.flag} takes a single path", EXIT_USAGE)
        arr, labels = load_input(inp, vals)
        positional.append(arr)
        headers.append(labels)
    return positional, headers[0]


def column_positions(
    columns: list[str] | tuple[str, ...], labels: tuple[str, ...] | None, name: str = "categorical"
) -> list[int]:
    """Turn ``categorical=`` or ``target=`` entries into 0-based column positions the facade reads.

    An entry names a column of the data file's header, or else gives a position as digits.
    """
    positions = []
    for column in columns:
        if labels is not None and column in labels:
            positions.append(labels.index(column))
        elif column.isdigit():
            positions.append(int(column))
        else:
            known = f"; the header names {list(labels)}" if labels is not None else ""
            raise CliError(
                "unknown_column",
                f"{name} names column {column!r}, which the data file does not have{known}",
                EXIT_USAGE,
            )
    return positions


def _prepare_call(
    spec: MethodSpec,
    positional: list,
    fkwargs: dict,
    resolved: dict,
    labels: tuple[str, ...] | None,
) -> tuple[list, dict, dict, tuple[str, ...] | None]:
    """Shape the loaded inputs and coerced params into what the facade takes.

    A method with its own ``labels`` param (GIN, whose graph adds latent nodes) takes the data's
    header there, unless ``labels=`` was given. ``categorical=`` names become positions and
    ``knowledge=`` becomes an ``andrey.Knowledge``. A method that reads column types or knowledge
    gets the data as a DataFrame, so its warnings name the header's columns. Returns the inputs,
    params, resolved params, and the labels still to apply to the result.
    """
    if labels is not None and any(p.name == "labels" for p in spec.params):
        if fkwargs["labels"] is None:
            fkwargs = {**fkwargs, "labels": labels}
            resolved = {**resolved, "labels": list(labels)}
        labels = None
    if fkwargs.get("categorical") is not None:
        fkwargs = {**fkwargs, "categorical": column_positions(fkwargs["categorical"], labels)}
    if fkwargs.get("target") is not None:
        fkwargs = {**fkwargs, "target": column_positions([fkwargs["target"]], labels, "target")[0]}
    if fkwargs.get("knowledge") is not None:
        try:
            fkwargs = {**fkwargs, "knowledge": andrey.Knowledge(**fkwargs["knowledge"])}
        except (TypeError, ValueError) as exc:
            raise CliError("bad_value", f"knowledge: {exc}", EXIT_USAGE) from exc
    if labels is not None and any(p.name in ("categorical", "knowledge") for p in spec.params):
        # A method that reads column types names the header's columns in its warnings and in
        # metadata["categorical"]; knowledge names them too, in each group of a panel.
        import pandas as pd

        first = positional[0]
        if spec.data.inputs[0].kind == "panel":
            first = [pd.DataFrame(X, columns=list(labels)) for X in first]
        else:
            first = pd.DataFrame(first, columns=list(labels))
        positional = [first, *positional[1:]]
    return positional, fkwargs, resolved, labels


@contextlib.contextmanager
def _fit_errors() -> Generator[None]:
    """Turn what a fit raises into the ``CliError`` that names its exit code."""
    try:
        yield
    except ImportError as exc:
        raise CliError(
            "missing_backend", str(exc), EXIT_USAGE, next=["uv pip install 'andrey-core[torch]'"]
        ) from exc
    except NotImplementedError as exc:  # a documented parameter limit -> fix the call
        raise CliError("unsupported_value", str(exc), EXIT_USAGE) from exc
    except KnowledgeError as exc:  # knowledge the data's columns or the search cannot satisfy
        raise CliError("bad_value", f"knowledge: {exc}", EXIT_USAGE) from exc
    except TypeError as exc:  # an unknown passthrough knob reaches the facade as a bad kwarg
        raise CliError("unsupported_value", f"{exc}", EXIT_USAGE) from exc
    except (ValueError, ArithmeticError) as exc:  # bad shape / NaN / singular / numeric breakdown
        raise CliError("data_error", f"{type(exc).__name__}: {exc}", EXIT_DATA) from exc
    except RuntimeError as exc:  # non-convergence / numerical breakdown -> retry or adjust
        raise CliError("algorithm_failure", str(exc), EXIT_ALGORITHM) from exc


def _run_info(
    positional: list,
    resolved: dict,
    elapsed: float,
    caught: list[warnings.WarningMessage],
) -> dict:
    """The envelope's ``run`` block: params, timing, sizes, versions, and the fit's warnings."""
    head = positional[0]
    first = head[0] if isinstance(head, list) else head
    run = {
        "params": resolved,
        "elapsed_s": round(elapsed, 6),
        "n_samples": int(first.shape[0]),
        "n_variables": int(first.shape[1]),
        "versions": versions(),
        "warnings": _warning_views(caught),
    }
    return run


def _emit(env: dict, human: Callable[[], str], *, output: str | None, as_json: bool, full: bool):
    """Write the envelope to ``output``, or print it: capped JSON, or ``human()`` on a terminal."""
    if output:  # a file is not context-bounded -> always the complete envelope
        _atomic_write(output, json.dumps(env, indent=2))
        return
    if not full:  # bound stdout so a large graph can't flood it
        _apply_output_cap(env, EDGE_LIMIT)
    if as_json or not sys.stdout.isatty():  # piped / agent / CI / forced -> JSON
        typer.echo(json.dumps(env, indent=2))
        return
    try:  # an interactive terminal -> a human summary (never fail to produce output)
        typer.echo(human())
    except Exception:
        typer.echo(json.dumps(env, indent=2))


def execute_run(
    spec: MethodSpec,
    positional: list,
    fkwargs: dict,
    resolved: dict,
    *,
    labels: tuple[str, ...] | None = None,
    output: str | None,
    as_json: bool,
    full: bool,
) -> None:
    """Call the facade on the loaded inputs + coerced params, then emit the result (shell-aware).

    ``labels`` (the data's header) name the result's nodes. A method with its own ``labels`` param
    (GIN, whose graph adds latent nodes) takes them there, unless ``labels=`` was given.
    """
    import time

    positional, fkwargs, resolved, labels = _prepare_call(
        spec, positional, fkwargs, resolved, labels
    )
    fn = getattr(andrey, spec.name)
    start = time.perf_counter()
    with _fit_errors():
        # The fit's warnings travel in the envelope: stderr carries only a failure's JSON.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn(*positional, **fkwargs)
    elapsed = time.perf_counter() - start
    if isinstance(out, list):
        out = [with_labels(o, labels) for o in out]
    else:
        out = with_labels(out, labels)

    run = _run_info(positional, resolved, elapsed, caught)
    if spec.seeded:  # report the *effective* seed (resolve_seed applies the ANDREY_SEED fallback)
        from andrey.core.seeding import resolve_seed

        run["seed"] = resolve_seed(resolved.get("seed", resolved.get("random_state")))
    env = envelope(spec, out, run)
    _emit(
        env,
        lambda: render_human(spec, out, run, full=full),
        output=output,
        as_json=as_json,
        full=full,
    )


def _table(data: object, labels: tuple[str, ...] | None) -> object:
    """Return ``data`` (an array, or a list of them) as tables with columns named ``labels``."""
    if labels is None or hasattr(data, "iloc"):
        return data
    import pandas as pd

    if isinstance(data, list):
        return [pd.DataFrame(x, columns=list(labels)) for x in data]
    return pd.DataFrame(data, columns=list(labels))


def execute_bootstrap(
    spec: MethodSpec,
    positional: list,
    fkwargs: dict,
    resolved: dict,
    *,
    labels: tuple[str, ...] | None = None,
    n_boot: int,
    seed: int | None,
    threshold: float,
    block_length: int | None,
    output: str | None,
    as_json: bool,
    full: bool,
) -> None:
    """Refit the method on bootstrap resamples, then emit the edge frequencies (shell-aware).

    The bootstrap seeds a seeded method itself, so ``seed=`` is the bootstrap's ``--seed``.
    """
    import time

    if fkwargs.get("seed") is not None:
        raise CliError(
            "bad_value", "the bootstrap seeds each fit itself; set --seed, not seed=", EXIT_USAGE
        )
    panel = spec.data.inputs[0].kind == "panel"
    bad = [
        message
        for ok, message in (
            (n_boot >= 1, f"--n-boot must be at least 1, got {n_boot}"),
            (0 < threshold <= 1, f"--threshold must lie in (0, 1], got {threshold}"),
            (seed is None or seed >= 0, f"--seed must not be negative, got {seed}"),
            (block_length is None or block_length >= 1, "--block-length must be at least 1"),
            (
                block_length is None or (spec.family == "temporal" and not panel),
                f"--block-length is for a time-series method on one series, not {spec.name}",
            ),
        )
        if not ok
    ]
    if bad:
        raise CliError("bad_value", "; ".join(bad), EXIT_USAGE)
    fkwargs = {k: v for k, v in fkwargs.items() if k != "seed"}
    resolved = {k: v for k, v in resolved.items() if k != "seed"}
    positional, fkwargs, resolved, labels = _prepare_call(
        spec, positional, fkwargs, resolved, labels
    )
    positional = [_table(positional[0], labels), *positional[1:]]
    aligned = {inp.name: x for inp, x in zip(spec.data.inputs[1:], positional[1:])}
    start = time.perf_counter()
    with _fit_errors():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = andrey.bootstrap(
                spec.name,
                positional[0],
                n_boot=n_boot,
                seed=seed,
                threshold=threshold,
                block_length=block_length,
                **aligned,
                **fkwargs,
            )
    run = _run_info(positional, resolved, time.perf_counter() - start, caught)
    env = bootstrap_envelope(spec, out, run)
    _emit(
        env,
        lambda: render_bootstrap_human(spec, out, run, full=full),
        output=output,
        as_json=as_json,
        full=full,
    )


def _warning_views(caught: list[warnings.WarningMessage]) -> list[dict]:
    """Each distinct warning once, in the order raised."""
    views = ({"category": w.category.__name__, "message": str(w.message)} for w in caught)
    return list({(v["category"], v["message"]): v for v in views}.values())


def _example(spec: MethodSpec, command: str = "run") -> str:
    """A runnable, copy-paste example for a method (list values quoted for zsh)."""
    parts = ["andrey", command, spec.name]
    for inp in spec.data.inputs:
        if inp.kind == "panel":
            parts += [inp.flag, "a.csv", inp.flag, "b.csv"]
        else:
            parts += [inp.flag, "data.csv" if inp.name == "data" else f"{inp.name}.csv"]
    if spec.params:
        p = spec.params[0]
        val = grammar_example(p)
        parts.append(
            f"'{p.name}={val}'" if p.type in ("int_pair", "str_list") else f"{p.name}={val}"
        )
    return " ".join(parts)


def render_method_help(spec: MethodSpec, *, command: str = "run") -> str:
    """Per-method help rendered from the registry -- what `andrey run <method> --help` prints.

    ``command="bootstrap"`` gives `andrey bootstrap <method> --help`: the same inputs and params,
    with the bootstrap flags.
    """
    lines = [f"andrey {command} {spec.name} -- {spec.summary}", ""]
    if spec.status == "experimental":
        lines += [
            "Experimental: no published benchmark; the API may change without deprecation.",
            "",
        ]
    lines.append("Inputs:")
    for inp in spec.data.inputs:
        rep = "  (repeatable, >= 2)" if inp.kind == "panel" else ""
        lines.append(f"  {inp.flag} <path>{rep}   {inp.doc}")
    lines += [
        "  Files: CSV, TSV, or whitespace-separated text (a header row names the nodes), or .npy.",
        '  A first row of numbers is data; quote a header name that is a number ("1").',
        "  A path of - reads text from stdin.",
    ]
    if spec.params:
        lines += ["", "Params (key=value):"]
        for p in spec.params:
            choices = f"  [{', '.join(p.choices)}]" if p.choices else ""
            lines.append(f"  {p.name}={grammar_default(p)}{choices}   {p.doc}")
    if spec.passthrough:
        lines += ["", "  <key>=<value>   extra optimizer knobs (passthrough)"]
    flags = "--config <file> (YAML/JSON base), -o <file>, --json, --full"
    if command == "bootstrap":
        lines += [
            "",
            "Bootstrap flags: --n-boot <n> (default 100), --seed <int>, --threshold <share> "
            "(default 0.5),",
            "  --block-length <rows> (time-series methods on one series). Workers follow "
            "ANDREY_NUM_WORKERS.",
        ]
    lines += [
        "",
        f"Run flags: {flags}",
        "",
        "Example:",
        f"  {_example(spec, command)}",
        "",
        f"Full schema: andrey {command} {spec.name} --help --json",
    ]
    return "\n".join(lines)


def check_environment() -> None:
    """Parse every ``ANDREY_*`` variable; name each bad value in one ``invalid_configuration``
    error."""
    errors = []
    for variable in ENV_VARS:
        try:
            SETTINGS[variable.name].read()
        except ValueError as exc:
            errors.append(str(exc))
    if errors:
        raise CliError(
            "invalid_configuration", "; ".join(errors), EXIT_USAGE, next=["andrey config --help"]
        )


# What each optional extra enables. `andrey config` reports whether each is installed.
EXTRAS = {
    "torch": "the `cpu`, `cuda`, and `mps` backends, GPU correlation and entropy tables, and CALM",
    "viz": (
        "`andrey.viz.mpl` chart styling (matplotlib) and Graphviz layouts for `andrey.viz.draw` "
        "(pygraphviz)"
    ),
}
SUPPORT = {
    "linux": "supported",
    "darwin": "runs, no guarantee yet",
    "win32": "runs, no guarantee yet",
}


def _extra_packages() -> dict[str, list[str]]:
    """Each extra's distributions, read from the installed package's metadata."""
    try:
        requires = importlib.metadata.requires("andrey-core") or []
    except importlib.metadata.PackageNotFoundError:
        return {}
    extras: dict[str, list[str]] = {}
    for requirement in requires:
        extra = re.search(r"""extra\s*==\s*["']([^"']+)["']""", requirement)
        name = re.match(r"[A-Za-z0-9._-]+", requirement)
        if extra and name:
            extras.setdefault(extra.group(1), []).append(name.group(0))
    return extras


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _unavailable_reason(backend: str, extras: dict[str, bool]) -> str:
    if not extras["torch"]:
        return "the [torch] extra is not installed"
    if backend == "mps" and sys.platform != "darwin":
        return "Apple MPS needs macOS"
    return f"torch finds no {backend.upper()} device"


def system_report(available: dict[str, bool]) -> dict:
    """The machine as Andrey sees it: the platform and its support tier, the usable cores, each
    optional extra with the command that installs it, and why each unavailable backend is off.
    Reads installed metadata only; it imports no optional package."""
    packages = _extra_packages()
    extras, installed = [], {}
    for name, effect in EXTRAS.items():
        versions = {dist: _version(dist) for dist in packages.get(name, [])}
        installed[name] = bool(versions) and all(versions.values())
        extras.append(
            {
                "name": name,
                "installed": installed[name],
                "packages": versions,
                "effect": effect,
                "install": f'pip install "andrey-core[{name}]"',
            }
        )
    return {
        "platform": sys.platform,
        "support": SUPPORT.get(sys.platform, "untested"),
        "python": platform.python_version(),
        "usable_cores": usable_cpus(),
        "extras": extras,
        "unavailable_backends": {
            name: _unavailable_reason(name, installed) for name, ok in available.items() if not ok
        },
    }


def config_report() -> dict:
    """Return current settings, detected backend availability, the machine, and raw ``ANDREY_*``
    values."""
    check_environment()
    state = andrey.describe().as_dict()
    environment = env_dict()
    for variable in environment["variables"]:
        variable["value"] = os.environ.get(variable["name"])
    return {
        "schema_version": SCHEMA_VERSION,
        "andrey_version": andrey.__version__,
        "configuration": {
            "backend": state["backend"],
            "num_workers": state["num_workers"],
            "tie_break": andrey.config.tie_break,
        },
        "available_backends": state["available_backends"],
        "system": system_report(state["available_backends"]),
        "environment": environment,
    }


def render_config(report: dict) -> str:
    """Render a configuration snapshot without claiming an operation's resolved backend."""
    settings = report["configuration"]
    available = ", ".join(
        "cpu (torch)" if name == "cpu" else name
        for name, present in report["available_backends"].items()
        if present
    )
    system = report["system"]
    lines = [
        f"Backend preference: {settings['backend']}",
        f"Worker setting: {settings['num_workers']}",
        f"Tie break: {settings['tie_break']}",
        f"Available backends: {available}",
        *[f"  {name}: {why}" for name, why in system["unavailable_backends"].items()],
        "Actual backend selection depends on the operation, input size, and runtime state.",
        "",
        f"Platform: {system['platform']} ({system['support']}), Python {system['python']}, "
        f"{system['usable_cores']} usable cores",
        "Extras:",
        *[
            f"  {extra['name']}: "
            + (
                ", ".join(f"{dist} {version}" for dist, version in extra["packages"].items())
                if extra["installed"]
                else f"not installed ({extra['install']})"
            )
            for extra in system["extras"]
        ],
        "",
        "Environment:",
    ]
    for variable in report["environment"]["variables"]:
        value = variable["value"]
        displayed = repr(value) if value is not None else f"unset (default: {variable['default']})"
        lines.append(f"  {variable['name']}: {displayed}")
    lines += ["", "Defaults and effects: andrey config --help"]
    return "\n".join(lines)


def render_config_help() -> str:
    """Explain the ``ANDREY_*`` variables and precedence from the shared specification."""
    environment = env_dict()
    lines = [
        "Environment variables (in andrey config --json, each value is the raw setting, null "
        "when unset):",
        "",
    ]
    for variable in environment["variables"]:
        lines += [
            f"  {variable['name']} (default: {variable['default']})",
            f"    Values: {variable['values']}",
            f"    {variable['effect']}",
        ]
    lines += ["", "Precedence:", *[f"  {rule}" for rule in environment["precedence"]]]
    lines += [
        "",
        "A blank value counts as unset. Any other value outside Values makes andrey config and",
        "andrey run exit 2 with invalid_configuration.",
        "",
        "andrey config imports torch to check for a device on a CUDA or ROCm build of torch, or on",
        "macOS, which can take seconds; andrey config --help imports no optional backend.",
    ]
    return "\n".join(lines)


def versions() -> dict:
    import numpy

    return {"andrey": andrey.__version__, "numpy": numpy.__version__, "python": _py_version()}


def _py_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _atomic_write(path: str, text: str) -> None:
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


# --- the Typer app --------------------------------------------------------------------------------

app = typer.Typer(
    add_completion=False,
    add_help_option=False,
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},  # -h works everywhere, not just `run`
    help=(
        "Accelerated, accurate, agentic causal discovery. List the methods, run one on a data "
        "file, and check the configuration."
    ),
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"andrey {andrey.__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True, add_help_option=False)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version and exit."
    ),
    show_help: bool = typer.Option(
        False,
        "--help",
        "-h",
        help="Show this message; add --json for every method's inputs, parameters, and output as "
        "JSON.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Use JSON with --help."),
    skill: bool = typer.Option(
        False, "--skill", help="Print the SKILL.md guide for coding agents."
    ),
) -> None:
    """Accelerated, accurate, agentic causal discovery.

    List the methods, run one on a data file, and check the configuration.
    """
    if show_help and skill:
        raise CliError("usage_error", "Choose one of --help or --skill.", EXIT_USAGE)
    if as_json and not show_help:
        raise CliError(
            "usage_error",
            "Root --json only pairs with --help; put --json after the command.",
            EXIT_USAGE,
            next=["andrey --help --json", "andrey list --json", "andrey config --json"],
        )
    if show_help:
        typer.echo(json.dumps(help_schema(), indent=2) if as_json else ctx.get_help())
        raise typer.Exit(EXIT_OK)
    if skill:
        typer.echo(render_skill())
        raise typer.Exit(EXIT_OK)
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(EXIT_OK)


@app.command("list")
def cmd_list(
    as_json: bool = typer.Option(False, "--json", help="Use JSON even on a terminal."),
) -> None:
    """Enumerate every method with its status and a one-line summary."""
    rows = [
        {
            "name": s.name,
            "family": s.family,
            "status": s.status,
            "output": s.output.graph_kind or s.output.structure_type,
            "accel": s.accel,
            "determinism": s.determinism,
            "summary": s.summary,
        }
        for s in list_specs()
    ]
    if sys.stdout.isatty() and not as_json:
        width = max(len(r["name"]) for r in rows)
        header = {"name": "method", "family": "family", "status": "status", "summary": "summary"}
        for r in (header, *rows):
            line = f"{r['name']:<{width}}  {r['family']:<12}  {r['status']:<12}  {r['summary']}"
            typer.echo(line)
    else:
        typer.echo(json.dumps(rows, indent=2))


@app.command("config", add_help_option=False)
def cmd_config(
    ctx: typer.Context,
    show_help: bool = typer.Option(
        False, "--help", "-h", help="Explain environment defaults and effects."
    ),
    as_json: bool = typer.Option(False, "--json", help="Use JSON for the report or help."),
) -> None:
    """Show current configuration and available backends without changing settings."""
    if show_help:
        if as_json:
            typer.echo(
                json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "andrey_version": andrey.__version__,
                        "environment": env_dict(),
                    },
                    indent=2,
                )
            )
        else:
            typer.echo(ctx.get_help() + "\n\n" + render_config_help())
        return
    report = config_report()
    typer.echo(json.dumps(report, indent=2) if as_json else render_config(report))


@app.command("calibrate")
def cmd_calibrate(
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        help="Pool size to measure; default: ANDREY_NUM_WORKERS if above 1, else all usable CPUs.",
    ),
    method: Optional[list[str]] = typer.Option(
        None,
        "--method",
        help="Measure only this method (ges or skeleton); repeatable. Default: both.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Use JSON even on a terminal."),
) -> None:
    """Measure the worker-pool cutoffs on this machine and print the settings to use.

    Times serial runs against pooled ones for a few sizes, whole fits for GES and single passes for
    the skeleton search of PC, FCI, GFCI, and CD-NOD, and sets nothing. It takes a minute or two,
    and several minutes when the pool never wins (the search then runs up to 256 variables).
    """
    from andrey.core import backend
    from andrey.search import calibrate as cal

    check_environment()
    methods = method or list(cal.METHODS)
    unknown = sorted(set(methods) - set(cal.METHODS))
    if unknown:
        raise CliError(
            "unsupported_value",
            f"--method must be {' or '.join(cal.METHODS)}, got {unknown[0]!r}",
            EXIT_USAGE,
            valid_options=list(cal.METHODS),
        )
    if workers is not None and workers < 2:
        raise CliError("unsupported_value", "--workers must be at least 2", EXIT_USAGE)
    configured = backend.worker_count()
    n = workers or (configured if configured > 1 else backend.usable_cpus())
    if n < 2:
        raise CliError(
            "unsupported_value",
            f"calibration needs at least 2 workers; this process can use {n} CPU",
            EXIT_USAGE,
        )
    report = calibration_report(cal.calibrate(n, methods), n)
    if as_json or not sys.stdout.isatty():
        typer.echo(json.dumps(report, indent=2))
    else:
        typer.echo(render_calibration(report))


def calibration_report(results: dict, workers: int) -> dict:
    """The measured crossovers as JSON: per method, the bracket, the timings, and the setting."""
    from andrey.search import calibrate as cal

    def side(c: cal.Crossover, d: int | None) -> dict | None:
        return None if d is None else {"d": d, "work": c.work(d)}

    methods = {
        m: {
            "variable": cal.METHODS[m].variable,
            "cutoff": c.cutoff,
            "serial_faster_at": side(c, c.below),
            "pool_faster_at": side(c, c.above),
            "pool_start_s": None if c.start_s is None else round(c.start_s, 4),
            "timings": [
                {"d": d, "work": c.work(d), "serial_s": round(s, 4), "pooled_s": round(p, 4)}
                for d, s, p in c.timings
            ],
        }
        for m, c in results.items()
    }
    settings = {"ANDREY_NUM_WORKERS": str(workers)}
    settings.update({v["variable"]: str(v["cutoff"]) for v in methods.values()})
    return {
        "schema_version": SCHEMA_VERSION,
        "andrey_version": andrey.__version__,
        "workers": workers,
        "methods": methods,
        "settings": settings,
        "note": cal.NOTE,
    }


def render_calibration(report: dict) -> str:
    """The human summary of :func:`calibration_report`, ending with the lines to export."""
    lines = [f"Measured with {report['workers']} workers."]
    for m, v in report["methods"].items():
        below, above = v["serial_faster_at"], v["pool_faster_at"]
        if above is None:
            where = f"the pool was slower up to d={below['d']}"
        elif below is None:
            where = f"the pool was faster down to d={above['d']}"
        else:
            where = (
                f"serial was faster at d={below['d']} (work {below['work']:,}), "
                f"the pool at d={above['d']} (work {above['work']:,})"
            )
        started = v.get("pool_start_s")
        took = "" if started is None else f" Starting the pool takes about {started:.2f} s."
        lines.append(f"{m.upper()}: {where}.{took}")
    lines += ["", "Set:"]
    lines += [f"  export {k}={v}" for k, v in report["settings"].items()]
    lines += ["", report["note"]]
    return "\n".join(lines)


def _method_or_help(
    ctx: typer.Context, method: str | None, show_help: bool, as_json: bool
) -> MethodSpec | None:
    """Return the method's spec, or ``None`` after printing its help (or the command's)."""
    command = ctx.command.name or "run"
    if method is None:
        if show_help:
            if as_json:
                raise CliError(
                    "missing_method",
                    "Specify a method, or use andrey --help --json.",
                    EXIT_USAGE,
                    next=["andrey --help --json"],
                )
            typer.echo(ctx.get_help())
            return None
        raise CliError(
            "missing_method",
            "Choose a method (see: andrey list).",
            EXIT_USAGE,
            next=["andrey list"],
        )
    if method not in REGISTRY:
        raise CliError(
            "unknown_method",
            f"unknown method {method!r}",
            EXIT_USAGE,
            valid_options=sorted(REGISTRY),
            next=["andrey list"],
        )
    spec = get_spec(method)
    if show_help:
        text = render_method_help(spec, command=command)
        typer.echo(json.dumps(help_schema(method), indent=2) if as_json else text)
        return None
    return spec


# The options `run` and `bootstrap` share.
_DataOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--data",
        metavar="PATH|-",
        help="Data file, one row per sample: CSV, TSV, or whitespace-separated text (a header row "
        "names the nodes), or .npy. - reads text from stdin.",
    ),
]
_DomainIndexOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--domain-index",
        metavar="PATH",
        help="cdnod only: the domain index, one value per row of --data.",
    ),
]
_InterventionsOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--interventions",
        metavar="PATH",
        help="gies only: a 0/1 matrix shaped like --data, 1 where the row's value of the variable "
        "was set by an intervention.",
    ),
]
_DataGroupsOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--data-groups",
        metavar="PATH",
        help="multi_group_direct_lingam only: one data file per group; repeat the flag.",
    ),
]
_DataListOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--data-list",
        metavar="PATH",
        help="longitudinal_lingam only: one data file per time point; repeat the flag.",
    ),
]
_ConfigOption = Annotated[
    Optional[str],
    typer.Option(
        "--config",
        metavar="PATH",
        help="YAML or JSON file of parameters; key=value arguments override it.",
    ),
]
_OutputOption = Annotated[
    Optional[str],
    typer.Option(
        "-o",
        "--output",
        metavar="PATH",
        help="Write the complete result as JSON to PATH instead of printing it.",
    ),
]
_JsonOption = Annotated[bool, typer.Option("--json", help="Use JSON even on a terminal.")]
_HelpOption = Annotated[
    bool,
    typer.Option(
        "--help",
        "-h",
        help="Show the method's inputs, parameters, and an example; add --json for the same as "
        "JSON.",
    ),
]


@app.command("run", add_help_option=False)
def cmd_run(
    ctx: typer.Context,
    method: str | None = typer.Argument(
        None, metavar="METHOD", help="Method to run (see: andrey list)."
    ),
    overrides: Optional[list[str]] = typer.Argument(
        None, metavar="[KEY=VALUE ...]", help="Hyperparameter overrides, for example 'alpha=0.05'."
    ),
    data: _DataOption = None,
    domain_index: _DomainIndexOption = None,
    interventions: _InterventionsOption = None,
    data_groups: _DataGroupsOption = None,
    data_list: _DataListOption = None,
    config: _ConfigOption = None,
    output: _OutputOption = None,
    as_json: _JsonOption = False,
    full: bool = typer.Option(
        False,
        "--full",
        help=f"Print every edge and weight. Without it, a terminal shows the first {SUMMARY_EDGES} "
        f"edges, and JSON output at most {EDGE_LIMIT}.",
    ),
    show_help: _HelpOption = False,
) -> None:
    r"""Run a method on a data file: andrey run <method> --data <path|-> \[key=value ...]."""
    spec = _method_or_help(ctx, method, show_help, as_json)
    if spec is None:
        return
    provided = {
        "data": data,
        "domain_index": domain_index,
        "interventions": interventions,
        "data_groups": data_groups,
        "data_list": data_list,
    }
    check_environment()
    cfg = _load_config(config) if config else {}
    fkwargs, resolved = _resolve_params(spec, cfg, overrides or [])  # validate before any I/O
    positional, labels = _load_method_inputs(spec, provided)
    execute_run(
        spec,
        positional,
        fkwargs,
        resolved,
        labels=labels,
        output=output,
        as_json=as_json,
        full=full,
    )


@app.command("bootstrap", add_help_option=False)
def cmd_bootstrap(
    ctx: typer.Context,
    method: str | None = typer.Argument(
        None, metavar="METHOD", help="Method to refit (see: andrey list)."
    ),
    overrides: Optional[list[str]] = typer.Argument(
        None, metavar="[KEY=VALUE ...]", help="Hyperparameter overrides, for example 'alpha=0.05'."
    ),
    data: _DataOption = None,
    domain_index: _DomainIndexOption = None,
    interventions: _InterventionsOption = None,
    data_groups: _DataGroupsOption = None,
    data_list: _DataListOption = None,
    n_boot: int = typer.Option(100, "--n-boot", metavar="N", help="Number of resamples and fits."),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Seed of the resamples. Without it, ANDREY_SEED, else 0.",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        help="Share of fits in which a pair must be adjacent to be in the averaged graph (0, 1].",
    ),
    block_length: Optional[int] = typer.Option(
        None,
        "--block-length",
        metavar="ROWS",
        help="Time-series methods on one series: rows per block. Default ceil(rows ** (1/3)).",
    ),
    config: _ConfigOption = None,
    output: _OutputOption = None,
    as_json: _JsonOption = False,
    full: bool = typer.Option(
        False,
        "--full",
        help=f"Print every edge and frequency. Without it, a terminal shows the first "
        f"{SUMMARY_EDGES} edges, and JSON output at most {EDGE_LIMIT} of each list.",
    ),
    show_help: _HelpOption = False,
) -> None:
    r"""Refit a method on resamples: andrey bootstrap <method> --data <path|-> \[key=value ...]."""
    spec = _method_or_help(ctx, method, show_help, as_json)
    if spec is None:
        return
    provided = {
        "data": data,
        "domain_index": domain_index,
        "interventions": interventions,
        "data_groups": data_groups,
        "data_list": data_list,
    }
    check_environment()
    cfg = _load_config(config) if config else {}
    fkwargs, resolved = _resolve_params(spec, cfg, overrides or [])  # validate before any I/O
    positional, labels = _load_method_inputs(spec, provided)
    execute_bootstrap(
        spec,
        positional,
        fkwargs,
        resolved,
        labels=labels,
        n_boot=n_boot,
        seed=seed,
        threshold=threshold,
        block_length=block_length,
        output=output,
        as_json=as_json,
        full=full,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry. Runs Typer with ``standalone_mode=False`` so every failure routes
    through one JSON error envelope: a command's ``CliError`` and Click's own usage errors (bad
    flag, unknown value/command) both land here as ``{"error": {...}}`` on stderr with a semantic
    exit code. Root ``--help`` and ``--version`` raise ``Exit``, which passes through.
    """
    args = list(argv) if argv is not None else sys.argv[1:]
    command = typer.main.get_command(app)
    try:
        command.main(args=args, prog_name="andrey", standalone_mode=False)
        return EXIT_OK
    except CliError as err:
        return emit_error(err)
    except typer.Exit as exc:  # --version / --help / bare-help
        return int(exc.exit_code)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_OK
    except BrokenPipeError:  # a downstream `| head` closed the pipe
        return EXIT_OK
    except Exception as exc:
        # A Click UsageError (bad flag, unknown value/command) exposes format_message(); re-emit it
        # as the error envelope. A stale `--param` flag gets a did_you_mean toward `param=value`.
        if hasattr(exc, "format_message"):
            msg = getattr(exc, "format_message")()  # noqa: B009  # duck-typed Click UsageError
            extra: dict = {}
            hit = re.search(r"(?i)no such option:?\s+(--[\w-]+)", msg)
            if hit and hit.group(1).lstrip("-").replace("-", "_") in {
                p.name for s in REGISTRY.values() for p in s.params
            }:
                key = hit.group(1).lstrip("-").replace("-", "_")
                extra = {"did_you_mean": f"{key}=<value> (params are key=value, not --flags)"}
            return emit_error(CliError("usage_error", msg, EXIT_USAGE, **extra))
        return emit_error(CliError("internal", f"{type(exc).__name__}: {exc}", EXIT_INTERNAL))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
