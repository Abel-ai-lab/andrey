"""The steps every public method shares, run around its facade from the method's spec.

:func:`method_spec` wraps a facade so that each call maps a former argument name or value to the
current one, checks every argument against its :class:`~andrey.spec.models.ParamSpec`, resolves
the seed, warns that an experimental method is experimental, picks a test or a score the call
leaves to the method, and names the method, the seed, the test and score the fit used, the score's
penalties, and the configuration's tie-break rule in the result's metadata. The data check that
follows, in :func:`andrey.api._adapt.data_matrix`, warns when the data are larger than the
method's measured boundary. A facade keeps only its method's own logic.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from andrey.api._adapt import default_score, default_test
from andrey.core import environments
from andrey.core.backend import config
from andrey.core.ci import SHIPPED_INDEP_TESTS, STOCHASTIC_INDEP_TESTS, available_indep_tests
from andrey.core.ci_missing import TESTWISE
from andrey.core.removed import removal_hint
from andrey.core.score import LINEAR_SCORES, PENALIZED_SCORES
from andrey.core.seeding import resolve_seed
from andrey.core.warning_policy import DeprecatedArgumentWarning, ExperimentalWarning, warn_once

if TYPE_CHECKING:
    from andrey.core.output import StructureOutput
    from andrey.spec.models import MethodSpec, ParamSpec

__all__ = ["method_spec"]

Facade = TypeVar("Facade", bound=Callable[..., Any])


@functools.cache
def spec_of(name: str) -> MethodSpec:
    """Return the method's spec, read from the registry on the first call."""
    from andrey.spec.registry import REGISTRY

    return REGISTRY[name]


def method_spec(name: str) -> Callable[[Facade], Facade]:
    """Run the facade of method ``name`` inside the steps its spec defines.

    Before the facade: a former argument name or value becomes the current one, and a dropped
    argument is removed, with one :class:`~andrey.DeprecatedArgumentWarning` for each; an argument
    or a choice that an earlier release removed raises as an unknown one does, and the message
    gives its current name (:mod:`andrey.core.removed`); every argument the call passes is checked
    and converted by :meth:`~andrey.spec.models.ParamSpec.check`, which raises ``TypeError`` or
    ``ValueError``; for a method that takes ``env``, data given as each environment's rows apart
    become one table and ``env`` (:func:`pool_environments`); a row-aligned argument that names a
    column of the data is taken out of it (:func:`take_columns`); a
    seeded method's seed is resolved (:func:`~andrey.core.seeding.resolve_seed`) and passed on as
    an int; an experimental method warns :class:`~andrey.ExperimentalWarning`
    (:func:`warn_experimental`); a ``None`` ``indep_test`` or ``score_func`` is picked from the
    declared column types, and the facade gets the pick and the columns it reads
    (:func:`chosen_settings`); a call that passes ``ebic_gamma`` or ``equivalent_sample_size`` with
    a score that does not use it raises ``ValueError`` (:func:`check_penalties`).
    After it: each result's ``metadata["algorithm"]`` is ``name``, ``metadata["indep_test"]`` and
    ``metadata["score_func"]`` are the test and the score the fit used, for a method that takes
    them, with the penalties that score used (:func:`used_penalties`);
    ``metadata["tie_break"]`` is :data:`andrey.config`'s ``tie_break`` for a method whose result it
    can change (``MethodSpec.tie_break``); when the fit draws random numbers
    (:func:`draws_random_numbers`), ``metadata["seed"]`` is the seed used; and, when its tests met
    missing values, ``metadata["missing"]`` is how it handled them (:func:`missing_used`).
    """

    def decorate(facade: Facade) -> Facade:
        signature = inspect.signature(facade)

        @functools.wraps(facade)
        def run(*args: Any, **kwargs: Any) -> Any:
            spec = spec_of(name)
            current = current_names(spec, kwargs)
            try:
                bound = signature.bind(*args, **current)
            except TypeError:
                if (hinted := removed_name_error(name, signature, current)) is None:
                    raise
                raise hinted from None
            params = {p.name: p for p in spec.params}
            for key, value in bound.arguments.items():
                if key in params:
                    bound.arguments[key] = check_argument(params[key], value, name)
            pool_environments(spec, bound.arguments)
            take_columns(spec, bound.arguments)
            stamp: dict[str, object] = {"algorithm": name}
            if spec.seeded:
                seed = bound.arguments["seed"] = resolve_seed(bound.arguments.get("seed"))
                if draws_random_numbers(spec, bound.arguments):
                    stamp["seed"] = seed
            if spec.status == "experimental":
                warn_experimental(name)
            chosen = chosen_settings(spec, bound.arguments)
            check_penalties(spec, bound.arguments, chosen.get("score_func"))
            out = facade(*bound.args, **bound.kwargs)
            stamp.update(chosen)
            stamp.update(used_penalties(spec, bound.arguments, chosen.get("score_func")))
            if spec.tie_break:
                stamp["tie_break"] = config.tie_break
            if isinstance(out, list):
                return [stamped(o, stamp | missing_used(spec, bound.arguments, o)) for o in out]
            return stamped(out, stamp | missing_used(spec, bound.arguments, out))

        return run  # ty: ignore[invalid-return-type]  # functools.wraps keeps the signature

    return decorate


def warn_experimental(name: str) -> None:
    """Warn on the first call of experimental method ``name`` in this process."""
    warn_once(
        f"andrey.{name} is experimental: it has no published benchmark, and its API may change "
        "without deprecation.",
        ExperimentalWarning,
    )


def removed_name_error(
    method: str, signature: inspect.Signature, kwargs: dict[str, Any]
) -> TypeError | None:
    """Return the error for the first argument in ``kwargs`` that a release removed, else ``None``.

    The message is Python's for an unknown argument, with what the name is called now
    (:func:`~andrey.core.removed.removal_hint`).
    """
    for key in kwargs:
        hint = None if key in signature.parameters else removal_hint(method, key)
        if hint is not None:
            return TypeError(f"{method}() got an unexpected keyword argument {key!r}; {hint}")
    return None


def pool_environments(spec: MethodSpec, arguments: dict[str, Any]) -> None:
    """Pool data given as each environment's rows apart into one table and ``env``, in place.

    Only for a method that takes ``env``: a list of tables labels each environment by its
    position, and a mapping by its key, so the method, its checks, and its metadata read one table
    and the label of each row (:func:`~andrey.core.environments.pool_environments`, which raises
    ``ValueError`` when the call also passes ``env``).
    """
    if not any(i.name == "env" for i in spec.data.inputs[1:]):
        return
    data_key = spec.data.inputs[0].name
    arguments[data_key], env = environments.pool_environments(
        arguments[data_key],
        arguments.get("env"),
        arguments.get("targets"),
        spec.name,
        missing=spec.data.missing,
    )
    if env is not None:
        arguments["env"] = env


def take_columns(spec: MethodSpec, arguments: dict[str, Any]) -> None:
    """Take each column that a row-aligned argument names out of the data, in ``arguments``.

    The row-aligned arguments are the inputs after the data and a ``targets`` parameter. One given
    as a string names a column of the DataFrame ``data``: the data lose that column, and the
    argument becomes the column's values, so the method, its default test or score, and its
    metadata read only the variables. Raises ``ValueError`` when the data have no such column or
    two arguments name the same one (:func:`~andrey.core.environments.take_columns`).
    """
    keys = [i.name for i in spec.data.inputs[1:]] + [
        p.name for p in spec.params if p.type == "targets"
    ]
    named = {key: arguments[key] for key in keys if isinstance(arguments.get(key), str)}
    data_key = spec.data.inputs[0].name
    arguments[data_key], values = environments.take_columns(arguments[data_key], named, spec.name)
    arguments.update(values)


def check_argument(param: ParamSpec, value: Any, method: str) -> object:
    """Return ``value`` checked and converted by ``param``.

    Raises what :meth:`~andrey.spec.models.ParamSpec.check` raises; when ``value`` is a choice that
    a release removed, the ``ValueError`` adds what the choice is called now
    (:func:`~andrey.core.removed.removal_hint`).
    """
    try:
        return param.check(value, method, choices=runtime_choices(param))
    except ValueError as error:
        hint = removal_hint(method, param.name, value)
        if hint is None:
            raise
        raise ValueError(f"{error}; {hint}") from None


def draws_random_numbers(spec: MethodSpec, arguments: dict[str, Any]) -> bool:
    """Whether a fit of the seeded method ``spec`` with ``arguments`` draws random numbers.

    A method with an ``indep_test`` draws them only under a randomized test
    (:data:`~andrey.core.ci.STOCHASTIC_INDEP_TESTS`) or, when it takes ``n_starts``, for the starts
    after the first, with ``n_starts`` above 1, unless it also takes a ``score_func``: then a
    search over variable orders, visited in a random order, runs before the tests (``boss_fci``,
    ``grasp_fci``), and it always does. A method with ``n_perturbations`` draws them only for the
    random changes before a later start, with ``n_starts`` above 1; every other seeded method
    always does. ``pc``'s ``missing="mvpc"`` draws them only on data with missing values, which
    this check does not read: ``pc`` records its seed itself when the correction ran.
    """
    test = next((p for p in spec.params if p.name == "indep_test"), None)
    starts = next((p for p in spec.params if p.name == "n_starts"), None)
    order_search = starts is not None and any(p.name == "score_func" for p in spec.params)
    later = starts is not None and arguments.get("n_starts", starts.default) > 1
    if test is not None and not order_search:
        return later or arguments.get("indep_test", test.default) in STOCHASTIC_INDEP_TESTS
    if any(p.concept == "n_perturbations" for p in spec.params):
        return later
    return True


#: The concepts of the arguments that name a test or a score, each with the function that picks it
#: from the declared column types when the call passes none.
CHOSEN = {"indep_test": default_test, "score_func": default_score}


def chosen_settings(spec: MethodSpec, arguments: dict[str, Any]) -> dict[str, str]:
    """Return the test and the score a fit of ``spec`` uses, by argument name, set in ``arguments``.

    A method with an ``indep_test`` or a ``score_func`` has it in the result: the name the call
    passed, or, for ``None``, the one picked from the declared column types
    (:func:`~andrey.api._adapt.default_test`, :func:`~andrey.api._adapt.default_score`), which
    replaces ``None`` in ``arguments``. A registered test is its registered name. ``categorical``
    in ``arguments`` becomes the columns the test reads, or the score for a method without a test:
    a mixed-data pick lists them by position, and an all-categorical pick needs none.
    """
    data = arguments[spec.data.inputs[0].name]
    declared = arguments.get("categorical")
    chosen: dict[str, str] = {}
    columns: dict[str, object] = {}
    for param in spec.params:
        concept = param.concept or ""
        pick = CHOSEN.get(concept)
        if pick is None:
            continue
        name, columns[concept] = pick(data, arguments.get(param.name), categorical=declared)
        arguments[param.name] = chosen[param.name] = name
    if columns:
        arguments["categorical"] = columns.get("indep_test", columns.get("score_func"))
    return chosen


#: The scores each penalty concept weighs; the other scores ignore it.
PENALTIES = {
    "penalty_discount": PENALIZED_SCORES,
    "ebic_gamma": LINEAR_SCORES,
    "equivalent_sample_size": ("bdeu",),
}
#: What each argument that only one score reads weighs; a call that passes it with another score
#: raises. ``penalty_discount`` is not among them: every score but BDeu uses it, and BDeu ignores
#: it.
SCORE_ONLY = {
    "ebic_gamma": "the extended-BIC term",
    "equivalent_sample_size": "the Dirichlet prior",
}


def used_penalties(
    spec: MethodSpec, arguments: dict[str, Any], score: str | None
) -> dict[str, float]:
    """Return the penalties the score ``score`` used in a fit of ``spec``, by argument name.

    ``penalty_discount`` weighs every score but BDeu, ``ebic_gamma`` only the linear-Gaussian BIC,
    and ``equivalent_sample_size`` only BDeu (:data:`PENALTIES`); a penalty the score ignores is
    left out. A method without ``score_func`` has one score, which uses every penalty it takes. The
    value is the call's, or the default.
    """
    one_score = not any(p.concept == "score_func" for p in spec.params)
    return {
        p.name: arguments.get(p.name, p.default)
        for p in spec.params
        if score in PENALTIES.get(p.concept or "", ()) or (one_score and p.concept in PENALTIES)
    }


def check_penalties(spec: MethodSpec, arguments: dict[str, Any], score: str | None) -> None:
    """Raise ``ValueError`` when the call passes an argument that only one score reads with another.

    ``ebic_gamma`` weighs the extended-BIC term of ``"bic"`` alone, and ``equivalent_sample_size``
    the Dirichlet prior of ``"bdeu"`` alone (:data:`SCORE_ONLY`). ``score`` is the one the fit uses
    (:func:`chosen_settings`). A method without ``score_func`` has one score, which reads every
    argument it takes.
    """
    passed = [
        (p.name, p.concept) for p in spec.params if p.concept in SCORE_ONLY and p.name in arguments
    ]
    if not passed or not any(p.concept == "score_func" for p in spec.params):
        return
    for name, concept in passed:
        used = PENALTIES[concept or ""]
        if score not in used:
            weighed = SCORE_ONLY[concept or ""]
            raise ValueError(
                f"{spec.name}: {name} weighs {weighed} of score_func={used[0]!r}, which "
                f"score_func={score!r} does not use; drop {name}, or pass score_func={used[0]!r}"
            )


def runtime_choices(param: ParamSpec) -> tuple[str, ...] | None:
    """Return the values ``param`` takes at run time where they differ from its spec, else ``None``.

    An ``indep_test`` takes its shipped choices and the tests registered with
    :func:`~andrey.core.ci.register_indep_test`.
    """
    if param.concept != "indep_test" or param.choices is None:
        return None
    registered = [name for name in available_indep_tests() if name not in SHIPPED_INDEP_TESTS]
    return (*param.choices, *registered)


def current_names(spec: MethodSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``kwargs`` with the former names and values of ``spec``'s arguments made current.

    The arguments are the data inputs and the parameters. Each former name or value warns once at
    the caller's line, and so does each dropped argument, which is removed. Raises ``ValueError``
    when a call passes an argument under both its former and its current name.
    """
    kwargs = dict(kwargs)
    for old in spec.dropped_args:
        if old in kwargs:
            del kwargs[old]
            warn_once(
                f"andrey.{spec.name}: {old} is ignored and stops being accepted in the next "
                "release; drop it",
                DeprecatedArgumentWarning,
            )
    for argument in (*spec.data.inputs, *spec.params):
        for old in argument.aliases:
            if old not in kwargs:
                continue
            if argument.name in kwargs:
                raise ValueError(
                    f"{spec.name}: {old} is the former name of {argument.name}; pass only "
                    f"{argument.name}"
                )
            kwargs[argument.name] = kwargs.pop(old)
            warn_once(
                f"andrey.{spec.name}: {old} is now {argument.name}; the former name stops working "
                "in the next release",
                DeprecatedArgumentWarning,
            )
    for param in spec.params:
        value = kwargs.get(param.name)
        for old, new in param.choice_aliases:
            if isinstance(value, str) and value == old:
                kwargs[param.name] = new
                warn_once(
                    f"andrey.{spec.name}: {param.name}={old!r} is now {param.name}={new!r}; the "
                    "former value stops working in the next release",
                    DeprecatedArgumentWarning,
                )
    return kwargs


def missing_used(
    spec: MethodSpec, arguments: dict[str, Any], out: StructureOutput
) -> dict[str, object]:
    """Return ``{"missing": value}`` when the fit ``out`` met missing values, else ``{}``.

    A fit met them when its tests counted the rows they read, ``metadata["rows_tested"]``, or when
    it ran structural EM, ``metadata["em"]``. The value is the call's ``missing`` or, for ``None``,
    the method's default handling, ``"testwise"``.
    """
    met = "rows_tested" in out.metadata or "em" in out.metadata
    if not met or not any(p.concept == "missing" for p in spec.params):
        return {}
    return {"missing": arguments.get("missing") or TESTWISE}


def stamped(out: StructureOutput, stamp: dict[str, object]) -> StructureOutput:
    """Return ``out`` with ``stamp``'s keys in its metadata, ``algorithm`` first."""
    metadata = {"algorithm": stamp["algorithm"], **out.metadata, **stamp}
    return dataclasses.replace(out, _metadata=metadata)
