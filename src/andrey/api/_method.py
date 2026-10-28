"""The steps every public method shares, run around its facade from the method's spec.

:func:`method_spec` wraps a facade so that each call maps a former argument name or value to the
current one, checks every argument against its :class:`~andrey.spec.models.ParamSpec`, resolves
the seed, warns that an experimental method is experimental, and names the method and the seed in
the result's metadata. The data check that follows, in :func:`andrey.api._adapt.data_matrix`, warns
when the data are larger than the method's measured boundary. A facade keeps only its method's own
logic.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from andrey.api._experimental import warn_experimental
from andrey.core.ci import SHIPPED_INDEP_TESTS, STOCHASTIC_INDEP_TESTS, available_indep_tests
from andrey.core.seeding import resolve_seed
from andrey.core.warning_policy import DeprecatedArgumentWarning, warn_once

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
    argument is removed, with one :class:`~andrey.DeprecatedArgumentWarning` for each; every
    argument the call passes is checked and converted by
    :meth:`~andrey.spec.models.ParamSpec.check`, which raises ``TypeError`` or ``ValueError``; a
    seeded method's seed is resolved (:func:`~andrey.core.seeding.resolve_seed`) and passed on as
    an int; an experimental method warns :class:`~andrey.ExperimentalWarning`.
    After it: each result's ``metadata["algorithm"]`` is ``name`` and, when the fit draws random
    numbers (:func:`draws_random_numbers`), ``metadata["seed"]`` is the seed used.
    """

    def decorate(facade: Facade) -> Facade:
        signature = inspect.signature(facade)

        @functools.wraps(facade)
        def run(*args: Any, **kwargs: Any) -> Any:
            spec = spec_of(name)
            bound = signature.bind(*args, **current_names(spec, kwargs))
            params = {p.name: p for p in spec.params}
            for key, value in bound.arguments.items():
                if key in params:
                    param = params[key]
                    choices = runtime_choices(param)
                    bound.arguments[key] = param.check(value, name, choices=choices)
            stamp: dict[str, object] = {"algorithm": name}
            if spec.seeded:
                seed = bound.arguments["seed"] = resolve_seed(bound.arguments.get("seed"))
                if draws_random_numbers(spec, bound.arguments):
                    stamp["seed"] = seed
            if spec.status == "experimental":
                warn_experimental(name)
            out = facade(*bound.args, **bound.kwargs)
            if isinstance(out, list):
                return [stamped(o, stamp) for o in out]
            return stamped(out, stamp)

        return run  # ty: ignore[invalid-return-type]  # functools.wraps keeps the signature

    return decorate


def draws_random_numbers(spec: MethodSpec, arguments: dict[str, Any]) -> bool:
    """Whether a fit of the seeded method ``spec`` with ``arguments`` draws random numbers.

    A method with an ``indep_test`` draws them only under a randomized test
    (:data:`~andrey.core.ci.STOCHASTIC_INDEP_TESTS`); every other seeded method always does.
    """
    test = next((p for p in spec.params if p.name == "indep_test"), None)
    return test is None or arguments.get("indep_test", test.default) in STOCHASTIC_INDEP_TESTS


def runtime_choices(param: ParamSpec) -> tuple[str, ...] | None:
    """Return the values ``param`` takes at run time where they differ from its spec, else ``None``.

    An ``indep_test`` that takes the shipped tests also takes the ones registered with
    :func:`~andrey.core.ci.register_indep_test`.
    """
    return available_indep_tests() if param.choices == SHIPPED_INDEP_TESTS else None


def current_names(spec: MethodSpec, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return ``kwargs`` with each former name and value of ``spec``'s parameters made current.

    Each former name or value warns once at the caller's line, and so does each dropped argument,
    which is removed. Raises ``ValueError`` when a call passes a parameter under both its former
    and its current name.
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
    for param in spec.params:
        for old in param.aliases:
            if old not in kwargs:
                continue
            if param.name in kwargs:
                raise ValueError(
                    f"{spec.name}: {old} is the former name of {param.name}; pass only {param.name}"
                )
            kwargs[param.name] = kwargs.pop(old)
            warn_once(
                f"andrey.{spec.name}: {old} is now {param.name}; the former name stops working "
                "in the next release",
                DeprecatedArgumentWarning,
            )
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


def stamped(out: StructureOutput, stamp: dict[str, object]) -> StructureOutput:
    """Return ``out`` with ``stamp``'s keys in its metadata, ``algorithm`` first."""
    metadata = {"algorithm": stamp["algorithm"], **out.metadata, **stamp}
    return dataclasses.replace(out, _metadata=metadata)
